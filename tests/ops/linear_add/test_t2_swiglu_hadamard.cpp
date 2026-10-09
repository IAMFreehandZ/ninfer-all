#include "core/arena.h"
#include "core/weight.h"
#include "ninfer/ops/hadamard_transform.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/linear_add.h"
#include "ninfer/ops/silu_mul.h"
#include "ops/device_profile_fixture.h"
#include "ops/linear/t2/t2_a8.h"
#include "ops/op_tester.h"
#include "ops/quantized_weight.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <exception>
#include <iostream>
#include <string>
#include <vector>

// swiglu_hadamard_linear_add must equal linear(gate_up), silu_mul_hadamard and linear_add word for
// word at the registered T2 FFN [34816, 5120] -> [5120, 17408], on both sides of the integer prefill
// tile. Where the fused route runs, its two stages are compared against their compositions as well:
// the paired gate_up tile's output word for word against silu_mul of the BF16 projection, and the
// transform-fused quantizer's codes and scales, padded columns included, byte for byte against
// t2_a8_quantize of the BF16 hadamard_transform. The composition's own FP64 qualification is in the
// hadamard_transform, T2 Linear and T2 LinearAdd suites.

using namespace ninfer;
using namespace ninfer::test;

namespace {

constexpr std::int32_t kHidden       = 5120;
constexpr std::int32_t kIntermediate = 17408;
constexpr std::int32_t kMaxT         = 1152;
// The prefill tile's activation contract pads T to whole 64-column tiles.
constexpr std::int32_t kTileT       = 64;
constexpr ops::LinearPolicy kPolicy = ops::LinearPolicy::AllowA8Int;

struct Buffers {
    DeviceBuffer plane;
    DeviceBuffer activation;
    DeviceBuffer transformed;
    DeviceBuffer paired;
};

int compare_stages(const std::string& label, std::int32_t t, const Tensor& x, const Weight& gate_up,
                   const Tensor& signs, Buffers& buffers, WorkspaceArena& workspace) {
    const Tensor plane(buffers.plane.p, DType::BF16, {2 * kIntermediate, t});
    Tensor activation(buffers.activation.p, DType::BF16, {kIntermediate, t});
    Tensor transformed(buffers.transformed.p, DType::BF16, {kIntermediate, t});
    Tensor paired(buffers.paired.p, DType::BF16, {kIntermediate, t});
    ops::silu_mul(plane.slice(0, 0, kIntermediate), plane.slice(0, kIntermediate, kIntermediate),
                  activation, nullptr);
    ops::hadamard_transform(activation, signs, false, transformed, nullptr);
    workspace.reset();
    ops::detail::t2_a8_project_swiglu(ops::detail::t2_a8_quantize(x, workspace, nullptr), gate_up,
                                      paired, nullptr);
    workspace.reset();
    const auto reference = ops::detail::t2_a8_quantize(transformed, workspace, nullptr);
    const auto quantized =
        ops::detail::t2_a8_quantize_hadamard(activation, signs, workspace, nullptr);
    cuda_check(cudaDeviceSynchronize(), "synchronize fused stages");

    const std::size_t words = static_cast<std::size_t>(kIntermediate) * static_cast<std::size_t>(t);
    int failures =
        verify_exact((label + " paired gate_up").c_str(),
                     from_device<std::uint16_t>(static_cast<const std::uint16_t*>(paired.data),
                                                words),
                     from_device<std::uint16_t>(static_cast<const std::uint16_t*>(activation.data),
                                                words));
    const std::size_t columns = static_cast<std::size_t>((t + kTileT - 1) / kTileT * kTileT);
    const std::size_t codes   = columns * kIntermediate;
    const std::size_t scales  = columns; // one per token
    failures += verify_exact((label + " codes").c_str(),
                             from_device<std::int8_t>(quantized.codes, codes),
                             from_device<std::int8_t>(reference.codes, codes));
    failures += verify_exact((label + " scales").c_str(),
                             from_device<std::uint16_t>(quantized.scales, scales),
                             from_device<std::uint16_t>(reference.scales, scales));
    return failures;
}

int run_case(std::int32_t t, const DeviceBuffer& input, const Weight& gate_up, const Weight& down,
             const DeviceBuffer& sign_buffer, const std::vector<std::uint16_t>& residual,
             Buffers& buffers, WorkspaceArena& workspace) {
    const std::string label = "swiglu_hadamard_linear_add T=" + std::to_string(t);
    const std::size_t output_words = static_cast<std::size_t>(kHidden) * static_cast<std::size_t>(t);
    const Tensor x(input.p, DType::BF16, {kHidden, t});
    const Tensor signs(sign_buffer.p, DType::BF16, {kIntermediate});
    Tensor plane(buffers.plane.p, DType::BF16, {2 * kIntermediate, t});
    Tensor transformed(buffers.transformed.p, DType::BF16, {kIntermediate, t});

    GuardedDeviceBuffer composed(output_words * sizeof(std::uint16_t));
    GuardedDeviceBuffer fused(output_words * sizeof(std::uint16_t));
    composed.copy_from_host(residual.data(), composed.bytes());
    fused.copy_from_host(residual.data(), fused.bytes());
    Tensor composed_tensor(composed.data(), DType::BF16, {kHidden, t});
    Tensor fused_tensor(fused.data(), DType::BF16, {kHidden, t});

    workspace.reset();
    ops::linear(x, gate_up, plane, kPolicy, workspace, nullptr);
    ops::silu_mul_hadamard(plane, signs, transformed, nullptr);
    workspace.reset();
    ops::linear_add(transformed, down, composed_tensor, kPolicy, workspace, nullptr);
    workspace.reset();
    ops::swiglu_hadamard_linear_add(x, gate_up, kPolicy, signs, down, kPolicy, fused_tensor,
                                    workspace, nullptr);
    cuda_check(cudaDeviceSynchronize(), "synchronize swiglu_hadamard_linear_add");

    int failures = composed.verify_guards((label + " composed").c_str());
    failures += fused.verify_guards(label.c_str());
    std::vector<std::uint16_t> expected(output_words);
    std::vector<std::uint16_t> actual(output_words);
    composed.copy_to_host(expected.data(), composed.bytes());
    fused.copy_to_host(actual.data(), fused.bytes());
    failures += verify_exact(label.c_str(), actual, expected);

    if (ops::detail::t2_a8_prefill_tile(t)) {
        failures += compare_stages(label, t, x, gate_up, signs, buffers, workspace);
    } else if (t >= kTileT * 2) {
        std::cerr << label << ": expected the fused prefill-tile route\n";
        ++failures;
    }
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    try {
        install_device_profile_fixture_from_env();
        const quantized_weight::PatternedWeightOptions options{
            quantized_weight::RowSplitScalePattern::Small,
            quantized_weight::RowSplitCodePattern::Hashed,
        };
        const auto host_gate_up = quantized_weight::make_patterned_weight(
            QType::T2_G128_FP16, 2 * kIntermediate, kHidden, 2206u, options);
        const auto host_down = quantized_weight::make_patterned_weight(
            QType::T2_G128_FP16, kHidden, kIntermediate, 2207u, options);
        DeviceBuffer device_gate_up = to_device(host_gate_up.payload);
        DeviceBuffer device_down    = to_device(host_down.payload);
        const Weight gate_up        = host_gate_up.device_weight(device_gate_up.p);
        const Weight down           = host_down.device_weight(device_down.p);

        std::vector<float> input(static_cast<std::size_t>(kHidden) * kMaxT);
        fill_uniform(input, 2208u, -2.0F, 2.0F);
        std::vector<float> signs(static_cast<std::size_t>(kIntermediate));
        fill_uniform(signs, 2209u, -1.0F, 1.0F);
        for (auto& sign : signs) { sign = sign < 0.0F ? -1.0F : 1.0F; }
        std::vector<float> residual_values(static_cast<std::size_t>(kHidden) * kMaxT);
        fill_uniform(residual_values, 2210u, -2.0F, 2.0F);
        std::vector<std::uint16_t> residual(residual_values.size());
        for (std::size_t i = 0; i < residual.size(); ++i) {
            residual[i] = f32_to_bf16(residual_values[i]);
        }
        const DeviceBuffer device_input = to_device_bf16(input);
        const DeviceBuffer device_signs = to_device_bf16(signs);
        const std::size_t plane_bytes =
            static_cast<std::size_t>(kIntermediate) * kMaxT * sizeof(std::uint16_t);
        Buffers buffers{DeviceBuffer(2 * plane_bytes), DeviceBuffer(plane_bytes),
                        DeviceBuffer(plane_bytes), DeviceBuffer(plane_bytes)};

        // Room for the fused op's own reservation and, for the stage comparison, two quantized sets.
        const std::size_t capacity = ops::swiglu_hadamard_linear_add_workspace_capacity_bytes(
            QType::T2_G128_FP16, 2 * kIntermediate, kHidden, kPolicy, QType::T2_G128_FP16, kHidden,
            kPolicy, 1, kMaxT);
        WorkspaceArena workspace(2 * capacity);

        int failures = 0;
        for (const std::int32_t t : {1, 8, 33, 63, 64, 65, 127, 128, 257, 1000, 1152}) {
            failures += run_case(t, device_input, gate_up, down, device_signs, residual, buffers,
                                 workspace);
        }
        std::cout << (failures == 0 ? "OK" : "FAIL") << " T2 swiglu_hadamard_linear_add\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "T2 swiglu_hadamard_linear_add: " << error.what() << '\n';
        return 1;
    }
}
