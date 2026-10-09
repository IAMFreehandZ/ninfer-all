#pragma once

// The forward Hadamard transform of a projection input quantized for the T2 prefill tile in the
// kernel that produces it: the down-projection input of a Hadamard-rotated Dense FFN goes from the
// SwiGLU activation straight to the tile's s8 codes and per-token scales. Each CTA transforms the
// 1024-blocks of one column in turn exactly as the hadamard producers do, rounds the results to
// BF16 as they store them and keeps them in registers until the column's amax is known, then
// quantizes them as t2_prefill_i8_quantize_kernel quantizes that BF16 column, so the codes and
// scale are bit-identical to the two kernels in sequence without the BF16 round trip.

#include "ops/common/memory.cuh"
#include "ops/common/warp.cuh"
#include "ops/kernel/hadamard_transform.cuh"
#include "ops/linear/t2/t2_prefill_i8.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

inline constexpr int kT2PrefillHadamardThreads = kHadamardQuarterWarps * kWarpSize;

// a [K, tokens] BF16 with K = kBlocks * 1024; codes [t_pad][K] and scales [t_pad] in the tile's
// activation contract, padded columns zero. The grid is one CTA per padded column.
template <int kBlocks>
__global__ __launch_bounds__(kT2PrefillHadamardThreads) void t2_prefill_hadamard_quantize_kernel(
    const __nv_bfloat16* __restrict__ a, const __nv_bfloat16* __restrict__ signs,
    std::int32_t tokens, std::int8_t* __restrict__ codes, __half* __restrict__ scales) {
    using C                    = T2PrefillI8;
    constexpr int kWidth       = kBlocks * kHadamardTransformBlock;
    constexpr int kBlockGroups = kHadamardTransformBlock / C::kGroupK;
    __shared__ HadamardQuarterShared shared;
    __shared__ float maxima[kHadamardQuarterWarps];
    __shared__ __align__(16) std::uint8_t staged[kHadamardTransformBlock];
    const int tid          = static_cast<int>(threadIdx.x);
    const int lane         = tid & (kWarpSize - 1);
    const int warp         = tid / kWarpSize;
    const int column       = static_cast<int>(blockIdx.x);
    std::int8_t* const dst = codes + static_cast<std::int64_t>(column) * kWidth;

    if (column >= tokens) {
        for (int i = tid; i < kWidth / 16; i += kT2PrefillHadamardThreads) {
            store_vec(dst + i * 16, make_uint4(0u, 0u, 0u, 0u));
        }
        if (tid == 0) { scales[column] = __float2half(0.0F); }
        return;
    }

    const __nv_bfloat16* const src = a + static_cast<std::int64_t>(column) * kWidth;
    uint4 raw[kBlocks];
#pragma unroll
    for (int b = 0; b < kBlocks; ++b) {
        raw[b] = load_vec<uint4>(src + b * kHadamardTransformBlock + warp * kWarpSize * 8 + lane * 8);
    }
    // Warp w holds elements 256r + 64w + 2l + k of each block: group 2r + (w >> 1) of the block.
    __nv_bfloat162 held[kBlocks][kHadamardQuarterWarps];
    float amax = 0.0F;
#pragma unroll
    for (int b = 0; b < kBlocks; ++b) {
        float v[8];
        hadamard_unpack8(raw[b], v);
        float t[kHadamardQuarterWarps][2];
        hadamard_quarter_forward(v, signs + b * kHadamardTransformBlock, shared, warp, lane,
                                 [] { __syncthreads(); }, t);
#pragma unroll
        for (int r = 0; r < kHadamardQuarterWarps; ++r) {
            held[b][r]       = __floats2bfloat162_rn(t[r][0], t[r][1]);
            const float2 f   = __bfloat1622float2(held[b][r]);
            amax             = fmaxf(amax, fmaxf(fabsf(f.x), fabsf(f.y)));
        }
    }
    amax = warp_max(amax);
    if (lane == 0) { maxima[warp] = amax; }
    __syncthreads();
#pragma unroll
    for (int w = 0; w < kHadamardQuarterWarps; ++w) { amax = fmaxf(amax, maxima[w]); }
    const __half scale = __float2half(amax / 127.0F);
    if (tid == 0) { scales[column] = scale; }
    const float s   = __half2float(scale);
    const float inv = s > 0.0F ? 1.0F / s : 0.0F;

    const int within = (64 * warp + 2 * lane) % C::kGroupK;
#pragma unroll
    for (int b = 0; b < kBlocks; ++b) {
#pragma unroll
        for (int r = 0; r < kHadamardQuarterWarps; ++r) {
            const float2 f = __bfloat1622float2(held[b][r]);
            // The even element of the pair lands at t2_prefill_permute16(within), the odd one four on.
            const int position = (2 * r + (warp >> 1)) * C::kGroupK + t2_prefill_permute16(within);
            staged[position]     = static_cast<std::uint8_t>(max(-127, min(127, __float2int_rn(f.x * inv))));
            staged[position + 4] = static_cast<std::uint8_t>(max(-127, min(127, __float2int_rn(f.y * inv))));
        }
        __syncthreads();
        if (tid < kHadamardTransformBlock / 16) {
            store_vec(dst + b * kHadamardTransformBlock + tid * 16, load_vec<uint4>(staged + tid * 16));
        }
        __syncthreads();
    }
    static_assert(kBlockGroups == 2 * kHadamardQuarterWarps);
}

} // namespace ninfer::ops::detail
