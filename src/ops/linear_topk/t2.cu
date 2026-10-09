// T2G128_F16S head profiles (the full head, or the indexed proposal head): the ternary head is
// projected through the public T2 Linear route into a dense BF16 logits tile, then a producer CTA
// per 128 rows publishes the top sixteen keys of every column, keyed by global token id, and the
// shared merge kernels finish the reduction. The merge sorts every partial it reads, so a
// producer publishes its sixteen keys as a set, in no particular order.

#include "ninfer/ops/linear.h"

#include "core/device.h"
#include "ops/common/memory.cuh"
#include "ops/common/score_id_order.cuh"
#include "ops/linear_topk/linear_topk_launch.h"

#include <cuda_bf16.h>

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

constexpr int kWarps   = 8;
constexpr int kThreads = kWarps * 32;
constexpr int kColumns = kLinearTopKT2ChunkColumns;

static_assert(kLinearTopKGroupedRows == 128 && kLinearTopK == 16,
              "one warp holds a group's 128 keys, four per lane, and keeps sixteen");

// One warp selects the sixteen largest of 128 order keys with a bitonic network. Key e = 4 * lane
// + j sits in register j of its lane, so network distances 1 and 2 stay in registers and the rest
// cross lanes. "first" below means the lower index of a pair receives the larger key.
__device__ __forceinline__ void topk_pair(std::uint64_t& a, std::uint64_t& b, bool larger_first) {
    const bool swap      = (a < b) == larger_first;
    const std::uint64_t x = swap ? b : a;
    b                     = swap ? a : b;
    a                     = x;
}

__device__ __forceinline__ std::uint64_t topk_lanes(std::uint64_t v, int mask, bool keep_larger) {
    const std::uint64_t p = __shfl_xor_sync(0xffffffffu, v, mask);
    return (v > p) == keep_larger ? v : p;
}

// Sorts each run of sixteen keys (four lanes) with `larger_first` the run's direction, whose
// distance-8 and distance-4 partners are lanes ^2 and ^1.
__device__ __forceinline__ void topk_merge16(std::uint64_t (&a)[4], int lane, bool larger_first) {
    const bool k8 = ((lane & 2) == 0) == larger_first;
    const bool k4 = ((lane & 1) == 0) == larger_first;
#pragma unroll
    for (int j = 0; j < 4; ++j) { a[j] = topk_lanes(a[j], 2, k8); }
#pragma unroll
    for (int j = 0; j < 4; ++j) { a[j] = topk_lanes(a[j], 1, k4); }
    topk_pair(a[0], a[2], larger_first);
    topk_pair(a[1], a[3], larger_first);
    topk_pair(a[0], a[1], larger_first);
    topk_pair(a[2], a[3], larger_first);
}

// The sixteen largest keys of the warp's 128, as a set in lanes 0..3 (every aligned group of four
// lanes ends with the same copy). Runs of sixteen are sorted in alternating directions; each
// round keeps the larger key of every pair of opposite runs, which leaves the larger half of
// their union as one bitonic run, and sorts it for the next round.
__device__ __forceinline__ void topk_select16(std::uint64_t (&a)[4], int lane) {
    topk_pair(a[0], a[1], true);
    topk_pair(a[2], a[3], false);
    {
        const bool first = (lane & 1) == 0;
        topk_pair(a[0], a[2], first);
        topk_pair(a[1], a[3], first);
        topk_pair(a[0], a[1], first);
        topk_pair(a[2], a[3], first);
    }
    {
        const bool first = (lane & 2) == 0;
        const bool k4    = ((lane & 1) == 0) == first;
#pragma unroll
        for (int j = 0; j < 4; ++j) { a[j] = topk_lanes(a[j], 1, k4); }
        topk_pair(a[0], a[2], first);
        topk_pair(a[1], a[3], first);
        topk_pair(a[0], a[1], first);
        topk_pair(a[2], a[3], first);
    }
    topk_merge16(a, lane, (lane & 4) == 0);
#pragma unroll
    for (int round = 0; round < 3; ++round) {
        const int partner = 4 << round;
#pragma unroll
        for (int j = 0; j < 4; ++j) { a[j] = topk_lanes(a[j], partner, true); }
        if (round < 2) { topk_merge16(a, lane, (lane & (partner << 1)) == 0); }
    }
}

__global__ __launch_bounds__(kThreads) void t2_logits_topk_kernel(
    const __nv_bfloat16* __restrict__ logits, std::int32_t rows, std::int32_t valid_rows,
    std::int32_t columns, std::uint64_t* __restrict__ partial_keys, std::int32_t producer_groups,
    const std::int32_t* __restrict__ row_ids) {
    const int lane  = static_cast<int>(threadIdx.x) & 31;
    const int warp  = static_cast<int>(threadIdx.x) >> 5;
    const int group = static_cast<int>(blockIdx.x);
    const int row0  = group * kLinearTopKGroupedRows + 4 * lane;
    for (int column = warp; column < columns; column += kWarps) {
        const __nv_bfloat16* const source = logits + static_cast<std::int64_t>(column) * rows;
        float value[4];
        if (row0 + 3 < rows) {
            const uint2 packed = load_vec<uint2>(source + row0);
            const float2 low   = __bfloat1622float2(load_vec<__nv_bfloat162>(&packed.x));
            const float2 high  = __bfloat1622float2(load_vec<__nv_bfloat162>(&packed.y));
            value[0]           = low.x;
            value[1]           = low.y;
            value[2]           = high.x;
            value[3]           = high.y;
        } else {
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                value[j] = row0 + j < rows ? __bfloat162float(source[row0 + j]) : 0.0f;
            }
        }
        std::uint64_t key[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int row = row0 + j;
            key[j]        = row < valid_rows
                                ? score_id_order_key(value[j], row_ids != nullptr ? row_ids[row] : row)
                                : 0;
        }
        topk_select16(key, lane);
        if (lane < 4) {
            std::uint64_t* const out =
                partial_keys + (static_cast<std::int64_t>(column) * producer_groups + group) *
                                   kLinearTopK + 4 * lane;
            store_vec(out, make_ulonglong2(key[0], key[1]));
            store_vec(out + 2, make_ulonglong2(key[2], key[3]));
        }
    }
}

} // namespace

void linear_topk_logits_launch(const Tensor& logits, std::int32_t valid_rows,
                               const Tensor* row_to_global_ids,
                               const LinearTopKWorkspace& workspace, cudaStream_t stream) {
    const std::int32_t columns = logits.ne[1];
    if (columns <= 0 || columns > kColumns ||
        workspace.rows_per_producer != kLinearTopKGroupedRows || workspace.tile_columns != 0 ||
        logits.dtype != DType::BF16 || logits.ne[0] % 4 != 0 ||
        reinterpret_cast<std::uintptr_t>(logits.data) % 8 != 0) {
        throw std::invalid_argument("linear_topk: invalid materialized-logits launch");
    }
    t2_logits_topk_kernel<<<workspace.producer_groups, kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(logits.data), logits.ne[0], valid_rows, columns,
        static_cast<std::uint64_t*>(workspace.partial_keys.data), workspace.producer_groups,
        row_to_global_ids != nullptr ? static_cast<const std::int32_t*>(row_to_global_ids->data)
                                     : nullptr);
    CUDA_CHECK(cudaGetLastError());
}

void linear_topk_t2_launch(const Tensor& hidden, const Weight& head, LinearPolicy policy,
                           std::int32_t valid_rows, const Tensor* row_to_global_ids,
                           Tensor& logits, const LinearTopKWorkspace& workspace,
                           WorkspaceArena& arena, cudaStream_t stream) {
    const std::int32_t columns = hidden.ne[1];
    if (columns <= 0 || columns > kColumns || logits.dtype != DType::BF16 ||
        logits.ne[0] != head.n || logits.ne[1] != columns) {
        throw std::invalid_argument("linear_topk: invalid T2 head launch");
    }
    {
        auto scope = arena.scope();
        ops::linear(hidden, head, logits, policy, arena, stream);
    }
    linear_topk_logits_launch(logits, valid_rows, row_to_global_ids, workspace, stream);
}

} // namespace ninfer::ops::detail
