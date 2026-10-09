#pragma once

// T2G128 prefill on sm_120: copy the existing packed weight bytes and permuted s8
// activation bytes with two TMA transactions per 128-k stage. The quantizer,
// ternary decoder and per-group FP32 accumulation contract are shared with the
// cp.async tile. Weight scales retain their cp.async pipeline.
//
// Each stage starts on a 1024-byte boundary so the 128-byte TMA swizzle has zero
// base bias in both activation buffers. A completed transaction barrier makes
// its copies visible; the CTA barrier also retires the preceding buffer before
// issue-next can reuse it. Descriptors are kernel values with Program-owned
// operand addresses, and require no device allocation or persistent cache.

#include "ops/linear/t2/t2_prefill_i8.cuh"

#include <cuda.h>
#include <cuda/ptx>

namespace ninfer::ops::detail {

struct T2PrefillTmaI8 {
    static constexpr int kGroupK    = T2PrefillI8::kGroupK;
    static constexpr int kStages    = T2PrefillI8::kStages;
    static constexpr int kMinBlocks = T2PrefillI8::kMinBlocks;
    static constexpr int kThreads   = T2PrefillI8::kThreads;
    static constexpr int kRows      = T2PrefillI8::kRows;
    static constexpr int kColumns   = T2PrefillI8::kColumns;
    static constexpr int kCodeBytes = T2PrefillI8::kCodeBytes;
    static constexpr int kXStride   = 128;
    static constexpr int kXBytes    = kColumns * kXStride;
    static constexpr int kStage = (kCodeBytes + kXBytes + 1023) / 1024 * 1024;
    static constexpr int kRing      = T2PrefillI8::kRing;
    static constexpr int kRingBytes = T2PrefillI8::kRingBytes;
    static constexpr int kSmem      = kStages * kStage + 2 * kRingBytes;

    static_assert(kGroupK == 128 && kStages == 2 && kRows == 128 && kColumns == 64);
    static_assert(kCodeBytes % 1024 == 0 && kStage % 1024 == 0);
    static_assert(kSmem + kStages * sizeof(std::uint64_t) <= 48 * 1024);
};

// out[row, t] = sum_k W[row, k] X[k, t] for t < tokens, through epilogue(row, token, value). With
// rows_fast false the grid is (column blocks, row blocks) and runs column blocks fastest, so a
// weight row block is read once from DRAM while the column blocks that need it pass through; with
// rows_fast it is (row blocks, column blocks), for activation bands too wide to stay in L2. A
// paired epilogue takes gate rows [0, up_offset) with their up partners (T2PrefillRows); w_map's
// box then holds half a tile, and each stage copies the gate and the up half.
template <std::int32_t kCols, class Epilogue>
__global__ __launch_bounds__(T2PrefillTmaI8::kThreads, T2PrefillTmaI8::kMinBlocks) void t2_prefill_tma_i8_kernel(
    const __grid_constant__ CUtensorMap w_map, const __grid_constant__ CUtensorMap x_map,
    const __half* __restrict__ w_scales, const __half* __restrict__ xs, std::int32_t tokens,
    bool rows_fast, std::int32_t up_offset, Epilogue epilogue) {
    using C                     = T2PrefillTmaI8;
    using Rows                  = T2PrefillRows<Epilogue::kPaired>;
    constexpr int kGroups       = kCols / C::kGroupK;
    static_assert(kCols % (C::kGroupK * C::kRing) == 0, "K must be whole scale rings");
    __shared__ __align__(1024) char smem[C::kSmem];
    __shared__ std::uint64_t barriers[C::kStages];
    namespace ptx = cuda::ptx;
    char* const s_ring = smem + C::kStages * C::kStage;

    const int tid    = static_cast<int>(threadIdx.x);
    const int lane   = tid & 31;
    const int warp   = tid >> 5;
    const int gid    = lane >> 2;
    const int tig    = lane & 3;
    const int col0   = static_cast<int>(rows_fast ? blockIdx.y : blockIdx.x) * C::kColumns;
    const int row0   = static_cast<int>(rows_fast ? blockIdx.x : blockIdx.y) * Rows::kOutputRows;

    if (tid == 0) {
#pragma unroll
        for (int b = 0; b < C::kStages; ++b) ptx::mbarrier_init(&barriers[b], 1);
        ptx::fence_proxy_async(ptx::space_shared);
    }
    __syncthreads();

    const auto issue = [&](int g, int buf) {
        char* const s_w = smem + buf * C::kStage;
        char* const s_x = s_w + C::kCodeBytes;
        if (tid == 0) {
            const std::int32_t wc[2] = {g * 32, row0};
            const std::int32_t xc[2] = {g * C::kGroupK, col0};
            ptx::cp_async_bulk_tensor(ptx::space_shared, ptx::space_global,
                                      s_w, &w_map, wc, &barriers[buf]);
            if constexpr (Epilogue::kPaired) {
                const std::int32_t uc[2] = {g * 32, up_offset + row0};
                ptx::cp_async_bulk_tensor(ptx::space_shared, ptx::space_global,
                                          s_w + Rows::kHalf * 32, &w_map, uc, &barriers[buf]);
            }
            ptx::cp_async_bulk_tensor(ptx::space_shared, ptx::space_global,
                                      s_x, &x_map, xc, &barriers[buf]);
            ptx::mbarrier_arrive_expect_tx(ptx::sem_release, ptx::scope_cta,
                ptx::space_shared, &barriers[buf], C::kCodeBytes + C::kXBytes);
        }
        if (g % C::kRing == 0) {
            char* const ring = s_ring + ((g / C::kRing) & 1) * C::kRingBytes;
            for (int r = tid; r < C::kRows; r += C::kThreads) {
                cp_async<16, Cache::cg>(
                    ring + r * 16,
                    w_scales + static_cast<std::size_t>(Rows::weight_row(row0, r, up_offset)) * kGroups + g);
            }
        }
        cp_commit();
    };

    float acc[T2PrefillI8::kMTiles][T2PrefillI8::kNTiles][4];
#pragma unroll
    for (int m = 0; m < T2PrefillI8::kMTiles; ++m) {
#pragma unroll
        for (int n = 0; n < T2PrefillI8::kNTiles; ++n) {
#pragma unroll
            for (int j = 0; j < 4; ++j) { acc[m][n][j] = 0.0F; }
        }
    }

#pragma unroll
    for (int i = 0; i < C::kStages - 1; ++i) { issue(i, i); }

    for (int g = 0; g < kGroups; ++g) {
        cp_wait<C::kStages - 2>();
        while (!ptx::mbarrier_try_wait_parity(ptx::sem_acquire, ptx::scope_cta,
                  &barriers[g % C::kStages], (g / C::kStages) & 1)) {}
        __syncthreads();
        const int next = g + C::kStages - 1;
        if (next < kGroups) {
            issue(next, next % C::kStages);
        } else {
            cp_commit();
        }

        const char* const s_w = smem + (g % C::kStages) * C::kStage;
        const char* const s_x = s_w + C::kCodeBytes;
        const __half* const ring =
            reinterpret_cast<const __half*>(s_ring + ((g / C::kRing) & 1) * C::kRingBytes);
        const int slot = g % C::kRing;

        t2_prefill_group<Rows>(s_w, ring, slot, warp, gid, tig, [&](int col, unsigned (&b)[8]) {
            const int swizzle = (col & 7) * 16;
            const char* p     = s_x + col * C::kXStride;
            const uint4 v0    = load_vec<uint4>(p + ((tig * 32) ^ swizzle));
            const uint4 v1    = load_vec<uint4>(p + ((tig * 32 + 16) ^ swizzle));
            b[0]              = v0.x;
            b[1]              = v0.y;
            b[2]              = v0.z;
            b[3]              = v0.w;
            b[4]              = v1.x;
            b[5]              = v1.y;
            b[6]              = v1.z;
            b[7]              = v1.w;
        }, acc);
    }

    t2_prefill_store(epilogue, acc, xs, row0, col0, warp, gid, tig, tokens);
}

} // namespace ninfer::ops::detail
