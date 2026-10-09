#pragma once

// T2G128 RowSplit x int8 prefill GEMM with its own activation contract, in place of the shared
// rowsplit_a8_mma.cuh mainloop with the ternary codec. On the RTX 5060 Ti (tools/t2_a8_tile_probe.cu,
// T = 1024, L2 flushed) that one ran the text-layer shapes at 74-94 TOP/s, cuBLAS int8 at 170-184;
// the original 128x64 tile ran at 134-138 and beat cuBLAS with conversions. Where the time went:
//
//   scale    the shared contract scales activations per (token, 64 k), so every two MMAs the int32
//            partial is converted and multiplied by two scales. Here the partial runs the whole
//            128-k weight group and accumulates with its weight scale alone: the activations carry
//            one scale per token, applied once to the finished sum. Per (token, 128 k) activation
//            scales cost a scale load and product per output and group, 5.7-6.8% of the GEMM time
//            on the RTX 5060 Ti; per token they moved held-out quick perplexity by +0.024%. The
//            partial's first MMA reads a zero accumulator and the exact sum converts with one I2FP,
//            which on sm_120 beat starting the accumulator at the bits of 1.5 * 2^23 and
//            subtracting that as a float (four register moves per partial) by 1.5-4% on the
//            text-layer shapes.
//   tile     the shared T2 tile is 64 rows and every such block re-streams its activation band; a
//            ternary row is a quarter of an int8 one, so the band is the traffic. 128 x 64 tiles
//            with three blocks per SM won over 64 x 128 and 256 x 64.
//   warps    each of the four warps owns 32 rows (two m-tiles) across all 64 columns, so a code
//            word is decoded once per tile; with 2 x 2 warps every word was decoded in two warps
//            and the text-layer shapes ran 4-13% slower. A warp loads the activation words of a
//            pair of n-tiles at a time, and the pair's two independent n8 partials interleave their
//            four integer MMAs before the per-group FP32 scaling; each output keeps the group order.
//   stages   two stages of 128 k with one barrier each; three or four stages measure the same.
//   decode   thread tig of an MMA quad owns k in [32 tig, 32 tig + 32) of each group, so its codes
//            for a row are one 8-byte shared load and every code word decodes in nine instructions:
//            masked with 0x3333 its nibbles are PRMT selectors of the even codes as they stand,
//            shifted by two those of the odd ones. The quantiser stores activations in the k order
//            that produces (t2_prefill_permute16).
//
// Activations: codes [Tpad][K] s8, scales [Tpad] binary16 (one per token), Tpad a multiple of
// kColumns; padded columns are zero. Output rows must be a multiple of kRows.

#include "ops/common/memory.cuh"
#include "ops/common/mma.cuh"
#include "ops/common/warp.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

struct T2PrefillI8 {
    static constexpr int kGroupK    = 128; // one weight scale group and one pipeline stage
    static constexpr int kWarpsM    = 4;
    static constexpr int kWarpsN    = 1;
    static constexpr int kMTiles    = 2;
    static constexpr int kNTiles    = 8;
    static constexpr int kStages    = 2;
    static constexpr int kMinBlocks = 3;
    static constexpr int kThreads   = kWarpsM * kWarpsN * 32;
    static constexpr int kRows      = kWarpsM * kMTiles * 16;
    static constexpr int kColumns   = kWarpsN * kNTiles * 8;
    // A column's 128 bytes padded to 144 so the eight columns an MMA quad reads miss each other.
    static constexpr int kXStride    = 144;
    static constexpr int kCodeBytes  = kRows * kGroupK / 4;
    static constexpr int kXBytes     = kColumns * kXStride;
    static constexpr int kStage      = kCodeBytes + kXBytes;
    // Weight scales arrive eight groups (16 bytes) per row at a time, double-buffered.
    static constexpr int kRing       = 8;
    static constexpr int kRingBytes  = kRows * kRing * 2;
    static constexpr int kSmem       = kStages * kStage + 2 * kRingBytes;
    static constexpr int kQuantThreads = 256;

    static_assert(kSmem <= 48 * 1024, "the T2 prefill tile must fit static-size shared memory");
};

// The weight rows a tile stages. A plain tile stages kRows consecutive rows from row0. A paired
// (SwiGLU) tile stages 64 gate rows from row0 and their up partners from up_offset + row0. Warp w's
// m-tile 0 is staged rows [16w, 16w + 16) and its m-tile 1 the same rows of the second half, so a
// paired thread's m-tile 0 holds gate rows whose up partners are its m-tile 1: the epilogue
// receives both halves of a pair without a shared exchange.
template <bool Paired>
struct T2PrefillRows {
    static constexpr int kHalf       = T2PrefillI8::kRows / 2;
    static constexpr int kOutputRows = Paired ? kHalf : T2PrefillI8::kRows;
    static_assert(T2PrefillI8::kWarpsN == 1 && T2PrefillI8::kMTiles == 2 &&
                  T2PrefillI8::kWarpsM * 16 == kHalf);

    __device__ static int weight_row(int row0, int staged, int up_offset) {
        if constexpr (Paired) {
            return staged < kHalf ? row0 + staged : up_offset + row0 + staged - kHalf;
        } else {
            (void)up_offset;
            return row0 + staged;
        }
    }
    // The first staged row of m-tile m of warp w; thread gid adds gid and gid + 8.
    __device__ static int staged_row(int w, int m) { return m * kHalf + w * 16; }
};

// The four outputs of thread (gid, tig) of warp w in m-tile m and n-tile n, each weight-scaled sum
// times its token's activation scale xs[token], through epilogue(row, token, value), or for a paired
// tile epilogue(row, token, gate, up) from m-tiles 0 and 1.
template <class Epilogue, class Acc>
__device__ __forceinline__ void t2_prefill_store(const Epilogue& epilogue, const Acc& acc,
                                                 const __half* __restrict__ xs, int row0, int col0,
                                                 int w, int gid, int tig, std::int32_t tokens) {
    using C = T2PrefillI8;
#pragma unroll
    for (int n = 0; n < C::kNTiles; ++n) {
        const int c0   = col0 + n * 8 + tig * 2;
        const float2 x = __half22float2(load_vec<__half2>(xs + c0));
        if constexpr (Epilogue::kPaired) {
            const int r0 = row0 + T2PrefillRows<true>::staged_row(w, 0) + gid;
            if (c0 < tokens) {
                epilogue(r0, c0, acc[0][n][0] * x.x, acc[1][n][0] * x.x);
                epilogue(r0 + 8, c0, acc[0][n][2] * x.x, acc[1][n][2] * x.x);
            }
            if (c0 + 1 < tokens) {
                epilogue(r0, c0 + 1, acc[0][n][1] * x.y, acc[1][n][1] * x.y);
                epilogue(r0 + 8, c0 + 1, acc[0][n][3] * x.y, acc[1][n][3] * x.y);
            }
        } else {
#pragma unroll
            for (int m = 0; m < C::kMTiles; ++m) {
                const int r0 = row0 + T2PrefillRows<false>::staged_row(w, m) + gid;
                if (c0 < tokens) {
                    epilogue(r0, c0, acc[m][n][0] * x.x);
                    epilogue(r0 + 8, c0, acc[m][n][2] * x.x);
                }
                if (c0 + 1 < tokens) {
                    epilogue(r0, c0 + 1, acc[m][n][1] * x.y);
                    epilogue(r0 + 8, c0 + 1, acc[m][n][3] * x.y);
                }
            }
        }
    }
}

// Where k (mod 16) of a code word lands among the sixteen activation bytes the MMA pairs it with:
// the decode yields k {0,2,4,6}, {1,3,5,7}, {8,10,12,14}, {9,11,13,15} per register.
__host__ __device__ constexpr int t2_prefill_permute16(int k) {
    return (k / 8) * 8 + (k % 2) * 4 + (k % 8) / 2;
}

// One code word (sixteen 2-bit two's-complement codes, lowest k lowest) to four registers of s8.
__device__ __forceinline__ void t2_prefill_decode(unsigned word, unsigned* d) {
    constexpr unsigned kTable = 0xFFFE0100u; // 0, +1, -2, -1
    const unsigned even       = word & 0x33333333u;
    const unsigned odd        = (word >> 2) & 0x33333333u;
    d[0]                      = __byte_perm(kTable, 0u, even);
    d[1]                      = __byte_perm(kTable, 0u, odd);
    d[2]                      = __byte_perm(kTable, 0u, even >> 16);
    d[3]                      = __byte_perm(kTable, 0u, odd >> 16);
}

// One 128-k group of a tile on staged codes s_w and weight scales ring[.][slot]: warp w decodes the
// codes of its two m-tiles once, then visits the n-tiles in pairs, whose activation code words
// load_column(col, b) fetches for this thread's k quarter.
template <class Rows, class LoadColumn, class Acc>
__device__ __forceinline__ void t2_prefill_group(const char* s_w, const __half* ring, int slot,
                                                 int warp, int gid, int tig,
                                                 const LoadColumn& load_column, Acc& acc) {
    using C = T2PrefillI8;
    unsigned d[C::kMTiles][2][8];
    float ws[C::kMTiles][2];
#pragma unroll
    for (int m = 0; m < C::kMTiles; ++m) {
        const int r0   = Rows::staged_row(warp, m) + gid;
        const uint2 w0 = load_vec<uint2>(s_w + r0 * 32 + tig * 8);
        const uint2 w1 = load_vec<uint2>(s_w + (r0 + 8) * 32 + tig * 8);
        t2_prefill_decode(w0.x, d[m][0]);
        t2_prefill_decode(w0.y, d[m][0] + 4);
        t2_prefill_decode(w1.x, d[m][1]);
        t2_prefill_decode(w1.y, d[m][1] + 4);
        ws[m][0] = __half2float(ring[r0 * C::kRing + slot]);
        ws[m][1] = __half2float(ring[(r0 + 8) * C::kRing + slot]);
    }
#pragma unroll
    for (int n0 = 0; n0 < C::kNTiles; n0 += 2) {
        unsigned b[2][8];
#pragma unroll
        for (int j = 0; j < 2; ++j) { load_column((n0 + j) * 8 + gid, b[j]); }
#pragma unroll
        for (int m = 0; m < C::kMTiles; ++m) {
            int c[2][4];
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                mma_s8_zero(c[j][0], c[j][1], c[j][2], c[j][3], d[m][0][0], d[m][1][0], d[m][0][1],
                            d[m][1][1], b[j][0], b[j][1]);
            }
#pragma unroll
            for (int s = 1; s < 4; ++s) {
#pragma unroll
                for (int j = 0; j < 2; ++j) {
                    mma_s8(c[j][0], c[j][1], c[j][2], c[j][3], d[m][0][2 * s], d[m][1][2 * s],
                           d[m][0][2 * s + 1], d[m][1][2 * s + 1], b[j][2 * s], b[j][2 * s + 1]);
                }
            }
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                const int n  = n0 + j;
                acc[m][n][0] = fmaf(__int2float_rn(c[j][0]), ws[m][0], acc[m][n][0]);
                acc[m][n][1] = fmaf(__int2float_rn(c[j][1]), ws[m][0], acc[m][n][1]);
                acc[m][n][2] = fmaf(__int2float_rn(c[j][2]), ws[m][1], acc[m][n][2]);
                acc[m][n][3] = fmaf(__int2float_rn(c[j][3]), ws[m][1], acc[m][n][3]);
            }
        }
    }
}

// out[row, t] = sum_k W[row, k] X[k, t] for t < tokens, through epilogue(row, token, value). With
// rows_fast false the grid is (column blocks, row blocks) and runs column blocks fastest, so a
// weight row block is read once from DRAM while the column blocks that need it pass through; with
// rows_fast it is (row blocks, column blocks), for activation bands too wide to stay in L2. A
// paired epilogue takes gate rows [0, up_offset) with their up partners (T2PrefillRows).
template <std::int32_t kCols, class Epilogue>
__global__ __launch_bounds__(T2PrefillI8::kThreads, T2PrefillI8::kMinBlocks) void t2_prefill_i8_kernel(
    const std::uint8_t* __restrict__ w_codes, const __half* __restrict__ w_scales,
    const std::int8_t* __restrict__ x, const __half* __restrict__ xs, std::int32_t tokens,
    bool rows_fast, std::int32_t up_offset, Epilogue epilogue) {
    using C                     = T2PrefillI8;
    using Rows                  = T2PrefillRows<Epilogue::kPaired>;
    constexpr int kGroups       = kCols / C::kGroupK;
    constexpr int kRowBytes     = kCols / 4;
    static_assert(kCols % (C::kGroupK * C::kRing) == 0, "K must be whole scale rings");
    __shared__ __align__(16) char smem[C::kSmem];
    char* const s_ring = smem + C::kStages * C::kStage;

    const int tid    = static_cast<int>(threadIdx.x);
    const int lane   = tid & 31;
    const int warp   = tid >> 5;
    const int gid    = lane >> 2;
    const int tig    = lane & 3;
    const int col0   = static_cast<int>(rows_fast ? blockIdx.y : blockIdx.x) * C::kColumns;
    const int row0   = static_cast<int>(rows_fast ? blockIdx.x : blockIdx.y) * Rows::kOutputRows;

    const auto issue = [&](int g, int buf) {
        char* const s_w = smem + buf * C::kStage;
        char* const s_x = s_w + C::kCodeBytes;
        for (int c = tid; c < C::kRows * 2; c += C::kThreads) {
            const int r = c >> 1;
            cp_async<16, Cache::cg>(
                s_w + r * 32 + (c & 1) * 16,
                w_codes + static_cast<std::size_t>(Rows::weight_row(row0, r, up_offset)) * kRowBytes +
                    g * 32 + (c & 1) * 16);
        }
        for (int c = tid; c < C::kColumns * 8; c += C::kThreads) {
            const int col = c >> 3;
            cp_async<16, Cache::cg>(s_x + col * C::kXStride + (c & 7) * 16,
                                    x + static_cast<std::size_t>(col0 + col) * kCols +
                                        g * C::kGroupK + (c & 7) * 16);
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

    float acc[C::kMTiles][C::kNTiles][4];
#pragma unroll
    for (int m = 0; m < C::kMTiles; ++m) {
#pragma unroll
        for (int n = 0; n < C::kNTiles; ++n) {
#pragma unroll
            for (int j = 0; j < 4; ++j) { acc[m][n][j] = 0.0F; }
        }
    }

#pragma unroll
    for (int i = 0; i < C::kStages - 1; ++i) { issue(i, i); }

    for (int g = 0; g < kGroups; ++g) {
        cp_wait<C::kStages - 2>();
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
            const char* p  = s_x + col * C::kXStride + tig * 32;
            const uint4 v0 = load_vec<uint4>(p);
            const uint4 v1 = load_vec<uint4>(p + 16);
            b[0]           = v0.x;
            b[1]           = v0.y;
            b[2]           = v0.z;
            b[3]           = v0.w;
            b[4]           = v1.x;
            b[5]           = v1.y;
            b[6]           = v1.z;
            b[7]           = v1.w;
        }, acc);
    }

    t2_prefill_store(epilogue, acc, xs, row0, col0, warp, gid, tig, tokens);
}

// One block per padded column: the token's amax over all of K sets its one scale, then a warp per
// 128-k group and a lane per four k store the codes, each warp holding its groups' inputs from the
// amax pass. Padded columns get zero codes and scales; a zero token keeps a zero scale and zero
// codes.
template <std::int32_t kCols>
__global__ __launch_bounds__(T2PrefillI8::kQuantThreads) void t2_prefill_i8_quantize_kernel(
    const __nv_bfloat16* __restrict__ x, std::int32_t tokens, std::int8_t* __restrict__ codes,
    __half* __restrict__ scales) {
    using C                 = T2PrefillI8;
    constexpr int kGroups   = kCols / C::kGroupK;
    constexpr int kWarps    = C::kQuantThreads / 32;
    constexpr int kPerWarp  = kGroups / kWarps;
    static_assert(kGroups % kWarps == 0, "every warp quantizes the same number of groups");
    __shared__ float maxima[kWarps];
    const int t            = static_cast<int>(blockIdx.x);
    const int lane         = static_cast<int>(threadIdx.x) & 31;
    const int warp         = static_cast<int>(threadIdx.x) >> 5;
    std::int8_t* const dst = codes + static_cast<std::size_t>(t) * kCols;
    if (t >= tokens) {
        for (int i = static_cast<int>(threadIdx.x); i < kCols / 16; i += C::kQuantThreads) {
            store_vec(dst + i * 16, make_uint4(0u, 0u, 0u, 0u));
        }
        if (threadIdx.x == 0) { scales[t] = __float2half(0.0F); }
        return;
    }
    const __nv_bfloat16* const src = x + static_cast<std::size_t>(t) * kCols;
    float f[kPerWarp][4];
    float amax = 0.0F;
#pragma unroll
    for (int i = 0; i < kPerWarp; ++i) {
        const uint2 raw = load_vec<uint2>(src + (warp + i * kWarps) * C::kGroupK + lane * 4);
        const auto* v   = reinterpret_cast<const __nv_bfloat16*>(&raw);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            f[i][j] = __bfloat162float(v[j]);
            amax    = fmaxf(amax, fabsf(f[i][j]));
        }
    }
    amax = warp_max(amax);
    if (lane == 0) { maxima[warp] = amax; }
    __syncthreads();
#pragma unroll
    for (int w = 0; w < kWarps; ++w) { amax = fmaxf(amax, maxima[w]); }
    const __half scale = __float2half(amax / 127.0F);
    if (threadIdx.x == 0) { scales[t] = scale; }
    const float s   = __half2float(scale);
    const float inv = s > 0.0F ? 1.0F / s : 0.0F;
    // Lane l writes bytes [4 l, 4 l + 4) of the group, which t2_prefill_permute16 fills from the
    // even (lanes 4q, 4q+2) or odd (4q+1, 4q+3) codes of lanes 4q + (l & 2) and the one after.
    const int source = lane & ~1;
    const int shift  = (lane & 1) * 16;
#pragma unroll
    for (int i = 0; i < kPerWarp; ++i) {
        unsigned q[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            q[j] = static_cast<unsigned>(max(-127, min(127, __float2int_rn(f[i][j] * inv)))) & 0xffu;
        }
        const unsigned packed = q[0] | (q[2] << 8) | (q[1] << 16) | (q[3] << 24);
        const unsigned a      = __shfl_sync(0xffffffffu, packed, source);
        const unsigned b      = __shfl_sync(0xffffffffu, packed, source + 1);
        const unsigned word   = ((a >> shift) & 0xffffu) | (((b >> shift) & 0xffffu) << 16);
        store_vec(dst + (warp + i * kWarps) * C::kGroupK + lane * 4, word);
    }
}

} // namespace ninfer::ops::detail
