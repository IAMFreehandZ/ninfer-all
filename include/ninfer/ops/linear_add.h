#pragma once

// ninfer::ops - fused residual += W @ x.

#include "core/weight.h"
#include "core/arena.h"
#include "core/tensor.h"
#include "ninfer/ops/linear.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops {

/**
 * Returns the A16-only transient capacity required by LinearAdd for every T in the inclusive
 * [min_tokens,max_tokens] interval. The QType and dimensions are the fixed implementation profile.
 * Invalid profiles or intervals throw; a legal static-zero route returns zero.
 */
[[nodiscard]] std::size_t linear_add_workspace_capacity_bytes(QType qtype, std::int32_t output_rows,
                                                              std::int32_t input_rows,
                                                              std::int32_t min_tokens,
                                                              std::int32_t max_tokens);

[[nodiscard]] std::size_t linear_add_workspace_capacity_bytes(QType qtype, std::int32_t output_rows,
                                                              std::int32_t input_rows,
                                                              LinearPolicy policy,
                                                              std::int32_t min_tokens,
                                                              std::int32_t max_tokens);

/**
 * Op: linear_add
 *
 * Math / indexing:
 *   ideal[:,t] = residual[:,t] + Linear(x,w)[:,t].
 *
 * Logical shapes:
 *   Contiguous BF16 x [K,T] and residual [N,T]. Registered weights are Q4_G64_FP16 RowSplit
 *   [5120,6144], Q5_G64_FP16 RowSplit [5120,17408] or [5120,6144], Q8_G32_FP16 RowSplit
 *   [2048,4096], [2048,6144], [5120,6144] or [5120,17408], NVFP4
 *   BlockScaleK16M128x4 [5120,6144] or [5120,17408], row-scaled
 *   FP8_E4M3FN_ROW_BF16 [5120,6144] or [5120,17408], or BF16 Contiguous [5120,6144]. T may
 *   be any positive value. T2_G128_FP16 RowSplit uses FP16 scales for 128-element groups and
 *   registers [N,5120] for N in {1024,4096,6144,7168,12288,14336,16384,34816,131072,248320},
 *   plus [5120,6144] and [5120,17408].
 *
 * Numeric:
 *   The oracle reads a registered BF16 weight directly or exact-decodes a registered packed
 *   weight, then evaluates `ideal` naively in FP64 from the represented inputs. The updated BF16
 *   residual is promoted and compared directly with that result; output storage rounding belongs
 *   to LinearAdd's selected A16, A8, or A4 criterion, not the oracle. Production routes may fuse
 *   or materialize the projection and may choose their natural accumulator, activation
 *   quantization, staging, and workspace precision; those private choices are not semantic
 *   rounding boundaries.
 *
 * Compute policy:
 *   All policies permit the A16 implementations of Q4, Q5, Q8 and BF16. NVFP4 uses A16 for
 *   A16Only/AllowA8 and may use A4 under AllowA4. FP8 may use A8 under AllowA8/AllowA4.
 *   T2 permits A16 under every policy and may use integer A8 under AllowA8Int,
 *   AllowA8IntDecode, or AllowPrefillCublas.
 *   Each registration owns its production plan. A permissive policy
 *   allows the private resolver to select either qualified
 *   arithmetic profile; it does not itself prescribe a kernel.
 *
 * Effects:
 *   Updates the full residual tensor in place; x/weight must not alias residual.
 *
 * Workspace:
 *   Caller-owned transient storage reported by linear_add_workspace_capacity_bytes(), scoped to
 *   the call. T2 A16 materializes a BF16 projection before adding the residual; its capacity query
 *   also covers the admitted integer A8 routes. Other registered A16 routes require no storage;
 *   quantized-activation routes use the reported capacity. There is no persistent state side effect.
 */
void linear_add(const Tensor& x, const Weight& w, Tensor& residual, WorkspaceArena& ws,
                cudaStream_t stream);

void linear_add(const Tensor& x, const Weight& w, Tensor& residual, LinearPolicy policy,
                WorkspaceArena& ws, cudaStream_t stream);

/**
 * Returns the transient capacity swiglu_hadamard_linear_add requires for every T in the inclusive
 * [min_tokens,max_tokens] interval: the composition's BF16 [gate_up_rows, T] plane, linear's
 * capacity, the BF16 [gate_up_rows / 2, T] transform and linear_add's capacity, for the given
 * profiles and policies. The fused route needs less.
 */
[[nodiscard]] std::size_t swiglu_hadamard_linear_add_workspace_capacity_bytes(
    QType gate_up_qtype, std::int32_t gate_up_rows, std::int32_t input_rows,
    LinearPolicy gate_up_policy, QType down_qtype, std::int32_t output_rows,
    LinearPolicy down_policy, std::int32_t min_tokens, std::int32_t max_tokens);

/**
 * Op: swiglu_hadamard_linear_add
 *
 * Math / indexing:
 *   residual[:,t] += Linear(a, down)[:,t] with a = silu_mul_hadamard(Linear(x, gate_up), signs):
 *   the Dense SwiGLU FFN of a Hadamard-rotated checkpoint whose down projection takes its input in
 *   the rotated basis, from the (already rotated) FFN input x.
 *
 * Logical shapes:
 *   x contiguous BF16 [gate_up.k, T]; gate_up [2I, K] holding the gate rows then the up rows;
 *   signs contiguous BF16 [I] of +1 / -1; down [N, I] with I a multiple of 1024; residual
 *   contiguous BF16 [N, T]. Each weight is any registration of linear / linear_add.
 *
 * Numeric:
 *   Bit-identical to linear(x, gate_up) into a BF16 [2I, T] plane, silu_mul_hadamard into a BF16
 *   [I, T] tensor and linear_add, under the given policies. Where both weights are T2 and T takes
 *   the integer prefill tile, the gate_up tile writes silu_mul of its BF16-rounded gate and up
 *   rows, and the down projection's input is quantised in the kernel that applies the transform;
 *   elsewhere the three Ops run in sequence.
 *
 * Effects:
 *   Updates residual in place; x, signs and both weights are preserved and must not overlap it.
 *
 * Workspace:
 *   swiglu_hadamard_linear_add_workspace_capacity_bytes(), scoped to the call. No persistent state
 *   side effect.
 */
void swiglu_hadamard_linear_add(const Tensor& x, const Weight& gate_up, LinearPolicy gate_up_policy,
                                const Tensor& signs, const Weight& down, LinearPolicy down_policy,
                                Tensor& residual, WorkspaceArena& ws, cudaStream_t stream);

} // namespace ninfer::ops
