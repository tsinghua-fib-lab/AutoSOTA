#pragma once

#include <cuda_bf16.h>
#include <cstdint>

namespace cutlass_mma {

// ============================================================================
// H100 (SM90) optimized wgmma-style MMA with better instruction interleaving
// ============================================================================

// Fused Q@K^T MMA with interleaved execution for maximum ILP
// This version issues multiple independent MMAs before waiting for results
template <int N_Q, int N_KV, int N_D>
__device__ __forceinline__ void fused_mma_qk_ilp(uint32_t Q[N_Q][N_D][4], uint32_t K[N_KV][N_D][2],
                                                 float S[N_Q][N_KV][4]) {
// Interleave MMA operations across different output tiles
// This allows the hardware to overlap MMA execution with register access
#pragma unroll
  for (int mma_d = 0; mma_d < N_D; ++mma_d) {
// Issue all Q x K MMAs for this dimension slice in sequence
// Hardware will overlap independent operations
#pragma unroll
    for (int mma_q = 0; mma_q < N_Q; ++mma_q) {
#pragma unroll
      for (int mma_kv = 0; mma_kv < N_KV; ++mma_kv) {
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
            "{%0, %1, %2, %3}, "
            "{%4, %5, %6, %7}, "
            "{%8, %9}, "
            "{%10, %11, %12, %13};"
            : "=f"(S[mma_q][mma_kv][0]), "=f"(S[mma_q][mma_kv][1]), "=f"(S[mma_q][mma_kv][2]), "=f"(S[mma_q][mma_kv][3])
            : "r"(Q[mma_q][mma_d][0]), "r"(Q[mma_q][mma_d][1]), "r"(Q[mma_q][mma_d][2]), "r"(Q[mma_q][mma_d][3]),
              "r"(K[mma_kv][mma_d][0]), "r"(K[mma_kv][mma_d][1]), "f"(S[mma_q][mma_kv][0]), "f"(S[mma_q][mma_kv][1]),
              "f"(S[mma_q][mma_kv][2]), "f"(S[mma_q][mma_kv][3]));
      }
    }
  }
}

// Fused P@V MMA with better loop ordering for register reuse
template <int N_Q, int N_D, int N_KV>
__device__ __forceinline__ void fused_mma_pv_ilp(uint32_t P[N_Q][N_KV][4], uint32_t V[N_KV][N_D][2],
                                                 float O[N_Q][N_D][4]) {
// Loop order: kv (accumulation) -> d -> q
// This keeps V registers in cache longer
#pragma unroll
  for (int mma_kv = 0; mma_kv < N_KV; ++mma_kv) {
#pragma unroll
    for (int mma_d = 0; mma_d < N_D; ++mma_d) {
#pragma unroll
      for (int mma_q = 0; mma_q < N_Q; ++mma_q) {
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
            "{%0, %1, %2, %3}, "
            "{%4, %5, %6, %7}, "
            "{%8, %9}, "
            "{%10, %11, %12, %13};"
            : "=f"(O[mma_q][mma_d][0]), "=f"(O[mma_q][mma_d][1]), "=f"(O[mma_q][mma_d][2]), "=f"(O[mma_q][mma_d][3])
            : "r"(P[mma_q][mma_kv][0]), "r"(P[mma_q][mma_kv][1]), "r"(P[mma_q][mma_kv][2]), "r"(P[mma_q][mma_kv][3]),
              "r"(V[mma_kv][mma_d][0]), "r"(V[mma_kv][mma_d][1]), "f"(O[mma_q][mma_d][0]), "f"(O[mma_q][mma_d][1]),
              "f"(O[mma_q][mma_d][2]), "f"(O[mma_q][mma_d][3]));
      }
    }
  }
}

// Original functions for backward compatibility
template <int N_Q, int N_KV, int N_D>
__device__ __forceinline__ void batched_mma_qk(uint32_t Q[N_Q][N_D][4], uint32_t K[N_KV][N_D][2],
                                               float S[N_Q][N_KV][4]) {
  fused_mma_qk_ilp<N_Q, N_KV, N_D>(Q, K, S);
}

template <int N_Q, int N_D, int N_KV>
__device__ __forceinline__ void batched_mma_pv(uint32_t P[N_Q][N_KV][4], uint32_t V[N_KV][N_D][2],
                                               float O[N_Q][N_D][4]) {
  fused_mma_pv_ilp<N_Q, N_D, N_KV>(P, V, O);
}

} // namespace cutlass_mma
