#pragma once

#include <iostream>
#include <cstdint>

#include <cuda_bf16.h>

#define CUDA_CHECK(x)                                                                                                  \
  {                                                                                                                    \
    auto error = x;                                                                                                    \
    if (error != cudaSuccess) {                                                                                        \
      std::cerr << "CUDA error - L" << __LINE__ << ": " << cudaGetErrorString(error) << std::endl;                     \
      exit(1);                                                                                                         \
    }                                                                                                                  \
  }

inline constexpr int WARP_SIZE = 32;

__device__ __host__ constexpr int cdiv(int a, int b) {
  return (a + b - 1) / b;
}

// NOTE: stride in bytes
template <int STRIDE> __device__ uint32_t swizzle(uint32_t index) {
  // no need swizzling
  if constexpr (STRIDE == 16)
    return index;

  uint32_t row_idx = (index / STRIDE) % 8;
  uint32_t bits_to_xor = row_idx / max(64 / STRIDE, 1);
  return index ^ (bits_to_xor << 4);
}

template <int HEIGHT, int WIDTH, int TB_SIZE>
__device__ inline void global_to_shared_swizzle_zero_pad(uint32_t dst, const nv_bfloat16* src, int src_stride, int tid,
                                                         int kv_id, int low_bound, int up_bound) {
  constexpr int num_elems = 16 / sizeof(nv_bfloat16);
  constexpr int num_iters = HEIGHT * WIDTH / (TB_SIZE * num_elems);

  const int kv_offset = kv_id * HEIGHT;
  const int zero = 0x0;
#pragma unroll
  for (int iter = 0; iter < num_iters; iter++) {
    const int idx = (iter * TB_SIZE + tid) * num_elems;
    const int row = idx / WIDTH;
    const int col = idx % WIDTH;
    const int global_row = kv_offset + row;

    const uint32_t dst_addr = swizzle<WIDTH * sizeof(nv_bfloat16)>(dst + (row * WIDTH + col) * sizeof(nv_bfloat16));

    if ((global_row < low_bound) || (global_row >= up_bound)) {
      asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};"
                   :
                   : "r"(dst_addr), "r"(zero), "r"(zero), "r"(zero), "r"(zero));

    } else {
      const nv_bfloat16* src_addr = src + (row * src_stride + col);
      asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(dst_addr), "l"(src_addr));
    }
  }
}

template <int HEIGHT, int WIDTH, int TB_SIZE>
__device__ inline void global_to_shared_swizzle_zero_pad_with_mask(uint32_t dst, const nv_bfloat16* src, int src_stride,
                                                                   int tid, int kv_id, int low_bound, int up_bound,
                                                                   const int* idx_map) {
  constexpr int num_elems = 16 / sizeof(nv_bfloat16);
  constexpr int num_iters = HEIGHT * WIDTH / (TB_SIZE * num_elems);

  const int kv_offset = kv_id * HEIGHT;
  const int zero = 0x0;
#pragma unroll
  for (int iter = 0; iter < num_iters; iter++) {
    const int idx = (iter * TB_SIZE + tid) * num_elems;
    const int row = idx / WIDTH;
    const int col = idx % WIDTH;
    const int global_row = kv_offset + row;
    const int idx_val = idx_map[global_row];

    const uint32_t dst_addr = swizzle<WIDTH * sizeof(nv_bfloat16)>(dst + (row * WIDTH + col) * sizeof(nv_bfloat16));

    if ((idx_val == -1) || ((global_row < low_bound) || (global_row >= up_bound))) {
      asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};"
                   :
                   : "r"(dst_addr), "r"(zero), "r"(zero), "r"(zero), "r"(zero));

    } else {
      const nv_bfloat16* src_addr = src + (row * src_stride + col);
      asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(dst_addr), "l"(src_addr));
    }
  }
}

__device__ inline void ldmatrix_x2(uint32_t regs[2], uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];" : "=r"(regs[0]), "=r"(regs[1]) : "r"(addr));
}

__device__ inline void ldmatrix_x4(uint32_t regs[4], uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
               : "=r"(regs[0]), "=r"(regs[1]), "=r"(regs[2]), "=r"(regs[3])
               : "r"(addr));
}

__device__ inline void ldmatrix_x2_trans(uint32_t regs[2], uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];"
               : "=r"(regs[0]), "=r"(regs[1])
               : "r"(addr));
}

__device__ inline void ldmatrix_x4_trans(uint32_t regs[4], uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];"
               : "=r"(regs[0]), "=r"(regs[1]), "=r"(regs[2]), "=r"(regs[3])
               : "r"(addr));
}

__device__ inline void mma_m16n8k16(uint32_t A[4], uint32_t B[2], float D[4]) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
               "{%0, %1, %2, %3}, "
               "{%4, %5, %6, %7}, "
               "{%8, %9}, "
               "{%10, %11, %12, %13};"
               : "=f"(D[0]), "=f"(D[1]), "=f"(D[2]), "=f"(D[3])
               : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]), "f"(D[0]), "f"(D[1]), "f"(D[2]),
                 "f"(D[3]));
}

template <typename T, typename... Args>
void launch_kernel(T* kernel, int num_blocks, int block_size, int smem_size, Args... args) {
  if (smem_size > 48'000)
    CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
  kernel<<<num_blocks, block_size, smem_size>>>(args...);
  CUDA_CHECK(cudaGetLastError());
}