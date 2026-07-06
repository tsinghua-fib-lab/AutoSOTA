#include <torch/extension.h>

#include <cstdint>

#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cute/tensor.hpp>

namespace {

__device__ __forceinline__ void kappa_cp_async_commit() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm volatile("cp.async.commit_group;\n" ::);
#endif
}

__device__ __forceinline__ void kappa_cp_async_wait() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm volatile("cp.async.wait_group 0;\n" ::);
#endif
}

__device__ __forceinline__ void kappa_cp_async_copy(
    float* smem_ptr,
    const float* gmem_ptr,
    bool pred) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  unsigned int smem_addr = __cvta_generic_to_shared(smem_ptr);
  unsigned int pred_int = pred ? 1u : 0u;
  asm volatile(
      "{ .reg .pred p; setp.ne.b32 p, %0, 0; "
      "@p cp.async.ca.shared.global [%1], [%2], 4; }\n"
      :
      : "r"(pred_int), "r"(smem_addr), "l"(gmem_ptr));
  if (!pred) {
    *smem_ptr = 0.0f;
  }
#else
  if (pred) {
    *smem_ptr = *gmem_ptr;
  } else {
    *smem_ptr = 0.0f;
  }
#endif
}

template<int B, int TC, typename LayoutS>
__device__ __forceinline__ void kappa_load_tile_rowmajor_async(
    const float* A,
    int n,
    int in_row0,
    int col0,
    float* sA,
    const LayoutS& sA_layout) {
  // Asynchronous (cp.async) load of a BxTC tile into shared memory.
  int tid = static_cast<int>(threadIdx.y) * blockDim.x + threadIdx.x;
  int stride = blockDim.x * blockDim.y;
  int total = B * TC;
  for (int idx = tid; idx < total; idx += stride) {
    int r = idx / TC;
    int c = idx - r * TC;
    int col = col0 + c;
    int64_t offset = static_cast<int64_t>(in_row0 + r) * static_cast<int64_t>(n) + col;
    float* smem_ptr = &sA[sA_layout(r, c)];
    const float* gmem_ptr = A + offset;
    bool pred = col < n;
    kappa_cp_async_copy(smem_ptr, gmem_ptr, pred);
  }
  kappa_cp_async_commit();
}

template<int B, int KAPPA, int S, int TC, int BR>
__global__ void flashblockrow_forward_kernel_static(
    const float* A,
    int n,
    int d_total,
    int k_total,
    float alpha,
    uint64_t seed,
    float* y) {
  using namespace cute;

  static_assert(TC == 32, "flashblockrow_cuda requires TC=32 for warp-sized tiles.");
  static_assert(B % BR == 0, "flashblockrow_cuda requires B divisible by BR.");

  int col0 = blockIdx.y * TC;
  int lane = threadIdx.x;
  int warp_row = threadIdx.y;
  // Each warp-row corresponds to a row within a BR-sized output block.
  if (warp_row >= BR) {
    return;
  }

  int row_tiles = B / BR;
  int g = blockIdx.x / row_tiles;
  int row_tile = blockIdx.x - g * row_tiles;
  int out_row = g * B + row_tile * BR + warp_row;
  if (out_row >= k_total) {
    return;
  }

  int d_blocks = d_total / B;
  int block_mask = d_blocks - 1;

  constexpr int kStride = TC;

  // Double-buffered shared tiles for A and a small table of input block ids.
  __shared__ __align__(16) float sA0[B * kStride];
  __shared__ __align__(16) float sA1[B * kStride];
  __shared__ int block_ids[KAPPA];

  auto sA_layout =
      make_layout(make_shape(Int<B>{}, Int<kStride>{}), make_stride(Int<kStride>{}, Int<1>{}));

  if (lane == 0 && warp_row == 0) {
    // Pick KAPPA input blocks using a simple affine sequence over power-of-two blocks.
    curandStatePhilox4_32_10_t block_state;
    curand_init(seed ^ 0x9e3779b97f4a7c15ULL, static_cast<uint64_t>(g), 0, &block_state);
    uint32_t start = curand(&block_state) & static_cast<uint32_t>(block_mask);
    uint32_t stride = curand(&block_state) & static_cast<uint32_t>(block_mask);
    stride |= 1u;
    #pragma unroll
    for (int t = 0; t < KAPPA; ++t) {
      block_ids[t] = static_cast<int>((start + t * stride) & block_mask);
    }
  }
  __syncthreads();

  float acc = 0.0f;

  float* sA_curr = sA0;
  float* sA_next = sA1;

  curandStatePhilox4_32_10_t row_state;
  if (lane == 0) {
    // Per-output-row RNG state (one lane drives sample choices).
    curand_init(seed ^ 0x94d049bb133111ebULL, static_cast<uint64_t>(out_row), 0, &row_state);
  }

  kappa_load_tile_rowmajor_async<B, TC>(
      A, n, block_ids[0] * B, col0, sA_curr, sA_layout);
  kappa_cp_async_wait();
  __syncthreads();

  unsigned mask = 0xffffffffu;
  int col = col0 + lane;

  #pragma unroll
  for (int t = 0; t < KAPPA; ++t) {
    #pragma unroll
    for (int ell = 0; ell < S; ++ell) {
      int u = 0;
      int sign = 1;
      if (lane == 0) {
        // Lane 0 selects a row in the current BxTC tile and a sign.
        uint32_t rnd = curand(&row_state);
        u = static_cast<int>(rnd & static_cast<uint32_t>(B - 1));
        sign = (rnd & 0x80000000u) ? 1 : -1;
      }
      // Broadcast u/sign to the rest of the warp.
      u = __shfl_sync(mask, u, 0);
      sign = __shfl_sync(mask, sign, 0);
      float v = sA_curr[sA_layout(u, lane)];
      acc += static_cast<float>(sign) * v;
    }

    if (t + 1 < KAPPA) {
      // Prefetch the next input block tile while finishing this one.
      kappa_load_tile_rowmajor_async<B, TC>(
          A, n, block_ids[t + 1] * B, col0, sA_next, sA_layout);
    }

    if (t + 1 < KAPPA) {
      kappa_cp_async_wait();
      __syncthreads();
      float* tmp = sA_curr;
      sA_curr = sA_next;
      sA_next = tmp;
    }
  }

  if (col < n) {
    // Final scaled output for this row/column.
    y[out_row * n + col] = alpha * acc;
  }
}

}  // namespace

torch::Tensor flashblockrow_forward_cuda(
    torch::Tensor x,
    int64_t k_total,
    int64_t block_size,
    int64_t kappa,
    int64_t s,
    int64_t tc,
    double alpha,
    uint64_t seed) {
  const auto d_total = static_cast<int>(x.size(0));
  const auto n = static_cast<int>(x.size(1));
  const auto k_total_i = static_cast<int>(k_total);
  const auto block_size_i = static_cast<int>(block_size);
  const auto kappa_i = static_cast<int>(kappa);
  const auto s_i = static_cast<int>(s);
  const auto tc_i = static_cast<int>(tc);
  const float alpha_f = static_cast<float>(alpha);
  const int d_blocks = d_total / block_size_i;

  auto y = torch::zeros({k_total_i, n}, x.options());

  constexpr int kBlockRows = 4;
  const int k_blocks = k_total_i / block_size_i;
  const int row_tiles = block_size_i / kBlockRows;
  const int col_tiles = (n + tc_i - 1) / tc_i;

  // blocks.x enumerates output row tiles, blocks.y enumerates column tiles.
  dim3 blocks(k_blocks * row_tiles, col_tiles);
  dim3 threads(tc_i, kBlockRows);

  cudaStream_t stream = at::cuda::getDefaultCUDAStream();
  TORCH_CHECK(block_size_i == 128, "flashblockrow_cuda requires block_size=128");
  const bool ks_1_1 = (kappa_i == 1 && s_i == 1);
  const bool ks_2_1 = (kappa_i == 2 && s_i == 1);
  const bool ks_4_1 = (kappa_i == 4 && s_i == 1);
  const bool ks_1_2 = (kappa_i == 1 && s_i == 2);
  const bool ks_2_2 = (kappa_i == 2 && s_i == 2);
  const bool ks_1_4 = (kappa_i == 1 && s_i == 4);
  TORCH_CHECK(
      ks_1_1 || ks_2_1 || ks_4_1 || ks_1_2 || ks_2_2 || ks_1_4,
      "flashblockrow_cuda requires (kappa,s) in {(1,1),(2,1),(4,1),(1,2),(2,2),(1,4)}");
  TORCH_CHECK(tc_i == 32, "flashblockrow_cuda requires tc=32");
  TORCH_CHECK(d_total % block_size_i == 0, "flashblockrow_cuda requires d_total divisible by block_size");
  TORCH_CHECK((d_blocks & (d_blocks - 1)) == 0, "flashblockrow_cuda requires power-of-two block count");
  TORCH_CHECK(block_size_i % kBlockRows == 0, "flashblockrow_cuda requires block_size divisible by block_rows");

  if (ks_1_1) {
    flashblockrow_forward_kernel_static<128, 1, 1, 32, 4><<<blocks, threads, 0, stream>>>(
        x.data_ptr<float>(),
        n,
        d_total,
        k_total_i,
        alpha_f,
        seed,
        y.data_ptr<float>());
  } else if (ks_2_1) {
    flashblockrow_forward_kernel_static<128, 2, 1, 32, 4><<<blocks, threads, 0, stream>>>(
        x.data_ptr<float>(),
        n,
        d_total,
        k_total_i,
        alpha_f,
        seed,
        y.data_ptr<float>());
  } else if (ks_4_1) {
    flashblockrow_forward_kernel_static<128, 4, 1, 32, 4><<<blocks, threads, 0, stream>>>(
        x.data_ptr<float>(),
        n,
        d_total,
        k_total_i,
        alpha_f,
        seed,
        y.data_ptr<float>());
  } else if (ks_1_2) {
    flashblockrow_forward_kernel_static<128, 1, 2, 32, 4><<<blocks, threads, 0, stream>>>(
        x.data_ptr<float>(),
        n,
        d_total,
        k_total_i,
        alpha_f,
        seed,
        y.data_ptr<float>());
  } else if (ks_2_2) {
    flashblockrow_forward_kernel_static<128, 2, 2, 32, 4><<<blocks, threads, 0, stream>>>(
        x.data_ptr<float>(),
        n,
        d_total,
        k_total_i,
        alpha_f,
        seed,
        y.data_ptr<float>());
  } else if (ks_1_4) {
    flashblockrow_forward_kernel_static<128, 1, 4, 32, 4><<<blocks, threads, 0, stream>>>(
        x.data_ptr<float>(),
        n,
        d_total,
        k_total_i,
        alpha_f,
        seed,
        y.data_ptr<float>());
  }

  return y;
}
