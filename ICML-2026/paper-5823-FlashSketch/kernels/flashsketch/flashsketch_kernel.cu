#include <torch/extension.h>

#include <cmath>
#include <cstdint>

#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cute/tensor.hpp>

namespace {

__device__ __forceinline__ uint32_t block_perm_mix32(uint32_t x) {
  x ^= x >> 16;
  x *= 0x7feb352dU;
  x ^= x >> 15;
  x *= 0x846ca68bU;
  x ^= x >> 16;
  return x;
}

template<int Br>
__device__ __forceinline__ int block_perm_mod_br(uint32_t x) {
  static_assert((Br & (Br - 1)) == 0, "Br must be a power of two");
  return static_cast<int>(x & static_cast<uint32_t>(Br - 1));
}

template<int Br, int Tn, int Tk, int KAPPA, int S, bool SkipZeros>
__global__ __launch_bounds__(256, 2) void block_perm_kernel(
    const float* __restrict__ A,
    float* __restrict__ Y,
    int d_total,
    int n,
    int k_total,
    int M,
    int Bc,
    float scale,
    uint32_t seed,
    int affine_a,
    int affine_b) {
  using namespace cute;

  const int g = static_cast<int>(blockIdx.x);
  const int col0 = static_cast<int>(blockIdx.y) * Tn;
  const int tid = static_cast<int>(threadIdx.x);
  const int threads = static_cast<int>(blockDim.x);
  const int64_t affine_a64 = static_cast<int64_t>(affine_a);
  const int64_t affine_b64 = static_cast<int64_t>(affine_b);
  const uint32_t g_term = static_cast<uint32_t>(g) * 0x9e3779b9U;
  static_assert(Tn % 4 == 0, "Tn must be divisible by 4");
  const bool full_n_tile = (col0 + Tn) <= n;
  const bool vec_ok = full_n_tile && ((n & 3) == 0) && ((col0 & 3) == 0);

  extern __shared__ float shared[];
  float* sA_ptr = shared;
  float* sY_ptr = sA_ptr + Tk * Tn;

  auto sA = make_tensor(
      make_smem_ptr(sA_ptr),
      make_layout(make_shape(Int<Tk>{}, Int<Tn>{}), make_stride(Int<Tn>{}, Int<1>{})));
  auto sY = make_tensor(
      make_smem_ptr(sY_ptr),
      make_layout(make_shape(Int<Br>{}, Int<Tn>{}), make_stride(Int<Tn>{}, Int<1>{})));

  for (int idx = tid; idx < Br * Tn; idx += threads) {
    sY_ptr[idx] = 0.0f;
  }
  __syncthreads();

  auto gA = make_tensor(
      make_gmem_ptr(A),
      make_layout(make_shape(d_total, n), make_stride(n, 1)));
  auto gY = make_tensor(
      make_gmem_ptr(Y),
      make_layout(make_shape(k_total, n), make_stride(n, 1)));

  int h = static_cast<int>((affine_a64 * g + affine_b64) % M);
  for (int ell = 0; ell < KAPPA; ++ell) {
    int h_use = h;
    const uint32_t h_term = static_cast<uint32_t>(h_use) * 0x85ebca6bU;
    int base_row = h_use * Bc;
    for (int u0 = 0; u0 < Bc; u0 += Tk) {
      const bool full_u_tile = (u0 + Tk) <= Bc;
      if (vec_ok && full_u_tile) {
        constexpr int kVec = 4;
        const int vec_cols = Tn / kVec;
        for (int idx = tid; idx < Tk * vec_cols; idx += threads) {
          int u = idx / vec_cols;
          int j4 = idx - u * vec_cols;
          int u_global = u0 + u;
          int row = base_row + u_global;
          int col = col0 + j4 * kVec;
          const float4* src = reinterpret_cast<const float4*>(A + row * n + col);
          float4 v = src[0];
          int base = u * Tn + j4 * kVec;
          sA_ptr[base + 0] = v.x;
          sA_ptr[base + 1] = v.y;
          sA_ptr[base + 2] = v.z;
          sA_ptr[base + 3] = v.w;
        }
      } else {
        for (int idx = tid; idx < Tk * Tn; idx += threads) {
          int u = idx / Tn;
          int j = idx - u * Tn;
          int u_global = u0 + u;
          int col = col0 + j;
          float val = 0.0f;
          if (u_global < Bc && col < n) {
            int row = base_row + u_global;
            val = gA(row, col);
          }
          sA(u, j) = val;
        }
      }
      __syncthreads();

      if (full_n_tile && full_u_tile) {
        for (int idx = tid; idx < Tk * Tn; idx += threads) {
          int u = idx / Tn;
          int j = idx - u * Tn;
          int u_global = u0 + u;
          float val = sA(u, j);
          if constexpr (SkipZeros) {
            if (val == 0.0f) {
              continue;
            }
          }
          uint32_t base = seed ^ g_term ^ h_term ^
              (static_cast<uint32_t>(u_global) * 0xc2b2ae35U);
          uint32_t base_hash = block_perm_mix32(base);
          uint32_t b = base_hash & static_cast<uint32_t>(Br - 1);
          uint32_t a = ((base_hash >> 8) | 1u) & static_cast<uint32_t>(Br - 1);
          #pragma unroll
          for (int t = 0; t < S; ++t) {
            uint32_t key = block_perm_mix32(base ^ (static_cast<uint32_t>(t) * 0x27d4eb2fU));
            int r = static_cast<int>((a * static_cast<uint32_t>(t) + b) &
                static_cast<uint32_t>(Br - 1));
            float sign = (key >> 31) ? 1.0f : -1.0f;
            atomicAdd(&sY(r, j), sign * val);
          }
        }
      } else {
        for (int idx = tid; idx < Tk * Tn; idx += threads) {
          int u = idx / Tn;
          int j = idx - u * Tn;
          int u_global = u0 + u;
          int col = col0 + j;
          if (u_global >= Bc || col >= n) {
            continue;
          }
          float val = sA(u, j);
          if constexpr (SkipZeros) {
            if (val == 0.0f) {
              continue;
            }
          }
          uint32_t base = seed ^ g_term ^ h_term ^
              (static_cast<uint32_t>(u_global) * 0xc2b2ae35U);
          uint32_t base_hash = block_perm_mix32(base);
          uint32_t b = base_hash & static_cast<uint32_t>(Br - 1);
          uint32_t a = ((base_hash >> 8) | 1u) & static_cast<uint32_t>(Br - 1);
          #pragma unroll
          for (int t = 0; t < S; ++t) {
            uint32_t key = block_perm_mix32(base ^ (static_cast<uint32_t>(t) * 0x27d4eb2fU));
            int r = static_cast<int>((a * static_cast<uint32_t>(t) + b) &
                static_cast<uint32_t>(Br - 1));
            float sign = (key >> 31) ? 1.0f : -1.0f;
            atomicAdd(&sY(r, j), sign * val);
          }
        }
      }
      __syncthreads();
    }
    if constexpr (KAPPA > 1) {
      h = static_cast<int>((affine_a64 * h + affine_b64) % M);
    }
  }

  if (vec_ok) {
    constexpr int kVec = 4;
    const int vec_cols = Tn / kVec;
    for (int r = tid; r < Br; r += threads) {
      float* out_ptr = Y + (g * Br + r) * n + col0;
      int base = r * Tn;
      for (int j4 = 0; j4 < vec_cols; ++j4) {
        float4 v = make_float4(
            scale * sY_ptr[base + j4 * kVec + 0],
            scale * sY_ptr[base + j4 * kVec + 1],
            scale * sY_ptr[base + j4 * kVec + 2],
            scale * sY_ptr[base + j4 * kVec + 3]);
        if constexpr (SkipZeros) {
          if (v.x == 0.0f && v.y == 0.0f && v.z == 0.0f && v.w == 0.0f) {
            continue;
          }
        }
        reinterpret_cast<float4*>(out_ptr + j4 * kVec)[0] = v;
      }
    }
  } else {
    for (int idx = tid; idx < Br * Tn; idx += threads) {
      int r = idx / Tn;
      int j = idx - r * Tn;
      int out_col = col0 + j;
      if (out_col < n) {
        float out = scale * sY(r, j);
        if constexpr (SkipZeros) {
          if (out == 0.0f) {
            continue;
          }
        }
        gY(g * Br + r, out_col) = out;
      }
    }
  }
}

template<int Br, int Tn, int Tk, int KAPPA, int S, bool SkipZeros>
__global__ __launch_bounds__(256, 2) void block_perm_split_kernel(
    const float* __restrict__ A,
    float* __restrict__ Y,
    int d_total,
    int n,
    int k_total,
    int M,
    int Bc,
    int bc_tile_size,
    float scale,
    uint32_t seed,
    int affine_a,
    int affine_b) {
  using namespace cute;

  const int g = static_cast<int>(blockIdx.x);
  const int col0 = static_cast<int>(blockIdx.y) * Tn;
  const int bc_tile = static_cast<int>(blockIdx.z);
  const int tid = static_cast<int>(threadIdx.x);
  const int threads = static_cast<int>(blockDim.x);
  const int64_t affine_a64 = static_cast<int64_t>(affine_a);
  const int64_t affine_b64 = static_cast<int64_t>(affine_b);
  const uint32_t g_term = static_cast<uint32_t>(g) * 0x9e3779b9U;
  static_assert(Tn % 4 == 0, "Tn must be divisible by 4");
  const bool full_n_tile = (col0 + Tn) <= n;
  const bool vec_ok = full_n_tile && ((n & 3) == 0) && ((col0 & 3) == 0);

  const int bc_start = bc_tile * bc_tile_size;
  int bc_end = bc_start + bc_tile_size;
  if (bc_end > Bc) {
    bc_end = Bc;
  }

  extern __shared__ float shared[];
  float* sA_ptr = shared;
  float* sY_ptr = sA_ptr + Tk * Tn;

  auto sA = make_tensor(
      make_smem_ptr(sA_ptr),
      make_layout(make_shape(Int<Tk>{}, Int<Tn>{}), make_stride(Int<Tn>{}, Int<1>{})));
  auto sY = make_tensor(
      make_smem_ptr(sY_ptr),
      make_layout(make_shape(Int<Br>{}, Int<Tn>{}), make_stride(Int<Tn>{}, Int<1>{})));

  for (int idx = tid; idx < Br * Tn; idx += threads) {
    sY_ptr[idx] = 0.0f;
  }
  __syncthreads();

  auto gA = make_tensor(
      make_gmem_ptr(A),
      make_layout(make_shape(d_total, n), make_stride(n, 1)));

  int h = static_cast<int>((affine_a64 * g + affine_b64) % M);
  for (int ell = 0; ell < KAPPA; ++ell) {
    int h_use = h;
    const uint32_t h_term = static_cast<uint32_t>(h_use) * 0x85ebca6bU;
    int base_row = h_use * Bc;
    for (int u0 = bc_start; u0 < bc_end; u0 += Tk) {
      const bool full_u_tile = (u0 + Tk) <= bc_end;
      if (vec_ok && full_u_tile) {
        constexpr int kVec = 4;
        const int vec_cols = Tn / kVec;
        for (int idx = tid; idx < Tk * vec_cols; idx += threads) {
          int u = idx / vec_cols;
          int j4 = idx - u * vec_cols;
          int u_global = u0 + u;
          int row = base_row + u_global;
          int col = col0 + j4 * kVec;
          const float4* src = reinterpret_cast<const float4*>(A + row * n + col);
          float4 v = src[0];
          int base = u * Tn + j4 * kVec;
          sA_ptr[base + 0] = v.x;
          sA_ptr[base + 1] = v.y;
          sA_ptr[base + 2] = v.z;
          sA_ptr[base + 3] = v.w;
        }
      } else {
        for (int idx = tid; idx < Tk * Tn; idx += threads) {
          int u = idx / Tn;
          int j = idx - u * Tn;
          int u_global = u0 + u;
          int col = col0 + j;
          float val = 0.0f;
          if (u_global < bc_end && col < n) {
            int row = base_row + u_global;
            val = gA(row, col);
          }
          sA(u, j) = val;
        }
      }
      __syncthreads();

      if (full_n_tile && full_u_tile) {
        for (int idx = tid; idx < Tk * Tn; idx += threads) {
          int u = idx / Tn;
          int j = idx - u * Tn;
          int u_global = u0 + u;
          float val = sA(u, j);
          if constexpr (SkipZeros) {
            if (val == 0.0f) {
              continue;
            }
          }
          uint32_t base = seed ^ g_term ^ h_term ^
              (static_cast<uint32_t>(u_global) * 0xc2b2ae35U);
          uint32_t base_hash = block_perm_mix32(base);
          uint32_t b = base_hash & static_cast<uint32_t>(Br - 1);
          uint32_t a = ((base_hash >> 8) | 1u) & static_cast<uint32_t>(Br - 1);
          #pragma unroll
          for (int t = 0; t < S; ++t) {
            uint32_t key = block_perm_mix32(base ^ (static_cast<uint32_t>(t) * 0x27d4eb2fU));
            int r = static_cast<int>((a * static_cast<uint32_t>(t) + b) &
                static_cast<uint32_t>(Br - 1));
            float sign = (key >> 31) ? 1.0f : -1.0f;
            atomicAdd(&sY(r, j), sign * val);
          }
        }
      } else {
        for (int idx = tid; idx < Tk * Tn; idx += threads) {
          int u = idx / Tn;
          int j = idx - u * Tn;
          int u_global = u0 + u;
          int col = col0 + j;
          if (u_global >= bc_end || col >= n) {
            continue;
          }
          float val = sA(u, j);
          if constexpr (SkipZeros) {
            if (val == 0.0f) {
              continue;
            }
          }
          uint32_t base = seed ^ g_term ^ h_term ^
              (static_cast<uint32_t>(u_global) * 0xc2b2ae35U);
          uint32_t base_hash = block_perm_mix32(base);
          uint32_t b = base_hash & static_cast<uint32_t>(Br - 1);
          uint32_t a = ((base_hash >> 8) | 1u) & static_cast<uint32_t>(Br - 1);
          #pragma unroll
          for (int t = 0; t < S; ++t) {
            uint32_t key = block_perm_mix32(base ^ (static_cast<uint32_t>(t) * 0x27d4eb2fU));
            int r = static_cast<int>((a * static_cast<uint32_t>(t) + b) &
                static_cast<uint32_t>(Br - 1));
            float sign = (key >> 31) ? 1.0f : -1.0f;
            atomicAdd(&sY(r, j), sign * val);
          }
        }
      }
      __syncthreads();
    }
    if constexpr (KAPPA > 1) {
      h = static_cast<int>((affine_a64 * h + affine_b64) % M);
    }
  }

  for (int idx = tid; idx < Br * Tn; idx += threads) {
    int r = idx / Tn;
    int j = idx - r * Tn;
    int out_col = col0 + j;
    if (out_col < n) {
      float out = scale * sY(r, j);
      if constexpr (SkipZeros) {
        if (out == 0.0f) {
          continue;
        }
      }
      atomicAdd(Y + (g * Br + r) * n + out_col, out);
    }
  }
}

template<int Br, int Tn, int Tk, int KAPPA, int S, bool SkipZeros>
void launch_block_perm(
    const float* A,
    float* Y,
    int d_total,
    int n,
    int k_total,
    int M,
    int Bc,
    float scale,
    uint32_t seed,
    int affine_a,
    int affine_b,
    cudaStream_t stream) {
  dim3 blocks(M, (n + Tn - 1) / Tn);
  dim3 threads(256);
  size_t shmem = static_cast<size_t>(Tk * Tn + Br * Tn) * sizeof(float);
  block_perm_kernel<Br, Tn, Tk, KAPPA, S, SkipZeros><<<blocks, threads, shmem, stream>>>(
      A, Y, d_total, n, k_total, M, Bc, scale, seed, affine_a, affine_b);
}

template<int Br, int KAPPA>
void launch_block_perm_s(
    const float* A,
    float* Y,
    int d_total,
    int n,
    int k_total,
    int M,
    int Bc,
    float scale,
    uint32_t seed,
    int affine_a,
    int affine_b,
    int s,
    bool skip_zeros,
    cudaStream_t stream) {
  if (s == 1) {
    if (skip_zeros) {
      launch_block_perm<Br, 32, 128, KAPPA, 1, true>(
          A, Y, d_total, n, k_total, M, Bc, scale, seed, affine_a, affine_b, stream);
    } else {
      launch_block_perm<Br, 32, 128, KAPPA, 1, false>(
          A, Y, d_total, n, k_total, M, Bc, scale, seed, affine_a, affine_b, stream);
    }
    return;
  }
  if (s == 2) {
    if (skip_zeros) {
      launch_block_perm<Br, 32, 128, KAPPA, 2, true>(
          A, Y, d_total, n, k_total, M, Bc, scale, seed, affine_a, affine_b, stream);
    } else {
      launch_block_perm<Br, 32, 128, KAPPA, 2, false>(
          A, Y, d_total, n, k_total, M, Bc, scale, seed, affine_a, affine_b, stream);
    }
    return;
  }
  if (s == 4) {
    if (skip_zeros) {
      launch_block_perm<Br, 32, 128, KAPPA, 4, true>(
          A, Y, d_total, n, k_total, M, Bc, scale, seed, affine_a, affine_b, stream);
    } else {
      launch_block_perm<Br, 32, 128, KAPPA, 4, false>(
          A, Y, d_total, n, k_total, M, Bc, scale, seed, affine_a, affine_b, stream);
    }
    return;
  }
  if (skip_zeros) {
    launch_block_perm<Br, 32, 128, KAPPA, 8, true>(
        A, Y, d_total, n, k_total, M, Bc, scale, seed, affine_a, affine_b, stream);
  } else {
    launch_block_perm<Br, 32, 128, KAPPA, 8, false>(
        A, Y, d_total, n, k_total, M, Bc, scale, seed, affine_a, affine_b, stream);
  }
}

template<int Br, int Tn, int Tk, int KAPPA, int S, bool SkipZeros>
void launch_block_perm_split(
    const float* A,
    float* Y,
    int d_total,
    int n,
    int k_total,
    int M,
    int Bc,
    int bc_tiles,
    int bc_tile_size,
    float scale,
    uint32_t seed,
    int affine_a,
    int affine_b,
    cudaStream_t stream) {
  dim3 blocks(M, (n + Tn - 1) / Tn, bc_tiles);
  dim3 threads(256);
  size_t shmem = static_cast<size_t>(Tk * Tn + Br * Tn) * sizeof(float);
  block_perm_split_kernel<Br, Tn, Tk, KAPPA, S, SkipZeros><<<blocks, threads, shmem, stream>>>(
      A, Y, d_total, n, k_total, M, Bc, bc_tile_size, scale, seed, affine_a, affine_b);
}

template<int Br, int KAPPA>
void launch_block_perm_split_s(
    const float* A,
    float* Y,
    int d_total,
    int n,
    int k_total,
    int M,
    int Bc,
    int bc_tiles,
    int bc_tile_size,
    float scale,
    uint32_t seed,
    int affine_a,
    int affine_b,
    int s,
    bool skip_zeros,
    cudaStream_t stream) {
  if (s == 1) {
    if (skip_zeros) {
      launch_block_perm_split<Br, 32, 128, KAPPA, 1, true>(
          A,
          Y,
          d_total,
          n,
          k_total,
          M,
          Bc,
          bc_tiles,
          bc_tile_size,
          scale,
          seed,
          affine_a,
          affine_b,
          stream);
    } else {
      launch_block_perm_split<Br, 32, 128, KAPPA, 1, false>(
          A,
          Y,
          d_total,
          n,
          k_total,
          M,
          Bc,
          bc_tiles,
          bc_tile_size,
          scale,
          seed,
          affine_a,
          affine_b,
          stream);
    }
    return;
  }
  if (s == 2) {
    if (skip_zeros) {
      launch_block_perm_split<Br, 32, 128, KAPPA, 2, true>(
          A,
          Y,
          d_total,
          n,
          k_total,
          M,
          Bc,
          bc_tiles,
          bc_tile_size,
          scale,
          seed,
          affine_a,
          affine_b,
          stream);
    } else {
      launch_block_perm_split<Br, 32, 128, KAPPA, 2, false>(
          A,
          Y,
          d_total,
          n,
          k_total,
          M,
          Bc,
          bc_tiles,
          bc_tile_size,
          scale,
          seed,
          affine_a,
          affine_b,
          stream);
    }
    return;
  }
  if (s == 4) {
    if (skip_zeros) {
      launch_block_perm_split<Br, 32, 128, KAPPA, 4, true>(
          A,
          Y,
          d_total,
          n,
          k_total,
          M,
          Bc,
          bc_tiles,
          bc_tile_size,
          scale,
          seed,
          affine_a,
          affine_b,
          stream);
    } else {
      launch_block_perm_split<Br, 32, 128, KAPPA, 4, false>(
          A,
          Y,
          d_total,
          n,
          k_total,
          M,
          Bc,
          bc_tiles,
          bc_tile_size,
          scale,
          seed,
          affine_a,
          affine_b,
          stream);
    }
    return;
  }
  if (skip_zeros) {
    launch_block_perm_split<Br, 32, 128, KAPPA, 8, true>(
        A,
        Y,
        d_total,
        n,
        k_total,
        M,
        Bc,
        bc_tiles,
        bc_tile_size,
        scale,
        seed,
        affine_a,
        affine_b,
        stream);
  } else {
    launch_block_perm_split<Br, 32, 128, KAPPA, 8, false>(
        A,
        Y,
        d_total,
        n,
        k_total,
        M,
        Bc,
        bc_tiles,
        bc_tile_size,
        scale,
        seed,
        affine_a,
        affine_b,
        stream);
  }
}

}  // namespace

torch::Tensor flashsketch_forward_cuda(
    torch::Tensor x,
    int64_t k_total,
    int64_t block_rows,
    int64_t kappa,
    int64_t s,
    double scale,
    uint64_t seed,
    int64_t affine_a,
    int64_t affine_b,
    bool skip_zeros) {
  const auto d_total = static_cast<int>(x.size(0));
  const auto n = static_cast<int>(x.size(1));
  const auto k_total_i = static_cast<int>(k_total);
  const auto block_rows_i = static_cast<int>(block_rows);
  const auto kappa_i = static_cast<int>(kappa);
  const auto s_i = static_cast<int>(s);
  const float scale_f = static_cast<float>(scale);
  const uint32_t seed32 = static_cast<uint32_t>(seed);
  const int affine_a_i = static_cast<int>(affine_a);
  const int affine_b_i = static_cast<int>(affine_b);

  TORCH_CHECK(k_total_i > 0, "k_total must be positive");
  TORCH_CHECK(block_rows_i == 64 || block_rows_i == 128, "block_rows must be 64 or 128");
  TORCH_CHECK(
      kappa_i == 1 || kappa_i == 2 || kappa_i == 4,
      "flashsketch requires kappa=1, 2, or 4");
  TORCH_CHECK(
      s_i == 1 || s_i == 2 || s_i == 4 || s_i == 8,
      "flashsketch requires s in {1,2,4,8}");
  TORCH_CHECK(affine_a_i > 0, "affine_a must be positive");
  TORCH_CHECK(affine_b_i >= 0, "affine_b must be non-negative");
  TORCH_CHECK(k_total_i % block_rows_i == 0, "k_total must be divisible by block_rows");

  const int M = k_total_i / block_rows_i;
  TORCH_CHECK(M > 0, "block_rows must be <= k_total");
  TORCH_CHECK(d_total % M == 0, "d_total must be divisible by M");
  const int Bc = d_total / M;

  auto y = torch::zeros({k_total_i, n}, x.options());
  cudaStream_t stream = at::cuda::getDefaultCUDAStream();

  constexpr int kTn = 32;
  constexpr int kMaxBcTiles = 16;
  int device = 0;
  TORCH_CHECK(cudaGetDevice(&device) == cudaSuccess, "cudaGetDevice failed");
  int sm_count = 0;
  TORCH_CHECK(
      cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device) == cudaSuccess,
      "cudaDeviceGetAttribute failed");
  const int64_t n_tiles = (n + kTn - 1) / kTn;
  const int64_t blocks = static_cast<int64_t>(M) * n_tiles;
  const int64_t min_blocks = static_cast<int64_t>(8 * sm_count);
  int bc_tiles = 1;
  if (blocks > 0 && blocks < min_blocks) {
    bc_tiles = static_cast<int>((min_blocks + blocks - 1) / blocks);
    if (bc_tiles > kMaxBcTiles) {
      bc_tiles = kMaxBcTiles;
    }
  }
  const int bc_tile_size = (Bc + bc_tiles - 1) / bc_tiles;

  if (block_rows_i == 128) {
    if (kappa_i == 1) {
      if (bc_tiles > 1) {
        launch_block_perm_split_s<128, 1>(
            x.data_ptr<float>(),
            y.data_ptr<float>(),
            d_total,
            n,
            k_total_i,
            M,
            Bc,
            bc_tiles,
            bc_tile_size,
            scale_f,
            seed32,
            affine_a_i,
            affine_b_i,
            s_i,
            skip_zeros,
            stream);
      } else {
        launch_block_perm_s<128, 1>(
            x.data_ptr<float>(),
            y.data_ptr<float>(),
            d_total,
            n,
            k_total_i,
            M,
            Bc,
            scale_f,
            seed32,
            affine_a_i,
            affine_b_i,
            s_i,
            skip_zeros,
            stream);
      }
      return y;
    }
    if (kappa_i == 2) {
      if (bc_tiles > 1) {
        launch_block_perm_split_s<128, 2>(
            x.data_ptr<float>(),
            y.data_ptr<float>(),
            d_total,
            n,
            k_total_i,
            M,
            Bc,
            bc_tiles,
            bc_tile_size,
            scale_f,
            seed32,
            affine_a_i,
            affine_b_i,
            s_i,
            skip_zeros,
            stream);
      } else {
        launch_block_perm_s<128, 2>(
            x.data_ptr<float>(),
            y.data_ptr<float>(),
            d_total,
            n,
            k_total_i,
            M,
            Bc,
            scale_f,
            seed32,
            affine_a_i,
            affine_b_i,
            s_i,
            skip_zeros,
            stream);
      }
      return y;
    }
    if (bc_tiles > 1) {
      launch_block_perm_split_s<128, 4>(
          x.data_ptr<float>(),
          y.data_ptr<float>(),
          d_total,
          n,
          k_total_i,
          M,
          Bc,
          bc_tiles,
          bc_tile_size,
          scale_f,
          seed32,
          affine_a_i,
          affine_b_i,
          s_i,
          skip_zeros,
          stream);
    } else {
      launch_block_perm_s<128, 4>(
          x.data_ptr<float>(),
          y.data_ptr<float>(),
          d_total,
          n,
          k_total_i,
          M,
          Bc,
          scale_f,
          seed32,
          affine_a_i,
          affine_b_i,
          s_i,
          skip_zeros,
          stream);
    }
    return y;
  }

  if (kappa_i == 1) {
    if (bc_tiles > 1) {
      launch_block_perm_split_s<64, 1>(
          x.data_ptr<float>(),
          y.data_ptr<float>(),
          d_total,
          n,
          k_total_i,
          M,
          Bc,
          bc_tiles,
          bc_tile_size,
          scale_f,
          seed32,
          affine_a_i,
          affine_b_i,
          s_i,
          skip_zeros,
          stream);
    } else {
      launch_block_perm_s<64, 1>(
          x.data_ptr<float>(),
          y.data_ptr<float>(),
          d_total,
          n,
          k_total_i,
          M,
          Bc,
          scale_f,
          seed32,
          affine_a_i,
          affine_b_i,
          s_i,
          skip_zeros,
          stream);
    }
    return y;
  }
  if (kappa_i == 2) {
    if (bc_tiles > 1) {
      launch_block_perm_split_s<64, 2>(
          x.data_ptr<float>(),
          y.data_ptr<float>(),
          d_total,
          n,
          k_total_i,
          M,
          Bc,
          bc_tiles,
          bc_tile_size,
          scale_f,
          seed32,
          affine_a_i,
          affine_b_i,
          s_i,
          skip_zeros,
          stream);
    } else {
      launch_block_perm_s<64, 2>(
          x.data_ptr<float>(),
          y.data_ptr<float>(),
          d_total,
          n,
          k_total_i,
          M,
          Bc,
          scale_f,
          seed32,
          affine_a_i,
          affine_b_i,
          s_i,
          skip_zeros,
          stream);
    }
    return y;
  }
  if (bc_tiles > 1) {
    launch_block_perm_split_s<64, 4>(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        d_total,
        n,
        k_total_i,
        M,
        Bc,
        bc_tiles,
        bc_tile_size,
        scale_f,
        seed32,
        affine_a_i,
        affine_b_i,
        s_i,
        skip_zeros,
        stream);
  } else {
    launch_block_perm_s<64, 4>(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        d_total,
        n,
        k_total_i,
        M,
        Bc,
        scale_f,
        seed32,
        affine_a_i,
        affine_b_i,
        s_i,
        skip_zeros,
        stream);
  }
  return y;
}
