#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

static inline int ceil_div_int(int a, int b) {
  return (a + b - 1) / b;
}

using vec16_t = uint4; // 16 bytes

#ifndef ROW_TILE
#define ROW_TILE 16
#endif

template <typename scalar_t>
__global__ void get_seqs_by_seq_tiled_vec16_kernel(const scalar_t* __restrict__ cache,     // [T, D]
                                                   scalar_t* __restrict__ out,             // [total_rows, D]
                                                   const int64_t* __restrict__ seq_starts, // [MaxSeqs]
                                                   const int32_t* __restrict__ cu,         // [B+1]
                                                   const int64_t* __restrict__ seq_ids,    // [B]
                                                   int32_t D, int32_t B) {
  int32_t b = (int32_t)blockIdx.y;
  if (b >= B)
    return;

  int32_t out0 = cu[b];
  int32_t out1 = cu[b + 1];
  int32_t L = out1 - out0;
  if (L <= 0)
    return;

  int32_t tile_id = (int32_t)blockIdx.z;
  int32_t row_base = tile_id * ROW_TILE;
  if (row_base >= L)
    return;

  int64_t gseq = seq_ids[b];
  int64_t src0 = seq_starts[gseq];

  int32_t vec_cols = D / 8;

  int32_t v = (int32_t)blockIdx.x * blockDim.x + threadIdx.x;
  int32_t v_stride = blockDim.x * gridDim.x;

  for (; v < vec_cols; v += v_stride) {
#pragma unroll
    for (int r = 0; r < ROW_TILE; ++r) {
      int32_t i = row_base + r;
      if (i >= L)
        break;

      int64_t src_row = src0 + (int64_t)i;
      int32_t out_row = out0 + i;

      const vec16_t* __restrict__ src = reinterpret_cast<const vec16_t*>(cache + src_row * (int64_t)D);
      vec16_t* __restrict__ dst = reinterpret_cast<vec16_t*>(out + (int64_t)out_row * (int64_t)D);

      dst[v] = src[v];
    }
  }
}

at::Tensor get_seqs_cuda(const at::Tensor& cache,      // [T, D]
                         const at::Tensor& seq_starts, // [MaxSeqs] int64
                         const at::Tensor& cu_seqlens, // [B+1] int32
                         const at::Tensor& seq_ids,    // [B] int64
                         const int64_t total_seqlen) {
  TORCH_CHECK(cache.is_cuda() && cache.is_contiguous(), "cache must be CUDA contiguous");
  TORCH_CHECK(seq_starts.is_cuda() && seq_starts.is_contiguous(), "seq_starts must be CUDA contiguous");
  TORCH_CHECK(cu_seqlens.is_cuda() && cu_seqlens.is_contiguous(), "cu_seqlens must be CUDA contiguous");
  TORCH_CHECK(seq_ids.is_cuda() && seq_ids.is_contiguous(), "seq_ids must be CUDA contiguous");
  TORCH_CHECK(cu_seqlens.scalar_type() == at::kInt, "cu_seqlens must be int32");
  TORCH_CHECK(cache.dim() == 2, "cache must be [T, D]");

  int32_t B = (int32_t)seq_ids.size(0);
  int32_t D = (int32_t)cache.size(1);

  TORCH_CHECK((D % 8) == 0, "D must be multiple of 8 for vec16 path (D=4096 ok)");

  int64_t total_rows64 = total_seqlen;
  auto out = at::empty({total_rows64, (int64_t)D}, cache.options());
  if (total_rows64 == 0 || B == 0)
    return out;

  constexpr int THREADS = 256;
  int32_t vec_cols = D / 8;
  int32_t xBlocks = ceil_div_int(vec_cols, THREADS);
  if (xBlocks > 4)
    xBlocks = 4; // keep reasonable

  int32_t zBlocks = ceil_div_int((int32_t)total_seqlen, (int32_t)ROW_TILE);

  dim3 block(THREADS);
  dim3 grid(xBlocks, B, zBlocks);

  auto stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, cache.scalar_type(), "get_seqs_by_seq_tiled_vec16", [&] {
    get_seqs_by_seq_tiled_vec16_kernel<scalar_t><<<grid, block, 0, stream>>>(
        cache.data_ptr<scalar_t>(), out.data_ptr<scalar_t>(), seq_starts.data_ptr<int64_t>(),
        cu_seqlens.data_ptr<int32_t>(), seq_ids.data_ptr<int64_t>(), D, B);
  });

  return out;
}

template <typename scalar_t>
__global__ void reset_full_by_seq_tiled_vec16_kernel(scalar_t* __restrict__ cache,           // [T, D]
                                                     const scalar_t* __restrict__ in,        // [total_rows, D]
                                                     const int64_t* __restrict__ seq_starts, // [MaxSeqs]
                                                     const int32_t* __restrict__ cu,         // [B+1]
                                                     const int64_t* __restrict__ seq_ids,    // [B]
                                                     int32_t D, int32_t B) {
  int32_t b = (int32_t)blockIdx.y;
  if (b >= B)
    return;

  int32_t in0 = cu[b];
  int32_t in1 = cu[b + 1];
  int32_t L = in1 - in0;
  if (L <= 0)
    return;

  int32_t tile_id = (int32_t)blockIdx.z;
  int32_t row_base = tile_id * ROW_TILE;
  if (row_base >= L)
    return;

  int64_t gseq = seq_ids[b];
  int64_t dst0 = seq_starts[gseq];

  int32_t vec_cols = D / 8;

  int32_t v = (int32_t)blockIdx.x * blockDim.x + threadIdx.x;
  int32_t v_stride = blockDim.x * gridDim.x;

  for (; v < vec_cols; v += v_stride) {
#pragma unroll
    for (int r = 0; r < ROW_TILE; ++r) {
      int32_t i = row_base + r;
      if (i >= L)
        break;

      int32_t in_row = in0 + i;
      int64_t dst_row = dst0 + (int64_t)i;

      const vec16_t* __restrict__ src = reinterpret_cast<const vec16_t*>(in + (int64_t)in_row * (int64_t)D);
      vec16_t* __restrict__ dst = reinterpret_cast<vec16_t*>(cache + dst_row * (int64_t)D);

      dst[v] = src[v];
    }
  }
}

void reset_full_cuda(at::Tensor& cache,            // [T, D]
                     const at::Tensor& in,         // [total_rows, D]
                     const at::Tensor& seq_starts, // [MaxSeqs] int64
                     const at::Tensor& cu_seqlens, // [B+1] int32
                     const at::Tensor& seq_ids,    // [B] int64
                     const int64_t total_seqlen) {
  TORCH_CHECK(cache.is_cuda() && cache.is_contiguous(), "cache must be CUDA contiguous");
  TORCH_CHECK(in.is_cuda() && in.is_contiguous(), "in must be CUDA contiguous");
  TORCH_CHECK(seq_starts.is_cuda() && seq_starts.is_contiguous(), "seq_starts must be CUDA contiguous");
  TORCH_CHECK(cu_seqlens.is_cuda() && cu_seqlens.is_contiguous(), "cu_seqlens must be CUDA contiguous");
  TORCH_CHECK(seq_ids.is_cuda() && seq_ids.is_contiguous(), "seq_ids must be CUDA contiguous");
  TORCH_CHECK(cu_seqlens.scalar_type() == at::kInt, "cu_seqlens must be int32");

  int32_t B = (int32_t)seq_ids.size(0);
  int32_t D = (int32_t)cache.size(1);

  TORCH_CHECK(in.size(1) == cache.size(1), "in and cache must have same D");
  TORCH_CHECK((D % 8) == 0, "D must be multiple of 8 for vec16 path");

  if (total_seqlen == 0 || B == 0)
    return;

  constexpr int THREADS = 256;
  int32_t vec_cols = D / 8;
  int32_t xBlocks = ceil_div_int(vec_cols, THREADS);
  if (xBlocks > 4)
    xBlocks = 4;

  int32_t zBlocks = ceil_div_int((int32_t)total_seqlen, (int32_t)ROW_TILE);

  dim3 block(THREADS);
  dim3 grid(xBlocks, B, zBlocks);

  auto stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, cache.scalar_type(), "reset_full_by_seq_tiled_vec16", [&] {
    reset_full_by_seq_tiled_vec16_kernel<scalar_t><<<grid, block, 0, stream>>>(
        cache.data_ptr<scalar_t>(), in.data_ptr<scalar_t>(), seq_starts.data_ptr<int64_t>(),
        cu_seqlens.data_ptr<int32_t>(), seq_ids.data_ptr<int64_t>(), D, B);
  });
}

template <typename scalar_t>
__global__ void get_block_by_seq_tiled_vec16_kernel(const scalar_t* __restrict__ cache,       // [T, D]
                                                    scalar_t* __restrict__ out,               // [total_rows, D]
                                                    const int64_t* __restrict__ seq_starts,   // [MaxSeqs]
                                                    const int32_t* __restrict__ cu,           // [B+1]
                                                    const int32_t* __restrict__ cu_seqlens_q, // [B+1]
                                                    const int64_t* __restrict__ seq_ids,      // [B]
                                                    int32_t D, int32_t B) {
  int32_t b = (int32_t)blockIdx.y;
  if (b >= B)
    return;

  int32_t out0 = cu_seqlens_q[b];
  int32_t out1 = cu_seqlens_q[b + 1];
  int32_t L = out1 - out0;
  if (L <= 0)
    return;

  int32_t tile_id = (int32_t)blockIdx.z;
  int32_t row_base = tile_id * ROW_TILE;
  if (row_base >= L)
    return;

  int64_t gseq = seq_ids[b];
  int64_t src0 = seq_starts[gseq] + cu[b + 1] - cu[b] - L;

  int32_t vec_cols = D / 8;

  int32_t v = (int32_t)blockIdx.x * blockDim.x + threadIdx.x;
  int32_t v_stride = blockDim.x * gridDim.x;

  for (; v < vec_cols; v += v_stride) {
#pragma unroll
    for (int r = 0; r < ROW_TILE; ++r) {
      int32_t i = row_base + r;
      if (i >= L)
        break;

      int64_t src_row = src0 + (int64_t)i;
      int32_t out_row = out0 + i;

      const vec16_t* __restrict__ src = reinterpret_cast<const vec16_t*>(cache + src_row * (int64_t)D);
      vec16_t* __restrict__ dst = reinterpret_cast<vec16_t*>(out + (int64_t)out_row * (int64_t)D);

      dst[v] = src[v];
    }
  }
}

at::Tensor get_block_cuda(const at::Tensor& cache,        // [T, D]
                          const at::Tensor& seq_starts,   // [MaxSeqs] int64
                          const at::Tensor& cu_seqlens,   // [B+1] int32
                          const at::Tensor& cu_seqlens_q, // [B+1] int32
                          const at::Tensor& seq_ids,      // [B] int64
                          const int64_t total_seqlen) {
  TORCH_CHECK(cache.is_cuda() && cache.is_contiguous(), "cache must be CUDA contiguous");
  TORCH_CHECK(seq_starts.is_cuda() && seq_starts.is_contiguous(), "seq_starts must be CUDA contiguous");
  TORCH_CHECK(cu_seqlens.is_cuda() && cu_seqlens.is_contiguous(), "cu_seqlens must be CUDA contiguous");
  TORCH_CHECK(cu_seqlens_q.is_cuda() && cu_seqlens_q.is_contiguous(), "cu_seqlens_q must be CUDA contiguous");
  TORCH_CHECK(seq_ids.is_cuda() && seq_ids.is_contiguous(), "seq_ids must be CUDA contiguous");
  TORCH_CHECK(cu_seqlens.scalar_type() == at::kInt, "cu_seqlens must be int32");
  TORCH_CHECK(cu_seqlens_q.scalar_type() == at::kInt, "cu_seqlens_q must be int32");
  TORCH_CHECK(cache.dim() == 2, "cache must be [T, D]");

  int32_t B = (int32_t)seq_ids.size(0);
  int32_t D = (int32_t)cache.size(1);

  TORCH_CHECK((D % 8) == 0, "D must be multiple of 8 for vec16 path (D=4096 ok)");

  int64_t total_rows64 = total_seqlen;
  auto out = at::empty({total_rows64, (int64_t)D}, cache.options());
  if (total_rows64 == 0 || B == 0)
    return out;

  constexpr int THREADS = 256;
  int32_t vec_cols = D / 8;
  int32_t xBlocks = ceil_div_int(vec_cols, THREADS);
  if (xBlocks > 4)
    xBlocks = 4; // keep reasonable

  int32_t zBlocks = ceil_div_int((int32_t)total_seqlen, (int32_t)ROW_TILE);

  dim3 block(THREADS);
  dim3 grid(xBlocks, B, zBlocks);

  auto stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, cache.scalar_type(), "get_block_by_seq_tiled_vec16", [&] {
    get_block_by_seq_tiled_vec16_kernel<scalar_t><<<grid, block, 0, stream>>>(
        cache.data_ptr<scalar_t>(), out.data_ptr<scalar_t>(), seq_starts.data_ptr<int64_t>(),
        cu_seqlens.data_ptr<int32_t>(), cu_seqlens_q.data_ptr<int32_t>(), seq_ids.data_ptr<int64_t>(), D, B);
  });

  return out;
}

template <typename scalar_t>
__global__ void reset_block_by_seq_tiled_vec16_kernel(scalar_t* __restrict__ cache,             // [T, D]
                                                      const scalar_t* __restrict__ in,          // [total_rows, D]
                                                      const int64_t* __restrict__ seq_starts,   // [MaxSeqs]
                                                      const int32_t* __restrict__ cu,           // [B+1]
                                                      const int32_t* __restrict__ cu_seqlens_q, // [B+1]
                                                      const int64_t* __restrict__ seq_ids,      // [B]
                                                      int32_t D, int32_t B) {
  int32_t b = (int32_t)blockIdx.y;
  if (b >= B)
    return;

  int32_t in0 = cu_seqlens_q[b];
  int32_t in1 = cu_seqlens_q[b + 1];
  int32_t L = in1 - in0;
  if (L <= 0)
    return;

  int32_t tile_id = (int32_t)blockIdx.z;
  int32_t row_base = tile_id * ROW_TILE;
  if (row_base >= L)
    return;

  int64_t gseq = seq_ids[b];
  int64_t dst0 = seq_starts[gseq] + cu[b + 1] - cu[b] - L;

  int32_t vec_cols = D / 8;

  int32_t v = (int32_t)blockIdx.x * blockDim.x + threadIdx.x;
  int32_t v_stride = blockDim.x * gridDim.x;

  for (; v < vec_cols; v += v_stride) {
#pragma unroll
    for (int r = 0; r < ROW_TILE; ++r) {
      int32_t i = row_base + r;
      if (i >= L)
        break;

      int32_t in_row = in0 + i;
      int64_t dst_row = dst0 + (int64_t)i;

      const vec16_t* __restrict__ src = reinterpret_cast<const vec16_t*>(in + (int64_t)in_row * (int64_t)D);
      vec16_t* __restrict__ dst = reinterpret_cast<vec16_t*>(cache + dst_row * (int64_t)D);

      dst[v] = src[v];
    }
  }
}

void reset_block_cuda(at::Tensor& cache,              // [T, D]
                      const at::Tensor& in,           // [total_rows, D]
                      const at::Tensor& seq_starts,   // [MaxSeqs] int64
                      const at::Tensor& cu_seqlens,   // [B+1] int32
                      const at::Tensor& cu_seqlens_q, // [B+1] int32
                      const at::Tensor& seq_ids,      // [B] int64
                      const int64_t total_seqlen) {
  TORCH_CHECK(cache.is_cuda() && cache.is_contiguous(), "cache must be CUDA contiguous");
  TORCH_CHECK(in.is_cuda() && in.is_contiguous(), "in must be CUDA contiguous");
  TORCH_CHECK(seq_starts.is_cuda() && seq_starts.is_contiguous(), "seq_starts must be CUDA contiguous");
  TORCH_CHECK(cu_seqlens.is_cuda() && cu_seqlens.is_contiguous(), "cu_seqlens must be CUDA contiguous");
  TORCH_CHECK(seq_ids.is_cuda() && seq_ids.is_contiguous(), "seq_ids must be CUDA contiguous");
  TORCH_CHECK(cu_seqlens.scalar_type() == at::kInt, "cu_seqlens must be int32");

  int32_t B = (int32_t)seq_ids.size(0);
  int32_t D = (int32_t)cache.size(1);

  TORCH_CHECK(in.size(1) == cache.size(1), "in and cache must have same D");
  TORCH_CHECK((D % 8) == 0, "D must be multiple of 8 for vec16 path");

  if (total_seqlen == 0 || B == 0)
    return;

  constexpr int THREADS = 256;
  int32_t vec_cols = D / 8;
  int32_t xBlocks = ceil_div_int(vec_cols, THREADS);
  if (xBlocks > 4)
    xBlocks = 4;

  int32_t zBlocks = ceil_div_int((int32_t)total_seqlen, (int32_t)ROW_TILE);

  dim3 block(THREADS);
  dim3 grid(xBlocks, B, zBlocks);

  auto stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, cache.scalar_type(), "reset_block_by_seq_tiled_vec16", [&] {
    reset_block_by_seq_tiled_vec16_kernel<scalar_t><<<grid, block, 0, stream>>>(
        cache.data_ptr<scalar_t>(), in.data_ptr<scalar_t>(), seq_starts.data_ptr<int64_t>(),
        cu_seqlens.data_ptr<int32_t>(), cu_seqlens_q.data_ptr<int32_t>(), seq_ids.data_ptr<int64_t>(), D, B);
  });
}

template <typename scalar_t, typename seqlen_t>
__global__ void scatter_update_kernel(scalar_t* __restrict__ cache,            // [Total_Size, D]
                                      const scalar_t* __restrict__ new_vals,   // [M, D]
                                      const int64_t* __restrict__ row_idx,     // [M]
                                      const seqlen_t* __restrict__ cu_seqlens, // [B+1]
                                      const int64_t* __restrict__ seq_starts,  // [MaxSeqs] (full array)
                                      const int64_t* __restrict__ seq_ids,     // [B]
                                      int64_t M, int64_t D, int64_t B) {
  int64_t row = blockIdx.x;
  if (row >= M)
    return;

  int64_t r_idx = row_idx[row];

  // Binary search to find seq_id (index in batch, 0..B-1)
  int64_t left = 0;
  int64_t right = B - 1;
  int64_t seq_id = 0;

  while (left <= right) {
    int64_t mid = (left + right) >> 1;
    if (static_cast<int64_t>(cu_seqlens[mid + 1]) <= r_idx) {
      left = mid + 1;
    } else if (static_cast<int64_t>(cu_seqlens[mid]) > r_idx) {
      right = mid - 1;
    } else {
      seq_id = mid;
      break;
    }
  }

  int64_t offset = r_idx - static_cast<int64_t>(cu_seqlens[seq_id]);
  int64_t global_id = seq_ids[seq_id];
  int64_t dest_row = seq_starts[global_id] + offset;

  for (int64_t col = threadIdx.x; col < D; col += blockDim.x) {
    cache[dest_row * D + col] = new_vals[row * D + col];
  }
}

void scatter_update_cuda(at::Tensor& cache, const at::Tensor& new_vals, const at::Tensor& row_idx,
                         const at::Tensor& cu_seqlens, const at::Tensor& seq_starts, const at::Tensor& seq_ids) {
  TORCH_CHECK(cache.is_cuda(), "cache must be CUDA");
  TORCH_CHECK(new_vals.is_cuda(), "new_vals must be CUDA");
  TORCH_CHECK(row_idx.is_cuda(), "row_idx must be CUDA");
  TORCH_CHECK(cu_seqlens.is_cuda(), "cu_seqlens must be CUDA");
  TORCH_CHECK(seq_starts.is_cuda(), "seq_starts must be CUDA");
  TORCH_CHECK(seq_ids.is_cuda(), "seq_ids must be CUDA");

  int64_t M = new_vals.size(0);
  int64_t D = new_vals.size(1);
  int64_t B = seq_ids.size(0);

  if (M == 0)
    return;

  const int threads = 256;
  const int blocks = M;

  auto stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, cache.scalar_type(), "scatter_update_kernel", [&] {
    if (cu_seqlens.scalar_type() == at::kInt) {
      scatter_update_kernel<scalar_t, int32_t><<<blocks, threads, 0, stream>>>(
          cache.data_ptr<scalar_t>(), new_vals.data_ptr<scalar_t>(), row_idx.data_ptr<int64_t>(),
          cu_seqlens.data_ptr<int32_t>(), seq_starts.data_ptr<int64_t>(), seq_ids.data_ptr<int64_t>(), M, D, B);
    } else {
      scatter_update_kernel<scalar_t, int64_t><<<blocks, threads, 0, stream>>>(
          cache.data_ptr<scalar_t>(), new_vals.data_ptr<scalar_t>(), row_idx.data_ptr<int64_t>(),
          cu_seqlens.data_ptr<int64_t>(), seq_starts.data_ptr<int64_t>(), seq_ids.data_ptr<int64_t>(), M, D, B);
    }
  });
}