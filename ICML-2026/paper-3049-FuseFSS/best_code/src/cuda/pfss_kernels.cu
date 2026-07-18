#include "suf/gpu_kernels.hpp"

#ifdef SUF_HAVE_CUDA

#include <cuda_runtime.h>

namespace suf {

namespace {
__device__ __forceinline__ u8 eval_pred_device(const GpuPredicate& p, u64 x) {
  switch (p.kind) {
    case static_cast<u8>(PredKind::LT):
      return static_cast<u8>((x + p.input_add) < p.param);
    case static_cast<u8>(PredKind::LTLOW): {
      const u64 xshift = x + p.input_add;
      if (p.f == 64) return static_cast<u8>(xshift < p.gamma);
      const u64 mask = (p.f == 64) ? ~0ULL : ((1ULL << p.f) - 1ULL);
      const u64 xlow = xshift & mask;
      return static_cast<u8>(xlow < p.gamma);
    }
    case static_cast<u8>(PredKind::MSB):
      return static_cast<u8>((x >> 63) & 1ULL);
    case static_cast<u8>(PredKind::MSB_ADD):
      return static_cast<u8>(((x + p.param) >> 63) & 1ULL);
    case static_cast<u8>(PredKind::CONST):
      return static_cast<u8>(p.param & 1ULL);
    default:
      return 0;
  }
}

__device__ __forceinline__ int interval_index_device(const u64* cuts, int m, u64 x) {
  int lo = 0;
  int hi = m; // exclusive
  while (lo + 1 < hi) {
    const int mid = (lo + hi) >> 1;
    if (x < cuts[mid]) {
      hi = mid;
    } else {
      lo = mid;
    }
  }
  return lo;
}

__global__ void kernel_eval_preds(const u64* x, std::size_t n,
                                  const GpuPredicate* preds, int num_preds,
                                  u8* out) {
  const std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  const u64 xi = x[idx];
  for (int p = 0; p < num_preds; ++p) {
    out[static_cast<std::size_t>(p) * n + idx] = eval_pred_device(preds[p], xi);
  }
}

__global__ void kernel_eval_poly(const u64* x, std::size_t n,
                                 const u64* cuts, int num_cuts,
                                 const u64* coeffs, int degree,
                                 u64* out) {
  const std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  const u64 xi = x[idx];
  const int interval = interval_index_device(cuts, num_cuts, xi);
  const int stride = degree + 1;
  const u64* coeff = coeffs + static_cast<std::size_t>(interval) * stride;
  u64 y = 0;
  for (int k = degree; k >= 0; --k) {
    y = y * xi + coeff[k];
  }
  out[idx] = y;
}

__global__ void kernel_eval_poly_from_coeffs(const u64* x,
                                             const u64* coeffs,
                                             std::size_t n,
                                             int degree,
                                             u64* out) {
  const std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  const u64 xi = x[idx];
  const std::size_t stride = static_cast<std::size_t>(degree + 1);
  const u64* coeff = coeffs + idx * stride;
  u64 y = 0;
  for (int k = degree; k >= 0; --k) {
    y = y * xi + coeff[k];
  }
  out[idx] = y;
}

__global__ void kernel_eval_poly_from_coeffs_multi(const u64* x,
                                                   const u64* coeffs,
                                                   std::size_t n,
                                                   int degree,
                                                   int outputs,
                                                   std::size_t coeff_words_per_input,
                                                   u64* out) {
  const std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  const u64 xi = x[idx];
  const std::size_t stride = static_cast<std::size_t>(degree + 1);
  const u64* row = coeffs + idx * coeff_words_per_input;
  for (int out_idx = 0; out_idx < outputs; ++out_idx) {
    const u64* coeff = row + static_cast<std::size_t>(out_idx) * stride;
    u64 y = 0;
    for (int k = degree; k >= 0; --k) {
      y = y * xi + coeff[k];
    }
    out[static_cast<std::size_t>(out_idx) * n + idx] = y;
  }
}

__global__ void kernel_eval_helper(const u8* pred_bits, std::size_t n,
                                   const GpuBoolNode* nodes, int num_nodes, int root,
                                   u64* out) {
  const std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) return;

  extern __shared__ u8 scratch[];
  u8* local = scratch + threadIdx.x * num_nodes;

  for (int i = 0; i < num_nodes; ++i) {
    const GpuBoolNode node = nodes[i];
    switch (node.kind) {
      case static_cast<u8>(BoolNode::Kind::PRED):
        local[i] = pred_bits[static_cast<std::size_t>(node.pred_index) * n + idx] & 1;
        break;
      case static_cast<u8>(BoolNode::Kind::NOT):
        local[i] = 1u ^ (local[node.lhs] & 1u);
        break;
      case static_cast<u8>(BoolNode::Kind::AND):
        local[i] = (local[node.lhs] & local[node.rhs]) & 1u;
        break;
      case static_cast<u8>(BoolNode::Kind::OR):
        local[i] = (local[node.lhs] | local[node.rhs]) & 1u;
        break;
      case static_cast<u8>(BoolNode::Kind::XOR):
        local[i] = (local[node.lhs] ^ local[node.rhs]) & 1u;
        break;
      default:
        local[i] = 0;
        break;
    }
  }
  out[idx] = static_cast<u64>(local[root] & 1u);
}

__global__ void kernel_fill_const_pred_bits(const u8* const_bits, std::size_t n,
                                            int num_preds, u8* out) {
  const std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  const std::size_t total = static_cast<std::size_t>(num_preds) * n;
  if (idx >= total) return;
  const int p = static_cast<int>(idx / n);
  out[idx] = const_bits ? (const_bits[p] & 1u) : 0u;
}

__global__ void kernel_scatter_pred_bits(const u8* query_bits, std::size_t n,
                                         int num_queries, const int* query_to_pred,
                                         u8* out) {
  const std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  const std::size_t total = static_cast<std::size_t>(num_queries) * n;
  if (idx >= total) return;
  const int q = static_cast<int>(idx / n);
  const std::size_t off = idx % n;
  const int pred = query_to_pred[q];
  out[static_cast<std::size_t>(pred) * n + off] = query_bits[idx];
}

} // namespace

void launch_eval_preds(const u64* d_in, std::size_t n,
                       const GpuPredicate* d_preds, int num_preds,
                       u8* d_out, cudaStream_t stream) {
  const int threads = 256;
  const int blocks = static_cast<int>((n + threads - 1) / threads);
  kernel_eval_preds<<<blocks, threads, 0, stream>>>(d_in, n, d_preds, num_preds, d_out);
}

void launch_eval_poly(const u64* d_in, std::size_t n,
                      const u64* d_cuts, int num_cuts,
                      const u64* d_coeffs, int degree,
                      u64* d_out, cudaStream_t stream) {
  const int threads = 256;
  const int blocks = static_cast<int>((n + threads - 1) / threads);
  kernel_eval_poly<<<blocks, threads, 0, stream>>>(d_in, n, d_cuts, num_cuts, d_coeffs, degree, d_out);
}

void launch_eval_poly_from_coeffs(const u64* d_in,
                                  const u64* d_coeffs,
                                  std::size_t n,
                                  int degree,
                                  u64* d_out,
                                  cudaStream_t stream) {
  const int threads = 256;
  const int blocks = static_cast<int>((n + threads - 1) / threads);
  kernel_eval_poly_from_coeffs<<<blocks, threads, 0, stream>>>(d_in, d_coeffs, n, degree, d_out);
}

void launch_eval_poly_from_coeffs_multi(const u64* d_in,
                                        const u64* d_coeffs,
                                        std::size_t n,
                                        int degree,
                                        int outputs,
                                        std::size_t coeff_words_per_input,
                                        u64* d_out,
                                        cudaStream_t stream) {
  if (!d_in || !d_coeffs || !d_out || n == 0 || outputs <= 0) return;
  const std::size_t min_words =
      static_cast<std::size_t>(outputs) * static_cast<std::size_t>(degree + 1);
  if (coeff_words_per_input < min_words) return;
  const int threads = 256;
  const int blocks = static_cast<int>((n + threads - 1) / threads);
  kernel_eval_poly_from_coeffs_multi<<<blocks, threads, 0, stream>>>(d_in, d_coeffs, n, degree,
                                                                     outputs, coeff_words_per_input,
                                                                     d_out);
}

void launch_eval_helper(const u8* d_pred_bits, std::size_t n,
                        const GpuBoolNode* d_nodes, int num_nodes, int root,
                        u64* d_out, cudaStream_t stream) {
  const int threads = 256;
  const int blocks = static_cast<int>((n + threads - 1) / threads);
  const std::size_t shared_bytes = static_cast<std::size_t>(num_nodes) * threads * sizeof(u8);
  kernel_eval_helper<<<blocks, threads, shared_bytes, stream>>>(d_pred_bits, n, d_nodes, num_nodes, root, d_out);
}

void launch_fill_const_pred_bits(const u8* d_const_bits, std::size_t n,
                                 int num_preds, u8* d_out,
                                 cudaStream_t stream) {
  if (!d_out || num_preds == 0 || n == 0) return;
  const std::size_t total = static_cast<std::size_t>(num_preds) * n;
  const int threads = 256;
  const int blocks = static_cast<int>((total + threads - 1) / threads);
  kernel_fill_const_pred_bits<<<blocks, threads, 0, stream>>>(d_const_bits, n, num_preds, d_out);
}

void launch_scatter_pred_bits(const u8* d_query_bits, std::size_t n,
                              int num_queries, const int* d_query_to_pred,
                              u8* d_out, cudaStream_t stream) {
  if (!d_query_bits || !d_query_to_pred || !d_out || num_queries == 0 || n == 0) return;
  const std::size_t total = static_cast<std::size_t>(num_queries) * n;
  const int threads = 256;
  const int blocks = static_cast<int>((total + threads - 1) / threads);
  kernel_scatter_pred_bits<<<blocks, threads, 0, stream>>>(d_query_bits, n, num_queries, d_query_to_pred, d_out);
}

} // namespace suf

#endif // SUF_HAVE_CUDA
