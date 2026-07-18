#include "suf/interval_lut.hpp"

#ifdef SUF_HAVE_CUDA

#include <cuda_runtime.h>

namespace suf {

__global__ void kernel_interval_lut_add_base(const u64* base, u64* out,
                                             std::size_t n, int out_words) {
  const std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  for (int w = 0; w < out_words; ++w) {
    out[idx * out_words + w] += base[w];
  }
}

__global__ void kernel_interval_lut_direct(const u64* table, const u64* in,
                                           std::size_t n, u64 mask,
                                           int out_words, u64* out) {
  const std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  const u64 x = in[idx] & mask;
  const std::size_t table_offset = static_cast<std::size_t>(x) * out_words;
  const std::size_t out_offset = idx * static_cast<std::size_t>(out_words);
  for (int w = 0; w < out_words; ++w) {
    out[out_offset + w] = table[table_offset + w];
  }
}

void interval_lut_add_base_gpu(const u64* d_base, u64* d_out,
                               std::size_t n, int out_words,
                               cudaStream_t stream) {
  const int threads = 256;
  const int blocks = static_cast<int>((n + threads - 1) / threads);
  kernel_interval_lut_add_base<<<blocks, threads, 0, stream>>>(d_base, d_out, n, out_words);
}

void interval_lut_direct_gpu(const u64* d_table, const u64* d_in,
                             std::size_t n, int in_bits, int out_words,
                             u64* d_out, cudaStream_t stream) {
  if (n == 0) return;
  const u64 mask = (in_bits >= 64) ? ~0ULL : ((1ULL << in_bits) - 1ULL);
  const int threads = 256;
  const int blocks = static_cast<int>((n + threads - 1) / threads);
  kernel_interval_lut_direct<<<blocks, threads, 0, stream>>>(d_table, d_in, n, mask, out_words, d_out);
}

} // namespace suf

#endif // SUF_HAVE_CUDA
