#include "suf/interval_lut.hpp"
#include "suf/validate.hpp"

#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <random>
#include <algorithm>

using namespace suf;

int main() {
  const int in_bits = 12;
  const std::size_t intervals = 8;
  const std::size_t out_words = 3;
  const std::size_t N = 256;

  std::mt19937_64 rng(1234);

  std::vector<u64> cutpoints(intervals);
  for (std::size_t i = 0; i < intervals; ++i) {
    cutpoints[i] = static_cast<u64>(i * (1ULL << (in_bits - 3)));
  }

  std::vector<std::vector<u64>> payloads(intervals, std::vector<u64>(out_words));
  for (std::size_t i = 0; i < intervals; ++i) {
    for (std::size_t w = 0; w < out_words; ++w) payloads[i][w] = rng();
  }

  std::vector<u64> inputs(N);
  for (std::size_t i = 0; i < N; ++i) inputs[i] = rng() & ((1ULL << in_bits) - 1ULL);

  std::mt19937_64 rng_keys(999);
  auto key0 = gen_interval_lut_v2(cutpoints, payloads, in_bits, 0, rng_keys);
  std::mt19937_64 rng_keys2(999);
  auto key1 = gen_interval_lut_v2(cutpoints, payloads, in_bits, 1, rng_keys2);

  IntervalLutKeyV2Gpu d_key0, d_key1;
  upload_interval_lut_v2(key0, d_key0);
  upload_interval_lut_v2(key1, d_key1);

  u64* d_in = nullptr;
  u64* d_out0 = nullptr;
  u64* d_out1 = nullptr;
  cudaMalloc(&d_in, N * sizeof(u64));
  cudaMalloc(&d_out0, N * out_words * sizeof(u64));
  cudaMalloc(&d_out1, N * out_words * sizeof(u64));
  cudaMemcpy(d_in, inputs.data(), N * sizeof(u64), cudaMemcpyHostToDevice);

  eval_interval_lut_v2_gpu(d_in, N, d_key0, d_out0, 0);
  eval_interval_lut_v2_gpu(d_in, N, d_key1, d_out1, 0);
  cudaDeviceSynchronize();

  std::vector<u64> out0(N * out_words), out1(N * out_words);
  cudaMemcpy(out0.data(), d_out0, N * out_words * sizeof(u64), cudaMemcpyDeviceToHost);
  cudaMemcpy(out1.data(), d_out1, N * out_words * sizeof(u64), cudaMemcpyDeviceToHost);

  // reference
  for (std::size_t i = 0; i < N; ++i) {
    const u64 x = inputs[i];
    std::size_t idx = 0;
    for (std::size_t j = 1; j < intervals; ++j) {
      if (x >= cutpoints[j]) idx = j;
    }
    for (std::size_t w = 0; w < out_words; ++w) {
      u64 got = out0[i * out_words + w] + out1[i * out_words + w];
      u64 exp = payloads[idx][w];
      if (got != exp) {
        std::cerr << "mismatch at i=" << i << " w=" << w << " got=" << got << " exp=" << exp << "\n";
        return 1;
      }
    }
  }

  free_interval_lut_v2(d_key0);
  free_interval_lut_v2(d_key1);
  cudaFree(d_in);
  cudaFree(d_out0);
  cudaFree(d_out1);

  std::cout << "ok\n";
  return 0;
}
