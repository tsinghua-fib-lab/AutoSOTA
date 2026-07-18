#pragma once

#include "suf/common.hpp"
#include "suf/dcf_batch.hpp"

#ifdef SUF_HAVE_CUDA
#include <cuda_runtime.h>
#endif

#include <vector>

namespace suf {

// Baseline DMPF core: uses scalar DCF batch internally (placeholder for shared-expansion DMPF).
// Interface matches suffix-sum evaluation expected by SIL2 IntervalLUT.
struct DmpfKey {
  int in_bits = 0;
  std::size_t points = 0;
  std::vector<u64> deltas; // size points * out_words
  std::size_t out_words = 0;
  DcfKeyBatch dcf_batch;
};

#ifdef SUF_HAVE_CUDA
struct DmpfKeyGpu {
  int in_bits = 0;
  std::size_t points = 0;
  std::size_t out_words = 0;
  u64* d_deltas = nullptr;
  DcfKeyBatchGpu dcf;
};

void upload_dmpf_key(const DmpfKey& key, DmpfKeyGpu& out);
void free_dmpf_key(DmpfKeyGpu& key);

// Computes suffix sum S(u) = sum_{i: u < c_i} delta[i] (per out_word),
// returned as additive shares.
void eval_dmpf_suffix_gpu(const u64* d_in, std::size_t n,
                          const DmpfKeyGpu& key,
                          u64* d_out, cudaStream_t stream = nullptr);

// Computes suffix sum and adds a constant base vector (per out_word) to each output.
void eval_dmpf_suffix_add_base_gpu(const u64* d_in, std::size_t n,
                                  const DmpfKeyGpu& key,
                                  const u64* d_base,
                                  u64* d_out, cudaStream_t stream = nullptr);
#endif

DmpfKey gen_dmpf_key(const std::vector<u64>& cutpoints,
                     const std::vector<u64>& deltas,
                     int in_bits,
                     int party,
                     std::mt19937_64& rng,
                     std::size_t out_words);

void eval_dmpf_suffix_cpu(const DmpfKey& key,
                          const std::vector<u64>& inputs,
                          std::vector<u64>& outputs);

} // namespace suf
