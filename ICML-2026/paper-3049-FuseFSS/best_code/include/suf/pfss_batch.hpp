#pragma once

#include "suf/crypto/dpf.hpp"

#ifdef SUF_HAVE_CUDA
#include <cuda_runtime.h>
#endif

#include <vector>

namespace suf {

struct DpfKeyPacked {
  Seed root;
  u64 tcwL = 0;
  u64 tcwR = 0;
  int scw_offset = 0;
  int n_bits = 0;
  u64 mask = ~0ULL;
  u8 t_init = 0;
  u8 invert = 0;
  u16 pad0 = 0;
  u64 input_add = 0;
};

struct DpfKeyBatch {
  int n_bits = 0;
  std::vector<DpfKeyPacked> keys;
  std::vector<Seed> scw;
};

#ifdef SUF_HAVE_CUDA
struct DpfKeyBatchGpu {
  DpfKeyPacked* d_keys = nullptr;
  Seed* d_scw = nullptr;
  int num_keys = 0;
};

void upload_dpf_batch(const DpfKeyBatch& batch, DpfKeyBatchGpu& out);
void free_dpf_batch(DpfKeyBatchGpu& batch);
void eval_dpf_batch_gpu(const u64* d_in, std::size_t n,
                        const DpfKeyBatchGpu& batch,
                        u8* d_out, cudaStream_t stream = nullptr);
#endif

} // namespace suf
