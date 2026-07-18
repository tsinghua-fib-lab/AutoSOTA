#pragma once

#include "suf/crypto/dcf.hpp"

#ifdef SUF_HAVE_CUDA
#include <cuda_runtime.h>
#endif

#include <vector>

namespace suf {

struct DcfKeyPacked {
  Seed root;
  u64 tcwL = 0;
  u64 tcwR = 0;
  int scw_offset = 0;
  int vcw_offset = 0;
  int n_bits = 0;
  u64 mask = ~0ULL;
  u64 g = 0;
  u8 party = 0;
  u8 pad0 = 0;
  u16 pad1 = 0;
};

struct DcfKeyBatch {
  int n_bits = 0;
  std::vector<DcfKeyPacked> keys;
  std::vector<Seed> scw;
  std::vector<u64> vcw;
};

#ifdef SUF_HAVE_CUDA
struct DcfKeyBatchGpu {
  DcfKeyPacked* d_keys = nullptr;
  Seed* d_scw = nullptr;
  u64* d_vcw = nullptr;
  int num_keys = 0;
};

void upload_dcf_batch(const DcfKeyBatch& batch, DcfKeyBatchGpu& out);
void free_dcf_batch(DcfKeyBatchGpu& batch);
void eval_dcf_batch_gpu(const u64* d_in, std::size_t n,
                        const DcfKeyBatchGpu& batch,
                        u64* d_out, cudaStream_t stream = nullptr);
#endif

} // namespace suf
