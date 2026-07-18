#include "suf/dcf_batch.hpp"
#include "suf/crypto/chacha.hpp"

#ifdef SUF_HAVE_CUDA

#include <cuda_runtime.h>

namespace suf {

namespace {
__device__ __forceinline__ u64 seed_value_dev(const Seed& s) {
  return (s.lo & ~3ULL);
}

__device__ __forceinline__ void prg_expand_dcf_dev(const Seed& seed, Seed& sL, u64& vL, Seed& sR, u64& vR) {
  Seed ss = seed;
  ss.lo &= ~3ULL;
  u32 block0[16];
  u32 block1[16];
  chacha_block(ss, 0, block0);
  chacha_block(ss, 1, block1);
  sL.lo = static_cast<u64>(block0[0]) | (static_cast<u64>(block0[1]) << 32);
  sL.hi = static_cast<u64>(block0[2]) | (static_cast<u64>(block0[3]) << 32);
  sR.lo = static_cast<u64>(block1[0]) | (static_cast<u64>(block1[1]) << 32);
  sR.hi = static_cast<u64>(block1[2]) | (static_cast<u64>(block1[3]) << 32);

  u32 block2[16];
  chacha_block(ss, 2, block2);
  vL = static_cast<u64>(block2[0]) | (static_cast<u64>(block2[1]) << 32);
  vR = static_cast<u64>(block2[2]) | (static_cast<u64>(block2[3]) << 32);
}

__device__ __forceinline__ u64 eval_dcf_lt_device(const DcfKeyPacked& key, const Seed* scw, const u64* vcw, u64 x) {
  u64 xmask = (key.n_bits == 64) ? x : (x & key.mask);
  Seed s = key.root;
  u8 t = seed_lsb(s);
  u64 v_share = 0;
  const u64 sign = (key.party == 1) ? static_cast<u64>(0) - 1ULL : 1ULL;

  for (int i = 0; i < key.n_bits; ++i) {
    const u8 bit = static_cast<u8>((xmask >> (key.n_bits - 1 - i)) & 1ULL);
    Seed sL, sR;
    u64 vL, vR;
    prg_expand_dcf_dev(s, sL, vL, sR, vR);

    Seed s_keep = bit ? sR : sL;
    u64 v_keep = bit ? vR : vL;

    const u8 t_prev = t;
    v_share = v_share + sign * (v_keep + static_cast<u64>(t_prev) * vcw[key.vcw_offset + i]);

    if (t_prev) {
      Seed scw_i = scw[key.scw_offset + i];
      const u8 t_cw = bit ? ((key.tcwR >> (key.n_bits - 1 - i)) & 1ULL)
                          : ((key.tcwL >> (key.n_bits - 1 - i)) & 1ULL);
      s_keep = seed_xor(s_keep, scw_i);
      if (t_cw) s_keep.lo ^= 1ULL;
    }

    s = s_keep;
    t = seed_lsb(s);
  }

  u64 final_term = seed_value_dev(s);
  if (t) final_term = final_term + key.g;
  if (key.party == 1) final_term = static_cast<u64>(0) - final_term;
  return v_share + final_term;
}

__global__ void kernel_eval_dcf_lt(const u64* x, std::size_t n,
                                   const DcfKeyPacked* keys, int num_keys,
                                   const Seed* scw, const u64* vcw,
                                   u64* out) {
  const std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  const u64 xi = x[idx];
  for (int k = 0; k < num_keys; ++k) {
    const DcfKeyPacked key = keys[k];
    out[static_cast<std::size_t>(k) * n + idx] = eval_dcf_lt_device(key, scw, vcw, xi);
  }
}

void check_cuda(cudaError_t err, const char* msg) {
  if (err != cudaSuccess) {
    fail(msg);
  }
}
} // namespace

void upload_dcf_batch(const DcfKeyBatch& batch, DcfKeyBatchGpu& out) {
  out.num_keys = static_cast<int>(batch.keys.size());
  if (out.num_keys == 0) return;
  check_cuda(cudaMalloc(&out.d_keys, batch.keys.size() * sizeof(DcfKeyPacked)), "cudaMalloc d_keys failed");
  check_cuda(cudaMalloc(&out.d_scw, batch.scw.size() * sizeof(Seed)), "cudaMalloc d_scw failed");
  check_cuda(cudaMalloc(&out.d_vcw, batch.vcw.size() * sizeof(u64)), "cudaMalloc d_vcw failed");

  check_cuda(cudaMemcpy(out.d_keys, batch.keys.data(), batch.keys.size() * sizeof(DcfKeyPacked), cudaMemcpyHostToDevice),
             "cudaMemcpy keys failed");
  check_cuda(cudaMemcpy(out.d_scw, batch.scw.data(), batch.scw.size() * sizeof(Seed), cudaMemcpyHostToDevice),
             "cudaMemcpy scw failed");
  check_cuda(cudaMemcpy(out.d_vcw, batch.vcw.data(), batch.vcw.size() * sizeof(u64), cudaMemcpyHostToDevice),
             "cudaMemcpy vcw failed");
}

void free_dcf_batch(DcfKeyBatchGpu& batch) {
  if (batch.d_keys) cudaFree(batch.d_keys);
  if (batch.d_scw) cudaFree(batch.d_scw);
  if (batch.d_vcw) cudaFree(batch.d_vcw);
  batch.d_keys = nullptr;
  batch.d_scw = nullptr;
  batch.d_vcw = nullptr;
  batch.num_keys = 0;
}

void eval_dcf_batch_gpu(const u64* d_in, std::size_t n,
                        const DcfKeyBatchGpu& batch,
                        u64* d_out, cudaStream_t stream) {
  const int threads = 256;
  const int blocks = static_cast<int>((n + threads - 1) / threads);
  kernel_eval_dcf_lt<<<blocks, threads, 0, stream>>>(d_in, n, batch.d_keys, batch.num_keys, batch.d_scw, batch.d_vcw, d_out);
}

} // namespace suf

#endif // SUF_HAVE_CUDA
