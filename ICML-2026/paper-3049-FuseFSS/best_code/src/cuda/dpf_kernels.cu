#include "suf/pfss_batch.hpp"
#include "suf/crypto/chacha.hpp"

#ifdef SUF_HAVE_CUDA

#include <cuda_runtime.h>

namespace suf {

namespace {
__device__ __forceinline__ u8 eval_dpf_lt_device(const DpfKeyPacked& key, const Seed* scw, u64 x) {
  const u64 xshift = x + key.input_add;
  u64 xmask = (key.n_bits == 64) ? xshift : (xshift & key.mask);
  Seed s = key.root;
  u8 t = key.t_init;
  u8 x_prev = 1;
  u8 t_dcf = 0;

  for (int i = 0; i < key.n_bits; ++i) {
    const u8 x_i = static_cast<u8>((xmask >> (key.n_bits - 1 - i)) & 1ULL);
    if (x_prev != x_i) t_dcf ^= t;
    x_prev = x_i;

    Seed sL, sR;
    u8 tL, tR;
    prg_expand(s, sL, tL, sR, tR);

    Seed s_next = x_i ? sR : sL;
    u8 t_next = x_i ? tR : tL;

    if (t) {
      s_next = seed_xor(s_next, scw[i]);
      const u64 tcw = x_i ? key.tcwR : key.tcwL;
      t_next = static_cast<u8>(t_next ^ ((tcw >> (key.n_bits - 1 - i)) & 1ULL));
    }

    s = s_next;
    t = t_next;
  }
  if (x_prev == 0) t_dcf ^= t;
  u8 out = t_dcf & 1u;
  if (key.invert) out ^= 1u;
  return out;
}

__global__ void kernel_eval_dpf_lt(const u64* x, std::size_t n,
                                   const DpfKeyPacked* keys, int num_keys,
                                   const Seed* scw, u8* out) {
  const std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  const u64 xi = x[idx];
  for (int k = 0; k < num_keys; ++k) {
    const DpfKeyPacked key = keys[k];
    const Seed* scw_key = scw + key.scw_offset;
    out[static_cast<std::size_t>(k) * n + idx] = eval_dpf_lt_device(key, scw_key, xi);
  }
}

void check_cuda(cudaError_t err, const char* msg) {
  if (err != cudaSuccess) {
    fail(msg);
  }
}
} // namespace

void upload_dpf_batch(const DpfKeyBatch& batch, DpfKeyBatchGpu& out) {
  out.num_keys = static_cast<int>(batch.keys.size());
  if (out.num_keys == 0) return;
  check_cuda(cudaMalloc(&out.d_keys, batch.keys.size() * sizeof(DpfKeyPacked)), "cudaMalloc d_keys failed");
  check_cuda(cudaMalloc(&out.d_scw, batch.scw.size() * sizeof(Seed)), "cudaMalloc d_scw failed");
  check_cuda(cudaMemcpy(out.d_keys, batch.keys.data(), batch.keys.size() * sizeof(DpfKeyPacked), cudaMemcpyHostToDevice),
             "cudaMemcpy keys failed");
  check_cuda(cudaMemcpy(out.d_scw, batch.scw.data(), batch.scw.size() * sizeof(Seed), cudaMemcpyHostToDevice),
             "cudaMemcpy scw failed");
}

void free_dpf_batch(DpfKeyBatchGpu& batch) {
  if (batch.d_keys) cudaFree(batch.d_keys);
  if (batch.d_scw) cudaFree(batch.d_scw);
  batch.d_keys = nullptr;
  batch.d_scw = nullptr;
  batch.num_keys = 0;
}

void eval_dpf_batch_gpu(const u64* d_in, std::size_t n,
                        const DpfKeyBatchGpu& batch,
                        u8* d_out, cudaStream_t stream) {
  const int threads = 256;
  const int blocks = static_cast<int>((n + threads - 1) / threads);
  kernel_eval_dpf_lt<<<blocks, threads, 0, stream>>>(d_in, n, batch.d_keys, batch.num_keys, batch.d_scw, d_out);
}

} // namespace suf

#endif // SUF_HAVE_CUDA
