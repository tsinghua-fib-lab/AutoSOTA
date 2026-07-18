#include "suf/dmpf.hpp"
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

__global__ void kernel_dmpf_suffix_base1(const u64* x, std::size_t n,
                                        const DcfKeyPacked* keys, int num_keys,
                                        const Seed* scw, const u64* vcw,
                                        const u64* deltas, const u64* base,
                                        u64* out) {
  const std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  const u64 xi = x[idx];
  u64 acc = base ? base[0] : 0;
  for (int k = 0; k < num_keys; ++k) {
    const DcfKeyPacked key = keys[k];
    const u64 bshare = eval_dcf_lt_device(key, scw, vcw, xi);
    acc += bshare * deltas[k];
  }
  out[idx] = acc;
}

__global__ void kernel_dmpf_suffix_baseN(const u64* x, std::size_t n,
                                        const DcfKeyPacked* keys, int num_keys,
                                        const Seed* scw, const u64* vcw,
                                        const u64* deltas, int out_words,
                                        const u64* base, u64* out) {
  const std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  const u64 xi = x[idx];
  const std::size_t out_offset = idx * static_cast<std::size_t>(out_words);
  if (base) {
    for (int w = 0; w < out_words; ++w) {
      out[out_offset + w] = base[w];
    }
  } else {
    for (int w = 0; w < out_words; ++w) {
      out[out_offset + w] = 0;
    }
  }
  for (int k = 0; k < num_keys; ++k) {
    const DcfKeyPacked key = keys[k];
    const u64 bshare = eval_dcf_lt_device(key, scw, vcw, xi);
    const std::size_t delta_offset = static_cast<std::size_t>(k) * out_words;
    for (int w = 0; w < out_words; ++w) {
      out[out_offset + w] += bshare * deltas[delta_offset + w];
    }
  }
}
} // namespace

void upload_dmpf_key(const DmpfKey& key, DmpfKeyGpu& out) {
  out.in_bits = key.in_bits;
  out.points = key.points;
  out.out_words = key.out_words;
  if (!key.deltas.empty()) {
    cudaMalloc(&out.d_deltas, key.deltas.size() * sizeof(u64));
    cudaMemcpy(out.d_deltas, key.deltas.data(),
               key.deltas.size() * sizeof(u64), cudaMemcpyHostToDevice);
  }
  upload_dcf_batch(key.dcf_batch, out.dcf);
}

void free_dmpf_key(DmpfKeyGpu& key) {
  if (key.d_deltas) cudaFree(key.d_deltas);
  free_dcf_batch(key.dcf);
  key.d_deltas = nullptr;
}

void eval_dmpf_suffix_gpu(const u64* d_in, std::size_t n,
                          const DmpfKeyGpu& key,
                          u64* d_out, cudaStream_t stream) {
  const int num_keys = key.dcf.num_keys;
  if (n == 0 || key.out_words == 0) return;
  if (num_keys == 0) {
    cudaMemsetAsync(d_out, 0, n * key.out_words * sizeof(u64), stream);
    return;
  }

  const int threads = 256;
  const int blocks = static_cast<int>((n + threads - 1) / threads);
  if (key.out_words == 1) {
    kernel_dmpf_suffix_base1<<<blocks, threads, 0, stream>>>(
        d_in, n, key.dcf.d_keys, num_keys, key.dcf.d_scw, key.dcf.d_vcw,
        key.d_deltas, nullptr, d_out);
  } else {
    kernel_dmpf_suffix_baseN<<<blocks, threads, 0, stream>>>(
        d_in, n, key.dcf.d_keys, num_keys, key.dcf.d_scw, key.dcf.d_vcw,
        key.d_deltas, static_cast<int>(key.out_words), nullptr, d_out);
  }
}

void eval_dmpf_suffix_add_base_gpu(const u64* d_in, std::size_t n,
                                   const DmpfKeyGpu& key,
                                   const u64* d_base,
                                   u64* d_out, cudaStream_t stream) {
  const int num_keys = key.dcf.num_keys;
  if (n == 0 || key.out_words == 0) return;

  const int threads = 256;
  const int blocks = static_cast<int>((n + threads - 1) / threads);
  if (key.out_words == 1) {
    kernel_dmpf_suffix_base1<<<blocks, threads, 0, stream>>>(
        d_in, n, key.dcf.d_keys, num_keys, key.dcf.d_scw, key.dcf.d_vcw,
        key.d_deltas, d_base, d_out);
  } else {
    kernel_dmpf_suffix_baseN<<<blocks, threads, 0, stream>>>(
        d_in, n, key.dcf.d_keys, num_keys, key.dcf.d_scw, key.dcf.d_vcw,
        key.d_deltas, static_cast<int>(key.out_words), d_base, d_out);
  }
}

} // namespace suf

#endif // SUF_HAVE_CUDA
