#include "suf/interval_lut.hpp"

#include <algorithm>
#include <cstdlib>

namespace suf {

IntervalLutKeyV2 gen_interval_lut_v2(const std::vector<u64>& cutpoints,
                                    const std::vector<std::vector<u64>>& payloads,
                                    int in_bits,
                                    int party,
                                    std::mt19937_64& rng) {
  ensure(!cutpoints.empty(), "interval LUT: empty cutpoints");
  ensure(cutpoints.size() == payloads.size(), "interval LUT: payload size mismatch");
  const std::size_t intervals = cutpoints.size();
  const std::size_t out_words = payloads[0].size();
  for (const auto& p : payloads) {
    ensure(p.size() == out_words, "interval LUT: payload word mismatch");
  }

  IntervalLutKeyV2 key;
  key.hdr.magic = 0x53494C32;
  key.hdr.version = 2;
  key.hdr.in_bits = static_cast<u8>(in_bits);
  key.hdr.out_bits = 64;
  key.hdr.out_words = static_cast<u8>(out_words);
  key.hdr.intervals = static_cast<u32>(intervals);

  const int kDirectTableMaxBits = []() {
    const char* v = std::getenv("SUF_DIRECT_LUT_BITS");
    if (!v || !*v) return 16;
    const int parsed = std::atoi(v);
    return parsed > 0 ? parsed : 16;
  }();
  if (in_bits < 64) {
    const u64 domain = (1ULL << in_bits);
    for (const auto c : cutpoints) {
      ensure(c < domain, "interval LUT: cutpoint out of range for in_bits");
    }
  }
  const bool use_direct_table = (in_bits > 0 && in_bits <= kDirectTableMaxBits
                                 && static_cast<std::size_t>(intervals) <= (1ULL << in_bits));
  if (use_direct_table) {
    key.hdr.flags |= kIntervalLutFlagDirectTable;
    key.base_share.assign(out_words, 0);

    const std::size_t table_size = static_cast<std::size_t>(1ULL << in_bits);
    key.dmpf.in_bits = in_bits;
    key.dmpf.points = table_size;
    key.dmpf.out_words = out_words;
    key.dmpf.dcf_batch.n_bits = in_bits;
    key.dmpf.deltas.resize(table_size * out_words);

    std::size_t interval_idx = 0;
    for (std::size_t x = 0; x < table_size; ++x) {
      while (interval_idx + 1 < intervals && x >= cutpoints[interval_idx + 1]) {
        ++interval_idx;
      }
      for (std::size_t w = 0; w < out_words; ++w) {
        const u64 value = payloads[interval_idx][w];
        const u64 share0 = rng();
        key.dmpf.deltas[x * out_words + w] = (party == 0) ? share0 : (value - share0);
      }
    }

    key.hdr.core_bytes = 0;
    key.hdr.payload_bytes = static_cast<u32>(key.dmpf.deltas.size() * sizeof(u64)
                                            + key.base_share.size() * sizeof(u64));
    return key;
  }

  // base = last payload
  key.base_share.resize(out_words);
  std::vector<u64> base(out_words);
  for (std::size_t w = 0; w < out_words; ++w) base[w] = payloads[intervals - 1][w];
  // random split base into shares
  std::vector<u64> base0(out_words);
  for (std::size_t w = 0; w < out_words; ++w) base0[w] = rng();
  if (party == 0) {
    key.base_share = base0;
  } else {
    for (std::size_t w = 0; w < out_words; ++w) {
      key.base_share[w] = base[w] - base0[w];
    }
  }

  // deltas and DMPF core (one key per boundary)
  std::vector<u64> deltas((intervals - 1) * out_words);
  for (std::size_t i = 0; i + 1 < intervals; ++i) {
    for (std::size_t w = 0; w < out_words; ++w) {
      deltas[i * out_words + w] = payloads[i][w] - payloads[i + 1][w];
    }
  }

  std::vector<u64> cutpoints_dmpf;
  cutpoints_dmpf.reserve(intervals - 1);
  for (std::size_t i = 1; i < intervals; ++i) {
    cutpoints_dmpf.push_back(cutpoints[i]);
  }

  key.dmpf = gen_dmpf_key(cutpoints_dmpf, deltas, in_bits, party, rng, out_words);

  key.hdr.core_bytes = static_cast<u32>(key.dmpf.dcf_batch.keys.size() * sizeof(DcfKeyPacked)
                                       + key.dmpf.dcf_batch.scw.size() * sizeof(Seed)
                                       + key.dmpf.dcf_batch.vcw.size() * sizeof(u64));
  key.hdr.payload_bytes = static_cast<u32>(key.dmpf.deltas.size() * sizeof(u64) + key.base_share.size() * sizeof(u64));

  return key;
}

void eval_interval_lut_v2_cpu(const IntervalLutKeyV2& key,
                              const std::vector<u64>& inputs,
                              std::vector<u64>& outputs) {
  const std::size_t n = inputs.size();
  const std::size_t out_words = key.hdr.out_words;
  if (key.hdr.flags & kIntervalLutFlagDirectTable) {
    const u64 mask = (key.hdr.in_bits >= 64) ? ~0ULL : ((1ULL << key.hdr.in_bits) - 1ULL);
    outputs.resize(n * out_words);
    for (std::size_t i = 0; i < n; ++i) {
      const std::size_t idx = static_cast<std::size_t>(inputs[i] & mask);
      for (std::size_t w = 0; w < out_words; ++w) {
        outputs[i * out_words + w] = key.dmpf.deltas[idx * out_words + w];
      }
    }
    return;
  }
  eval_dmpf_suffix_cpu(key.dmpf, inputs, outputs);
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t w = 0; w < out_words; ++w) {
      outputs[i * out_words + w] += key.base_share[w];
    }
  }
}

#ifdef SUF_HAVE_CUDA
void upload_interval_lut_v2(const IntervalLutKeyV2& key, IntervalLutKeyV2Gpu& out) {
  out.hdr = key.hdr;
  cudaMalloc(&out.d_base, key.base_share.size() * sizeof(u64));
  cudaMemcpy(out.d_base, key.base_share.data(), key.base_share.size() * sizeof(u64), cudaMemcpyHostToDevice);
  upload_dmpf_key(key.dmpf, out.dmpf);
}

void free_interval_lut_v2(IntervalLutKeyV2Gpu& key) {
  if (key.d_base) cudaFree(key.d_base);
  free_dmpf_key(key.dmpf);
  key.d_base = nullptr;
}

void eval_interval_lut_v2_gpu(const u64* d_in, std::size_t n,
                              const IntervalLutKeyV2Gpu& key,
                              u64* d_out, cudaStream_t stream) {
  if (n == 0) return;
  if (key.hdr.flags & kIntervalLutFlagDirectTable) {
    interval_lut_direct_gpu(key.dmpf.d_deltas, d_in, n, key.hdr.in_bits,
                            key.hdr.out_words, d_out, stream);
    return;
  }
  eval_dmpf_suffix_add_base_gpu(d_in, n, key.dmpf, key.d_base, d_out, stream);
}
#endif

} // namespace suf
