#include "suf/dmpf.hpp"
#include "suf/crypto/dcf.hpp"

namespace suf {

static void append_dcf_key(const DcfKey& k, int party, DcfKeyBatch& batch) {
  DcfKeyPacked packed;
  packed.root = k.root;
  packed.tcwL = k.tcwL;
  packed.tcwR = k.tcwR;
  packed.scw_offset = static_cast<int>(batch.scw.size());
  packed.vcw_offset = static_cast<int>(batch.vcw.size());
  packed.n_bits = k.n_bits;
  packed.mask = (k.n_bits == 64) ? ~0ULL : ((1ULL << k.n_bits) - 1ULL);
  packed.g = k.g;
  packed.party = static_cast<u8>(party & 1);

  batch.keys.push_back(packed);
  batch.scw.insert(batch.scw.end(), k.scw.begin(), k.scw.end());
  batch.vcw.insert(batch.vcw.end(), k.vcw.begin(), k.vcw.end());
}

DmpfKey gen_dmpf_key(const std::vector<u64>& cutpoints,
                     const std::vector<u64>& deltas,
                     int in_bits,
                     int party,
                     std::mt19937_64& rng,
                     std::size_t out_words) {
  ensure(in_bits > 0 && in_bits <= 64, "dmpf: in_bits 1..64");
  const std::size_t points = cutpoints.size();
  ensure(out_words > 0, "dmpf: out_words must be > 0");
  ensure(deltas.size() == points * out_words, "dmpf: delta size mismatch");

  DmpfKey key;
  key.in_bits = in_bits;
  key.points = points;
  key.out_words = out_words;
  key.deltas = deltas;
  key.dcf_batch.n_bits = in_bits;

  key.dcf_batch.keys.reserve(points);
  key.dcf_batch.scw.reserve(points * in_bits);
  key.dcf_batch.vcw.reserve(points * in_bits);

  for (std::size_t i = 0; i < points; ++i) {
    auto kp = keygen_dcf_lt(in_bits, cutpoints[i], rng);
    const DcfKey& k = (party == 0) ? kp.k0 : kp.k1;
    append_dcf_key(k, party, key.dcf_batch);
  }

  return key;
}

void eval_dmpf_suffix_cpu(const DmpfKey& key,
                          const std::vector<u64>& inputs,
                          std::vector<u64>& outputs) {
  const std::size_t n = inputs.size();
  const std::size_t out_words = key.out_words;
  outputs.assign(n * out_words, 0);

  const auto& batch = key.dcf_batch;
  for (std::size_t b = 0; b < batch.keys.size(); ++b) {
    const DcfKeyPacked& packed = batch.keys[b];
    DcfKey view;
    view.n_bits = packed.n_bits;
    view.root = packed.root;
    view.scw.assign(batch.scw.begin() + packed.scw_offset,
                    batch.scw.begin() + packed.scw_offset + packed.n_bits);
    view.vcw.assign(batch.vcw.begin() + packed.vcw_offset,
                    batch.vcw.begin() + packed.vcw_offset + packed.n_bits);
    view.tcwL = packed.tcwL;
    view.tcwR = packed.tcwR;
    view.g = packed.g;

    for (std::size_t i = 0; i < n; ++i) {
      const u64 share = eval_dcf_lt_cpu(packed.party, view, inputs[i]);
      for (std::size_t w = 0; w < out_words; ++w) {
        outputs[i * out_words + w] += share * key.deltas[b * out_words + w];
      }
    }
  }
}

} // namespace suf
