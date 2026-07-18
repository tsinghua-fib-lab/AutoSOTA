#pragma once

#include "suf/crypto/chacha.hpp"

#include <vector>
#include <random>

namespace suf {

struct DpfKey {
  int n = 0;
  Seed root;
  std::vector<Seed> scw; // size n
  u64 tcwL = 0;
  u64 tcwR = 0;
};

inline std::pair<DpfKey, DpfKey> keygen_dpf(int n, u64 alpha, std::mt19937_64& rng) {
  ensure(n > 0 && n <= 64, "dpf: n must be 1..64");
  const u64 mask_bit = 1ULL;

  DpfKey k0;
  DpfKey k1;
  k0.n = n;
  k1.n = n;
  k0.scw.resize(n);
  k1.scw.resize(n);

  Seed s0{rng(), rng()};
  Seed s1{rng(), rng()};
  s0.lo &= ~mask_bit;
  s1.lo &= ~mask_bit;
  k0.root = s0;
  k1.root = s1;

  u8 t0 = 0;
  u8 t1 = 1;

  for (int i = 0; i < n; ++i) {
    const u8 keep = static_cast<u8>((alpha >> (n - 1 - i)) & 1ULL);
    const u8 loose = keep ^ 1u;

    Seed s0L, s0R, s1L, s1R;
    u8 t0L, t0R, t1L, t1R;
    prg_expand(s0, s0L, t0L, s0R, t0R);
    prg_expand(s1, s1L, t1L, s1R, t1R);

    const Seed s0_loose = loose ? s0R : s0L;
    const Seed s1_loose = loose ? s1R : s1L;
    const Seed scw = seed_xor(s0_loose, s1_loose);

    u8 tLcw = static_cast<u8>(t0L ^ t1L ^ keep ^ 1u);
    u8 tRcw = static_cast<u8>(t0R ^ t1R ^ keep);

    k0.scw[i] = scw;
    k1.scw[i] = scw;
    if (tLcw) k0.tcwL |= (1ULL << (n - 1 - i));
    if (tRcw) k0.tcwR |= (1ULL << (n - 1 - i));

    const Seed s0_keep = keep ? s0R : s0L;
    const Seed s1_keep = keep ? s1R : s1L;
    const u8 t0_keep = keep ? t0R : t0L;
    const u8 t1_keep = keep ? t1R : t1L;

    if (t0 == 0) {
      s0 = s0_keep;
      t0 = t0_keep;
    } else {
      s0 = seed_xor(s0_keep, scw);
      t0 = static_cast<u8>(t0_keep ^ (keep ? tRcw : tLcw));
    }

    if (t1 == 0) {
      s1 = s1_keep;
      t1 = t1_keep;
    } else {
      s1 = seed_xor(s1_keep, scw);
      t1 = static_cast<u8>(t1_keep ^ (keep ? tRcw : tLcw));
    }
  }

  k1.tcwL = k0.tcwL;
  k1.tcwR = k0.tcwR;

  return {k0, k1};
}

inline u8 eval_dpf_lt_cpu(int party, const DpfKey& key, u64 x) {
  Seed s = key.root;
  u8 t = static_cast<u8>(party & 1);
  u8 x_prev = 1;
  u8 t_dcf = 0;

  for (int i = 0; i < key.n; ++i) {
    const u8 x_i = static_cast<u8>((x >> (key.n - 1 - i)) & 1ULL);
    if (x_prev != x_i) t_dcf ^= t;
    x_prev = x_i;

    Seed sL, sR;
    u8 tL, tR;
    prg_expand(s, sL, tL, sR, tR);

    Seed s_next = x_i ? sR : sL;
    u8 t_next = x_i ? tR : tL;

    if (t) {
      s_next = seed_xor(s_next, key.scw[i]);
      const u64 tcw = x_i ? key.tcwR : key.tcwL;
      t_next = static_cast<u8>(t_next ^ ((tcw >> (key.n - 1 - i)) & 1ULL));
    }

    s = s_next;
    t = t_next;
  }

  if (x_prev == 0) t_dcf ^= t;
  return t_dcf & 1u;
}

} // namespace suf
