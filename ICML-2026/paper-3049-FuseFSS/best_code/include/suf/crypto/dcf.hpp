#pragma once

#include "suf/crypto/chacha.hpp"

#include <random>
#include <vector>

namespace suf {

struct DcfKey {
  int n_bits = 0;
  Seed root;
  std::vector<Seed> scw; // size n_bits
  u64 tcwL = 0;
  u64 tcwR = 0;
  std::vector<u64> vcw; // size n_bits
  u64 g = 0; // final correction
};

struct DcfKeyPair {
  DcfKey k0;
  DcfKey k1;
};

inline u64 seed_value(const Seed& s) {
  return (s.lo & ~3ULL); // clear last 2 bits like notThreeBlock
}

inline void prg_expand_dcf(const Seed& seed, Seed& sL, u64& vL, Seed& sR, u64& vR) {
  // derive two child seeds (counter 0/1) without clearing LSBs (t bits come from PRG)
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

inline DcfKeyPair keygen_dcf_lt(int n_bits, u64 alpha, std::mt19937_64& rng) {
  ensure(n_bits > 0 && n_bits <= 64, "dcf: n_bits 1..64");

  DcfKeyPair out;
  out.k0.n_bits = n_bits;
  out.k1.n_bits = n_bits;
  out.k0.scw.resize(n_bits);
  out.k1.scw.resize(n_bits);
  out.k0.vcw.resize(n_bits);
  out.k1.vcw.resize(n_bits);

  Seed s0{rng(), rng()};
  Seed s1{rng(), rng()};
  // set s0 lsb to ~lsb(s1) to ensure t0 != t1
  const u8 t1_init = seed_lsb(s1);
  s0.lo = (s0.lo & ~1ULL) | (static_cast<u64>(t1_init ^ 1u));
  out.k0.root = s0;
  out.k1.root = s1;

  u8 t0 = seed_lsb(s0);
  u8 t1 = seed_lsb(s1);
  u64 v_alpha = 0;

  const u64 payload = 1;

  for (int i = 0; i < n_bits; ++i) {
    const u8 keep = static_cast<u8>((alpha >> (n_bits - 1 - i)) & 1ULL);
    const u8 loose = keep ^ 1u;

    Seed s0L, s0R, s1L, s1R;
    u64 v0L, v0R, v1L, v1R;
    prg_expand_dcf(s0, s0L, v0L, s0R, v0R);
    prg_expand_dcf(s1, s1L, v1L, s1R, v1R);

    u64 v0_keep = keep ? v0R : v0L;
    u64 v1_keep = keep ? v1R : v1L;
    u64 v0_loose = loose ? v0R : v0L;
    u64 v1_loose = loose ? v1R : v1L;

    const u8 sign = (t1 == 1) ? 1 : 0;
    u64 sign_val = sign ? (~0ULL) : 1ULL; // -1 or +1 mod 2^64

    u64 vcw = sign_val * (static_cast<u64>(0) - v_alpha - v0_loose + v1_loose);
    if (keep == 1) {
      vcw = vcw + sign_val * payload;
    }
    v_alpha = v_alpha - v1_keep + v0_keep + sign_val * vcw;

    out.k0.vcw[i] = vcw;
    out.k1.vcw[i] = vcw;

    // scw and tcw bits
    const Seed s0_keep = keep ? s0R : s0L;
    const Seed s1_keep = keep ? s1R : s1L;
    const Seed s0_loose = loose ? s0R : s0L;
    const Seed s1_loose = loose ? s1R : s1L;

    Seed scw = seed_xor(s0_loose, s1_loose);
    scw.lo &= ~3ULL; // clear last 2 bits

    u8 t0L = seed_lsb(s0L);
    u8 t0R = seed_lsb(s0R);
    u8 t1L = seed_lsb(s1L);
    u8 t1R = seed_lsb(s1R);

    u8 tLcw = static_cast<u8>(t0L ^ t1L ^ keep ^ 1u);
    u8 tRcw = static_cast<u8>(t0R ^ t1R ^ keep);

    out.k0.scw[i] = scw;
    out.k1.scw[i] = scw;
    if (tLcw) out.k0.tcwL |= (1ULL << (n_bits - 1 - i));
    if (tRcw) out.k0.tcwR |= (1ULL << (n_bits - 1 - i));

    const u8 t0_keep = keep ? t0R : t0L;
    const u8 t1_keep = keep ? t1R : t1L;

    if (t0 == 0) {
      s0 = s0_keep;
      t0 = t0_keep;
    } else {
      s0 = seed_xor(s0_keep, scw);
      const u8 t_cw = keep ? tRcw : tLcw;
      t0 = static_cast<u8>(t0_keep ^ t_cw);
    }

    if (t1 == 0) {
      s1 = s1_keep;
      t1 = t1_keep;
    } else {
      s1 = seed_xor(s1_keep, scw);
      const u8 t_cw = keep ? tRcw : tLcw;
      t1 = static_cast<u8>(t1_keep ^ t_cw);
    }
  }

  out.k1.tcwL = out.k0.tcwL;
  out.k1.tcwR = out.k0.tcwR;

  const u64 s0_val = seed_value(s0);
  const u64 s1_val = seed_value(s1);
  u64 g = s1_val - s0_val - v_alpha;
  if ((s1.lo & 1ULL) == 1ULL) {
    g = static_cast<u64>(0) - g;
  }
  out.k0.g = g;
  out.k1.g = g;

  return out;
}

inline u64 eval_dcf_lt_cpu(int party, const DcfKey& key, u64 x) {
  const u64 mask = (key.n_bits == 64) ? ~0ULL : ((1ULL << key.n_bits) - 1ULL);
  u64 xmask = x & mask;
  Seed s = key.root;
  u8 t = seed_lsb(s);
  u64 v_share = 0;
  const u64 sign = (party == 1) ? static_cast<u64>(0) - 1ULL : 1ULL;

  for (int i = 0; i < key.n_bits; ++i) {
    const u8 bit = static_cast<u8>((xmask >> (key.n_bits - 1 - i)) & 1ULL);
    Seed sL, sR;
    u64 vL, vR;
    prg_expand_dcf(s, sL, vL, sR, vR);

    Seed s_keep = bit ? sR : sL;
    u64 v_keep = bit ? vR : vL;
    const u8 t_prev = t;
    v_share = v_share + sign * (v_keep + static_cast<u64>(t_prev) * key.vcw[i]);

    if (t_prev) {
      Seed scw_i = key.scw[i];
      const u8 t_cw = bit ? ((key.tcwR >> (key.n_bits - 1 - i)) & 1ULL)
                          : ((key.tcwL >> (key.n_bits - 1 - i)) & 1ULL);
      s_keep = seed_xor(s_keep, scw_i);
      if (t_cw) s_keep.lo ^= 1ULL;
    }
    s = s_keep;
    t = seed_lsb(s);
  }

  u64 final_term = seed_value(s);
  if (t) final_term = final_term + key.g;
  if (party == 1) final_term = static_cast<u64>(0) - final_term;
  return v_share + final_term;
}

} // namespace suf
