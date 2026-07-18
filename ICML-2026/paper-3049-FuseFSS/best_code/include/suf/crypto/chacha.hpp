#pragma once

#include "suf/common.hpp"

namespace suf {

#ifdef __CUDACC__
#define SUF_HOST_DEVICE __host__ __device__
#else
#define SUF_HOST_DEVICE
#endif

struct Seed {
  u64 lo = 0;
  u64 hi = 0;
};

SUF_HOST_DEVICE inline Seed seed_xor(const Seed& a, const Seed& b) {
  return Seed{a.lo ^ b.lo, a.hi ^ b.hi};
}

SUF_HOST_DEVICE inline Seed seed_and_mask(const Seed& a, u64 mask_lo, u64 mask_hi) {
  return Seed{a.lo & mask_lo, a.hi & mask_hi};
}

SUF_HOST_DEVICE inline u8 seed_lsb(const Seed& a) {
  return static_cast<u8>(a.lo & 1ULL);
}

SUF_HOST_DEVICE inline u32 rotl32(u32 x, int r) {
  return (x << r) | (x >> (32 - r));
}

SUF_HOST_DEVICE inline void quarter_round(u32& a, u32& b, u32& c, u32& d) {
  a += b; d ^= a; d = rotl32(d, 16);
  c += d; b ^= c; b = rotl32(b, 12);
  a += b; d ^= a; d = rotl32(d, 8);
  c += d; b ^= c; b = rotl32(b, 7);
}

SUF_HOST_DEVICE inline void chacha_block(const Seed& seed, u32 counter, u32 out[16]) {
  // Expand 128-bit seed to 256-bit key by repeating.
  const u32 k0 = static_cast<u32>(seed.lo & 0xffffffffu);
  const u32 k1 = static_cast<u32>((seed.lo >> 32) & 0xffffffffu);
  const u32 k2 = static_cast<u32>(seed.hi & 0xffffffffu);
  const u32 k3 = static_cast<u32>((seed.hi >> 32) & 0xffffffffu);

  const u32 constants[4] = {0x61707865u, 0x3320646eu, 0x79622d32u, 0x6b206574u};

  u32 state[16];
  state[0] = constants[0];
  state[1] = constants[1];
  state[2] = constants[2];
  state[3] = constants[3];
  state[4] = k0;
  state[5] = k1;
  state[6] = k2;
  state[7] = k3;
  state[8] = k0;
  state[9] = k1;
  state[10] = k2;
  state[11] = k3;
  state[12] = counter;
  state[13] = 0;
  state[14] = 0;
  state[15] = 0;

  u32 working[16];
  for (int i = 0; i < 16; ++i) working[i] = state[i];

  // 12 rounds (6 double rounds)
  for (int i = 0; i < 6; ++i) {
    // column rounds
    quarter_round(working[0], working[4], working[8], working[12]);
    quarter_round(working[1], working[5], working[9], working[13]);
    quarter_round(working[2], working[6], working[10], working[14]);
    quarter_round(working[3], working[7], working[11], working[15]);
    // diagonal rounds
    quarter_round(working[0], working[5], working[10], working[15]);
    quarter_round(working[1], working[6], working[11], working[12]);
    quarter_round(working[2], working[7], working[8], working[13]);
    quarter_round(working[3], working[4], working[9], working[14]);
  }

  for (int i = 0; i < 16; ++i) {
    out[i] = working[i] + state[i];
  }
}

SUF_HOST_DEVICE inline Seed seed_from_words(const u32 w[16]) {
  Seed s;
  s.lo = static_cast<u64>(w[0]) | (static_cast<u64>(w[1]) << 32);
  s.hi = static_cast<u64>(w[2]) | (static_cast<u64>(w[3]) << 32);
  return s;
}

SUF_HOST_DEVICE inline void prg_expand(const Seed& seed, Seed& left, u8& tL, Seed& right, u8& tR) {
  u32 block0[16];
  u32 block1[16];
  chacha_block(seed, 0, block0);
  chacha_block(seed, 1, block1);
  Seed sL = seed_from_words(block0);
  Seed sR = seed_from_words(block1);
  tL = seed_lsb(sL);
  tR = seed_lsb(sR);
  // clear LSBs
  sL.lo &= ~1ULL;
  sR.lo &= ~1ULL;
  left = sL;
  right = sR;
}

#undef SUF_HOST_DEVICE

} // namespace suf
