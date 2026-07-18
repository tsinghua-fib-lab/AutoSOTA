#pragma once

#include "suf/common.hpp"

namespace suf {

struct RotCmp64Recipe {
  u64 theta0 = 0; // r
  u64 theta1 = 0; // r + beta (mod 2^64)
  u8 wrap = 0;    // wrap bit (secret-shared in real protocol)
};

inline RotCmp64Recipe rewrite_lt_u64(u64 r, u64 beta) {
  const u64 theta0 = r;
  const u64 theta1 = r + beta;
  const u8 wrap = (theta1 < theta0) ? 1 : 0;
  return RotCmp64Recipe{theta0, theta1, wrap};
}

struct RotLowRecipe {
  int f = 0;
  u64 theta0 = 0; // r_low
  u64 theta1 = 0; // r_low + gamma (mod 2^f)
  u8 wrap = 0;
};

inline RotLowRecipe rewrite_ltlow(u64 r, int f, u64 gamma) {
  const u64 mask = (f == 64) ? ~0ULL : ((1ULL << f) - 1ULL);
  const u64 rlow = r & mask;
  const u64 theta1 = (rlow + gamma) & mask;
  const u8 wrap = (theta1 < rlow) ? 1 : 0;
  return RotLowRecipe{f, rlow, theta1, wrap};
}

inline RotCmp64Recipe rewrite_msb_add(u64 r, u64 c) {
  // MSB(x + c) is threshold on (x + c) < 2^63.
  // Under mask, we use interval of length 2^63 starting at (r - c).
  const u64 start = r - c;
  return rewrite_lt_u64(start, (1ULL << 63));
}

} // namespace suf
