#pragma once

#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace suf {

using u8 = std::uint8_t;
using u16 = std::uint16_t;
using u32 = std::uint32_t;
using u64 = std::uint64_t;
using i64 = std::int64_t;

constexpr int kFracBits = 12;

[[noreturn]] inline void fail(const char* msg) {
  throw std::runtime_error(std::string("suf: ") + msg);
}
inline void ensure(bool ok, const char* msg) {
  if (!ok) fail(msg);
}

template <class T>
inline u32 checked_u32_len(T n) {
  static_assert(std::is_integral_v<T>, "integral only");
  ensure(n >= 0, "negative length");
  ensure(static_cast<std::uint64_t>(n) <= std::numeric_limits<u32>::max(), "length overflow u32");
  return static_cast<u32>(n);
}

inline void store_u32_le(u32 x, u8 out[4]) {
  out[0] = static_cast<u8>(x >> 0);
  out[1] = static_cast<u8>(x >> 8);
  out[2] = static_cast<u8>(x >> 16);
  out[3] = static_cast<u8>(x >> 24);
}

inline u32 load_u32_le(const u8 in[4]) {
  return (u32(in[0]) << 0) | (u32(in[1]) << 8) | (u32(in[2]) << 16) | (u32(in[3]) << 24);
}

inline void store_u64_le(u64 x, u8 out[8]) {
  out[0] = static_cast<u8>(x >> 0);
  out[1] = static_cast<u8>(x >> 8);
  out[2] = static_cast<u8>(x >> 16);
  out[3] = static_cast<u8>(x >> 24);
  out[4] = static_cast<u8>(x >> 32);
  out[5] = static_cast<u8>(x >> 40);
  out[6] = static_cast<u8>(x >> 48);
  out[7] = static_cast<u8>(x >> 56);
}

inline u64 load_u64_le(const u8 in[8]) {
  return (u64(in[0]) << 0) | (u64(in[1]) << 8) | (u64(in[2]) << 16) | (u64(in[3]) << 24) |
         (u64(in[4]) << 32) | (u64(in[5]) << 40) | (u64(in[6]) << 48) | (u64(in[7]) << 56);
}

inline u64 rotl64(u64 x, int r) {
  return (x << r) | (x >> (64 - r));
}

inline u64 rotr64(u64 x, int r) {
  return (x >> r) | (x << (64 - r));
}

} // namespace suf
