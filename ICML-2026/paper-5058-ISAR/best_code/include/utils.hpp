#pragma once
#include <vector>
#include <cstddef>
#include <cstdint>

class Bitset {
public:
    Bitset(size_t n) : size_(n), bits_((n + 63) / 64, 0) {}
    size_t size() const { return size_; }

    bool operator[](size_t i) const { return bits_[i >> 6] & (1ULL << (i & 63)); }
    void set(size_t i) { bits_[i >> 6] |= (1ULL << (i & 63)); }
    void unset(size_t i) { bits_[i >> 6] &= ~(1ULL << (i & 63)); }
    void flip(size_t i) { bits_[i >> 6] ^= (1ULL << (i & 63)); }
    void clear() { std::fill(bits_.begin(), bits_.end(), 0); }

private:
    size_t size_;
    std::vector<uint64_t> bits_;
};