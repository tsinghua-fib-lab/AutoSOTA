#pragma once

#include "suf/ir.hpp"

#include <random>
#include <vector>

namespace suf {

struct MaskedGateInstance {
  SUFDescriptor desc;            // masked descriptor (predicates are masked comparison atoms)
  std::vector<u8> const_pred_bits; // XOR shares for PredKind::CONST entries (0 for others)
  int in_bits = 64;
  u64 r_in = 0;
};

MaskedGateInstance compile_masked_gate_instance(const SUFDescriptor& base,
                                                int in_bits,
                                                u64 r_in,
                                                int party,
                                                std::mt19937_64& rng);

} // namespace suf
