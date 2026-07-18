#pragma once

#include "suf/common.hpp"

#include <vector>
#include <cstddef>

namespace suf {

enum class PredKind : u8 {
  LT = 0,     // x < beta (unsigned)
  LTLOW = 1,  // (x mod 2^f) < gamma
  MSB = 2,    // msb(x)
  MSB_ADD = 3, // msb(x + c)
  CONST = 4   // constant bit (secret-shared in secure mode)
};

struct Predicate {
  PredKind kind = PredKind::LT;
  u64 param = 0;   // beta for LT, c for MSB_ADD, unused for MSB
  u64 gamma = 0;   // gamma for LTLOW
  u8 f = 0;        // f for LTLOW
  u64 input_add = 0; // added to x before comparison (for view_{k,c})
};

struct BoolNode {
  enum class Kind : u8 { PRED, NOT, AND, OR, XOR } kind = Kind::PRED;
  int lhs = -1;
  int rhs = -1;
  int pred_index = -1; // for PRED
};

struct BoolExpr {
  std::vector<BoolNode> nodes;
  int root = -1; // index into nodes
};

struct Polynomial {
  // coeffs[0] + coeffs[1]*x + ... (mod 2^64)
  std::vector<u64> coeffs;
};

struct SUFDescriptor {
  // Interval boundaries in unsigned order. If cuts.size()==m, then intervals are:
  // [cuts[i], cuts[i+1]) for i=0..m-2 and [cuts[m-1], 2^64) for last.
  std::vector<u64> cuts;
  std::vector<Polynomial> polys; // size = number of intervals
  std::vector<Predicate> predicates;
  std::vector<BoolExpr> helpers; // boolean outputs
};

} // namespace suf
