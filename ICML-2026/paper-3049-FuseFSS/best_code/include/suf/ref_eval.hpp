#pragma once

#include "suf/ir.hpp"

#include <algorithm>
#include <vector>

namespace suf {

inline u64 eval_poly(const Polynomial& poly, u64 x) {
  u64 y = 0;
  for (std::size_t i = poly.coeffs.size(); i-- > 0;) {
    y = y * x + poly.coeffs[i];
  }
  return y;
}

inline u64 eval_predicate(const Predicate& p, u64 x) {
  switch (p.kind) {
    case PredKind::LT:
      return (x + p.input_add) < p.param ? 1ULL : 0ULL;
    case PredKind::LTLOW: {
      const u64 xshift = x + p.input_add;
      if (p.f == 64) {
        return xshift < p.gamma ? 1ULL : 0ULL;
      }
      const u64 mask = (p.f == 64) ? ~0ULL : ((1ULL << p.f) - 1ULL);
      const u64 xlow = xshift & mask;
      return xlow < p.gamma ? 1ULL : 0ULL;
    }
    case PredKind::MSB:
      return (x >> 63) & 1ULL;
    case PredKind::MSB_ADD:
      return ((x + p.param) >> 63) & 1ULL;
    case PredKind::CONST:
      return p.param & 1ULL;
    default:
      return 0ULL;
  }
}

inline u64 eval_bool_expr(const BoolExpr& e, const std::vector<u64>& pred_bits) {
  std::vector<u64> memo(e.nodes.size(), 0);
  for (std::size_t i = 0; i < e.nodes.size(); ++i) {
    const auto& n = e.nodes[i];
    switch (n.kind) {
      case BoolNode::Kind::PRED:
        memo[i] = pred_bits[static_cast<std::size_t>(n.pred_index)] & 1ULL;
        break;
      case BoolNode::Kind::NOT:
        memo[i] = 1ULL ^ (memo[static_cast<std::size_t>(n.lhs)] & 1ULL);
        break;
      case BoolNode::Kind::AND:
        memo[i] = (memo[static_cast<std::size_t>(n.lhs)] & memo[static_cast<std::size_t>(n.rhs)]) & 1ULL;
        break;
      case BoolNode::Kind::OR:
        memo[i] = (memo[static_cast<std::size_t>(n.lhs)] | memo[static_cast<std::size_t>(n.rhs)]) & 1ULL;
        break;
      case BoolNode::Kind::XOR:
        memo[i] = (memo[static_cast<std::size_t>(n.lhs)] ^ memo[static_cast<std::size_t>(n.rhs)]) & 1ULL;
        break;
      default:
        memo[i] = 0ULL;
        break;
    }
  }
  return memo[static_cast<std::size_t>(e.root)] & 1ULL;
}

struct RefEvalResult {
  u64 arith = 0;
  std::vector<u64> helpers; // 0/1
};

inline std::size_t interval_index(const std::vector<u64>& cuts, u64 x) {
  auto it = std::upper_bound(cuts.begin(), cuts.end(), x);
  if (it == cuts.begin()) return 0;
  return static_cast<std::size_t>(std::distance(cuts.begin(), it) - 1);
}

inline RefEvalResult eval_suf_ref(const SUFDescriptor& d, u64 x) {
  std::vector<u64> pred_bits(d.predicates.size());
  for (std::size_t i = 0; i < d.predicates.size(); ++i) {
    pred_bits[i] = eval_predicate(d.predicates[i], x);
  }
  const std::size_t idx = interval_index(d.cuts, x);
  RefEvalResult out;
  out.arith = eval_poly(d.polys[idx], x);
  out.helpers.resize(d.helpers.size());
  for (std::size_t i = 0; i < d.helpers.size(); ++i) {
    out.helpers[i] = eval_bool_expr(d.helpers[i], pred_bits);
  }
  return out;
}

} // namespace suf
