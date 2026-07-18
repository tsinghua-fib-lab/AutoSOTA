#pragma once

#include "suf/ir.hpp"

namespace suf {

inline void validate_predicate(const Predicate& p) {
  switch (p.kind) {
    case PredKind::LT:
      break;
    case PredKind::LTLOW:
      ensure(p.f > 0 && p.f <= 64, "LTLOW: bad f");
      if (p.f < 64) {
        ensure(p.gamma < (1ULL << p.f), "LTLOW: gamma out of range");
      }
      break;
    case PredKind::MSB:
      break;
    case PredKind::MSB_ADD:
      break;
    case PredKind::CONST:
      ensure(p.param <= 1, "CONST: param must be 0/1");
      break;
    default:
      fail("unknown predicate kind");
  }
}

inline void validate_bool_expr(const BoolExpr& e, std::size_t num_preds) {
  ensure(e.root >= 0 && static_cast<std::size_t>(e.root) < e.nodes.size(), "BoolExpr: bad root");
  for (std::size_t i = 0; i < e.nodes.size(); ++i) {
    const auto& n = e.nodes[i];
    switch (n.kind) {
      case BoolNode::Kind::PRED:
        ensure(n.pred_index >= 0 && static_cast<std::size_t>(n.pred_index) < num_preds, "BoolExpr: bad pred index");
        break;
      case BoolNode::Kind::NOT:
        ensure(n.lhs >= 0 && static_cast<std::size_t>(n.lhs) < e.nodes.size(), "BoolExpr: bad not lhs");
        break;
      case BoolNode::Kind::AND:
      case BoolNode::Kind::OR:
      case BoolNode::Kind::XOR:
        ensure(n.lhs >= 0 && static_cast<std::size_t>(n.lhs) < e.nodes.size(), "BoolExpr: bad lhs");
        ensure(n.rhs >= 0 && static_cast<std::size_t>(n.rhs) < e.nodes.size(), "BoolExpr: bad rhs");
        break;
      default:
        fail("BoolExpr: unknown node kind");
    }
  }
}

inline void validate_suf(const SUFDescriptor& d) {
  ensure(!d.cuts.empty(), "cuts empty");
  for (std::size_t i = 1; i < d.cuts.size(); ++i) {
    ensure(d.cuts[i] > d.cuts[i-1], "cuts not strictly increasing");
  }
  const std::size_t intervals = d.cuts.size();
  ensure(d.polys.size() == intervals, "polys size mismatch intervals");
  for (const auto& p : d.predicates) validate_predicate(p);
  for (const auto& h : d.helpers) validate_bool_expr(h, d.predicates.size());
}

} // namespace suf
