#pragma once

#include "suf/ir.hpp"
#include "suf/masked_compile.hpp"
#include "suf/pfss_plan.hpp"
#include "suf/postprocess.hpp"
#include "suf/validate.hpp"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <random>
#include <unordered_map>
#include <vector>

namespace suf {

struct OperatorPiece {
  std::vector<Polynomial> polys;     // arithmetic outputs for this interval
  std::vector<u64> aux_words;         // per-interval constants returned after polynomial coeffs
  std::vector<BoolExpr> bool_outputs; // Boolean outputs for this interval
};

struct OperatorSpecification {
  int in_bits = 64;
  // Interval starts. The first boundary must be 0; the final 2^in_bits
  // sentinel is implicit so the n=64 domain is representable.
  std::vector<u64> boundaries;
  std::vector<OperatorPiece> pieces;
  std::vector<Predicate> predicates;
  PostprocessProgram postprocess;
};

struct IntervalLookupInstance {
  int in_bits = 64;
  std::size_t payload_words = 0;
  std::vector<u64> cutpoints;
  std::vector<std::vector<u64>> payloads;
};

struct PackedComparisonInstance {
  PfssPlan plan;
  std::vector<int> predicate_to_query;
  std::vector<int> query_to_predicate;
  std::vector<u8> const_pred_bits;
};

struct ShapeLeakage {
  std::size_t comparison_queries = 0;
  std::vector<int> comparison_bit_widths;
  std::size_t interval_count = 0;
  std::size_t payload_words = 0;
  std::size_t arithmetic_outputs = 0;
  std::size_t boolean_outputs = 0;
  std::size_t final_arithmetic_outputs = 0;
  std::size_t final_boolean_outputs = 0;
  std::size_t ring_multiplications = 0;
  std::size_t boolean_ands = 0;
  std::size_t b2a_conversions = 0;
  std::size_t a2b_conversions = 0;
  std::size_t kappa_a_words = 0;
  std::size_t kappa_b_bits = 0;
  int max_degree = 0;
};

struct OperatorCompileOptions {
  bool mask_aware = false;
  u64 input_mask = 0;
  int party = 0;
  u64 seed = 0xC0FFEEULL;
};

struct CompiledOperator {
  OperatorSpecification spec;
  SUFDescriptor descriptor;          // normalized descriptor before masking
  MaskedGateInstance masked_instance; // populated when mask_aware=true
  SUFDescriptor backend_descriptor;   // descriptor consumed by predicate backend
  PackedComparisonInstance packed_comparison;
  IntervalLookupInstance interval_lookup;
  ShapeLeakage shape;
};

namespace detail {

inline unsigned __int128 domain_size_u128(int in_bits) {
  ensure(in_bits > 0 && in_bits <= 64, "operator spec: in_bits must be 1..64");
  return static_cast<unsigned __int128>(1) << in_bits;
}

inline u64 mask_for_bits(int in_bits) {
  if (in_bits >= 64) return ~0ULL;
  return (1ULL << in_bits) - 1ULL;
}

inline std::size_t arithmetic_outputs(const OperatorSpecification& spec) {
  if (spec.pieces.empty()) return 0;
  return spec.pieces.front().polys.size();
}

inline std::size_t boolean_outputs(const OperatorSpecification& spec) {
  if (spec.pieces.empty()) return 0;
  return spec.pieces.front().bool_outputs.size();
}

inline std::size_t aux_words(const OperatorSpecification& spec) {
  if (spec.pieces.empty()) return 0;
  return spec.pieces.front().aux_words.size();
}

inline std::size_t final_arithmetic_outputs(const OperatorSpecification& spec) {
  return spec.postprocess.arithmetic_outputs.empty()
             ? arithmetic_outputs(spec)
             : spec.postprocess.arithmetic_outputs.size();
}

inline std::size_t final_boolean_outputs(const OperatorSpecification& spec) {
  return spec.postprocess.boolean_outputs.empty()
             ? boolean_outputs(spec)
             : spec.postprocess.boolean_outputs.size();
}

inline int max_degree(const OperatorSpecification& spec) {
  int degree = 0;
  for (const auto& piece : spec.pieces) {
    for (const auto& poly : piece.polys) {
      if (!poly.coeffs.empty()) {
        degree = std::max<int>(degree, static_cast<int>(poly.coeffs.size()) - 1);
      }
    }
  }
  return degree;
}

inline u64 mul_mod64(u64 a, u64 b) {
  return static_cast<u64>(static_cast<unsigned __int128>(a) *
                          static_cast<unsigned __int128>(b));
}

inline std::vector<std::vector<u64>> build_binom(int degree) {
  std::vector<std::vector<u64>> binom(static_cast<std::size_t>(degree + 1),
                                      std::vector<u64>(static_cast<std::size_t>(degree + 1), 0));
  binom[0][0] = 1;
  for (int k = 1; k <= degree; ++k) {
    binom[k][0] = 1;
    binom[k][k] = 1;
    for (int i = 1; i < k; ++i) {
      binom[k][i] = binom[k - 1][i - 1] + binom[k - 1][i];
    }
  }
  return binom;
}

inline std::vector<u64> build_pow_offset(int degree, u64 offset) {
  std::vector<u64> pow(static_cast<std::size_t>(degree + 1), 0);
  pow[0] = 1;
  for (int i = 1; i <= degree; ++i) {
    pow[static_cast<std::size_t>(i)] =
        mul_mod64(pow[static_cast<std::size_t>(i - 1)], offset);
  }
  return pow;
}

inline std::vector<u64> shift_poly_coeffs(const std::vector<u64>& coeffs,
                                          int degree,
                                          const std::vector<std::vector<u64>>& binom,
                                          const std::vector<u64>& pow_neg_r) {
  std::vector<u64> out(static_cast<std::size_t>(degree + 1), 0);
  const int max_k = std::min<int>(degree, static_cast<int>(coeffs.size()) - 1);
  for (int k = 0; k <= max_k; ++k) {
    const u64 c = coeffs[static_cast<std::size_t>(k)];
    if (c == 0) continue;
    for (int i = 0; i <= k; ++i) {
      const u64 term0 = mul_mod64(c, binom[k][i]);
      const u64 term = mul_mod64(term0, pow_neg_r[static_cast<std::size_t>(k - i)]);
      out[static_cast<std::size_t>(i)] += term;
    }
  }
  return out;
}

inline u64 sub_mod_bits(u64 a, u64 b, int bits) {
  if (bits >= 64) return a - b;
  const u64 modulus = 1ULL << bits;
  return (a + modulus - (b & (modulus - 1ULL))) & (modulus - 1ULL);
}

inline std::size_t interval_index_for(const std::vector<u64>& cuts, u64 x) {
  auto it = std::upper_bound(cuts.begin(), cuts.end(), x);
  if (it == cuts.begin()) return 0;
  return static_cast<std::size_t>(std::distance(cuts.begin(), it) - 1);
}

inline int clone_bool_node(BoolExpr& dst,
                           const BoolExpr& src,
                           int src_idx,
                           int pred_offset,
                           std::vector<int>& memo) {
  ensure(src_idx >= 0 && static_cast<std::size_t>(src_idx) < src.nodes.size(),
         "operator spec: bad BoolExpr clone index");
  auto& cached = memo[static_cast<std::size_t>(src_idx)];
  if (cached >= 0) return cached;
  const auto& in = src.nodes[static_cast<std::size_t>(src_idx)];
  BoolNode out;
  out.kind = in.kind;
  switch (in.kind) {
    case BoolNode::Kind::PRED:
      out.pred_index = in.pred_index + pred_offset;
      break;
    case BoolNode::Kind::NOT:
      out.lhs = clone_bool_node(dst, src, in.lhs, pred_offset, memo);
      break;
    case BoolNode::Kind::AND:
    case BoolNode::Kind::OR:
    case BoolNode::Kind::XOR:
      out.lhs = clone_bool_node(dst, src, in.lhs, pred_offset, memo);
      out.rhs = clone_bool_node(dst, src, in.rhs, pred_offset, memo);
      break;
    default:
      fail("operator spec: unknown BoolExpr node");
  }
  dst.nodes.push_back(out);
  cached = static_cast<int>(dst.nodes.size() - 1);
  return cached;
}

inline int append_bool_expr(BoolExpr& dst, const BoolExpr& src, int pred_offset = 0) {
  ensure(src.root >= 0, "operator spec: empty BoolExpr");
  std::vector<int> memo(src.nodes.size(), -1);
  return clone_bool_node(dst, src, src.root, pred_offset, memo);
}

inline BoolExpr pred_expr(int pred_index) {
  BoolExpr e;
  BoolNode n;
  n.kind = BoolNode::Kind::PRED;
  n.pred_index = pred_index;
  e.nodes.push_back(n);
  e.root = 0;
  return e;
}

inline BoolExpr not_expr(const BoolExpr& a) {
  BoolExpr out;
  const int lhs = append_bool_expr(out, a);
  BoolNode n;
  n.kind = BoolNode::Kind::NOT;
  n.lhs = lhs;
  out.nodes.push_back(n);
  out.root = static_cast<int>(out.nodes.size() - 1);
  return out;
}

inline BoolExpr binary_expr(BoolNode::Kind kind, const BoolExpr& a, const BoolExpr& b) {
  BoolExpr out;
  const int lhs = append_bool_expr(out, a);
  const int rhs = append_bool_expr(out, b);
  BoolNode n;
  n.kind = kind;
  n.lhs = lhs;
  n.rhs = rhs;
  out.nodes.push_back(n);
  out.root = static_cast<int>(out.nodes.size() - 1);
  return out;
}

inline BoolExpr xor_expr(const BoolExpr& a, const BoolExpr& b) {
  return binary_expr(BoolNode::Kind::XOR, a, b);
}

inline BoolExpr and_expr(const BoolExpr& a, const BoolExpr& b) {
  return binary_expr(BoolNode::Kind::AND, a, b);
}

inline BoolExpr const_expr(SUFDescriptor& d, u8 bit) {
  Predicate p;
  p.kind = PredKind::CONST;
  p.param = bit & 1u;
  d.predicates.push_back(p);
  return pred_expr(static_cast<int>(d.predicates.size() - 1));
}

inline BoolExpr comparison_expr(SUFDescriptor& d, u64 boundary) {
  Predicate p;
  p.kind = PredKind::LT;
  p.param = boundary;
  d.predicates.push_back(p);
  return pred_expr(static_cast<int>(d.predicates.size() - 1));
}

inline std::vector<u64> build_payload_row(const OperatorSpecification& spec,
                                          std::size_t piece_idx,
                                          int degree,
                                          const std::vector<std::vector<u64>>& binom,
                                          u64 x_offset) {
  const auto r = arithmetic_outputs(spec);
  const std::size_t coeff_words = r * static_cast<std::size_t>(degree + 1);
  const std::size_t words = coeff_words + aux_words(spec);
  std::vector<u64> row(words, 0);
  const auto pow_offset = build_pow_offset(degree, x_offset);
  for (std::size_t out = 0; out < r; ++out) {
    const auto& coeffs = spec.pieces[piece_idx].polys[out].coeffs;
    const auto shifted = shift_poly_coeffs(coeffs, degree, binom, pow_offset);
    for (int k = 0; k <= degree; ++k) {
      const std::size_t pos = out * static_cast<std::size_t>(degree + 1) + static_cast<std::size_t>(k);
      row[pos] = shifted[static_cast<std::size_t>(k)];
    }
  }
  const auto& aux = spec.pieces[piece_idx].aux_words;
  for (std::size_t i = 0; i < aux.size(); ++i) {
    row[coeff_words + i] = aux[i];
  }
  return row;
}

inline std::vector<std::vector<u64>> build_payloads(const OperatorSpecification& spec,
                                                    int degree) {
  const auto r = arithmetic_outputs(spec);
  const std::size_t words = r * static_cast<std::size_t>(degree + 1) + aux_words(spec);
  std::vector<std::vector<u64>> payloads(spec.pieces.size(), std::vector<u64>(words, 0));
  const auto binom = build_binom(degree);
  for (std::size_t i = 0; i < spec.pieces.size(); ++i) {
    payloads[i] = build_payload_row(spec, i, degree, binom, 0);
  }
  return payloads;
}

inline unsigned __int128 interval_length(u64 start, u64 end, int in_bits) {
  const auto domain = domain_size_u128(in_bits);
  const unsigned __int128 s = start;
  const unsigned __int128 e = (end == 0) ? domain : static_cast<unsigned __int128>(end);
  return e >= s ? (e - s) : 0;
}

inline IntervalLookupInstance translate_payload_lookup(const OperatorSpecification& spec,
                                                       int degree,
                                                       u64 input_mask) {
  const int in_bits = spec.in_bits;
  const std::size_t m = spec.boundaries.size();
  const u64 mask = mask_for_bits(in_bits);
  const auto domain = domain_size_u128(in_bits);
  const unsigned __int128 target_intervals =
      (static_cast<unsigned __int128>(m) < domain) ? static_cast<unsigned __int128>(m + 1)
                                                   : static_cast<unsigned __int128>(m);

  struct Start {
    u64 value = 0;
    std::size_t index = 0;
  };
  std::vector<Start> starts;
  starts.reserve(m);
  for (std::size_t i = 0; i < m; ++i) {
    starts.push_back(Start{(spec.boundaries[i] + input_mask) & mask, i});
  }
  std::sort(starts.begin(), starts.end(), [](const Start& a, const Start& b) {
    return a.value < b.value;
  });

  IntervalLookupInstance out;
  out.in_bits = in_bits;
  out.payload_words = arithmetic_outputs(spec) * static_cast<std::size_t>(degree + 1) +
                      aux_words(spec);

  if (!starts.empty() && starts.front().value == 0) {
    out.cutpoints.reserve(static_cast<std::size_t>(target_intervals));
    for (const auto& s : starts) {
      out.cutpoints.push_back(s.value);
    }
    if (static_cast<unsigned __int128>(out.cutpoints.size()) < target_intervals) {
      for (std::size_t i = 0; i < out.cutpoints.size(); ++i) {
        const u64 start = out.cutpoints[i];
        const u64 end = (i + 1 < out.cutpoints.size()) ? out.cutpoints[i + 1] : 0ULL;
        if (interval_length(start, end, in_bits) > 1) {
          const unsigned __int128 split128 = static_cast<unsigned __int128>(start) + 1;
          const u64 split = (in_bits == 64) ? static_cast<u64>(split128)
                                            : static_cast<u64>(split128) & mask;
          out.cutpoints.insert(out.cutpoints.begin() + static_cast<std::ptrdiff_t>(i + 1), split);
          break;
        }
      }
    }
  } else {
    out.cutpoints.reserve(m + 1);
    out.cutpoints.push_back(0);
    for (const auto& s : starts) {
      out.cutpoints.push_back(s.value);
    }
  }

  const auto binom = build_binom(degree);
  out.payloads.reserve(out.cutpoints.size());
  for (const u64 cut : out.cutpoints) {
    const u64 plain = sub_mod_bits(cut, input_mask, in_bits);
    const auto piece = interval_index_for(spec.boundaries, plain);
    const u64 x_offset = plain - cut;
    out.payloads.push_back(build_payload_row(spec, piece, degree, binom, x_offset));
  }

  ensure(out.cutpoints.size() == out.payloads.size(), "operator spec: lookup shape mismatch");
  ensure(static_cast<unsigned __int128>(out.cutpoints.size()) == target_intervals,
         "operator spec: lookup padding failed");
  return out;
}

} // namespace detail

inline void validate_operator_spec(const OperatorSpecification& spec) {
  ensure(spec.in_bits > 0 && spec.in_bits <= 64, "operator spec: in_bits must be 1..64");
  ensure(!spec.boundaries.empty(), "operator spec: empty boundaries");
  ensure(spec.boundaries.front() == 0, "operator spec: first boundary must be 0");
  ensure(spec.pieces.size() == spec.boundaries.size(), "operator spec: pieces/boundaries mismatch");
  const auto domain = detail::domain_size_u128(spec.in_bits);
  for (std::size_t i = 0; i < spec.boundaries.size(); ++i) {
    ensure(static_cast<unsigned __int128>(spec.boundaries[i]) < domain,
           "operator spec: boundary outside domain");
    if (i > 0) ensure(spec.boundaries[i] > spec.boundaries[i - 1],
                      "operator spec: boundaries not strictly increasing");
  }

  const auto r = detail::arithmetic_outputs(spec);
  const auto ell = detail::boolean_outputs(spec);
  const auto aux = detail::aux_words(spec);
  for (const auto& p : spec.predicates) validate_predicate(p);
  for (const auto& piece : spec.pieces) {
    ensure(piece.polys.size() == r, "operator spec: inconsistent arithmetic output count");
    ensure(piece.aux_words.size() == aux, "operator spec: inconsistent aux payload word count");
    ensure(piece.bool_outputs.size() == ell, "operator spec: inconsistent boolean output count");
    for (const auto& b : piece.bool_outputs) validate_bool_expr(b, spec.predicates.size());
  }
  validate_postprocess_program(spec.postprocess);
}

inline SUFDescriptor lower_operator_spec_to_suf(const OperatorSpecification& spec,
                                                std::size_t arithmetic_output = 0) {
  validate_operator_spec(spec);
  const auto r = detail::arithmetic_outputs(spec);
  if (r > 0) {
    ensure(arithmetic_output < r, "operator spec: arithmetic output index out of range");
  }

  SUFDescriptor d;
  d.cuts = spec.boundaries;
  d.predicates = spec.predicates;
  d.polys.resize(spec.pieces.size());
  for (std::size_t i = 0; i < spec.pieces.size(); ++i) {
    if (r == 0) {
      d.polys[i].coeffs = {0};
    } else {
      d.polys[i] = spec.pieces[i].polys[arithmetic_output];
    }
  }

  const auto ell = detail::boolean_outputs(spec);
  if (ell == 0) {
    validate_suf(d);
    return d;
  }

  const BoolExpr const0 = detail::const_expr(d, 0);
  const BoolExpr const1 = detail::const_expr(d, 1);
  std::unordered_map<u64, BoolExpr> boundary_cache;
  auto boundary_expr = [&](std::size_t boundary_idx) -> BoolExpr {
    if (boundary_idx == 0) return const0;
    if (boundary_idx == spec.boundaries.size()) return const1;
    const u64 b = spec.boundaries[boundary_idx];
    auto it = boundary_cache.find(b);
    if (it != boundary_cache.end()) return it->second;
    auto expr = detail::comparison_expr(d, b);
    boundary_cache.emplace(b, expr);
    return expr;
  };

  for (std::size_t out = 0; out < ell; ++out) {
    BoolExpr merged = const0;
    for (std::size_t i = 0; i < spec.pieces.size(); ++i) {
      const BoolExpr lo = boundary_expr(i);
      const BoolExpr hi = boundary_expr(i + 1);
      const BoolExpr indicator = detail::xor_expr(hi, lo);
      const BoolExpr gated = detail::and_expr(indicator, spec.pieces[i].bool_outputs[out]);
      merged = detail::xor_expr(merged, gated);
    }
    d.helpers.push_back(std::move(merged));
  }

  validate_suf(d);
  return d;
}

inline PackedComparisonInstance compile_packed_comparison_instance(const SUFDescriptor& d,
                                                                   int in_bits = 64) {
  PackedComparisonInstance out;
  out.plan = compile_pfss_plan(d, in_bits);
  out.predicate_to_query.reserve(d.predicates.size());
  out.const_pred_bits.reserve(d.predicates.size());
  int q = 0;
  for (std::size_t i = 0; i < d.predicates.size(); ++i) {
    const auto& p = d.predicates[i];
    if (p.kind == PredKind::CONST) {
      out.predicate_to_query.push_back(-1);
      out.const_pred_bits.push_back(static_cast<u8>(p.param & 1ULL));
    } else {
      out.predicate_to_query.push_back(q++);
      out.query_to_predicate.push_back(static_cast<int>(i));
      out.const_pred_bits.push_back(0);
    }
  }
  return out;
}

inline std::size_t count_helper_boolean_ands(const SUFDescriptor& d) {
  std::size_t out = 0;
  for (const auto& helper : d.helpers) {
    for (const auto& node : helper.nodes) {
      if (node.kind == BoolNode::Kind::AND || node.kind == BoolNode::Kind::OR) {
        ++out;
      }
    }
  }
  return out;
}

inline CompiledOperator compile_operator_spec(const OperatorSpecification& spec,
                                              const OperatorCompileOptions& opts = {}) {
  validate_operator_spec(spec);

  CompiledOperator out;
  out.spec = spec;
  out.descriptor = lower_operator_spec_to_suf(spec, 0);
  out.backend_descriptor = out.descriptor;

  if (opts.mask_aware) {
    std::mt19937_64 rng(opts.seed);
    out.masked_instance = compile_masked_gate_instance(out.descriptor,
                                                       spec.in_bits,
                                                       opts.input_mask,
                                                       opts.party,
                                                       rng);
    out.backend_descriptor = out.masked_instance.desc;
  }

  out.packed_comparison = compile_packed_comparison_instance(out.backend_descriptor,
                                                            spec.in_bits);
  if (opts.mask_aware && !out.masked_instance.const_pred_bits.empty()) {
    ensure(out.masked_instance.const_pred_bits.size() ==
               out.packed_comparison.const_pred_bits.size(),
           "operator spec: masked const bits shape mismatch");
    out.packed_comparison.const_pred_bits = out.masked_instance.const_pred_bits;
  }

  const int degree = detail::max_degree(spec);
  const u64 input_mask = opts.input_mask & detail::mask_for_bits(spec.in_bits);
  const auto payloads = detail::build_payloads(spec, degree);
  if (opts.mask_aware) {
    out.interval_lookup = detail::translate_payload_lookup(spec, degree, input_mask);
  } else {
    out.interval_lookup.in_bits = spec.in_bits;
    out.interval_lookup.payload_words = payloads.empty() ? 0 : payloads.front().size();
    out.interval_lookup.cutpoints = spec.boundaries;
    out.interval_lookup.payloads = payloads;
  }

  out.shape.comparison_queries = out.packed_comparison.plan.queries.size();
  out.shape.comparison_bit_widths.reserve(out.packed_comparison.plan.queries.size());
  for (const auto& q : out.packed_comparison.plan.queries) {
    out.shape.comparison_bit_widths.push_back(q.n_bits);
  }
  out.shape.interval_count = out.interval_lookup.cutpoints.size();
  out.shape.payload_words = out.interval_lookup.payload_words;
  out.shape.arithmetic_outputs = detail::arithmetic_outputs(spec);
  out.shape.boolean_outputs = detail::boolean_outputs(spec);
  out.shape.final_arithmetic_outputs = detail::final_arithmetic_outputs(spec);
  out.shape.final_boolean_outputs = detail::final_boolean_outputs(spec);
  out.shape.max_degree = degree;
  const auto phi_cost = count_postprocess_cost(spec.postprocess);
  const auto kappa_shape = required_postprocess_kappa_shape(spec.postprocess);
  out.shape.ring_multiplications =
      out.shape.arithmetic_outputs * static_cast<std::size_t>(degree) +
      phi_cost.ring_multiplications;
  out.shape.boolean_ands = count_helper_boolean_ands(out.descriptor) + phi_cost.boolean_ands;
  out.shape.b2a_conversions = phi_cost.b2a_conversions;
  out.shape.a2b_conversions = phi_cost.a2b_conversions;
  out.shape.kappa_a_words = kappa_shape.arithmetic;
  out.shape.kappa_b_bits = kappa_shape.boolean;
  return out;
}

inline std::size_t operator_spec_arithmetic_outputs(const OperatorSpecification& spec) {
  validate_operator_spec(spec);
  return detail::arithmetic_outputs(spec);
}

inline std::size_t operator_spec_boolean_outputs(const OperatorSpecification& spec) {
  validate_operator_spec(spec);
  return detail::boolean_outputs(spec);
}

inline std::size_t operator_spec_final_arithmetic_outputs(const OperatorSpecification& spec) {
  validate_operator_spec(spec);
  return detail::final_arithmetic_outputs(spec);
}

inline std::size_t operator_spec_final_boolean_outputs(const OperatorSpecification& spec) {
  validate_operator_spec(spec);
  return detail::final_boolean_outputs(spec);
}

inline std::size_t operator_spec_aux_words(const OperatorSpecification& spec) {
  validate_operator_spec(spec);
  return detail::aux_words(spec);
}

} // namespace suf
