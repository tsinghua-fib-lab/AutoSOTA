#include "suf/operator_spec.hpp"
#include "suf/ref_eval.hpp"

#include <algorithm>
#include <cassert>
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace suf;

namespace {

struct LsbOnlyRuntime final : MpcRuntime {
  u64 mul(u64 a, u64 b) const override { return a * b; }
  u8 bool_and(u8 a, u8 b) const override { return static_cast<u8>((a & b) & 1U); }
  u64 b2a(u8 b) const override { return static_cast<u64>(b & 1U); }
  u8 a2b(u64 x) const override { return static_cast<u8>(x & 1ULL); }
};

BoolExpr pred_expr_for_test(int pred_index) {
  BoolExpr e;
  BoolNode n;
  n.kind = BoolNode::Kind::PRED;
  n.pred_index = pred_index;
  e.nodes.push_back(n);
  e.root = 0;
  return e;
}

BoolExpr not_pred_expr_for_test(int pred_index) {
  BoolExpr e;
  BoolNode p;
  p.kind = BoolNode::Kind::PRED;
  p.pred_index = pred_index;
  e.nodes.push_back(p);

  BoolNode n;
  n.kind = BoolNode::Kind::NOT;
  n.lhs = 0;
  e.nodes.push_back(n);
  e.root = 1;
  return e;
}

OperatorSpecification make_spec() {
  OperatorSpecification spec;
  spec.in_bits = 8;
  spec.boundaries = {0, 64, 128};

  Predicate low;
  low.kind = PredKind::LTLOW;
  low.f = 4;
  low.gamma = 8;
  spec.predicates.push_back(low);

  Predicate one;
  one.kind = PredKind::CONST;
  one.param = 1;
  spec.predicates.push_back(one);

  spec.pieces.resize(3);
  spec.pieces[0].polys = {Polynomial{{1}}, Polynomial{{10, 1}}};
  spec.pieces[1].polys = {Polynomial{{2}}, Polynomial{{20, 2}}};
  spec.pieces[2].polys = {Polynomial{{3}}, Polynomial{{30, 3}}};

  spec.pieces[0].bool_outputs = {pred_expr_for_test(0)};
  spec.pieces[1].bool_outputs = {not_pred_expr_for_test(0)};
  spec.pieces[2].bool_outputs = {pred_expr_for_test(1)};
  return spec;
}

OperatorSpecification make_relu_phi_spec() {
  OperatorSpecification spec;
  spec.in_bits = 8;
  spec.boundaries = {0, 128};

  Predicate msb;
  msb.kind = PredKind::MSB;
  spec.predicates.push_back(msb);

  spec.pieces.resize(2);
  spec.pieces[0].polys = {Polynomial{{1}}, Polynomial{{3}}}; // a=1,b=3
  spec.pieces[0].aux_words = {11, 12};
  spec.pieces[0].bool_outputs = {pred_expr_for_test(0)};
  spec.pieces[1].polys = {Polynomial{{0}}, Polynomial{{5}}}; // a=0,b=5
  spec.pieces[1].aux_words = {21, 22};
  spec.pieces[1].bool_outputs = {pred_expr_for_test(0)};

  PostprocessArithExpr y;
  y.nodes.push_back(PostprocessArithNode{PostprocessArithOp::POLY_OUT, -1, -1, 0, 0});
  y.nodes.push_back(PostprocessArithNode{PostprocessArithOp::X, -1, -1, -1, 0});
  y.nodes.push_back(PostprocessArithNode{PostprocessArithOp::MUL, 0, 1, -1, 0});
  y.nodes.push_back(PostprocessArithNode{PostprocessArithOp::POLY_OUT, -1, -1, 1, 0});
  y.nodes.push_back(PostprocessArithNode{PostprocessArithOp::ADD, 2, 3, -1, 0});
  y.root = 4;
  spec.postprocess.arith_exprs.push_back(y);
  spec.postprocess.arithmetic_outputs = {0};

  PostprocessBoolExpr z;
  z.nodes.push_back(PostprocessBoolNode{PostprocessBoolOp::BOOL_OUT, -1, -1, 0, 0});
  z.root = 0;
  spec.postprocess.bool_exprs.push_back(z);
  spec.postprocess.boolean_outputs = {0};
  return spec;
}

OperatorSpecification make_kappa_phi_spec() {
  OperatorSpecification spec;
  spec.in_bits = 8;
  spec.boundaries = {0};
  spec.pieces.resize(1);
  spec.pieces[0].polys = {Polynomial{{9}}};

  PostprocessArithExpr y;
  y.nodes.push_back(PostprocessArithNode{PostprocessArithOp::POLY_OUT, -1, -1, 0, 0});
  y.nodes.push_back(PostprocessArithNode{PostprocessArithOp::KAPPA_A, -1, -1, 1, 0});
  y.nodes.push_back(PostprocessArithNode{PostprocessArithOp::ADD, 0, 1, -1, 0});
  y.root = 2;
  spec.postprocess.arith_exprs.push_back(y);
  spec.postprocess.arithmetic_outputs = {0};

  PostprocessBoolExpr z;
  z.nodes.push_back(PostprocessBoolNode{PostprocessBoolOp::KAPPA_B, -1, -1, 2, 0});
  z.root = 0;
  spec.postprocess.bool_exprs.push_back(z);
  spec.postprocess.boolean_outputs = {0};
  return spec;
}

std::vector<u64> expected_payload(const OperatorSpecification& spec,
                                  std::size_t piece,
                                  int degree) {
  const std::size_t outputs = spec.pieces.front().polys.size();
  const std::size_t coeff_words = outputs * static_cast<std::size_t>(degree + 1);
  std::vector<u64> row(coeff_words + spec.pieces.front().aux_words.size(), 0);
  for (std::size_t out = 0; out < outputs; ++out) {
    const auto& coeffs = spec.pieces[piece].polys[out].coeffs;
    for (int k = 0; k <= degree; ++k) {
      row[out * static_cast<std::size_t>(degree + 1) + static_cast<std::size_t>(k)] =
          (static_cast<std::size_t>(k) < coeffs.size()) ? coeffs[static_cast<std::size_t>(k)] : 0;
    }
  }
  for (std::size_t i = 0; i < spec.pieces[piece].aux_words.size(); ++i) {
    row[coeff_words + i] = spec.pieces[piece].aux_words[i];
  }
  return row;
}

std::size_t lookup_interval(const std::vector<u64>& cutpoints, u64 x) {
  auto it = std::upper_bound(cutpoints.begin(), cutpoints.end(), x);
  if (it == cutpoints.begin()) return 0;
  return static_cast<std::size_t>(std::distance(cutpoints.begin(), it) - 1);
}

u64 eval_payload_poly(const std::vector<u64>& payload,
                      std::size_t output,
                      int degree,
                      u64 x) {
  const std::size_t base = output * static_cast<std::size_t>(degree + 1);
  u64 y = 0;
  for (int k = degree; k >= 0; --k) {
    y = y * x + payload[base + static_cast<std::size_t>(k)];
  }
  return y;
}

u64 expected_helper(u64 x) {
  if (x < 64) return ((x & 0xFULL) < 8) ? 1ULL : 0ULL;
  if (x < 128) return ((x & 0xFULL) < 8) ? 0ULL : 1ULL;
  return 1ULL;
}

} // namespace

int main() {
  const auto spec = make_spec();

  const auto d = lower_operator_spec_to_suf(spec, 0);
  assert(d.cuts == spec.boundaries);
  assert(d.helpers.size() == 1);
  for (int x = 0; x < 256; ++x) {
    const auto res = eval_suf_ref(d, static_cast<u64>(x));
    if (res.helpers[0] != expected_helper(static_cast<u64>(x))) {
      std::cerr << "lowered helper mismatch at " << x << " got " << res.helpers[0]
                << " expected " << expected_helper(static_cast<u64>(x)) << "\n";
      return 1;
    }
  }

  const auto unmasked = compile_operator_spec(spec);
  assert(unmasked.shape.arithmetic_outputs == 2);
  assert(unmasked.shape.boolean_outputs == 1);
  assert(unmasked.shape.max_degree == 1);
  assert(unmasked.shape.payload_words == 4);
  assert(unmasked.shape.interval_count == 3);
  assert(unmasked.packed_comparison.query_to_predicate.size() ==
         unmasked.packed_comparison.plan.queries.size());
  for (const auto& q : unmasked.packed_comparison.plan.queries) {
    assert(q.n_bits == 8 || q.n_bits == 4);
  }
  for (std::size_t i = 0; i < spec.pieces.size(); ++i) {
    assert(unmasked.interval_lookup.payloads[i] == expected_payload(spec, i, 1));
  }

  OperatorCompileOptions opts;
  opts.mask_aware = true;
  opts.input_mask = 37;
  opts.party = 0;
  opts.seed = 1234;
  const auto masked = compile_operator_spec(spec, opts);
  assert(masked.shape.interval_count == 4);
  assert(masked.shape.payload_words == 4);
  assert(masked.packed_comparison.query_to_predicate.size() ==
         masked.packed_comparison.plan.queries.size());
  assert(masked.packed_comparison.const_pred_bits.size() ==
         masked.backend_descriptor.predicates.size());
  bool saw_low_width = false;
  bool saw_input_width = false;
  for (const auto& q : masked.packed_comparison.plan.queries) {
    assert(q.n_bits == 8 || q.n_bits == 4);
    saw_low_width = saw_low_width || (q.n_bits == 4);
    saw_input_width = saw_input_width || (q.n_bits == 8);
  }
  assert(saw_low_width);
  assert(saw_input_width);

  for (int x = 0; x < 256; ++x) {
    const u64 x_plain = static_cast<u64>(x);
    const u64 x_masked = (x_plain + opts.input_mask) & 0xFFULL;
    const auto piece = interval_index(spec.boundaries, x_plain);
    const auto row = lookup_interval(masked.interval_lookup.cutpoints, x_masked);
    const auto& payload = masked.interval_lookup.payloads[row];
    for (std::size_t out = 0; out < spec.pieces[piece].polys.size(); ++out) {
      const u64 got = eval_payload_poly(payload, out, 1, x_masked);
      const u64 expected = eval_poly(spec.pieces[piece].polys[out], x_plain);
      if (got != expected) {
        std::cerr << "masked payload eval mismatch at x=" << x << " x_hat=" << x_masked
                  << " row=" << row << " piece=" << piece << " out=" << out
                  << " got=" << got << " expected=" << expected << "\n";
        return 1;
      }
    }
  }

  const auto relu = make_relu_phi_spec();
  const auto relu_compiled = compile_operator_spec(relu);
  assert(relu_compiled.shape.arithmetic_outputs == 2);
  assert(relu_compiled.shape.final_arithmetic_outputs == 1);
  assert(relu_compiled.shape.boolean_outputs == 1);
  assert(relu_compiled.shape.final_boolean_outputs == 1);
  assert(relu_compiled.shape.payload_words == 4);
  assert(relu_compiled.shape.ring_multiplications == 1);
  assert(relu_compiled.shape.b2a_conversions == 0);
  assert(relu_compiled.interval_lookup.payloads[0] == expected_payload(relu, 0, 0));
  assert(relu_compiled.interval_lookup.payloads[1] == expected_payload(relu, 1, 0));

  OperatorCompileOptions relu_opts_a;
  relu_opts_a.mask_aware = true;
  relu_opts_a.input_mask = 37;
  relu_opts_a.party = 0;
  relu_opts_a.seed = 88;
  OperatorCompileOptions relu_opts_b = relu_opts_a;
  relu_opts_b.input_mask = 91;
  const auto relu_mask_a = compile_operator_spec(relu, relu_opts_a);
  const auto relu_mask_b = compile_operator_spec(relu, relu_opts_b);
  assert(relu_mask_a.shape.comparison_queries == relu_mask_b.shape.comparison_queries);
  assert(relu_mask_a.shape.comparison_bit_widths == relu_mask_b.shape.comparison_bit_widths);
  assert(relu_mask_a.shape.interval_count == relu_mask_b.shape.interval_count);
  assert(relu_mask_a.shape.payload_words == 4);
  assert(relu_mask_a.shape.payload_words == relu_mask_b.shape.payload_words);

  const auto kappa = make_kappa_phi_spec();
  const auto kappa_shape = required_postprocess_kappa_shape(kappa.postprocess);
  assert(kappa_shape.arithmetic == 2);
  assert(kappa_shape.boolean == 3);
  const auto kappa_compiled = compile_operator_spec(kappa);
  assert(kappa_compiled.shape.kappa_a_words == 2);
  assert(kappa_compiled.shape.kappa_b_bits == 3);

  PostprocessProgram unused_program;
  PostprocessArithExpr unused_kappa;
  unused_kappa.nodes.push_back(PostprocessArithNode{PostprocessArithOp::KAPPA_A, -1, -1, 7, 0});
  unused_kappa.root = 0;
  unused_program.arith_exprs.push_back(unused_kappa);
  PostprocessBoolExpr unused_a2b;
  PostprocessBoolNode unused_a2b_node;
  unused_a2b_node.op = PostprocessBoolOp::A2B;
  unused_a2b_node.index = 0;
  unused_a2b_node.bit_index = 3;
  unused_a2b.nodes.push_back(unused_a2b_node);
  unused_a2b.root = 0;
  unused_program.bool_exprs.push_back(unused_a2b);
  validate_postprocess_program(unused_program);
  const auto unused_cost = count_postprocess_cost(unused_program);
  assert(!unused_cost.requires_runtime());
  assert(required_postprocess_kappa_shape(unused_program).empty());
  assert(max_postprocess_a2b_bit_index(unused_program) == -1);

  PostprocessProgram bit_program;
  PostprocessArithExpr x_expr;
  x_expr.nodes.push_back(PostprocessArithNode{PostprocessArithOp::X, -1, -1, -1, 0});
  x_expr.root = 0;
  bit_program.arith_exprs.push_back(x_expr);
  PostprocessBoolExpr bit_expr;
  PostprocessBoolNode a2b3;
  a2b3.op = PostprocessBoolOp::A2B;
  a2b3.index = 0;
  a2b3.bit_index = 3;
  bit_expr.nodes.push_back(a2b3);
  bit_expr.root = 0;
  bit_program.bool_exprs.push_back(bit_expr);
  bit_program.boolean_outputs = {0};
  validate_postprocess_program(bit_program);
  ReferenceMpcRuntime runtime;
  PostprocessEvalContext bit_ctx;
  bit_ctx.x = 0b1010ULL;
  std::vector<u64> bit_arith;
  std::vector<u8> bit_bool;
  eval_postprocess_program(bit_program, bit_ctx, &runtime, bit_arith, bit_bool);
  assert(bit_bool.size() == 1);
  assert(bit_bool[0] == 1);

  LsbOnlyRuntime lsb_only;
  bool non_lsb_default_threw = false;
  try {
    eval_postprocess_program(bit_program, bit_ctx, &lsb_only, bit_arith, bit_bool);
  } catch (const std::runtime_error&) {
    non_lsb_default_threw = true;
  }
  assert(non_lsb_default_threw);

  PostprocessProgram xhat_program;
  PostprocessArithExpr xhat_expr;
  xhat_expr.nodes.push_back(PostprocessArithNode{PostprocessArithOp::X_HAT, -1, -1, -1, 0});
  xhat_expr.root = 0;
  xhat_program.arith_exprs.push_back(xhat_expr);
  xhat_program.arithmetic_outputs = {0};
  PostprocessEvalContext xhat_ctx;
  xhat_ctx.x = 17;
  xhat_ctx.x_hat = 93;
  std::vector<u64> xhat_arith;
  std::vector<u8> xhat_bool;
  eval_postprocess_program(xhat_program, xhat_ctx, nullptr, xhat_arith, xhat_bool);
  assert(xhat_arith.size() == 1);
  assert(xhat_arith[0] == 93);

  std::cout << "ok\n";
  return 0;
}
