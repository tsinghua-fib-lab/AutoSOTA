#pragma once

#include "suf/common.hpp"

#include <algorithm>
#include <cstddef>
#include <vector>

namespace suf {

enum class PostprocessArithOp : u8 {
  CONST = 0,
  X = 1,
  X_HAT = 2,
  POLY_OUT = 3,
  AUX = 4,
  KAPPA_A = 5,
  ADD = 6,
  SUB = 7,
  MUL = 8,
  B2A = 9
};

enum class PostprocessBoolOp : u8 {
  CONST = 0,
  BOOL_OUT = 1,
  KAPPA_B = 2,
  NOT = 3,
  XOR = 4,
  AND = 5,
  OR = 6,
  A2B = 7
};

struct PostprocessArithNode {
  PostprocessArithOp op = PostprocessArithOp::CONST;
  int lhs = -1;
  int rhs = -1;
  int index = -1;
  u64 value = 0;
};

struct PostprocessBoolNode {
  PostprocessBoolOp op = PostprocessBoolOp::CONST;
  int lhs = -1;
  int rhs = -1;
  int index = -1;
  u8 value = 0;
  int bit_index = 0;
};

struct PostprocessArithExpr {
  std::vector<PostprocessArithNode> nodes;
  int root = -1;
};

struct PostprocessBoolExpr {
  std::vector<PostprocessBoolNode> nodes;
  int root = -1;
};

struct PostprocessProgram {
  std::vector<PostprocessArithExpr> arith_exprs;
  std::vector<PostprocessBoolExpr> bool_exprs;
  std::vector<int> arithmetic_outputs;
  std::vector<int> boolean_outputs;

  bool empty() const {
    return arithmetic_outputs.empty() && boolean_outputs.empty();
  }
};

struct PostprocessCost {
  std::size_t ring_multiplications = 0;
  std::size_t boolean_ands = 0;
  std::size_t b2a_conversions = 0;
  std::size_t a2b_conversions = 0;

  bool requires_runtime() const {
    return ring_multiplications != 0 || boolean_ands != 0 ||
           b2a_conversions != 0 || a2b_conversions != 0;
  }
};

struct PostprocessKappaShape {
  std::size_t arithmetic = 0;
  std::size_t boolean = 0;

  bool empty() const {
    return arithmetic == 0 && boolean == 0;
  }
};

struct MpcRuntime {
  virtual ~MpcRuntime() = default;
  virtual u64 mul(u64 a, u64 b) const = 0;
  virtual u8 bool_and(u8 a, u8 b) const = 0;
  virtual u64 b2a(u8 b) const = 0;
  virtual u8 a2b(u64 x) const = 0;
  virtual u8 a2b_bit(u64 x, int bit_index) const {
    ensure(bit_index >= 0 && bit_index < 64, "postprocess: A2B bit index out of range");
    if (bit_index == 0) return a2b(x);
    fail("postprocess: non-LSB A2B requires an explicit runtime implementation");
  }
};

struct ReferenceMpcRuntime final : MpcRuntime {
  u64 mul(u64 a, u64 b) const override {
    return static_cast<u64>(static_cast<unsigned __int128>(a) *
                            static_cast<unsigned __int128>(b));
  }
  u8 bool_and(u8 a, u8 b) const override {
    return static_cast<u8>((a & b) & 1u);
  }
  u64 b2a(u8 b) const override {
    return static_cast<u64>(b & 1u);
  }
  u8 a2b(u64 x) const override {
    return static_cast<u8>(x & 1ULL);
  }
  u8 a2b_bit(u64 x, int bit_index) const override {
    ensure(bit_index >= 0 && bit_index < 64, "postprocess: A2B bit index out of range");
    return static_cast<u8>((x >> bit_index) & 1ULL);
  }
};

struct PostprocessEvalContext {
  const std::vector<u64>* poly_outputs = nullptr;
  const std::vector<u8>* bool_outputs = nullptr;
  const std::vector<u64>* aux_words = nullptr;
  const std::vector<u64>* kappa_a = nullptr;
  const std::vector<u8>* kappa_b = nullptr;
  u64 x = 0;
  u64 x_hat = 0;
};

namespace detail {

inline const u64& checked_word_ref(const std::vector<u64>* values, int index,
                                   const char* msg) {
  ensure(values != nullptr, msg);
  ensure(index >= 0 && static_cast<std::size_t>(index) < values->size(), msg);
  return (*values)[static_cast<std::size_t>(index)];
}

inline u8 checked_bit_value(const std::vector<u8>* values, int index,
                            const char* msg) {
  ensure(values != nullptr, msg);
  ensure(index >= 0 && static_cast<std::size_t>(index) < values->size(), msg);
  return static_cast<u8>((*values)[static_cast<std::size_t>(index)] & 1u);
}

inline PostprocessCost count_arith_expr_cost(const PostprocessArithExpr& e) {
  PostprocessCost out;
  for (const auto& node : e.nodes) {
    if (node.op == PostprocessArithOp::MUL) ++out.ring_multiplications;
    if (node.op == PostprocessArithOp::B2A) ++out.b2a_conversions;
  }
  return out;
}

inline PostprocessCost count_bool_expr_cost(const PostprocessBoolExpr& e) {
  PostprocessCost out;
  for (const auto& node : e.nodes) {
    if (node.op == PostprocessBoolOp::AND || node.op == PostprocessBoolOp::OR) {
      ++out.boolean_ands;
    }
    if (node.op == PostprocessBoolOp::A2B) ++out.a2b_conversions;
  }
  return out;
}

inline void add_cost(PostprocessCost& dst, const PostprocessCost& src) {
  dst.ring_multiplications += src.ring_multiplications;
  dst.boolean_ands += src.boolean_ands;
  dst.b2a_conversions += src.b2a_conversions;
  dst.a2b_conversions += src.a2b_conversions;
}

struct ReachablePostprocess {
  std::vector<u8> arith;
  std::vector<u8> boolean;
  std::vector<u8> visiting_arith;
  std::vector<u8> visiting_boolean;
};

inline void mark_reachable_arith_expr(const PostprocessProgram& program,
                                      int expr_index,
                                      ReachablePostprocess& reach);

inline void mark_reachable_bool_expr(const PostprocessProgram& program,
                                     int expr_index,
                                     ReachablePostprocess& reach);

inline void mark_reachable_arith_expr(const PostprocessProgram& program,
                                      int expr_index,
                                      ReachablePostprocess& reach) {
  ensure(expr_index >= 0 && static_cast<std::size_t>(expr_index) < program.arith_exprs.size(),
         "postprocess: arithmetic output expression index out of range");
  const std::size_t idx = static_cast<std::size_t>(expr_index);
  if (reach.arith[idx]) return;
  ensure(!reach.visiting_arith[idx], "postprocess: cyclic arithmetic/Boolean Phi dependency");
  reach.visiting_arith[idx] = 1;
  const auto& expr = program.arith_exprs[idx];
  for (const auto& node : expr.nodes) {
    if (node.op == PostprocessArithOp::B2A) {
      mark_reachable_bool_expr(program, node.index, reach);
    }
  }
  reach.visiting_arith[idx] = 0;
  reach.arith[idx] = 1;
}

inline void mark_reachable_bool_expr(const PostprocessProgram& program,
                                     int expr_index,
                                     ReachablePostprocess& reach) {
  ensure(expr_index >= 0 && static_cast<std::size_t>(expr_index) < program.bool_exprs.size(),
         "postprocess: Boolean output expression index out of range");
  const std::size_t idx = static_cast<std::size_t>(expr_index);
  if (reach.boolean[idx]) return;
  ensure(!reach.visiting_boolean[idx], "postprocess: cyclic Boolean/arithmetic Phi dependency");
  reach.visiting_boolean[idx] = 1;
  const auto& expr = program.bool_exprs[idx];
  for (const auto& node : expr.nodes) {
    if (node.op == PostprocessBoolOp::A2B) {
      mark_reachable_arith_expr(program, node.index, reach);
    }
  }
  reach.visiting_boolean[idx] = 0;
  reach.boolean[idx] = 1;
}

inline ReachablePostprocess reachable_postprocess_exprs(
    const PostprocessProgram& program) {
  ReachablePostprocess reach;
  reach.arith.assign(program.arith_exprs.size(), 0);
  reach.boolean.assign(program.bool_exprs.size(), 0);
  reach.visiting_arith.assign(program.arith_exprs.size(), 0);
  reach.visiting_boolean.assign(program.bool_exprs.size(), 0);
  for (const int out : program.arithmetic_outputs) {
    mark_reachable_arith_expr(program, out, reach);
  }
  for (const int out : program.boolean_outputs) {
    mark_reachable_bool_expr(program, out, reach);
  }
  return reach;
}

} // namespace detail

inline PostprocessCost count_postprocess_cost(const PostprocessProgram& program) {
  PostprocessCost out;
  const auto reach = detail::reachable_postprocess_exprs(program);
  for (std::size_t i = 0; i < program.arith_exprs.size(); ++i) {
    if (reach.arith[i]) {
      detail::add_cost(out, detail::count_arith_expr_cost(program.arith_exprs[i]));
    }
  }
  for (std::size_t i = 0; i < program.bool_exprs.size(); ++i) {
    if (reach.boolean[i]) {
      detail::add_cost(out, detail::count_bool_expr_cost(program.bool_exprs[i]));
    }
  }
  return out;
}

inline PostprocessKappaShape required_postprocess_kappa_shape(
    const PostprocessProgram& program) {
  PostprocessKappaShape out;
  const auto reach = detail::reachable_postprocess_exprs(program);
  for (std::size_t ei = 0; ei < program.arith_exprs.size(); ++ei) {
    if (!reach.arith[ei]) continue;
    const auto& expr = program.arith_exprs[ei];
    for (const auto& node : expr.nodes) {
      if (node.op == PostprocessArithOp::KAPPA_A) {
        ensure(node.index >= 0, "postprocess: KAPPA_A index must be non-negative");
        out.arithmetic = std::max(out.arithmetic,
                                  static_cast<std::size_t>(node.index + 1));
      }
    }
  }
  for (std::size_t ei = 0; ei < program.bool_exprs.size(); ++ei) {
    if (!reach.boolean[ei]) continue;
    const auto& expr = program.bool_exprs[ei];
    for (const auto& node : expr.nodes) {
      if (node.op == PostprocessBoolOp::KAPPA_B) {
        ensure(node.index >= 0, "postprocess: KAPPA_B index must be non-negative");
        out.boolean = std::max(out.boolean,
                               static_cast<std::size_t>(node.index + 1));
      }
    }
  }
  return out;
}

inline int max_postprocess_a2b_bit_index(const PostprocessProgram& program) {
  int max_bit = -1;
  const auto reach = detail::reachable_postprocess_exprs(program);
  for (std::size_t ei = 0; ei < program.bool_exprs.size(); ++ei) {
    if (!reach.boolean[ei]) continue;
    const auto& expr = program.bool_exprs[ei];
    for (const auto& node : expr.nodes) {
      if (node.op == PostprocessBoolOp::A2B) {
        max_bit = std::max(max_bit, node.bit_index);
      }
    }
  }
  return max_bit;
}

inline void validate_postprocess_program(const PostprocessProgram& program) {
  for (std::size_t ei = 0; ei < program.arith_exprs.size(); ++ei) {
    const auto& expr = program.arith_exprs[ei];
    ensure(!expr.nodes.empty(), "postprocess: empty arithmetic expression");
    ensure(expr.root >= 0 && static_cast<std::size_t>(expr.root) < expr.nodes.size(),
           "postprocess: arithmetic root out of range");
    for (std::size_t ni = 0; ni < expr.nodes.size(); ++ni) {
      const auto& node = expr.nodes[ni];
      if (node.op == PostprocessArithOp::ADD ||
          node.op == PostprocessArithOp::SUB ||
          node.op == PostprocessArithOp::MUL) {
        ensure(node.lhs >= 0 && node.rhs >= 0, "postprocess: malformed arithmetic binary node");
        ensure(static_cast<std::size_t>(node.lhs) < ni &&
               static_cast<std::size_t>(node.rhs) < ni,
               "postprocess: arithmetic expression must be topologically ordered");
      }
      if (node.op == PostprocessArithOp::B2A) {
        ensure(node.index >= 0 &&
                   static_cast<std::size_t>(node.index) < program.bool_exprs.size(),
               "postprocess: B2A Boolean expression index out of range");
      }
      if (node.op == PostprocessArithOp::KAPPA_A) {
        ensure(node.index >= 0, "postprocess: KAPPA_A index must be non-negative");
      }
    }
  }
  for (std::size_t ei = 0; ei < program.bool_exprs.size(); ++ei) {
    const auto& expr = program.bool_exprs[ei];
    ensure(!expr.nodes.empty(), "postprocess: empty Boolean expression");
    ensure(expr.root >= 0 && static_cast<std::size_t>(expr.root) < expr.nodes.size(),
           "postprocess: Boolean root out of range");
    for (std::size_t ni = 0; ni < expr.nodes.size(); ++ni) {
      const auto& node = expr.nodes[ni];
      if (node.op == PostprocessBoolOp::NOT) {
        ensure(node.lhs >= 0 && static_cast<std::size_t>(node.lhs) < ni,
               "postprocess: malformed NOT node");
      }
      if (node.op == PostprocessBoolOp::XOR ||
          node.op == PostprocessBoolOp::AND ||
          node.op == PostprocessBoolOp::OR) {
        ensure(node.lhs >= 0 && node.rhs >= 0, "postprocess: malformed Boolean binary node");
        ensure(static_cast<std::size_t>(node.lhs) < ni &&
               static_cast<std::size_t>(node.rhs) < ni,
               "postprocess: Boolean expression must be topologically ordered");
      }
      if (node.op == PostprocessBoolOp::A2B) {
        ensure(node.index >= 0 &&
                   static_cast<std::size_t>(node.index) < program.arith_exprs.size(),
               "postprocess: A2B arithmetic expression index out of range");
        ensure(node.bit_index >= 0 && node.bit_index < 64,
               "postprocess: A2B bit index out of range");
      }
      if (node.op == PostprocessBoolOp::KAPPA_B) {
        ensure(node.index >= 0, "postprocess: KAPPA_B index must be non-negative");
      }
    }
  }
  for (const int out : program.arithmetic_outputs) {
    ensure(out >= 0 && static_cast<std::size_t>(out) < program.arith_exprs.size(),
           "postprocess: arithmetic output expression index out of range");
  }
  for (const int out : program.boolean_outputs) {
    ensure(out >= 0 && static_cast<std::size_t>(out) < program.bool_exprs.size(),
           "postprocess: Boolean output expression index out of range");
  }
}

u64 eval_postprocess_arith_expr(const PostprocessProgram& program,
                                int expr_index,
                                const PostprocessEvalContext& ctx,
                                const MpcRuntime* runtime);

u8 eval_postprocess_bool_expr(const PostprocessProgram& program,
                              int expr_index,
                              const PostprocessEvalContext& ctx,
                              const MpcRuntime* runtime);

inline u64 eval_arith_node_value(const PostprocessProgram& program,
                                 const PostprocessArithExpr& expr,
                                 const std::vector<u64>& values,
                                 const PostprocessEvalContext& ctx,
                                 const MpcRuntime* runtime,
                                 const PostprocessArithNode& node) {
  switch (node.op) {
    case PostprocessArithOp::CONST:
      return node.value;
    case PostprocessArithOp::X:
      return ctx.x;
    case PostprocessArithOp::X_HAT:
      return ctx.x_hat;
    case PostprocessArithOp::POLY_OUT:
      return detail::checked_word_ref(ctx.poly_outputs, node.index,
                                      "postprocess: polynomial output index out of range");
    case PostprocessArithOp::AUX:
      return detail::checked_word_ref(ctx.aux_words, node.index,
                                      "postprocess: aux payload index out of range");
    case PostprocessArithOp::KAPPA_A:
      return detail::checked_word_ref(ctx.kappa_a, node.index,
                                      "postprocess: kappaA index out of range");
    case PostprocessArithOp::ADD:
      ensure(node.lhs >= 0 && node.rhs >= 0, "postprocess: malformed ADD node");
      return values[static_cast<std::size_t>(node.lhs)] +
             values[static_cast<std::size_t>(node.rhs)];
    case PostprocessArithOp::SUB:
      ensure(node.lhs >= 0 && node.rhs >= 0, "postprocess: malformed SUB node");
      return values[static_cast<std::size_t>(node.lhs)] -
             values[static_cast<std::size_t>(node.rhs)];
    case PostprocessArithOp::MUL:
      ensure(runtime != nullptr, "postprocess: MUL requires MpcRuntime");
      ensure(node.lhs >= 0 && node.rhs >= 0, "postprocess: malformed MUL node");
      return runtime->mul(values[static_cast<std::size_t>(node.lhs)],
                          values[static_cast<std::size_t>(node.rhs)]);
    case PostprocessArithOp::B2A:
      ensure(runtime != nullptr, "postprocess: B2A requires MpcRuntime");
      return runtime->b2a(eval_postprocess_bool_expr(program, node.index, ctx, runtime));
  }
  fail("postprocess: unknown arithmetic op");
}

inline u8 eval_bool_node_value(const PostprocessProgram& program,
                               const PostprocessBoolExpr& expr,
                               const std::vector<u8>& values,
                               const PostprocessEvalContext& ctx,
                               const MpcRuntime* runtime,
                               const PostprocessBoolNode& node) {
  switch (node.op) {
    case PostprocessBoolOp::CONST:
      return static_cast<u8>(node.value & 1u);
    case PostprocessBoolOp::BOOL_OUT:
      return detail::checked_bit_value(ctx.bool_outputs, node.index,
                                       "postprocess: Boolean output index out of range");
    case PostprocessBoolOp::KAPPA_B:
      return detail::checked_bit_value(ctx.kappa_b, node.index,
                                       "postprocess: kappaB index out of range");
    case PostprocessBoolOp::NOT:
      ensure(node.lhs >= 0, "postprocess: malformed NOT node");
      return static_cast<u8>(values[static_cast<std::size_t>(node.lhs)] ^ 1u);
    case PostprocessBoolOp::XOR:
      ensure(node.lhs >= 0 && node.rhs >= 0, "postprocess: malformed XOR node");
      return static_cast<u8>((values[static_cast<std::size_t>(node.lhs)] ^
                              values[static_cast<std::size_t>(node.rhs)]) & 1u);
    case PostprocessBoolOp::AND:
      ensure(runtime != nullptr, "postprocess: AND requires MpcRuntime");
      ensure(node.lhs >= 0 && node.rhs >= 0, "postprocess: malformed AND node");
      return runtime->bool_and(values[static_cast<std::size_t>(node.lhs)],
                               values[static_cast<std::size_t>(node.rhs)]);
    case PostprocessBoolOp::OR: {
      ensure(runtime != nullptr, "postprocess: OR requires MpcRuntime");
      ensure(node.lhs >= 0 && node.rhs >= 0, "postprocess: malformed OR node");
      const u8 a = values[static_cast<std::size_t>(node.lhs)] & 1u;
      const u8 b = values[static_cast<std::size_t>(node.rhs)] & 1u;
      const u8 ab = runtime->bool_and(a, b);
      return static_cast<u8>((a ^ b ^ ab) & 1u);
    }
    case PostprocessBoolOp::A2B:
      ensure(runtime != nullptr, "postprocess: A2B requires MpcRuntime");
      return runtime->a2b_bit(eval_postprocess_arith_expr(program, node.index, ctx, runtime),
                              node.bit_index);
  }
  fail("postprocess: unknown Boolean op");
}

inline u64 eval_postprocess_arith_expr(const PostprocessProgram& program,
                                       int expr_index,
                                       const PostprocessEvalContext& ctx,
                                       const MpcRuntime* runtime) {
  ensure(expr_index >= 0 && static_cast<std::size_t>(expr_index) < program.arith_exprs.size(),
         "postprocess: arithmetic expression index out of range");
  const auto& expr = program.arith_exprs[static_cast<std::size_t>(expr_index)];
  ensure(expr.root >= 0 && static_cast<std::size_t>(expr.root) < expr.nodes.size(),
         "postprocess: malformed arithmetic expression root");
  std::vector<u64> values(expr.nodes.size(), 0);
  for (std::size_t i = 0; i < expr.nodes.size(); ++i) {
    values[i] = eval_arith_node_value(program, expr, values, ctx, runtime, expr.nodes[i]);
  }
  return values[static_cast<std::size_t>(expr.root)];
}

inline u8 eval_postprocess_bool_expr(const PostprocessProgram& program,
                                     int expr_index,
                                     const PostprocessEvalContext& ctx,
                                     const MpcRuntime* runtime) {
  ensure(expr_index >= 0 && static_cast<std::size_t>(expr_index) < program.bool_exprs.size(),
         "postprocess: Boolean expression index out of range");
  const auto& expr = program.bool_exprs[static_cast<std::size_t>(expr_index)];
  ensure(expr.root >= 0 && static_cast<std::size_t>(expr.root) < expr.nodes.size(),
         "postprocess: malformed Boolean expression root");
  std::vector<u8> values(expr.nodes.size(), 0);
  for (std::size_t i = 0; i < expr.nodes.size(); ++i) {
    values[i] = eval_bool_node_value(program, expr, values, ctx, runtime, expr.nodes[i]);
  }
  return static_cast<u8>(values[static_cast<std::size_t>(expr.root)] & 1u);
}

inline void eval_postprocess_program(const PostprocessProgram& program,
                                     const PostprocessEvalContext& ctx,
                                     const MpcRuntime* runtime,
                                     std::vector<u64>& arith_outputs,
                                     std::vector<u8>& bool_outputs) {
  if (program.arithmetic_outputs.empty()) {
    arith_outputs = ctx.poly_outputs ? *ctx.poly_outputs : std::vector<u64>{};
  } else {
    arith_outputs.resize(program.arithmetic_outputs.size());
    for (std::size_t i = 0; i < program.arithmetic_outputs.size(); ++i) {
      arith_outputs[i] = eval_postprocess_arith_expr(program, program.arithmetic_outputs[i], ctx, runtime);
    }
  }

  if (program.boolean_outputs.empty()) {
    bool_outputs = ctx.bool_outputs ? *ctx.bool_outputs : std::vector<u8>{};
  } else {
    bool_outputs.resize(program.boolean_outputs.size());
    for (std::size_t i = 0; i < program.boolean_outputs.size(); ++i) {
      bool_outputs[i] = eval_postprocess_bool_expr(program, program.boolean_outputs[i], ctx, runtime);
    }
  }
}

} // namespace suf
