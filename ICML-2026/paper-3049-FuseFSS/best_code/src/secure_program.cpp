#include "suf/secure_program.hpp"

#ifdef SUF_HAVE_CUDA

#include "suf/masked_compile.hpp"
#include "suf/gpu_kernels.hpp"
#include "suf/pfss_batch.hpp"

namespace suf {

namespace {
inline u64 mul_mod64(u64 a, u64 b) {
  return static_cast<u64>(static_cast<unsigned __int128>(a) * static_cast<unsigned __int128>(b));
}

std::vector<std::vector<u64>> build_binom(int degree) {
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

std::vector<u64> build_pow_offset(int degree, u64 offset) {
  std::vector<u64> pow(static_cast<std::size_t>(degree + 1), 0);
  pow[0] = 1;
  for (int i = 1; i <= degree; ++i) {
    pow[static_cast<std::size_t>(i)] = mul_mod64(pow[static_cast<std::size_t>(i - 1)], offset);
  }
  return pow;
}

u64 sub_mod_bits(u64 a, u64 b, int bits) {
  if (bits >= 64) return a - b;
  const u64 modulus = 1ULL << bits;
  return (a + modulus - (b & (modulus - 1ULL))) & (modulus - 1ULL);
}

std::vector<u64> shift_poly_coeffs(const std::vector<u64>& coeffs,
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

std::vector<u8> const_predicate_shares(const SUFDescriptor& desc, int party) {
  std::vector<u8> bits(desc.predicates.size(), 0);
  for (std::size_t i = 0; i < desc.predicates.size(); ++i) {
    if (desc.predicates[i].kind == PredKind::CONST) {
      bits[i] = (party == 0) ? 0u : static_cast<u8>(desc.predicates[i].param & 1ULL);
    }
  }
  return bits;
}

bool helpers_need_interactive_and(const SUFDescriptor& desc) {
  for (const auto& helper : desc.helpers) {
    for (const auto& node : helper.nodes) {
      if (node.kind == BoolNode::Kind::AND || node.kind == BoolNode::Kind::OR) {
        return true;
      }
    }
  }
  return false;
}

u64 mask_for_bits_local(int bits) {
  return bits >= 64 ? ~0ULL : ((1ULL << bits) - 1ULL);
}

u64 eval_predicate_plain(const Predicate& p, u64 x, int in_bits) {
  const u64 in_mask = mask_for_bits_local(in_bits);
  switch (p.kind) {
    case PredKind::LT: {
      const u64 shifted = (x + p.input_add) & in_mask;
      return shifted < (p.param & in_mask) ? 1ULL : 0ULL;
    }
    case PredKind::LTLOW: {
      const int f = static_cast<int>(p.f);
      const u64 low_mask = mask_for_bits_local(f);
      const u64 shifted = (x + p.input_add) & low_mask;
      return shifted < (p.gamma & low_mask) ? 1ULL : 0ULL;
    }
    case PredKind::MSB:
      return (x >> (in_bits - 1)) & 1ULL;
    case PredKind::MSB_ADD:
      return ((x + p.param) & in_mask) >> (in_bits - 1);
    case PredKind::CONST:
      return p.param & 1ULL;
  }
  fail("paper-strict: unknown predicate kind");
}

u8 eval_bool_expr_plain(const BoolExpr& expr,
                        const std::vector<Predicate>& predicates,
                        u64 x,
                        int in_bits,
                        const MpcRuntime& runtime) {
  ensure(expr.root >= 0 && static_cast<std::size_t>(expr.root) < expr.nodes.size(),
         "paper-strict: malformed BoolExpr root");
  std::vector<u8> values(expr.nodes.size(), 0);
  for (std::size_t i = 0; i < expr.nodes.size(); ++i) {
    const auto& node = expr.nodes[i];
    switch (node.kind) {
      case BoolNode::Kind::PRED:
        ensure(node.pred_index >= 0 &&
                   static_cast<std::size_t>(node.pred_index) < predicates.size(),
               "paper-strict: predicate index out of range");
        values[i] = static_cast<u8>(eval_predicate_plain(predicates[static_cast<std::size_t>(node.pred_index)],
                                                         x, in_bits) & 1ULL);
        break;
      case BoolNode::Kind::NOT:
        values[i] = static_cast<u8>(values[static_cast<std::size_t>(node.lhs)] ^ 1u);
        break;
      case BoolNode::Kind::XOR:
        values[i] = static_cast<u8>((values[static_cast<std::size_t>(node.lhs)] ^
                                     values[static_cast<std::size_t>(node.rhs)]) & 1u);
        break;
      case BoolNode::Kind::AND:
        values[i] = runtime.bool_and(values[static_cast<std::size_t>(node.lhs)],
                                     values[static_cast<std::size_t>(node.rhs)]);
        break;
      case BoolNode::Kind::OR: {
        const u8 a = values[static_cast<std::size_t>(node.lhs)] & 1u;
        const u8 b = values[static_cast<std::size_t>(node.rhs)] & 1u;
        values[i] = static_cast<u8>((a ^ b ^ runtime.bool_and(a, b)) & 1u);
        break;
      }
    }
  }
  return static_cast<u8>(values[static_cast<std::size_t>(expr.root)] & 1u);
}

u64 eval_poly_horner_runtime(const Polynomial& poly,
                             int degree,
                             u64 x,
                             const MpcRuntime& runtime) {
  u64 y = 0;
  for (int k = degree; k >= 0; --k) {
    if (k != degree) {
      y = runtime.mul(y, x);
    }
    if (static_cast<std::size_t>(k) < poly.coeffs.size()) {
      y += poly.coeffs[static_cast<std::size_t>(k)];
    }
  }
  return y;
}

u64 split_arith_share(u64 value, std::uint64_t seed, std::size_t item, std::size_t out, int party) {
  std::mt19937_64 rng(seed ^ 0xA0761D6478BD642FULL ^
                      (static_cast<u64>(item) * 0xE7037ED1A0B428DBULL) ^
                      (static_cast<u64>(out) * 0x8EBC6AF09C88C6E3ULL));
  const u64 share0 = rng();
  return party == 0 ? share0 : value - share0;
}

u64 split_bool_share(u8 value, std::uint64_t seed, std::size_t item, std::size_t out, int party) {
  std::mt19937_64 rng(seed ^ 0x589965CC75374CC3ULL ^
                      (static_cast<u64>(item) * 0x1D8E4E27C47D124FULL) ^
                      (static_cast<u64>(out) * 0xEB44ACCAB455D165ULL));
  const u64 share0 = rng() & 1ULL;
  return party == 0 ? share0 : (share0 ^ static_cast<u64>(value & 1u));
}
} // namespace

GpuSecureSufProgram::GpuSecureSufProgram(const OperatorSpecification& spec,
                                         int party,
                                         std::uint64_t seed,
                                         bool mask_aware,
                                         u64 mask_in)
  : GpuSecureSufProgram(spec, party, seed,
                        SecureEvalOptions{mask_aware, mask_in, nullptr,
                                          SecureEvalMode::FastPublicX, nullptr}) {}

GpuSecureSufProgram::GpuSecureSufProgram(const OperatorSpecification& spec,
                                         int party,
                                         std::uint64_t seed,
                                         const SecureEvalOptions& options)
  : has_operator_spec_(true),
    spec_(spec),
    party_(party),
    seed_(seed),
    in_bits_(spec.in_bits),
	    eval_mode_(options.mode),
	    runtime_(options.runtime),
	    mask_aware_(options.mask_aware),
	    r_in_(options.mask_in) {
	  if (options.mask_vector != nullptr) {
	    ensure(mask_aware_, "GpuSecureSufProgram: mask_vector requires mask_aware=true");
	    ensure(eval_mode_ == SecureEvalMode::PaperStrictSharedX,
	           "GpuSecureSufProgram: per-wire mask_vector is only supported by PaperStrictSharedX");
	    r_in_vector_ = *options.mask_vector;
	    ensure(!r_in_vector_.empty(), "GpuSecureSufProgram: mask_vector must not be empty");
	  }
  if (options.kappa_a != nullptr) {
    kappa_a_values_ = *options.kappa_a;
    kappa_a_count_ = options.kappa_a_count;
    ensure(!kappa_a_values_.empty(), "GpuSecureSufProgram: kappa_a must not be empty");
  }
  if (options.kappa_b != nullptr) {
    kappa_b_values_ = *options.kappa_b;
    kappa_b_count_ = options.kappa_b_count;
    ensure(!kappa_b_values_.empty(), "GpuSecureSufProgram: kappa_b must not be empty");
  }

	  OperatorCompileOptions opts;
	  opts.mask_aware = options.mask_aware;
	  opts.input_mask = options.mask_in;
  opts.party = party;
  opts.seed = seed;
  const auto compiled = compile_operator_spec(spec, opts);

  desc_ = compiled.backend_descriptor;
  helpers_require_interactive_and_ = helpers_need_interactive_and(desc_);
  plan_ = compiled.packed_comparison.plan;
  source_arith_outputs_ = compiled.shape.arithmetic_outputs;
  arith_outputs_ = compiled.shape.final_arithmetic_outputs;
  helper_outputs_ = compiled.shape.final_boolean_outputs;
  poly_degree_ = compiled.shape.max_degree;
  coeff_words_per_input_ = compiled.interval_lookup.payload_words;
  aux_words_ = operator_spec_aux_words(spec);
  has_postprocess_ = !spec.postprocess.empty();
  postprocess_requires_runtime_ = count_postprocess_cost(spec.postprocess).requires_runtime();
  ensure(source_arith_outputs_ > 0, "GpuSecureSufProgram: OperatorSpecification requires arithmetic outputs");
  ensure(arith_outputs_ > 0, "GpuSecureSufProgram: OperatorSpecification final outputs are empty");
  ensure(coeff_words_per_input_ >= source_arith_outputs_ * static_cast<std::size_t>(poly_degree_ + 1),
         "GpuSecureSufProgram: compiled payload shape does not match polynomial outputs");

  if (mask_aware_) {
    const_pred_bits_ = compiled.packed_comparison.const_pred_bits;
  } else {
    const_pred_bits_ = const_predicate_shares(desc_, party);
  }

  gpu_prog_ = std::make_unique<GpuSufProgram>(desc_);

  pred_to_query_.clear();
  pred_to_query_.assign(desc_.predicates.size(), -1);
  for (std::size_t q = 0; q < compiled.packed_comparison.query_to_predicate.size(); ++q) {
    const int pred = compiled.packed_comparison.query_to_predicate[q];
    if (pred >= 0 && static_cast<std::size_t>(pred) < pred_to_query_.size()) {
      pred_to_query_[static_cast<std::size_t>(pred)] = static_cast<int>(q);
    }
  }

  std::mt19937_64 rng(seed ^ 0x9E3779B97F4A7C15ULL);
  if (!plan_.queries.empty()) {
    auto batch = build_dpf_batch(plan_, party, rng);
    upload_dpf_batch(batch, dpf_gpu_);
    dpf_loaded_ = true;
  }

  if (!compiled.packed_comparison.query_to_predicate.empty()) {
    cudaMalloc(&d_query_to_pred_,
               compiled.packed_comparison.query_to_predicate.size() * sizeof(int));
    cudaMemcpy(d_query_to_pred_,
               compiled.packed_comparison.query_to_predicate.data(),
               compiled.packed_comparison.query_to_predicate.size() * sizeof(int),
               cudaMemcpyHostToDevice);
  }
  if (!const_pred_bits_.empty()) {
    cudaMalloc(&d_const_pred_bits_, const_pred_bits_.size() * sizeof(u8));
    cudaMemcpy(d_const_pred_bits_, const_pred_bits_.data(),
               const_pred_bits_.size() * sizeof(u8), cudaMemcpyHostToDevice);
  }

  auto lut_key = gen_interval_lut_v2(compiled.interval_lookup.cutpoints,
                                     compiled.interval_lookup.payloads,
                                     spec.in_bits,
                                     party,
                                     rng);
  upload_interval_lut_v2(lut_key, coeff_key_);
  use_coeff_lut_ = true;
}

GpuSecureSufProgram::GpuSecureSufProgram(const SUFDescriptor& d, int party, std::uint64_t seed,
                                         int in_bits_override, bool mask_aware, u64 mask_in)
  : desc_(d),
    party_(party),
    seed_(seed),
    mask_aware_(mask_aware),
    r_in_(mask_in) {
  std::mt19937_64 rng(seed);
  int in_bits = 64;
  if (in_bits_override > 0) {
    ensure(in_bits_override <= 64, "GpuSecureSufProgram: in_bits_override must be 1..64");
    in_bits = in_bits_override;
  }
  in_bits_ = in_bits;

  if (mask_aware_) {
    auto inst = compile_masked_gate_instance(d, in_bits, r_in_, party, rng);
    desc_ = std::move(inst.desc);
    const_pred_bits_ = std::move(inst.const_pred_bits);
  } else {
    const_pred_bits_ = const_predicate_shares(desc_, party);
  }
  helpers_require_interactive_and_ = helpers_need_interactive_and(desc_);
  plan_ = compile_pfss_plan(desc_, in_bits);

  gpu_prog_ = std::make_unique<GpuSufProgram>(desc_);

  pred_to_query_.clear();
  pred_to_query_.reserve(desc_.predicates.size());
  std::vector<int> query_to_pred;
  query_to_pred.reserve(plan_.queries.size());
  int qidx = 0;
  for (std::size_t i = 0; i < desc_.predicates.size(); ++i) {
    if (desc_.predicates[i].kind == PredKind::CONST) {
      pred_to_query_.push_back(-1);
    } else {
      pred_to_query_.push_back(qidx);
      query_to_pred.push_back(static_cast<int>(i));
      ++qidx;
    }
  }

  if (!plan_.queries.empty()) {
    auto batch = build_dpf_batch(plan_, party, rng);
    upload_dpf_batch(batch, dpf_gpu_);
    dpf_loaded_ = true;
  }

  if (!query_to_pred.empty()) {
    cudaMalloc(&d_query_to_pred_, query_to_pred.size() * sizeof(int));
    cudaMemcpy(d_query_to_pred_, query_to_pred.data(),
               query_to_pred.size() * sizeof(int), cudaMemcpyHostToDevice);
  }
  if (!const_pred_bits_.empty()) {
    cudaMalloc(&d_const_pred_bits_, const_pred_bits_.size() * sizeof(u8));
    cudaMemcpy(d_const_pred_bits_, const_pred_bits_.data(),
               const_pred_bits_.size() * sizeof(u8), cudaMemcpyHostToDevice);
  }

  poly_degree_ = 0;
  for (const auto& p : desc_.polys) {
    if (!p.coeffs.empty()) {
      poly_degree_ = std::max<int>(poly_degree_, static_cast<int>(p.coeffs.size()) - 1);
    }
  }
  arith_outputs_ = 1;
  source_arith_outputs_ = 1;
  helper_outputs_ = desc_.helpers.size();
  coeff_words_per_input_ = static_cast<std::size_t>(poly_degree_ + 1);

  if (poly_degree_ == 0) {
    std::vector<u64> cutpoints = desc_.cuts;
    std::vector<std::vector<u64>> payloads(desc_.polys.size(), std::vector<u64>(1));
    for (std::size_t i = 0; i < desc_.polys.size(); ++i) {
      payloads[i][0] = desc_.polys[i].coeffs.empty() ? 0ULL : desc_.polys[i].coeffs[0];
    }
    auto lut_key = gen_interval_lut_v2(cutpoints, payloads, in_bits, party, rng);
    upload_interval_lut_v2(lut_key, interval_key_);
    use_interval_lut_ = true;
  } else {
    const auto binom = build_binom(poly_degree_);
    std::vector<u64> cutpoints = desc_.cuts;
    std::vector<std::vector<u64>> payloads(desc_.polys.size());
    for (std::size_t i = 0; i < desc_.polys.size(); ++i) {
      const u64 cut = cutpoints[i];
      const u64 plain = mask_aware_ ? sub_mod_bits(cut, r_in_, in_bits) : cut;
      const u64 x_offset = mask_aware_ ? (plain - cut) : 0ULL;
      const auto pow_offset = build_pow_offset(poly_degree_, x_offset);
      payloads[i] = shift_poly_coeffs(desc_.polys[i].coeffs, poly_degree_, binom, pow_offset);
    }
    auto lut_key = gen_interval_lut_v2(cutpoints, payloads, in_bits, party, rng);
    upload_interval_lut_v2(lut_key, coeff_key_);
    use_coeff_lut_ = true;
  }
}

GpuSecureSufProgram::~GpuSecureSufProgram() {
  if (dpf_loaded_) free_dpf_batch(dpf_gpu_);
  if (use_interval_lut_) free_interval_lut_v2(interval_key_);
  if (use_coeff_lut_) free_interval_lut_v2(coeff_key_);
  if (d_pred_bits_) cudaFree(d_pred_bits_);
  if (d_query_bits_) cudaFree(d_query_bits_);
  if (d_coeffs_) cudaFree(d_coeffs_);
  if (d_query_to_pred_) cudaFree(d_query_to_pred_);
  if (d_const_pred_bits_) cudaFree(d_const_pred_bits_);
  d_pred_bits_ = nullptr;
  pred_capacity_ = 0;
  d_query_bits_ = nullptr;
  query_capacity_ = 0;
  d_coeffs_ = nullptr;
  coeff_capacity_ = 0;
  d_query_to_pred_ = nullptr;
  d_const_pred_bits_ = nullptr;
}

u8* GpuSecureSufProgram::ensure_pred_bits(std::size_t n) const {
  const std::size_t needed = desc_.predicates.size() * n;
  if (needed == 0) return nullptr;
  if (needed > pred_capacity_) {
    if (d_pred_bits_) cudaFree(d_pred_bits_);
    cudaMalloc(&d_pred_bits_, needed * sizeof(u8));
    pred_capacity_ = needed;
  }
  return d_pred_bits_;
}

u8* GpuSecureSufProgram::ensure_query_bits(std::size_t n) const {
  const std::size_t needed = plan_.queries.size() * n;
  if (needed == 0) return nullptr;
  if (needed > query_capacity_) {
    if (d_query_bits_) cudaFree(d_query_bits_);
    cudaMalloc(&d_query_bits_, needed * sizeof(u8));
    query_capacity_ = needed;
  }
  return d_query_bits_;
}

u64* GpuSecureSufProgram::ensure_coeffs(std::size_t n) const {
  if (!use_coeff_lut_) return nullptr;
  const std::size_t words = coeff_words_per_input_ == 0
                                ? static_cast<std::size_t>(poly_degree_ + 1)
                                : coeff_words_per_input_;
  const std::size_t needed = n * words;
  if (needed == 0) return nullptr;
  if (needed > coeff_capacity_) {
    if (d_coeffs_) cudaFree(d_coeffs_);
    cudaMalloc(&d_coeffs_, needed * sizeof(u64));
    coeff_capacity_ = needed;
  }
  return d_coeffs_;
}

void GpuSecureSufProgram::eval_paper_strict_host(const std::vector<u64>& h_masked_input,
                                                 std::vector<u64>& h_out_arith,
                                                 std::vector<u64>* h_out_helpers,
                                                 const MpcRuntime& runtime) const {
  ensure(has_operator_spec_, "paper-strict eval requires OperatorSpecification constructor");
  validate_operator_spec(spec_);
  const std::size_t n = h_masked_input.size();
  h_out_arith.assign(arith_outputs_ * n, 0);
  if (h_out_helpers) {
    h_out_helpers->assign(helper_outputs_ * n, 0);
  }

	  const u64 mask = mask_for_bits_local(in_bits_);
	  const std::size_t aux = operator_spec_aux_words(spec_);
	  if (!r_in_vector_.empty()) {
	    ensure(mask_aware_, "paper-strict: mask_vector requires mask-aware evaluation");
	    ensure(r_in_vector_.size() == n, "paper-strict: mask_vector length must match input batch");
	  }
	  for (std::size_t i = 0; i < n; ++i) {
	    const u64 x_hat = h_masked_input[i] & mask;
	    const u64 r_in = r_in_vector_.empty() ? r_in_ : (r_in_vector_[i] & mask);
	    const u64 x = mask_aware_ ? sub_mod_bits(x_hat, r_in, in_bits_) : x_hat;
    const std::size_t piece = detail::interval_index_for(spec_.boundaries, x);
    const auto& op_piece = spec_.pieces[piece];

    std::vector<u64> poly_outputs(source_arith_outputs_, 0);
    for (std::size_t out = 0; out < source_arith_outputs_; ++out) {
      poly_outputs[out] = eval_poly_horner_runtime(op_piece.polys[out], poly_degree_, x, runtime);
    }

    std::vector<u8> bool_outputs(op_piece.bool_outputs.size(), 0);
    for (std::size_t out = 0; out < op_piece.bool_outputs.size(); ++out) {
      bool_outputs[out] = eval_bool_expr_plain(op_piece.bool_outputs[out],
                                               spec_.predicates,
                                               x,
                                               in_bits_,
                                               runtime);
    }

    std::vector<u64> aux_words(aux, 0);
    for (std::size_t a = 0; a < aux; ++a) {
      aux_words[a] = op_piece.aux_words[a];
    }

    PostprocessEvalContext ctx;
    ctx.poly_outputs = &poly_outputs;
    ctx.bool_outputs = &bool_outputs;
    ctx.aux_words = &aux_words;
    ctx.x = x;
    ctx.x_hat = x_hat;
    std::vector<u64> kappa_a_current;
    if (!kappa_a_values_.empty()) {
      const std::size_t count =
          kappa_a_count_ == 0 ? kappa_a_values_.size() : kappa_a_count_;
      ensure(count != 0, "paper-strict: kappaA count must be non-zero");
      if (kappa_a_values_.size() == count) {
        kappa_a_current = kappa_a_values_;
      } else {
        ensure(kappa_a_values_.size() == count * n,
               "paper-strict: kappaA vector must have count or count*n entries");
        kappa_a_current.assign(kappa_a_values_.begin() +
                                   static_cast<std::ptrdiff_t>(i * count),
                               kappa_a_values_.begin() +
                                   static_cast<std::ptrdiff_t>((i + 1) * count));
      }
      ctx.kappa_a = &kappa_a_current;
    }
    std::vector<u8> kappa_b_current;
    if (!kappa_b_values_.empty()) {
      const std::size_t count =
          kappa_b_count_ == 0 ? kappa_b_values_.size() : kappa_b_count_;
      ensure(count != 0, "paper-strict: kappaB count must be non-zero");
      if (kappa_b_values_.size() == count) {
        kappa_b_current = kappa_b_values_;
      } else {
        ensure(kappa_b_values_.size() == count * n,
               "paper-strict: kappaB vector must have count or count*n entries");
        kappa_b_current.assign(kappa_b_values_.begin() +
                                   static_cast<std::ptrdiff_t>(i * count),
                               kappa_b_values_.begin() +
                                   static_cast<std::ptrdiff_t>((i + 1) * count));
      }
      ctx.kappa_b = &kappa_b_current;
    }

    std::vector<u64> final_arith;
    std::vector<u8> final_bool;
    eval_postprocess_program(spec_.postprocess, ctx, &runtime, final_arith, final_bool);
    ensure(final_arith.size() == arith_outputs_, "paper-strict: arithmetic output count mismatch");
    ensure(final_bool.size() == helper_outputs_, "paper-strict: helper output count mismatch");

    for (std::size_t out = 0; out < arith_outputs_; ++out) {
      h_out_arith[out * n + i] = split_arith_share(final_arith[out], seed_, i, out, party_);
    }
    if (h_out_helpers) {
      for (std::size_t out = 0; out < helper_outputs_; ++out) {
        (*h_out_helpers)[out * n + i] = split_bool_share(final_bool[out], seed_, i, out, party_);
      }
    }
  }
}

void GpuSecureSufProgram::eval_paper_strict_device(const u64* d_in,
                                                   std::size_t n,
                                                   u64* d_out_arith,
                                                   u64* d_out_helpers,
                                                   cudaStream_t stream) const {
  (void)stream;
  ensure(runtime_ != nullptr,
         "GpuSecureSufProgram: PaperStrictSharedX requires SecureEvalOptions.runtime");
  std::vector<u64> h_in(n);
  cudaMemcpy(h_in.data(), d_in, n * sizeof(u64), cudaMemcpyDeviceToHost);

  std::vector<u64> h_arith;
  std::vector<u64> h_helpers;
  eval_paper_strict_host(h_in, h_arith, d_out_helpers ? &h_helpers : nullptr, *runtime_);
  if (d_out_arith) {
    cudaMemcpy(d_out_arith, h_arith.data(), h_arith.size() * sizeof(u64), cudaMemcpyHostToDevice);
  }
  if (d_out_helpers) {
    cudaMemcpy(d_out_helpers, h_helpers.data(), h_helpers.size() * sizeof(u64), cudaMemcpyHostToDevice);
  }
}

void GpuSecureSufProgram::eval(const u64* d_in, std::size_t n,
                               u64* d_out_arith, u64* d_out_helpers,
                               cudaStream_t stream) const {
  if (eval_mode_ == SecureEvalMode::PaperStrictSharedX) {
    eval_paper_strict_device(d_in, n, d_out_arith, d_out_helpers, stream);
    return;
  }
  ensure(!has_postprocess_,
         "GpuSecureSufProgram: post-processing Phi requires PaperStrictSharedX or Sigma runtime");
  u8* d_pred_bits = ensure_pred_bits(n);
  u8* d_query_bits = ensure_query_bits(n);
  if (d_query_bits && dpf_loaded_) {
    eval_dpf_batch_gpu(d_in, n, dpf_gpu_, d_query_bits, stream);
  }
  if (d_pred_bits) {
    launch_fill_const_pred_bits(d_const_pred_bits_, n,
                                static_cast<int>(desc_.predicates.size()),
                                d_pred_bits, stream);
    if (d_query_bits && d_query_to_pred_) {
      launch_scatter_pred_bits(d_query_bits, n,
                               static_cast<int>(plan_.queries.size()),
                               d_query_to_pred_, d_pred_bits, stream);
    }
  }

  if (d_out_arith) {
    if (use_interval_lut_) {
      eval_interval_lut_v2_gpu(d_in, n, interval_key_, d_out_arith, stream);
    } else if (use_coeff_lut_) {
      u64* d_coeffs = ensure_coeffs(n);
      eval_interval_lut_v2_gpu(d_in, n, coeff_key_, d_coeffs, stream);
      launch_eval_poly_from_coeffs_multi(d_in, d_coeffs, n, poly_degree_,
                                         static_cast<int>(arith_outputs_),
                                         coeff_words_per_input_,
                                         d_out_arith, stream);
    } else {
      gpu_prog_->eval_poly_only(d_in, n, d_out_arith, stream);
    }
  }
  if (d_out_helpers && d_pred_bits) {
    ensure(!helpers_require_interactive_and_,
           "GpuSecureSufProgram: helper AND/OR requires interactive Boolean-AND preprocessing; "
           "standalone generic runtime refuses insecure local post-processing");
    gpu_prog_->eval_helpers_from_pred_bits(d_pred_bits, n, d_out_helpers, stream);
  }

}

} // namespace suf

#endif // SUF_HAVE_CUDA
