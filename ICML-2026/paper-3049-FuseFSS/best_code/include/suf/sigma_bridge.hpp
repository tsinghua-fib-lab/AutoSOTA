#pragma once

#include <cstddef>
#include <cstdint>

struct SigmaPeer;
struct Stats;
namespace suf {
struct OperatorSpecification;
}

enum SufSigmaCapabilityFlags : int {
  SUF_SIGMA_CAP_SUPPORTED = 1 << 0,
  SUF_SIGMA_CAP_NEEDS_MUL = 1 << 1,
  SUF_SIGMA_CAP_NEEDS_AND = 1 << 2,
  SUF_SIGMA_CAP_NEEDS_B2A = 1 << 3,
  SUF_SIGMA_CAP_NEEDS_A2B = 1 << 4,
  SUF_SIGMA_CAP_STRICT_SUPPORTED = 1 << 5,
  SUF_SIGMA_CAP_NEEDS_KAPPA_A = 1 << 6,
  SUF_SIGMA_CAP_NEEDS_KAPPA_B = 1 << 7,
};

struct SufSigmaCompiledOperatorResult {
  std::size_t n = 0;
  std::size_t arithmetic_outputs = 0;
  std::size_t boolean_outputs = 0;
  std::uint64_t** d_arithmetic = nullptr;
  std::uint64_t** d_boolean = nullptr;
  int capability_flags = 0;
};

struct SufSigmaPostprocessContext {
  // Each entry is a device vector of length n in the same wire
  // representation as the executor phase: masks during keygen and public
  // masked openings during eval.
  std::size_t kappa_a_count = 0;
  const std::uint64_t* const* d_kappa_a = nullptr;
  // Optional length arrays. When provided, strict production execution
  // validates each kappa vector before consuming the key stream.
  const std::size_t* kappa_a_lengths = nullptr;
  std::size_t kappa_b_count = 0;
  const std::uint64_t* const* d_kappa_b = nullptr;
  const std::size_t* kappa_b_lengths = nullptr;
};

extern "C" void suf_sigma_reset_keygen();
extern "C" void suf_sigma_reset_eval();
extern "C" void suf_sigma_consume_key();
extern "C" void suf_sigma_consume_nexp_key(int bw, int scale, std::size_t n);
extern "C" void suf_sigma_consume_inverse_key(int bw, int scale, int nmax, std::size_t n);
extern "C" void suf_sigma_consume_rsqrt_key(int bw, int scale, int extradiv, std::size_t n);
extern "C" void suf_sigma_clear();
extern "C" void suf_sigma_set_keybuf_ptr(std::uint8_t** keybuf_ptr);
extern "C" bool suf_softmax_enabled();
extern "C" bool suf_layernorm_enabled();

extern "C" int suf_sigma_register_operator_spec(const suf::OperatorSpecification* spec);
extern "C" int suf_sigma_register_operator_spec_with_id(int operator_id,
                                                        const suf::OperatorSpecification* spec);
extern "C" int suf_sigma_compiled_operator_capability_flags(int operator_id);
extern "C" bool suf_sigma_compiled_operator_supported(int operator_id);
extern "C" bool suf_sigma_compiled_operator_strict_supported(int operator_id);

extern "C" std::uint64_t* suf_sigma_keygen_compiled_operator(int operator_id,
                                                             int party,
                                                             int bw,
                                                             int scale,
                                                             const std::uint64_t* d_input_mask,
                                                             std::size_t n);

extern "C" std::uint64_t* suf_sigma_eval_compiled_operator(SigmaPeer* peer,
                                                           int operator_id,
                                                           int party,
                                                           int bw,
                                                           int scale,
                                                           const std::uint64_t* d_input_masked,
                                                           std::size_t n,
                                                           Stats* s);

extern "C" SufSigmaCompiledOperatorResult* suf_sigma_keygen_compiled_operator_v2(
    int operator_id,
    int party,
    int bw,
    int scale,
    const std::uint64_t* d_input_mask,
    std::size_t n);

extern "C" SufSigmaCompiledOperatorResult* suf_sigma_eval_compiled_operator_v2(
    SigmaPeer* peer,
    int operator_id,
    int party,
    int bw,
    int scale,
    const std::uint64_t* d_input_masked,
    std::size_t n,
    Stats* s);

extern "C" SufSigmaCompiledOperatorResult* suf_sigma_keygen_compiled_operator_v3(
    int operator_id,
    int party,
    int bw,
    int scale,
    const std::uint64_t* d_input_mask,
    std::size_t n,
    const SufSigmaPostprocessContext* ctx);

extern "C" SufSigmaCompiledOperatorResult* suf_sigma_eval_compiled_operator_v3(
    SigmaPeer* peer,
    int operator_id,
    int party,
    int bw,
    int scale,
    const std::uint64_t* d_input_masked,
    std::size_t n,
    Stats* s,
    const SufSigmaPostprocessContext* ctx);

extern "C" void suf_sigma_free_compiled_operator_result_v2(
    SufSigmaCompiledOperatorResult* result);

extern "C" bool suf_sigma_postprocess_mul_supported();
extern "C" bool suf_sigma_postprocess_and_supported();
extern "C" bool suf_sigma_postprocess_b2a_supported();
extern "C" bool suf_sigma_postprocess_a2b_supported();

extern "C" std::uint64_t* suf_sigma_keygen_postprocess_mul_u64(int party,
                                                               int bw,
                                                               int scale,
                                                               const std::uint64_t* d_lhs_mask,
                                                               const std::uint64_t* d_rhs_mask,
                                                               std::size_t n);

extern "C" std::uint64_t* suf_sigma_eval_postprocess_mul_u64(SigmaPeer* peer,
                                                             int party,
                                                             int bw,
                                                             int scale,
                                                             const std::uint64_t* d_lhs,
                                                             const std::uint64_t* d_rhs,
                                                             std::size_t n,
                                                             Stats* s);

extern "C" std::uint64_t* suf_sigma_keygen_postprocess_and_u64(int party,
                                                               const std::uint64_t* d_lhs_mask,
                                                               const std::uint64_t* d_rhs_mask,
                                                               std::size_t n);

extern "C" std::uint64_t* suf_sigma_eval_postprocess_and_u64(SigmaPeer* peer,
                                                             int party,
                                                             const std::uint64_t* d_lhs,
                                                             const std::uint64_t* d_rhs,
                                                             std::size_t n,
                                                             Stats* s);

extern "C" std::uint64_t* suf_sigma_keygen_postprocess_b2a_u64(int party,
                                                               int bw,
                                                               const std::uint64_t* d_bool_mask,
                                                               std::size_t n);

extern "C" std::uint64_t* suf_sigma_eval_postprocess_b2a_u64(SigmaPeer* peer,
                                                             int party,
                                                             int bw,
                                                             const std::uint64_t* d_bool_open,
                                                             std::size_t n,
                                                             Stats* s);

extern "C" std::uint64_t* suf_sigma_keygen_postprocess_a2b_lsb_u64(int party,
                                                                    const std::uint64_t* d_arith_mask,
                                                                    std::size_t n);

extern "C" std::uint64_t* suf_sigma_eval_postprocess_a2b_lsb_u64(SigmaPeer* peer,
                                                                  int party,
                                                                  const std::uint64_t* d_arith_open,
                                                                  std::size_t n,
                                                                  Stats* s);

extern "C" std::uint64_t* suf_sigma_keygen_postprocess_a2b_bit_u64(int party,
                                                                    int bw,
                                                                    int bit_index,
                                                                    const std::uint64_t* d_arith_mask,
                                                                    std::size_t n);

extern "C" std::uint64_t* suf_sigma_eval_postprocess_a2b_bit_u64(SigmaPeer* peer,
                                                                  int party,
                                                                  int bw,
                                                                  int bit_index,
                                                                  const std::uint64_t* d_arith_open,
                                                                  std::size_t n,
                                                                  Stats* s);

extern "C" std::uint64_t* suf_sigma_keygen_activation(int party,
                                                       int bw,
                                                       int scale,
                                                       bool silu,
                                                       const std::uint64_t* d_input_mask,
                                                       std::size_t n);

extern "C" std::uint64_t* suf_sigma_eval_activation(SigmaPeer* peer,
                                                    int party,
                                                    int bw,
                                                    int scale,
                                                    bool silu,
                                                    const std::uint64_t* d_input_masked,
                                                    std::size_t n,
                                                    Stats* s);

extern "C" std::uint64_t* suf_sigma_keygen_nexp(int party,
                                                int bw,
                                                int scale,
                                                const std::uint64_t* d_input_mask,
                                                std::size_t n);

extern "C" std::uint64_t* suf_sigma_eval_nexp(SigmaPeer* peer,
                                              int party,
                                              int bw,
                                              int scale,
                                              const std::uint64_t* d_input_masked,
                                              std::size_t n,
                                              Stats* s);

extern "C" std::uint64_t* suf_sigma_keygen_inverse(int party,
                                                   int bw,
                                                   int scale,
                                                   int nmax,
                                                   const std::uint16_t* d_input_mask,
                                                   std::size_t n);

extern "C" std::uint64_t* suf_sigma_eval_inverse(SigmaPeer* peer,
                                                 int party,
                                                 int bw,
                                                 int scale,
                                                 int nmax,
                                                 const std::uint16_t* d_input_masked,
                                                 std::size_t n,
                                                 Stats* s);

extern "C" std::uint64_t* suf_sigma_keygen_rsqrt(int party,
                                                 int bw,
                                                 int scale,
                                                 int extradiv,
                                                 const std::uint16_t* d_input_mask,
                                                 std::size_t n);

extern "C" std::uint64_t* suf_sigma_eval_rsqrt(SigmaPeer* peer,
                                               int party,
                                               int bw,
                                               int scale,
                                               int extradiv,
                                               const std::uint16_t* d_input_masked,
                                               std::size_t n,
                                               Stats* s);
