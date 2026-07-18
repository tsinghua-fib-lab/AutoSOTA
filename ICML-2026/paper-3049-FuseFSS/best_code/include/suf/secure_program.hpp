#pragma once

#include "suf/gpu_backend.hpp"
#include "suf/pfss_plan.hpp"
#include "suf/interval_lut.hpp"
#include "suf/operator_spec.hpp"

#ifdef SUF_HAVE_CUDA
#include <cuda_runtime.h>
#endif

#include <random>
#include <memory>
#include <vector>

namespace suf {

#ifdef SUF_HAVE_CUDA

enum class SecureEvalMode {
  FastPublicX = 0,
  PaperStrictSharedX = 1
};

struct SecureEvalOptions {
  bool mask_aware = false;
  u64 mask_in = 0;
  const std::vector<u64>* mask_vector = nullptr;
  SecureEvalMode mode = SecureEvalMode::FastPublicX;
  const MpcRuntime* runtime = nullptr;
  const std::vector<u64>* kappa_a = nullptr;
  std::size_t kappa_a_count = 0;
  const std::vector<u8>* kappa_b = nullptr;
  std::size_t kappa_b_count = 0;
};

class GpuSecureSufProgram {
public:
  GpuSecureSufProgram(const SUFDescriptor& d, int party, std::uint64_t seed,
                      int in_bits_override = 0,
                      bool mask_aware = false,
                      u64 mask_in = 0);
  GpuSecureSufProgram(const OperatorSpecification& spec, int party, std::uint64_t seed,
                      bool mask_aware = false,
                      u64 mask_in = 0);
  GpuSecureSufProgram(const OperatorSpecification& spec, int party, std::uint64_t seed,
                      const SecureEvalOptions& options);
  ~GpuSecureSufProgram();

  GpuSecureSufProgram(const GpuSecureSufProgram&) = delete;
  GpuSecureSufProgram& operator=(const GpuSecureSufProgram&) = delete;

  void eval(const u64* d_in, std::size_t n,
            u64* d_out_arith, u64* d_out_helpers,
            cudaStream_t stream = nullptr) const;
  void eval_paper_strict_host(const std::vector<u64>& h_masked_input,
                              std::vector<u64>& h_out_arith,
                              std::vector<u64>* h_out_helpers,
                              const MpcRuntime& runtime) const;

  std::size_t num_arithmetic_outputs() const { return arith_outputs_; }
  std::size_t num_predicates() const { return desc_.predicates.size(); }
  std::size_t num_helpers() const { return helper_outputs_; }

private:
  u8* ensure_pred_bits(std::size_t n) const;
  u8* ensure_query_bits(std::size_t n) const;
  u64* ensure_coeffs(std::size_t n) const;
  void eval_paper_strict_device(const u64* d_in, std::size_t n,
                                u64* d_out_arith, u64* d_out_helpers,
                                cudaStream_t stream) const;

  bool has_operator_spec_ = false;
  OperatorSpecification spec_{};
  SUFDescriptor desc_;
  PfssPlan plan_;
  std::unique_ptr<GpuSufProgram> gpu_prog_; // used for poly + helper eval

  int party_ = 0;
  std::uint64_t seed_ = 0;
  int in_bits_ = 64;
  SecureEvalMode eval_mode_ = SecureEvalMode::FastPublicX;
  const MpcRuntime* runtime_ = nullptr;
  bool has_postprocess_ = false;
  bool postprocess_requires_runtime_ = false;
  std::size_t source_arith_outputs_ = 1;
  std::size_t aux_words_ = 0;
  std::size_t helper_outputs_ = 0;

  DpfKeyBatchGpu dpf_gpu_{};
  bool dpf_loaded_ = false;
  mutable u8* d_pred_bits_ = nullptr;
  mutable u8* d_query_bits_ = nullptr;
  mutable std::size_t pred_capacity_ = 0;
  mutable std::size_t query_capacity_ = 0;

	  bool mask_aware_ = false;
	  u64 r_in_ = 0;
	  std::vector<u64> r_in_vector_;
  std::vector<u64> kappa_a_values_;
  std::size_t kappa_a_count_ = 0;
  std::vector<u8> kappa_b_values_;
  std::size_t kappa_b_count_ = 0;
  std::vector<int> pred_to_query_;
  std::vector<u8> const_pred_bits_;
  int* d_query_to_pred_ = nullptr;
  u8* d_const_pred_bits_ = nullptr;
  bool helpers_require_interactive_and_ = false;

  int poly_degree_ = 0;
  std::size_t arith_outputs_ = 1;
  std::size_t coeff_words_per_input_ = 0;
  bool use_interval_lut_ = false;
  IntervalLutKeyV2Gpu interval_key_{};
  bool use_coeff_lut_ = false;
  IntervalLutKeyV2Gpu coeff_key_{};
  mutable u64* d_coeffs_ = nullptr;
  mutable std::size_t coeff_capacity_ = 0;
};

#endif // SUF_HAVE_CUDA

} // namespace suf
