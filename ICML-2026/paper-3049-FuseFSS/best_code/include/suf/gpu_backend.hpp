#pragma once

#include "suf/ir.hpp"

#ifdef SUF_HAVE_CUDA
#include <cuda_runtime.h>
#endif

#include <vector>
#include <cstddef>

namespace suf {

struct GpuPredicate {
  u8 kind = 0;
  u8 f = 0;
  u8 pad0 = 0;
  u8 pad1 = 0;
  u64 param = 0;
  u64 gamma = 0;
  u64 input_add = 0;
};

struct GpuBoolNode {
  u8 kind = 0; // matches BoolNode::Kind
  u8 pad0 = 0;
  u16 pad1 = 0;
  int lhs = -1;
  int rhs = -1;
  int pred_index = -1;
};

struct GpuBoolExpr {
  int root = -1;
  int offset = 0; // into nodes array
  int num_nodes = 0;
};

#ifdef SUF_HAVE_CUDA

class GpuSufProgram {
public:
  explicit GpuSufProgram(const SUFDescriptor& d);
  ~GpuSufProgram();

  GpuSufProgram(const GpuSufProgram&) = delete;
  GpuSufProgram& operator=(const GpuSufProgram&) = delete;

  std::size_t helpers_count() const { return helpers_.size(); }
  std::size_t predicates_count() const { return preds_.size(); }
  int poly_degree() const { return poly_degree_; }

  void eval(const u64* d_in, std::size_t n,
            u64* d_out_arith, u64* d_out_helpers,
            cudaStream_t stream = nullptr) const;

  void eval_poly_only(const u64* d_in, std::size_t n,
                      u64* d_out_arith, cudaStream_t stream = nullptr) const;

  void eval_helpers_from_pred_bits(const u8* d_pred_bits, std::size_t n,
                                   u64* d_out_helpers, cudaStream_t stream = nullptr) const;

private:
  void upload(const SUFDescriptor& d);
  void release();
  u8* ensure_pred_bits(std::size_t n) const;

  u64* d_cuts_ = nullptr;
  u64* d_coeffs_ = nullptr;
  GpuPredicate* d_preds_ = nullptr;
  GpuBoolNode* d_nodes_ = nullptr;
  mutable u8* d_pred_bits_ = nullptr;
  mutable std::size_t pred_capacity_ = 0;

  std::vector<u64> cuts_;
  std::vector<u64> coeffs_;
  std::vector<GpuPredicate> preds_;
  std::vector<GpuBoolExpr> helpers_;
  std::vector<GpuBoolNode> nodes_;

  int poly_degree_ = 0;
};

#endif // SUF_HAVE_CUDA

} // namespace suf
