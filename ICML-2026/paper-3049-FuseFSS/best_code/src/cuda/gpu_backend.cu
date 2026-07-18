#include "suf/gpu_backend.hpp"
#include "suf/gpu_kernels.hpp"
#include "suf/validate.hpp"

#ifdef SUF_HAVE_CUDA

#include <cuda_runtime.h>
#include <algorithm>

namespace suf {

namespace {
void check_cuda(cudaError_t err, const char* msg) {
  if (err != cudaSuccess) {
    std::string detail = std::string(msg) + ": " + cudaGetErrorString(err);
    fail(detail.c_str());
  }
}

GpuPredicate to_gpu_pred(const Predicate& p) {
  GpuPredicate out;
  out.kind = static_cast<u8>(p.kind);
  out.f = p.f;
  out.param = p.param;
  out.gamma = p.gamma;
  out.input_add = p.input_add;
  return out;
}
}

GpuSufProgram::GpuSufProgram(const SUFDescriptor& d) {
  validate_suf(d);
  upload(d);
}

GpuSufProgram::~GpuSufProgram() {
  release();
}

void GpuSufProgram::upload(const SUFDescriptor& d) {
  cuts_ = d.cuts;
  poly_degree_ = 0;
  for (const auto& p : d.polys) {
    poly_degree_ = std::max<int>(poly_degree_, static_cast<int>(p.coeffs.size()) - 1);
  }
  const int stride = poly_degree_ + 1;
  coeffs_.assign(cuts_.size() * stride, 0);
  for (std::size_t i = 0; i < d.polys.size(); ++i) {
    for (std::size_t k = 0; k < d.polys[i].coeffs.size(); ++k) {
      coeffs_[i * stride + k] = d.polys[i].coeffs[k];
    }
  }

  preds_.resize(d.predicates.size());
  for (std::size_t i = 0; i < d.predicates.size(); ++i) preds_[i] = to_gpu_pred(d.predicates[i]);

  helpers_.clear();
  nodes_.clear();
  helpers_.reserve(d.helpers.size());
  for (const auto& h : d.helpers) {
    GpuBoolExpr he;
    he.offset = static_cast<int>(nodes_.size());
    he.num_nodes = static_cast<int>(h.nodes.size());
    he.root = h.root;
    for (const auto& n : h.nodes) {
      GpuBoolNode gn;
      gn.kind = static_cast<u8>(n.kind);
      gn.lhs = n.lhs;
      gn.rhs = n.rhs;
      gn.pred_index = n.pred_index;
      nodes_.push_back(gn);
    }
    helpers_.push_back(he);
  }

  check_cuda(cudaMalloc(&d_cuts_, cuts_.size() * sizeof(u64)), "cudaMalloc cuts failed");
  check_cuda(cudaMalloc(&d_coeffs_, coeffs_.size() * sizeof(u64)), "cudaMalloc coeffs failed");
  if (!preds_.empty()) {
    check_cuda(cudaMalloc(&d_preds_, preds_.size() * sizeof(GpuPredicate)), "cudaMalloc preds failed");
  }
  if (!nodes_.empty()) {
    check_cuda(cudaMalloc(&d_nodes_, nodes_.size() * sizeof(GpuBoolNode)), "cudaMalloc nodes failed");
  }

  check_cuda(cudaMemcpy(d_cuts_, cuts_.data(), cuts_.size() * sizeof(u64), cudaMemcpyHostToDevice),
             "cudaMemcpy cuts failed");
  check_cuda(cudaMemcpy(d_coeffs_, coeffs_.data(), coeffs_.size() * sizeof(u64), cudaMemcpyHostToDevice),
             "cudaMemcpy coeffs failed");
  if (!preds_.empty()) {
    check_cuda(cudaMemcpy(d_preds_, preds_.data(), preds_.size() * sizeof(GpuPredicate), cudaMemcpyHostToDevice),
               "cudaMemcpy preds failed");
  }
  if (!nodes_.empty()) {
    check_cuda(cudaMemcpy(d_nodes_, nodes_.data(), nodes_.size() * sizeof(GpuBoolNode), cudaMemcpyHostToDevice),
               "cudaMemcpy nodes failed");
  }
}

void GpuSufProgram::release() {
  if (d_cuts_) cudaFree(d_cuts_);
  if (d_coeffs_) cudaFree(d_coeffs_);
  if (d_preds_) cudaFree(d_preds_);
  if (d_nodes_) cudaFree(d_nodes_);
  if (d_pred_bits_) cudaFree(d_pred_bits_);
  d_cuts_ = nullptr;
  d_coeffs_ = nullptr;
  d_preds_ = nullptr;
  d_nodes_ = nullptr;
  d_pred_bits_ = nullptr;
  pred_capacity_ = 0;
}

u8* GpuSufProgram::ensure_pred_bits(std::size_t n) const {
  const std::size_t needed = preds_.size() * n;
  if (needed == 0) return nullptr;
  if (needed > pred_capacity_) {
    if (d_pred_bits_) cudaFree(d_pred_bits_);
    check_cuda(cudaMalloc(&d_pred_bits_, needed * sizeof(u8)), "cudaMalloc pred bits failed");
    pred_capacity_ = needed;
  }
  return d_pred_bits_;
}

void GpuSufProgram::eval(const u64* d_in, std::size_t n,
                         u64* d_out_arith, u64* d_out_helpers,
                         cudaStream_t stream) const {
  u8* d_pred_bits = ensure_pred_bits(n);
  if (d_pred_bits) {
    launch_eval_preds(d_in, n, d_preds_, static_cast<int>(preds_.size()), d_pred_bits, stream);
  }

  launch_eval_poly(d_in, n, d_cuts_, static_cast<int>(cuts_.size()), d_coeffs_, poly_degree_, d_out_arith, stream);

  if (!helpers_.empty() && d_out_helpers) {
    for (std::size_t h = 0; h < helpers_.size(); ++h) {
      const auto& he = helpers_[h];
      const GpuBoolNode* nodes = d_nodes_ + he.offset;
      u64* out_ptr = d_out_helpers + h * n;
      launch_eval_helper(d_pred_bits, n, nodes, he.num_nodes, he.root, out_ptr, stream);
    }
  }

}

void GpuSufProgram::eval_poly_only(const u64* d_in, std::size_t n,
                                   u64* d_out_arith, cudaStream_t stream) const {
  launch_eval_poly(d_in, n, d_cuts_, static_cast<int>(cuts_.size()), d_coeffs_, poly_degree_, d_out_arith, stream);
}

void GpuSufProgram::eval_helpers_from_pred_bits(const u8* d_pred_bits, std::size_t n,
                                                u64* d_out_helpers, cudaStream_t stream) const {
  if (!helpers_.empty() && d_out_helpers) {
    for (std::size_t h = 0; h < helpers_.size(); ++h) {
      const auto& he = helpers_[h];
      const GpuBoolNode* nodes = d_nodes_ + he.offset;
      u64* out_ptr = d_out_helpers + h * n;
      launch_eval_helper(d_pred_bits, n, nodes, he.num_nodes, he.root, out_ptr, stream);
    }
  }
}

} // namespace suf

#endif // SUF_HAVE_CUDA
