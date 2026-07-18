#pragma once

#include "suf/gpu_backend.hpp"

#ifdef SUF_HAVE_CUDA
#include <cuda_runtime.h>
#endif

namespace suf {

#ifdef SUF_HAVE_CUDA
void launch_eval_preds(const u64* d_in, std::size_t n,
                       const GpuPredicate* d_preds, int num_preds,
                       u8* d_out, cudaStream_t stream);

void launch_eval_poly(const u64* d_in, std::size_t n,
                      const u64* d_cuts, int num_cuts,
                      const u64* d_coeffs, int degree,
                      u64* d_out, cudaStream_t stream);

void launch_eval_poly_from_coeffs(const u64* d_in,
                                  const u64* d_coeffs,
                                  std::size_t n,
                                  int degree,
                                  u64* d_out,
                                  cudaStream_t stream);

void launch_eval_poly_from_coeffs_multi(const u64* d_in,
                                        const u64* d_coeffs,
                                        std::size_t n,
                                        int degree,
                                        int outputs,
                                        std::size_t coeff_words_per_input,
                                        u64* d_out,
                                        cudaStream_t stream);

void launch_eval_helper(const u8* d_pred_bits, std::size_t n,
                        const GpuBoolNode* d_nodes, int num_nodes, int root,
                        u64* d_out, cudaStream_t stream);

void launch_fill_const_pred_bits(const u8* d_const_bits, std::size_t n,
                                 int num_preds, u8* d_out,
                                 cudaStream_t stream);

void launch_scatter_pred_bits(const u8* d_query_bits, std::size_t n,
                              int num_queries, const int* d_query_to_pred,
                              u8* d_out, cudaStream_t stream);
#endif

} // namespace suf
