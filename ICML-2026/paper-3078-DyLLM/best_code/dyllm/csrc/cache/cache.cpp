#include <torch/extension.h>

at::Tensor get_seqs_cuda(const at::Tensor& cache, const at::Tensor& seq_starts, const at::Tensor& cu_seqlens,
                         const at::Tensor& seq_ids, const int64_t total_seqlen);

at::Tensor get_block_cuda(const at::Tensor& cache, const at::Tensor& seq_starts, const at::Tensor& cu_seqlens,
                          const at::Tensor& cu_seqlens_q, const at::Tensor& seq_ids, const int64_t total_seqlen);

void reset_full_cuda(at::Tensor& cache, const at::Tensor& in, const at::Tensor& seq_starts,
                     const at::Tensor& cu_seqlens, const at::Tensor& seq_ids, const int64_t total_seqlen);

void reset_block_cuda(at::Tensor& cache, const at::Tensor& in, const at::Tensor& seq_starts,
                      const at::Tensor& cu_seqlens, const at::Tensor& cu_seqlens_q, const at::Tensor& seq_ids,
                      const int64_t total_seqlen);

void scatter_update_cuda(at::Tensor& cache, const at::Tensor& new_vals, const at::Tensor& row_idx,
                         const at::Tensor& cu_seqlens, const at::Tensor& seq_starts, const at::Tensor& seq_ids);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("get_seqs_cuda", &get_seqs_cuda, "Gather cached sequences (CUDA)");
  m.def("get_block_cuda", &get_block_cuda, "Gather cached blocks (CUDA)");
  m.def("reset_full_cuda", &reset_full_cuda, "Reset full cached sequences (CUDA)");
  m.def("reset_block_cuda", &reset_block_cuda, "Reset block cached sequences (CUDA)");
  m.def("scatter_update_cuda", &scatter_update_cuda, "Scatter update cached sequences (CUDA)");
}
