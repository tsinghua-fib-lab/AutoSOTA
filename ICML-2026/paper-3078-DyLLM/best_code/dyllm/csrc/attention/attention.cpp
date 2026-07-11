#include <torch/extension.h>

template <typename scalar_t>
void attention_sparse_varlen(const scalar_t* q, const scalar_t* k, const scalar_t* v, const scalar_t* c,
                             const scalar_t* o_sal, scalar_t* o, const int B, const int H, const int H_kv,
                             const int* cu_seqlens_q, const int* cu_seqlens_k, const int max_seqlen_q,
                             const int max_seqlen_k, const int total_seqlen_q, const int num_salient,
                             const int* cu_salientlens, const int* idx_salient_row_k, const int* idx_salient_row_q,
                             bool* block_mask, int* idx_map_k, int* idx_map_q, float* cosine_stats, bool* cosine_out,
                             float threshold, int dim);

at::Tensor attention_sparse_varlen_cuda(const at::Tensor& Q, // [T_q, H, D], bf16, CUDA
                                        const at::Tensor& K, // [T_k, H, D], bf16, CUDA
                                        const at::Tensor& V, // [T_k, H, D], bf16, CUDA
                                        const at::Tensor& C, const at::Tensor& o_sal,
                                        const at::Tensor& cu_seqlens_q, // [B+1], CUDA
                                        const at::Tensor& cu_seqlens_k, // [B+1], CUDA
                                        const int max_seqlen_q, const int max_seqlen_k, const int total_seqlen,
                                        const at::Tensor& cu_salientlens, const at::Tensor& idx_salient_row,
                                        const at::Tensor& cosine_stats,
                                        const at::Tensor& cosine_out, // [total_seqlen], bool mask
                                        float threshold = 0.0f, bool is_q_pruned = false,
                                        const at::Tensor& idx_salient_row_k = at::Tensor()) {
  TORCH_CHECK(Q.is_cuda() && K.is_cuda() && V.is_cuda(), "Q, K, V must be CUDA");
  TORCH_CHECK(Q.scalar_type() == at::kBFloat16 && K.scalar_type() == at::kBFloat16 && V.scalar_type() == at::kBFloat16,
              "Q, K, V must be bfloat16");
  TORCH_CHECK(Q.dim() == 3 && K.dim() == 3 && V.dim() == 3, "Q, K, V must be [T, H, D]");

  auto Oc = at::empty_like(Q);

  const int* cu_q_ptr = cu_seqlens_q.data_ptr<int>();
  const int* cu_k_ptr = cu_seqlens_k.data_ptr<int>();
  const int* cu_sal_ptr = cu_salientlens.data_ptr<int>();
  const int* idx_sal_ptr = (is_q_pruned) ? idx_salient_row_k.data_ptr<int>() : idx_salient_row.data_ptr<int>();
  const int* idx_sal_q_ptr = (is_q_pruned) ? idx_salient_row.data_ptr<int>() : idx_salient_row.data_ptr<int>();

  auto cosine_stats_ptr = cosine_stats.data_ptr<float>();
  auto cosine_out_ptr = cosine_out.data_ptr<bool>();

  int B = static_cast<int>(cu_seqlens_q.size(0) - 1);
  int H = static_cast<int>(Q.size(1));
  int H_kv = static_cast<int>(K.size(1));
  constexpr int BLOCK_Q = 64; // Must match BLOCK_Q in attention_ops_kernels.cu
  constexpr int BLOCK_K = 64; // Must match BLOCK_KV in attention_ops_kernels.cu
  int block_kv = (max_seqlen_k + BLOCK_K - 1) / BLOCK_K;
  int block_q = (max_seqlen_q + BLOCK_Q - 1) / BLOCK_Q;

  auto opts_gpu = at::TensorOptions().device(at::kCUDA);
  auto block_mask = at::empty({B * block_kv}, opts_gpu.dtype(at::kBool));
  auto idx_map_k = at::full({B * block_kv * BLOCK_K}, -1, opts_gpu.dtype(at::kInt));

  // Always allocate idx_map_q - it's used for O_sal indexing in both pruned and non-pruned cases
  auto idx_map_q = at::full({B * block_q * BLOCK_Q}, -1, opts_gpu.dtype(at::kInt));
  int* idx_map_q_ptr = idx_map_q.data_ptr<int>();

  bool* block_mask_ptr = block_mask.data_ptr<bool>();
  int* idx_map_k_ptr = idx_map_k.data_ptr<int>();
  int num_salient = static_cast<int>(idx_salient_row.numel());

  AT_DISPATCH_SWITCH(Q.scalar_type(), "attention_sparse_varlen", AT_DISPATCH_CASE(at::kBFloat16, [&] {
                       using scalar_t = at::BFloat16;
                       attention_sparse_varlen<scalar_t>(
                           Q.data_ptr<scalar_t>(), K.data_ptr<scalar_t>(), V.data_ptr<scalar_t>(),
                           C.data_ptr<scalar_t>(), o_sal.data_ptr<scalar_t>(), Oc.data_ptr<scalar_t>(), B, H, H_kv,
                           cu_q_ptr, cu_k_ptr, max_seqlen_q, max_seqlen_k, total_seqlen, num_salient, cu_sal_ptr,
                           idx_sal_ptr, idx_sal_q_ptr, block_mask_ptr, idx_map_k_ptr, idx_map_q_ptr, cosine_stats_ptr,
                           cosine_out_ptr, threshold, static_cast<int>(Q.size(2)));
                     }));

  return Oc;
}

void fused_offset_launch(torch::Tensor idxs, torch::Tensor num_idxs, torch::Tensor cu_promptlens,
                         torch::Tensor cu_salientlens, torch::Tensor out_tensor, int max_threads);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("attention_sparse_varlen", &attention_sparse_varlen_cuda, "Sparse Variable Length Attention (CUDA)");
  m.def("fused_offset_launch", &fused_offset_launch, "Fused Offset Launch");
}
