#include <torch/extension.h>

void fused_add_rms_norm(torch::Tensor& input, torch::Tensor& residual, torch::Tensor& weight, double epsilon);

void rms_norm(torch::Tensor& out, torch::Tensor& input, torch::Tensor& weight, double epsilon);

void rotary_embedding(torch::Tensor& positions, torch::Tensor& query, std::optional<torch::Tensor> key,
                      int64_t head_size, torch::Tensor& cos_sin_cache, bool is_neox);

void rotary_embedding_separate(torch::Tensor& positions_q, torch::Tensor& query,
                               std::optional<torch::Tensor> positions_k, std::optional<torch::Tensor> key,
                               int64_t head_size, torch::Tensor& cos_sin_cache, bool is_neox);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("rotary_embedding", &rotary_embedding, "Rotary Positional Embedding (CUDA)");
  m.def("rotary_embedding_separate", &rotary_embedding_separate,
        "Rotary Positional Embedding for Q and K separately (CUDA)");
  m.def("rms_norm", &rms_norm, "RMS norm (CUDA)");
  m.def("fused_add_rms_norm", &fused_add_rms_norm, "Fused Add + RMS norm (CUDA)");
}