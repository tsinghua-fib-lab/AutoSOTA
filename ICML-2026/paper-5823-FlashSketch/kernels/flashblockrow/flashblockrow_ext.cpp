#include <torch/extension.h>

#include <vector>

torch::Tensor flashblockrow_forward_cuda(
    torch::Tensor x,
    int64_t k_total,
    int64_t block_size,
    int64_t kappa,
    int64_t s,
    int64_t tc,
    double alpha,
    uint64_t seed);

torch::Tensor flashblockrow_forward(
    torch::Tensor x,
    int64_t k_total,
    int64_t block_size,
    int64_t kappa,
    int64_t s,
    int64_t tc,
    double alpha,
    uint64_t seed) {
  TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
  TORCH_CHECK(x.scalar_type() == torch::kFloat32, "x must be float32");
  TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
  TORCH_CHECK(x.dim() == 2, "x must be 2D");
  TORCH_CHECK(k_total > 0, "k_total must be positive");
  TORCH_CHECK(block_size > 0, "block_size must be positive");
  TORCH_CHECK(kappa > 0, "kappa must be positive");
  TORCH_CHECK(s > 0, "s must be positive");
  TORCH_CHECK(tc > 0, "tc must be positive");
  TORCH_CHECK(tc <= 32, "tc must be <= 32");
  TORCH_CHECK(block_size <= 1024, "block_size must be <= 1024");
  TORCH_CHECK(alpha > 0.0, "alpha must be positive");

  const auto d_total = static_cast<int64_t>(x.size(0));
  TORCH_CHECK(d_total % block_size == 0, "d_total must be divisible by block_size");
  TORCH_CHECK(k_total % block_size == 0, "k_total must be divisible by block_size");
  TORCH_CHECK(kappa <= d_total / block_size, "kappa must be <= number of input blocks");

  return flashblockrow_forward_cuda(x, k_total, block_size, kappa, s, tc, alpha, seed);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &flashblockrow_forward, "FlashBlockRow forward (CUDA)");
}
