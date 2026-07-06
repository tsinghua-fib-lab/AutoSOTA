#include <torch/extension.h>

#include <numeric>
#include <vector>

torch::Tensor flashsketch_forward_cuda(
    torch::Tensor x,
    int64_t k_total,
    int64_t block_rows,
    int64_t kappa,
    int64_t s,
    double scale,
    uint64_t seed,
    int64_t affine_a,
    int64_t affine_b,
    bool skip_zeros);

namespace {

std::vector<int64_t> unique_prime_factors(int64_t value) {
  std::vector<int64_t> factors;
  int64_t n = value;
  for (int64_t p = 2; p * p <= n; ++p) {
    if (n % p != 0) {
      continue;
    }
    factors.push_back(p);
    while (n % p == 0) {
      n /= p;
    }
  }
  if (n > 1) {
    factors.push_back(n);
  }
  return factors;
}

int64_t pick_full_period_a(int64_t modulus) {
  if (modulus <= 1) {
    return 1;
  }
  const auto factors = unique_prime_factors(modulus);
  int64_t base = 1;
  for (int64_t factor : factors) {
    base = std::lcm(base, factor);
  }
  if (modulus % 4 == 0) {
    base = std::lcm(base, static_cast<int64_t>(4));
  }
  int64_t a = (1 + base) % modulus;
  if (a == 0) {
    a = 1;
  }
  return a;
}

int64_t pick_coprime_b(uint64_t seed, int64_t modulus) {
  if (modulus <= 1) {
    return 0;
  }
  int64_t b = static_cast<int64_t>(seed % static_cast<uint64_t>(modulus));
  if (b == 0) {
    b = 1;
  }
  while (std::gcd(b, modulus) != 1) {
    b = (b + 1) % modulus;
    if (b == 0) {
      b = 1;
    }
  }
  return b;
}

}  // namespace

torch::Tensor flashsketch_forward(
    torch::Tensor x,
    int64_t k_total,
    int64_t block_rows,
    int64_t kappa,
    int64_t s,
    double scale,
    uint64_t seed,
    bool skip_zeros) {
  TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
  TORCH_CHECK(x.scalar_type() == torch::kFloat32, "x must be float32");
  TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
  TORCH_CHECK(x.dim() == 2, "x must be 2D");
  TORCH_CHECK(k_total > 0, "k_total must be positive");
  TORCH_CHECK(block_rows > 0, "block_rows must be positive");
  TORCH_CHECK(kappa > 0, "kappa must be positive");
  TORCH_CHECK(s > 0, "s must be positive");
  TORCH_CHECK(scale > 0.0, "scale must be positive");
  const int64_t M = k_total / block_rows;
  TORCH_CHECK(M > 0, "block_rows must be <= k_total");
  const int64_t affine_a = pick_full_period_a(M);
  const int64_t affine_b = pick_coprime_b(seed, M);

  TORCH_CHECK(affine_a > 0, "affine_a must be positive");
  TORCH_CHECK(affine_b >= 0, "affine_b must be non-negative");

  return flashsketch_forward_cuda(
      x,
      k_total,
      block_rows,
      kappa,
      s,
      scale,
      seed,
      affine_a,
      affine_b,
      skip_zeros);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &flashsketch_forward, "FlashSketch SJLT forward (CUDA)");
}
