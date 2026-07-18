#include "suf/secure_program.hpp"
#include "suf/ref_eval.hpp"

#include <cuda_runtime.h>
#include <vector>
#include <iostream>

using namespace suf;

static SUFDescriptor make_desc() {
  SUFDescriptor d;
  d.cuts = {0, 128};
  d.polys.resize(2);
  d.polys[0].coeffs = {0, 1};
  d.polys[1].coeffs = {1, 1};

  Predicate p0; p0.kind = PredKind::LT; p0.param = 128; d.predicates.push_back(p0);
  Predicate p1; p1.kind = PredKind::LTLOW; p1.f = 4; p1.gamma = 7; d.predicates.push_back(p1);
  Predicate p2; p2.kind = PredKind::MSB; d.predicates.push_back(p2);
  Predicate p3; p3.kind = PredKind::MSB_ADD; p3.param = 5; d.predicates.push_back(p3);

  for (int i = 0; i < 4; ++i) {
    BoolExpr e;
    e.nodes.push_back(BoolNode{BoolNode::Kind::PRED, -1, -1, i});
    e.root = 0;
    d.helpers.push_back(e);
  }
  return d;
}

int main() {
  const std::size_t N = 512;
  std::vector<u64> h_in(N);
  for (std::size_t i = 0; i < N; ++i) h_in[i] = static_cast<u64>(i * 13);

  u64* d_in = nullptr;
  u64* d_out0 = nullptr;
  u64* d_out1 = nullptr;
  u64* d_help0 = nullptr;
  u64* d_help1 = nullptr;

  cudaMalloc(&d_in, N * sizeof(u64));
  cudaMemcpy(d_in, h_in.data(), N * sizeof(u64), cudaMemcpyHostToDevice);
  cudaMalloc(&d_out0, N * sizeof(u64));
  cudaMalloc(&d_out1, N * sizeof(u64));
  cudaMalloc(&d_help0, 4 * N * sizeof(u64));
  cudaMalloc(&d_help1, 4 * N * sizeof(u64));

  auto desc = make_desc();
  GpuSecureSufProgram prog0(desc, 0, 123);
  GpuSecureSufProgram prog1(desc, 1, 123);

  prog0.eval(d_in, N, d_out0, d_help0, 0);
  prog1.eval(d_in, N, d_out1, d_help1, 0);
  cudaDeviceSynchronize();

  std::vector<u64> h_out0(N), h_out1(N);
  std::vector<u64> h_help0(4 * N), h_help1(4 * N);
  cudaMemcpy(h_out0.data(), d_out0, N * sizeof(u64), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_out1.data(), d_out1, N * sizeof(u64), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_help0.data(), d_help0, 4 * N * sizeof(u64), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_help1.data(), d_help1, 4 * N * sizeof(u64), cudaMemcpyDeviceToHost);

  for (std::size_t i = 0; i < N; ++i) {
    auto ref = eval_suf_ref(desc, h_in[i]);
    const u64 got = h_out0[i] + h_out1[i];
    if (got != ref.arith) {
      std::cerr << "arith mismatch at " << i << " got " << got << " expected " << ref.arith << "\n";
      return 1;
    }
    for (int h = 0; h < 4; ++h) {
      const u64 bit = (h_help0[h * N + i] ^ h_help1[h * N + i]) & 1ULL;
      if (bit != ref.helpers[h]) {
        std::cerr << "helper mismatch at " << i << " helper=" << h << " got " << bit
                  << " expected " << ref.helpers[h] << "\n";
        return 1;
      }
    }
  }

  cudaFree(d_in);
  cudaFree(d_out0);
  cudaFree(d_out1);
  cudaFree(d_help0);
  cudaFree(d_help1);

  std::cout << "ok\n";
  return 0;
}
