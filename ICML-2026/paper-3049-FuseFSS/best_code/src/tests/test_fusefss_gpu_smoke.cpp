#include "suf/gpu_backend.hpp"
#include "suf/ref_eval.hpp"

#include <cuda_runtime.h>
#include <vector>
#include <iostream>

using namespace suf;

static SUFDescriptor make_toy() {
  SUFDescriptor d;
  d.cuts = {0, 128};
  d.polys.resize(2);
  d.polys[0].coeffs = {0, 1}; // f(x)=x
  d.polys[1].coeffs = {1, 1}; // f(x)=x+1

  Predicate p;
  p.kind = PredKind::LT;
  p.param = 128;
  d.predicates.push_back(p);

  BoolExpr e;
  e.nodes.push_back(BoolNode{BoolNode::Kind::PRED, -1, -1, 0});
  e.root = 0;
  d.helpers.push_back(e);
  return d;
}

int main() {
  const std::size_t N = 256;
  std::vector<u64> h_in(N);
  for (std::size_t i = 0; i < N; ++i) h_in[i] = static_cast<u64>(i);

  u64* d_in = nullptr;
  u64* d_out = nullptr;
  u64* d_helpers = nullptr;

  cudaMalloc(&d_in, N * sizeof(u64));
  cudaMalloc(&d_out, N * sizeof(u64));
  cudaMalloc(&d_helpers, N * sizeof(u64));
  cudaMemcpy(d_in, h_in.data(), N * sizeof(u64), cudaMemcpyHostToDevice);

  GpuSufProgram prog(make_toy());
  prog.eval(d_in, N, d_out, d_helpers, 0);
  cudaDeviceSynchronize();

  std::vector<u64> h_out(N);
  std::vector<u64> h_help(N);
  cudaMemcpy(h_out.data(), d_out, N * sizeof(u64), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_help.data(), d_helpers, N * sizeof(u64), cudaMemcpyDeviceToHost);

  for (std::size_t i = 0; i < N; ++i) {
    auto ref = eval_suf_ref(make_toy(), h_in[i]);
    if (h_out[i] != ref.arith || h_help[i] != ref.helpers[0]) {
      std::cerr << "mismatch at " << i << " got arith=" << h_out[i] << " helper=" << h_help[i]
                << " expected arith=" << ref.arith << " helper=" << ref.helpers[0] << "\n";
      return 1;
    }
  }

  cudaFree(d_in);
  cudaFree(d_out);
  cudaFree(d_helpers);

  std::cout << "ok\n";
  return 0;
}
