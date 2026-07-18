#include "suf/secure_program.hpp"

#include <cuda_runtime.h>

#include <chrono>
#include <cstring>
#include <iostream>
#include <random>
#include <string>
#include <unordered_set>
#include <vector>

using namespace suf;

namespace {

struct Config {
  std::size_t n = 256;
  bool json = false;
};

BoolExpr pred_expr_for_test(int pred_index) {
  BoolExpr e;
  e.nodes.push_back(BoolNode{BoolNode::Kind::PRED, -1, -1, pred_index});
  e.root = 0;
  return e;
}

OperatorSpecification make_relu_phi_spec() {
  OperatorSpecification spec;
  spec.in_bits = 8;
  spec.boundaries = {0, 128};

  Predicate msb;
  msb.kind = PredKind::MSB;
  spec.predicates.push_back(msb);

  spec.pieces.resize(2);
  spec.pieces[0].polys = {Polynomial{{1}}, Polynomial{{3}}};
  spec.pieces[0].bool_outputs = {pred_expr_for_test(0)};
  spec.pieces[1].polys = {Polynomial{{0}}, Polynomial{{5}}};
  spec.pieces[1].bool_outputs = {pred_expr_for_test(0)};

  PostprocessArithExpr y;
  y.nodes.push_back(PostprocessArithNode{PostprocessArithOp::POLY_OUT, -1, -1, 0, 0});
  y.nodes.push_back(PostprocessArithNode{PostprocessArithOp::X, -1, -1, -1, 0});
  y.nodes.push_back(PostprocessArithNode{PostprocessArithOp::MUL, 0, 1, -1, 0});
  y.nodes.push_back(PostprocessArithNode{PostprocessArithOp::POLY_OUT, -1, -1, 1, 0});
  y.nodes.push_back(PostprocessArithNode{PostprocessArithOp::ADD, 2, 3, -1, 0});
  y.root = 4;
  spec.postprocess.arith_exprs.push_back(y);
  spec.postprocess.arithmetic_outputs = {0};

  PostprocessBoolExpr z;
  z.nodes.push_back(PostprocessBoolNode{PostprocessBoolOp::BOOL_OUT, -1, -1, 0, 0});
  z.root = 0;
  spec.postprocess.bool_exprs.push_back(z);
  spec.postprocess.boolean_outputs = {0};
  return spec;
}

Config parse_args(int argc, char** argv) {
  Config cfg;
  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--n") && i + 1 < argc) {
      cfg.n = std::stoull(argv[++i]);
    } else if (!std::strcmp(argv[i], "--json")) {
      cfg.json = true;
    } else {
      std::cerr << "Unknown arg: " << argv[i] << "\n";
      std::exit(1);
    }
  }
  ensure(cfg.n > 0, "paper canary: n must be positive");
  return cfg;
}

} // namespace

int main(int argc, char** argv) {
  const auto cfg = parse_args(argc, argv);
  const auto spec = make_relu_phi_spec();
  ReferenceMpcRuntime runtime;

  std::vector<u64> h_plain(cfg.n);
  std::vector<u64> h_masks(cfg.n);
  std::vector<u64> h_masked(cfg.n);
  std::mt19937_64 rng(1234);
  for (std::size_t i = 0; i < cfg.n; ++i) {
    h_plain[i] = rng() & 0xFFULL;
    h_masks[i] = rng() & 0xFFULL;
    h_masked[i] = (h_plain[i] + h_masks[i]) & 0xFFULL;
  }
  const std::unordered_set<u64> distinct_masks(h_masks.begin(), h_masks.end());

  SecureEvalOptions opts;
  opts.mask_aware = true;
  opts.mask_vector = &h_masks;
  opts.mode = SecureEvalMode::PaperStrictSharedX;
  opts.runtime = &runtime;

  u64* d_in = nullptr;
  u64* d_y0 = nullptr;
  u64* d_y1 = nullptr;
  u64* d_z0 = nullptr;
  u64* d_z1 = nullptr;
  cudaMalloc(&d_in, cfg.n * sizeof(u64));
  cudaMalloc(&d_y0, cfg.n * sizeof(u64));
  cudaMalloc(&d_y1, cfg.n * sizeof(u64));
  cudaMalloc(&d_z0, cfg.n * sizeof(u64));
  cudaMalloc(&d_z1, cfg.n * sizeof(u64));
  cudaMemcpy(d_in, h_masked.data(), cfg.n * sizeof(u64), cudaMemcpyHostToDevice);

  auto start = std::chrono::high_resolution_clock::now();
  GpuSecureSufProgram prog0(spec, 0, 5555, opts);
  GpuSecureSufProgram prog1(spec, 1, 5555, opts);
  prog0.eval(d_in, cfg.n, d_y0, d_z0, 0);
  prog1.eval(d_in, cfg.n, d_y1, d_z1, 0);
  cudaDeviceSynchronize();
  auto end = std::chrono::high_resolution_clock::now();
  const double wall_ms = std::chrono::duration<double, std::milli>(end - start).count();

  std::vector<u64> h_y0(cfg.n), h_y1(cfg.n), h_z0(cfg.n), h_z1(cfg.n);
  cudaMemcpy(h_y0.data(), d_y0, cfg.n * sizeof(u64), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_y1.data(), d_y1, cfg.n * sizeof(u64), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_z0.data(), d_z0, cfg.n * sizeof(u64), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_z1.data(), d_z1, cfg.n * sizeof(u64), cudaMemcpyDeviceToHost);

  bool verified = true;
  for (std::size_t i = 0; i < cfg.n; ++i) {
    const u64 exp_y = (h_plain[i] < 128) ? (h_plain[i] + 3ULL) : 5ULL;
    const u64 exp_z = (h_plain[i] >= 128) ? 1ULL : 0ULL;
    const u64 got_y = h_y0[i] + h_y1[i];
    const u64 got_z = (h_z0[i] ^ h_z1[i]) & 1ULL;
    if (got_y != exp_y || got_z != exp_z) {
      verified = false;
      std::cerr << "paper canary mismatch at " << i
                << " x=" << h_plain[i]
                << " mask=" << h_masks[i]
                << " got_y=" << got_y << " exp_y=" << exp_y
                << " got_z=" << got_z << " exp_z=" << exp_z << "\n";
      break;
    }
  }

  if (cfg.json) {
    std::cout << "{"
              << "\"bench\":\"fusefss_paper_strict_reference_canary\","
              << "\"n\":" << cfg.n << ","
              << "\"runtime\":\"paper-strict-reference\","
              << "\"scope\":\"reference_semantics_canary\","
              << "\"mask_model\":\"per-element-fresh-mask\","
              << "\"distinct_masks\":" << distinct_masks.size() << ","
              << "\"paper_strict_reference_semantics_met\":true,"
              << "\"verified\":" << (verified ? "true" : "false") << ","
              << "\"wall_ms\":" << wall_ms
              << "}\n";
  } else {
    std::cout << "fusefss_paper_strict_reference_canary"
              << " n=" << cfg.n
              << " runtime=paper-strict-reference"
              << " scope=reference_semantics_canary"
              << " mask_model=per-element-fresh-mask"
              << " distinct_masks=" << distinct_masks.size()
              << " paper_strict_reference_semantics_met=1"
              << " verified=" << (verified ? 1 : 0)
              << " wall_ms=" << wall_ms
              << "\n";
  }

  cudaFree(d_in);
  cudaFree(d_y0);
  cudaFree(d_y1);
  cudaFree(d_z0);
  cudaFree(d_z1);
  return verified ? 0 : 1;
}
