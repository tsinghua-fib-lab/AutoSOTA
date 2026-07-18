#include "suf/gpu_backend.hpp"
#include "suf/ref_eval.hpp"
#include "suf/secure_program.hpp"

#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <random>
#include <chrono>
#include <cstring>
#include <memory>

using namespace suf;

struct BenchConfig {
  std::size_t n = 1 << 20;
  int iters = 50;
  int intervals = 16;
  int degree = 3;
  int helpers = 4;
  bool verify = false;
  bool secure = false;
  bool mask_aware = false;
  u64 mask_in = 0;
};

static BenchConfig parse_args(int argc, char** argv) {
  BenchConfig cfg;
  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--n") && i + 1 < argc) cfg.n = std::stoull(argv[++i]);
    else if (!std::strcmp(argv[i], "--iters") && i + 1 < argc) cfg.iters = std::stoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--intervals") && i + 1 < argc) cfg.intervals = std::stoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--degree") && i + 1 < argc) cfg.degree = std::stoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--helpers") && i + 1 < argc) cfg.helpers = std::stoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--verify")) cfg.verify = true;
    else if (!std::strcmp(argv[i], "--secure")) cfg.secure = true;
    else if (!std::strcmp(argv[i], "--mask-aware")) cfg.mask_aware = true;
    else if (!std::strcmp(argv[i], "--mask") && i + 1 < argc) cfg.mask_in = std::stoull(argv[++i]);
    else {
      std::cerr << "Unknown arg: " << argv[i] << "\n";
      std::exit(1);
    }
  }
  return cfg;
}

static SUFDescriptor make_bench_desc(const BenchConfig& cfg) {
  SUFDescriptor d;
  d.cuts.resize(cfg.intervals);
  const std::uint64_t domain = (1ULL << 32); // use 32-bit domain inside u64
  const std::uint64_t step = domain / cfg.intervals;
  for (int i = 0; i < cfg.intervals; ++i) {
    d.cuts[i] = static_cast<u64>(i * step);
  }
  d.polys.resize(cfg.intervals);
  for (int i = 0; i < cfg.intervals; ++i) {
    d.polys[i].coeffs.resize(cfg.degree + 1);
    for (int k = 0; k <= cfg.degree; ++k) {
      d.polys[i].coeffs[k] = static_cast<u64>((i + 1) * (k + 1));
    }
  }

  // predicates for helper bits
  for (int i = 0; i < cfg.helpers; ++i) {
    Predicate p;
    p.kind = PredKind::LT;
    p.param = d.cuts[static_cast<std::size_t>(i % cfg.intervals)] + step / 2;
    d.predicates.push_back(p);

    BoolExpr e;
    e.nodes.push_back(BoolNode{BoolNode::Kind::PRED, -1, -1, i});
    e.root = 0;
    d.helpers.push_back(e);
  }
  return d;
}

int main(int argc, char** argv) {
  auto cfg = parse_args(argc, argv);
  auto desc = make_bench_desc(cfg);

  std::vector<u64> h_in(cfg.n);
  std::mt19937_64 rng(1234);
  for (std::size_t i = 0; i < cfg.n; ++i) {
    h_in[i] = rng();
    if (cfg.mask_aware) {
      h_in[i] += cfg.mask_in;
    }
  }

  u64* d_in = nullptr;
  u64* d_out = nullptr;
  u64* d_helpers = nullptr;
  cudaMalloc(&d_in, cfg.n * sizeof(u64));
  cudaMalloc(&d_out, cfg.n * sizeof(u64));
  if (cfg.helpers > 0) cudaMalloc(&d_helpers, cfg.n * cfg.helpers * sizeof(u64));
  cudaMemcpy(d_in, h_in.data(), cfg.n * sizeof(u64), cudaMemcpyHostToDevice);

  std::unique_ptr<GpuSufProgram> prog_clear;
  std::unique_ptr<GpuSecureSufProgram> prog_secure;
  if (cfg.secure) {
    prog_secure = std::make_unique<GpuSecureSufProgram>(desc, 0, 1234, 0, cfg.mask_aware, cfg.mask_in);
  } else {
    prog_clear = std::make_unique<GpuSufProgram>(desc);
  }

  // warmup
  if (cfg.secure) {
    prog_secure->eval(d_in, cfg.n, d_out, d_helpers, 0);
  } else {
    prog_clear->eval(d_in, cfg.n, d_out, d_helpers, 0);
  }
  cudaDeviceSynchronize();

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);
  for (int i = 0; i < cfg.iters; ++i) {
    if (cfg.secure) {
      prog_secure->eval(d_in, cfg.n, d_out, d_helpers, 0);
    } else {
      prog_clear->eval(d_in, cfg.n, d_out, d_helpers, 0);
    }
  }
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float ms = 0.0f;
  cudaEventElapsedTime(&ms, start, stop);
  const double avg_ms = ms / cfg.iters;
  const double elems_per_s = (cfg.n / (avg_ms / 1000.0));

  std::cout << "FuseFSS GPU eval" << (cfg.secure ? " (secure-pfss)" : "") << ": n=" << cfg.n
            << " intervals=" << cfg.intervals << " degree=" << cfg.degree
            << " helpers=" << cfg.helpers << " avg_ms=" << avg_ms << " throughput=" << elems_per_s << " elems/s\n";

  if (cfg.verify) {
    std::vector<u64> h_out0(cfg.n);
    cudaMemcpy(h_out0.data(), d_out, cfg.n * sizeof(u64), cudaMemcpyDeviceToHost);
    std::vector<u64> h_out1;
    if (cfg.secure) {
      u64* d_out1 = nullptr;
      cudaMalloc(&d_out1, cfg.n * sizeof(u64));
      auto prog1 = std::make_unique<GpuSecureSufProgram>(desc, 1, 1234, 0, cfg.mask_aware, cfg.mask_in);
      prog1->eval(d_in, cfg.n, d_out1, nullptr, 0);
      cudaDeviceSynchronize();
      h_out1.resize(cfg.n);
      cudaMemcpy(h_out1.data(), d_out1, cfg.n * sizeof(u64), cudaMemcpyDeviceToHost);
      cudaFree(d_out1);
    }
    for (std::size_t i = 0; i < std::min<std::size_t>(cfg.n, 1024); ++i) {
      u64 x = h_in[i];
      if (cfg.mask_aware) {
        x -= cfg.mask_in;
      }
      auto ref = eval_suf_ref(desc, x);
      u64 got = h_out0[i];
      if (cfg.secure) {
        got += h_out1[i];
      }
      if (got != ref.arith) {
        std::cerr << "verify mismatch at " << i << " got " << got << " expected " << ref.arith << "\n";
        return 1;
      }
    }
    std::cout << "verify ok\n";
  }

  cudaFree(d_in);
  cudaFree(d_out);
  if (d_helpers) cudaFree(d_helpers);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  return 0;
}
