#include "suf/secure_program.hpp"
#include "suf/ref_eval.hpp"
#include "suf/pfss_plan.hpp"
#include "suf/interval_lut.hpp"
#include "suf/masked_compile.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <string>
#include <vector>

using namespace suf;

struct SigmoidConfig {
  int in_bits = 16;
  int bw_out = 16;
  int scale_in = 12;
  int scale_out = 16;
  int intervals = 256;
  int degree = 1; // 0 for LUT-only, 1 for linear
  std::size_t n = 1 << 18;
  int iters = 50;
  int warmup = 5;
  bool mask_aware = false;
  u64 mask_in = 0;
  bool verify = true;
  bool json = true;
  int invariance = 0;
};

static void usage() {
  std::cerr << "bench_sigmoid [--in-bits N] [--bw-out N] [--scale-in N] [--scale-out N]\n";
  std::cerr << "  [--intervals N] [--degree 0|1] [--n N] [--iters N] [--warmup N]\n";
  std::cerr << "  [--mask-aware 0|1] [--mask M] [--verify 0|1] [--json 0|1] [--invariance K]\n";
}

static SigmoidConfig parse_args(int argc, char** argv) {
  SigmoidConfig cfg;
  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--in-bits") && i + 1 < argc) cfg.in_bits = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--bw-out") && i + 1 < argc) cfg.bw_out = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--scale-in") && i + 1 < argc) cfg.scale_in = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--scale-out") && i + 1 < argc) cfg.scale_out = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--intervals") && i + 1 < argc) cfg.intervals = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--degree") && i + 1 < argc) cfg.degree = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--n") && i + 1 < argc) cfg.n = std::stoull(argv[++i]);
    else if (!std::strcmp(argv[i], "--iters") && i + 1 < argc) cfg.iters = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--warmup") && i + 1 < argc) cfg.warmup = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--mask-aware") && i + 1 < argc) cfg.mask_aware = std::atoi(argv[++i]) != 0;
    else if (!std::strcmp(argv[i], "--mask") && i + 1 < argc) cfg.mask_in = std::stoull(argv[++i]);
    else if (!std::strcmp(argv[i], "--verify") && i + 1 < argc) cfg.verify = std::atoi(argv[++i]) != 0;
    else if (!std::strcmp(argv[i], "--json") && i + 1 < argc) cfg.json = std::atoi(argv[++i]) != 0;
    else if (!std::strcmp(argv[i], "--invariance") && i + 1 < argc) cfg.invariance = std::atoi(argv[++i]);
    else {
      std::cerr << "Unknown arg: " << argv[i] << "\n";
      usage();
      std::exit(1);
    }
  }
  return cfg;
}

static inline u64 mask_for_bits(int bits) {
  if (bits >= 64) return ~0ULL;
  return (1ULL << bits) - 1ULL;
}

static inline long double sigmoid(long double x) {
  return 1.0L / (1.0L + std::exp(-x));
}

static inline long double to_real(u64 x, int in_bits, int scale_in) {
  const u64 sign_bit = 1ULL << (in_bits - 1);
  const u64 domain = (1ULL << in_bits);
  long double signed_x = (x & sign_bit) ? static_cast<long double>(static_cast<long long>(x) - static_cast<long long>(domain))
                                        : static_cast<long double>(x);
  return signed_x / static_cast<long double>(1ULL << scale_in);
}

static SUFDescriptor build_sigmoid_desc(const SigmoidConfig& cfg) {
  SUFDescriptor d;
  d.cuts.resize(cfg.intervals);
  const u64 domain = (1ULL << cfg.in_bits);
  const u64 step = domain / static_cast<u64>(cfg.intervals);
  for (int i = 0; i < cfg.intervals; ++i) {
    d.cuts[static_cast<std::size_t>(i)] = static_cast<u64>(i) * step;
  }
  d.polys.resize(cfg.intervals);
  const u64 mask = mask_for_bits(cfg.bw_out);

  for (int i = 0; i < cfg.intervals; ++i) {
    const u64 x0 = d.cuts[static_cast<std::size_t>(i)];
    const u64 x1 = (x0 + step) & mask_for_bits(cfg.in_bits);
    const long double x0_real = to_real(x0, cfg.in_bits, cfg.scale_in);
    const long double x1_real = to_real(x1, cfg.in_bits, cfg.scale_in);
    const long double y0 = sigmoid(x0_real);
    const long double y1 = sigmoid(x1_real);

    if (cfg.degree <= 0) {
      const u64 mid = (x0 + step / 2) & mask_for_bits(cfg.in_bits);
      const long double mid_real = to_real(mid, cfg.in_bits, cfg.scale_in);
      const long double y_mid = sigmoid(mid_real);
      const long double y_scaled = y_mid * static_cast<long double>(1ULL << cfg.scale_out);
      const std::int64_t y_fixed = llroundl(y_scaled);
      d.polys[i].coeffs = { static_cast<u64>(y_fixed) & mask };
    } else {
      long double denom = (x1_real - x0_real);
      long double m = (denom == 0.0L) ? 0.0L : (y1 - y0) / denom;
      long double c = y0 - m * x0_real;
      const long double scale_y = static_cast<long double>(1ULL << cfg.scale_out);
      const long double scale_x = static_cast<long double>(1ULL << cfg.scale_in);
      const std::int64_t c0 = llroundl(c * scale_y);
      const std::int64_t c1 = llroundl(m * scale_y / scale_x);
      d.polys[i].coeffs = { static_cast<u64>(c0) & mask, static_cast<u64>(c1) & mask };
    }
  }
  return d;
}

static std::size_t bytes_dpf_batch(const DpfKeyBatch& batch) {
  return batch.keys.size() * sizeof(DpfKeyPacked) + batch.scw.size() * sizeof(Seed);
}

static std::size_t bytes_dcf_batch(const DcfKeyBatch& batch) {
  return batch.keys.size() * sizeof(DcfKeyPacked) + batch.scw.size() * sizeof(Seed)
       + batch.vcw.size() * sizeof(u64);
}

static std::size_t bytes_interval_lut(const IntervalLutKeyV2& key) {
  return sizeof(IntervalLutHeaderV2)
       + key.base_share.size() * sizeof(u64)
       + key.dmpf.deltas.size() * sizeof(u64)
       + bytes_dcf_batch(key.dmpf.dcf_batch);
}

struct KeyStats {
  std::size_t pred_bytes = 0;
  std::size_t lut_bytes = 0;
  double pred_ms = 0.0;
  double lut_ms = 0.0;
  double keygen_ms = 0.0;
};

static KeyStats measure_keygen(const SUFDescriptor& desc, int in_bits, int party) {
  KeyStats stats;
  std::mt19937_64 rng(1234);
  auto start = std::chrono::high_resolution_clock::now();

  auto plan = compile_pfss_plan(desc);
  auto pred_start = std::chrono::high_resolution_clock::now();
  auto dpf_batch = build_dpf_batch(plan, party, rng);
  auto pred_end = std::chrono::high_resolution_clock::now();

  std::vector<std::vector<u64>> payloads(desc.polys.size());
  for (std::size_t i = 0; i < desc.polys.size(); ++i) {
    payloads[i] = desc.polys[i].coeffs;
  }
  auto lut_start = pred_end;
  auto lut_key = gen_interval_lut_v2(desc.cuts, payloads, in_bits, party, rng);
  auto end = std::chrono::high_resolution_clock::now();

  stats.pred_ms = std::chrono::duration<double, std::milli>(pred_end - pred_start).count();
  stats.lut_ms = std::chrono::duration<double, std::milli>(end - lut_start).count();
  stats.keygen_ms = std::chrono::duration<double, std::milli>(end - start).count();
  stats.pred_bytes = bytes_dpf_batch(dpf_batch);
  stats.lut_bytes = bytes_interval_lut(lut_key);
  return stats;
}

static void run_invariance(const SigmoidConfig& cfg, const SUFDescriptor& base) {
  if (cfg.invariance <= 0) return;
  std::mt19937_64 rng(42);
  std::size_t ref_pred = 0;
  std::size_t ref_lut = 0;
  int ref_cuts = -1;
  int ref_queries = -1;
  bool ok = true;

  for (int i = 0; i < cfg.invariance; ++i) {
    const u64 mask = rng();
    auto inst = compile_masked_gate_instance(base, cfg.in_bits, mask, 0, rng);
    auto plan = compile_pfss_plan(inst.desc);
    KeyStats stats = measure_keygen(inst.desc, cfg.in_bits, 0);
    if (i == 0) {
      ref_pred = stats.pred_bytes;
      ref_lut = stats.lut_bytes;
      ref_cuts = static_cast<int>(inst.desc.cuts.size());
      ref_queries = static_cast<int>(plan.queries.size());
    } else {
      if (stats.pred_bytes != ref_pred || stats.lut_bytes != ref_lut ||
          static_cast<int>(inst.desc.cuts.size()) != ref_cuts ||
          static_cast<int>(plan.queries.size()) != ref_queries) {
        ok = false;
      }
    }
  }
  if (!cfg.json) {
    std::cout << "mask-shape invariance: " << (ok ? "ok" : "mismatch") << "\n";
  }
}

int main(int argc, char** argv) {
  auto cfg = parse_args(argc, argv);
  if (cfg.in_bits <= 0 || cfg.in_bits >= 63) {
    std::cerr << "in-bits must be in [1, 62] for this benchmark.\n";
    return 1;
  }
  if ((cfg.intervals & (cfg.intervals - 1)) != 0) {
    std::cerr << "intervals should be power-of-two for signed domain alignment.\n";
    return 1;
  }

  auto desc = build_sigmoid_desc(cfg);
  run_invariance(cfg, desc);

  // Key size stats (party 0).
  KeyStats key_stats = measure_keygen(desc, cfg.in_bits, 0);
  const std::size_t key_bytes = key_stats.pred_bytes + key_stats.lut_bytes;

  // Build input.
  std::vector<u64> h_in(cfg.n);
  std::mt19937_64 rng(123);
  const u64 mask = mask_for_bits(cfg.in_bits);
  for (std::size_t i = 0; i < cfg.n; ++i) {
    h_in[i] = rng() & mask;
    if (cfg.mask_aware) {
      h_in[i] = (h_in[i] + cfg.mask_in) & mask;
    }
  }

  u64* d_in = nullptr;
  u64* d_out0 = nullptr;
  u64* d_out1 = nullptr;
  cudaMalloc(&d_in, cfg.n * sizeof(u64));
  cudaMalloc(&d_out0, cfg.n * sizeof(u64));
  cudaMemcpy(d_in, h_in.data(), cfg.n * sizeof(u64), cudaMemcpyHostToDevice);

  GpuSecureSufProgram prog0(desc, 0, 1234, cfg.in_bits, cfg.mask_aware, cfg.mask_in);

  // warmup
  for (int i = 0; i < cfg.warmup; ++i) {
    prog0.eval(d_in, cfg.n, d_out0, nullptr, 0);
  }
  cudaDeviceSynchronize();

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);
  for (int i = 0; i < cfg.iters; ++i) {
    prog0.eval(d_in, cfg.n, d_out0, nullptr, 0);
  }
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float ms = 0.0f;
  cudaEventElapsedTime(&ms, start, stop);
  const double avg_ms = ms / cfg.iters;

  double max_abs = 0.0;
  double mean_abs = 0.0;
  double cos_num = 0.0;
  double cos_den0 = 0.0;
  double cos_den1 = 0.0;
  if (cfg.verify) {
    const u64 out_mask = mask_for_bits(cfg.bw_out);
    for (std::size_t i = 0; i < cfg.n; ++i) {
      u64 x = h_in[i];
      if (cfg.mask_aware) {
        x = (x - cfg.mask_in) & mask;
      }
      long double x_real = to_real(x, cfg.in_bits, cfg.scale_in);
      long double y_ref = sigmoid(x_real);
      long double y_scaled = y_ref * static_cast<long double>(1ULL << cfg.scale_out);
      long double y_fixed = std::llround(y_scaled);
      u64 approx = eval_suf_ref(desc, x).arith & out_mask;
      long double got = static_cast<long double>(approx);
      long double err = std::fabs(got - y_fixed);
      max_abs = std::max(max_abs, static_cast<double>(err));
      mean_abs += static_cast<double>(err);
      cos_num += static_cast<double>(got * y_fixed);
      cos_den0 += static_cast<double>(got * got);
      cos_den1 += static_cast<double>(y_fixed * y_fixed);
    }
    mean_abs /= static_cast<double>(cfg.n);
  }

  if (cfg.json) {
    std::cout << "{\"gate\":\"sigmoid\",\"mode\":\"" << (cfg.degree <= 0 ? "lut" : "poly")
              << "\",\"intervals\":" << cfg.intervals
              << ",\"degree\":" << cfg.degree
              << ",\"in_bits\":" << cfg.in_bits
              << ",\"bw_out\":" << cfg.bw_out
              << ",\"scale_in\":" << cfg.scale_in
              << ",\"scale_out\":" << cfg.scale_out
              << ",\"n\":" << cfg.n
              << ",\"avg_ms\":" << avg_ms
              << ",\"key_bytes\":" << key_bytes
              << ",\"pred_bytes\":" << key_stats.pred_bytes
              << ",\"lut_bytes\":" << key_stats.lut_bytes
              << ",\"max_abs_err\":" << max_abs
              << ",\"mean_abs_err\":" << mean_abs
              << ",\"cos_sim\":" << (cos_den0 > 0 && cos_den1 > 0 ? (cos_num / std::sqrt(cos_den0 * cos_den1)) : 0.0)
              << "}\n";
  } else {
    std::cout << "sigmoid " << (cfg.degree <= 0 ? "lut" : "poly") << " intervals=" << cfg.intervals
              << " degree=" << cfg.degree << " in_bits=" << cfg.in_bits
              << " scale_in=" << cfg.scale_in << " scale_out=" << cfg.scale_out << "\n";
    std::cout << "avg_ms=" << avg_ms << " key_bytes=" << key_bytes
              << " pred_bytes=" << key_stats.pred_bytes << " lut_bytes=" << key_stats.lut_bytes << "\n";
    if (cfg.verify) {
      std::cout << "max_abs_err=" << max_abs << " mean_abs_err=" << mean_abs << "\n";
    }
  }

  cudaFree(d_in);
  cudaFree(d_out0);
  if (d_out1) cudaFree(d_out1);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  return 0;
}
