#include "suf/gpu_backend.hpp"
#include "suf/pfss_plan.hpp"
#include "suf/interval_lut.hpp"
#include "suf/secure_program.hpp"
#include "suf/ref_eval.hpp"
#include "suf/masked_compile.hpp"

#include <cuda_runtime.h>
#include <chrono>
#include <cctype>
#include <cstring>
#include <iostream>
#include <random>
#include <string>
#include <vector>

using namespace suf;

struct BenchConfig {
  std::string model;
  std::size_t seq = 128;
  int iters = 20;
  int intervals = -1; // auto
  int degree = 0;
  int helpers = -1; // auto
  bool verify = false;
  bool json = false;
  bool mask_aware = false;
  u64 mask_in = 0;
  bool template_cache = true;
  bool cpu_eval = false;
};

struct ModelSpec {
  const char* name;
  int n_layer;
  int n_embd;
  int intermediate; // for silu
  bool silu;
};

static const ModelSpec kModels[] = {
  {"bert-tiny", 2, 128, 0, false},
  {"bert-base", 12, 768, 0, false},
  {"bert-large", 24, 1024, 0, false},
  {"gpt2", 12, 768, 0, false},
  {"gpt-neo", 24, 2048, 0, false},
  {"gpt-neo-large", 32, 2560, 0, false},
  {"llama7b", 32, 4096, 11008, true},
  {"llama13b", 40, 5120, 13824, true},
  {"llama3-8b", 32, 4096, 14336, true},
};

static int infer_in_bits_from_intervals(int intervals) {
  if (intervals <= 1) return 1;
  int bits = 0;
  std::uint64_t v = static_cast<std::uint64_t>(intervals - 1);
  while (v) {
    ++bits;
    v >>= 1;
  }
  return bits > 0 ? bits : 1;
}

static BenchConfig parse_args(int argc, char** argv) {
  BenchConfig cfg;
  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--model") && i + 1 < argc) cfg.model = argv[++i];
    else if (!std::strcmp(argv[i], "--seq") && i + 1 < argc) cfg.seq = std::stoull(argv[++i]);
    else if (!std::strcmp(argv[i], "--iters") && i + 1 < argc) cfg.iters = std::stoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--intervals") && i + 1 < argc) cfg.intervals = std::stoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--degree") && i + 1 < argc) cfg.degree = std::stoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--helpers") && i + 1 < argc) cfg.helpers = std::stoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--verify")) cfg.verify = true;
    else if (!std::strcmp(argv[i], "--json")) cfg.json = true;
    else if (!std::strcmp(argv[i], "--mask-aware")) cfg.mask_aware = true;
    else if (!std::strcmp(argv[i], "--mask") && i + 1 < argc) cfg.mask_in = std::stoull(argv[++i]);
    else if (!std::strcmp(argv[i], "--no-template-cache")) cfg.template_cache = false;
    else if (!std::strcmp(argv[i], "--cpu-eval")) cfg.cpu_eval = true;
    else {
      std::cerr << "Unknown arg: " << argv[i] << "\n";
      std::exit(1);
    }
  }
  if (cfg.model.empty()) {
    std::cerr << "Must provide --model\n";
    std::exit(1);
  }
  return cfg;
}

static const ModelSpec* find_model(const std::string& name) {
  for (const auto& m : kModels) {
    if (name == m.name) return &m;
  }
  return nullptr;
}

static SUFDescriptor make_bench_desc(const BenchConfig& cfg, int in_bits) {
  SUFDescriptor d;
  d.cuts.resize(cfg.intervals);
  const std::uint64_t domain = (1ULL << in_bits);
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

  for (int i = 0; i < cfg.helpers; ++i) {
    Predicate p;
    p.kind = PredKind::LTLOW;
    p.f = static_cast<u8>(in_bits);
    p.gamma = d.cuts[static_cast<std::size_t>(i % cfg.intervals)] + step / 2;
    d.predicates.push_back(p);

    BoolExpr e;
    e.nodes.push_back(BoolNode{BoolNode::Kind::PRED, -1, -1, i});
    e.root = 0;
    d.helpers.push_back(e);
  }
  return d;
}

static SUFDescriptor make_effective_desc(const SUFDescriptor& base,
                                         int in_bits,
                                         bool mask_aware,
                                         u64 mask_in,
                                         int party) {
  if (!mask_aware) return base;
  std::mt19937_64 rng(1234);
  auto inst = compile_masked_gate_instance(base, in_bits, mask_in, party, rng);
  return inst.desc;
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

static KeyStats measure_keygen(const SUFDescriptor& desc, int party, int in_bits) {
  KeyStats stats;
  std::mt19937_64 rng(1234);
  auto start = std::chrono::high_resolution_clock::now();

  auto plan = compile_pfss_plan(desc);
  auto pred_start = std::chrono::high_resolution_clock::now();
  auto dpf_batch = build_dpf_batch(plan, party, rng);
  auto pred_end = std::chrono::high_resolution_clock::now();

  bool piecewise_constant = true;
  for (const auto& p : desc.polys) {
    if (p.coeffs.size() != 1) {
      piecewise_constant = false;
      break;
    }
  }

  IntervalLutKeyV2 lut_key;
  auto lut_start = pred_end;
  std::vector<std::vector<u64>> payloads(desc.polys.size());
  for (std::size_t i = 0; i < desc.polys.size(); ++i) {
    payloads[i] = desc.polys[i].coeffs;
  }
  lut_start = std::chrono::high_resolution_clock::now();
  lut_key = gen_interval_lut_v2(desc.cuts, payloads, in_bits, party, rng);
  auto end = std::chrono::high_resolution_clock::now();
  stats.pred_ms = std::chrono::duration<double, std::milli>(pred_end - pred_start).count();
  stats.lut_ms = std::chrono::duration<double, std::milli>(end - lut_start).count();
  stats.keygen_ms = std::chrono::duration<double, std::milli>(end - start).count();
  stats.pred_bytes = bytes_dpf_batch(dpf_batch);
  stats.lut_bytes = bytes_interval_lut(lut_key);
  return stats;
}

int main(int argc, char** argv) {
  auto cfg = parse_args(argc, argv);
  const auto* model = find_model(cfg.model);
  if (!model) {
    std::cerr << "Unknown model: " << cfg.model << "\n";
    return 1;
  }
  if (cfg.intervals < 0) {
    cfg.intervals = model->silu ? 1024 : 256;
  }
  if (cfg.helpers < 0) {
    cfg.helpers = 2;
  }
  const int in_bits = infer_in_bits_from_intervals(cfg.intervals);

  const std::size_t gate_elems = model->silu
    ? cfg.seq * static_cast<std::size_t>(model->intermediate)
    : cfg.seq * static_cast<std::size_t>(4 * model->n_embd);
  const int gate_count = model->n_layer;

  auto desc = make_bench_desc(cfg, in_bits);
  auto effective_desc = make_effective_desc(desc, in_bits, cfg.mask_aware, cfg.mask_in, 0);

  KeyStats key_stats = measure_keygen(effective_desc, 0, in_bits);
  const std::size_t per_gate_key_bytes = key_stats.pred_bytes + key_stats.lut_bytes;
  const double per_gate_key_ms = key_stats.keygen_ms;
  const std::size_t total_key_bytes = per_gate_key_bytes * static_cast<std::size_t>(gate_count);
  const double total_key_ms = per_gate_key_ms * gate_count;

  std::vector<u64> h_x(gate_elems);
  std::vector<u64> h_in(gate_elems);
  std::mt19937_64 rng(1234);
  const u64 mask = (in_bits == 64) ? ~0ULL : ((1ULL << in_bits) - 1ULL);
  for (std::size_t i = 0; i < gate_elems; ++i) {
    h_x[i] = rng() & mask;
    h_in[i] = cfg.mask_aware ? ((h_x[i] + cfg.mask_in) & mask) : h_x[i];
  }

  u64* d_in = nullptr;
  u64* d_out = nullptr;
  u64* d_helpers = nullptr;
  cudaMalloc(&d_in, gate_elems * sizeof(u64));
  cudaMalloc(&d_out, gate_elems * sizeof(u64));
  if (cfg.helpers > 0) cudaMalloc(&d_helpers, gate_elems * cfg.helpers * sizeof(u64));
  cudaMemcpy(d_in, h_in.data(), gate_elems * sizeof(u64), cudaMemcpyHostToDevice);

  double init_ms = 0.0;
  double total_init_ms = 0.0;
  double per_gate_eval_ms = 0.0;
  double total_eval_ms = 0.0;

  if (cfg.template_cache) {
    auto init_start = std::chrono::high_resolution_clock::now();
    GpuSecureSufProgram prog(effective_desc, 0, 1234, in_bits, cfg.mask_aware, cfg.mask_in);
    auto init_end = std::chrono::high_resolution_clock::now();
    init_ms = std::chrono::duration<double, std::milli>(init_end - init_start).count();

    prog.eval(d_in, gate_elems, d_out, d_helpers, 0);
    cudaDeviceSynchronize();

    cudaEvent_t start_evt, stop_evt;
    cudaEventCreate(&start_evt);
    cudaEventCreate(&stop_evt);
    cudaEventRecord(start_evt);
    for (int i = 0; i < cfg.iters; ++i) {
      prog.eval(d_in, gate_elems, d_out, d_helpers, 0);
    }
    cudaEventRecord(stop_evt);
    cudaEventSynchronize(stop_evt);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start_evt, stop_evt);
    per_gate_eval_ms = ms / cfg.iters;
    total_eval_ms = per_gate_eval_ms * gate_count;
    cudaEventDestroy(start_evt);
    cudaEventDestroy(stop_evt);
  } else {
    double eval_sum_ms = 0.0;
    for (int g = 0; g < gate_count; ++g) {
      auto init_start = std::chrono::high_resolution_clock::now();
      GpuSecureSufProgram prog(effective_desc, 0, 1234, in_bits, cfg.mask_aware, cfg.mask_in);
      auto init_end = std::chrono::high_resolution_clock::now();
      total_init_ms += std::chrono::duration<double, std::milli>(init_end - init_start).count();

      prog.eval(d_in, gate_elems, d_out, d_helpers, 0);
      cudaDeviceSynchronize();

      cudaEvent_t start_evt, stop_evt;
      cudaEventCreate(&start_evt);
      cudaEventCreate(&stop_evt);
      cudaEventRecord(start_evt);
      for (int i = 0; i < cfg.iters; ++i) {
        prog.eval(d_in, gate_elems, d_out, d_helpers, 0);
      }
      cudaEventRecord(stop_evt);
      cudaEventSynchronize(stop_evt);
      float ms = 0.0f;
      cudaEventElapsedTime(&ms, start_evt, stop_evt);
      eval_sum_ms += (ms / cfg.iters);
      cudaEventDestroy(start_evt);
      cudaEventDestroy(stop_evt);
    }
    per_gate_eval_ms = eval_sum_ms / gate_count;
    total_eval_ms = eval_sum_ms;
  }

  double cpu_per_gate_ms = 0.0;
  double cpu_total_ms = 0.0;
  if (cfg.cpu_eval) {
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < cfg.iters; ++i) {
      for (std::size_t j = 0; j < gate_elems; ++j) {
        const u64 x = cfg.mask_aware ? h_x[j] : h_in[j];
        auto ref = eval_suf_ref(effective_desc, x);
        (void)ref;
      }
    }
    auto end = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(end - start).count();
    cpu_per_gate_ms = ms / cfg.iters;
    cpu_total_ms = cpu_per_gate_ms * gate_count;
  }

  if (cfg.verify) {
    std::vector<u64> h_out0(gate_elems);
    cudaMemcpy(h_out0.data(), d_out, gate_elems * sizeof(u64), cudaMemcpyDeviceToHost);
    std::vector<u64> h_out1;
    u64* d_out1 = nullptr;
    cudaMalloc(&d_out1, gate_elems * sizeof(u64));
    GpuSecureSufProgram prog1(effective_desc, 1, 1234, in_bits, cfg.mask_aware, cfg.mask_in);
    prog1.eval(d_in, gate_elems, d_out1, nullptr, 0);
    cudaDeviceSynchronize();
    h_out1.resize(gate_elems);
    cudaMemcpy(h_out1.data(), d_out1, gate_elems * sizeof(u64), cudaMemcpyDeviceToHost);
    cudaFree(d_out1);
    for (std::size_t i = 0; i < std::min<std::size_t>(gate_elems, 1024); ++i) {
      const u64 x = cfg.mask_aware ? h_x[i] : h_in[i];
      auto ref = eval_suf_ref(effective_desc, x);
      u64 got = h_out0[i];
      if (!h_out1.empty()) {
        got += h_out1[i];
      }
      if (got != ref.arith) {
        std::cerr << "verify mismatch at " << i << " got " << got << " expected " << ref.arith << "\n";
        return 1;
      }
    }
  }

  if (cfg.json) {
    std::cout << "{"
              << "\"model\":\"" << cfg.model << "\","
              << "\"seq\":" << cfg.seq << ","
              << "\"gate\":\"" << (model->silu ? "silu" : "gelu") << "\","
              << "\"gate_elems\":" << gate_elems << ","
              << "\"gate_count\":" << gate_count << ","
              << "\"in_bits\":" << in_bits << ","
              << "\"intervals\":" << cfg.intervals << ","
              << "\"degree\":" << cfg.degree << ","
              << "\"helpers\":" << cfg.helpers << ","
              << "\"mask_aware\":" << (cfg.mask_aware ? "true" : "false") << ","
              << "\"mask_in\":" << cfg.mask_in << ","
              << "\"mask_model\":\"" << (cfg.mask_aware ? "representative-single-mask" : "unmasked-synthetic-input") << "\","
              << "\"scope\":\"synthetic_gate_microbenchmark\","
              << "\"template_cache\":" << (cfg.template_cache ? "true" : "false") << ","
              << "\"pred_bytes\":" << key_stats.pred_bytes << ","
              << "\"lut_bytes\":" << key_stats.lut_bytes << ","
              << "\"per_gate_key_bytes\":" << per_gate_key_bytes << ","
              << "\"total_key_bytes\":" << total_key_bytes << ","
              << "\"pred_ms\":" << key_stats.pred_ms << ","
              << "\"lut_ms\":" << key_stats.lut_ms << ","
              << "\"per_gate_key_ms\":" << per_gate_key_ms << ","
              << "\"total_key_ms\":" << total_key_ms << ","
              << "\"init_ms\":" << init_ms << ","
              << "\"total_init_ms\":" << total_init_ms << ","
              << "\"per_gate_eval_ms\":" << per_gate_eval_ms << ","
              << "\"total_eval_ms\":" << total_eval_ms << ","
              << "\"cpu_per_gate_ms\":" << cpu_per_gate_ms << ","
              << "\"cpu_total_ms\":" << cpu_total_ms
              << "}\n";
  } else {
    std::cout << "FuseFSS model bench: model=" << cfg.model
              << " gate=" << (model->silu ? "silu" : "gelu")
              << " seq=" << cfg.seq
              << " gate_elems=" << gate_elems
              << " gate_count=" << gate_count
              << " in_bits=" << in_bits
              << " intervals=" << cfg.intervals
              << " degree=" << cfg.degree
              << " helpers=" << cfg.helpers
              << " mask_aware=" << (cfg.mask_aware ? 1 : 0)
              << " mask_in=" << cfg.mask_in
              << " mask_model=" << (cfg.mask_aware ? "representative-single-mask" : "unmasked-synthetic-input")
              << " scope=synthetic_gate_microbenchmark"
              << " template_cache=" << (cfg.template_cache ? 1 : 0)
              << " pred_bytes=" << key_stats.pred_bytes
              << " lut_bytes=" << key_stats.lut_bytes
              << " per_gate_key_bytes=" << per_gate_key_bytes
              << " total_key_bytes=" << total_key_bytes
              << " pred_ms=" << key_stats.pred_ms
              << " lut_ms=" << key_stats.lut_ms
              << " per_gate_key_ms=" << per_gate_key_ms
              << " total_key_ms=" << total_key_ms
              << " init_ms=" << init_ms
              << " total_init_ms=" << total_init_ms
              << " per_gate_eval_ms=" << per_gate_eval_ms
              << " total_eval_ms=" << total_eval_ms
              << " cpu_per_gate_ms=" << cpu_per_gate_ms
              << " cpu_total_ms=" << cpu_total_ms
              << "\n";
  }

  cudaFree(d_in);
  cudaFree(d_out);
  if (d_helpers) cudaFree(d_helpers);

  return 0;
}
