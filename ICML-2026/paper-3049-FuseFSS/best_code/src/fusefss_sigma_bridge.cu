#include "suf/sigma_bridge.hpp"

#include "suf/ir.hpp"
#include "suf/operator_spec.hpp"
#include "suf/postprocess.hpp"

#include "utils/gpu_data_types.h"
#include "utils/gpu_mem.h"
#include "utils/sigma_comms.h"

#include <cuda_runtime.h>
#include <curand.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <limits>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

struct AESGlobalContext {
  u32 *t0_g;
  u8 *Sbox_g;
  u32 *t4_0G;
  u32 *t4_1G;
  u32 *t4_2G;
  u32 *t4_3G;
};

extern curandGenerator_t gpuGen[2];

void initAESContext(AESGlobalContext *g);

template <typename T>
struct GPUSelectKey {
  int N;
  T *a;
  T *b;
  T *c;
  T *d1;
  T *d2;
};

template <typename T>
static GPUSelectKey<T> readGPUSelectKey(uint8_t **key_as_bytes, int N) {
  GPUSelectKey<T> k;
  k.N = N;

  const std::size_t size_in_bytes = static_cast<std::size_t>(N) * sizeof(T);

  k.a = reinterpret_cast<T *>(*key_as_bytes);
  *key_as_bytes += size_in_bytes;

  k.b = reinterpret_cast<T *>(*key_as_bytes);
  *key_as_bytes += size_in_bytes;

  k.c = reinterpret_cast<T *>(*key_as_bytes);
  *key_as_bytes += size_in_bytes;

  k.d1 = reinterpret_cast<T *>(*key_as_bytes);
  *key_as_bytes += size_in_bytes;

  k.d2 = reinterpret_cast<T *>(*key_as_bytes);
  *key_as_bytes += size_in_bytes;

  return k;
}

struct GPUSSTabKey {
  int bin;
  int N;
  u8 *ss;
  u64 memSzSS;
  u64 memSzOut;
};

struct GPUDPFTreeKey {
  int bin;
  int N;
  int evalAll;
  AESBlock *scw;
  AESBlock *l0;
  AESBlock *l1;
  u32 *tR;
  u64 szScw;
  u64 memSzScw;
  u64 memSzL;
  u64 memSzT;
  u64 memSzOut;
};

struct GPUDPFKey {
  int bin;
  int M;
  int B;
  u64 memSzOut;
  GPUDPFTreeKey *dpfTreeKey;
  GPUSSTabKey ssKey;
};

GPUDPFKey readGPUDPFKey(u8 **key_as_bytes);

template <typename T>
struct GPULUTKey {
  int bout;
  GPUDPFKey k;
  u32 *maskU;
  GPUSelectKey<T> s;
};

template <typename T>
struct GPUVectorLUTKey {
  int bout;
  int outWords;
  GPUDPFKey k;
  u32 *maskU;
  GPUSelectKey<T> s;
};

template <typename T>
static GPULUTKey<T> readGPULUTKey(uint8_t **key_as_bytes) {
  GPULUTKey<T> l;
  l.bout = static_cast<int>(**key_as_bytes);
  *key_as_bytes += sizeof(int);
  l.k = readGPUDPFKey(reinterpret_cast<u8 **>(key_as_bytes));
  l.maskU = reinterpret_cast<u32 *>(*key_as_bytes);
  *key_as_bytes += l.k.memSzOut;
  l.s = readGPUSelectKey<T>(key_as_bytes, l.k.M);
  return l;
}

template <typename T>
static GPUVectorLUTKey<T> readGPUVectorLUTKey(uint8_t **key_as_bytes) {
  GPUVectorLUTKey<T> l;
  std::memcpy(&l.bout, *key_as_bytes, sizeof(int));
  *key_as_bytes += sizeof(int);
  std::memcpy(&l.outWords, *key_as_bytes, sizeof(int));
  *key_as_bytes += sizeof(int);
  l.k = readGPUDPFKey(reinterpret_cast<u8 **>(key_as_bytes));
  l.maskU = reinterpret_cast<u32 *>(*key_as_bytes);
  *key_as_bytes += l.k.memSzOut;
  l.s = readGPUSelectKey<T>(key_as_bytes, l.k.M * l.outWords);
  return l;
}

template <typename TIn, typename TOut>
TOut *gpuKeyGenLUT(uint8_t **key_as_bytes, int party, int bin, int bout, int N,
                   TIn *d_rin, AESGlobalContext *gaes);

template <typename TIn, typename TOut>
TOut *gpuDpfLUT(GPULUTKey<TOut> k0, SigmaPeer *peer, int party, TIn *d_X, TOut *d_tab,
               AESGlobalContext *g, Stats *s, bool opMasked = true);

template <typename TIn, typename TOut>
TOut *gpuKeyGenVectorLUT(uint8_t **key_as_bytes, int party, int bin, int bout,
                         int outWords, int N, TIn *d_rin, AESGlobalContext *gaes);

template <typename TIn, typename TOut>
TOut *gpuDpfVectorLUT(GPUVectorLUTKey<TOut> k0, SigmaPeer *peer, int party,
                      TIn *d_X, TOut *d_tab, AESGlobalContext *g, Stats *s,
                      bool opMasked = true);

extern template u64 *gpuKeyGenLUT<u16, u64>(uint8_t **key_as_bytes, int party, int bin, int bout, int N,
                                           u16 *d_rin, AESGlobalContext *gaes);
extern template u64 *gpuDpfLUT<u16, u64>(GPULUTKey<u64> k0, SigmaPeer *peer, int party, u16 *d_X,
                                        u64 *d_tab, AESGlobalContext *g, Stats *s, bool opMasked);
extern template u64 *gpuKeyGenVectorLUT<u16, u64>(uint8_t **key_as_bytes, int party, int bin,
                                                 int bout, int outWords, int N, u16 *d_rin,
                                                 AESGlobalContext *gaes);
extern template u64 *gpuDpfVectorLUT<u16, u64>(GPUVectorLUTKey<u64> k0, SigmaPeer *peer,
                                              int party, u16 *d_X, u64 *d_tab,
                                              AESGlobalContext *g, Stats *s, bool opMasked);

enum TruncateType {
  None,
  LocalLRS,
  LocalARS,
  TrWithSlack,
  TrFloor
};

struct GPUDReluKey {
  GPUDPFKey dpfKey;
  u32 *mask;
};

template <typename T>
struct GPUTrCorrKey {
  GPUDReluKey mDpfKey;
  T *corr;
};

template <typename T>
struct GPUTruncateKey {
  int bin;
  int shift;
  int bout;
  int N;
  GPUTrCorrKey<T> lsbKey;
  GPUTrCorrKey<T> msbKey;
};

template <typename T>
struct GPUMulKey {
  u64 szA;
  u64 szB;
  u64 szC;
  T *a;
  T *b;
  T *c;
  GPUTruncateKey<T> trKey;
};

struct GPUAndKey {
  int N;
  std::uint32_t* b0;
  std::uint32_t* b1;
  std::uint32_t* b2;
};

template <typename T>
T *gpuKeygenMul(u8 **key_as_bytes, int party, int bw, int scale, int N,
                T *d_mask_A, T *d_mask_B, TruncateType t, AESGlobalContext *gaes);

template <typename T>
T *gpuMul(SigmaPeer *peer, int party, int bw, int scale, int N,
          GPUMulKey<T> k, T *d_X, T *d_Y, TruncateType t,
          AESGlobalContext *gaes, Stats *s);

extern template u64 *gpuKeygenMul<u64>(u8 **key_as_bytes, int party, int bw, int scale, int N,
                                       u64 *d_mask_A, u64 *d_mask_B, TruncateType t,
                                       AESGlobalContext *gaes);
extern template u64 *gpuMul<u64>(SigmaPeer *peer, int party, int bw, int scale, int N,
                                 GPUMulKey<u64> k, u64 *d_X, u64 *d_Y, TruncateType t,
                                 AESGlobalContext *gaes, Stats *s);

template <typename T>
T *gpuKeyGenAnd(u8 **key_as_bytes, int party, int bout, int N, T *d_b0, T *d_b1);

extern template u64 *gpuKeyGenAnd<u64>(u8 **key_as_bytes, int party, int bout, int N,
                                       u64 *d_b0, u64 *d_b1);

template <typename T>
T *randomGEOnGpu(const u64 n, int bw);

extern template u64 *randomGEOnGpu<u64>(const u64 n, int bw);

namespace {

enum class GateKind : int {
  Gelu = 0,
  Silu = 1,
  NExp = 2,
  Inv = 3,
  Rsqrt = 4,
};

struct DescKey {
  bool silu = false;
  int bw = 0;
  int scale = 0;
  int intervals = 0;

  bool operator==(const DescKey& other) const {
    return silu == other.silu && bw == other.bw && scale == other.scale && intervals == other.intervals;
  }
};

struct DescKeyHash {
  std::size_t operator()(const DescKey& k) const noexcept {
    std::size_t h = static_cast<std::size_t>(k.silu);
    h = h * 1315423911u + static_cast<std::size_t>(k.bw);
    h = h * 1315423911u + static_cast<std::size_t>(k.scale);
    h = h * 1315423911u + static_cast<std::size_t>(k.intervals);
    return h;
  }
};

struct TableKey {
  GateKind kind = GateKind::NExp;
  int in_bits = 0;
  int scale_in = 0;
  int scale_out = 0;
  int out_bits = 0;
  std::uint64_t clamp_min = 0;
  std::uint64_t clamp_max = 0;
  std::uint64_t extra = 0;

  bool operator==(const TableKey& other) const {
    return kind == other.kind && in_bits == other.in_bits &&
           scale_in == other.scale_in && scale_out == other.scale_out &&
           out_bits == other.out_bits && clamp_min == other.clamp_min &&
           clamp_max == other.clamp_max && extra == other.extra;
  }
};

struct TableKeyHash {
  std::size_t operator()(const TableKey& k) const noexcept {
    std::size_t h = static_cast<std::size_t>(k.kind);
    h = h * 1315423911u + static_cast<std::size_t>(k.in_bits);
    h = h * 1315423911u + static_cast<std::size_t>(k.scale_in);
    h = h * 1315423911u + static_cast<std::size_t>(k.scale_out);
    h = h * 1315423911u + static_cast<std::size_t>(k.out_bits);
    h = h * 1315423911u + static_cast<std::size_t>(k.clamp_min);
    h = h * 1315423911u + static_cast<std::size_t>(k.clamp_max);
    h = h * 1315423911u + static_cast<std::size_t>(k.extra);
    return h;
  }
};

struct GenericDeviceTableKey {
  int operator_id = 0;
  int output_index = 0;

  bool operator==(const GenericDeviceTableKey& other) const {
    return operator_id == other.operator_id && output_index == other.output_index;
  }
};

struct GenericDeviceTableKeyHash {
  std::size_t operator()(const GenericDeviceTableKey& k) const noexcept {
    std::size_t h = static_cast<std::size_t>(k.operator_id);
    h = h * 1315423911u + static_cast<std::size_t>(k.output_index);
    return h;
  }
};

struct GenericOperatorRecord {
  int id = 0;
  suf::OperatorSpecification spec;
};

enum class StrictTableKind : int {
  IdentityX = 0,
  PayloadWord = 1,
  SourceBool = 2,
  B2A = 3,
  A2BBit = 4,
};

struct StrictDeviceTableKey {
  int operator_id = 0;
  int kind = 0;
  int word_index = 0;
  int bit_width = 0;

  bool operator==(const StrictDeviceTableKey& other) const {
    return operator_id == other.operator_id && kind == other.kind &&
           word_index == other.word_index && bit_width == other.bit_width;
  }
};

struct StrictDeviceTableKeyHash {
  std::size_t operator()(const StrictDeviceTableKey& k) const noexcept {
    std::size_t h = static_cast<std::size_t>(k.operator_id);
    h = h * 1315423911u + static_cast<std::size_t>(k.kind);
    h = h * 1315423911u + static_cast<std::size_t>(k.word_index);
    h = h * 1315423911u + static_cast<std::size_t>(k.bit_width);
    return h;
  }
};

struct StrictVectorDeviceTableKey {
  int operator_id = 0;
  int kind = 0;
  int word_start = 0;
  int word_count = 0;
  int bit_width = 0;

  bool operator==(const StrictVectorDeviceTableKey& other) const {
    return operator_id == other.operator_id && kind == other.kind &&
           word_start == other.word_start && word_count == other.word_count &&
           bit_width == other.bit_width;
  }
};

struct StrictVectorDeviceTableKeyHash {
  std::size_t operator()(const StrictVectorDeviceTableKey& k) const noexcept {
    std::size_t h = static_cast<std::size_t>(k.operator_id);
    h = h * 1315423911u + static_cast<std::size_t>(k.kind);
    h = h * 1315423911u + static_cast<std::size_t>(k.word_start);
    h = h * 1315423911u + static_cast<std::size_t>(k.word_count);
    h = h * 1315423911u + static_cast<std::size_t>(k.bit_width);
    return h;
  }
};

struct VectorLutPlan {
  StrictTableKind kind = StrictTableKind::PayloadWord;
  int word_start = 0;
  int word_count = 0;
  int bit_width = 0;
  bool fused = false;

  const char* lowering_name() const {
    return fused ? "fused-vector-lut" : "legacy-per-word-lut";
  }
};

std::mutex g_desc_mutex;
std::unordered_map<DescKey, suf::SUFDescriptor, DescKeyHash> g_desc_cache;
std::mutex g_table_mutex;
std::unordered_map<TableKey, std::vector<std::uint64_t>, TableKeyHash> g_table_cache;
std::mutex g_device_table_mutex;
std::unordered_map<TableKey, std::uint64_t*, TableKeyHash> g_device_table_cache;
std::mutex g_generic_mutex;
std::unordered_map<int, GenericOperatorRecord> g_generic_ops;
std::unordered_map<TableKey, int, TableKeyHash> g_builtin_generic_ops;
int g_next_generic_op_id = 1;
std::mutex g_generic_table_mutex;
std::unordered_map<GenericDeviceTableKey, std::vector<std::uint64_t>, GenericDeviceTableKeyHash>
    g_generic_table_cache;
std::unordered_map<GenericDeviceTableKey, std::uint64_t*, GenericDeviceTableKeyHash>
    g_generic_device_table_cache;
std::mutex g_strict_table_mutex;
std::unordered_map<StrictDeviceTableKey, std::vector<std::uint64_t>, StrictDeviceTableKeyHash>
    g_strict_table_cache;
std::unordered_map<StrictDeviceTableKey, std::uint64_t*, StrictDeviceTableKeyHash>
    g_strict_device_table_cache;
std::unordered_map<StrictVectorDeviceTableKey, std::vector<std::uint64_t>, StrictVectorDeviceTableKeyHash>
    g_strict_vector_table_cache;
std::unordered_map<StrictVectorDeviceTableKey, std::uint64_t*, StrictVectorDeviceTableKeyHash>
    g_strict_vector_device_table_cache;

enum class PendingLutKind : int {
  Unknown = 0,
  BuiltinTable = 1,
  GenericOperator = 2,
  StrictTable = 3,
  StrictPayloadVector = 4,
};

struct PendingLutMeta {
  PendingLutKind kind = PendingLutKind::Unknown;
  TableKey table;
  GenericDeviceTableKey generic;
  StrictDeviceTableKey strict;
  int bin = 0;
  int bout = 0;
  std::size_t n = 0;
  std::size_t out_words = 1;
};

struct PendingLutEntry {
  std::uint8_t* key = nullptr;
  PendingLutMeta meta;
};

std::mutex g_pending_lut_mutex;
std::vector<PendingLutEntry> g_pending_lut_keys;
std::size_t g_pending_lut_idx = 0;
std::uint8_t** g_keybuf_ptr = nullptr;
AESGlobalContext g_aes{};
bool g_aes_ready = false;

void ensure_aes_ready() {
  if (!g_aes_ready) {
    initAESContext(&g_aes);
    g_aes_ready = true;
  }
}

const char* env_value(const char* name) {
  if (std::strncmp(name, "SUF_", 4) == 0) {
    std::string alias = "FUSEFSS_";
    alias += name + 4;
    const char* v = std::getenv(alias.c_str());
    if (v && *v) return v;
  }
  return std::getenv(name);
}

int env_int(const char* name, int fallback) {
  const char* v = env_value(name);
  if (!v || !*v) return fallback;
  return std::atoi(v);
}

bool env_enabled(const char* name) {
  const char* v = env_value(name);
  return v && *v && std::atoi(v) != 0;
}

bool env_has_value(const char* name) {
  const char* v = env_value(name);
  return v && *v;
}

bool use_strict_generic_hooks() {
  if (env_has_value("SUF_SIGMA_GENERIC_STRICT")) {
    return env_enabled("SUF_SIGMA_GENERIC_STRICT");
  }
  if (env_enabled("SUF_SIGMA_GENERIC") ||
      env_enabled("SUF_SIGMA_LEGACY_SPECIALIZED")) {
    return false;
  }
  return true;
}

bool fused_vector_lut_shape_supported(std::size_t words, std::size_t n, int bw) {
  const auto int_max = static_cast<std::size_t>(std::numeric_limits<int>::max());
  return words > 1 && bw > 2 && words <= int_max && n <= int_max &&
         n != 0 && words <= int_max / n;
}

double env_double(const char* name, double fallback) {
  const char* v = env_value(name);
  if (!v || !*v) return fallback;
  char* end = nullptr;
  const double val = std::strtod(v, &end);
  if (end == v) return fallback;
  return val;
}

const std::vector<std::uint64_t>& get_table(const TableKey& key);
int max_degree_for_spec(const suf::OperatorSpecification& spec);

std::size_t lookup_piece_index(const std::vector<std::uint64_t>& cuts, std::uint64_t x) {
  auto it = std::upper_bound(cuts.begin(), cuts.end(), x);
  if (it == cuts.begin()) return 0;
  return static_cast<std::size_t>(std::distance(cuts.begin(), it) - 1);
}

std::uint64_t eval_poly_mod64(const suf::Polynomial& poly, std::uint64_t x) {
  std::uint64_t y = 0;
  for (auto it = poly.coeffs.rbegin(); it != poly.coeffs.rend(); ++it) {
    y = static_cast<std::uint64_t>(
        static_cast<unsigned __int128>(y) * static_cast<unsigned __int128>(x));
    y += *it;
  }
  return y;
}

bool postprocess_trivially_empty(const suf::PostprocessProgram& program) {
  return program.arith_exprs.empty() && program.bool_exprs.empty() &&
         program.arithmetic_outputs.empty() && program.boolean_outputs.empty();
}

bool strict_runtime_supported_for_spec(const suf::OperatorSpecification& spec) {
  const bool lut_width_supported = spec.in_bits >= 8 && spec.in_bits <= 16;
  if (!lut_width_supported) return false;
  const std::size_t source_arith_outputs = suf::operator_spec_arithmetic_outputs(spec);
  if (source_arith_outputs == 0) return false;
  const int degree = max_degree_for_spec(spec);
  const auto cost = suf::count_postprocess_cost(spec.postprocess);
  if (source_arith_outputs * static_cast<std::size_t>(degree) != 0 &&
      !suf_sigma_postprocess_mul_supported()) {
    return false;
  }
  if (cost.ring_multiplications != 0 && !suf_sigma_postprocess_mul_supported()) return false;
  if (cost.boolean_ands != 0 && !suf_sigma_postprocess_and_supported()) return false;
  if (cost.b2a_conversions != 0 && !suf_sigma_postprocess_b2a_supported()) return false;
  if (cost.a2b_conversions != 0 && !suf_sigma_postprocess_a2b_supported()) return false;
  return true;
}

int capability_flags_for_spec(const suf::OperatorSpecification& spec) {
  int flags = 0;
  const auto cost = suf::count_postprocess_cost(spec.postprocess);
  const auto kappa = suf::required_postprocess_kappa_shape(spec.postprocess);
  const std::size_t source_arith_outputs = suf::operator_spec_arithmetic_outputs(spec);
  const int degree = max_degree_for_spec(spec);
  if (cost.ring_multiplications != 0 ||
      source_arith_outputs * static_cast<std::size_t>(degree) != 0) {
    flags |= SUF_SIGMA_CAP_NEEDS_MUL;
  }
  if (cost.boolean_ands != 0) flags |= SUF_SIGMA_CAP_NEEDS_AND;
  if (cost.b2a_conversions != 0) flags |= SUF_SIGMA_CAP_NEEDS_B2A;
  if (cost.a2b_conversions != 0) flags |= SUF_SIGMA_CAP_NEEDS_A2B;
  if (kappa.arithmetic != 0) flags |= SUF_SIGMA_CAP_NEEDS_KAPPA_A;
  if (kappa.boolean != 0) flags |= SUF_SIGMA_CAP_NEEDS_KAPPA_B;

  const bool single_arith = suf::operator_spec_final_arithmetic_outputs(spec) == 1;
  const bool no_bool = suf::operator_spec_final_boolean_outputs(spec) == 0;
  const bool lut_width_supported = spec.in_bits > 0 && spec.in_bits <= 16;
  const bool no_phi = postprocess_trivially_empty(spec.postprocess);
  if (single_arith && no_bool && lut_width_supported && no_phi) {
    flags |= SUF_SIGMA_CAP_SUPPORTED;
  }
  if (strict_runtime_supported_for_spec(spec)) {
    flags |= SUF_SIGMA_CAP_STRICT_SUPPORTED;
  }
  return flags;
}

GenericOperatorRecord get_generic_operator_record(int operator_id) {
  std::lock_guard<std::mutex> lock(g_generic_mutex);
  auto it = g_generic_ops.find(operator_id);
  suf::ensure(it != g_generic_ops.end(), "Sigma generic operator id is not registered");
  return it->second;
}

std::vector<std::uint64_t> build_generic_dense_table(const suf::OperatorSpecification& spec,
                                                     int output_index) {
  suf::validate_operator_spec(spec);
  suf::ensure(spec.in_bits > 0 && spec.in_bits <= 16,
              "Sigma generic operator LUT input width must be 1..16");
  suf::ensure(spec.postprocess.empty(),
              "Sigma generic compiled operator does not execute Phi in production");
  const auto outputs = suf::operator_spec_arithmetic_outputs(spec);
  suf::ensure(outputs == 1, "Sigma generic compiled operator requires one arithmetic output");
  suf::ensure(output_index == 0, "Sigma generic compiled operator output index out of range");

  const std::size_t domain = static_cast<std::size_t>(1ULL << spec.in_bits);
  std::vector<std::uint64_t> table(domain, 0);
  for (std::size_t x = 0; x < domain; ++x) {
    const auto piece = lookup_piece_index(spec.boundaries, static_cast<std::uint64_t>(x));
    table[x] = eval_poly_mod64(spec.pieces[piece].polys[static_cast<std::size_t>(output_index)],
                               static_cast<std::uint64_t>(x));
  }
  return table;
}

const std::vector<std::uint64_t>& get_generic_table(int operator_id, int output_index) {
  const GenericDeviceTableKey key{.operator_id = operator_id, .output_index = output_index};
  {
    std::lock_guard<std::mutex> lock(g_generic_table_mutex);
    auto it = g_generic_table_cache.find(key);
    if (it != g_generic_table_cache.end()) return it->second;
  }

  const auto rec = get_generic_operator_record(operator_id);
  auto table = build_generic_dense_table(rec.spec, output_index);
  std::lock_guard<std::mutex> lock(g_generic_table_mutex);
  auto res = g_generic_table_cache.emplace(key, std::move(table));
  return res.first->second;
}

std::uint64_t* get_generic_device_table(int operator_id, int output_index) {
  const GenericDeviceTableKey key{.operator_id = operator_id, .output_index = output_index};
  {
    std::lock_guard<std::mutex> lock(g_generic_table_mutex);
    auto it = g_generic_device_table_cache.find(key);
    if (it != g_generic_device_table_cache.end()) return it->second;
  }

  const auto& table = get_generic_table(operator_id, output_index);
  auto* d_table = reinterpret_cast<std::uint64_t*>(gpuMalloc(table.size() * sizeof(std::uint64_t)));
  cudaMemcpy(d_table, table.data(), table.size() * sizeof(std::uint64_t), cudaMemcpyHostToDevice);
  std::lock_guard<std::mutex> lock(g_generic_table_mutex);
  g_generic_device_table_cache.emplace(key, d_table);
  return d_table;
}

void clear_generic_table_cache() {
  std::lock_guard<std::mutex> lock(g_generic_table_mutex);
  for (auto& item : g_generic_device_table_cache) {
    if (item.second) gpuFree(item.second);
  }
  g_generic_device_table_cache.clear();
  g_generic_table_cache.clear();
}

suf::OperatorSpecification make_table_operator_spec(const TableKey& key) {
  const auto& table = get_table(key);
  suf::OperatorSpecification spec;
  spec.in_bits = key.in_bits;
  spec.boundaries.reserve(table.size());
  spec.pieces.resize(table.size());
  for (std::size_t i = 0; i < table.size(); ++i) {
    spec.boundaries.push_back(static_cast<std::uint64_t>(i));
    spec.pieces[i].polys = {suf::Polynomial{{table[i]}}};
  }
  suf::validate_operator_spec(spec);
  return spec;
}

int register_operator_spec_copy(const suf::OperatorSpecification& spec) {
  suf::validate_operator_spec(spec);
  std::lock_guard<std::mutex> lock(g_generic_mutex);
  const int id = g_next_generic_op_id++;
  g_generic_ops.emplace(id, GenericOperatorRecord{.id = id, .spec = spec});
  return id;
}

int register_operator_spec_copy_with_id(int operator_id, const suf::OperatorSpecification& spec) {
  suf::validate_operator_spec(spec);
  suf::ensure(operator_id > 0, "Sigma generic operator id must be positive");
  std::lock_guard<std::mutex> lock(g_generic_mutex);
  auto it = g_generic_ops.find(operator_id);
  if (it != g_generic_ops.end()) return operator_id;
  g_generic_ops.emplace(operator_id, GenericOperatorRecord{.id = operator_id, .spec = spec});
  if (operator_id >= g_next_generic_op_id) {
    g_next_generic_op_id = operator_id + 1;
  }
  return operator_id;
}

int get_or_register_builtin_generic_operator(const TableKey& key) {
  {
    std::lock_guard<std::mutex> lock(g_generic_mutex);
    auto it = g_builtin_generic_ops.find(key);
    if (it != g_builtin_generic_ops.end()) return it->second;
  }

  auto spec = make_table_operator_spec(key);
  std::lock_guard<std::mutex> lock(g_generic_mutex);
  auto it = g_builtin_generic_ops.find(key);
  if (it != g_builtin_generic_ops.end()) return it->second;
  const int id = g_next_generic_op_id++;
  g_generic_ops.emplace(id, GenericOperatorRecord{.id = id, .spec = std::move(spec)});
  g_builtin_generic_ops.emplace(key, id);
  return id;
}

} // namespace

extern "C" void suf_sigma_set_keybuf_ptr(std::uint8_t** keybuf_ptr) {
  g_keybuf_ptr = keybuf_ptr;
}

extern "C" bool suf_softmax_enabled() {
  return env_enabled("SUF_SOFTMAX") || env_enabled("SUF_NONLINEAR");
}

extern "C" bool suf_layernorm_enabled() {
  return env_enabled("SUF_LAYERNORM") || env_enabled("SUF_NONLINEAR");
}

namespace {

int bits_needed(std::uint64_t v) {
  int bits = 0;
  while (v > 0) {
    ++bits;
    v >>= 1;
  }
  return bits > 0 ? bits : 1;
}

std::uint64_t mask_for_bw(int bw) {
  if (bw >= 64) return ~0ULL;
  return (1ULL << bw) - 1ULL;
}

std::uint64_t mod_pow2(std::int64_t v, int bw) {
  if (bw >= 64) return static_cast<std::uint64_t>(v);
  const __int128 mod = (__int128)1 << bw;
  __int128 x = static_cast<__int128>(v) % mod;
  if (x < 0) x += mod;
  return static_cast<std::uint64_t>(x);
}

std::uint64_t mul_mod_bw(std::uint64_t a, std::uint64_t b, std::uint64_t mask) {
  const unsigned __int128 prod = static_cast<unsigned __int128>(a) * static_cast<unsigned __int128>(b);
  return static_cast<std::uint64_t>(prod) & mask;
}

std::vector<std::vector<std::uint64_t>> build_binom_mod(int degree, std::uint64_t mask) {
  std::vector<std::vector<std::uint64_t>> binom(static_cast<std::size_t>(degree + 1),
                                                std::vector<std::uint64_t>(static_cast<std::size_t>(degree + 1), 0));
  binom[0][0] = 1;
  for (int k = 1; k <= degree; ++k) {
    binom[k][0] = 1;
    binom[k][k] = 1;
    for (int i = 1; i < k; ++i) {
      binom[k][i] = (binom[k - 1][i - 1] + binom[k - 1][i]) & mask;
    }
  }
  return binom;
}

std::vector<std::uint64_t> build_pow_neg_r_mod(int degree, std::uint64_t r, std::uint64_t mask) {
  std::vector<std::uint64_t> pow(static_cast<std::size_t>(degree + 1), 0);
  pow[0] = 1;
  const std::uint64_t r_neg = (0ULL - r) & mask;
  for (int i = 1; i <= degree; ++i) {
    pow[static_cast<std::size_t>(i)] = mul_mod_bw(pow[static_cast<std::size_t>(i - 1)], r_neg, mask);
  }
  return pow;
}

std::vector<std::uint64_t> shift_poly_coeffs_mod(const std::vector<std::uint64_t>& coeffs,
                                                 int degree,
                                                 const std::vector<std::vector<std::uint64_t>>& binom,
                                                 const std::vector<std::uint64_t>& pow_neg_r,
                                                 std::uint64_t mask) {
  std::vector<std::uint64_t> out(static_cast<std::size_t>(degree + 1), 0);
  const int max_k = std::min<int>(degree, static_cast<int>(coeffs.size()) - 1);
  for (int k = 0; k <= max_k; ++k) {
    const std::uint64_t c = coeffs[static_cast<std::size_t>(k)] & mask;
    if (c == 0) continue;
    for (int i = 0; i <= k; ++i) {
      const std::uint64_t term0 = mul_mod_bw(c, binom[k][i], mask);
      const std::uint64_t term = mul_mod_bw(term0, pow_neg_r[static_cast<std::size_t>(k - i)], mask);
      out[static_cast<std::size_t>(i)] = (out[static_cast<std::size_t>(i)] + term) & mask;
    }
  }
  return out;
}

suf::SUFDescriptor build_activation_desc(bool silu, int bw, int scale, int intervals) {
  suf::SUFDescriptor d;
  d.cuts.resize(intervals);
  d.polys.resize(intervals);

  const __int128 domain = (__int128)1 << bw;
  const __int128 step = domain / intervals;
  const std::uint64_t sign_bit = (bw >= 64) ? (1ULL << 63) : (1ULL << (bw - 1));

  for (int i = 0; i < intervals; ++i) {
    const __int128 cut = step * i;
    d.cuts[i] = static_cast<std::uint64_t>(cut);
  }

  for (int i = 0; i < intervals; ++i) {
    const __int128 mid = step * i + step / 2;
    const std::uint64_t x = static_cast<std::uint64_t>(mid);
    __int128 signed_x = (x & sign_bit) ? (static_cast<__int128>(x) - domain) : static_cast<__int128>(x);
    const long double x_real = static_cast<long double>(signed_x) / static_cast<long double>(1ULL << scale);
    long double y_real = 0.0L;
    if (silu) {
      y_real = x_real / (1.0L + std::exp(-x_real));
    } else {
      const long double t = x_real / std::sqrt(2.0L);
      y_real = x_real * 0.5L * (1.0L + std::erf(t));
    }
    const long double y_scaled = y_real * static_cast<long double>(1ULL << scale);
    const std::int64_t y_fixed = llroundl(y_scaled);
    const std::uint64_t y_mod = mod_pow2(y_fixed, bw);

    d.polys[i].coeffs.clear();
    d.polys[i].coeffs.push_back(y_mod);
  }
  return d;
}

const suf::SUFDescriptor& get_activation_desc(bool silu, int bw, int scale, int intervals) {
  DescKey key{.silu = silu, .bw = bw, .scale = scale, .intervals = intervals};
  std::lock_guard<std::mutex> lock(g_desc_mutex);
  auto it = g_desc_cache.find(key);
  if (it != g_desc_cache.end()) return it->second;
  auto desc = build_activation_desc(silu, bw, scale, intervals);
  auto res = g_desc_cache.emplace(key, std::move(desc));
  return res.first->second;
}

std::size_t table_size_for_bits(int in_bits) {
  suf::ensure(in_bits > 0 && in_bits < 32, "table size bits out of range");
  return static_cast<std::size_t>(1ULL << in_bits);
}

int max_degree_for_spec(const suf::OperatorSpecification& spec) {
  int degree = 0;
  for (const auto& piece : spec.pieces) {
    for (const auto& poly : piece.polys) {
      if (!poly.coeffs.empty()) {
        degree = std::max<int>(degree, static_cast<int>(poly.coeffs.size()) - 1);
      }
    }
  }
  return degree;
}

bool postprocess_uses_shared_x(const suf::PostprocessProgram& program) {
  for (const auto& expr : program.arith_exprs) {
    for (const auto& node : expr.nodes) {
      if (node.op == suf::PostprocessArithOp::X) {
        return true;
      }
    }
  }
  return false;
}

bool strict_spec_requires_identity_lookup(const suf::OperatorSpecification& spec) {
  return max_degree_for_spec(spec) > 0 || postprocess_uses_shared_x(spec.postprocess);
}

std::uint64_t eval_predicate_for_bits(const suf::Predicate& p,
                                      std::uint64_t x,
                                      int in_bits) {
  const std::uint64_t in_mask = (in_bits >= 64) ? ~0ULL : ((1ULL << in_bits) - 1ULL);
  switch (p.kind) {
    case suf::PredKind::LT: {
      const std::uint64_t shifted = (x + p.input_add) & in_mask;
      return shifted < (p.param & in_mask) ? 1ULL : 0ULL;
    }
    case suf::PredKind::LTLOW: {
      const int f = static_cast<int>(p.f);
      const std::uint64_t low_mask = (f >= 64) ? ~0ULL : ((1ULL << f) - 1ULL);
      const std::uint64_t shifted = (x + p.input_add) & low_mask;
      return shifted < (p.gamma & low_mask) ? 1ULL : 0ULL;
    }
    case suf::PredKind::MSB:
      return (x >> (in_bits - 1)) & 1ULL;
    case suf::PredKind::MSB_ADD:
      return ((x + p.param) & in_mask) >> (in_bits - 1);
    case suf::PredKind::CONST:
      return p.param & 1ULL;
  }
  suf::fail("Sigma strict: unknown predicate kind");
}

std::uint8_t eval_bool_expr_for_table(const suf::BoolExpr& expr,
                                      const std::vector<suf::Predicate>& predicates,
                                      std::uint64_t x,
                                      int in_bits) {
  suf::ensure(expr.root >= 0 && static_cast<std::size_t>(expr.root) < expr.nodes.size(),
              "Sigma strict: malformed Boolean expression root");
  std::vector<std::uint8_t> values(expr.nodes.size(), 0);
  for (std::size_t i = 0; i < expr.nodes.size(); ++i) {
    const auto& node = expr.nodes[i];
    switch (node.kind) {
      case suf::BoolNode::Kind::PRED:
        suf::ensure(node.pred_index >= 0 &&
                        static_cast<std::size_t>(node.pred_index) < predicates.size(),
                    "Sigma strict: predicate index out of range");
        values[i] = static_cast<std::uint8_t>(
            eval_predicate_for_bits(predicates[static_cast<std::size_t>(node.pred_index)],
                                    x, in_bits) & 1ULL);
        break;
      case suf::BoolNode::Kind::NOT:
        values[i] = static_cast<std::uint8_t>(values[static_cast<std::size_t>(node.lhs)] ^ 1u);
        break;
      case suf::BoolNode::Kind::XOR:
        values[i] = static_cast<std::uint8_t>(
            (values[static_cast<std::size_t>(node.lhs)] ^
             values[static_cast<std::size_t>(node.rhs)]) & 1u);
        break;
      case suf::BoolNode::Kind::AND:
        values[i] = static_cast<std::uint8_t>(
            (values[static_cast<std::size_t>(node.lhs)] &
             values[static_cast<std::size_t>(node.rhs)]) & 1u);
        break;
      case suf::BoolNode::Kind::OR:
        values[i] = static_cast<std::uint8_t>(
            (values[static_cast<std::size_t>(node.lhs)] |
             values[static_cast<std::size_t>(node.rhs)]) & 1u);
        break;
    }
  }
  return static_cast<std::uint8_t>(values[static_cast<std::size_t>(expr.root)] & 1u);
}

std::vector<std::uint64_t> build_strict_dense_table(const StrictDeviceTableKey& key) {
  if (key.kind == static_cast<int>(StrictTableKind::B2A)) {
    std::vector<std::uint64_t> table(256, 0);
    table[1] = 1ULL & mask_for_bw(key.bit_width);
    return table;
  }
  if (key.kind == static_cast<int>(StrictTableKind::A2BBit)) {
    suf::ensure(key.bit_width >= 1 && key.bit_width <= 16,
                "Sigma strict A2B LUT input width must be 1..16");
    suf::ensure(key.word_index >= 0 && key.word_index < key.bit_width,
                "Sigma strict A2B bit index out of range");
    const std::size_t domain = table_size_for_bits(key.bit_width);
    std::vector<std::uint64_t> table(domain, 0);
    for (std::size_t x = 0; x < domain; ++x) {
      table[x] = (static_cast<std::uint64_t>(x) >> key.word_index) & 1ULL;
    }
    return table;
  }

  const auto rec = get_generic_operator_record(key.operator_id);
  const auto& spec = rec.spec;
  suf::validate_operator_spec(spec);
  suf::ensure(spec.in_bits >= 8 && spec.in_bits <= 16,
              "Sigma strict generic LUT input width must be 8..16");
  const std::size_t domain = table_size_for_bits(spec.in_bits);
  const std::uint64_t out_mask = mask_for_bw(key.bit_width);
  std::vector<std::uint64_t> table(domain, 0);
  const int degree = max_degree_for_spec(spec);
  const std::size_t source_outputs = suf::operator_spec_arithmetic_outputs(spec);
  const std::size_t coeff_words = source_outputs * static_cast<std::size_t>(degree + 1);
  const std::size_t aux_words = suf::operator_spec_aux_words(spec);
  const std::size_t payload_words = coeff_words + aux_words;

  for (std::size_t xi = 0; xi < domain; ++xi) {
    const auto x = static_cast<std::uint64_t>(xi);
    if (key.kind == static_cast<int>(StrictTableKind::IdentityX)) {
      table[xi] = x & out_mask;
      continue;
    }

    const std::size_t piece_idx = lookup_piece_index(spec.boundaries, x);
    const auto& piece = spec.pieces[piece_idx];
    if (key.kind == static_cast<int>(StrictTableKind::PayloadWord)) {
      suf::ensure(key.word_index >= 0 &&
                      static_cast<std::size_t>(key.word_index) < payload_words,
                  "Sigma strict payload word index out of range");
      const std::size_t word = static_cast<std::size_t>(key.word_index);
      if (word < coeff_words) {
        const std::size_t out = word / static_cast<std::size_t>(degree + 1);
        const std::size_t k = word % static_cast<std::size_t>(degree + 1);
        const auto& coeffs = piece.polys[out].coeffs;
        table[xi] = (k < coeffs.size() ? coeffs[k] : 0ULL) & out_mask;
      } else {
        table[xi] = piece.aux_words[word - coeff_words] & out_mask;
      }
    } else if (key.kind == static_cast<int>(StrictTableKind::SourceBool)) {
      suf::ensure(key.word_index >= 0 &&
                      static_cast<std::size_t>(key.word_index) < piece.bool_outputs.size(),
                  "Sigma strict Boolean output index out of range");
      table[xi] = eval_bool_expr_for_table(piece.bool_outputs[static_cast<std::size_t>(key.word_index)],
                                           spec.predicates, x, spec.in_bits) & 1ULL;
    } else {
      suf::fail("Sigma strict: unknown dense table kind");
    }
  }
  return table;
}

const std::vector<std::uint64_t>& get_strict_table(const StrictDeviceTableKey& key) {
  {
    std::lock_guard<std::mutex> lock(g_strict_table_mutex);
    auto it = g_strict_table_cache.find(key);
    if (it != g_strict_table_cache.end()) return it->second;
  }
  auto table = build_strict_dense_table(key);
  std::lock_guard<std::mutex> lock(g_strict_table_mutex);
  auto res = g_strict_table_cache.emplace(key, std::move(table));
  return res.first->second;
}

std::uint64_t* get_strict_device_table(const StrictDeviceTableKey& key) {
  {
    std::lock_guard<std::mutex> lock(g_strict_table_mutex);
    auto it = g_strict_device_table_cache.find(key);
    if (it != g_strict_device_table_cache.end()) return it->second;
  }

  const auto& table = get_strict_table(key);
  auto* d_table = reinterpret_cast<std::uint64_t*>(gpuMalloc(table.size() * sizeof(std::uint64_t)));
  cudaMemcpy(d_table, table.data(), table.size() * sizeof(std::uint64_t), cudaMemcpyHostToDevice);
  std::lock_guard<std::mutex> lock(g_strict_table_mutex);
  g_strict_device_table_cache.emplace(key, d_table);
  return d_table;
}

std::vector<std::uint64_t> build_strict_vector_dense_table(const StrictVectorDeviceTableKey& key) {
  suf::ensure(key.word_count > 0, "Sigma strict vector LUT word count must be positive");
  suf::ensure(key.kind == static_cast<int>(StrictTableKind::PayloadWord),
              "Sigma strict vector LUT currently supports payload words only");
  const auto rec = get_generic_operator_record(key.operator_id);
  const std::size_t domain = table_size_for_bits(rec.spec.in_bits);
  std::vector<std::uint64_t> table(static_cast<std::size_t>(key.word_count) * domain);
  for (int word = 0; word < key.word_count; ++word) {
    const StrictDeviceTableKey scalar_key{.operator_id = key.operator_id,
                                          .kind = key.kind,
                                          .word_index = key.word_start + word,
                                          .bit_width = key.bit_width};
    const auto& scalar = get_strict_table(scalar_key);
    suf::ensure(scalar.size() == domain, "Sigma strict vector LUT scalar table shape mismatch");
    std::copy(scalar.begin(), scalar.end(),
              table.begin() + static_cast<std::size_t>(word) * domain);
  }
  return table;
}

const std::vector<std::uint64_t>& get_strict_vector_table(
    const StrictVectorDeviceTableKey& key) {
  {
    std::lock_guard<std::mutex> lock(g_strict_table_mutex);
    auto it = g_strict_vector_table_cache.find(key);
    if (it != g_strict_vector_table_cache.end()) return it->second;
  }
  auto table = build_strict_vector_dense_table(key);
  std::lock_guard<std::mutex> lock(g_strict_table_mutex);
  auto res = g_strict_vector_table_cache.emplace(key, std::move(table));
  return res.first->second;
}

std::uint64_t* get_strict_vector_device_table(const StrictVectorDeviceTableKey& key) {
  {
    std::lock_guard<std::mutex> lock(g_strict_table_mutex);
    auto it = g_strict_vector_device_table_cache.find(key);
    if (it != g_strict_vector_device_table_cache.end()) return it->second;
  }

  const auto& table = get_strict_vector_table(key);
  auto* d_table = reinterpret_cast<std::uint64_t*>(gpuMalloc(table.size() * sizeof(std::uint64_t)));
  cudaMemcpy(d_table, table.data(), table.size() * sizeof(std::uint64_t), cudaMemcpyHostToDevice);
  std::lock_guard<std::mutex> lock(g_strict_table_mutex);
  g_strict_vector_device_table_cache.emplace(key, d_table);
  return d_table;
}

void clear_strict_table_cache() {
  std::lock_guard<std::mutex> lock(g_strict_table_mutex);
  for (auto& item : g_strict_device_table_cache) {
    if (item.second) gpuFree(item.second);
  }
  for (auto& item : g_strict_vector_device_table_cache) {
    if (item.second) gpuFree(item.second);
  }
  g_strict_vector_device_table_cache.clear();
  g_strict_vector_table_cache.clear();
  g_strict_device_table_cache.clear();
  g_strict_table_cache.clear();
}

std::vector<std::uint64_t> build_table(const TableKey& key) {
  const std::size_t table_size = table_size_for_bits(key.in_bits);
  std::vector<std::uint64_t> table(table_size);
  const long double scale_in = static_cast<long double>(1ULL << key.scale_in);
  const long double scale_out = static_cast<long double>(1ULL << key.scale_out);
  for (std::size_t i = 0; i < table_size; ++i) {
    std::uint64_t x_fixed = static_cast<std::uint64_t>(i);
    long double y_real = 0.0L;
    if (key.kind == GateKind::Gelu || key.kind == GateKind::Silu) {
      const std::uint64_t sign_bit = (key.in_bits >= 64) ? (1ULL << 63) : (1ULL << (key.in_bits - 1));
      const __int128 domain = (__int128)1 << key.in_bits;
      __int128 signed_x = (x_fixed & sign_bit) ? (static_cast<__int128>(x_fixed) - domain)
                                               : static_cast<__int128>(x_fixed);
      const long double x_real = static_cast<long double>(signed_x) / scale_in;
      if (key.kind == GateKind::Silu) {
        y_real = x_real / (1.0L + std::exp(-x_real));
      } else {
        const long double t = x_real / std::sqrt(2.0L);
        y_real = x_real * 0.5L * (1.0L + std::erf(t));
      }
    } else {
      if (x_fixed < key.clamp_min) x_fixed = key.clamp_min;
      if (x_fixed > key.clamp_max) x_fixed = key.clamp_max;
      const long double x_real = static_cast<long double>(x_fixed) / scale_in;
      switch (key.kind) {
        case GateKind::NExp:
          y_real = std::exp(-x_real);
          break;
        case GateKind::Inv:
          y_real = (x_real <= 0.0L) ? 0.0L : (1.0L / x_real);
          break;
        case GateKind::Rsqrt: {
          if (x_real <= 0.0L) {
            y_real = 0.0L;
          } else {
            const long double denom = x_real / static_cast<long double>(key.extra);
            y_real = (denom <= 0.0L) ? 0.0L : (1.0L / std::sqrt(denom));
          }
          break;
        }
        default:
          y_real = 0.0L;
          break;
      }
    }
    const long double y_scaled = y_real * scale_out;
    const std::int64_t y_fixed = llroundl(y_scaled);
    table[i] = mod_pow2(y_fixed, key.out_bits);
  }
  return table;
}

const std::vector<std::uint64_t>& get_table(const TableKey& key) {
  std::lock_guard<std::mutex> lock(g_table_mutex);
  auto it = g_table_cache.find(key);
  if (it != g_table_cache.end()) return it->second;
  auto table = build_table(key);
  auto res = g_table_cache.emplace(key, std::move(table));
  return res.first->second;
}

TableKey make_table_key(GateKind kind,
                        int bw_out,
                        int scale_out,
                        int in_bits,
                        int scale_in,
                        std::uint64_t clamp_min,
                        std::uint64_t clamp_max,
                        std::uint64_t extra) {
  TableKey tkey;
  tkey.kind = kind;
  tkey.in_bits = in_bits;
  tkey.scale_in = scale_in;
  tkey.scale_out = scale_out;
  tkey.out_bits = bw_out;
  tkey.clamp_min = clamp_min;
  tkey.clamp_max = clamp_max;
  tkey.extra = extra;
  return tkey;
}

std::uint64_t* get_device_table(const TableKey& key) {
  std::lock_guard<std::mutex> lock(g_device_table_mutex);
  auto it = g_device_table_cache.find(key);
  if (it != g_device_table_cache.end()) return it->second;

  const auto& table = get_table(key);
  auto* d_table = reinterpret_cast<std::uint64_t*>(gpuMalloc(table.size() * sizeof(std::uint64_t)));
  cudaMemcpy(d_table, table.data(), table.size() * sizeof(std::uint64_t), cudaMemcpyHostToDevice);
  g_device_table_cache.emplace(key, d_table);
  return d_table;
}

void clear_device_table_cache() {
  std::lock_guard<std::mutex> lock(g_device_table_mutex);
  for (auto& item : g_device_table_cache) {
    if (item.second) gpuFree(item.second);
  }
  g_device_table_cache.clear();
  clear_generic_table_cache();
  clear_strict_table_cache();
}

template <typename T>
void release_lut_key(GPULUTKey<T>& key) {
  if (key.k.bin > 7) {
    delete[] key.k.dpfTreeKey;
    key.k.dpfTreeKey = nullptr;
  }
}

template <typename T>
void release_vector_lut_key(GPUVectorLUTKey<T>& key) {
  if (key.k.bin > 7) {
    delete[] key.k.dpfTreeKey;
    key.k.dpfTreeKey = nullptr;
  }
}

bool keybuf_ready() {
  return g_keybuf_ptr && *g_keybuf_ptr;
}

void clear_pending_lut_keys() {
  std::lock_guard<std::mutex> lock(g_pending_lut_mutex);
  g_pending_lut_keys.clear();
  g_pending_lut_idx = 0;
}

PendingLutMeta unknown_lut_meta() {
  return PendingLutMeta{};
}

PendingLutMeta builtin_lut_meta(const TableKey& key, std::size_t n) {
  PendingLutMeta meta;
  meta.kind = PendingLutKind::BuiltinTable;
  meta.table = key;
  meta.bin = key.in_bits;
  meta.bout = key.out_bits;
  meta.n = n;
  return meta;
}

PendingLutMeta generic_lut_meta(int operator_id,
                               int output_index,
                               int bin,
                               int bout,
                               std::size_t n) {
  PendingLutMeta meta;
  meta.kind = PendingLutKind::GenericOperator;
  meta.generic = GenericDeviceTableKey{.operator_id = operator_id,
                                       .output_index = output_index};
  meta.bin = bin;
  meta.bout = bout;
  meta.n = n;
  return meta;
}

PendingLutMeta strict_lut_meta(int operator_id,
                              StrictTableKind kind,
                              int word_index,
                              int bit_width,
                              int bin,
                              int bout,
                              std::size_t n) {
  PendingLutMeta meta;
  meta.kind = PendingLutKind::StrictTable;
  meta.strict = StrictDeviceTableKey{.operator_id = operator_id,
                                    .kind = static_cast<int>(kind),
                                    .word_index = word_index,
                                    .bit_width = bit_width};
  meta.bin = bin;
  meta.bout = bout;
  meta.n = n;
  return meta;
}

PendingLutMeta strict_payload_vector_lut_meta(int operator_id,
                                             int word_start,
                                             int word_count,
                                             int bit_width,
                                             int bin,
                                             int bout,
                                             std::size_t n) {
  PendingLutMeta meta;
  meta.kind = PendingLutKind::StrictPayloadVector;
  meta.strict = StrictDeviceTableKey{.operator_id = operator_id,
                                    .kind = static_cast<int>(StrictTableKind::PayloadWord),
                                    .word_index = word_start,
                                    .bit_width = bit_width};
  meta.bin = bin;
  meta.bout = bout;
  meta.n = n;
  meta.out_words = static_cast<std::size_t>(word_count);
  return meta;
}

void ensure_lut_meta_matches(const PendingLutMeta& actual,
                             const PendingLutMeta& expected) {
  if (actual.kind == PendingLutKind::Unknown ||
      expected.kind == PendingLutKind::Unknown) {
    return;
  }
  suf::ensure(actual.kind == expected.kind,
              "FuseFSS queued LUT key kind mismatch");
  suf::ensure(actual.bin == expected.bin,
              "FuseFSS queued LUT key input width metadata mismatch");
  suf::ensure(actual.bout == expected.bout,
              "FuseFSS queued LUT key output width metadata mismatch");
  if (expected.n != 0) {
    suf::ensure(actual.n == expected.n,
                "FuseFSS queued LUT key vector length metadata mismatch");
  }
  switch (expected.kind) {
    case PendingLutKind::BuiltinTable:
      suf::ensure(actual.table == expected.table,
                  "FuseFSS queued LUT key builtin table metadata mismatch");
      break;
    case PendingLutKind::GenericOperator:
      suf::ensure(actual.generic == expected.generic,
                  "FuseFSS queued LUT key generic operator metadata mismatch");
      break;
    case PendingLutKind::StrictTable:
      suf::ensure(actual.strict == expected.strict,
                  "FuseFSS queued LUT key strict table metadata mismatch");
      break;
    case PendingLutKind::StrictPayloadVector:
      suf::ensure(actual.strict == expected.strict,
                  "FuseFSS queued vector LUT strict table metadata mismatch");
      suf::ensure(actual.out_words == expected.out_words,
                  "FuseFSS queued vector LUT payload width metadata mismatch");
      break;
    case PendingLutKind::Unknown:
      break;
  }
}

void queue_lut_key(std::uint8_t* key_begin, const PendingLutMeta& meta) {
  std::lock_guard<std::mutex> lock(g_pending_lut_mutex);
  g_pending_lut_keys.push_back(PendingLutEntry{.key = key_begin, .meta = meta});
}

PendingLutEntry pop_lut_key() {
  std::lock_guard<std::mutex> lock(g_pending_lut_mutex);
  if (g_pending_lut_idx >= g_pending_lut_keys.size()) return PendingLutEntry{};
  auto entry = g_pending_lut_keys[g_pending_lut_idx++];
  if (g_pending_lut_idx == g_pending_lut_keys.size()) {
    g_pending_lut_keys.clear();
    g_pending_lut_idx = 0;
  }
  return entry;
}

GPULUTKey<std::uint64_t> read_lut_key(bool prefer_queued,
                                      const PendingLutMeta& expected = unknown_lut_meta()) {
  if (prefer_queued) {
    auto queued = pop_lut_key();
    if (queued.key) {
      ensure_lut_meta_matches(queued.meta, expected);
      auto* key_ptr = queued.key;
      return readGPULUTKey<std::uint64_t>(&key_ptr);
    }
  }
  suf::ensure(keybuf_ready(), "FuseFSS key buffer is not initialized");
  return readGPULUTKey<std::uint64_t>(g_keybuf_ptr);
}

GPUVectorLUTKey<std::uint64_t> read_vector_lut_key(
    bool prefer_queued,
    const PendingLutMeta& expected = unknown_lut_meta()) {
  if (prefer_queued) {
    auto queued = pop_lut_key();
    if (queued.key) {
      ensure_lut_meta_matches(queued.meta, expected);
      auto* key_ptr = queued.key;
      return readGPUVectorLUTKey<std::uint64_t>(&key_ptr);
    }
  }
  suf::ensure(keybuf_ready(), "FuseFSS key buffer is not initialized");
  return readGPUVectorLUTKey<std::uint64_t>(g_keybuf_ptr);
}

__global__ void kernel_u64_to_u16(const std::uint64_t* in,
                                  std::uint16_t* out,
                                  int in_bits,
                                  std::size_t n) {
  const std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  std::uint64_t v = in[idx];
  if (in_bits < 16) {
    v &= (std::uint64_t(1) << in_bits) - 1ULL;
  }
  out[idx] = static_cast<std::uint16_t>(v);
}

__global__ void kernel_u16_to_u64(const std::uint16_t* in,
                                  std::uint64_t* out,
                                  int in_bits,
                                  std::size_t n) {
  const std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  std::uint64_t v = static_cast<std::uint64_t>(in[idx]);
  if (in_bits < 16) {
    v &= (std::uint64_t(1) << in_bits) - 1ULL;
  }
  out[idx] = v;
}

std::uint64_t* keygen_table_gate_u64(GateKind kind,
                                     int bw_out,
                                     int scale_out,
                                     int in_bits,
                                     int scale_in,
                                     std::uint64_t clamp_min,
                                     std::uint64_t clamp_max,
                                     std::uint64_t extra,
                                     int party,
                                     const std::uint64_t* d_input_mask,
                                     std::size_t n) {
  if (!g_keybuf_ptr || !*g_keybuf_ptr) return nullptr;
  ensure_aes_ready();

  std::uint16_t* d_input_u16 = reinterpret_cast<std::uint16_t*>(gpuMalloc(n * sizeof(std::uint16_t)));
  const int threads = 256;
  const int blocks = static_cast<int>((n + threads - 1) / threads);
  kernel_u64_to_u16<<<blocks, threads>>>(d_input_mask, d_input_u16, in_bits, n);
  cudaDeviceSynchronize();

  std::uint8_t* key_begin = *g_keybuf_ptr;
  auto d_out = gpuKeyGenLUT<u16, u64>(g_keybuf_ptr, party,
                                      in_bits, bw_out,
                                      static_cast<int>(n),
                                      d_input_u16,
                                      &g_aes);
  gpuFree(d_input_u16);
  const std::size_t key_size = static_cast<std::size_t>(*g_keybuf_ptr - key_begin);

  if (env_enabled("SUF_DEBUG")) {
    std::fprintf(stderr,
                 "[FuseFSS] keygen gate kind=%d bw=%d scale=%d in_bits=%d scale_in=%d extra=%llu n=%zu key_bytes=%zu\n",
                 static_cast<int>(kind), bw_out, scale_out, in_bits, scale_in,
                 static_cast<unsigned long long>(extra), n, key_size);
  }

  return d_out;
}

std::uint64_t* keygen_table_gate_u16(GateKind kind,
                                     int bw_out,
                                     int scale_out,
                                     int in_bits,
                                     int scale_in,
                                     std::uint64_t clamp_min,
                                     std::uint64_t clamp_max,
                                     std::uint64_t extra,
                                     int party,
                                     const std::uint16_t* d_input_mask,
                                     std::size_t n) {
  if (!g_keybuf_ptr || !*g_keybuf_ptr) return nullptr;
  ensure_aes_ready();

  std::uint8_t* key_begin = *g_keybuf_ptr;
  auto d_out = gpuKeyGenLUT<std::uint16_t, std::uint64_t>(g_keybuf_ptr, party,
                                                          in_bits, bw_out,
                                                          static_cast<int>(n),
                                                          const_cast<std::uint16_t*>(d_input_mask),
                                                          &g_aes);
  const std::size_t key_size = static_cast<std::size_t>(*g_keybuf_ptr - key_begin);

  if (env_enabled("SUF_DEBUG")) {
    std::fprintf(stderr,
                 "[FuseFSS] keygen gate kind=%d bw=%d scale=%d in_bits=%d scale_in=%d extra=%llu n=%zu key_bytes=%zu\n",
                 static_cast<int>(kind), bw_out, scale_out, in_bits, scale_in,
                 static_cast<unsigned long long>(extra), n, key_size);
  }

  return d_out;
}

std::uint64_t* eval_gate_u64(GateKind expected_kind,
                             int bw_out,
                             int scale_out,
                             int scale_in,
                             int in_bits,
                             std::uint64_t clamp_min,
                             std::uint64_t clamp_max,
                             std::uint64_t extra,
                             SigmaPeer* peer,
                             int party,
                             const std::uint64_t* d_input_masked,
                             std::size_t n,
                             Stats* s,
                             bool prefer_queued_key = true) {
  if (!keybuf_ready()) return nullptr;
  ensure_aes_ready();
  const auto tkey = make_table_key(expected_kind, bw_out, scale_out, in_bits, scale_in,
                                   clamp_min, clamp_max, extra);
  auto* d_table = get_device_table(tkey);
  auto lut_key = read_lut_key(prefer_queued_key, builtin_lut_meta(tkey, n));
  suf::ensure(lut_key.bout == bw_out, "FuseFSS LUT key output bit-width mismatch");
  suf::ensure(lut_key.k.bin == in_bits, "FuseFSS LUT key input bit-width mismatch");
  suf::ensure(lut_key.k.M == static_cast<int>(n), "FuseFSS LUT key vector length mismatch");
  if (env_enabled("SUF_DEBUG")) {
    std::fprintf(stderr,
                 "[FuseFSS] eval gate kind=%d bw=%d scale=%d in_bits=%d scale_in=%d extra=%llu n=%zu\n",
                 static_cast<int>(expected_kind), bw_out, scale_out, in_bits, scale_in,
                 static_cast<unsigned long long>(extra), n);
  }

  auto* d_open = reinterpret_cast<std::uint64_t*>(gpuMalloc(n * sizeof(std::uint64_t)));
  cudaMemcpy(d_open, d_input_masked, n * sizeof(std::uint64_t), cudaMemcpyDeviceToDevice);
  peer->reconstructInPlace(d_open, in_bits, n, s);

  std::uint16_t* d_input_u16 = reinterpret_cast<std::uint16_t*>(gpuMalloc(n * sizeof(std::uint16_t)));
  const int threads = 256;
  const int blocks = static_cast<int>((n + threads - 1) / threads);
  kernel_u64_to_u16<<<blocks, threads>>>(d_open, d_input_u16, in_bits, n);
  cudaDeviceSynchronize();
  auto d_out = gpuDpfLUT<std::uint16_t, std::uint64_t>(lut_key, peer, party,
                                                       d_input_u16,
                                                       d_table, &g_aes, s);
  gpuFree(d_input_u16);
  gpuFree(d_open);
  release_lut_key(lut_key);
  return d_out;
}

std::uint64_t* eval_gate_u16(GateKind expected_kind,
                             int bw_out,
                             int scale_out,
                             int scale_in,
                             int in_bits,
                             std::uint64_t clamp_min,
                             std::uint64_t clamp_max,
                             std::uint64_t extra,
                             SigmaPeer* peer,
                             int party,
                             const std::uint16_t* d_input_masked,
                             std::size_t n,
                             Stats* s,
                             bool prefer_queued_key = true) {
  if (!keybuf_ready()) return nullptr;
  ensure_aes_ready();
  const auto tkey = make_table_key(expected_kind, bw_out, scale_out, in_bits, scale_in,
                                   clamp_min, clamp_max, extra);
  auto* d_table = get_device_table(tkey);
  auto lut_key = read_lut_key(prefer_queued_key, builtin_lut_meta(tkey, n));
  suf::ensure(lut_key.bout == bw_out, "FuseFSS LUT key output bit-width mismatch");
  suf::ensure(lut_key.k.bin == in_bits, "FuseFSS LUT key input bit-width mismatch");
  suf::ensure(lut_key.k.M == static_cast<int>(n), "FuseFSS LUT key vector length mismatch");
  if (env_enabled("SUF_DEBUG")) {
    std::fprintf(stderr,
                 "[FuseFSS] eval gate kind=%d bw=%d scale=%d in_bits=%d scale_in=%d extra=%llu n=%zu\n",
                 static_cast<int>(expected_kind), bw_out, scale_out, in_bits, scale_in,
                 static_cast<unsigned long long>(extra), n);
  }

  auto* d_open = reinterpret_cast<std::uint16_t*>(gpuMalloc(n * sizeof(std::uint16_t)));
  cudaMemcpy(d_open, d_input_masked, n * sizeof(std::uint16_t), cudaMemcpyDeviceToDevice);
  peer->reconstructInPlace(d_open, in_bits, n, s);

  auto d_out = gpuDpfLUT<std::uint16_t, std::uint64_t>(lut_key, peer, party,
                                                       d_open,
                                                       d_table, &g_aes, s);
  gpuFree(d_open);
  release_lut_key(lut_key);
  return d_out;
}

std::uint64_t* keygen_compiled_operator_u64(int operator_id,
                                            int party,
                                            int bw,
                                            int scale,
                                            const std::uint64_t* d_input_mask,
                                            std::size_t n) {
  (void)scale;
  if (!g_keybuf_ptr || !*g_keybuf_ptr) return nullptr;
  ensure_aes_ready();
  const auto rec = get_generic_operator_record(operator_id);
  const int caps = capability_flags_for_spec(rec.spec);
  suf::ensure((caps & SUF_SIGMA_CAP_SUPPORTED) != 0,
              "Sigma generic operator is not supported by production LUT runtime");
  suf::ensure(bw > 0 && bw <= 64, "Sigma generic operator output bw must be 1..64");

  std::uint16_t* d_input_u16 = reinterpret_cast<std::uint16_t*>(gpuMalloc(n * sizeof(std::uint16_t)));
  const int threads = 256;
  const int blocks = static_cast<int>((n + threads - 1) / threads);
  kernel_u64_to_u16<<<blocks, threads>>>(d_input_mask, d_input_u16, rec.spec.in_bits, n);
  cudaDeviceSynchronize();

  std::uint8_t* key_begin = *g_keybuf_ptr;
  auto d_out = gpuKeyGenLUT<u16, u64>(g_keybuf_ptr, party,
                                      rec.spec.in_bits, bw,
                                      static_cast<int>(n),
                                      d_input_u16,
                                      &g_aes);
  gpuFree(d_input_u16);
  const std::size_t key_size = static_cast<std::size_t>(*g_keybuf_ptr - key_begin);
  if (env_enabled("SUF_DEBUG")) {
    std::fprintf(stderr,
                 "[FuseFSS] keygen generic operator=%d bw=%d in_bits=%d n=%zu key_bytes=%zu\n",
                 operator_id, bw, rec.spec.in_bits, n, key_size);
  }
  return d_out;
}

std::uint64_t* keygen_compiled_operator_u16(int operator_id,
                                            int party,
                                            int bw,
                                            int scale,
                                            const std::uint16_t* d_input_mask,
                                            std::size_t n) {
  (void)scale;
  if (!g_keybuf_ptr || !*g_keybuf_ptr) return nullptr;
  ensure_aes_ready();
  const auto rec = get_generic_operator_record(operator_id);
  const int caps = capability_flags_for_spec(rec.spec);
  suf::ensure((caps & SUF_SIGMA_CAP_SUPPORTED) != 0,
              "Sigma generic operator is not supported by production LUT runtime");
  suf::ensure(bw > 0 && bw <= 64, "Sigma generic operator output bw must be 1..64");

  std::uint8_t* key_begin = *g_keybuf_ptr;
  auto d_out = gpuKeyGenLUT<std::uint16_t, std::uint64_t>(g_keybuf_ptr, party,
                                                          rec.spec.in_bits, bw,
                                                          static_cast<int>(n),
                                                          const_cast<std::uint16_t*>(d_input_mask),
                                                          &g_aes);
  const std::size_t key_size = static_cast<std::size_t>(*g_keybuf_ptr - key_begin);
  if (env_enabled("SUF_DEBUG")) {
    std::fprintf(stderr,
                 "[FuseFSS] keygen generic-u16 operator=%d bw=%d in_bits=%d n=%zu key_bytes=%zu\n",
                 operator_id, bw, rec.spec.in_bits, n, key_size);
  }
  return d_out;
}

std::uint64_t* eval_compiled_operator_u64(SigmaPeer* peer,
                                          int operator_id,
                                          int party,
                                          int bw,
                                          int scale,
                                          const std::uint64_t* d_input_masked,
                                          std::size_t n,
                                          Stats* s,
                                          bool prefer_queued_key) {
  (void)scale;
  if (!keybuf_ready()) return nullptr;
  ensure_aes_ready();
  const auto rec = get_generic_operator_record(operator_id);
  const int caps = capability_flags_for_spec(rec.spec);
  suf::ensure((caps & SUF_SIGMA_CAP_SUPPORTED) != 0,
              "Sigma generic operator is not supported by production LUT runtime");
  suf::ensure(bw > 0 && bw <= 64, "Sigma generic operator output bw must be 1..64");

  auto* d_table = get_generic_device_table(operator_id, 0);
  auto lut_key = read_lut_key(prefer_queued_key,
                              generic_lut_meta(operator_id, 0, rec.spec.in_bits, bw, n));
  suf::ensure(lut_key.bout == bw, "FuseFSS generic LUT key output bit-width mismatch");
  suf::ensure(lut_key.k.bin == rec.spec.in_bits, "FuseFSS generic LUT key input bit-width mismatch");
  suf::ensure(lut_key.k.M == static_cast<int>(n), "FuseFSS generic LUT key vector length mismatch");
  if (env_enabled("SUF_DEBUG")) {
    std::fprintf(stderr,
                 "[FuseFSS] eval generic operator=%d bw=%d in_bits=%d n=%zu\n",
                 operator_id, bw, rec.spec.in_bits, n);
  }

  auto* d_open = reinterpret_cast<std::uint64_t*>(gpuMalloc(n * sizeof(std::uint64_t)));
  cudaMemcpy(d_open, d_input_masked, n * sizeof(std::uint64_t), cudaMemcpyDeviceToDevice);
  peer->reconstructInPlace(d_open, rec.spec.in_bits, n, s);

  std::uint16_t* d_input_u16 = reinterpret_cast<std::uint16_t*>(gpuMalloc(n * sizeof(std::uint16_t)));
  const int threads = 256;
  const int blocks = static_cast<int>((n + threads - 1) / threads);
  kernel_u64_to_u16<<<blocks, threads>>>(d_open, d_input_u16, rec.spec.in_bits, n);
  cudaDeviceSynchronize();
  auto d_out = gpuDpfLUT<std::uint16_t, std::uint64_t>(lut_key, peer, party,
                                                       d_input_u16,
                                                       d_table, &g_aes, s);
  gpuFree(d_input_u16);
  gpuFree(d_open);
  release_lut_key(lut_key);
  return d_out;
}

std::uint64_t* eval_compiled_operator_u16(SigmaPeer* peer,
                                          int operator_id,
                                          int party,
                                          int bw,
                                          int scale,
                                          const std::uint16_t* d_input_masked,
                                          std::size_t n,
                                          Stats* s,
                                          bool prefer_queued_key) {
  (void)scale;
  if (!keybuf_ready()) return nullptr;
  ensure_aes_ready();
  const auto rec = get_generic_operator_record(operator_id);
  const int caps = capability_flags_for_spec(rec.spec);
  suf::ensure((caps & SUF_SIGMA_CAP_SUPPORTED) != 0,
              "Sigma generic operator is not supported by production LUT runtime");
  suf::ensure(bw > 0 && bw <= 64, "Sigma generic operator output bw must be 1..64");

  auto* d_table = get_generic_device_table(operator_id, 0);
  auto lut_key = read_lut_key(prefer_queued_key,
                              generic_lut_meta(operator_id, 0, rec.spec.in_bits, bw, n));
  suf::ensure(lut_key.bout == bw, "FuseFSS generic LUT key output bit-width mismatch");
  suf::ensure(lut_key.k.bin == rec.spec.in_bits, "FuseFSS generic LUT key input bit-width mismatch");
  suf::ensure(lut_key.k.M == static_cast<int>(n), "FuseFSS generic LUT key vector length mismatch");
  if (env_enabled("SUF_DEBUG")) {
    std::fprintf(stderr,
                 "[FuseFSS] eval generic-u16 operator=%d bw=%d in_bits=%d n=%zu\n",
                 operator_id, bw, rec.spec.in_bits, n);
  }

  auto* d_open = reinterpret_cast<std::uint16_t*>(gpuMalloc(n * sizeof(std::uint16_t)));
  cudaMemcpy(d_open, d_input_masked, n * sizeof(std::uint16_t), cudaMemcpyDeviceToDevice);
  peer->reconstructInPlace(d_open, rec.spec.in_bits, n, s);
  auto d_out = gpuDpfLUT<std::uint16_t, std::uint64_t>(lut_key, peer, party,
                                                       d_open,
                                                       d_table, &g_aes, s);
  gpuFree(d_open);
  release_lut_key(lut_key);
  return d_out;
}

template <typename T>
GPUMulKey<T> read_mul_key_none(std::uint8_t** key_as_bytes, int n) {
  GPUMulKey<T> k;
  k.szA = static_cast<u64>(n);
  k.szB = static_cast<u64>(n);
  k.szC = static_cast<u64>(n);
  const std::size_t size_in_bytes = static_cast<std::size_t>(n) * sizeof(T);
  k.a = reinterpret_cast<T*>(*key_as_bytes);
  *key_as_bytes += size_in_bytes;
  k.b = reinterpret_cast<T*>(*key_as_bytes);
  *key_as_bytes += size_in_bytes;
  k.c = reinterpret_cast<T*>(*key_as_bytes);
  *key_as_bytes += size_in_bytes;
  std::memset(&k.trKey, 0, sizeof(k.trKey));
  return k;
}

GPUAndKey read_and_key(std::uint8_t** key_as_bytes) {
  GPUAndKey k;
  k.N = *reinterpret_cast<int*>(*key_as_bytes);
  *key_as_bytes += sizeof(int);
  const int num_ints = (k.N - 1) / PACKING_SIZE + 1;
  const std::size_t size_in_bytes = static_cast<std::size_t>(num_ints) * sizeof(std::uint32_t);
  k.b0 = reinterpret_cast<std::uint32_t*>(*key_as_bytes);
  *key_as_bytes += size_in_bytes;
  k.b1 = reinterpret_cast<std::uint32_t*>(*key_as_bytes);
  *key_as_bytes += size_in_bytes;
  k.b2 = reinterpret_cast<std::uint32_t*>(*key_as_bytes);
  *key_as_bytes += size_in_bytes;
  return k;
}

__global__ void kernel_eval_and_packed(int party,
                                       int n,
                                       int num_words,
                                       const std::uint64_t* lhs,
                                       const std::uint64_t* rhs,
                                       const std::uint32_t* a_packed,
                                       const std::uint32_t* b_packed,
                                       const std::uint32_t* c_packed,
                                       std::uint32_t* out_packed) {
  const int word = blockIdx.x * blockDim.x + threadIdx.x;
  if (word >= num_words) return;
  std::uint32_t packed = 0;
  const std::uint32_t a_word = a_packed[word];
  const std::uint32_t b_word = b_packed[word];
  const std::uint32_t c_word = c_packed[word];
  for (int bit = 0; bit < PACKING_SIZE; ++bit) {
    const int idx = word * PACKING_SIZE + bit;
    if (idx >= n) break;
    const std::uint32_t x = static_cast<std::uint32_t>(lhs[idx] & 1ULL);
    const std::uint32_t y = static_cast<std::uint32_t>(rhs[idx] & 1ULL);
    const std::uint32_t a = (a_word >> bit) & 1U;
    const std::uint32_t b = (b_word >> bit) & 1U;
    const std::uint32_t c = (c_word >> bit) & 1U;
    const std::uint32_t out =
        (((party == SERVER1) ? (x & y) : 0U) ^ (x & b) ^ (a & y) ^ c) & 1U;
    packed |= (out << bit);
  }
  out_packed[word] = packed;
}

__global__ void kernel_unpack_bits_to_u64(const std::uint32_t* packed,
                                          std::uint64_t* out,
                                          int n) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  out[idx] = (packed[idx / PACKING_SIZE] >> (idx & (PACKING_SIZE - 1))) & 1U;
}

__global__ void kernel_fill_u64(std::uint64_t* out,
                                std::uint64_t value,
                                std::uint64_t mask,
                                std::size_t n) {
  const std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  out[idx] = value & mask;
}

__global__ void kernel_add_u64(const std::uint64_t* a,
                               const std::uint64_t* b,
                               std::uint64_t* out,
                               std::uint64_t mask,
                               std::size_t n) {
  const std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  out[idx] = (a[idx] + b[idx]) & mask;
}

__global__ void kernel_sub_u64(const std::uint64_t* a,
                               const std::uint64_t* b,
                               std::uint64_t* out,
                               std::uint64_t mask,
                               std::size_t n) {
  const std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  out[idx] = (a[idx] - b[idx]) & mask;
}

__global__ void kernel_xor_bit_u64(const std::uint64_t* a,
                                   const std::uint64_t* b,
                                   std::uint64_t* out,
                                   std::size_t n) {
  const std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  out[idx] = (a[idx] ^ b[idx]) & 1ULL;
}

__global__ void kernel_not_bit_u64(const std::uint64_t* in,
                                   std::uint64_t* out,
                                   std::size_t n) {
  const std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  out[idx] = (in[idx] ^ 1ULL) & 1ULL;
}

__global__ void kernel_bit_u64(const std::uint64_t* in,
                               std::uint64_t* out,
                               int bit_index,
                               std::size_t n) {
  const std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  out[idx] = (in[idx] >> bit_index) & 1ULL;
}

__global__ void kernel_b2a_finalize_key_shares(int party,
                                               const std::uint64_t* bool_mask,
                                               const std::uint64_t* out_mask,
                                               std::uint64_t* r_share,
                                               std::uint64_t* m_share,
                                               std::uint64_t mask,
                                               std::size_t n) {
  const std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  const std::uint64_t r = bool_mask[idx] & 1ULL;
  const std::uint64_t m = out_mask[idx] & mask;
  if (party == SERVER1) {
    r_share[idx] = (r - r_share[idx]) & mask;
    m_share[idx] = (m - m_share[idx]) & mask;
  }
}

__global__ void kernel_b2a_eval_share(int party,
                                      const std::uint64_t* bool_open,
                                      const std::uint64_t* r_share,
                                      const std::uint64_t* m_share,
                                      std::uint64_t* out_share,
                                      std::uint64_t mask,
                                      std::size_t n) {
  const std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  const std::uint64_t o = bool_open[idx] & 1ULL;
  const std::uint64_t r = r_share[idx] & mask;
  const std::uint64_t m = m_share[idx] & mask;
  const std::uint64_t public_term = (party == SERVER1) ? o : 0ULL;
  out_share[idx] = (public_term + r - (2ULL * o * r) + m) & mask;
}

void write_key_int(std::uint8_t** key_as_bytes, int value) {
  std::memcpy(*key_as_bytes, &value, sizeof(int));
  *key_as_bytes += sizeof(int);
}

int read_key_int(std::uint8_t** key_as_bytes) {
  int value = 0;
  std::memcpy(&value, *key_as_bytes, sizeof(int));
  *key_as_bytes += sizeof(int);
  return value;
}

std::uint64_t* alloc_device_words(std::size_t n) {
  return reinterpret_cast<std::uint64_t*>(gpuMalloc(n * sizeof(std::uint64_t)));
}

void launch_fill(std::uint64_t* out, std::uint64_t value, int bw, std::size_t n) {
  const int threads = 256;
  const int blocks = static_cast<int>((n + threads - 1) / threads);
  kernel_fill_u64<<<blocks, threads>>>(out, value, mask_for_bw(bw), n);
  cudaDeviceSynchronize();
}

std::uint64_t* local_add_words(const std::uint64_t* a,
                               const std::uint64_t* b,
                               int bw,
                               std::size_t n) {
  auto* out = alloc_device_words(n);
  const int threads = 256;
  const int blocks = static_cast<int>((n + threads - 1) / threads);
  kernel_add_u64<<<blocks, threads>>>(a, b, out, mask_for_bw(bw), n);
  cudaDeviceSynchronize();
  return out;
}

std::uint64_t* local_sub_words(const std::uint64_t* a,
                               const std::uint64_t* b,
                               int bw,
                               std::size_t n) {
  auto* out = alloc_device_words(n);
  const int threads = 256;
  const int blocks = static_cast<int>((n + threads - 1) / threads);
  kernel_sub_u64<<<blocks, threads>>>(a, b, out, mask_for_bw(bw), n);
  cudaDeviceSynchronize();
  return out;
}

std::uint64_t* local_xor_bits(const std::uint64_t* a,
                              const std::uint64_t* b,
                              std::size_t n) {
  auto* out = alloc_device_words(n);
  const int threads = 256;
  const int blocks = static_cast<int>((n + threads - 1) / threads);
  kernel_xor_bit_u64<<<blocks, threads>>>(a, b, out, n);
  cudaDeviceSynchronize();
  return out;
}

std::uint64_t* local_not_bits(const std::uint64_t* in, std::size_t n) {
  auto* out = alloc_device_words(n);
  const int threads = 256;
  const int blocks = static_cast<int>((n + threads - 1) / threads);
  kernel_not_bit_u64<<<blocks, threads>>>(in, out, n);
  cudaDeviceSynchronize();
  return out;
}

std::uint64_t* local_bit_bits(const std::uint64_t* in, int bit_index, std::size_t n) {
  suf::ensure(bit_index >= 0 && bit_index < 64, "Sigma local bit index out of range");
  auto* out = alloc_device_words(n);
  const int threads = 256;
  const int blocks = static_cast<int>((n + threads - 1) / threads);
  kernel_bit_u64<<<blocks, threads>>>(in, out, bit_index, n);
  cudaDeviceSynchronize();
  return out;
}

std::uint64_t* local_lsb_bits(const std::uint64_t* in, std::size_t n) {
  return local_bit_bits(in, 0, n);
}

std::uint64_t* clone_device_words(const std::uint64_t* in, std::size_t n) {
  auto* out = alloc_device_words(n);
  cudaMemcpy(out, in, n * sizeof(std::uint64_t), cudaMemcpyDeviceToDevice);
  return out;
}

SufSigmaCompiledOperatorResult* make_empty_v2_result(std::size_t n,
                                                     std::size_t arith,
                                                     std::size_t bools,
                                                     int capability_flags) {
  auto* result = new SufSigmaCompiledOperatorResult();
  result->n = n;
  result->arithmetic_outputs = arith;
  result->boolean_outputs = bools;
  result->capability_flags = capability_flags;
  result->d_arithmetic = arith == 0 ? nullptr : new std::uint64_t*[arith]();
  result->d_boolean = bools == 0 ? nullptr : new std::uint64_t*[bools]();
  return result;
}

struct StrictDeviceValue {
  std::uint64_t* ptr = nullptr;
};

class StrictCompiledExecutor {
 public:
  StrictCompiledExecutor(bool keygen,
                         SigmaPeer* peer,
                         int operator_id,
                         const suf::OperatorSpecification& spec,
                         int party,
                         int bw,
                         int scale,
                         const std::uint64_t* d_input,
                         std::size_t n,
                         Stats* stats,
                         const SufSigmaPostprocessContext* postprocess_ctx)
      : keygen_(keygen),
        peer_(peer),
        operator_id_(operator_id),
        spec_(spec),
        party_(party),
        bw_(bw),
        scale_(scale),
        d_input_(d_input),
        n_(n),
        stats_(stats),
        postprocess_ctx_(postprocess_ctx),
        degree_(max_degree_for_spec(spec)),
        source_arith_outputs_(suf::operator_spec_arithmetic_outputs(spec)),
        source_bool_outputs_(suf::operator_spec_boolean_outputs(spec)),
        aux_words_(suf::operator_spec_aux_words(spec)),
        payload_words_(source_arith_outputs_ * static_cast<std::size_t>(degree_ + 1) +
                       aux_words_),
        needs_identity_lookup_(strict_spec_requires_identity_lookup(spec)) {}

  SufSigmaCompiledOperatorResult* run() {
    suf::ensure(d_input_ != nullptr, "Sigma strict generic received null input");
    suf::ensure(n_ <= static_cast<std::size_t>(std::numeric_limits<int>::max()),
                "Sigma strict generic length exceeds int range");
    suf::ensure(bw_ > 0 && bw_ <= 64, "Sigma strict generic bw must be 1..64");
    suf::ensure(strict_runtime_supported_for_spec(spec_),
                "Sigma strict generic operator is not supported by production runtime");
    preflight_postprocess_context();
    ensure_aes_ready();

    const std::uint64_t* d_lut_input_words = d_input_;
    if (!keygen_) {
      suf::ensure(peer_ != nullptr, "Sigma strict generic eval received null peer");
      auto* d_open = track(clone_device_words(d_input_, n_));
      peer_->reconstructInPlace(d_open, spec_.in_bits, n_, stats_);
      d_lut_input_words = d_open;
      x_hat_ = StrictDeviceValue{d_open};
    } else {
      x_hat_ = StrictDeviceValue{track(fill_word(0, bw_))};
    }

    auto* d_input_u16 = reinterpret_cast<std::uint16_t*>(gpuMalloc(n_ * sizeof(std::uint16_t)));
    const int threads = 256;
    const int blocks = static_cast<int>((n_ + threads - 1) / threads);
    kernel_u64_to_u16<<<blocks, threads>>>(d_lut_input_words, d_input_u16, spec_.in_bits, n_);
    cudaDeviceSynchronize();

    if (needs_identity_lookup_) {
      x_ = StrictDeviceValue{track(lookup_table(d_input_u16,
                                                StrictTableKind::IdentityX,
                                                0,
                                                bw_))};
    }

    payload_.reserve(payload_words_);
    if (payload_words_ != 0) {
      const VectorLutPlan payload_plan = payload_vector_lut_plan();
      if (payload_plan.fused) {
        auto* base = track(lookup_payload_vector(d_input_u16, payload_plan));
        for (std::size_t word = 0; word < payload_words_; ++word) {
          payload_.push_back(StrictDeviceValue{base + word * n_});
        }
      } else {
        for (std::size_t word = 0; word < payload_words_; ++word) {
          payload_.push_back(StrictDeviceValue{
              track(lookup_table(d_input_u16,
                                 StrictTableKind::PayloadWord,
                                 static_cast<int>(word),
                                 bw_))});
        }
      }
    }

    source_bool_.reserve(source_bool_outputs_);
    for (std::size_t out = 0; out < source_bool_outputs_; ++out) {
      auto raw_bool = StrictDeviceValue{
          track(lookup_table(d_input_u16,
                             StrictTableKind::SourceBool,
                             static_cast<int>(out),
                             bw_))};
      source_bool_.push_back(StrictDeviceValue{track(local_lsb_bits(raw_bool.ptr, n_))});
    }
    gpuFree(d_input_u16);

    source_arith_.reserve(source_arith_outputs_);
    for (std::size_t out = 0; out < source_arith_outputs_; ++out) {
      source_arith_.push_back(eval_source_poly(out));
    }

    std::vector<StrictDeviceValue> final_arith;
    std::vector<StrictDeviceValue> final_bool;
    if (spec_.postprocess.arithmetic_outputs.empty()) {
      final_arith = source_arith_;
    } else {
      for (const int out : spec_.postprocess.arithmetic_outputs) {
        final_arith.push_back(eval_arith_expr(out));
      }
    }
    if (spec_.postprocess.boolean_outputs.empty()) {
      final_bool = source_bool_;
    } else {
      for (const int out : spec_.postprocess.boolean_outputs) {
        final_bool.push_back(eval_bool_expr(out));
      }
    }
    return finish(final_arith, final_bool);
  }

 private:
  std::uint64_t* track(std::uint64_t* ptr) {
    suf::ensure(ptr != nullptr, "Sigma strict generic produced null device pointer");
    owned_.push_back(ptr);
    return ptr;
  }

  bool is_owned(std::uint64_t* ptr) const {
    return std::find(owned_.begin(), owned_.end(), ptr) != owned_.end();
  }

  void preflight_postprocess_context() const {
    const int max_a2b_bit = suf::max_postprocess_a2b_bit_index(spec_.postprocess);
    if (max_a2b_bit >= 0) {
      suf::ensure(max_a2b_bit < bw_,
                  "Sigma strict generic A2B bit index is outside arithmetic bw");
      if (max_a2b_bit != 0) {
        suf::ensure(bw_ <= 16,
                    "Sigma strict generic non-LSB A2B requires bw<=16");
      }
    }
    const auto kappa = suf::required_postprocess_kappa_shape(spec_.postprocess);
    if (kappa.arithmetic == 0 && kappa.boolean == 0) return;
    suf::ensure(postprocess_ctx_ != nullptr,
                "Sigma strict generic operator requires explicit kappa context");
    suf::ensure(postprocess_ctx_->kappa_a_count >= kappa.arithmetic,
                "Sigma strict generic kappaA context is too small");
    suf::ensure(postprocess_ctx_->kappa_b_count >= kappa.boolean,
                "Sigma strict generic kappaB context is too small");
    if (kappa.arithmetic != 0) {
      suf::ensure(postprocess_ctx_->d_kappa_a != nullptr,
                  "Sigma strict generic kappaA context is null");
      for (std::size_t i = 0; i < kappa.arithmetic; ++i) {
        suf::ensure(postprocess_ctx_->d_kappa_a[i] != nullptr,
                    "Sigma strict generic kappaA device vector is null");
        if (postprocess_ctx_->kappa_a_lengths) {
          suf::ensure(postprocess_ctx_->kappa_a_lengths[i] >= n_,
                      "Sigma strict generic kappaA device vector is shorter than n");
        }
      }
    }
    if (kappa.boolean != 0) {
      suf::ensure(postprocess_ctx_->d_kappa_b != nullptr,
                  "Sigma strict generic kappaB context is null");
      for (std::size_t i = 0; i < kappa.boolean; ++i) {
        suf::ensure(postprocess_ctx_->d_kappa_b[i] != nullptr,
                    "Sigma strict generic kappaB device vector is null");
        if (postprocess_ctx_->kappa_b_lengths) {
          suf::ensure(postprocess_ctx_->kappa_b_lengths[i] >= n_,
                      "Sigma strict generic kappaB device vector is shorter than n");
        }
      }
    }
  }

  std::uint64_t* fill_word(std::uint64_t value, int bw) {
    auto* ptr = alloc_device_words(n_);
    launch_fill(ptr, value, bw, n_);
    return ptr;
  }

  std::uint64_t* lookup_table(const std::uint16_t* d_input_u16,
                              StrictTableKind kind,
                              int word_index,
                              int bout) {
    if (keygen_) {
      suf::ensure(g_keybuf_ptr && *g_keybuf_ptr,
                  "Sigma strict generic keygen key buffer is not initialized");
      return gpuKeyGenLUT<std::uint16_t, std::uint64_t>(
          g_keybuf_ptr, party_, spec_.in_bits, bout, static_cast<int>(n_),
          const_cast<std::uint16_t*>(d_input_u16), &g_aes);
    }

    suf::ensure(peer_ != nullptr, "Sigma strict generic eval received null peer");
    auto lut_key = read_lut_key(
        true,
        strict_lut_meta(operator_id_, kind, word_index, bout,
                        spec_.in_bits, bout, n_));
    suf::ensure(lut_key.bout == bout, "Sigma strict generic LUT key output width mismatch");
    suf::ensure(lut_key.k.bin == spec_.in_bits, "Sigma strict generic LUT key input width mismatch");
    suf::ensure(lut_key.k.M == static_cast<int>(n_), "Sigma strict generic LUT key length mismatch");
    auto* table = get_strict_device_table(
        StrictDeviceTableKey{.operator_id = operator_id_,
                             .kind = static_cast<int>(kind),
                             .word_index = word_index,
                             .bit_width = bout});
    auto* out = gpuDpfLUT<std::uint16_t, std::uint64_t>(
        lut_key, peer_, party_, const_cast<std::uint16_t*>(d_input_u16),
        table, &g_aes, stats_);
    release_lut_key(lut_key);
    return out;
  }

  VectorLutPlan payload_vector_lut_plan() const {
    VectorLutPlan plan;
    plan.kind = StrictTableKind::PayloadWord;
    plan.word_start = 0;
    plan.word_count = static_cast<int>(payload_words_);
    plan.bit_width = bw_;
    plan.fused = !env_enabled("SUF_SIGMA_VECTOR_LUT_LEGACY") &&
                 fused_vector_lut_shape_supported(payload_words_, n_, bw_);
    if (env_enabled("SUF_DEBUG")) {
      std::fprintf(stderr,
                   "[FuseFSS] strict payload vector plan operator=%d words=%d bw=%d lowering=%s n=%zu\n",
                   operator_id_, plan.word_count, plan.bit_width,
                   plan.lowering_name(), n_);
    }
    return plan;
  }

  std::uint64_t* lookup_payload_vector(const std::uint16_t* d_input_u16,
                                       const VectorLutPlan& plan) {
    suf::ensure(plan.fused, "Sigma strict payload vector called with legacy plan");
    suf::ensure(plan.word_count > 0, "Sigma strict payload vector word count is empty");
    suf::ensure(plan.bit_width > 2,
                "Sigma strict payload vector LUT requires arithmetic output width >2");
    if (keygen_) {
      suf::ensure(g_keybuf_ptr && *g_keybuf_ptr,
                  "Sigma strict generic vector keygen key buffer is not initialized");
      std::uint8_t* key_begin = *g_keybuf_ptr;
      auto* out = gpuKeyGenVectorLUT<std::uint16_t, std::uint64_t>(
          g_keybuf_ptr, party_, spec_.in_bits, plan.bit_width, plan.word_count,
          static_cast<int>(n_), const_cast<std::uint16_t*>(d_input_u16), &g_aes);
      if (env_enabled("SUF_DEBUG")) {
        std::fprintf(stderr,
                     "[FuseFSS] strict fused vector LUT keygen operator=%d words=%d n=%zu key_bytes=%zu\n",
                     operator_id_, plan.word_count, n_,
                     static_cast<std::size_t>(*g_keybuf_ptr - key_begin));
      }
      return out;
    }

    suf::ensure(peer_ != nullptr, "Sigma strict generic vector eval received null peer");
    auto lut_key = read_vector_lut_key(
        true,
        strict_payload_vector_lut_meta(operator_id_, plan.word_start, plan.word_count,
                                       plan.bit_width, spec_.in_bits, plan.bit_width,
                                       n_));
    suf::ensure(lut_key.bout == plan.bit_width,
                "Sigma strict vector LUT key output width mismatch");
    suf::ensure(lut_key.outWords == plan.word_count,
                "Sigma strict vector LUT payload width mismatch");
    suf::ensure(lut_key.k.bin == spec_.in_bits,
                "Sigma strict vector LUT key input width mismatch");
    suf::ensure(lut_key.k.M == static_cast<int>(n_),
                "Sigma strict vector LUT key length mismatch");
    auto* table = get_strict_vector_device_table(
        StrictVectorDeviceTableKey{.operator_id = operator_id_,
                                   .kind = static_cast<int>(plan.kind),
                                   .word_start = plan.word_start,
                                   .word_count = plan.word_count,
                                   .bit_width = plan.bit_width});
    auto* out = gpuDpfVectorLUT<std::uint16_t, std::uint64_t>(
        lut_key, peer_, party_, const_cast<std::uint16_t*>(d_input_u16),
        table, &g_aes, stats_);
    release_vector_lut_key(lut_key);
    return out;
  }

  StrictDeviceValue eval_source_poly(std::size_t output) {
    const std::size_t stride = static_cast<std::size_t>(degree_ + 1);
    std::uint64_t* acc = payload_[output * stride + static_cast<std::size_t>(degree_)].ptr;
    for (int k = degree_ - 1; k >= 0; --k) {
      suf::ensure(x_.ptr != nullptr,
                  "Sigma strict generic polynomial evaluation requires shared x");
      auto product = mul_value(StrictDeviceValue{acc}, x_);
      auto sum = add_value(product, payload_[output * stride + static_cast<std::size_t>(k)]);
      acc = sum.ptr;
    }
    return StrictDeviceValue{acc};
  }

  StrictDeviceValue add_value(StrictDeviceValue lhs, StrictDeviceValue rhs) {
    return StrictDeviceValue{track(local_add_words(lhs.ptr, rhs.ptr, bw_, n_))};
  }

  StrictDeviceValue sub_value(StrictDeviceValue lhs, StrictDeviceValue rhs) {
    return StrictDeviceValue{track(local_sub_words(lhs.ptr, rhs.ptr, bw_, n_))};
  }

  StrictDeviceValue mul_value(StrictDeviceValue lhs, StrictDeviceValue rhs) {
    std::uint64_t* out = nullptr;
    if (keygen_) {
      out = suf_sigma_keygen_postprocess_mul_u64(party_, bw_, scale_, lhs.ptr, rhs.ptr, n_);
    } else {
      out = suf_sigma_eval_postprocess_mul_u64(peer_, party_, bw_, scale_,
                                               lhs.ptr, rhs.ptr, n_, stats_);
    }
    return StrictDeviceValue{track(out)};
  }

  StrictDeviceValue xor_value(StrictDeviceValue lhs, StrictDeviceValue rhs) {
    return StrictDeviceValue{track(local_xor_bits(lhs.ptr, rhs.ptr, n_))};
  }

  StrictDeviceValue not_value(StrictDeviceValue in) {
    if (keygen_) return in;
    return StrictDeviceValue{track(local_not_bits(in.ptr, n_))};
  }

  StrictDeviceValue and_value(StrictDeviceValue lhs, StrictDeviceValue rhs) {
    std::uint64_t* out = nullptr;
    if (keygen_) {
      out = suf_sigma_keygen_postprocess_and_u64(party_, lhs.ptr, rhs.ptr, n_);
    } else {
      out = suf_sigma_eval_postprocess_and_u64(peer_, party_, lhs.ptr, rhs.ptr, n_, stats_);
    }
    return StrictDeviceValue{track(out)};
  }

  StrictDeviceValue or_value(StrictDeviceValue lhs, StrictDeviceValue rhs) {
    auto x = xor_value(lhs, rhs);
    auto a = and_value(lhs, rhs);
    return xor_value(x, a);
  }

  StrictDeviceValue b2a_value(StrictDeviceValue in) {
    std::uint64_t* out = nullptr;
    if (keygen_) {
      out = suf_sigma_keygen_postprocess_b2a_u64(party_, bw_, in.ptr, n_);
    } else {
      out = suf_sigma_eval_postprocess_b2a_u64(peer_, party_, bw_, in.ptr, n_, stats_);
    }
    return StrictDeviceValue{track(out)};
  }

  StrictDeviceValue a2b_value(StrictDeviceValue in, int bit_index) {
    std::uint64_t* out = nullptr;
    if (keygen_) {
      out = suf_sigma_keygen_postprocess_a2b_bit_u64(party_, bw_, bit_index, in.ptr, n_);
    } else {
      out = suf_sigma_eval_postprocess_a2b_bit_u64(peer_, party_, bw_, bit_index,
                                                   in.ptr, n_, stats_);
    }
    return StrictDeviceValue{track(out)};
  }

  StrictDeviceValue kappa_arith_value(int index) {
    suf::ensure(postprocess_ctx_ != nullptr,
                "Sigma strict Phi KAPPA_A requires postprocess context");
    suf::ensure(index >= 0 &&
                    static_cast<std::size_t>(index) < postprocess_ctx_->kappa_a_count,
                "Sigma strict Phi KAPPA_A index out of range");
    suf::ensure(postprocess_ctx_->d_kappa_a != nullptr &&
                    postprocess_ctx_->d_kappa_a[index] != nullptr,
                "Sigma strict Phi KAPPA_A received null device vector");
    const StrictDeviceValue raw{
        const_cast<std::uint64_t*>(postprocess_ctx_->d_kappa_a[index])};
    return add_value(raw, StrictDeviceValue{track(fill_word(0, bw_))});
  }

  StrictDeviceValue kappa_bool_value(int index) {
    suf::ensure(postprocess_ctx_ != nullptr,
                "Sigma strict Phi KAPPA_B requires postprocess context");
    suf::ensure(index >= 0 &&
                    static_cast<std::size_t>(index) < postprocess_ctx_->kappa_b_count,
                "Sigma strict Phi KAPPA_B index out of range");
    suf::ensure(postprocess_ctx_->d_kappa_b != nullptr &&
                    postprocess_ctx_->d_kappa_b[index] != nullptr,
                "Sigma strict Phi KAPPA_B received null device vector");
    return StrictDeviceValue{track(local_lsb_bits(postprocess_ctx_->d_kappa_b[index], n_))};
  }

  StrictDeviceValue aux_value(int index) const {
    suf::ensure(index >= 0 && static_cast<std::size_t>(index) < aux_words_,
                "Sigma strict Phi AUX index out of range");
    const std::size_t base =
        source_arith_outputs_ * static_cast<std::size_t>(degree_ + 1);
    return payload_[base + static_cast<std::size_t>(index)];
  }

  StrictDeviceValue eval_arith_expr(int expr_index) {
    suf::ensure(expr_index >= 0 &&
                    static_cast<std::size_t>(expr_index) < spec_.postprocess.arith_exprs.size(),
                "Sigma strict Phi arithmetic expression index out of range");
    const auto& expr = spec_.postprocess.arith_exprs[static_cast<std::size_t>(expr_index)];
    std::vector<StrictDeviceValue> values;
    values.reserve(expr.nodes.size());
    for (const auto& node : expr.nodes) {
      switch (node.op) {
        case suf::PostprocessArithOp::CONST:
          values.push_back(StrictDeviceValue{track(fill_word(keygen_ ? 0ULL : node.value, bw_))});
          break;
        case suf::PostprocessArithOp::X:
          values.push_back(x_);
          break;
        case suf::PostprocessArithOp::X_HAT:
          values.push_back(x_hat_);
          break;
        case suf::PostprocessArithOp::POLY_OUT:
          suf::ensure(node.index >= 0 &&
                          static_cast<std::size_t>(node.index) < source_arith_.size(),
                      "Sigma strict Phi POLY_OUT index out of range");
          values.push_back(source_arith_[static_cast<std::size_t>(node.index)]);
          break;
        case suf::PostprocessArithOp::AUX:
          values.push_back(aux_value(node.index));
          break;
        case suf::PostprocessArithOp::KAPPA_A:
          values.push_back(kappa_arith_value(node.index));
          break;
        case suf::PostprocessArithOp::ADD:
          values.push_back(add_value(values[static_cast<std::size_t>(node.lhs)],
                                     values[static_cast<std::size_t>(node.rhs)]));
          break;
        case suf::PostprocessArithOp::SUB:
          values.push_back(sub_value(values[static_cast<std::size_t>(node.lhs)],
                                     values[static_cast<std::size_t>(node.rhs)]));
          break;
        case suf::PostprocessArithOp::MUL:
          values.push_back(mul_value(values[static_cast<std::size_t>(node.lhs)],
                                     values[static_cast<std::size_t>(node.rhs)]));
          break;
        case suf::PostprocessArithOp::B2A:
          values.push_back(b2a_value(eval_bool_expr(node.index)));
          break;
      }
    }
    suf::ensure(expr.root >= 0 && static_cast<std::size_t>(expr.root) < values.size(),
                "Sigma strict Phi malformed arithmetic expression root");
    return values[static_cast<std::size_t>(expr.root)];
  }

  StrictDeviceValue eval_bool_expr(int expr_index) {
    suf::ensure(expr_index >= 0 &&
                    static_cast<std::size_t>(expr_index) < spec_.postprocess.bool_exprs.size(),
                "Sigma strict Phi Boolean expression index out of range");
    const auto& expr = spec_.postprocess.bool_exprs[static_cast<std::size_t>(expr_index)];
    std::vector<StrictDeviceValue> values;
    values.reserve(expr.nodes.size());
    for (const auto& node : expr.nodes) {
      switch (node.op) {
        case suf::PostprocessBoolOp::CONST:
          values.push_back(StrictDeviceValue{track(fill_word(keygen_ ? 0ULL : node.value, 1))});
          break;
        case suf::PostprocessBoolOp::BOOL_OUT:
          suf::ensure(node.index >= 0 &&
                          static_cast<std::size_t>(node.index) < source_bool_.size(),
                      "Sigma strict Phi BOOL_OUT index out of range");
          values.push_back(source_bool_[static_cast<std::size_t>(node.index)]);
          break;
        case suf::PostprocessBoolOp::KAPPA_B:
          values.push_back(kappa_bool_value(node.index));
          break;
        case suf::PostprocessBoolOp::NOT:
          values.push_back(not_value(values[static_cast<std::size_t>(node.lhs)]));
          break;
        case suf::PostprocessBoolOp::XOR:
          values.push_back(xor_value(values[static_cast<std::size_t>(node.lhs)],
                                     values[static_cast<std::size_t>(node.rhs)]));
          break;
        case suf::PostprocessBoolOp::AND:
          values.push_back(and_value(values[static_cast<std::size_t>(node.lhs)],
                                     values[static_cast<std::size_t>(node.rhs)]));
          break;
        case suf::PostprocessBoolOp::OR:
          values.push_back(or_value(values[static_cast<std::size_t>(node.lhs)],
                                    values[static_cast<std::size_t>(node.rhs)]));
          break;
        case suf::PostprocessBoolOp::A2B:
          values.push_back(a2b_value(eval_arith_expr(node.index), node.bit_index));
          break;
      }
    }
    suf::ensure(expr.root >= 0 && static_cast<std::size_t>(expr.root) < values.size(),
                "Sigma strict Phi malformed Boolean expression root");
    return values[static_cast<std::size_t>(expr.root)];
  }

  SufSigmaCompiledOperatorResult* finish(std::vector<StrictDeviceValue>& final_arith,
                                         std::vector<StrictDeviceValue>& final_bool) {
    const int caps = capability_flags_for_spec(spec_);
    auto* result = make_empty_v2_result(n_, final_arith.size(), final_bool.size(), caps);
    std::unordered_set<std::uint64_t*> keep;

    auto keep_output = [&](StrictDeviceValue value) -> std::uint64_t* {
      auto* ptr = value.ptr;
      if (!is_owned(ptr)) {
        ptr = track(clone_device_words(ptr, n_));
      }
      if (keep.find(ptr) != keep.end()) {
        ptr = track(clone_device_words(ptr, n_));
      }
      keep.insert(ptr);
      return ptr;
    };

    for (std::size_t i = 0; i < final_arith.size(); ++i) {
      result->d_arithmetic[i] = keep_output(final_arith[i]);
    }
    for (std::size_t i = 0; i < final_bool.size(); ++i) {
      result->d_boolean[i] = keep_output(final_bool[i]);
    }
    for (auto* ptr : owned_) {
      if (ptr && keep.find(ptr) == keep.end()) {
        gpuFree(ptr);
      }
    }
    owned_.clear();
    return result;
  }

  bool keygen_ = true;
  SigmaPeer* peer_ = nullptr;
  int operator_id_ = 0;
  const suf::OperatorSpecification& spec_;
  int party_ = 0;
  int bw_ = 0;
  int scale_ = 0;
  const std::uint64_t* d_input_ = nullptr;
  std::size_t n_ = 0;
  Stats* stats_ = nullptr;
  const SufSigmaPostprocessContext* postprocess_ctx_ = nullptr;
  int degree_ = 0;
  std::size_t source_arith_outputs_ = 0;
  std::size_t source_bool_outputs_ = 0;
  std::size_t aux_words_ = 0;
  std::size_t payload_words_ = 0;
  bool needs_identity_lookup_ = false;
  StrictDeviceValue x_;
  StrictDeviceValue x_hat_;
  std::vector<StrictDeviceValue> payload_;
  std::vector<StrictDeviceValue> source_arith_;
  std::vector<StrictDeviceValue> source_bool_;
  std::vector<std::uint64_t*> owned_;
};

SufSigmaCompiledOperatorResult* run_strict_compiled_operator(bool keygen,
                                                             SigmaPeer* peer,
                                                             int operator_id,
                                                             int party,
                                                             int bw,
                                                             int scale,
                                                             const std::uint64_t* d_input,
                                                             std::size_t n,
                                                             Stats* s,
                                                             const SufSigmaPostprocessContext* ctx) {
  const auto rec = get_generic_operator_record(operator_id);
  StrictCompiledExecutor exec(keygen, peer, operator_id, rec.spec, party, bw, scale,
                              d_input, n, s, ctx);
  return exec.run();
}

void consume_one_lut_key_to_pending(std::size_t expected_n,
                                    const char* label,
                                    const PendingLutMeta& meta = unknown_lut_meta()) {
  suf::ensure(keybuf_ready(), "FuseFSS key buffer is not initialized");
  auto* key_begin = *g_keybuf_ptr;
  auto* key_end = key_begin;
  auto lut_key = readGPULUTKey<std::uint64_t>(&key_end);
  if (expected_n != 0) {
    suf::ensure(lut_key.k.M == static_cast<int>(expected_n),
                "FuseFSS typed key consume vector length mismatch");
  }
  if (meta.kind != PendingLutKind::Unknown) {
    suf::ensure(lut_key.k.bin == meta.bin,
                "FuseFSS typed key consume input width metadata mismatch");
    suf::ensure(lut_key.bout == meta.bout,
                "FuseFSS typed key consume output width metadata mismatch");
    suf::ensure(lut_key.k.M == static_cast<int>(meta.n),
                "FuseFSS typed key consume vector length metadata mismatch");
  }
  release_lut_key(lut_key);
  queue_lut_key(key_begin, meta);
  *g_keybuf_ptr = key_end;
  if (env_enabled("SUF_DEBUG")) {
    std::fprintf(stderr, "[FuseFSS] queued %s LUT key bytes=%zu\n",
                 label ? label : "generic",
                 static_cast<std::size_t>(key_end - key_begin));
  }
}

void consume_one_vector_lut_key_to_pending(std::size_t expected_n,
                                           int expected_words,
                                           const char* label,
                                           const PendingLutMeta& meta = unknown_lut_meta()) {
  suf::ensure(keybuf_ready(), "FuseFSS key buffer is not initialized");
  auto* key_begin = *g_keybuf_ptr;
  auto* key_end = key_begin;
  auto lut_key = readGPUVectorLUTKey<std::uint64_t>(&key_end);
  if (expected_n != 0) {
    suf::ensure(lut_key.k.M == static_cast<int>(expected_n),
                "FuseFSS typed vector key consume vector length mismatch");
  }
  if (expected_words != 0) {
    suf::ensure(lut_key.outWords == expected_words,
                "FuseFSS typed vector key consume payload width mismatch");
  }
  if (meta.kind != PendingLutKind::Unknown) {
    suf::ensure(lut_key.k.bin == meta.bin,
                "FuseFSS typed vector key consume input width metadata mismatch");
    suf::ensure(lut_key.bout == meta.bout,
                "FuseFSS typed vector key consume output width metadata mismatch");
    suf::ensure(lut_key.k.M == static_cast<int>(meta.n),
                "FuseFSS typed vector key consume vector length metadata mismatch");
    suf::ensure(lut_key.outWords == static_cast<int>(meta.out_words),
                "FuseFSS typed vector key consume payload width metadata mismatch");
  }
  release_vector_lut_key(lut_key);
  queue_lut_key(key_begin, meta);
  *g_keybuf_ptr = key_end;
  if (env_enabled("SUF_DEBUG")) {
    std::fprintf(stderr, "[FuseFSS] queued %s vector LUT key bytes=%zu words=%d\n",
                 label ? label : "generic-vector",
                 static_cast<std::size_t>(key_end - key_begin),
                 expected_words);
  }
}

void consume_strict_compiled_operator_key(int operator_id,
                                          int bw,
                                          std::size_t n,
                                          const char* label) {
  const auto rec = get_generic_operator_record(operator_id);
  const int degree = max_degree_for_spec(rec.spec);
  const std::size_t source_arith_outputs =
      suf::operator_spec_arithmetic_outputs(rec.spec);
  const std::size_t source_bool_outputs =
      suf::operator_spec_boolean_outputs(rec.spec);
  const std::size_t aux_words = suf::operator_spec_aux_words(rec.spec);
  const std::size_t payload_words =
      source_arith_outputs * static_cast<std::size_t>(degree + 1) + aux_words;
  const auto phi_cost = suf::count_postprocess_cost(rec.spec.postprocess);

  // These consume hooks are used by Sigma key-struct readers for built-in
  // softmax/layernorm operators. Full Phi post-processing is consumed directly
  // by the v2 executor when it is called without key-struct pre-reading.
  suf::ensure(degree == 0 && phi_cost.ring_multiplications == 0 &&
                  phi_cost.boolean_ands == 0 && phi_cost.b2a_conversions == 0 &&
                  phi_cost.a2b_conversions == 0,
              "FuseFSS strict key consume only supports LUT-only builtins");

  if (strict_spec_requires_identity_lookup(rec.spec)) {
    consume_one_lut_key_to_pending(
        n, label,
        strict_lut_meta(operator_id, StrictTableKind::IdentityX, 0,
                        /*bit_width=*/bw,
                        rec.spec.in_bits, bw, n));
  }
  const VectorLutPlan payload_plan{
      .kind = StrictTableKind::PayloadWord,
      .word_start = 0,
      .word_count = static_cast<int>(payload_words),
      .bit_width = bw,
      .fused = !env_enabled("SUF_SIGMA_VECTOR_LUT_LEGACY") &&
               fused_vector_lut_shape_supported(payload_words, n, bw)};
  if (payload_plan.fused) {
    consume_one_vector_lut_key_to_pending(
        n, payload_plan.word_count, label,
        strict_payload_vector_lut_meta(operator_id, payload_plan.word_start,
                                       payload_plan.word_count,
                                       /*bit_width=*/bw,
                                       rec.spec.in_bits, bw, n));
  } else {
    for (std::size_t i = 0; i < payload_words; ++i) {
      consume_one_lut_key_to_pending(
          n, label,
          strict_lut_meta(operator_id, StrictTableKind::PayloadWord,
                          static_cast<int>(i),
                          /*bit_width=*/bw,
                          rec.spec.in_bits, bw, n));
    }
  }
  for (std::size_t i = 0; i < source_bool_outputs; ++i) {
    consume_one_lut_key_to_pending(
        n, label,
        strict_lut_meta(operator_id, StrictTableKind::SourceBool,
                        static_cast<int>(i),
                        /*bit_width=*/bw,
                        rec.spec.in_bits, bw, n));
  }
}

} // namespace

extern "C" void suf_sigma_reset_keygen() {
  clear_device_table_cache();
  clear_pending_lut_keys();
}

extern "C" void suf_sigma_reset_eval() {
  clear_pending_lut_keys();
}

extern "C" void suf_sigma_clear() {
  clear_device_table_cache();
  clear_pending_lut_keys();
}

extern "C" void suf_sigma_consume_key() {
  if (!keybuf_ready()) return;
  consume_one_lut_key_to_pending(0, "single");
}

extern "C" void suf_sigma_consume_nexp_key(int bw, int scale, std::size_t n) {
  if (!use_strict_generic_hooks()) {
    suf_sigma_consume_key();
    return;
  }
  const double xmax = env_double("SUF_NEXP_XMAX", 16.0);
  const std::uint64_t clamp_raw =
      static_cast<std::uint64_t>(llroundl(xmax * (1ULL << scale)));
  int in_bits = env_int("SUF_NEXP_BITS", 0);
  if (in_bits <= 0) {
    in_bits = std::min<int>(16, std::min<int>(bw, bits_needed(clamp_raw)));
  }
  const std::uint64_t clamp_max =
      std::min<std::uint64_t>(clamp_raw, mask_for_bw(in_bits));
  const auto tkey = make_table_key(GateKind::NExp, bw, scale, in_bits, scale,
                                   0, clamp_max, 0);
  consume_strict_compiled_operator_key(
      get_or_register_builtin_generic_operator(tkey), bw, n, "strict-nexp");
}

extern "C" void suf_sigma_consume_inverse_key(int bw,
                                               int scale,
                                               int nmax,
                                               std::size_t n) {
  if (!use_strict_generic_hooks()) {
    suf_sigma_consume_key();
    return;
  }
  const int scale_in = env_int("SUF_INV_FRAC", 6);
  int in_bits = env_int("SUF_INV_BITS", 0);
  const std::uint64_t max_fixed =
      (nmax > 0) ? (static_cast<std::uint64_t>(nmax) << scale_in) : 0;
  if (in_bits <= 0) {
    std::uint64_t tmp = max_fixed;
    int bits = 0;
    while (tmp > 0) {
      ++bits;
      tmp >>= 1;
    }
    in_bits = std::max(1, std::min(16, bits));
  }
  const std::uint64_t clamp_min = (1ULL << scale_in);
  std::uint64_t clamp_max =
      std::min<std::uint64_t>(max_fixed, mask_for_bw(in_bits));
  if (clamp_max < clamp_min) clamp_max = clamp_min;
  const auto tkey = make_table_key(GateKind::Inv, bw, scale, in_bits, scale_in,
                                   clamp_min, clamp_max,
                                   static_cast<std::uint64_t>(nmax));
  consume_strict_compiled_operator_key(
      get_or_register_builtin_generic_operator(tkey), bw, n, "strict-inv");
}

extern "C" void suf_sigma_consume_rsqrt_key(int bw,
                                             int scale,
                                             int extradiv,
                                             std::size_t n) {
  if (!use_strict_generic_hooks()) {
    suf_sigma_consume_key();
    return;
  }
  const int target_frac = env_int("SUF_RSQRT_FRAC", 6);
  const int shift = std::max(0, 2 * scale - target_frac);
  const int max_bits = std::max(1, std::min(16, bw - shift));
  const int scale_in = 2 * scale - shift;
  const double vmax_real = env_double("SUF_RSQRT_VMAX", 16.0);
  const double eps_real = env_double("SUF_RSQRT_EPS", 0.0);
  const std::uint64_t clamp_min = std::max<std::uint64_t>(
      1, static_cast<std::uint64_t>(llroundl(eps_real * (1ULL << scale_in))));
  const std::uint64_t vmax_fixed =
      static_cast<std::uint64_t>(llroundl(vmax_real * (1ULL << scale_in)));
  int in_bits = env_int("SUF_RSQRT_BITS", 0);
  if (in_bits <= 0) {
    in_bits = bits_needed(vmax_fixed);
    in_bits = std::max(8, std::min(max_bits, in_bits));
  } else {
    in_bits = std::max(1, std::min(max_bits, in_bits));
  }
  std::uint64_t clamp_max =
      std::min<std::uint64_t>(vmax_fixed, mask_for_bw(in_bits));
  if (clamp_max < clamp_min) clamp_max = clamp_min;
  const auto tkey = make_table_key(GateKind::Rsqrt, bw, scale, in_bits, scale_in,
                                   clamp_min, clamp_max,
                                   static_cast<std::uint64_t>(extradiv));
  consume_strict_compiled_operator_key(
      get_or_register_builtin_generic_operator(tkey), bw, n, "strict-rsqrt");
}

extern "C" int suf_sigma_register_operator_spec(const suf::OperatorSpecification* spec) {
  suf::ensure(spec != nullptr, "Sigma generic operator registration received null spec");
  return register_operator_spec_copy(*spec);
}

extern "C" int suf_sigma_register_operator_spec_with_id(int operator_id,
                                                        const suf::OperatorSpecification* spec) {
  suf::ensure(spec != nullptr, "Sigma generic operator registration received null spec");
  return register_operator_spec_copy_with_id(operator_id, *spec);
}

extern "C" int suf_sigma_compiled_operator_capability_flags(int operator_id) {
  const auto rec = get_generic_operator_record(operator_id);
  return capability_flags_for_spec(rec.spec);
}

extern "C" bool suf_sigma_compiled_operator_supported(int operator_id) {
  return (suf_sigma_compiled_operator_capability_flags(operator_id) &
          SUF_SIGMA_CAP_SUPPORTED) != 0;
}

extern "C" bool suf_sigma_compiled_operator_strict_supported(int operator_id) {
  return (suf_sigma_compiled_operator_capability_flags(operator_id) &
          SUF_SIGMA_CAP_STRICT_SUPPORTED) != 0;
}

extern "C" std::uint64_t* suf_sigma_keygen_compiled_operator(int operator_id,
                                                             int party,
                                                             int bw,
                                                             int scale,
                                                             const std::uint64_t* d_input_mask,
                                                             std::size_t n) {
  return keygen_compiled_operator_u64(operator_id, party, bw, scale, d_input_mask, n);
}

extern "C" std::uint64_t* suf_sigma_eval_compiled_operator(SigmaPeer* peer,
                                                           int operator_id,
                                                           int party,
                                                           int bw,
                                                           int scale,
                                                           const std::uint64_t* d_input_masked,
                                                           std::size_t n,
                                                           Stats* s) {
  return eval_compiled_operator_u64(peer, operator_id, party, bw, scale,
                                    d_input_masked, n, s, true);
}

extern "C" SufSigmaCompiledOperatorResult* suf_sigma_keygen_compiled_operator_v2(
    int operator_id,
    int party,
    int bw,
    int scale,
    const std::uint64_t* d_input_mask,
    std::size_t n) {
  return run_strict_compiled_operator(true, nullptr, operator_id, party, bw, scale,
                                      d_input_mask, n, nullptr, nullptr);
}

extern "C" SufSigmaCompiledOperatorResult* suf_sigma_eval_compiled_operator_v2(
    SigmaPeer* peer,
    int operator_id,
    int party,
    int bw,
    int scale,
    const std::uint64_t* d_input_masked,
    std::size_t n,
    Stats* s) {
  return run_strict_compiled_operator(false, peer, operator_id, party, bw, scale,
                                      d_input_masked, n, s, nullptr);
}

extern "C" SufSigmaCompiledOperatorResult* suf_sigma_keygen_compiled_operator_v3(
    int operator_id,
    int party,
    int bw,
    int scale,
    const std::uint64_t* d_input_mask,
    std::size_t n,
    const SufSigmaPostprocessContext* ctx) {
  return run_strict_compiled_operator(true, nullptr, operator_id, party, bw, scale,
                                      d_input_mask, n, nullptr, ctx);
}

extern "C" SufSigmaCompiledOperatorResult* suf_sigma_eval_compiled_operator_v3(
    SigmaPeer* peer,
    int operator_id,
    int party,
    int bw,
    int scale,
    const std::uint64_t* d_input_masked,
    std::size_t n,
    Stats* s,
    const SufSigmaPostprocessContext* ctx) {
  return run_strict_compiled_operator(false, peer, operator_id, party, bw, scale,
                                      d_input_masked, n, s, ctx);
}

extern "C" void suf_sigma_free_compiled_operator_result_v2(
    SufSigmaCompiledOperatorResult* result) {
  if (!result) return;
  std::unordered_set<std::uint64_t*> freed;
  for (std::size_t i = 0; i < result->arithmetic_outputs; ++i) {
    auto* ptr = result->d_arithmetic ? result->d_arithmetic[i] : nullptr;
    if (ptr && freed.insert(ptr).second) gpuFree(ptr);
  }
  for (std::size_t i = 0; i < result->boolean_outputs; ++i) {
    auto* ptr = result->d_boolean ? result->d_boolean[i] : nullptr;
    if (ptr && freed.insert(ptr).second) gpuFree(ptr);
  }
  delete[] result->d_arithmetic;
  delete[] result->d_boolean;
  delete result;
}

extern "C" bool suf_sigma_postprocess_mul_supported() {
  return true;
}

extern "C" bool suf_sigma_postprocess_and_supported() {
  return true;
}

extern "C" bool suf_sigma_postprocess_b2a_supported() {
  return true;
}

extern "C" bool suf_sigma_postprocess_a2b_supported() {
  return true;
}

extern "C" std::uint64_t* suf_sigma_keygen_postprocess_mul_u64(int party,
                                                               int bw,
                                                               int scale,
                                                               const std::uint64_t* d_lhs_mask,
                                                               const std::uint64_t* d_rhs_mask,
                                                               std::size_t n) {
  (void)scale;
  if (!g_keybuf_ptr || !*g_keybuf_ptr) return nullptr;
  ensure_aes_ready();
  suf::ensure(d_lhs_mask != nullptr && d_rhs_mask != nullptr,
              "Sigma MUL keygen received null mask");
  suf::ensure(bw > 0 && bw <= 64, "Sigma MUL keygen bw must be 1..64");
  suf::ensure(bw > 2, "Sigma MUL production path requires arithmetic bw>2");
  suf::ensure(n <= static_cast<std::size_t>(std::numeric_limits<int>::max()),
              "Sigma MUL keygen length exceeds int range");
  return gpuKeygenMul<std::uint64_t>(
      reinterpret_cast<u8**>(g_keybuf_ptr), party, bw, /*scale=*/0,
      static_cast<int>(n), const_cast<std::uint64_t*>(d_lhs_mask),
      const_cast<std::uint64_t*>(d_rhs_mask), TruncateType::None, &g_aes);
}

extern "C" std::uint64_t* suf_sigma_eval_postprocess_mul_u64(SigmaPeer* peer,
                                                             int party,
                                                             int bw,
                                                             int scale,
                                                             const std::uint64_t* d_lhs,
                                                             const std::uint64_t* d_rhs,
                                                             std::size_t n,
                                                             Stats* s) {
  (void)scale;
  if (!keybuf_ready()) return nullptr;
  ensure_aes_ready();
  suf::ensure(peer != nullptr, "Sigma MUL eval received null peer");
  suf::ensure(d_lhs != nullptr && d_rhs != nullptr, "Sigma MUL eval received null input");
  suf::ensure(bw > 0 && bw <= 64, "Sigma MUL eval bw must be 1..64");
  suf::ensure(bw > 2, "Sigma MUL production path requires arithmetic bw>2");
  suf::ensure(n <= static_cast<std::size_t>(std::numeric_limits<int>::max()),
              "Sigma MUL eval length exceeds int range");
  auto key = read_mul_key_none<std::uint64_t>(g_keybuf_ptr, static_cast<int>(n));
  return gpuMul<std::uint64_t>(
      peer, party, bw, /*scale=*/0, static_cast<int>(n), key,
      const_cast<std::uint64_t*>(d_lhs), const_cast<std::uint64_t*>(d_rhs),
      TruncateType::None, &g_aes, s);
}

extern "C" std::uint64_t* suf_sigma_keygen_postprocess_and_u64(int party,
                                                               const std::uint64_t* d_lhs_mask,
                                                               const std::uint64_t* d_rhs_mask,
                                                               std::size_t n) {
  if (!g_keybuf_ptr || !*g_keybuf_ptr) return nullptr;
  suf::ensure(d_lhs_mask != nullptr && d_rhs_mask != nullptr,
              "Sigma AND keygen received null mask");
  suf::ensure(n <= static_cast<std::size_t>(std::numeric_limits<int>::max()),
              "Sigma AND keygen length exceeds int range");
  return gpuKeyGenAnd<u64>(reinterpret_cast<u8**>(g_keybuf_ptr), party, 1,
                           static_cast<int>(n),
                           const_cast<u64*>(d_lhs_mask),
                           const_cast<u64*>(d_rhs_mask));
}

extern "C" std::uint64_t* suf_sigma_eval_postprocess_and_u64(SigmaPeer* peer,
                                                             int party,
                                                             const std::uint64_t* d_lhs,
                                                             const std::uint64_t* d_rhs,
                                                             std::size_t n,
                                                             Stats* s) {
  if (!keybuf_ready()) return nullptr;
  suf::ensure(peer != nullptr, "Sigma AND eval received null peer");
  suf::ensure(d_lhs != nullptr && d_rhs != nullptr, "Sigma AND eval received null input");
  suf::ensure(n <= static_cast<std::size_t>(std::numeric_limits<int>::max()),
              "Sigma AND eval length exceeds int range");

  const auto k = read_and_key(g_keybuf_ptr);
  suf::ensure(k.N == static_cast<int>(n), "Sigma AND key vector length mismatch");
  const int num_ints = (k.N - 1) / PACKING_SIZE + 1;
  const std::size_t key_bytes = static_cast<std::size_t>(3 * num_ints) * sizeof(std::uint32_t);
  auto* d_key = reinterpret_cast<std::uint32_t*>(
      moveToGPU(reinterpret_cast<std::uint8_t*>(k.b0), key_bytes, s));
  auto* d_a = d_key;
  auto* d_b = d_a + num_ints;
  auto* d_c = d_b + num_ints;
  auto* d_out_packed = reinterpret_cast<std::uint32_t*>(
      gpuMalloc(static_cast<std::size_t>(num_ints) * sizeof(std::uint32_t)));
  const int threads = 256;
  const int word_blocks = (num_ints + threads - 1) / threads;
  const auto bytes0 = peer->bytesSent() + peer->bytesReceived();
  kernel_eval_and_packed<<<word_blocks, threads>>>(party, static_cast<int>(n), num_ints,
                                                   d_lhs, d_rhs, d_a, d_b, d_c,
                                                   d_out_packed);
  cudaDeviceSynchronize();
  gpuFree(d_key);
  peer->reconstructInPlace(d_out_packed, 1, n, s);
  const auto bytes1 = peer->bytesSent() + peer->bytesReceived();
  if (s) {
    s->linear_comm_bytes += (bytes1 - bytes0);
  }
  auto* d_out = reinterpret_cast<std::uint64_t*>(gpuMalloc(n * sizeof(std::uint64_t)));
  const int blocks = static_cast<int>((n + threads - 1) / threads);
  kernel_unpack_bits_to_u64<<<blocks, threads>>>(d_out_packed, d_out, static_cast<int>(n));
  cudaDeviceSynchronize();
  gpuFree(d_out_packed);
  return d_out;
}

extern "C" std::uint64_t* suf_sigma_keygen_postprocess_b2a_u64(int party,
                                                               int bw,
                                                               const std::uint64_t* d_bool_mask,
                                                               std::size_t n) {
  if (!g_keybuf_ptr || !*g_keybuf_ptr) return nullptr;
  suf::ensure(d_bool_mask != nullptr, "Sigma B2A keygen received null mask");
  suf::ensure(bw > 0 && bw <= 64, "Sigma B2A keygen bw must be 1..64");
  suf::ensure(n <= static_cast<std::size_t>(std::numeric_limits<int>::max()),
              "Sigma B2A keygen length exceeds int range");
  write_key_int(g_keybuf_ptr, static_cast<int>(n));
  auto* d_out_mask = randomGEOnGpu<std::uint64_t>(static_cast<u64>(n), bw);
  auto* d_r_share = randomGEOnGpu<std::uint64_t>(static_cast<u64>(n), bw);
  auto* d_m_share = randomGEOnGpu<std::uint64_t>(static_cast<u64>(n), bw);
  const int threads = 256;
  const int blocks = static_cast<int>((n + threads - 1) / threads);
  kernel_b2a_finalize_key_shares<<<blocks, threads>>>(
      party, d_bool_mask, d_out_mask, d_r_share, d_m_share, mask_for_bw(bw), n);
  cudaDeviceSynchronize();
  const std::size_t bytes = n * sizeof(std::uint64_t);
  moveIntoCPUMem(*g_keybuf_ptr, reinterpret_cast<std::uint8_t*>(d_r_share), bytes, nullptr);
  *g_keybuf_ptr += bytes;
  moveIntoCPUMem(*g_keybuf_ptr, reinterpret_cast<std::uint8_t*>(d_m_share), bytes, nullptr);
  *g_keybuf_ptr += bytes;
  gpuFree(d_r_share);
  gpuFree(d_m_share);
  return d_out_mask;
}

extern "C" std::uint64_t* suf_sigma_eval_postprocess_b2a_u64(SigmaPeer* peer,
                                                             int party,
                                                             int bw,
                                                             const std::uint64_t* d_bool_open,
                                                             std::size_t n,
                                                             Stats* s) {
  if (!keybuf_ready()) return nullptr;
  ensure_aes_ready();
  suf::ensure(peer != nullptr, "Sigma B2A eval received null peer");
  suf::ensure(d_bool_open != nullptr, "Sigma B2A eval received null input");
  suf::ensure(bw > 0 && bw <= 64, "Sigma B2A eval bw must be 1..64");
  suf::ensure(n <= static_cast<std::size_t>(std::numeric_limits<int>::max()),
              "Sigma B2A eval length exceeds int range");
  const int key_n = read_key_int(g_keybuf_ptr);
  suf::ensure(key_n == static_cast<int>(n), "Sigma B2A key vector length mismatch");
  const std::size_t bytes = n * sizeof(std::uint64_t);
  auto* d_r_share = reinterpret_cast<std::uint64_t*>(
      moveToGPU(*g_keybuf_ptr, bytes, s));
  *g_keybuf_ptr += bytes;
  auto* d_m_share = reinterpret_cast<std::uint64_t*>(
      moveToGPU(*g_keybuf_ptr, bytes, s));
  *g_keybuf_ptr += bytes;
  auto* d_out = alloc_device_words(n);
  const int threads = 256;
  const int blocks = static_cast<int>((n + threads - 1) / threads);
  const auto bytes0 = peer->bytesSent() + peer->bytesReceived();
  kernel_b2a_eval_share<<<blocks, threads>>>(party, d_bool_open, d_r_share,
                                             d_m_share, d_out, mask_for_bw(bw), n);
  cudaDeviceSynchronize();
  gpuFree(d_r_share);
  gpuFree(d_m_share);
  peer->reconstructInPlace(d_out, bw, n, s);
  const auto bytes1 = peer->bytesSent() + peer->bytesReceived();
  if (s) {
    s->linear_comm_bytes += (bytes1 - bytes0);
  }
  return d_out;
}

extern "C" std::uint64_t* suf_sigma_keygen_postprocess_a2b_lsb_u64(
    int party,
    const std::uint64_t* d_arith_mask,
    std::size_t n) {
  return suf_sigma_keygen_postprocess_a2b_bit_u64(party, 64, 0, d_arith_mask, n);
}

extern "C" std::uint64_t* suf_sigma_eval_postprocess_a2b_lsb_u64(
    SigmaPeer* peer,
    int party,
    const std::uint64_t* d_arith_open,
    std::size_t n,
    Stats* s) {
  return suf_sigma_eval_postprocess_a2b_bit_u64(peer, party, 64, 0, d_arith_open, n, s);
}

extern "C" std::uint64_t* suf_sigma_keygen_postprocess_a2b_bit_u64(
    int party,
    int bw,
    int bit_index,
    const std::uint64_t* d_arith_mask,
    std::size_t n) {
  suf::ensure(d_arith_mask != nullptr, "Sigma A2B keygen received null mask");
  suf::ensure(bw > 0 && bw <= 64, "Sigma A2B keygen bw must be 1..64");
  suf::ensure(bit_index >= 0 && bit_index < bw, "Sigma A2B keygen bit index out of range");
  suf::ensure(n <= static_cast<std::size_t>(std::numeric_limits<int>::max()),
              "Sigma A2B keygen length exceeds int range");
  if (bit_index == 0) {
    return local_lsb_bits(d_arith_mask, n);
  }
  suf::ensure(bw <= 16, "Sigma A2B non-LSB production path currently supports bw<=16");
  suf::ensure(g_keybuf_ptr && *g_keybuf_ptr, "Sigma A2B keygen key buffer is not initialized");
  ensure_aes_ready();
  write_key_int(g_keybuf_ptr, static_cast<int>(n));
  write_key_int(g_keybuf_ptr, bw);
  write_key_int(g_keybuf_ptr, bit_index);

  auto* d_input_u16 = reinterpret_cast<std::uint16_t*>(gpuMalloc(n * sizeof(std::uint16_t)));
  const int threads = 256;
  const int blocks = static_cast<int>((n + threads - 1) / threads);
  kernel_u64_to_u16<<<blocks, threads>>>(d_arith_mask, d_input_u16, bw, n);
  cudaDeviceSynchronize();
  auto* raw = gpuKeyGenLUT<std::uint16_t, std::uint64_t>(
      g_keybuf_ptr, party, bw, bw, static_cast<int>(n), d_input_u16, &g_aes);
  gpuFree(d_input_u16);
  auto* out = local_lsb_bits(raw, n);
  gpuFree(raw);
  return out;
}

extern "C" std::uint64_t* suf_sigma_eval_postprocess_a2b_bit_u64(
    SigmaPeer* peer,
    int party,
    int bw,
    int bit_index,
    const std::uint64_t* d_arith_open,
    std::size_t n,
    Stats* s) {
  (void)party;
  suf::ensure(d_arith_open != nullptr, "Sigma A2B eval received null input");
  suf::ensure(bw > 0 && bw <= 64, "Sigma A2B eval bw must be 1..64");
  suf::ensure(bit_index >= 0 && bit_index < bw, "Sigma A2B eval bit index out of range");
  suf::ensure(n <= static_cast<std::size_t>(std::numeric_limits<int>::max()),
              "Sigma A2B eval length exceeds int range");
  if (bit_index == 0) {
    return local_lsb_bits(d_arith_open, n);
  }
  suf::ensure(peer != nullptr, "Sigma A2B eval received null peer");
  suf::ensure(bw <= 16, "Sigma A2B non-LSB production path currently supports bw<=16");
  suf::ensure(keybuf_ready(), "Sigma A2B eval key buffer is not initialized");
  ensure_aes_ready();
  const int key_n = read_key_int(g_keybuf_ptr);
  const int key_bw = read_key_int(g_keybuf_ptr);
  const int key_bit = read_key_int(g_keybuf_ptr);
  suf::ensure(key_n == static_cast<int>(n), "Sigma A2B key vector length mismatch");
  suf::ensure(key_bw == bw, "Sigma A2B key bit-width mismatch");
  suf::ensure(key_bit == bit_index, "Sigma A2B key bit index mismatch");

  auto lut_key = read_lut_key(false);
  suf::ensure(lut_key.bout == bw, "Sigma A2B LUT key output width mismatch");
  suf::ensure(lut_key.k.bin == bw, "Sigma A2B LUT key input width mismatch");
  suf::ensure(lut_key.k.M == static_cast<int>(n), "Sigma A2B LUT key length mismatch");

  auto* d_input_u16 = reinterpret_cast<std::uint16_t*>(gpuMalloc(n * sizeof(std::uint16_t)));
  const int threads = 256;
  const int blocks = static_cast<int>((n + threads - 1) / threads);
  kernel_u64_to_u16<<<blocks, threads>>>(d_arith_open, d_input_u16, bw, n);
  cudaDeviceSynchronize();
  auto* table = get_strict_device_table(
      StrictDeviceTableKey{.operator_id = 0,
                           .kind = static_cast<int>(StrictTableKind::A2BBit),
                           .word_index = bit_index,
                           .bit_width = bw});
  auto* raw = gpuDpfLUT<std::uint16_t, std::uint64_t>(
      lut_key, peer, party, d_input_u16, table, &g_aes, s);
  gpuFree(d_input_u16);
  release_lut_key(lut_key);
  auto* out = local_lsb_bits(raw, n);
  gpuFree(raw);
  return out;
}

static std::uint64_t* detach_single_arithmetic_result(SufSigmaCompiledOperatorResult* result) {
  suf::ensure(result != nullptr, "Sigma strict generic returned null result");
  suf::ensure(result->arithmetic_outputs == 1 && result->d_arithmetic != nullptr,
              "Sigma strict generic builtin requires exactly one arithmetic output");
  auto* out = result->d_arithmetic[0];
  result->d_arithmetic[0] = nullptr;
  suf_sigma_free_compiled_operator_result_v2(result);
  return out;
}

static std::uint64_t* strict_keygen_builtin_u64(int operator_id,
                                                int party,
                                                int bw,
                                                int scale,
                                                const std::uint64_t* d_input_mask,
                                                std::size_t n) {
  return detach_single_arithmetic_result(
      suf_sigma_keygen_compiled_operator_v2(operator_id, party, bw, scale, d_input_mask, n));
}

static std::uint64_t* strict_eval_builtin_u64(SigmaPeer* peer,
                                              int operator_id,
                                              int party,
                                              int bw,
                                              int scale,
                                              const std::uint64_t* d_input_masked,
                                              std::size_t n,
                                              Stats* s) {
  return detach_single_arithmetic_result(
      suf_sigma_eval_compiled_operator_v2(peer, operator_id, party, bw, scale,
                                          d_input_masked, n, s));
}

static std::uint64_t* strict_keygen_builtin_u16(int operator_id,
                                                int party,
                                                int bw,
                                                int scale,
                                                int in_bits,
                                                const std::uint16_t* d_input_mask,
                                                std::size_t n) {
  auto* d_input_u64 = alloc_device_words(n);
  const int threads = 256;
  const int blocks = static_cast<int>((n + threads - 1) / threads);
  kernel_u16_to_u64<<<blocks, threads>>>(d_input_mask, d_input_u64, in_bits, n);
  cudaDeviceSynchronize();
  auto* out = strict_keygen_builtin_u64(operator_id, party, bw, scale, d_input_u64, n);
  gpuFree(d_input_u64);
  return out;
}

static std::uint64_t* strict_eval_builtin_u16(SigmaPeer* peer,
                                              int operator_id,
                                              int party,
                                              int bw,
                                              int scale,
                                              int in_bits,
                                              const std::uint16_t* d_input_masked,
                                              std::size_t n,
                                              Stats* s) {
  auto* d_input_u64 = alloc_device_words(n);
  const int threads = 256;
  const int blocks = static_cast<int>((n + threads - 1) / threads);
  kernel_u16_to_u64<<<blocks, threads>>>(d_input_masked, d_input_u64, in_bits, n);
  cudaDeviceSynchronize();
  auto* out = strict_eval_builtin_u64(peer, operator_id, party, bw, scale, d_input_u64, n, s);
  gpuFree(d_input_u64);
  return out;
}

bool use_legacy_specialized_hooks() {
  return env_enabled("SUF_SIGMA_LEGACY_SPECIALIZED");
}

const char* gate_kind_name(GateKind kind) {
  switch (kind) {
    case GateKind::Gelu:
      return "gelu";
    case GateKind::Silu:
      return "silu";
    case GateKind::NExp:
      return "nexp";
    case GateKind::Inv:
      return "inv";
    case GateKind::Rsqrt:
      return "rsqrt";
  }
  return "unknown";
}

void emit_builtin_operator_metadata(int op_id,
                                    const TableKey& tkey,
                                    const char* lowering_name) {
  const char* path = env_value("SUF_SIGMA_OPERATOR_METADATA");
  if (!path || !*path) return;
  FILE* f = std::fopen(path, "a");
  if (!f) return;
  const int caps = capability_flags_for_spec(get_generic_operator_record(op_id).spec);
  std::fprintf(
      f,
      "{\"operator_id\":%d,\"operator\":\"%s\",\"shape_leakage\":{\"comparison_queries\":0,"
      "\"comparison_bit_widths\":[],\"interval_count\":%zu,\"payload_words\":1,"
      "\"arithmetic_outputs\":1,\"boolean_outputs\":0,\"final_arithmetic_outputs\":1,"
      "\"final_boolean_outputs\":0,\"ring_multiplications\":0,\"boolean_ands\":0,"
      "\"b2a_conversions\":0,\"a2b_conversions\":0},\"in_bits\":%d,"
      "\"out_bits\":%d,\"scale_in\":%d,\"scale_out\":%d,\"clamp_min\":%llu,"
      "\"clamp_max\":%llu,\"extra\":%llu,\"lowering_kind\":\"%s\","
      "\"capability_flags\":%d,\"strict_vector_lut\":\"%s\"}\n",
      op_id, gate_kind_name(tkey.kind), table_size_for_bits(tkey.in_bits), tkey.in_bits,
      tkey.out_bits, tkey.scale_in, tkey.scale_out,
      static_cast<unsigned long long>(tkey.clamp_min),
      static_cast<unsigned long long>(tkey.clamp_max),
      static_cast<unsigned long long>(tkey.extra),
      lowering_name ? lowering_name : "optimized-table", caps,
      env_enabled("SUF_SIGMA_VECTOR_LUT_LEGACY") ? "legacy" : "fused");
  std::fclose(f);
}

int ensure_builtin_generic_for_optimized_lowering(const TableKey& tkey,
                                                  const char* lowering_name) {
  if (use_legacy_specialized_hooks()) return 0;
  const int op_id = get_or_register_builtin_generic_operator(tkey);
  emit_builtin_operator_metadata(op_id, tkey, lowering_name);
  if (env_enabled("SUF_DEBUG")) {
    std::fprintf(stderr,
                 "[FuseFSS] production generic operator=%d lowering=%s kind=%d bw=%d scale=%d in_bits=%d scale_in=%d extra=%llu\n",
                 op_id, lowering_name ? lowering_name : "optimized-table",
                 static_cast<int>(tkey.kind), tkey.out_bits, tkey.scale_out,
                 tkey.in_bits, tkey.scale_in,
                 static_cast<unsigned long long>(tkey.extra));
  }
  return op_id;
}

std::uint64_t* keygen_builtin_generic_optimized_u64(const TableKey& tkey,
                                                    const char* lowering_name,
                                                    int party,
                                                    const std::uint64_t* d_input_mask,
                                                    std::size_t n) {
  ensure_builtin_generic_for_optimized_lowering(tkey, lowering_name);
  return keygen_table_gate_u64(tkey.kind, tkey.out_bits, tkey.scale_out,
                               tkey.in_bits, tkey.scale_in,
                               tkey.clamp_min, tkey.clamp_max, tkey.extra,
                               party, d_input_mask, n);
}

std::uint64_t* keygen_builtin_generic_optimized_u16(const TableKey& tkey,
                                                    const char* lowering_name,
                                                    int party,
                                                    const std::uint16_t* d_input_mask,
                                                    std::size_t n) {
  ensure_builtin_generic_for_optimized_lowering(tkey, lowering_name);
  return keygen_table_gate_u16(tkey.kind, tkey.out_bits, tkey.scale_out,
                               tkey.in_bits, tkey.scale_in,
                               tkey.clamp_min, tkey.clamp_max, tkey.extra,
                               party, d_input_mask, n);
}

std::uint64_t* eval_builtin_generic_optimized_u64(const TableKey& tkey,
                                                  const char* lowering_name,
                                                  SigmaPeer* peer,
                                                  int party,
                                                  const std::uint64_t* d_input_masked,
                                                  std::size_t n,
                                                  Stats* s,
                                                  bool prefer_queued_key = true) {
  ensure_builtin_generic_for_optimized_lowering(tkey, lowering_name);
  return eval_gate_u64(tkey.kind, tkey.out_bits, tkey.scale_out, tkey.scale_in,
                       tkey.in_bits, tkey.clamp_min, tkey.clamp_max, tkey.extra,
                       peer, party, d_input_masked, n, s, prefer_queued_key);
}

std::uint64_t* eval_builtin_generic_optimized_u16(const TableKey& tkey,
                                                  const char* lowering_name,
                                                  SigmaPeer* peer,
                                                  int party,
                                                  const std::uint16_t* d_input_masked,
                                                  std::size_t n,
                                                  Stats* s,
                                                  bool prefer_queued_key = true) {
  ensure_builtin_generic_for_optimized_lowering(tkey, lowering_name);
  return eval_gate_u16(tkey.kind, tkey.out_bits, tkey.scale_out, tkey.scale_in,
                       tkey.in_bits, tkey.clamp_min, tkey.clamp_max, tkey.extra,
                       peer, party, d_input_masked, n, s, prefer_queued_key);
}

extern "C" std::uint64_t* suf_sigma_keygen_activation(int party,
                                                       int bw,
                                                       int scale,
                                                       bool silu,
                                                       const std::uint64_t* d_input_mask,
                                                       std::size_t n) {
  const int intervals = env_int(silu ? "SUF_SILU_INTERVALS" : "SUF_GELU_INTERVALS",
                                silu ? 1024 : 256);
  const int default_bits = bits_needed(static_cast<std::uint64_t>(intervals - 1));
  const int in_bits = env_int(silu ? "SUF_SILU_BITS" : "SUF_GELU_BITS", default_bits);
  const std::uint64_t clamp_max = mask_for_bw(in_bits);
  if (use_strict_generic_hooks()) {
    const auto tkey = make_table_key(silu ? GateKind::Silu : GateKind::Gelu,
                                     bw, scale, in_bits, scale, 0, clamp_max, 0);
    const int op_id = get_or_register_builtin_generic_operator(tkey);
    return strict_keygen_builtin_u64(op_id, party, bw, scale, d_input_mask, n);
  }
  if (env_enabled("SUF_SIGMA_GENERIC")) {
    const auto tkey = make_table_key(silu ? GateKind::Silu : GateKind::Gelu,
                                     bw, scale, in_bits, scale, 0, clamp_max, 0);
    const int op_id = get_or_register_builtin_generic_operator(tkey);
    return keygen_compiled_operator_u64(op_id, party, bw, scale, d_input_mask, n);
  }
  const auto tkey = make_table_key(silu ? GateKind::Silu : GateKind::Gelu,
                                   bw, scale, in_bits, scale, 0, clamp_max, 0);
  return keygen_builtin_generic_optimized_u64(tkey, silu ? "silu-dpf-lut" : "gelu-dpf-lut",
                                              party, d_input_mask, n);
}

extern "C" std::uint64_t* suf_sigma_keygen_nexp(int party,
                                                int bw,
                                                int scale,
                                                const std::uint64_t* d_input_mask,
                                                std::size_t n) {
  const double xmax = env_double("SUF_NEXP_XMAX", 16.0);
  const std::uint64_t clamp_raw = static_cast<std::uint64_t>(llroundl(xmax * (1ULL << scale)));
  int in_bits = env_int("SUF_NEXP_BITS", 0);
  if (in_bits <= 0) {
    in_bits = std::min<int>(16, std::min<int>(bw, bits_needed(clamp_raw)));
  }
  const std::uint64_t clamp_max = std::min<std::uint64_t>(clamp_raw, mask_for_bw(in_bits));
  if (use_strict_generic_hooks()) {
    const auto tkey = make_table_key(GateKind::NExp, bw, scale, in_bits, scale,
                                     0, clamp_max, 0);
    const int op_id = get_or_register_builtin_generic_operator(tkey);
    return strict_keygen_builtin_u64(op_id, party, bw, scale, d_input_mask, n);
  }
  if (env_enabled("SUF_SIGMA_GENERIC")) {
    const auto tkey = make_table_key(GateKind::NExp, bw, scale, in_bits, scale,
                                     0, clamp_max, 0);
    const int op_id = get_or_register_builtin_generic_operator(tkey);
    return keygen_compiled_operator_u64(op_id, party, bw, scale, d_input_mask, n);
  }
  const auto tkey = make_table_key(GateKind::NExp, bw, scale, in_bits, scale,
                                   0, clamp_max, 0);
  return keygen_builtin_generic_optimized_u64(tkey, "nexp-dpf-lut",
                                              party, d_input_mask, n);
}

extern "C" std::uint64_t* suf_sigma_eval_nexp(SigmaPeer* peer,
                                              int party,
                                              int bw,
                                              int scale,
                                              const std::uint64_t* d_input_masked,
                                              std::size_t n,
                                              Stats* s) {
  const double xmax = env_double("SUF_NEXP_XMAX", 16.0);
  const std::uint64_t clamp_raw = static_cast<std::uint64_t>(llroundl(xmax * (1ULL << scale)));
  int in_bits = env_int("SUF_NEXP_BITS", 0);
  if (in_bits <= 0) {
    in_bits = std::min<int>(16, std::min<int>(bw, bits_needed(clamp_raw)));
  }
  const std::uint64_t clamp_max = std::min<std::uint64_t>(clamp_raw, mask_for_bw(in_bits));
  if (use_strict_generic_hooks()) {
    const auto tkey = make_table_key(GateKind::NExp, bw, scale, in_bits, scale,
                                     0, clamp_max, 0);
    const int op_id = get_or_register_builtin_generic_operator(tkey);
    return strict_eval_builtin_u64(peer, op_id, party, bw, scale, d_input_masked, n, s);
  }
  if (env_enabled("SUF_SIGMA_GENERIC")) {
    const auto tkey = make_table_key(GateKind::NExp, bw, scale, in_bits, scale,
                                     0, clamp_max, 0);
    const int op_id = get_or_register_builtin_generic_operator(tkey);
    return eval_compiled_operator_u64(peer, op_id, party, bw, scale,
                                      d_input_masked, n, s, true);
  }
  const auto tkey = make_table_key(GateKind::NExp, bw, scale, in_bits, scale,
                                   0, clamp_max, 0);
  return eval_builtin_generic_optimized_u64(tkey, "nexp-dpf-lut",
                                            peer, party, d_input_masked, n, s);
}

extern "C" std::uint64_t* suf_sigma_keygen_inverse(int party,
                                                   int bw,
                                                   int scale,
                                                   int nmax,
                                                   const std::uint16_t* d_input_mask,
                                                   std::size_t n) {
  const int scale_in = env_int("SUF_INV_FRAC", 6);
  int in_bits = env_int("SUF_INV_BITS", 0);
  const std::uint64_t max_fixed = (nmax > 0) ? (static_cast<std::uint64_t>(nmax) << scale_in) : 0;
  if (in_bits <= 0) {
    std::uint64_t tmp = max_fixed;
    int bits = 0;
    while (tmp > 0) {
      ++bits;
      tmp >>= 1;
    }
    in_bits = std::max(1, std::min(16, bits));
  }
  const std::uint64_t clamp_min = (1ULL << scale_in);
  std::uint64_t clamp_max = std::min<std::uint64_t>(max_fixed, mask_for_bw(in_bits));
  if (clamp_max < clamp_min) clamp_max = clamp_min;
  if (use_strict_generic_hooks()) {
    const auto tkey = make_table_key(GateKind::Inv, bw, scale, in_bits, scale_in,
                                     clamp_min, clamp_max, static_cast<std::uint64_t>(nmax));
    const int op_id = get_or_register_builtin_generic_operator(tkey);
    return strict_keygen_builtin_u16(op_id, party, bw, scale, in_bits, d_input_mask, n);
  }
  if (env_enabled("SUF_SIGMA_GENERIC")) {
    const auto tkey = make_table_key(GateKind::Inv, bw, scale, in_bits, scale_in,
                                     clamp_min, clamp_max, static_cast<std::uint64_t>(nmax));
    const int op_id = get_or_register_builtin_generic_operator(tkey);
    return keygen_compiled_operator_u16(op_id, party, bw, scale, d_input_mask, n);
  }
  const auto tkey = make_table_key(GateKind::Inv, bw, scale, in_bits, scale_in,
                                   clamp_min, clamp_max, static_cast<std::uint64_t>(nmax));
  return keygen_builtin_generic_optimized_u16(tkey, "inv-dpf-lut",
                                              party, d_input_mask, n);
}

extern "C" std::uint64_t* suf_sigma_eval_inverse(SigmaPeer* peer,
                                                 int party,
                                                 int bw,
                                                 int scale,
                                                 int nmax,
                                                 const std::uint16_t* d_input_masked,
                                                 std::size_t n,
                                                 Stats* s) {
  const int scale_in = env_int("SUF_INV_FRAC", 6);
  int in_bits = env_int("SUF_INV_BITS", 0);
  const std::uint64_t max_fixed = (nmax > 0) ? (static_cast<std::uint64_t>(nmax) << scale_in) : 0;
  if (in_bits <= 0) {
    std::uint64_t tmp = max_fixed;
    int bits = 0;
    while (tmp > 0) {
      ++bits;
      tmp >>= 1;
    }
    in_bits = std::max(1, std::min(16, bits));
  }
  const std::uint64_t clamp_min = (1ULL << scale_in);
  std::uint64_t clamp_max = std::min<std::uint64_t>(max_fixed, mask_for_bw(in_bits));
  if (clamp_max < clamp_min) clamp_max = clamp_min;
  if (use_strict_generic_hooks()) {
    const auto tkey = make_table_key(GateKind::Inv, bw, scale, in_bits, scale_in,
                                     clamp_min, clamp_max, static_cast<std::uint64_t>(nmax));
    const int op_id = get_or_register_builtin_generic_operator(tkey);
    return strict_eval_builtin_u16(peer, op_id, party, bw, scale, in_bits,
                                   d_input_masked, n, s);
  }
  if (env_enabled("SUF_SIGMA_GENERIC")) {
    const auto tkey = make_table_key(GateKind::Inv, bw, scale, in_bits, scale_in,
                                     clamp_min, clamp_max, static_cast<std::uint64_t>(nmax));
    const int op_id = get_or_register_builtin_generic_operator(tkey);
    return eval_compiled_operator_u16(peer, op_id, party, bw, scale,
                                      d_input_masked, n, s, true);
  }
  const auto tkey = make_table_key(GateKind::Inv, bw, scale, in_bits, scale_in,
                                   clamp_min, clamp_max, static_cast<std::uint64_t>(nmax));
  return eval_builtin_generic_optimized_u16(tkey, "inv-dpf-lut",
                                            peer, party, d_input_masked, n, s);
}

extern "C" std::uint64_t* suf_sigma_keygen_rsqrt(int party,
                                                 int bw,
                                                 int scale,
                                                 int extradiv,
                                                 const std::uint16_t* d_input_mask,
                                                 std::size_t n) {
  const int target_frac = env_int("SUF_RSQRT_FRAC", 6);
  const int shift = std::max(0, 2 * scale - target_frac);
  const int max_bits = std::max(1, std::min(16, bw - shift));
  const int scale_in = 2 * scale - shift;
  const double vmax_real = env_double("SUF_RSQRT_VMAX", 16.0);
  const double eps_real = env_double("SUF_RSQRT_EPS", 0.0);
  const std::uint64_t clamp_min = std::max<std::uint64_t>(1, static_cast<std::uint64_t>(llroundl(eps_real * (1ULL << scale_in))));
  const std::uint64_t vmax_fixed = static_cast<std::uint64_t>(llroundl(vmax_real * (1ULL << scale_in)));
  int in_bits = env_int("SUF_RSQRT_BITS", 0);
  if (in_bits <= 0) {
    in_bits = bits_needed(vmax_fixed);
    in_bits = std::max(8, std::min(max_bits, in_bits));
  } else {
    in_bits = std::max(1, std::min(max_bits, in_bits));
  }
  std::uint64_t clamp_max = std::min<std::uint64_t>(vmax_fixed, mask_for_bw(in_bits));
  if (clamp_max < clamp_min) clamp_max = clamp_min;
  if (use_strict_generic_hooks()) {
    const auto tkey = make_table_key(GateKind::Rsqrt, bw, scale, in_bits, scale_in,
                                     clamp_min, clamp_max, static_cast<std::uint64_t>(extradiv));
    const int op_id = get_or_register_builtin_generic_operator(tkey);
    return strict_keygen_builtin_u16(op_id, party, bw, scale, in_bits, d_input_mask, n);
  }
  if (env_enabled("SUF_SIGMA_GENERIC")) {
    const auto tkey = make_table_key(GateKind::Rsqrt, bw, scale, in_bits, scale_in,
                                     clamp_min, clamp_max, static_cast<std::uint64_t>(extradiv));
    const int op_id = get_or_register_builtin_generic_operator(tkey);
    return keygen_compiled_operator_u16(op_id, party, bw, scale, d_input_mask, n);
  }
  const auto tkey = make_table_key(GateKind::Rsqrt, bw, scale, in_bits, scale_in,
                                   clamp_min, clamp_max, static_cast<std::uint64_t>(extradiv));
  return keygen_builtin_generic_optimized_u16(tkey, "rsqrt-dpf-lut",
                                              party, d_input_mask, n);
}

extern "C" std::uint64_t* suf_sigma_eval_rsqrt(SigmaPeer* peer,
                                               int party,
                                               int bw,
                                               int scale,
                                               int extradiv,
                                               const std::uint16_t* d_input_masked,
                                               std::size_t n,
                                               Stats* s) {
  const int target_frac = env_int("SUF_RSQRT_FRAC", 6);
  const int shift = std::max(0, 2 * scale - target_frac);
  const int max_bits = std::max(1, std::min(16, bw - shift));
  const int scale_in = 2 * scale - shift;
  const double vmax_real = env_double("SUF_RSQRT_VMAX", 16.0);
  const double eps_real = env_double("SUF_RSQRT_EPS", 0.0);
  const std::uint64_t clamp_min = std::max<std::uint64_t>(1, static_cast<std::uint64_t>(llroundl(eps_real * (1ULL << scale_in))));
  const std::uint64_t vmax_fixed = static_cast<std::uint64_t>(llroundl(vmax_real * (1ULL << scale_in)));
  int in_bits = env_int("SUF_RSQRT_BITS", 0);
  if (in_bits <= 0) {
    in_bits = bits_needed(vmax_fixed);
    in_bits = std::max(8, std::min(max_bits, in_bits));
  } else {
    in_bits = std::max(1, std::min(max_bits, in_bits));
  }
  std::uint64_t clamp_max = std::min<std::uint64_t>(vmax_fixed, mask_for_bw(in_bits));
  if (clamp_max < clamp_min) clamp_max = clamp_min;
  if (use_strict_generic_hooks()) {
    const auto tkey = make_table_key(GateKind::Rsqrt, bw, scale, in_bits, scale_in,
                                     clamp_min, clamp_max, static_cast<std::uint64_t>(extradiv));
    const int op_id = get_or_register_builtin_generic_operator(tkey);
    return strict_eval_builtin_u16(peer, op_id, party, bw, scale, in_bits,
                                   d_input_masked, n, s);
  }
  if (env_enabled("SUF_SIGMA_GENERIC")) {
    const auto tkey = make_table_key(GateKind::Rsqrt, bw, scale, in_bits, scale_in,
                                     clamp_min, clamp_max, static_cast<std::uint64_t>(extradiv));
    const int op_id = get_or_register_builtin_generic_operator(tkey);
    return eval_compiled_operator_u16(peer, op_id, party, bw, scale,
                                      d_input_masked, n, s, true);
  }
  const auto tkey = make_table_key(GateKind::Rsqrt, bw, scale, in_bits, scale_in,
                                   clamp_min, clamp_max, static_cast<std::uint64_t>(extradiv));
  return eval_builtin_generic_optimized_u16(tkey, "rsqrt-dpf-lut",
                                            peer, party, d_input_masked, n, s);
}

extern "C" std::uint64_t* suf_sigma_eval_activation(SigmaPeer* peer,
                                                    int party,
                                                    int bw,
                                                    int scale,
                                                    bool silu,
                                                    const std::uint64_t* d_input_masked,
                                                    std::size_t n,
                                                    Stats* s) {
  const int intervals = env_int(silu ? "SUF_SILU_INTERVALS" : "SUF_GELU_INTERVALS",
                                silu ? 1024 : 256);
  const int default_bits = bits_needed(static_cast<std::uint64_t>(intervals - 1));
  const int in_bits = env_int(silu ? "SUF_SILU_BITS" : "SUF_GELU_BITS", default_bits);
  const std::uint64_t clamp_max = mask_for_bw(in_bits);
  const GateKind kind = silu ? GateKind::Silu : GateKind::Gelu;
  if (use_strict_generic_hooks()) {
    const auto tkey = make_table_key(kind, bw, scale, in_bits, scale, 0, clamp_max, 0);
    const int op_id = get_or_register_builtin_generic_operator(tkey);
    return strict_eval_builtin_u64(peer, op_id, party, bw, scale, d_input_masked, n, s);
  }
  if (env_enabled("SUF_SIGMA_GENERIC")) {
    const auto tkey = make_table_key(kind, bw, scale, in_bits, scale, 0, clamp_max, 0);
    const int op_id = get_or_register_builtin_generic_operator(tkey);
    return eval_compiled_operator_u64(peer, op_id, party, bw, scale,
                                      d_input_masked, n, s, false);
  }
  const auto tkey = make_table_key(kind, bw, scale, in_bits, scale, 0, clamp_max, 0);
  return eval_builtin_generic_optimized_u64(tkey, silu ? "silu-dpf-lut" : "gelu-dpf-lut",
                                            peer, party, d_input_masked, n, s, false);
}
