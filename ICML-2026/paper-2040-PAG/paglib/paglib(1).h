#pragma once
#ifndef NO_MANUAL_VECTORIZATION
#ifdef __SSE__
#define USE_SSE
#ifdef __AVX__
#define USE_AVX
#ifdef __AVX512F__
#define USE_AVX512
#endif
#endif
#endif
#endif

#if defined(USE_AVX) || defined(USE_SSE)
#ifdef _MSC_VER
#include <intrin.h>
#include <stdexcept>
#else
#include <x86intrin.h>
#endif

#if defined(USE_AVX512)
#include <immintrin.h>
#endif

#if defined(__GNUC__)
#define PORTABLE_ALIGN32 __attribute__((aligned(32)))
#define PORTABLE_ALIGN64 __attribute__((aligned(64)))
#else
#define PORTABLE_ALIGN32 __declspec(align(32))
#define PORTABLE_ALIGN64 __declspec(align(64))
#endif
#endif

#include <iostream>
#include <queue>
#include <string.h>
#include <vector>
struct Neighbor {
  unsigned id;
  float distance;
  float inner_product;
  bool needs_expansion;
  const void *edge_record;

  Neighbor() = default;

  Neighbor(unsigned id, float distance, float inner_product,
           bool needs_expansion, const void *edge_record = nullptr)
      : id{id}, distance{distance}, inner_product{inner_product},
        needs_expansion{needs_expansion}, edge_record{edge_record} {}

  inline bool operator<(const Neighbor &other) const {
    return distance < other.distance;
  }
};

struct NeighborIndex {
  unsigned id;
  float distance;
  float inner_product;
  bool needs_expansion;

  NeighborIndex() = default;

  NeighborIndex(unsigned id, float distance, float inner_product,
                bool needs_expansion)
      : id{id}, distance{distance}, inner_product{inner_product},
        needs_expansion{needs_expansion} {}

  inline bool operator<(const NeighborIndex &other) const {
    return distance < other.distance;
  }
};

struct PIFEntry {
  float value;
  int index;
};

namespace paglib {
// typedef size_t labeltype;
typedef unsigned int labeltype;

enum class PAGMetric {
  L2,
  Cosine,
  MaximumInnerProduct,
};

enum class PAGIndexMode {
  Static,
  OnlineInsert,
};

enum class PAGStorageBackend {
  Memory,
  SSD,
};

enum class PAGComputeBackend {
  CPU,
  GPU,
};

struct PAGIndexOptions {
  PAGMetric metric = PAGMetric::L2;
  PAGIndexMode mode = PAGIndexMode::Static;
  PAGStorageBackend storage = PAGStorageBackend::Memory;
  PAGComputeBackend compute = PAGComputeBackend::CPU;
};

template <typename T> class pairGreater {
public:
  bool operator()(const T &p1, const T &p2) { return p1.first > p2.first; }
};

template <typename T>
static void writeBinaryPOD(std::ostream &out, const T &podRef) {
  out.write((char *)&podRef, sizeof(T));
}

template <typename T> static void readBinaryPOD(std::istream &in, T &podRef) {
  in.read((char *)&podRef, sizeof(T));
}

template <typename MTYPE>
using DISTFUNC = MTYPE (*)(const void *, const void *, const void *);

template <typename MTYPE> class SpaceInterface {
public:
  // virtual void search(void *);
  virtual size_t get_data_size() = 0;

  virtual DISTFUNC<MTYPE> get_dist_func() = 0;

  virtual void *get_dist_func_param() = 0;

  virtual ~SpaceInterface() {}
};

template <typename dist_t> struct PAGSpaceBundle {
  SpaceInterface<dist_t> *vector_space = nullptr;
  SpaceInterface<dist_t> *inner_product_space = nullptr;
  SpaceInterface<dist_t> *projection_space = nullptr;
  PAGIndexOptions options;
};

template <typename dist_t> class PAGAlgorithmInterface {
public:
  virtual void insertPoint(const void *datapoint, labeltype label,
                           float *norm) = 0;
  virtual void setConstructionEf(int ef_construction) = 0;
  virtual void packSearchLayout(int indexed_count) = 0;
  virtual void completePESForNode(int node_id) = 0;
  virtual void reservePESProjectionStorageForNode(int node_id) = 0;
  virtual void encodePESProjectionsForNode(int node_id) = 0;
  virtual void selectEntryPointSeeds(float *center_vector) = 0;
  virtual void initializeFallbackEntryPointSeeds() = 0;
  virtual void query(float *query_point, float *query_extended_point,
                     size_t top_k, std::vector<Neighbor> &result,
                     float *projection_table, int step) const = 0;

  virtual void save() = 0;
  virtual ~PAGAlgorithmInterface() {}
};

} // namespace paglib

#include "bruteforce.h"
#include "pag_index_core.h"
#include "space_ip.h"
#include "space_l2.h"
