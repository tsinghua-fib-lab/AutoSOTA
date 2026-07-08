#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace pag {

using Label = unsigned int;

enum class Metric {
  L2,
  Cosine,
  MaximumInnerProduct,
};

enum class IndexMode {
  Static,
  Online,
};

struct BuildOptions {
  std::string index_path;
  Metric metric = Metric::L2;
  IndexMode mode = IndexMode::Static;
  // Public API build-time cap for search().top_k. CLI benchmarks may infer the
  // L2/Cosine cap from ground-truth files; the API makes this explicit.
  // Internal working sets still use at least 10 entries.
  size_t max_search_k = 10;
  size_t max_elements = 0;
  int ef_construction = 200;
  // Target graph degree M. The base layer can store up to 2 * target_degree
  // neighbors after reverse-link updates.
  int target_degree = 16;
  // Must be a positive multiple of 8. The per-level projection code width is
  // fixed internally by the 4-bit encoding format.
  int projection_levels = 64;
};

struct LoadOptions {
  std::string index_path;
  Metric metric = Metric::L2;
  size_t max_elements = 0;
};

struct SearchOptions {
  size_t top_k = 10;
  size_t ef_search = 0;
};

struct SearchResult {
  Label id = 0;
  float distance = 0.0f;
  float inner_product = 0.0f;
};

class Index {
public:
  Index();
  ~Index();

  Index(Index &&) noexcept;
  Index &operator=(Index &&) noexcept;

  Index(const Index &) = delete;
  Index &operator=(const Index &) = delete;

  void build(const float *row_major_vectors, size_t count, size_t dimension,
             const BuildOptions &options);
  void load(const LoadOptions &options);
  void save();
  std::vector<SearchResult> search(const float *query,
                                   const SearchOptions &options) const;
  std::vector<std::vector<SearchResult>>
  search_batch(const float *row_major_queries, size_t query_count,
               const SearchOptions &options) const;

  Label add(const float *vector);
  void insert(const float *vector, Label label);
  std::vector<Label> add_batch(const float *row_major_vectors, size_t count);
  void insert_batch(const float *row_major_vectors, const Label *labels,
                    size_t count);

  bool is_loaded() const;
  Metric metric() const;
  size_t dimension() const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace pag
