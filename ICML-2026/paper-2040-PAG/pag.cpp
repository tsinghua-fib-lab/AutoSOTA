#include "pag_config.h"
#include "pag_index.h"
#include "paglib/paglib.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <omp.h>
#include <queue>
#include <random>
#include <time.h>
#include <unordered_set>

static constexpr size_t kAlignment = 64;
static constexpr int kProjectionWidth = 8;
static constexpr int kMaxConstructionBeamWidth = 100;

using namespace std;
using namespace paglib;
namespace fs = std::filesystem;

class MicrosecondTimer {
  std::chrono::steady_clock::time_point time_begin;

public:
  MicrosecondTimer() { time_begin = std::chrono::steady_clock::now(); }

  float elapsedMicros() {
    std::chrono::steady_clock::time_point time_end =
        std::chrono::steady_clock::now();
    return (std::chrono::duration_cast<std::chrono::microseconds>(time_end -
                                                                  time_begin)
                .count());
  }

  void reset() { time_begin = std::chrono::steady_clock::now(); }
};

static std::string IndexFilePath(const char *index_dir_path,
                                 const char *file_name) {
  std::string index_dir(index_dir_path);
  if (!index_dir.empty() &&
      (index_dir.back() == '/' || index_dir.back() == '\\')) {
    return index_dir + file_name;
  }
  return index_dir + "/" + file_name;
}

struct MatrixHeader {
  uint32_t rows;
  uint32_t cols;
};

struct MatrixLayout {
  bool has_header = true;
  size_t rows = 0;
  size_t cols = 0;
};

static MatrixHeader ReadMatrixHeader(std::ifstream &input) {
  MatrixHeader header{};
  input.read((char *)&header.rows, sizeof(header.rows));
  input.read((char *)&header.cols, sizeof(header.cols));
  return header;
}

static MatrixLayout DetectFloatMatrixLayout(const char *path,
                                            size_t expected_rows,
                                            size_t expected_cols) {
  const uintmax_t file_bytes = fs::file_size(path);
  const uintmax_t raw_bytes = expected_rows * expected_cols * sizeof(float);
  const uintmax_t headered_bytes = sizeof(MatrixHeader) + raw_bytes;

  if (file_bytes == headered_bytes) {
    std::ifstream input(path, std::ios::binary);
    MatrixHeader header = ReadMatrixHeader(input);
    if (header.rows == expected_rows && header.cols == expected_cols) {
      return MatrixLayout{true, expected_rows, expected_cols};
    }
  }
  if (file_bytes == raw_bytes) {
    return MatrixLayout{false, expected_rows, expected_cols};
  }

  throw std::runtime_error("Float matrix dimensions do not match file size");
}

static std::streampos VectorFileOffset(const MatrixLayout &layout,
                                       size_t vector_id, size_t dim) {
  const std::streamoff header_bytes =
      layout.has_header ? static_cast<std::streamoff>(sizeof(MatrixHeader)) : 0;
  return header_bytes +
         static_cast<std::streamoff>(vector_id * dim * sizeof(float));
}

static void SeekVector(std::ifstream &input, const MatrixLayout &layout,
                       size_t vector_id, size_t dim) {
  input.seekg(VectorFileOffset(layout, vector_id, dim), std::ios::beg);
}

static float NormalizeVectorInPlace(float *vector, size_t dim) {
  float squared_norm = 0.0f;
  for (size_t i = 0; i < dim; i++) {
    squared_norm += vector[i] * vector[i];
  }
  float norm = std::sqrt(squared_norm);
  if (norm > 0.0f) {
    const float inv_norm = 1.0f / norm;
    for (size_t i = 0; i < dim; i++) {
      vector[i] *= inv_norm;
    }
  }
  return norm;
}

static PAGMetric ParseMetricName(const char *metric_name) {
  if (metric_name != nullptr && (std::strcmp(metric_name, "cosine") == 0 ||
                                 std::strcmp(metric_name, "COSINE") == 0)) {
    return PAGMetric::Cosine;
  }
  if (metric_name != nullptr && (std::strcmp(metric_name, "mips") == 0 ||
                                 std::strcmp(metric_name, "MIPS") == 0 ||
                                 std::strcmp(metric_name, "ip") == 0 ||
                                 std::strcmp(metric_name, "IP") == 0)) {
    return PAGMetric::MaximumInnerProduct;
  }
  return PAGMetric::L2;
}

static int ComputeWorkingSetSize(int topk) {
  if (topk <= 10) {
    return 10;
  }
  return topk;
}

static size_t RequiredOnlineInitialCount(int construction_beam_width) {
  return static_cast<size_t>(std::max(1, construction_beam_width));
}

static void BuildRandomOrthogonalMatrix(int projection_subspace_dim,
                                        std::mt19937 &rng,
                                        std::vector<std::vector<float>> &R) {
  std::normal_distribution<float> nd(0.0f, 1.0f);
  R.assign(projection_subspace_dim,
           std::vector<float>(projection_subspace_dim, 0.0f));

  for (int i = 0; i < projection_subspace_dim; i++)
    for (int j = 0; j < projection_subspace_dim; j++)
      R[i][j] = nd(rng);

  for (int i = 0; i < projection_subspace_dim; i++) {
    for (int j = 0; j < i; j++) {
      float dot = 0.0f;
      for (int k = 0; k < projection_subspace_dim; k++)
        dot += R[i][k] * R[j][k];
      for (int k = 0; k < projection_subspace_dim; k++)
        R[i][k] -= dot * R[j][k];
    }
    float norm = 0.0f;
    for (int k = 0; k < projection_subspace_dim; k++)
      norm += R[i][k] * R[i][k];
    norm = std::sqrt(norm);
    for (int k = 0; k < projection_subspace_dim; k++)
      R[i][k] /= norm;
  }
}

static void BuildBalancedPermutation(std::vector<float> &dim_norm,
                                     int vector_dim, int level,
                                     std::vector<int> &permutation,
                                     std::vector<int> &zero_positions) {
  permutation.resize(vector_dim);

  zero_positions.resize(level);

  std::vector<int> idx(vector_dim);
  std::iota(idx.begin(), idx.end(), 0);
  std::sort(idx.begin(), idx.end(),
            [&dim_norm](int a, int b) { return dim_norm[a] > dim_norm[b]; });

  std::vector<float> seg_norm(level, 0.0f);
  std::vector<int> seg_size(level, 0);
  std::vector<std::vector<int>> segments(level);

  int K = vector_dim / level;
  for (int k = 0; k < vector_dim; k++) {
    int dim_id = idx[k];

    int best_seg = -1;
    float best_norm = std::numeric_limits<float>::max();

    for (int s = 0; s < level; s++) {
      if (seg_size[s] < K && seg_norm[s] < best_norm) {
        best_norm = seg_norm[s];
        best_seg = s;
      }
    }

    segments[best_seg].push_back(dim_id);
    seg_norm[best_seg] += dim_norm[dim_id];
    seg_size[best_seg]++;
  }

  for (int l = 0; l < level; l++) {
    int projection_subspace_dim = segments[l].size();
    int subdim0 = 0;

    for (int k = projection_subspace_dim - 1; k >= 0; k--) {
      int dim_id = segments[l][k];
      if (dim_norm[dim_id] > 0.0f) {
        subdim0 = k + 1;
        break;
      }
    }
    zero_positions[l] = subdim0;
  }

  int pos = 0;
  for (int l = 0; l < level; l++)
    for (int d : segments[l])
      permutation[pos++] = d;
}

static void BuildProjectionVectors(
    std::vector<std::vector<std::vector<float>>> &projection_vectors, int level,
    int projection_subspace_dim, int projection_width,
    std::vector<int> &zero_positions) {
  std::normal_distribution<float> nd(0.0f, 1.0f);

#pragma omp parallel for
  for (int l = 0; l < level; l++) {
    std::random_device rd;
    std::mt19937 rng_thread(rd() + l);

    int subdim0 = zero_positions[l];
    if (subdim0 <= 0)
      subdim0 = projection_subspace_dim;

    std::vector<std::vector<float>> vectors(projection_width,
                                            std::vector<float>(subdim0, 0.0f));
    if (projection_width <= subdim0) {
      for (int i = 0; i < projection_width; i++)
        vectors[i][i] = 1.0f;

      std::vector<std::vector<float>> R;
      BuildRandomOrthogonalMatrix(subdim0, rng_thread, R);
      for (int i = 0; i < projection_width; i++) {
        std::vector<float> tmp(subdim0, 0.0f);
        for (int r = 0; r < subdim0; r++)
          for (int c = 0; c < subdim0; c++)
            tmp[r] += R[r][c] * vectors[i][c];
        vectors[i] = tmp;
      }
    } else {
      int n_poly = projection_width / subdim0;
      int remainder = projection_width % subdim0;
      int idx = 0;

      for (int p = 0; p < n_poly; p++) {
        for (int i = 0; i < subdim0; i++) {
          std::fill(vectors[idx + i].begin(), vectors[idx + i].end(), 0.0f);
          vectors[idx + i][i] = 1.0f;
        }

        std::vector<std::vector<float>> R;
        BuildRandomOrthogonalMatrix(subdim0, rng_thread, R);
        for (int i = 0; i < subdim0; i++) {
          std::vector<float> tmp(subdim0, 0.0f);
          for (int r = 0; r < subdim0; r++)
            for (int c = 0; c < subdim0; c++)
              tmp[r] += R[r][c] * vectors[idx + i][c];
          vectors[idx + i] = tmp;
        }
        idx += subdim0;
      }

      if (remainder > 0) {
        for (int i = 0; i < remainder; i++) {
          std::fill(vectors[idx + i].begin(), vectors[idx + i].end(), 0.0f);
          vectors[idx + i][i] = 1.0f;
        }

        std::vector<std::vector<float>> R;
        BuildRandomOrthogonalMatrix(subdim0, rng_thread, R);
        for (int i = idx; i < idx + remainder; i++) {
          std::vector<float> tmp(subdim0, 0.0f);
          for (int r = 0; r < subdim0; r++)
            for (int c = 0; c < subdim0; c++)
              tmp[r] += R[r][c] * vectors[i][c];

          vectors[i] = tmp;
        }
      }
    }

    float scale = 1.0f / std::sqrt((float)level);
    for (int i = 0; i < projection_width; i++) {
      for (int j = 0; j < subdim0; j++)
        projection_vectors[l][i][j] = vectors[i][j] * scale;

      for (int j = subdim0; j < projection_subspace_dim; j++)
        projection_vectors[l][i][j] = 0;
    }
  }
}

static void BuildGroundTruthQueues(
    unsigned int *truth_ids, size_t query_count,
    vector<std::priority_queue<std::pair<float, labeltype>>> &answers,
    size_t topk, size_t truth_stride) {
  (vector<std::priority_queue<std::pair<float, labeltype>>>(query_count))
      .swap(answers);
  for (int query_id = 0; query_id < query_count; query_id++) {
    for (int rank = 0; rank < topk; rank++) {
      answers[query_id].emplace(0.0f,
                                truth_ids[truth_stride * query_id + rank]);
    }
  }
}

static void PrintBuildProgress(size_t completed, size_t total) {
  constexpr int kBarWidth = 40;
  if (total == 0) {
    return;
  }
  if (completed > total) {
    completed = total;
  }
  const int percent = static_cast<int>((100 * completed) / total);
  const int filled = static_cast<int>((kBarWidth * completed) / total);

  std::cerr << "\rBuilding PAG [";
  for (int i = 0; i < kBarWidth; ++i) {
    std::cerr << (i < filled ? '=' : ' ');
  }
  std::cerr << "] " << percent << "% (" << completed << "/" << total << ")"
            << std::flush;
  if (completed == total) {
    std::cerr << "\n";
  }
}

static std::vector<size_t> BuildSearchEfList(size_t topk) {
  const int search_step = ComputeWorkingSetSize((int)topk);
  const size_t first_ef = std::max<size_t>(topk, search_step);
  int ef_count = 10;
  if (const char *ef_count_env = std::getenv("PAG_EF_POINTS")) {
    ef_count = std::max(1, std::atoi(ef_count_env));
  }

  std::vector<size_t> search_efs;
  search_efs.reserve(ef_count);
  for (int i = 0; i < ef_count; ++i) {
    const size_t ef = first_ef + static_cast<size_t>(i) * search_step;
    search_efs.push_back(ef);
  }
  return search_efs;
}

static std::vector<int> LoadPermutation(const char *index_dir,
                                        size_t extended_dim) {
  std::vector<int> permutation(extended_dim);
  std::string permutation_path = IndexFilePath(index_dir, "permutation.bin");

  std::ifstream permutation_input(permutation_path, std::ios::binary);
  if (!permutation_input) {
    std::cerr << "Failed to open permutation file!" << std::endl;
  }
  permutation_input.read((char *)permutation.data(),
                         permutation.size() * sizeof(int));
  return permutation;
}

static void SavePermutation(const char *index_dir,
                            const std::vector<int> &permutation) {
  std::string permutation_path = IndexFilePath(index_dir, "permutation.bin");
  std::ofstream permutation_output(permutation_path, std::ios::binary);
  permutation_output.write((char *)permutation.data(),
                           permutation.size() * sizeof(int));
  permutation_output.close();
}

static void
RunSearchBenchmark(float *queries, size_t query_count,
                   PAGIndexCore<float> &index,
                   vector<std::priority_queue<std::pair<float, labeltype>>>
                       &ground_truth_answers,
                   size_t topk, float *query_table, size_t dim,
                   size_t extended_dim, const char *index_dir) {
  size_t padded_dim = (dim + 15) & ~0xF;
  std::vector<int> permutation = LoadPermutation(index_dir, extended_dim);

  float *permuted_queries =
      new (std::align_val_t{kAlignment}) float[query_count * extended_dim];

  if (topk > index.maxQueryTopK()) {
    throw std::runtime_error(
        "Requested topk exceeds this index's build-time max_search_k");
  }
  const int search_step = ComputeWorkingSetSize((int)topk);
  std::vector<size_t> search_efs = BuildSearchEfList(topk);

  std::vector<std::vector<Neighbor>> results;
  results.resize(query_count);
  for (size_t query_id = 0; query_id < query_count; ++query_id) {
    results[query_id].resize(topk);
  }

  cout << "efs\tRecall\tQPS\n";
  for (size_t ef : search_efs) {
    index.setSearchEf(ef);
    MicrosecondTimer stage_timer;

    for (int query_id = 0; query_id < query_count; query_id++) {
      float *query = queries + query_id * padded_dim;
      float *permuted_query = permuted_queries + query_id * extended_dim;

      for (int dim_id = 0; dim_id < extended_dim; dim_id++) {
        int source_dim = permutation[dim_id];
        if (source_dim < padded_dim)
          permuted_query[dim_id] = query[source_dim];
        else
          permuted_query[dim_id] = 0.0f;
      }
    }

    for (int query_id = 0; query_id < query_count; query_id++) {
      float *query = queries + query_id * padded_dim;
      float *permuted_query = permuted_queries + extended_dim * query_id;
      index.searchKnn(query, permuted_query, topk, results[query_id],
                      query_table, search_step);
    }
    float time_us_per_query = stage_timer.elapsedMicros() / query_count;

    size_t correct = 0;
    size_t total = 0;
    for (int query_id = 0; query_id < query_count; query_id++) {
      std::priority_queue<std::pair<float, labeltype>> gt(
          ground_truth_answers[query_id]);
      unordered_set<labeltype> truth_set;
      total += gt.size();

      while (gt.size()) {

        truth_set.insert(gt.top().second);
        gt.pop();
      }

      for (int rank = 0; rank < topk; rank++) {
        if (truth_set.find(results[query_id][rank].id) != truth_set.end()) {
          correct++;
        }
      }
    }

    float recall = 1.0f * correct / total;
    cout << ef << "\t" << recall << "\t" << 1e6 / time_us_per_query << "\n";
  }

  delete[] permuted_queries;
}

struct PAGRuntimeConfig {
  int ef_construction;
  int target_degree;
  int construction_beam_width;
  size_t base_count;
  size_t max_elements;
  size_t query_count;
  size_t dim;
  size_t padded_dim;
  size_t extended_dim;
  int projection_levels;
  int projection_width;
  int projection_subspace_dim;
  size_t topk;
  size_t max_query_top_k;
  size_t max_truth_k;
};

static PAGRuntimeConfig MakeRuntimeConfig(const PAGRunConfig &command) {
  if (command.projection_levels <= 0) {
    throw std::invalid_argument("PAG projection_levels must be positive");
  }
  if (command.projection_levels % 8 != 0) {
    throw std::invalid_argument(
        "PAG projection_levels must be a multiple of 8");
  }

  int ef_construction = command.ef_construction;
  int target_degree = command.target_degree;
  const int min_ef_construction = 2 * target_degree;

  if (ef_construction <= min_ef_construction) {
    ef_construction = min_ef_construction;
  }

  int construction_beam_width;
  if (ef_construction <= kMaxConstructionBeamWidth) {
    construction_beam_width = ef_construction;
  } else {
    ef_construction = ((ef_construction + kMaxConstructionBeamWidth - 1) /
                       kMaxConstructionBeamWidth) *
                      kMaxConstructionBeamWidth;
    construction_beam_width = kMaxConstructionBeamWidth;
  }

  PAGRuntimeConfig config{};
  config.ef_construction = ef_construction;
  config.target_degree = target_degree;
  config.construction_beam_width = construction_beam_width;
  config.base_count = command.base_count;
  config.max_elements = command.base_count;
  config.query_count = command.query_count;
  config.dim = command.dim;
  config.padded_dim = (config.dim + 15) & ~0xF;
  config.extended_dim = ((config.padded_dim + command.projection_levels - 1) /
                         command.projection_levels) *
                        command.projection_levels;
  config.projection_levels = command.projection_levels;
  config.projection_width = kProjectionWidth;
  config.projection_subspace_dim =
      config.extended_dim / config.projection_levels;
  config.topk = command.result_k;
  config.max_query_top_k = config.topk;
  config.max_truth_k = ComputeWorkingSetSize(static_cast<int>(config.topk));
  return config;
}

struct GroundTruthData {
  std::vector<unsigned int> ids;
  size_t query_count = 0;
  size_t max_truth_k = 0;
};

struct GroundTruthShape {
  size_t query_count = 0;
  size_t stride = 0;
  size_t payload_ids = 0;
  bool has_header = true;
};

static GroundTruthShape InferGroundTruthShape(const PAGRunConfig &command,
                                              size_t expected_query_count) {
  const uintmax_t truth_file_bytes = fs::file_size(command.truth_file);
  if (truth_file_bytes % sizeof(unsigned int) != 0) {
    throw std::runtime_error("Ground truth file size is not int-aligned");
  }

  ifstream truth_input(command.truth_file, ios::binary);
  MatrixHeader header = ReadMatrixHeader(truth_input);
  truth_input.close();

  GroundTruthShape shape;
  const bool header_plausible =
      truth_file_bytes >= sizeof(MatrixHeader) &&
      header.rows == expected_query_count && header.cols > 0 &&
      truth_file_bytes ==
          sizeof(MatrixHeader) + static_cast<uintmax_t>(header.rows) *
                                     header.cols * sizeof(unsigned int);

  if (header_plausible) {
    shape.query_count = header.rows;
    shape.stride = header.cols;
    shape.payload_ids = shape.query_count * shape.stride;
    shape.has_header = true;
    return shape;
  }

  if ((truth_file_bytes / sizeof(unsigned int)) % expected_query_count != 0) {
    const size_t header_payload_ids =
        (truth_file_bytes - sizeof(MatrixHeader)) / sizeof(unsigned int);
    if (truth_file_bytes >= sizeof(MatrixHeader) &&
        header_payload_ids % expected_query_count == 0) {
      shape.query_count = expected_query_count;
      shape.stride = header_payload_ids / expected_query_count;
      shape.payload_ids = header_payload_ids;
      shape.has_header = true;
      std::cerr << "Warning: Ground truth header says " << header.rows << "x"
                << header.cols << ", inferred " << shape.query_count << "x"
                << shape.stride << " from file size and query count.\n";
      return shape;
    }
    throw std::runtime_error(
        "Ground truth dimensions do not match query count");
  }

  shape.query_count = expected_query_count;
  shape.payload_ids = truth_file_bytes / sizeof(unsigned int);
  shape.stride = shape.payload_ids / expected_query_count;
  shape.has_header = false;
  return shape;
}

static GroundTruthData LoadGroundTruthIds(const PAGRunConfig &command,
                                          const PAGRuntimeConfig &runtime) {
  GroundTruthShape shape = InferGroundTruthShape(command, runtime.query_count);

  if (runtime.topk > shape.stride) {
    throw std::runtime_error("Requested topk is larger than truth width");
  }

  ifstream truth_input(command.truth_file, ios::binary);
  if (shape.has_header) {
    ReadMatrixHeader(truth_input);
  }

  GroundTruthData truth;
  truth.query_count = shape.query_count;
  truth.max_truth_k = shape.stride;
  truth.ids.resize(shape.payload_ids);
  truth_input.read((char *)truth.ids.data(),
                   truth.ids.size() * sizeof(unsigned int));
  truth_input.close();
  return truth;
}

static float *LoadQueryVectors(const PAGRunConfig &command,
                               const PAGRuntimeConfig &runtime,
                               PAGMetric metric) {

  float *query_vectors = (float *)std::aligned_alloc(
      kAlignment, runtime.query_count * runtime.padded_dim * sizeof(float));
  std::vector<float> input_buffer(runtime.dim);
  ifstream query_input(command.query_file, ios::binary);
  MatrixLayout query_layout = DetectFloatMatrixLayout(
      command.query_file, runtime.query_count, runtime.dim);

  if (query_layout.has_header) {
    ReadMatrixHeader(query_input);
  }
  for (int i = 0; i < runtime.query_count; i++) {
    query_input.read((char *)input_buffer.data(), sizeof(float) * runtime.dim);
    float *dst = &query_vectors[i * runtime.padded_dim];

    for (int j = 0; j < runtime.dim; j++)
      dst[j] = input_buffer[j];

    for (int j = runtime.dim; j < runtime.padded_dim; j++)
      dst[j] = 0.0f;

    if (metric == PAGMetric::Cosine ||
        metric == PAGMetric::MaximumInnerProduct) {
      NormalizeVectorInPlace(dst, runtime.padded_dim);
    }
  }
  query_input.close();
  return query_vectors;
}

struct SavedIndexMetadata {
  size_t projection_width = 0;
  int projection_level_count = 0;
  int projection_subspace_dim = 0;
  size_t max_elements = 0;
  size_t vector_dim = 0;
  size_t padded_dim = 0;
  size_t extended_dim = 0;
  size_t max_entry_points = 0;
};

static PAGMetric ToInternalMetric(pag::Metric metric) {
  switch (metric) {
  case pag::Metric::L2:
    return PAGMetric::L2;
  case pag::Metric::Cosine:
    return PAGMetric::Cosine;
  case pag::Metric::MaximumInnerProduct:
    return PAGMetric::MaximumInnerProduct;
  }
  throw std::invalid_argument("Unknown PAG metric");
}

static pag::Metric ToPublicMetric(PAGMetric metric) {
  switch (metric) {
  case PAGMetric::L2:
    return pag::Metric::L2;
  case PAGMetric::Cosine:
    return pag::Metric::Cosine;
  case PAGMetric::MaximumInnerProduct:
    return pag::Metric::MaximumInnerProduct;
  }
  throw std::invalid_argument("Unknown PAG metric");
}

static SavedIndexMetadata
ReadSavedIndexMetadata(const std::string &index_path) {
  SavedIndexMetadata metadata;
  std::ifstream input(IndexFilePath(index_path.c_str(), "info.bin"),
                      std::ios::binary);
  if (!input) {
    throw std::runtime_error("Cannot open PAG info.bin");
  }

  readBinaryPOD(input, metadata.projection_width);
  readBinaryPOD(input, metadata.projection_level_count);
  readBinaryPOD(input, metadata.projection_subspace_dim);
  readBinaryPOD(input, metadata.max_elements);

  size_t ignored_size = 0;
  int ignored_int = 0;
  tableint ignored_tableint = 0;
  double ignored_double = 0.0;

  readBinaryPOD(input, ignored_size);
  readBinaryPOD(input, ignored_size);
  readBinaryPOD(input, ignored_size);
  readBinaryPOD(input, ignored_size);
  readBinaryPOD(input, ignored_size);
  readBinaryPOD(input, ignored_int);
  readBinaryPOD(input, ignored_tableint);
  readBinaryPOD(input, ignored_size);
  readBinaryPOD(input, ignored_size);
  readBinaryPOD(input, ignored_size);
  readBinaryPOD(input, ignored_double);
  readBinaryPOD(input, ignored_size);
  readBinaryPOD(input, ignored_size);
  readBinaryPOD(input, ignored_size);
  readBinaryPOD(input, ignored_size);
  readBinaryPOD(input, metadata.vector_dim);
  readBinaryPOD(input, metadata.padded_dim);
  readBinaryPOD(input, metadata.extended_dim);
  readBinaryPOD(input, metadata.max_entry_points);
  return metadata;
}

namespace pag {

struct Index::Impl {
  PAGMetric metric = PAGMetric::L2;
  PAGIndexOptions index_options;
  std::string index_path;
  size_t vector_dim = 0;
  size_t padded_dim = 0;
  size_t extended_dim = 0;
  int projection_level_count = 0;
  int projection_subspace_dim = 0;
  size_t projection_width = 0;
  size_t max_entry_points = 0;
  size_t max_query_top_k = 0;
  bool mutable_graph = false;
  size_t element_count = 0;
  size_t max_elements = 0;
  Label next_label = 0;
  std::vector<int> permutation;

  std::unique_ptr<L2Space> vector_space;
  std::unique_ptr<InnerProductSpace> inner_product_space;
  std::unique_ptr<InnerProductSpace> projection_space;
  std::unique_ptr<PAGIndexCore<float>> core;

  void resetSpaces(size_t padded_dim_value, int projection_subspace_dim_value,
                   PAGMetric metric_value) {
    metric = metric_value;
    index_options.metric = metric_value;
    vector_space = std::make_unique<L2Space>(padded_dim_value);
    inner_product_space = std::make_unique<InnerProductSpace>(padded_dim_value);
    projection_space =
        std::make_unique<InnerProductSpace>(projection_subspace_dim_value);
  }

  PAGSpaceBundle<float> spaces() const {
    PAGSpaceBundle<float> bundle;
    bundle.vector_space = vector_space.get();
    bundle.inner_product_space = inner_product_space.get();
    bundle.projection_space = projection_space.get();
    bundle.options = index_options;
    return bundle;
  }

  void validateSearch(const SearchOptions &options, int &search_step,
                      size_t &ef_search) const {
    if (!core) {
      throw std::logic_error("Cannot search an empty PAG index");
    }
    if (options.top_k == 0) {
      throw std::invalid_argument("PAG search top_k must be positive");
    }
    const size_t effective_max_query_top_k =
        this->max_query_top_k == 0 ? max_entry_points : this->max_query_top_k;
    if (options.top_k > effective_max_query_top_k) {
      throw std::invalid_argument(
          "Requested top_k exceeds this index's build-time max_search_k");
    }
    if (options.top_k > element_count) {
      throw std::invalid_argument(
          "Requested top_k exceeds the current number of indexed vectors");
    }
    search_step = ComputeWorkingSetSize(static_cast<int>(options.top_k));
    if (static_cast<size_t>(search_step) > max_entry_points) {
      throw std::invalid_argument(
          "Requested top_k exceeds this index's configured max_search_k");
    }
    ef_search =
        options.ef_search == 0
            ? std::max<size_t>(options.top_k, static_cast<size_t>(search_step))
            : options.ef_search;
  }

  std::vector<SearchResult>
  searchOnePrepared(const float *query, const SearchOptions &options,
                    int search_step) const {
    if (query == nullptr) {
      throw std::invalid_argument("PAG search query must not be null");
    }

    std::vector<float> padded_query(padded_dim, 0.0f);
    std::copy(query, query + vector_dim, padded_query.begin());
    float half_query_squared_norm = 0.0f;
    if (metric == PAGMetric::L2) {
      for (size_t d = 0; d < vector_dim; ++d) {
        half_query_squared_norm += padded_query[d] * padded_query[d];
      }
      half_query_squared_norm *= 0.5f;
    }
    if (metric == PAGMetric::Cosine ||
        metric == PAGMetric::MaximumInnerProduct) {
      NormalizeVectorInPlace(padded_query.data(), padded_query.size());
    }

    std::vector<float> permuted_query(extended_dim, 0.0f);
    for (size_t dim_id = 0; dim_id < extended_dim; ++dim_id) {
      const int source_dim = permutation[dim_id];
      permuted_query[dim_id] =
          source_dim < static_cast<int>(padded_dim) ? padded_query[source_dim]
                                                    : 0.0f;
    }

    const size_t query_table_size =
        projection_level_count * 2 * projection_width;
    const size_t query_table_bytes =
        ((query_table_size * sizeof(float) + kAlignment - 1) / kAlignment) *
        kAlignment;
    float *query_table =
        static_cast<float *>(std::aligned_alloc(kAlignment, query_table_bytes));
    if (query_table == nullptr) {
      throw std::runtime_error("PAG search failed to allocate query table");
    }
    std::unique_ptr<float, decltype(&std::free)> query_table_guard(query_table,
                                                                   std::free);
    std::fill_n(query_table, query_table_size, 0.0f);

    std::vector<Neighbor> neighbors(options.top_k);
    if (mutable_graph) {
      core->onlineSearchKnn(padded_query.data(), permuted_query.data(),
                            options.top_k, neighbors, query_table,
                            search_step);
    } else {
      core->searchKnn(padded_query.data(), permuted_query.data(),
                      options.top_k, neighbors, query_table, search_step);
    }

    std::vector<SearchResult> results;
    results.reserve(neighbors.size());
    for (const Neighbor &neighbor : neighbors) {
      float public_distance = neighbor.distance;
      if (metric == PAGMetric::L2) {
        public_distance = 2.0f * (neighbor.distance + half_query_squared_norm);
        if (public_distance < 0.0f && public_distance > -1.0e-4f) {
          public_distance = 0.0f;
        }
      }
      results.push_back(
          SearchResult{neighbor.id, public_distance, neighbor.inner_product});
    }
    return results;
  }

  void prepareInsertPoint(const float *vector, std::vector<float> &point,
                          float &vector_norm) const {
    point.assign(extended_dim, 0.0f);
    std::copy(vector, vector + vector_dim, point.begin());

    vector_norm = 0.0f;
    if (metric == PAGMetric::Cosine) {
      NormalizeVectorInPlace(point.data(), vector_dim);
      vector_norm = 1.0f;
    } else {
      for (size_t d = 0; d < vector_dim; ++d) {
        vector_norm += point[d] * point[d];
      }
      vector_norm = std::sqrt(vector_norm);
    }
  }
};

Index::Index() : impl_(std::make_unique<Impl>()) {}
Index::~Index() = default;
Index::Index(Index &&) noexcept = default;
Index &Index::operator=(Index &&) noexcept = default;

void Index::build(const float *row_major_vectors, size_t count,
                  size_t dimension, const BuildOptions &options) {
  if (row_major_vectors == nullptr) {
    throw std::invalid_argument("PAG build input vectors must not be null");
  }
  if (count == 0 || dimension == 0) {
    throw std::invalid_argument("PAG build requires non-empty input");
  }
  if (options.index_path.empty()) {
    throw std::invalid_argument("PAG build requires an index_path");
  }
  if (options.projection_levels <= 0) {
    throw std::invalid_argument("PAG projection_levels must be positive");
  }
  if (options.projection_levels % 8 != 0) {
    throw std::invalid_argument(
        "PAG projection_levels must be a multiple of 8");
  }
  if (options.target_degree <= 0 || options.ef_construction <= 0) {
    throw std::invalid_argument("PAG graph parameters must be positive");
  }

  PAGRuntimeConfig runtime{};
  runtime.ef_construction = options.ef_construction;
  runtime.target_degree = options.target_degree;
  const int min_ef_construction = 2 * runtime.target_degree;
  if (runtime.ef_construction <= min_ef_construction) {
    runtime.ef_construction = min_ef_construction;
  }
  if (runtime.ef_construction <= kMaxConstructionBeamWidth) {
    runtime.construction_beam_width = runtime.ef_construction;
  } else {
    runtime.ef_construction =
        ((runtime.ef_construction + kMaxConstructionBeamWidth - 1) /
         kMaxConstructionBeamWidth) *
        kMaxConstructionBeamWidth;
    runtime.construction_beam_width = kMaxConstructionBeamWidth;
  }
  const int online_warmup_step = runtime.construction_beam_width;
  runtime.construction_beam_width =
      std::min<int>(runtime.construction_beam_width, static_cast<int>(count));
  runtime.base_count = count;
  runtime.max_elements = options.max_elements == 0 ? count : options.max_elements;
  if (runtime.max_elements < runtime.base_count) {
    throw std::invalid_argument("PAG max_elements must be at least count");
  }
  runtime.dim = dimension;
  runtime.padded_dim = (runtime.dim + 15) & ~0xF;
  runtime.projection_levels = options.projection_levels;
  runtime.projection_width = kProjectionWidth;
  runtime.extended_dim = ((runtime.padded_dim + runtime.projection_levels - 1) /
                          runtime.projection_levels) *
                         runtime.projection_levels;
  runtime.projection_subspace_dim =
      runtime.extended_dim / runtime.projection_levels;
  runtime.topk = std::max<size_t>(1, options.max_search_k);
  runtime.max_truth_k = ComputeWorkingSetSize(static_cast<int>(runtime.topk));

  const PAGMetric metric = ToInternalMetric(options.metric);
  if (options.mode == IndexMode::Online) {
    const size_t required_initial_count =
        RequiredOnlineInitialCount(online_warmup_step);
    if (runtime.base_count < required_initial_count) {
      throw std::invalid_argument(
          "PAG online build requires at least construction_beam_width "
          "warm-up vectors");
    }
  }
  if (metric == PAGMetric::MaximumInnerProduct &&
      options.mode != IndexMode::Online &&
      runtime.base_count < runtime.max_truth_k) {
    throw std::invalid_argument(
        "PAG MIPS build requires at least ComputeWorkingSetSize(max_search_k) "
        "vectors for projection metadata");
  }
  impl_->index_path = options.index_path;
  impl_->vector_dim = runtime.dim;
  impl_->padded_dim = runtime.padded_dim;
  impl_->extended_dim = runtime.extended_dim;
  impl_->projection_level_count = runtime.projection_levels;
  impl_->projection_subspace_dim = runtime.projection_subspace_dim;
  impl_->projection_width = runtime.projection_width;
  impl_->max_entry_points = runtime.max_truth_k;
  impl_->max_query_top_k = runtime.topk;
  impl_->resetSpaces(runtime.padded_dim, runtime.projection_subspace_dim,
                     metric);
  impl_->index_options.mode = options.mode == IndexMode::Online
                                  ? PAGIndexMode::OnlineInsert
                                  : PAGIndexMode::Static;

  fs::create_directories(options.index_path);

  std::vector<std::vector<std::vector<float>>> projection_vectors(
      runtime.projection_levels,
      std::vector<std::vector<float>>(
          runtime.projection_width,
          std::vector<float>(runtime.projection_subspace_dim, 0.0f)));
  std::vector<float> half_squared_norms(runtime.base_count, 0.0f);
  std::vector<float> vector_norms(runtime.base_count, 0.0f);
  std::vector<float> center_vector(runtime.padded_dim, 0.0f);
  std::vector<double> center_accumulator(runtime.padded_dim, 0.0f);
  std::vector<float> dim_energy(runtime.extended_dim, 0.0f);
  std::vector<double> dim_energy_accumulator(runtime.extended_dim, 0.0f);
  std::vector<float> padded_vector(runtime.extended_dim, 0.0f);

  for (size_t i = 0; i < runtime.base_count; ++i) {
    std::fill(padded_vector.begin(), padded_vector.end(), 0.0f);
    std::memcpy(padded_vector.data(), row_major_vectors + i * runtime.dim,
                runtime.dim * sizeof(float));

    float squared_norm = 0.0f;
    for (size_t j = 0; j < runtime.dim; ++j) {
      squared_norm += padded_vector[j] * padded_vector[j];
    }

    if (metric == PAGMetric::Cosine) {
      vector_norms[i] =
          NormalizeVectorInPlace(padded_vector.data(), runtime.dim);
      half_squared_norms[i] = 0.5f;
      vector_norms[i] = 1.0f;
    } else if (metric == PAGMetric::MaximumInnerProduct) {
      half_squared_norms[i] = squared_norm / 2.0f;
      vector_norms[i] = std::sqrt(squared_norm);
    } else {
      half_squared_norms[i] = squared_norm / 2.0f;
      vector_norms[i] = std::sqrt(squared_norm);
    }

    for (size_t d = 0; d < runtime.padded_dim; ++d) {
      center_accumulator[d] += padded_vector[d];
    }
    for (size_t d = 0; d < runtime.dim; ++d) {
      dim_energy_accumulator[d] += padded_vector[d] * padded_vector[d];
    }
  }

  for (size_t d = 0; d < runtime.dim; ++d) {
    dim_energy[d] = dim_energy_accumulator[d] / runtime.base_count;
  }
  for (size_t d = 0; d < runtime.padded_dim; ++d) {
    center_vector[d] = center_accumulator[d] / runtime.base_count;
  }
  if (metric == PAGMetric::Cosine) {
    NormalizeVectorInPlace(center_vector.data(), runtime.padded_dim);
  }

  std::vector<std::pair<float, int>> distance_to_center(runtime.base_count);
  for (size_t i = 0; i < runtime.base_count; ++i) {
    const float *src = row_major_vectors + i * runtime.dim;
    float score = 0.0f;
    if (metric == PAGMetric::MaximumInnerProduct) {
      for (size_t j = 0; j < runtime.dim; ++j) {
        score += src[j] * src[j];
      }
      distance_to_center[i] = {-score, static_cast<int>(i)};
    } else {
      std::vector<float> point(src, src + runtime.dim);
      if (metric == PAGMetric::Cosine) {
        NormalizeVectorInPlace(point.data(), point.size());
      }
      for (size_t j = 0; j < runtime.dim; ++j) {
        const float delta = point[j] - center_vector[j];
        score += delta * delta;
      }
      distance_to_center[i] = {score, static_cast<int>(i)};
    }
  }

  if (runtime.construction_beam_width <
      static_cast<int>(distance_to_center.size())) {
    std::nth_element(distance_to_center.begin(),
                     distance_to_center.begin() +
                         runtime.construction_beam_width,
                     distance_to_center.end());
  }
  std::sort(distance_to_center.begin(),
            distance_to_center.begin() + runtime.construction_beam_width);

  std::vector<std::pair<float, int>> initial_seed_ids(
      distance_to_center.begin(),
      distance_to_center.begin() + runtime.construction_beam_width);
  if (metric != PAGMetric::MaximumInnerProduct) {
    std::sort(initial_seed_ids.begin(), initial_seed_ids.end(),
              [](const auto &a, const auto &b) { return a.second < b.second; });
  }

  std::vector<bool> is_initial_seed(runtime.base_count, false);
  std::vector<std::vector<float>> initial_vectors(
      runtime.construction_beam_width, std::vector<float>(runtime.dim));
  for (int i = 0; i < runtime.construction_beam_width; ++i) {
    const int target_id = initial_seed_ids[i].second;
    is_initial_seed[target_id] = true;
    const float *src =
        row_major_vectors + static_cast<size_t>(target_id) * runtime.dim;
    std::copy(src, src + runtime.dim, initial_vectors[i].begin());
    if (metric == PAGMetric::Cosine) {
      NormalizeVectorInPlace(initial_vectors[i].data(), runtime.dim);
    }
  }

  std::vector<int> permutation;
  std::vector<int> zero_positions;
  BuildBalancedPermutation(dim_energy, runtime.extended_dim,
                           runtime.projection_levels, permutation,
                           zero_positions);
  BuildProjectionVectors(projection_vectors, runtime.projection_levels,
                         runtime.projection_subspace_dim,
                         runtime.projection_width, zero_positions);

  std::vector<int> mips_entry_permutation;
  std::vector<std::vector<std::vector<float>>> mips_entry_directions;
  if (metric == PAGMetric::MaximumInnerProduct) {
    std::vector<int> mips_entry_zero_positions;
    const int mips_entry_levels = 4;
    const int mips_entry_width = 8;
    const int mips_entry_subdim = runtime.extended_dim / mips_entry_levels;
    mips_entry_directions.assign(
        mips_entry_levels,
        std::vector<std::vector<float>>(
            mips_entry_width, std::vector<float>(mips_entry_subdim, 0.0f)));
    BuildBalancedPermutation(dim_energy, runtime.extended_dim,
                             mips_entry_levels, mips_entry_permutation,
                             mips_entry_zero_positions);
    BuildProjectionVectors(mips_entry_directions, mips_entry_levels,
                           mips_entry_subdim, mips_entry_width,
                           mips_entry_zero_positions);
  }

  std::vector<std::pair<float, int>>().swap(distance_to_center);

  impl_->core = std::make_unique<PAGIndexCore<float>>(
      runtime.max_truth_k, runtime.construction_beam_width,
      runtime.projection_width, runtime.dim, projection_vectors,
      runtime.projection_levels, runtime.projection_subspace_dim,
      half_squared_norms, permutation, impl_->vector_space.get(),
      impl_->inner_product_space.get(), impl_->projection_space.get(),
      runtime.max_elements, options.index_path.c_str(), runtime.target_degree,
      runtime.ef_construction, impl_->index_options);
  impl_->core->setMaxQueryTopK(runtime.topk);

  if (metric == PAGMetric::MaximumInnerProduct) {
    const int pif_entries_per_bucket = ComputeWorkingSetSize(runtime.topk);
    std::vector<int> initial_internal_ids(runtime.construction_beam_width);
    std::iota(initial_internal_ids.begin(), initial_internal_ids.end(), 0);
    impl_->core->configurePIF(mips_entry_permutation, mips_entry_directions,
                              pif_entries_per_bucket);
    impl_->core->buildInitialPIFTable(initial_vectors, initial_internal_ids);
  }

  for (int i = 0; i < runtime.construction_beam_width; ++i) {
    const int node_id = initial_seed_ids[i].second;
    std::vector<float> point(runtime.extended_dim, 0.0f);
    std::copy(initial_vectors[i].begin(), initial_vectors[i].end(),
              point.begin());
    impl_->core->insertPoint(point.data(), static_cast<labeltype>(node_id),
                             &vector_norms[node_id]);
  }

#pragma omp parallel for schedule(dynamic)
  for (size_t insertion_ord = 0; insertion_ord < runtime.base_count;
       ++insertion_ord) {
    const int node_id = static_cast<int>(insertion_ord);
    if (is_initial_seed[node_id]) {
      continue;
    }
    std::vector<float> point(runtime.extended_dim, 0.0f);
    const float *src =
        row_major_vectors + static_cast<size_t>(node_id) * runtime.dim;
    std::copy(src, src + runtime.dim, point.begin());
    if (metric == PAGMetric::Cosine) {
      NormalizeVectorInPlace(point.data(), runtime.dim);
    }
    impl_->core->insertPoint(point.data(), static_cast<labeltype>(node_id),
                             &vector_norms[node_id]);
  }

#pragma omp parallel for schedule(dynamic)
  for (size_t i = 0; i < runtime.base_count; ++i) {
    impl_->core->completePESForNode(static_cast<int>(i));
  }
#pragma omp parallel for schedule(dynamic)
  for (size_t i = 0; i < runtime.base_count; ++i) {
    impl_->core->reservePESProjectionStorageForNode(static_cast<int>(i));
  }
#pragma omp parallel for schedule(dynamic)
  for (size_t i = 0; i < runtime.base_count; ++i) {
    impl_->core->encodePESProjectionsForNode(static_cast<int>(i));
  }

  if (metric == PAGMetric::MaximumInnerProduct) {
    impl_->core->initializeFallbackEntryPointSeeds();
  } else {
    impl_->core->selectEntryPointSeeds(center_vector.data());
  }
  impl_->permutation = permutation;
  impl_->mutable_graph = options.mode == IndexMode::Online;
  impl_->element_count = runtime.base_count;
  impl_->max_elements = runtime.max_elements;
  impl_->next_label = static_cast<Label>(runtime.base_count);

  if (impl_->mutable_graph) {
    impl_->core->enableOnlineEntryPointTracking();
    return;
  }

  impl_->core->packSearchLayout(runtime.base_count);
  if (!impl_->index_path.empty() && !impl_->permutation.empty()) {
    fs::create_directories(impl_->index_path);
    SavePermutation(impl_->index_path.c_str(), impl_->permutation);
  }
  impl_->core->save();
  load(LoadOptions{options.index_path, options.metric, runtime.base_count});
}

void Index::load(const LoadOptions &options) {
  if (options.index_path.empty()) {
    throw std::invalid_argument("PAG load requires an index_path");
  }
  const SavedIndexMetadata metadata =
      ReadSavedIndexMetadata(options.index_path);
  const PAGMetric metric = ToInternalMetric(options.metric);

  impl_->index_path = options.index_path;
  impl_->vector_dim = metadata.vector_dim;
  impl_->padded_dim = metadata.padded_dim;
  impl_->extended_dim = metadata.extended_dim;
  impl_->projection_level_count = metadata.projection_level_count;
  impl_->projection_subspace_dim = metadata.projection_subspace_dim;
  impl_->projection_width = metadata.projection_width;
  impl_->max_entry_points = metadata.max_entry_points;
  impl_->resetSpaces(metadata.padded_dim, metadata.projection_subspace_dim,
                     metric);
  impl_->permutation =
      LoadPermutation(options.index_path.c_str(), metadata.extended_dim);

  const size_t max_elements =
      options.max_elements == 0 ? metadata.max_elements : options.max_elements;
  impl_->core = std::make_unique<PAGIndexCore<float>>(
      impl_->spaces(), options.index_path.c_str(), max_elements);
  impl_->max_query_top_k = impl_->core->maxQueryTopK();
  impl_->mutable_graph = false;
  impl_->element_count = impl_->core->currentCount();
  impl_->max_elements = max_elements;
  impl_->next_label = static_cast<Label>(impl_->element_count);
}

void Index::save() {
  if (!impl_->core) {
    throw std::logic_error("Cannot save an empty PAG index");
  }
  const bool finalize_mutable_graph = impl_->mutable_graph;
  if (!finalize_mutable_graph) {
    return;
  }
  if (finalize_mutable_graph) {
    impl_->core->packSearchLayout(static_cast<int>(impl_->element_count));
    impl_->mutable_graph = false;
  }
  if (!impl_->index_path.empty() && !impl_->permutation.empty()) {
    fs::create_directories(impl_->index_path);
    SavePermutation(impl_->index_path.c_str(), impl_->permutation);
  }
  impl_->core->save();
  if (finalize_mutable_graph) {
    load(LoadOptions{impl_->index_path, ToPublicMetric(impl_->metric),
                     impl_->element_count});
  }
}

std::vector<SearchResult> Index::search(const float *query,
                                        const SearchOptions &options) const {
  int search_step = 0;
  size_t ef_search = 0;
  impl_->validateSearch(options, search_step, ef_search);
  impl_->core->setSearchEf(ef_search);
  return impl_->searchOnePrepared(query, options, search_step);
}

std::vector<std::vector<SearchResult>>
Index::search_batch(const float *row_major_queries, size_t query_count,
                    const SearchOptions &options) const {
  if (query_count == 0) {
    return {};
  }
  if (row_major_queries == nullptr) {
    throw std::invalid_argument("PAG batch search queries must not be null");
  }

  int search_step = 0;
  size_t ef_search = 0;
  impl_->validateSearch(options, search_step, ef_search);
  impl_->core->setSearchEf(ef_search);

  std::vector<std::vector<SearchResult>> results(query_count);
  std::exception_ptr first_exception;
  std::mutex exception_mutex;
  std::atomic<bool> failed(false);

#pragma omp parallel for schedule(static)
  for (size_t query_id = 0; query_id < query_count; ++query_id) {
    if (failed.load(std::memory_order_relaxed)) {
      continue;
    }
    try {
      const float *query = row_major_queries + query_id * impl_->vector_dim;
      results[query_id] = impl_->searchOnePrepared(query, options, search_step);
    } catch (...) {
      failed.store(true, std::memory_order_relaxed);
      std::lock_guard<std::mutex> lock(exception_mutex);
      if (!first_exception) {
        first_exception = std::current_exception();
      }
    }
  }

  if (first_exception) {
    std::rethrow_exception(first_exception);
  }
  return results;
}

Label Index::add(const float *vector) {
  const Label label = impl_->next_label;
  insert(vector, label);
  return label;
}

void Index::insert(const float *vector, Label label) {
  if (!impl_->core) {
    throw std::logic_error("Cannot insert before building a PAG index");
  }
  if (!impl_->mutable_graph) {
    throw std::logic_error(
        "PAG insert requires BuildOptions::mode=IndexMode::Online");
  }
  if (vector == nullptr) {
    throw std::invalid_argument("PAG insert vector must not be null");
  }
  if (impl_->element_count >= impl_->max_elements) {
    throw std::runtime_error("PAG insert exceeds max_elements capacity");
  }

  std::vector<float> point;
  float vector_norm = 0.0f;
  impl_->prepareInsertPoint(vector, point, vector_norm);
  if (vector_norm <= 0.0f) {
    throw std::invalid_argument("PAG insert does not support zero-norm vectors");
  }

  impl_->core->insertPoint(point.data(), label, &vector_norm);
  impl_->element_count = impl_->core->currentCount();
  impl_->next_label = std::max<Label>(impl_->next_label, label + 1);
}

std::vector<Label> Index::add_batch(const float *row_major_vectors,
                                    size_t count) {
  if (count == 0) {
    return {};
  }
  if (!impl_->core) {
    throw std::logic_error("Cannot insert before building a PAG index");
  }
  if (!impl_->mutable_graph) {
    throw std::logic_error(
        "PAG insert requires BuildOptions::mode=IndexMode::Online");
  }
  if (row_major_vectors == nullptr) {
    throw std::invalid_argument("PAG batch insert vectors must not be null");
  }
  if (impl_->element_count + count > impl_->max_elements) {
    throw std::runtime_error("PAG insert exceeds max_elements capacity");
  }

  std::vector<Label> labels(count);
  const Label first_label = impl_->next_label;
  if (count > static_cast<size_t>(std::numeric_limits<Label>::max()) -
                  static_cast<size_t>(first_label)) {
    throw std::runtime_error("PAG add_batch label range exceeds Label capacity");
  }
  for (size_t i = 0; i < count; ++i) {
    labels[i] = first_label + static_cast<Label>(i);
  }
  insert_batch(row_major_vectors, labels.data(), count);
  return labels;
}

void Index::insert_batch(const float *row_major_vectors, const Label *labels,
                         size_t count) {
  if (count == 0) {
    return;
  }
  if (!impl_->core) {
    throw std::logic_error("Cannot insert before building a PAG index");
  }
  if (!impl_->mutable_graph) {
    throw std::logic_error(
        "PAG insert requires BuildOptions::mode=IndexMode::Online");
  }
  if (row_major_vectors == nullptr || labels == nullptr) {
    throw std::invalid_argument(
        "PAG batch insert vectors and labels must not be null");
  }
  if (impl_->element_count + count > impl_->max_elements) {
    throw std::runtime_error("PAG insert exceeds max_elements capacity");
  }

  std::atomic<size_t> first_bad(count);
#pragma omp parallel for schedule(static)
  for (size_t i = 0; i < count; ++i) {
    const float *vector = row_major_vectors + i * impl_->vector_dim;
    float norm = 0.0f;
    for (size_t d = 0; d < impl_->vector_dim; ++d) {
      norm += vector[d] * vector[d];
    }
    norm = std::sqrt(norm);
    if (norm <= 0.0f) {
      size_t expected = first_bad.load(std::memory_order_relaxed);
      while (i < expected &&
             !first_bad.compare_exchange_weak(expected, i,
                                              std::memory_order_relaxed)) {
      }
    }
  }
  if (first_bad.load(std::memory_order_relaxed) != count) {
    throw std::invalid_argument("PAG insert does not support zero-norm vectors");
  }
  impl_->core->reserveLabelMetadataForLabels(labels, count);

  std::exception_ptr first_exception;
  std::mutex exception_mutex;
  std::atomic<bool> failed(false);

#pragma omp parallel for schedule(dynamic)
  for (size_t i = 0; i < count; ++i) {
    if (failed.load(std::memory_order_relaxed)) {
      continue;
    }
    try {
      const float *vector = row_major_vectors + i * impl_->vector_dim;
      std::vector<float> point;
      float vector_norm = 0.0f;
      impl_->prepareInsertPoint(vector, point, vector_norm);
      impl_->core->insertPoint(point.data(), labels[i], &vector_norm);
    } catch (...) {
      failed.store(true, std::memory_order_relaxed);
      std::lock_guard<std::mutex> lock(exception_mutex);
      if (!first_exception) {
        first_exception = std::current_exception();
      }
    }
  }

  if (first_exception) {
    std::rethrow_exception(first_exception);
  }

  impl_->element_count = impl_->core->currentCount();
  Label next_label = impl_->next_label;
  for (size_t i = 0; i < count; ++i) {
    next_label = std::max<Label>(next_label, labels[i] + 1);
  }
  impl_->next_label = next_label;
}


bool Index::is_loaded() const { return impl_->core != nullptr; }

Metric Index::metric() const { return ToPublicMetric(impl_->metric); }

size_t Index::dimension() const { return impl_->vector_dim; }

} // namespace pag

static void RunPAGWorkflow(const PAGRunConfig &command) {
  PAGRuntimeConfig runtime = MakeRuntimeConfig(command);
  const PAGMetric metric = ParseMetricName(command.metric_name);
  GroundTruthShape truth_shape =
      InferGroundTruthShape(command, runtime.query_count);
  if (runtime.topk > truth_shape.stride) {
    throw std::runtime_error("Requested topk is larger than truth width");
  }
  const size_t requested_max_search_k =
      command.max_search_k > 0 ? static_cast<size_t>(command.max_search_k) : 0;
  if (requested_max_search_k != 0 && requested_max_search_k < runtime.topk) {
    throw std::runtime_error("max_search_k must be at least topk");
  }
  if (requested_max_search_k > truth_shape.stride) {
    throw std::runtime_error("max_search_k is larger than truth width");
  }
  const size_t benchmark_max_query_k =
      requested_max_search_k != 0
          ? requested_max_search_k
          : (metric == PAGMetric::MaximumInnerProduct ? runtime.topk
                                                      : truth_shape.stride);
  runtime.max_query_top_k = benchmark_max_query_k;
  runtime.max_truth_k =
      ComputeWorkingSetSize(static_cast<int>(benchmark_max_query_k));
  if (metric == PAGMetric::MaximumInnerProduct &&
      runtime.base_count < runtime.max_truth_k) {
    throw std::runtime_error(
        "PAG MIPS build requires at least ComputeWorkingSetSize(max_search_k) "
        "vectors for projection metadata");
  }

  std::vector<std::vector<std::vector<float>>> projection_vectors(
      runtime.projection_levels,
      std::vector<std::vector<float>>(
          runtime.projection_width,
          std::vector<float>(runtime.projection_subspace_dim, 0.0f)));

  L2Space l2space(runtime.padded_dim);
  InnerProductSpace ipsubspace(runtime.projection_subspace_dim);
  InnerProductSpace ipspace(runtime.padded_dim);
  PAGIndexCore<float> *pag_index;
  PAGIndexOptions index_options;
  index_options.metric = metric;

  std::vector<float> input_buffer(runtime.dim);
  fs::path dir(command.index_dir);
  if (fs::exists(dir)) {
    pag_index =
        new PAGIndexCore<float>(&l2space, &ipspace, &ipsubspace,
                                command.index_dir, false, 0, index_options);

    GroundTruthData truth = LoadGroundTruthIds(command, runtime);
    float *query_vectors = LoadQueryVectors(command, runtime, metric);

    vector<std::priority_queue<std::pair<float, labeltype>>> answers;
    size_t k = runtime.topk;
    BuildGroundTruthQueues(truth.ids.data(), runtime.query_count, answers, k,
                           truth.max_truth_k);

    const int query_table_size =
        runtime.projection_levels * 2 * runtime.projection_width;
    float *query_table = (float *)std::aligned_alloc(
        kAlignment, query_table_size * sizeof(float));
    RunSearchBenchmark(query_vectors, runtime.query_count, *pag_index, answers,
                       k, query_table, runtime.dim, runtime.extended_dim,
                       command.index_dir);
  } else {
    fs::create_directories(dir);
    MicrosecondTimer total_build_timer;

    ifstream input(command.base_file, ios::binary);
    MatrixLayout base_layout = DetectFloatMatrixLayout(
        command.base_file, runtime.base_count, runtime.dim);
    float *padded_vector = new float[runtime.extended_dim];
    std::vector<float> half_squared_norms(runtime.base_count, 0.0f);
    std::vector<float> vector_norms(runtime.base_count, 0.0f);
    std::vector<float> center_vector(runtime.padded_dim, 0.0f);
    std::vector<double> center_accumulator(runtime.padded_dim, 0.0f);
    std::vector<float> dim_energy(runtime.extended_dim, 0.0f);
    std::vector<double> dim_energy_accumulator(runtime.extended_dim, 0.0f);

    double squared_value;

    if (base_layout.has_header) {
      ReadMatrixHeader(input);
    }

    for (int i = 0; i < runtime.base_count; i++) {
      input.read((char *)input_buffer.data(), sizeof(float) * runtime.dim);
      memcpy(padded_vector, input_buffer.data(), sizeof(float) * runtime.dim);
      memset(padded_vector + runtime.dim, 0,
             sizeof(float) * (runtime.extended_dim - runtime.dim));

      float sum = 0;
      for (int j = 0; j < runtime.dim; j++) {
        squared_value = padded_vector[j] * padded_vector[j];
        sum += squared_value;
      }

      if (metric == PAGMetric::Cosine) {
        vector_norms[i] = std::sqrt(sum);
        if (vector_norms[i] > 0.0f) {
          const float inv_norm = 1.0f / vector_norms[i];
          for (int j = 0; j < runtime.dim; j++) {
            padded_vector[j] *= inv_norm;
          }
        }
        half_squared_norms[i] = 0.5f;
        vector_norms[i] = 1.0f;
      } else if (metric == PAGMetric::MaximumInnerProduct) {
        half_squared_norms[i] = sum / 2.0f;
        vector_norms[i] = std::sqrt(sum);
      } else {
        half_squared_norms[i] = sum;
        vector_norms[i] = sqrt(half_squared_norms[i]);
        half_squared_norms[i] = half_squared_norms[i] / 2;
      }

      for (int d = 0; d < runtime.padded_dim; d++)
        center_accumulator[d] += padded_vector[d];

      for (int j = 0; j < runtime.dim; j++)
        dim_energy_accumulator[j] += padded_vector[j] * padded_vector[j];
    }

    for (int d = 0; d < runtime.dim; d++)
      dim_energy[d] = dim_energy_accumulator[d] / runtime.base_count;

    for (int d = 0; d < runtime.padded_dim; d++)
      center_vector[d] = center_accumulator[d] / runtime.base_count;

    if (metric == PAGMetric::Cosine) {
      NormalizeVectorInPlace(center_vector.data(), runtime.padded_dim);
    }

    std::vector<std::pair<float, int>> distance_to_center(runtime.base_count);
    std::vector<bool> is_initial_seed(runtime.base_count, false);

    input.clear();
    input.seekg(0, std::ios::beg);

    if (base_layout.has_header) {
      ReadMatrixHeader(input);
    }

    for (int i = 0; i < runtime.base_count; i++) {
      input.read((char *)input_buffer.data(), sizeof(float) * runtime.dim);

      if (metric == PAGMetric::Cosine) {
        NormalizeVectorInPlace(input_buffer.data(), runtime.dim);
      }

      float sum = 0;
      if (metric == PAGMetric::MaximumInnerProduct) {
        for (int j = 0; j < runtime.dim; j++) {
          sum += input_buffer[j] * input_buffer[j];
        }
        distance_to_center[i].first = -sum;
      } else {
        for (int j = 0; j < runtime.dim; j++) {
          float tmp = input_buffer[j] - center_vector[j];
          sum += tmp * tmp;
        }
        distance_to_center[i].first = sum;
      }

      distance_to_center[i].second = i;
    }

    if (runtime.construction_beam_width <
        static_cast<int>(distance_to_center.size())) {
      std::nth_element(distance_to_center.begin(),
                       distance_to_center.begin() +
                           runtime.construction_beam_width,
                       distance_to_center.end());
    }
    std::sort(distance_to_center.begin(),
              distance_to_center.begin() + runtime.construction_beam_width);

    std::vector<std::pair<float, int>> initial_seed_ids(
        distance_to_center.begin(),
        distance_to_center.begin() + runtime.construction_beam_width);
    if (metric != PAGMetric::MaximumInnerProduct) {
      std::sort(
          initial_seed_ids.begin(), initial_seed_ids.end(),
          [](const auto &a, const auto &b) { return a.second < b.second; });
    }

    std::vector<std::vector<float>> initial_vectors(
        runtime.construction_beam_width, std::vector<float>(runtime.dim));

    for (int i = 0; i < runtime.construction_beam_width; i++) {
      int target_id = initial_seed_ids[i].second;
      is_initial_seed[target_id] = true;
      SeekVector(input, base_layout, target_id, runtime.dim);
      input.read((char *)initial_vectors[i].data(),
                 sizeof(float) * runtime.dim);
      if (metric == PAGMetric::Cosine) {
        NormalizeVectorInPlace(initial_vectors[i].data(), runtime.dim);
      }
    }

    input.clear();
    input.seekg(0, std::ios::beg);

    if (base_layout.has_header) {
      ReadMatrixHeader(input);
    }

    std::vector<int> permutation;
    std::vector<int> zero_positions;

    BuildBalancedPermutation(dim_energy, runtime.extended_dim,
                             runtime.projection_levels, permutation,
                             zero_positions);
    BuildProjectionVectors(projection_vectors, runtime.projection_levels,
                           runtime.projection_subspace_dim,
                           runtime.projection_width, zero_positions);

    std::vector<int> mips_entry_permutation;
    std::vector<std::vector<std::vector<float>>> mips_entry_directions;
    if (metric == PAGMetric::MaximumInnerProduct) {
      std::vector<int> mips_entry_zero_positions;
      const int mips_entry_levels = 4;
      const int mips_entry_width = 8;
      const int mips_entry_subdim = runtime.extended_dim / mips_entry_levels;
      mips_entry_directions.assign(
          mips_entry_levels,
          std::vector<std::vector<float>>(
              mips_entry_width, std::vector<float>(mips_entry_subdim, 0.0f)));
      BuildBalancedPermutation(dim_energy, runtime.extended_dim,
                               mips_entry_levels, mips_entry_permutation,
                               mips_entry_zero_positions);
      BuildProjectionVectors(mips_entry_directions, mips_entry_levels,
                             mips_entry_subdim, mips_entry_width,
                             mips_entry_zero_positions);
    }

    SavePermutation(command.index_dir, permutation);

    std::vector<std::pair<float, int>>().swap(distance_to_center);

    pag_index = new PAGIndexCore<float>(
        runtime.max_truth_k, runtime.construction_beam_width,
        runtime.projection_width, runtime.dim, projection_vectors,
        runtime.projection_levels, runtime.projection_subspace_dim,
        half_squared_norms, permutation, &l2space, &ipspace, &ipsubspace,
        runtime.base_count, command.index_dir, runtime.target_degree,
        runtime.ef_construction, index_options);
    pag_index->setMaxQueryTopK(runtime.max_query_top_k);

    if (metric == PAGMetric::MaximumInnerProduct) {
      const int pif_entries_per_bucket =
          ComputeWorkingSetSize(static_cast<int>(runtime.max_query_top_k));
      pag_index->configurePIF(mips_entry_permutation, mips_entry_directions,
                              pif_entries_per_bucket);
      pag_index->buildInitialPIFTable(initial_vectors);
    }

    for (int i = 0; i < runtime.construction_beam_width; i++) {
      float *point_buffer = new float[runtime.extended_dim];
      std::memset(point_buffer, 0, runtime.extended_dim * sizeof(float));
      std::memcpy(point_buffer, initial_vectors[i].data(),
                  runtime.dim * sizeof(float));

      int node_id = initial_seed_ids[i].second;
      pag_index->insertPoint((void *)point_buffer, (size_t)node_id,
                             &(vector_norms[node_id]));
      delete[] point_buffer;
    }

    std::atomic<size_t> build_progress(runtime.construction_beam_width);
    size_t last_progress_reported = runtime.construction_beam_width;
    const size_t progress_interval =
        std::max<size_t>(1, runtime.base_count / 100);
    PrintBuildProgress(last_progress_reported, runtime.base_count);

#pragma omp parallel for schedule(dynamic)
    for (int insertion_ord = 0; insertion_ord < runtime.base_count;
         insertion_ord++) {
      int i = insertion_ord;
      if (is_initial_seed[i])
        continue;

      float *point_buffer = new float[runtime.extended_dim];
      std::memset(point_buffer, 0, runtime.extended_dim * sizeof(float));

#pragma omp critical
      {
        SeekVector(input, base_layout, i, runtime.dim);
        input.read((char *)point_buffer, sizeof(float) * runtime.dim);
        if (metric == PAGMetric::Cosine) {
          NormalizeVectorInPlace(point_buffer, runtime.dim);
        }
      }

      pag_index->insertPoint((void *)point_buffer, (size_t)i,
                             &(vector_norms[i]));
      delete[] point_buffer;

      const size_t completed =
          build_progress.fetch_add(1, std::memory_order_relaxed) + 1;
      if (completed == runtime.base_count ||
          completed % progress_interval == 0) {
#pragma omp critical(PAGBuildProgress)
        {
          if (completed > last_progress_reported) {
            last_progress_reported = completed;
            PrintBuildProgress(completed, runtime.base_count);
          }
        }
      }
    }

#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < runtime.base_count; i++) {
      pag_index->completePESForNode(i);
    }
#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < runtime.base_count; i++) {
      pag_index->reservePESProjectionStorageForNode(i);
    }
#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < runtime.base_count; i++) {
      pag_index->encodePESProjectionsForNode(i);
    }

    if (metric == PAGMetric::MaximumInnerProduct) {
      pag_index->initializeFallbackEntryPointSeeds();
    } else {
      pag_index->selectEntryPointSeeds(center_vector.data());
    }
    pag_index->packSearchLayout(runtime.base_count);
    pag_index->save();
    input.close();
    std::cerr << "PAG build time: " << 1e-6 * total_build_timer.elapsedMicros()
              << " seconds\n";
  }
  return;
}

void RunPAG(const PAGRunConfig &config) { RunPAGWorkflow(config); }
