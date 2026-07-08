#pragma once
#include "paglib.h"
#include "visited_list_pool.h"
#include <algorithm>
#include <array>
#include <assert.h>
#include <atomic>
#include <cmath>
#include <cstring>
#include <immintrin.h>
#include <limits>
#include <list>
#include <random>
#include <stdexcept>
#include <stdlib.h>
#include <unordered_set>
#include <unordered_map>

#include "pag_search_primitives.h"

namespace paglib {
typedef unsigned int tableint;
typedef unsigned int linklistsizeint;

template <typename dist_t>
class PAGIndexCore : public PAGAlgorithmInterface<dist_t> {
public:
  static const tableint max_update_element_locks = 65536;
  PAGIndexCore(SpaceInterface<dist_t> *s) {}

  PAGIndexCore(SpaceInterface<dist_t> *s1, SpaceInterface<dist_t> *s2,
               SpaceInterface<dist_t> *s3, const char *path_index,
               bool nmslib = false, size_t max_elements = 0,
               PAGIndexOptions options = PAGIndexOptions()) {
    options_ = options;
    loadIndexFile(path_index, s1, s2, s3, max_elements);
  }

  PAGIndexCore(const PAGSpaceBundle<dist_t> &spaces, const char *path_index,
               size_t max_elements = 0) {
    options_ = spaces.options;
    loadIndexFile(path_index, spaces.vector_space, spaces.inner_product_space,
                  spaces.projection_space, max_elements);
  }

  PAGIndexCore(
      size_t max_entry_points, int construction_beam_width,
      size_t projection_width, size_t vector_dim,
      std::vector<std::vector<std::vector<float>>> &projection_directions,
      int projection_level_count, int projection_subspace_dim,
      std::vector<float> &norm, std::vector<int> &permutation,
      SpaceInterface<dist_t> *s1, SpaceInterface<dist_t> *s2,
      SpaceInterface<dist_t> *s3, size_t max_elements, const char *path_index,
      size_t target_degree = 16, size_t ef_construction = 200,
      PAGIndexOptions options = PAGIndexOptions())
      : link_list_locks_(max_elements), element_levels_(max_elements) {

    options_ = options;
    max_elements_ = max_elements;

    max_entry_points_ = max_entry_points;
    max_query_top_k_ = max_entry_points;
    construction_beam_width_ = construction_beam_width;

    permutation_ = permutation;
    vecsize_ = max_elements;
    projection_directions_ = projection_directions;
    half_squared_norms_ = norm;
    vector_dim_ = vector_dim;

    padded_dim_ = (vector_dim_ + 15) & ~0xF;
    extended_dim_ =
        ((padded_dim_ + projection_level_count - 1) / projection_level_count) *
        projection_level_count;

    vector_distance_func_ = s1->get_dist_func();
    inner_product_func_ = s2->get_dist_func();
    vector_space_param_ = s1->get_dist_func_param();
    projection_distance_func_ = s3->get_dist_func();
    projection_distance_param_ = s3->get_dist_func_param();

    projection_level_count_ = projection_level_count;
    projection_subspace_dim_ = projection_subspace_dim;

    query_projection_directions_ =
        new float[(size_t)projection_level_count_ * projection_width *
                  projection_subspace_dim_];
    float *projection_direction_dst = query_projection_directions_;
    for (int level_id = 0; level_id < projection_level_count_; ++level_id) {
      for (size_t direction_id = 0; direction_id < projection_width;
           ++direction_id) {
        std::memcpy(projection_direction_dst,
                    projection_directions_[level_id][direction_id].data(),
                    projection_subspace_dim_ * sizeof(float));
        projection_direction_dst += projection_subspace_dim_;
      }
    }

    target_degree_ = target_degree;
    max_upper_degree_ = target_degree_;
    max_base_degree_ = target_degree_ * 2;
    ef_construction_ = std::max(ef_construction, target_degree_);
    ef_ = 10;
    projection_width_ = projection_width;

    int half_level = projection_level_count_ / 2;
    const int prediction_term_count = options_.metric == PAGMetric::L2 ? 3 : 2;

    offset0 = 0;
    offset1 = 16 * (half_level);
    offset2 = 16 * (half_level + sizeof(int16_t));
    offset3 = 16 * (half_level + 2 * sizeof(int16_t));

    segment_size_ = 16 * (half_level + prediction_term_count * sizeof(int16_t));

    // level_generator_.seed(random_seed);
    // update_probability_generator_.seed(random_seed + 1);

    // pre_size_links_level0_ = max_base_degree_ * (3 * sizeof(float) +
    // sizeof(tableint) + half_level);
    size_links_level0_ =
        max_base_degree_ * (prediction_term_count * sizeof(int16_t) +
                            sizeof(tableint) + half_level) +
        sizeof(linklistsizeint); // new

    data_size_ = padded_dim_ * sizeof(int16_t);

    size_data_per_element_ =
        size_links_level0_ + data_size_ + sizeof(labeltype) + 2 * sizeof(float);
    offsetData_ = size_links_level0_;
    label_offset_ = data_size_ + 2 * sizeof(float);
    vector_record_bytes_ = data_size_ + 2 * sizeof(float) + sizeof(labeltype);

    initial_adjacency_record_bytes_ = max_base_degree_ * sizeof(tableint) +
                                      segment_size_ + sizeof(linklistsizeint);
    adjacency_records_ = (char **)malloc(vecsize_ * sizeof(char *));
    for (int i = 0; i < vecsize_; i++) {
      adjacency_records_[i] = (char *)malloc(initial_adjacency_record_bytes_);
    }

    vector_records_ = (char *)malloc(vecsize_ * vector_record_bytes_);

    cur_element_count_ = 0;

    visited_list_pool_ = new VisitedListPool(1, max_elements);

    enterpoint_node_ = -1;
    maxlevel_ = -1;

    size_links_per_element_ =
        max_upper_degree_ * sizeof(tableint) + sizeof(linklistsizeint);
    mult_ = 1 / log(1.0 * target_degree_);
    revSize_ = 1.0 / mult_;
    pes_candidates_.resize(vecsize_);
    base_record_is_full_ = new bool[vecsize_]();
    pes_start_pos_ = new int[vecsize_];

    std::string folderPath(path_index);
    std::string fullPath;
    if (!folderPath.empty() &&
        (folderPath.back() == '/' || folderPath.back() == '\\')) {
      indexPath_ = folderPath + "index.bin";
      infoPath_ = folderPath + "info.bin";
    } else {
      indexPath_ = folderPath + "/index.bin";
      infoPath_ = folderPath + "/info.bin";
    }
  }

  inline float dot_product_avx512(const float *__restrict a,
                                  const float *__restrict b) const {

    __m512 sum0 = _mm512_setzero_ps();
    __m512 sum1 = _mm512_setzero_ps();
    __m512 sum2 = _mm512_setzero_ps();
    __m512 sum3 = _mm512_setzero_ps();

    int i = 0;
    int limit = padded_dim_ & (~63);

    for (; i < limit; i += 64) {
      __m512 va0 = _mm512_loadu_ps(a + i);
      __m512 vb0 = _mm512_loadu_ps(b + i);
      sum0 = _mm512_fmadd_ps(va0, vb0, sum0);

      __m512 va1 = _mm512_loadu_ps(a + i + 16);
      __m512 vb1 = _mm512_loadu_ps(b + i + 16);
      sum1 = _mm512_fmadd_ps(va1, vb1, sum1);

      __m512 va2 = _mm512_loadu_ps(a + i + 32);
      __m512 vb2 = _mm512_loadu_ps(b + i + 32);
      sum2 = _mm512_fmadd_ps(va2, vb2, sum2);

      __m512 va3 = _mm512_loadu_ps(a + i + 48);
      __m512 vb3 = _mm512_loadu_ps(b + i + 48);
      sum3 = _mm512_fmadd_ps(va3, vb3, sum3);
    }

    __m512 final_sum = _mm512_add_ps(sum0, sum1);
    final_sum = _mm512_add_ps(final_sum, sum2);
    final_sum = _mm512_add_ps(final_sum, sum3);

    float result = _mm512_reduce_add_ps(final_sum);

    for (; i < padded_dim_; i++) {
      result += a[i] * b[i];
    }
    return result;
  }

  float maxAbsValue(const float *__restrict vec) const {
    __m512 max_abs_vec = _mm512_setzero_ps();

    for (int i = 0; i < padded_dim_; i += 16) {
      __m512 v = _mm512_loadu_ps(vec + i);

      __m512 abs_v = _mm512_andnot_ps(_mm512_set1_ps(-0.0f), v);

      max_abs_vec = _mm512_max_ps(max_abs_vec, abs_v);
    }

    float max_abs = _mm512_reduce_max_ps(max_abs_vec);
    return std::max(max_abs, 1e-6f);
  }

  inline int16_t encodeFloatToInt16(float x, float scale) const {
    float scaled = x * scale;

    int32_t q_int32 = static_cast<int32_t>(std::round(scaled));

    int16_t q = static_cast<int16_t>(q_int32);

    const int16_t MAX_VAL = 32767;
    const int16_t MIN_VAL = -32767;

    if (q_int32 > MAX_VAL) {
      q = MAX_VAL;
    } else if (q_int32 < MIN_VAL) {
      q = MIN_VAL;
    }
    return q;
  }

  inline float dot_product_avx512_f32_i16(
      const float *__restrict q,    // float32 query
      const int16_t *__restrict vq, // stored int16 vector
      float scale_v                 // int16 restore scale
  ) const {
    __m512 acc0 = _mm512_setzero_ps();
    __m512 acc1 = _mm512_setzero_ps();

    const __m512 inv_scale = _mm512_set1_ps(1.0f / scale_v);

    int i = 0;
    int limit = padded_dim_ & ~31; // multiples of 32

    for (; i < limit; i += 32) {
      __m512 q0 = _mm512_loadu_ps(q + i);
      __m512 q1 = _mm512_loadu_ps(q + i + 16);

      // ---- v0 ----
      __m256i v0_i16 = _mm256_loadu_si256((const __m256i *)(vq + i));
      __m512i v0_i32 = _mm512_cvtepi16_epi32(v0_i16);
      __m512 v0 = _mm512_cvtepi32_ps(v0_i32);

      // ---- v1 ----
      __m256i v1_i16 = _mm256_loadu_si256((const __m256i *)(vq + i + 16));
      __m512i v1_i32 = _mm512_cvtepi16_epi32(v1_i16);
      __m512 v1 = _mm512_cvtepi32_ps(v1_i32);

      v0 = _mm512_mul_ps(v0, inv_scale);
      v1 = _mm512_mul_ps(v1, inv_scale);

      acc0 = _mm512_fmadd_ps(q0, v0, acc0);
      acc1 = _mm512_fmadd_ps(q1, v1, acc1);
    }

    if (i < padded_dim_) {
      __m512 q_tail = _mm512_loadu_ps(q + i);

      __m256i v_tail_i16 = _mm256_loadu_si256((const __m256i *)(vq + i));
      __m512i v_tail_i32 = _mm512_cvtepi16_epi32(v_tail_i16);
      __m512 v_tail = _mm512_cvtepi32_ps(v_tail_i32);

      v_tail = _mm512_mul_ps(v_tail, inv_scale);
      acc0 = _mm512_fmadd_ps(q_tail, v_tail, acc0);
    }

    __m512 final_acc = _mm512_add_ps(acc0, acc1);
    float dot = _mm512_reduce_add_ps(final_acc);

    return dot;
  }

  inline float dot_product_avx512_int16(const int16_t *__restrict qa,
                                        const int16_t *__restrict qb,
                                        float scale_a, float scale_b) const {
    __m512i acc64 = _mm512_setzero_si512();
    const __m512i zero = _mm512_setzero_si512();

    int i = 0;

    int limit = padded_dim_ & ~31;

    for (; i < limit; i += 32) {
      __m512i va = _mm512_loadu_si512((const __m512i *)(qa + i));
      __m512i vb = _mm512_loadu_si512((const __m512i *)(qb + i));

      __m512i prod32 = _mm512_dpwssd_epi32(zero, va, vb);

      acc64 = _mm512_add_epi64(
          acc64, _mm512_cvtepi32_epi64(_mm512_extracti64x4_epi64(prod32, 0)));
      acc64 = _mm512_add_epi64(
          acc64, _mm512_cvtepi32_epi64(_mm512_extracti64x4_epi64(prod32, 1)));
    }

    int64_t dot = _mm512_reduce_add_epi64(acc64);

    if (i < padded_dim_) {
      __m256i va = _mm256_loadu_si256((const __m256i *)(qa + i));
      __m256i vb = _mm256_loadu_si256((const __m256i *)(qb + i));

      __m256i prod32 = _mm256_madd_epi16(va, vb);

      alignas(32) int32_t lane_sums[8];
      _mm256_store_si256((__m256i *)lane_sums, prod32);
      for (int k = 0; k < 8; ++k)
        dot += lane_sums[k];
    }

    return static_cast<float>(dot) / (scale_a * scale_b);
  }

  inline float dot_product_avx512_extended(const float *__restrict a,
                                           const float *__restrict b) const {

    __m512 sum0 = _mm512_setzero_ps();
    __m512 sum1 = _mm512_setzero_ps();
    __m512 sum2 = _mm512_setzero_ps();
    __m512 sum3 = _mm512_setzero_ps();

    int i = 0;
    int limit = extended_dim_ & (~63);

    for (; i < limit; i += 64) {
      __m512 va0 = _mm512_loadu_ps(a + i);
      __m512 vb0 = _mm512_loadu_ps(b + i);
      sum0 = _mm512_fmadd_ps(va0, vb0, sum0); // FMA: va0*vb0 + sum0

      __m512 va1 = _mm512_loadu_ps(a + i + 16);
      __m512 vb1 = _mm512_loadu_ps(b + i + 16);
      sum1 = _mm512_fmadd_ps(va1, vb1, sum1);

      __m512 va2 = _mm512_loadu_ps(a + i + 32);
      __m512 vb2 = _mm512_loadu_ps(b + i + 32);
      sum2 = _mm512_fmadd_ps(va2, vb2, sum2);

      __m512 va3 = _mm512_loadu_ps(a + i + 48);
      __m512 vb3 = _mm512_loadu_ps(b + i + 48);
      sum3 = _mm512_fmadd_ps(va3, vb3, sum3);
    }

    __m512 final_sum = _mm512_add_ps(sum0, sum1);
    final_sum = _mm512_add_ps(final_sum, sum2);
    final_sum = _mm512_add_ps(final_sum, sum3);

    float result = _mm512_reduce_add_ps(final_sum);

    for (; i < extended_dim_; i++) {
      result += a[i] * b[i];
    }

    return result;
  }

  struct CompareByFirst {
    constexpr bool
    operator()(std::pair<dist_t, tableint> const &a,
               std::pair<dist_t, tableint> const &b) const noexcept {
      return a.first < b.first;
    }
  };

  ~PAGIndexCore() {
    if (adjacency_records_ != nullptr) {
      for (size_t i = 0; i < vecsize_; ++i) {
        free(adjacency_records_[i]);
      }
      free(adjacency_records_);
    }
    if (linkLists_ != nullptr) {
      for (tableint i = 0; i < cur_element_count_; i++) {
        if (element_levels_[i] > 0) {
          free(linkLists_[i]);
        }
      }
      free(linkLists_);
    }
    delete visited_list_pool_;
    delete[] query_projection_directions_;
    if (loaded_index_memory_ != nullptr) {
      free(loaded_index_memory_);
    } else {
      free(vector_records_);
    }
    free(entry_point_vector_records_);
    delete[] pes_start_pos_;
    delete[] base_record_is_full_;
    delete[] packed_to_internal_id_;
    delete[] packed_group_record_bytes_;
    delete[] packed_group_offsets_;
    delete[] edge_group_counts_;
    delete[] edge_group_offsets_;
  }

  void setSearchEf(size_t ef) { setEf(ef); }

  void setConstructionEf(int ef_construction) override {
    setEfc(ef_construction);
  }

  void insertPoint(const void *data_point, labeltype label,
                   float *norm) override {
    insertPointImpl(data_point, label, norm);
  }

  void query(float *query_point, float *query_extended_point, size_t top_k,
             std::vector<Neighbor> &result, float *projection_table,
             int step) const override {
    queryImpl(query_point, query_extended_point, top_k, result,
              projection_table, step);
  }

  inline void searchKnn(float *query_point, float *query_extended_point,
                        size_t top_k, std::vector<Neighbor> &result,
                        float *projection_table, int step) const {
    queryImpl(query_point, query_extended_point, top_k, result,
              projection_table, step);
  }

  void exactSearch(float *query_point, size_t top_k,
                   std::vector<Neighbor> &result) const {
    result.clear();
    if (top_k == 0) {
      return;
    }

    std::priority_queue<Neighbor> heap;
    for (tableint internal_id = 0; internal_id < cur_element_count_;
         ++internal_id) {
      const Neighbor candidate = scoreInternalId(query_point, internal_id, false);
      if (heap.size() < top_k) {
        heap.push(candidate);
      } else if (candidate.distance < heap.top().distance) {
        heap.pop();
        heap.push(candidate);
      }
    }

    result.resize(heap.size());
    for (size_t i = heap.size(); i > 0; --i) {
      result[i - 1] = heap.top();
      heap.pop();
    }
    std::sort(result.begin(), result.end());
  }

  void onlineGraphSearchKnn(float *query_point, size_t top_k,
                            std::vector<Neighbor> &result) const {
    result.clear();
    if (top_k == 0 || cur_element_count_ == 0) {
      return;
    }

    struct MinCandidate {
      bool operator()(const std::pair<float, tableint> &a,
                      const std::pair<float, tableint> &b) const {
        return a.first > b.first;
      }
    };

    const size_t ef_search = std::max<size_t>(ef_, top_k);
    VisitedList *vl = visited_list_pool_->getFreeVisitedList();
    vl_type *visited_array = vl->mass;
    vl_type visited_array_tag = vl->curV;

    std::priority_queue<std::pair<float, tableint>,
                        std::vector<std::pair<float, tableint>>, MinCandidate>
        candidate_queue;
    std::priority_queue<std::pair<float, tableint>> top_candidates;

    auto add_seed = [&](tableint internal_id) {
      if (internal_id >= cur_element_count_ ||
          visited_array[internal_id] == visited_array_tag) {
        return;
      }
      visited_array[internal_id] = visited_array_tag;
      Neighbor scored = scoreInternalId(query_point, internal_id, true);
      candidate_queue.emplace(scored.distance, internal_id);
      top_candidates.emplace(scored.distance, internal_id);
      if (top_candidates.size() > ef_search) {
        top_candidates.pop();
      }
    };

    for (tableint seed : entry_point_ids_) {
      add_seed(seed);
    }
    if (candidate_queue.empty()) {
      add_seed(0);
    }

    while (!candidate_queue.empty()) {
      const auto current = candidate_queue.top();
      if (top_candidates.size() >= ef_search &&
          current.first > top_candidates.top().first) {
        break;
      }
      candidate_queue.pop();

      int *link_record = reinterpret_cast<int *>(get_linklist0(current.second));
      const int degree = getListCount(reinterpret_cast<linklistsizeint *>(link_record));
      int *neighbors = link_record + 1;
      for (int i = 0; i < degree; ++i) {
        const tableint candidate_id = static_cast<tableint>(neighbors[i]);
        if (candidate_id >= cur_element_count_ ||
            visited_array[candidate_id] == visited_array_tag) {
          continue;
        }
        visited_array[candidate_id] = visited_array_tag;
        Neighbor scored = scoreInternalId(query_point, candidate_id, true);
        if (top_candidates.size() < ef_search ||
            scored.distance < top_candidates.top().first) {
          candidate_queue.emplace(scored.distance, candidate_id);
          top_candidates.emplace(scored.distance, candidate_id);
          if (top_candidates.size() > ef_search) {
            top_candidates.pop();
          }
        }
      }
    }

    std::vector<std::pair<float, tableint>> ordered;
    ordered.reserve(top_candidates.size());
    while (!top_candidates.empty()) {
      ordered.push_back(top_candidates.top());
      top_candidates.pop();
    }
    std::sort(ordered.begin(), ordered.end());

    result.reserve(std::min(top_k, ordered.size()));
    for (const auto &candidate : ordered) {
      result.push_back(scoreInternalId(query_point, candidate.second, false));
      if (result.size() == top_k) {
        break;
      }
    }

    visited_list_pool_->releaseVisitedList(vl);
  }

  void onlineSearchKnn(float *query_point, float *query_extended_point,
                       size_t top_k, std::vector<Neighbor> &result,
                       float *projection_table, int step) const {
    queryImpl(query_point, query_extended_point, top_k, result, projection_table,
              step);
  }

  void completePESForNode(int node_id) override {
    completePESForNodeImpl(node_id);
  }

  void reservePESProjectionStorageForNode(int node_id) override {
    reservePESProjectionStorageForNodeImpl(node_id);
  }

  void encodePESProjectionsForNode(int node_id) override {
    encodePESProjectionsForNodeImpl(node_id);
  }

  void selectEntryPointSeeds(float *center_vector) override {
    selectEntryPointSeedsImpl(center_vector);
  }

  void initializeFallbackEntryPointSeeds() override {
    initializeFallbackEntryPointSeedsImpl();
  }

  void packSearchLayout(int indexed_count) override {
    packSearchLayoutImpl(indexed_count);
  }

  void save() override { saveIndexFile(); }

  void load(const char *path_index, const PAGSpaceBundle<dist_t> &spaces,
            size_t max_elements = 0) {
    options_ = spaces.options;
    loadIndexFile(path_index, spaces.vector_space, spaces.inner_product_space,
                  spaces.projection_space, max_elements);
  }

  PAGIndexOptions options() const { return options_; }

  PAGMetric metric() const { return options_.metric; }

  void setMaxQueryTopK(size_t top_k) {
    max_query_top_k_ = std::max<size_t>(1, top_k);
  }

  size_t maxQueryTopK() const {
    return max_query_top_k_ == 0 ? max_entry_points_ : max_query_top_k_;
  }

  size_t currentCount() const { return cur_element_count_; }

  void enableOnlineEntryPointTracking() {
    rebuildLabelLookup();
    track_online_inserted_entry_points_ = true;
    online_inserted_entry_ids_.clear();
  }

  void rebuildLabelLookup() {
    label_to_internal_id_.clear();
    label_to_internal_id_.reserve(cur_element_count_);
    for (tableint internal_id = 0; internal_id < cur_element_count_;
         ++internal_id) {
      label_to_internal_id_[*getExternalLabelPtr(internal_id)] = internal_id;
    }
  }

  void reserveLabelMetadataForLabels(const labeltype *labels, size_t count) {
    size_t required_size = 0;
    for (size_t i = 0; i < count; ++i) {
      required_size =
          std::max(required_size, static_cast<size_t>(labels[i]) + 1);
    }
    std::unique_lock<std::mutex> label_lock(label_metadata_guard_);
    if (required_size > half_squared_norms_.size()) {
      half_squared_norms_.resize(required_size, 0.0f);
    }
  }

  Neighbor scoreInternalId(float *query_point, tableint internal_id,
                           bool needs_expansion) const {
    float *record = reinterpret_cast<float *>(getNormByInternalIdQuery(internal_id));
    const float vector_norm = *record++;
    const float vector_scale = *record++;
    const int16_t *stored_vector = reinterpret_cast<int16_t *>(record);
    const float inner_product =
        dot_product_avx512_f32_i16(query_point, stored_vector, vector_scale);
    float distance;
    if (isCosineMetric()) {
      distance = 0.5f - inner_product;
    } else if (isMipsMetric()) {
      distance = -vector_norm * inner_product;
    } else {
      distance = vector_norm * (vector_norm * 0.5f - inner_product);
    }
    const labeltype label =
        *reinterpret_cast<const labeltype *>(stored_vector + padded_dim_);
    return Neighbor(label, distance, inner_product, needs_expansion,
                    adjacency_records_ == nullptr ? nullptr
                                                  : adjacency_records_[internal_id]);
  }
  void configurePIF(
      const std::vector<int> &entry_permutation,
      const std::vector<std::vector<std::vector<float>>> &entry_directions,
      int entry_count) {
    if (!isMipsMetric()) {
      return;
    }
    entry_count = std::max(10, entry_count);
    pif_projection_width_ =
        entry_directions.empty() ? 0 : (int)entry_directions[0].size();
    pif_subspace_dim_ = extended_dim_ / 4;
    pif_entries_per_bucket_ = entry_count;
    pif_permutation_ = entry_permutation;
    pif_projection_directions_.assign(
        (size_t)4 * pif_projection_width_ * pif_subspace_dim_, 0.0f);
    for (int s = 0; s < 4; ++s) {
      for (int i = 0; i < pif_projection_width_; ++i) {
        std::memcpy(
            pif_projection_directions_.data() +
                ((size_t)s * pif_projection_width_ + i) * pif_subspace_dim_,
            entry_directions[s][i].data(), pif_subspace_dim_ * sizeof(float));
      }
    }
  }

  void buildInitialPIFTable(
      const std::vector<std::vector<float>> &entry_vectors,
      const std::vector<int> &entry_internal_ids = std::vector<int>()) {
    if (!isMipsMetric() || pif_projection_width_ <= 0 ||
        pif_entries_per_bucket_ <= 0) {
      return;
    }
    const int requested_entries_per_bucket = std::max(10, pif_entries_per_bucket_);
    const int npts =
        std::min<int>(requested_entries_per_bucket,
                      static_cast<int>(entry_vectors.size()));
    if (!entry_internal_ids.empty() &&
        static_cast<int>(entry_internal_ids.size()) < npts) {
      throw std::invalid_argument(
          "PIF entry id list is smaller than the entry vector list");
    }
    std::vector<int> entry_ids(npts);
    pif_preloaded_internal_ids_.assign(max_elements_, 0);
    for (int i = 0; i < npts; ++i) {
      entry_ids[i] = entry_internal_ids.empty() ? i : entry_internal_ids[i];
      if (entry_ids[i] < 0 ||
          static_cast<size_t>(entry_ids[i]) >= max_elements_) {
        throw std::out_of_range("PIF entry internal id is outside index capacity");
      }
      pif_preloaded_internal_ids_[entry_ids[i]] = 1;
    }
    const int signed_m = 2 * pif_projection_width_;
    const int pair_cols = signed_m * signed_m;
    const size_t table_cols = static_cast<size_t>(pair_cols) * pair_cols;
    const PIFEntry empty_entry{
        -std::numeric_limits<float>::infinity(),
        entry_ids.empty() ? 0 : entry_ids[0]};
    pif_bucket_top_entries_.assign(
        table_cols * static_cast<size_t>(requested_entries_per_bucket),
        empty_entry);
    pif_bucket_tail_scores_.assign(table_cols,
                                   -std::numeric_limits<float>::infinity());
    std::vector<std::mutex>(table_cols).swap(pif_bucket_locks_);
    pif_entries_per_bucket_ = requested_entries_per_bucket;

    std::vector<float> permuted_points(
        static_cast<size_t>(npts) * extended_dim_, 0.0f);
#pragma omp parallel for schedule(static)
    for (int i = 0; i < npts; ++i) {
      float *dst =
          permuted_points.data() + static_cast<size_t>(i) * extended_dim_;
      const auto &src = entry_vectors[i];
      for (int j = 0; j < extended_dim_; ++j) {
        const int src_dim = pif_permutation_[j];
        dst[j] = src_dim < static_cast<int>(src.size()) ? src[src_dim] : 0.0f;
      }
    }

    std::vector<float> projection_values(static_cast<size_t>(4) * signed_m *
                                         npts);
#pragma omp parallel for schedule(static)
    for (int i = 0; i < npts; ++i) {
      const float *point =
          permuted_points.data() + static_cast<size_t>(i) * extended_dim_;
      for (int s = 0; s < 4; ++s) {
        const float *subvector =
            point + static_cast<size_t>(s) * pif_subspace_dim_;
        for (int j = 0; j < pif_projection_width_; ++j) {
          const float *direction =
              pif_projection_directions_.data() +
              ((size_t)s * pif_projection_width_ + j) * pif_subspace_dim_;
          const float value = projection_distance_func_(
              (void *)subvector, (void *)direction, projection_distance_param_);
          projection_values[((size_t)s * signed_m + j) * npts + i] = value;
          projection_values[((size_t)s * signed_m +
                             (pif_projection_width_ + j)) *
                                npts +
                            i] = -value;
        }
      }
    }

    std::vector<float> pair01(static_cast<size_t>(pair_cols) * npts);
    std::vector<float> pair23(static_cast<size_t>(pair_cols) * npts);
#pragma omp parallel for collapse(2) schedule(dynamic)
    for (int a = 0; a < signed_m; ++a) {
      for (int b = 0; b < signed_m; ++b) {
        const size_t pair_id = static_cast<size_t>(a) * signed_m + b;
        const float *left = projection_values.data() +
                            (static_cast<size_t>(0) * signed_m + a) * npts;
        const float *right = projection_values.data() +
                             (static_cast<size_t>(1) * signed_m + b) * npts;
        float *out = pair01.data() + pair_id * npts;
        for (int i = 0; i < npts; ++i) {
          out[i] = left[i] + right[i];
        }
      }
    }
#pragma omp parallel for collapse(2) schedule(dynamic)
    for (int c = 0; c < signed_m; ++c) {
      for (int d = 0; d < signed_m; ++d) {
        const size_t pair_id = static_cast<size_t>(c) * signed_m + d;
        const float *left = projection_values.data() +
                            (static_cast<size_t>(2) * signed_m + c) * npts;
        const float *right = projection_values.data() +
                             (static_cast<size_t>(3) * signed_m + d) * npts;
        float *out = pair23.data() + pair_id * npts;
        for (int i = 0; i < npts; ++i) {
          out[i] = left[i] + right[i];
        }
      }
    }

#pragma omp parallel
    {
      std::vector<PIFEntry> buffer(pif_entries_per_bucket_, empty_entry);
#pragma omp for collapse(2) schedule(dynamic)
      for (int p01 = 0; p01 < pair_cols; ++p01) {
        for (int p23 = 0; p23 < pair_cols; ++p23) {
          std::fill(buffer.begin(), buffer.end(), empty_entry);
          const size_t col_id =
              static_cast<size_t>(p01) * pair_cols + static_cast<size_t>(p23);
          const float *left = pair01.data() + static_cast<size_t>(p01) * npts;
          const float *right = pair23.data() + static_cast<size_t>(p23) * npts;
          for (int i = 0; i < npts; ++i) {
            buffer[i] = PIFEntry{left[i] + right[i], entry_ids[i]};
          }
          std::sort(buffer.begin(), buffer.end(),
                    [](const PIFEntry &a, const PIFEntry &b) {
                      return a.value > b.value;
                    });
          std::memcpy(pif_bucket_top_entries_.data() +
                          col_id * pif_entries_per_bucket_,
                      buffer.data(),
                      static_cast<size_t>(pif_entries_per_bucket_) *
                          sizeof(PIFEntry));
          pif_bucket_tail_scores_[col_id] =
              buffer[pif_entries_per_bucket_ - 1].value;
        }
      }
    }
    has_mips_entry_table_ = true;
  }

  inline void updatePIFTableForInsertedPoint(const float *point,
                                             int internal_id) {
    if (!has_mips_entry_table_ || pif_projection_width_ <= 0 ||
        pif_entries_per_bucket_ <= 0 || pif_bucket_locks_.empty()) {
      return;
    }

    const int signed_m = 2 * pif_projection_width_;
    const int pair_cols = signed_m * signed_m;
    std::array<float, 32> raw_projection;
    std::array<float, 256> pair01;
    std::array<float, 256> pair23;
    assert(pif_projection_width_ == 8);

    for (int s = 0; s < 4; ++s) {
      for (int j = 0; j < pif_projection_width_; ++j) {
        const float *direction =
            pif_projection_directions_.data() +
            ((size_t)s * pif_projection_width_ + j) * pif_subspace_dim_;
        float value = 0.0f;
        for (int d = 0; d < pif_subspace_dim_; ++d) {
          const int source_dim = pif_permutation_[s * pif_subspace_dim_ + d];
          const float source_value =
              source_dim < static_cast<int>(extended_dim_) ? point[source_dim]
                                                           : 0.0f;
          value += source_value * direction[d];
        }
        raw_projection[(size_t)s * pif_projection_width_ + j] = value;
      }
    }

    for (int a = 0; a < signed_m; ++a) {
      const float va = (a < pif_projection_width_)
                           ? raw_projection[(size_t)a]
                           : -raw_projection[(size_t)a - pif_projection_width_];
      for (int b = 0; b < signed_m; ++b) {
        const float vb = (b < pif_projection_width_)
                             ? raw_projection[(size_t)pif_projection_width_ + b]
                             : -raw_projection[(size_t)pif_projection_width_ +
                                               (b - pif_projection_width_)];
        pair01[(size_t)a * signed_m + b] = va + vb;
      }
    }

    for (int c = 0; c < signed_m; ++c) {
      const float vc =
          (c < pif_projection_width_)
              ? raw_projection[(size_t)2 * pif_projection_width_ + c]
              : -raw_projection[(size_t)2 * pif_projection_width_ +
                                (c - pif_projection_width_)];
      for (int d = 0; d < signed_m; ++d) {
        const float vd =
            (d < pif_projection_width_)
                ? raw_projection[(size_t)3 * pif_projection_width_ + d]
                : -raw_projection[(size_t)3 * pif_projection_width_ +
                                  (d - pif_projection_width_)];
        pair23[(size_t)c * signed_m + d] = vc + vd;
      }
    }

    for (int p01 = 0; p01 < pair_cols; ++p01) {
      const float left = pair01[(size_t)p01];
      for (int p23 = 0; p23 < pair_cols; ++p23) {
        const size_t col_id =
            static_cast<size_t>(p01) * pair_cols + static_cast<size_t>(p23);
        const float score = left + pair23[(size_t)p23];
        if (score <= pif_bucket_tail_scores_[col_id]) {
          continue;
        }

        std::unique_lock<std::mutex> lock(pif_bucket_locks_[col_id]);
        if (score <= pif_bucket_tail_scores_[col_id]) {
          continue;
        }

        PIFEntry *column =
            pif_bucket_top_entries_.data() + col_id * pif_entries_per_bucket_;
        int pos = pif_entries_per_bucket_ - 1;
        while (pos > 0 && score > column[pos - 1].value) {
          column[pos] = column[pos - 1];
          --pos;
        }
        column[pos] = PIFEntry{score, internal_id};
        pif_bucket_tail_scores_[col_id] =
            column[pif_entries_per_bucket_ - 1].value;
      }
    }
  }

  inline bool isCosineMetric() const {
    return options_.metric == PAGMetric::Cosine;
  }

  inline bool isMipsMetric() const {
    return options_.metric == PAGMetric::MaximumInnerProduct;
  }

  inline bool storesNeighborNormInEdges() const {
    return options_.metric == PAGMetric::L2;
  }

  inline float distanceFromStoredInnerProduct(float vector_norm,
                                              float inner_product) const {
    if (isCosineMetric()) {
      return 0.5f - inner_product;
    }
    if (isMipsMetric()) {
      return -vector_norm * inner_product;
    }
    return vector_norm * (vector_norm * 0.5f - inner_product);
  }

  inline float graphCandidateDistance(float stored_distance,
                                      float half_query_squared_norm) const {
    if (isMipsMetric()) {
      return stored_distance;
    }
    return 2.0f * (stored_distance + half_query_squared_norm);
  }

  inline float rawQueryToStoredVectorDistance(float *query_point,
                                              float query_squared_norm,
                                              float vector_norm,
                                              float *vector_record_ptr) const {
    float vector_scale = *vector_record_ptr;
    vector_record_ptr += 1;
    float inner_product = dot_product_avx512_f32_i16(
        query_point, reinterpret_cast<int16_t *>(vector_record_ptr),
        vector_scale);
    if (isMipsMetric()) {
      return -vector_norm * inner_product;
    }
    return query_squared_norm + vector_norm * vector_norm -
           2.0f * vector_norm * inner_product;
  }

  inline const PIFEntry *selectPIFSeeds(float *query_point) const {
    if (!has_mips_entry_table_ || pif_projection_width_ <= 0 ||
        pif_entries_per_bucket_ <= 0) {
      return nullptr;
    }
    const int signed_m = 2 * pif_projection_width_;
    const int pair_cols = signed_m * signed_m;
    int selected[4] = {0, 0, 0, 0};

    for (int s = 0; s < 4; ++s) {
      float best_abs = -1.0f;
      int best_id = 0;
      for (int j = 0; j < pif_projection_width_; ++j) {
        const float *direction =
            pif_projection_directions_.data() +
            ((size_t)s * pif_projection_width_ + j) * pif_subspace_dim_;
        float value = 0.0f;
        for (int d = 0; d < pif_subspace_dim_; ++d) {
          const int source_dim = pif_permutation_[s * pif_subspace_dim_ + d];
          const float query_value = source_dim < static_cast<int>(padded_dim_)
                                        ? query_point[source_dim]
                                        : 0.0f;
          value += query_value * direction[d];
        }
        const float abs_value = value < 0.0f ? -value : value;
        if (abs_value > best_abs) {
          best_abs = abs_value;
          best_id = value < 0.0f ? j + pif_projection_width_ : j;
        }
      }
      selected[s] = best_id;
    }

    const int pair01 = selected[0] * signed_m + selected[1];
    const int pair23 = selected[2] * signed_m + selected[3];
    const size_t col_id =
        static_cast<size_t>(pair01) * pair_cols + static_cast<size_t>(pair23);
    return pif_bucket_top_entries_.data() + col_id * pif_entries_per_bucket_;
  }

  PAGIndexMode indexMode() const { return options_.mode; }

  PAGStorageBackend storageBackend() const { return options_.storage; }

  PAGComputeBackend computeBackend() const { return options_.compute; }

private:
  PAGIndexOptions options_;

  std::string indexPath_;
  std::string infoPath_;

  int *pes_start_pos_ = nullptr;
  bool *base_record_is_full_ = nullptr;
  size_t initial_adjacency_record_bytes_ = 0;
  char *entry_point_vector_records_ = nullptr;
  size_t max_elements_ = 0;
  size_t cur_element_count_ = 0;
  size_t size_data_per_element_ = 0;
  size_t size_links_per_element_ = 0;

  size_t max_entry_points_ = 0;
  size_t max_query_top_k_ = 0;
  int construction_beam_width_ = 0;

  int projection_level_count_ = 0;
  int projection_subspace_dim_ = 0;

  std::vector<float> half_squared_norms_;
  std::vector<std::vector<std::vector<float>>> projection_directions_;

  float *query_projection_directions_ = nullptr;
  size_t vector_dim_ = 0;
  size_t padded_dim_ = 0;
  size_t extended_dim_ = 0;

  size_t vecsize_ = 0;
  size_t loaded_index_bytes_ = 0;
  size_t packed_index_bytes_ = 0;

  size_t segment_size_ = 0;
  size_t target_degree_ = 0;
  size_t projection_width_ = 0;
  size_t max_upper_degree_ = 0;
  size_t max_base_degree_ = 0;
  size_t ef_construction_ = 0;
  int *packed_to_internal_id_ = nullptr;

  double mult_ = 0.0, revSize_ = 0.0;
  int maxlevel_ = 0;

  VisitedListPool *visited_list_pool_ = nullptr;
  std::mutex cur_element_count_guard_;
  std::vector<int> permutation_;
  std::vector<std::mutex> link_list_locks_;
  tableint enterpoint_node_ = 0;

  size_t size_links_level0_ = 0;
  size_t pre_size_links_level0_ = 0;
  size_t offsetData_ = 0;

  size_t vector_record_bytes_ = 0;

  size_t *packed_group_record_bytes_ = nullptr;
  size_t *packed_group_offsets_ = nullptr;
  tableint *edge_group_counts_ = nullptr;
  tableint *edge_group_offsets_ = nullptr;
  std::vector<char *> packed_edge_record_ptrs_;

  std::vector<int> entry_point_ids_;
  bool has_mips_entry_table_ = false;
  int pif_projection_width_ = 0;
  int pif_subspace_dim_ = 0;
  int pif_entries_per_bucket_ = 0;
  std::vector<int> pif_permutation_;
  std::vector<float> pif_projection_directions_;
  std::vector<PIFEntry> pif_bucket_top_entries_;
  std::vector<float> pif_bucket_tail_scores_;
  std::vector<std::mutex> pif_bucket_locks_;
  std::vector<unsigned char> pif_preloaded_internal_ids_;
  std::vector<tableint> online_inserted_entry_ids_;
  std::unordered_map<labeltype, tableint> label_to_internal_id_;
  std::mutex label_metadata_guard_;
  bool track_online_inserted_entry_points_ = false;

  size_t offset0 = 0;
  size_t offset1 = 0;
  size_t offset2 = 0;
  // size_t offset2_new;
  size_t offset3 = 0;
  // size_t offset3_new;

  char **adjacency_records_ = nullptr;
  char *loaded_index_memory_ = nullptr;
  char *vector_records_ = nullptr;
  char **linkLists_ = nullptr;
  std::vector<int> element_levels_;

  std::vector<std::vector<PESCandidate>> pes_candidates_;

  size_t data_size_ = 0;
  size_t label_offset_ = 0;
  DISTFUNC<dist_t> vector_distance_func_ = nullptr;
  DISTFUNC<dist_t> inner_product_func_ = nullptr;
  DISTFUNC<dist_t> projection_distance_func_ = nullptr;

  void *vector_space_param_ = nullptr;
  void *projection_distance_param_ = nullptr;

  std::default_random_engine level_generator_;
  std::default_random_engine update_probability_generator_;

  inline labeltype *getExternalLabelPtr(tableint internal_id) const {
    return (labeltype *)(vector_records_ + internal_id * vector_record_bytes_ +
                         label_offset_);
  }

  inline unsigned char *getProjectionCodePairSlot(char *data, int i,
                                                  int pair_id) const {
    int block_index = i / 16;
    int slot_in_block = i % 16;
    char *data_offset =
        data + block_index * segment_size_ + 16 * pair_id + slot_in_block;
    return reinterpret_cast<unsigned char *>(data_offset);
  }

  inline const unsigned char *getProjectionCodePairSlot(const char *data, int i,
                                                        int pair_id) const {
    int block_index = i / 16;
    int slot_in_block = i % 16;
    const char *data_offset =
        data + block_index * segment_size_ + 16 * pair_id + slot_in_block;
    return reinterpret_cast<const unsigned char *>(data_offset);
  }

  inline unsigned char *getProjectionCodeSlot(char *data, int i,
                                              int j) const { // j = level index
    return getProjectionCodePairSlot(data, i, j >> 1);
  }

  inline const unsigned char *getProjectionCodeSlot(const char *data, int i,
                                                    int j) const {
    return getProjectionCodePairSlot(data, i, j >> 1);
  }

  inline unsigned char getProjectionCodeNibble(const char *data, int i,
                                               int j) const {
    unsigned char packed = *getProjectionCodeSlot(data, i, j);
    return (j & 1) ? get_low4(packed) : get_high4(packed);
  }

  inline void setProjectionCodeNibble(char *data, int i, int j,
                                      unsigned char val) const {
    unsigned char *dst = getProjectionCodeSlot(data, i, j);
    if (j & 1)
      write_low4(dst, val);
    else
      write_high4(dst, val);
  }

  inline void setProjectionCodePair(char *data, int i, int pair_id,
                                    unsigned char even_val,
                                    unsigned char odd_val) const {
    *getProjectionCodePairSlot(data, i, pair_id) =
        static_cast<unsigned char>(((even_val & 0x0F) << 4) | (odd_val & 0x0F));
  }

  inline uint16_t encodeBFloat16(float value) const {
    uint32_t bits = *(uint32_t *)&value;
    uint32_t lsb = (bits >> 16) & 1;
    bits += 0x7FFF + lsb;
    return (uint16_t)(bits >> 16);
  }

  inline uint16_t *getPredictionTerm1(char *data, int i) const {
    int block_index = i / 16;
    int slot_in_block = i % 16;
    char *data_offset = data + block_index * segment_size_ + offset1 +
                        slot_in_block * sizeof(int16_t);
    return (uint16_t *)data_offset;
  }

  inline uint16_t *getPredictionTerm2(char *data, int i) const {
    int block_index = i / 16;
    int slot_in_block = i % 16;
    char *data_offset = data + block_index * segment_size_ + offset2 +
                        slot_in_block * sizeof(int16_t);
    return (uint16_t *)data_offset;
  }

  inline uint16_t *getNeighborNorm(char *data, int i) const {
    int block_index = i / 16;
    int slot_in_block = i % 16;
    char *data_offset = data + block_index * segment_size_ + offset3 +
                        slot_in_block * sizeof(int16_t);
    return (uint16_t *)data_offset;
  }

  inline char *getNormByInternalIdQuery(tableint internal_id) const {
    return (vector_records_ + internal_id * vector_record_bytes_);
  }

  inline labeltype *getExternalLabelQuery(tableint internal_id) const {
    return (labeltype *)(vector_records_ + internal_id * vector_record_bytes_ +
                         label_offset_);
  }

  inline char *getNormByInternalId(tableint internal_id) const {
    return (vector_records_ + internal_id * vector_record_bytes_);
  }

  int getRandomLevel(double reverse_size) {
    std::uniform_real_distribution<double> distribution(0.0, 1.0);
    double r = -log(distribution(level_generator_)) * reverse_size;
    return (int)r;
  }

  std::priority_queue<std::pair<dist_t, tableint>,
                      std::vector<std::pair<dist_t, tableint>>, CompareByFirst>
  searchBaseLayer(tableint ep_id, const void *data_point, int layer) {
    VisitedList *vl = visited_list_pool_->getFreeVisitedList();
    vl_type *visited_array = vl->mass;
    vl_type visited_array_tag = vl->curV;

    std::priority_queue<std::pair<dist_t, tableint>,
                        std::vector<std::pair<dist_t, tableint>>,
                        CompareByFirst>
        top_candidates;
    std::priority_queue<std::pair<dist_t, tableint>,
                        std::vector<std::pair<dist_t, tableint>>,
                        CompareByFirst>
        candidateSet;

    dist_t lowerBound;

    float query_squared_norm =
        inner_product_func_(data_point, data_point, vector_space_param_);
    float *norm_pointer = (float *)getNormByInternalId(ep_id);
    float v_norm = *norm_pointer;
    norm_pointer += 1;
    dist_t dist = rawQueryToStoredVectorDistance(
        (float *)data_point, query_squared_norm, v_norm, norm_pointer);

    top_candidates.emplace(dist, ep_id);
    lowerBound = dist;
    candidateSet.emplace(-dist, ep_id);
    visited_array[ep_id] = visited_array_tag;

    while (!candidateSet.empty()) {
      std::pair<dist_t, tableint> curr_el_pair = candidateSet.top();
      if ((-curr_el_pair.first) > lowerBound &&
          top_candidates.size() == ef_construction_) {
        break;
      }
      candidateSet.pop();

      tableint curNodeNum = curr_el_pair.second;

      std::unique_lock<std::mutex> lock(link_list_locks_[curNodeNum]);

      int *data = (int *)get_linklist(curNodeNum, layer);
      size_t size = getListCount((linklistsizeint *)data);
      int *neighborIds = data + 1;

      int *prefetch_neighbor_ptr = neighborIds;
      int *next_neighbor_ptr = neighborIds + 1;

#ifdef USE_SSE
      _mm_prefetch((char *)(visited_array + *(prefetch_neighbor_ptr)),
                   _MM_HINT_T0);
      _mm_prefetch((char *)(visited_array + *(prefetch_neighbor_ptr) + 64),
                   _MM_HINT_T0);
      _mm_prefetch(getNormByInternalId(*prefetch_neighbor_ptr), _MM_HINT_T0);
      _mm_prefetch(getNormByInternalId(*(next_neighbor_ptr)), _MM_HINT_T0);
#endif

      for (size_t j = 0; j < size; j++) {
        int candidate_id = *(neighborIds + j);
        int *lookahead_neighbor_ptr = neighborIds + j;
#ifdef USE_SSE
        _mm_prefetch((char *)(visited_array + *(lookahead_neighbor_ptr)),
                     _MM_HINT_T0);
        _mm_prefetch(getNormByInternalId(*(lookahead_neighbor_ptr)),
                     _MM_HINT_T0);
#endif
        if (visited_array[candidate_id] == visited_array_tag)
          continue;
        visited_array[candidate_id] = visited_array_tag;

        norm_pointer = (float *)getNormByInternalId(candidate_id);
        v_norm = *norm_pointer;
        norm_pointer += 1;
        float dist1 = rawQueryToStoredVectorDistance(
            (float *)data_point, query_squared_norm, v_norm, norm_pointer);

        if (top_candidates.size() < ef_construction_ || lowerBound > dist1) {
          candidateSet.emplace(-dist1, candidate_id);
#ifdef USE_SSE
          _mm_prefetch(getNormByInternalId(candidateSet.top().second),
                       _MM_HINT_T0);
#endif

          top_candidates.emplace(dist1, candidate_id);

          if (top_candidates.size() > ef_construction_)
            top_candidates.pop();

          if (!top_candidates.empty())
            lowerBound = top_candidates.top().first;
        }
      }
    }
    visited_list_pool_->releaseVisitedList(vl);

    return top_candidates;
  }

  inline __attribute__((always_inline)) __m512
  evalBuildPairLayoutBlock16(const unsigned char *&char_pointer,
                             const __m128i nibble_mask, const float *table,
                             int level_blocks, const __m512 v_center_ip) const {
    if (storesNeighborNormInEdges()) {
      return (level_blocks == 4)
                 ? eval_pairlayout_block16_unrolled_lb4(
                       char_pointer, nibble_mask, table, v_center_ip)
                 : eval_pairlayout_block16(char_pointer, nibble_mask, table,
                                           level_blocks, v_center_ip);
    }

    __m512 sum0 = _mm512_setzero_ps();
    __m512 sum1 = _mm512_setzero_ps();
    __m512 sum2 = _mm512_setzero_ps();
    __m512 sum3 = _mm512_setzero_ps();
    for (int rr2 = 0; rr2 < level_blocks; rr2++) {
      add_8level_scores_64bytes(char_pointer, nibble_mask, table + rr2 * 128,
                                sum0, sum1, sum2, sum3);
      char_pointer += 64;
    }
    __m512 sum =
        _mm512_add_ps(_mm512_add_ps(sum0, sum1), _mm512_add_ps(sum2, sum3));

    const uint16_t *bf16_ptr = reinterpret_cast<const uint16_t *>(char_pointer);
    const __m512i bf16_01 =
        _mm512_loadu_si512(reinterpret_cast<const __m512i *>(bf16_ptr));
    bf16_ptr += 32;
    const __m256i bf16_0 = _mm512_castsi512_si256(bf16_01);
    const __m256i bf16_1 = _mm512_extracti64x4_epi64(bf16_01, 1);

    __m512i v0_i32 = _mm512_cvtepu16_epi32(bf16_0);
    v0_i32 = _mm512_slli_epi32(v0_i32, 16);
    sum = _mm512_fmadd_ps(_mm512_castsi512_ps(v0_i32), v_center_ip, sum);

    __m512i v1_i32 = _mm512_cvtepu16_epi32(bf16_1);
    v1_i32 = _mm512_slli_epi32(v1_i32, 16);
    if (isMipsMetric()) {
      sum = _mm512_mul_ps(sum, _mm512_castsi512_ps(v1_i32));
    } else {
      sum = _mm512_fmadd_ps(sum, _mm512_castsi512_ps(v1_i32),
                            _mm512_set1_ps(0.5f));
    }
    char_pointer = reinterpret_cast<const unsigned char *>(bf16_ptr);
    return sum;
  }

  std::priority_queue<std::pair<dist_t, tableint>,
                      std::vector<std::pair<dist_t, tableint>>, CompareByFirst>
  searchInitialBuildLayer(tableint cur_id, tableint ep_id,
                          const void *query_point) {
    std::priority_queue<std::pair<dist_t, tableint>,
                        std::vector<std::pair<dist_t, tableint>>,
                        CompareByFirst>
        top_candidates;

    float *query_extended_point = new float[extended_dim_];
    permute(query_extended_point, (float *)query_point);

    VisitedList *vl = visited_list_pool_->getFreeVisitedList();
    vl_type *visited_array = vl->mass;
    vl_type visited_array_tag = vl->curV;

    int working_set_capacity = ef_construction_;
    std::vector<Neighbor> working_set(working_set_capacity + 1);
    int prt_passed_ids[max_base_degree_];

    const int projection_table_values =
        projection_level_count_ * 2 * projection_width_;
    const int projection_table_alignment = 64;
    const int level_blocks = projection_level_count_ / 8;
    const __m128i m128_4 = _mm_set1_epi8(0x0F);

    float *table = (float *)std::aligned_alloc(
        projection_table_alignment, projection_table_values * sizeof(float));
    std::memset(table, 0, projection_table_values * sizeof(float));

    for (int j = 0; j < projection_level_count_; j++) {
      float *y = query_extended_point + j * projection_subspace_dim_;
      float *projection_table_block = table + j * (2 * projection_width_);
      for (int i = 0; i < projection_width_; i++) {
        _mm_prefetch(
            reinterpret_cast<const char *>(projection_directions_[j][i].data()),
            _MM_HINT_T0);
        projection_table_block[i] =
            -1 * projection_distance_func_(
                     (void *)y, (void *)projection_directions_[j][i].data(),
                     projection_distance_param_);
        projection_table_block[i + projection_width_] =
            -1 * projection_table_block[i];
      }
    }

    float query_squared_norm =
        inner_product_func_(query_point, query_point, vector_space_param_);
    float half_query_squared_norm = query_squared_norm / 2.0f;
    float *norm_pointer0 = (float *)getNormByInternalId(ep_id);
    float v_norm0 = *norm_pointer0;
    norm_pointer0 += 1;
    float v_scale0 = *norm_pointer0;
    norm_pointer0 += 1;
    float ip0 = dot_product_avx512_f32_i16((float *)query_point,
                                           (int16_t *)norm_pointer0, v_scale0);
    float dist0 = distanceFromStoredInnerProduct(v_norm0, ip0);

    working_set[0] = Neighbor(ep_id, dist0, ip0, true);
    visited_array[ep_id] = visited_array_tag;

    auto filterNeighborsByPRT16 =
        [&](int *neighborIds, int size, float center_ip, float lower_bound,
            float reverse_bound, int *out_ids, bool &reverse_flag) -> int {
      const int round = (size + 15) >> 4;
      const int div = (size % 16 == 0 ? 16 : size % 16);
      const __m512 v_center = _mm512_set1_ps(center_ip);
      const __m512 v_lower = _mm512_set1_ps(lower_bound);
      const __m512 v_reverse = _mm512_set1_ps(reverse_bound);

      int count = 0;
      const int *ids = neighborIds;
      const unsigned char *char_pointer =
          reinterpret_cast<const unsigned char *>(neighborIds +
                                                  max_base_degree_);
      reverse_flag = true;

      for (int rr = 0; rr < round; rr++) {
        const __m512 pred = evalBuildPairLayoutBlock16(
            char_pointer, m128_4, table, level_blocks, v_center);

        const __mmask16 tail_mask = (rr == round - 1 && div != 16)
                                        ? ((__mmask16)((1u << div) - 1u))
                                        : (__mmask16)0xFFFF;

        const __mmask16 lower_mask =
            _mm512_cmp_ps_mask(pred, v_lower, _CMP_LE_OS) & tail_mask;

        if (reverse_flag && ((_mm512_cmp_ps_mask(pred, v_reverse, _CMP_LE_OS) &
                              lower_mask) != 0)) {
          reverse_flag = false;
        }

        const __m512i v_ids =
            _mm512_loadu_si512(reinterpret_cast<const __m512i *>(ids));
        _mm512_mask_compressstoreu_epi32(out_ids + count, lower_mask, v_ids);
        count += _mm_popcnt_u32((unsigned)lower_mask);

        ids += 16;
      }

      return count;
    };

    int k = 0;
    int l_num = 1;

    while (k < l_num) {
      int nk = l_num;

      if (working_set[k].needs_expansion) {
        working_set[k].needs_expansion = false;
        unsigned n = working_set[k].id;
        float current_distance = working_set[k].distance;

        std::unique_lock<std::mutex> lock(link_list_locks_[n]);

        if (l_num < working_set_capacity) {
          int *data = (int *)get_linklist0(n);
          size_t size = getListCount((linklistsizeint *)data);
          int *neighborIds = data + 1;
          int *prefetch_neighbor_ptr = neighborIds;
          int *next_neighbor_ptr = neighborIds + 1;

#ifdef USE_SSE
          _mm_prefetch((char *)(visited_array + *prefetch_neighbor_ptr),
                       _MM_HINT_T0);
          _mm_prefetch((char *)(visited_array + *prefetch_neighbor_ptr + 64),
                       _MM_HINT_T0);
          _mm_prefetch(getNormByInternalId(*prefetch_neighbor_ptr),
                       _MM_HINT_T0);
          _mm_prefetch((char *)(next_neighbor_ptr), _MM_HINT_T0);
#endif

          bool reverse_flag = true;
          for (size_t j = 1; j <= size; j++) {
            int candidate_id = *(neighborIds + j - 1);
            int *lookahead_neighbor_ptr = neighborIds + j;

#ifdef USE_SSE
            _mm_prefetch((char *)(visited_array + *lookahead_neighbor_ptr),
                         _MM_HINT_T0);
            _mm_prefetch(getNormByInternalId(*lookahead_neighbor_ptr),
                         _MM_HINT_T0);
#endif

            if (!(visited_array[candidate_id] == visited_array_tag)) {
              visited_array[candidate_id] = visited_array_tag;

              float *norm_pointer = (float *)getNormByInternalId(candidate_id);
              float v_norm = *norm_pointer;
              norm_pointer += 1;
              float candidate_inner_product = dot_product_avx512_f32_i16(
                  (float *)query_point, (int16_t *)(norm_pointer + 1),
                  *norm_pointer);
              float dist =
                  isMipsMetric()
                      ? -v_norm * candidate_inner_product
                      : v_norm * (v_norm * 0.5f - candidate_inner_product);

              if (reverse_flag && dist < current_distance) {
                reverse_flag = false;
              }

              if (l_num == working_set_capacity &&
                  dist >= working_set[working_set_capacity - 1].distance)
                continue;

              int r;
              if (l_num == working_set_capacity) {
                Neighbor accepted_candidate(candidate_id, dist,
                                            candidate_inner_product, true);
                r = InsertIntoPool(working_set.data(), working_set_capacity,
                                   accepted_candidate);
              } else {
                Neighbor inserted_candidate(candidate_id, dist,
                                            candidate_inner_product, true);
                r = InsertIntoPool(working_set.data(), l_num,
                                   inserted_candidate);
                l_num++;
              }
              if (r < nk) {
                nk = r;
              }
            }
          }

          if (reverse_flag == true) {
            insertPESCandidate(pes_candidates_, n, cur_id,
                               graphCandidateDistance(current_distance,
                                                      half_query_squared_norm));
          }
        } else {
          int *data = (int *)get_linklist0(n);
          size_t size = getListCount((linklistsizeint *)data);
          int *neighborIds = data + 1;

          bool reverse_flag = true;
          const int count = filterNeighborsByPRT16(
              neighborIds, (int)size, working_set[k].inner_product,
              working_set[working_set_capacity - 1].distance, current_distance,
              prt_passed_ids, reverse_flag);

          if (reverse_flag == true) {
            insertPESCandidate(pes_candidates_, n, cur_id,
                               graphCandidateDistance(current_distance,
                                                      half_query_squared_norm));
          }

          size = count;
          neighborIds = prt_passed_ids;

          if (size > 0) {
            int *prefetch_neighbor_ptr = neighborIds;
            int *next_neighbor_ptr = neighborIds + 1;
#ifdef USE_SSE
            _mm_prefetch((char *)(visited_array + *prefetch_neighbor_ptr),
                         _MM_HINT_T0);
            _mm_prefetch((char *)(visited_array + *prefetch_neighbor_ptr + 64),
                         _MM_HINT_T0);
            _mm_prefetch(getNormByInternalId(*prefetch_neighbor_ptr),
                         _MM_HINT_T0);
            _mm_prefetch((char *)(next_neighbor_ptr), _MM_HINT_T0);
#endif
          }

          for (size_t j = 0; j < size; j++) {
            int candidate_id = *(neighborIds + j);
            int *lookahead_neighbor_ptr = neighborIds + j + 1;

#ifdef USE_SSE
            if (j < size - 1) {
              _mm_prefetch((char *)(visited_array + *lookahead_neighbor_ptr),
                           _MM_HINT_T0);
              _mm_prefetch(getNormByInternalId(*lookahead_neighbor_ptr),
                           _MM_HINT_T0);
            }
#endif
            if (!(visited_array[candidate_id] == visited_array_tag)) {
              visited_array[candidate_id] = visited_array_tag;

              float *norm_pointer = (float *)getNormByInternalId(candidate_id);
              float v_norm = *norm_pointer;
              norm_pointer += 1;
              float candidate_inner_product = dot_product_avx512_f32_i16(
                  (float *)query_point, (int16_t *)(norm_pointer + 1),
                  *norm_pointer);
              float dist =
                  isMipsMetric()
                      ? -v_norm * candidate_inner_product
                      : v_norm * (v_norm * 0.5f - candidate_inner_product);

              if (l_num == working_set_capacity &&
                  dist >= working_set[working_set_capacity - 1].distance)
                continue;

              int r;
              if (l_num == working_set_capacity) {
                Neighbor accepted_candidate(candidate_id, dist,
                                            candidate_inner_product, true);
                r = InsertIntoPool(working_set.data(), working_set_capacity,
                                   accepted_candidate);
              } else {
                Neighbor inserted_candidate(candidate_id, dist,
                                            candidate_inner_product, true);
                r = InsertIntoPool(working_set.data(), l_num,
                                   inserted_candidate);
                l_num++;
              }
              if (r < nk) {
                nk = r;
              }
            }
          }
        }
      }
      if (nk <= k)
        k = nk;
      else
        ++k;
    }

    visited_list_pool_->releaseVisitedList(vl);
    for (int i = 0; i < l_num; i++) {
      top_candidates.emplace(graphCandidateDistance(working_set[i].distance,
                                                    half_query_squared_norm),
                             working_set[i].id);
    }
    delete[] query_extended_point;
    std::free(table);
    return top_candidates;
  }

  std::priority_queue<std::pair<dist_t, tableint>,
                      std::vector<std::pair<dist_t, tableint>>, CompareByFirst>
  searchIncrementalBuildLayer(tableint cur_id, const void *query_point) {

    int step = construction_beam_width_;

    std::priority_queue<std::pair<dist_t, tableint>,
                        std::vector<std::pair<dist_t, tableint>>,
                        CompareByFirst>
        top_candidates;

    float half_query_squared_norm =
        inner_product_func_(query_point, query_point, vector_space_param_) / 2;
    float query_norm = sqrt(2 * half_query_squared_norm);

    float normalized_query[padded_dim_];
    float *cur_query_point = (float *)query_point;
    for (int i = 0; i < padded_dim_; i++)
      normalized_query[i] = cur_query_point[i] / query_norm;

    float query_int16_scale = 32767.0f / maxAbsValue(normalized_query);

    int16_t query_int16[padded_dim_];
    for (int i = 0; i < padded_dim_; i++)
      query_int16[i] =
          encodeFloatToInt16(normalized_query[i], query_int16_scale);

    float *query_extended_point = new float[extended_dim_];
    permute(query_extended_point, (float *)query_point);

    VisitedList *vl = visited_list_pool_->getFreeVisitedList();
    vl_type *visited_array = vl->mass;
    vl_type visited_array_tag = vl->curV;

    int max_tfb_rounds = ef_construction_ / step;
    int working_set_capacity = step;
    int ejected_ring_capacity = step;
    int rejected_ring_capacity = step;

    std::vector<NeighborIndex> working_set(working_set_capacity);
    std::vector<NeighborIndex> ejected_ring(ejected_ring_capacity);
    std::vector<NeighborIndex> rejected_ring(rejected_ring_capacity);
    std::vector<NeighborIndex> ejected_ring_buffer(ejected_ring_capacity);

    int ejected_head = 0;
    int rejected_head = 0;
    int ejected_size = 0;
    int rejected_size = 0;

    const int projection_table_values =
        projection_level_count_ * 2 * projection_width_;
    const int projection_table_alignment = 64;
    const int level_blocks = projection_level_count_ / 8;
    const __m128i m128_4 = _mm_set1_epi8(0x0F);

    float *table = (float *)std::aligned_alloc(
        projection_table_alignment, projection_table_values * sizeof(float));
    std::memset(table, 0, projection_table_values * sizeof(float));

    for (int j = 0; j < projection_level_count_; j++) {
      float *y = query_extended_point + j * projection_subspace_dim_;
      float *projection_table_block = table + j * (2 * projection_width_);
      for (int i = 0; i < projection_width_; i++) {
        _mm_prefetch(
            reinterpret_cast<const char *>(projection_directions_[j][i].data()),
            _MM_HINT_T0);
        projection_table_block[i] =
            -1 * projection_distance_func_(
                     (void *)y, (void *)projection_directions_[j][i].data(),
                     projection_distance_param_);
        projection_table_block[i + projection_width_] =
            -1 * projection_table_block[i];
      }
    }

    for (int i = 0; i < step; i++) {
      float *norm_pointer = (float *)getNormByInternalId(i);
      float v_norm = *norm_pointer;
      norm_pointer += 1;
      float v_scale = *norm_pointer;
      norm_pointer += 1;
      float ip = query_norm *
                 dot_product_avx512_int16(query_int16, (int16_t *)norm_pointer,
                                          query_int16_scale, v_scale);
      float dist = distanceFromStoredInnerProduct(v_norm, ip);

      working_set[i] = NeighborIndex(i, dist, ip, true);
      visited_array[i] = visited_array_tag;
    }

    std::sort(working_set.begin(), working_set.begin() + step);
    int working_set_size = step;
    int prt_passed_ids[max_base_degree_];

    auto filterNeighborsByPRT16 =
        [&](int *neighborIds, int size, float center_ip, float lower_bound,
            float reverse_bound, int *out_ids, bool &reverse_flag) -> int {
      const int round = (size + 15) >> 4;
      const int div = (size % 16 == 0 ? 16 : size % 16);
      const __m512 v_center = _mm512_set1_ps(center_ip);
      const __m512 v_lower = _mm512_set1_ps(lower_bound);
      const __m512 v_reverse = _mm512_set1_ps(reverse_bound);

      int count = 0;
      const int *ids = neighborIds;
      const unsigned char *char_pointer =
          reinterpret_cast<const unsigned char *>(neighborIds +
                                                  max_base_degree_);
      reverse_flag = true;

      for (int rr = 0; rr < round; rr++) {
        const __m512 pred = evalBuildPairLayoutBlock16(
            char_pointer, m128_4, table, level_blocks, v_center);

        const __mmask16 tail_mask = (rr == round - 1 && div != 16)
                                        ? ((__mmask16)((1u << div) - 1u))
                                        : (__mmask16)0xFFFF;

        const __mmask16 lower_mask =
            _mm512_cmp_ps_mask(pred, v_lower, _CMP_LE_OS) & tail_mask;

        if (reverse_flag && ((_mm512_cmp_ps_mask(pred, v_reverse, _CMP_LE_OS) &
                              lower_mask) != 0)) {
          reverse_flag = false;
        }

        const __m512i v_ids =
            _mm512_loadu_si512(reinterpret_cast<const __m512i *>(ids));
        _mm512_mask_compressstoreu_epi32(out_ids + count, lower_mask, v_ids);
        count += _mm_popcnt_u32((unsigned)lower_mask);

        ids += 16;
      }

      return count;
    };

    for (int seg = 0; seg < max_tfb_rounds; seg++) {
      if (working_set_size < 1)
        break;

      int k = 0;
      int next_k = 0;
      int next_next_k = 0;

      while (k < working_set_size) {
        working_set[k].needs_expansion = false;
        float current_distance = working_set[k].distance;
        unsigned n = working_set[k].id;

        if (k == next_k) {
          next_k = working_set_size;
          for (int ii = k + 1; ii < working_set_size; ii++) {
            if (working_set[ii].needs_expansion == true) {
              next_k = ii;
              _mm_prefetch((char *)(get_linklist0(working_set[ii].id)),
                           _MM_HINT_T0);
              break;
            }
          }
        }

        std::unique_lock<std::mutex> lock(link_list_locks_[n]);

        int *data = (int *)get_linklist0(n);
        int size = *data;
        int *neighborIds = data + 1;

        bool reverse_flag = true;
        const int count = filterNeighborsByPRT16(
            neighborIds, size, working_set[k].inner_product,
            working_set[working_set_size - 1].distance, current_distance,
            prt_passed_ids, reverse_flag);

        if (count == 0) {
          if (reverse_flag == true) {
            insertPESCandidate(pes_candidates_, n, cur_id,
                               graphCandidateDistance(current_distance,
                                                      half_query_squared_norm));
          }
          k = next_k;
          continue;
        }

        if (reverse_flag == true) {
          insertPESCandidate(pes_candidates_, n, cur_id,
                             graphCandidateDistance(current_distance,
                                                    half_query_squared_norm));
        }

        next_next_k = next_k;
        size = count;
        neighborIds = prt_passed_ids;

        for (size_t j = 0; j < (size_t)size; j++) {
          int candidate_id = neighborIds[j];
          int *prefetch_neighbor_ptr = neighborIds + j + 1;

          if (j + 1 < (size_t)size) {
            _mm_prefetch((char *)(visited_array + *prefetch_neighbor_ptr),
                         _MM_HINT_T0);
            _mm_prefetch(getNormByInternalId(*prefetch_neighbor_ptr),
                         _MM_HINT_T0);
          }

          if (!(visited_array[candidate_id] == visited_array_tag)) {
            visited_array[candidate_id] = visited_array_tag;

            float *norm_pointer = (float *)getNormByInternalId(candidate_id);
            float v_norm = *norm_pointer;
            norm_pointer += 1;
            float v_scale = *norm_pointer;
            norm_pointer += 1;
            float ip = query_norm * dot_product_avx512_int16(
                                        query_int16, (int16_t *)norm_pointer,
                                        query_int16_scale, v_scale);
            float dist = distanceFromStoredInnerProduct(v_norm, ip);

            if (dist >= working_set[working_set_size - 1].distance) {
              NeighborIndex rejected_candidate(candidate_id, dist, ip, true);
              PushTFBRing(rejected_ring, rejected_head, rejected_size,
                          rejected_ring_capacity, rejected_candidate);
              continue;
            }

            if (working_set[working_set_size - 1].needs_expansion == true) {
              PushTFBRing(ejected_ring, ejected_head, ejected_size,
                          ejected_ring_capacity,
                          working_set[working_set_size - 1]);
            } else {
              top_candidates.emplace(
                  graphCandidateDistance(
                      working_set[working_set_size - 1].distance,
                      half_query_squared_norm),
                  working_set[working_set_size - 1].id);
            }

            NeighborIndex accepted_candidate(candidate_id, dist, ip, true);
            int r = InsertIntoPoolIndex(working_set.data(), working_set_size,
                                        accepted_candidate);

            if (r <= next_next_k) {
              if (r <= next_k) {
                next_next_k = next_k + 1;
                next_k = r;
                _mm_prefetch((char *)(get_linklist0(candidate_id)),
                             _MM_HINT_T0);
              } else {
                next_next_k = r;
              }
            }
          }
        }

        k = next_k;
        next_k = next_next_k;
      }

      for (int i = 0; i < working_set_size; i++) {
        top_candidates.emplace(graphCandidateDistance(working_set[i].distance,
                                                      half_query_squared_norm),
                               working_set[i].id);
      }

      RefillTFBWorkingSet(working_set, step, rejected_ring, rejected_head,
                          rejected_size, ejected_ring, ejected_head,
                          ejected_size, ejected_ring_buffer, working_set_size);
    }

    for (int i = 0; i < working_set_size; i++) {
      top_candidates.emplace(graphCandidateDistance(working_set[i].distance,
                                                    half_query_squared_norm),
                             working_set[i].id);
    }

    visited_list_pool_->releaseVisitedList(vl);
    delete[] query_extended_point;
    std::free(table);
    return top_candidates;
  }

#include "pag_search_engine.inc"

  inline void permute(float *dst, float *src) {
    for (int i = 0; i < extended_dim_; i++) {
      int new_pos = permutation_[i];
      if (new_pos < padded_dim_)
        dst[i] = src[new_pos];
      else
        dst[i] = 0.0f;
    }
  }

  void getNeighborsByHeuristic2(
      std::priority_queue<std::pair<dist_t, tableint>,
                          std::vector<std::pair<dist_t, tableint>>,
                          CompareByFirst> &top_candidates,
      const size_t M) {
    if (top_candidates.size() < M) {
      return;
    }

    std::priority_queue<std::pair<dist_t, tableint>> queue_closest;
    std::vector<std::pair<dist_t, tableint>> return_list;
    while (top_candidates.size() > 0) {
      queue_closest.emplace(-top_candidates.top().first,
                            top_candidates.top().second);
      top_candidates.pop();
    }

    while (queue_closest.size()) {
      if (return_list.size() >= M)
        break;
      std::pair<dist_t, tableint> current_pair = queue_closest.top(); // minus
      dist_t dist_to_query = -current_pair.first; // positive
      queue_closest.pop();
      bool good = true;

      for (std::pair<dist_t, tableint> second_pair : return_list) {

        dist_t curdist = computeStoredVectorDistance(second_pair.second,
                                                     current_pair.second);

        if (curdist < dist_to_query) {
          good = false;
          break;
        }
      }
      if (good) {
        return_list.push_back(current_pair); // minus
      }
    }

    for (std::pair<dist_t, tableint> current_pair : return_list) {
      top_candidates.emplace(-current_pair.first, current_pair.second);
    }
  }

  inline linklistsizeint *get_linklist0(tableint internal_id) const {
    return (linklistsizeint *)(adjacency_records_[internal_id]);
  };

  inline linklistsizeint *get_linklist(tableint internal_id, int level) const {
    return (linklistsizeint *)(linkLists_[internal_id] +
                               (level - 1) * size_links_per_element_);
  };

  linklistsizeint *get_linklist_at_level(tableint internal_id,
                                         int level) const {
    return level == 0 ? get_linklist0(internal_id)
                      : get_linklist(internal_id, level);
  };

  tableint mutuallyConnectNewElement(
      tableint new_internal_id,
      std::priority_queue<std::pair<dist_t, tableint>,
                          std::vector<std::pair<dist_t, tableint>>,
                          CompareByFirst> &top_candidates,
      int level) {
    size_t max_degree_at_level = level ? max_upper_degree_ : max_base_degree_;

    getNeighborsByHeuristic2(top_candidates, target_degree_);

    if (top_candidates.size() > target_degree_)
      throw std::runtime_error("Should be not be more than target_degree_ "
                               "candidates returned by the heuristic");

    std::vector<tableint> selected_neighbors;
    selected_neighbors.reserve(target_degree_);
    while (top_candidates.size() > 0) {
      selected_neighbors.push_back(top_candidates.top().second);
      top_candidates.pop();
    }

    tableint next_closest_entry_point = selected_neighbors.back();

    {
      linklistsizeint *new_link_list;
      if (level == 0)
        new_link_list = get_linklist0(new_internal_id);
      else
        new_link_list = get_linklist(new_internal_id, level);

      setListCount(new_link_list, selected_neighbors.size());

      int *new_links = (int *)(new_link_list + 1);

      if (level == 0) {
        int required_edge_blocks = (selected_neighbors.size() + 15) / 16;
        if (required_edge_blocks > 1) {
          size_t new_size = initial_adjacency_record_bytes_ +
                            (required_edge_blocks - 1) * segment_size_;
          adjacency_records_[new_internal_id] =
              (char *)realloc(adjacency_records_[new_internal_id], new_size);
          new_link_list = get_linklist0(new_internal_id);
          new_links = (int *)(new_link_list + 1);
        }
      }

      for (size_t idx = 0; idx < selected_neighbors.size(); idx++) {
        int *link_slot = new_links + idx;

        if (level > element_levels_[selected_neighbors[idx]])
          throw std::runtime_error(
              "Trying to make a link on a non-existent level");

        *link_slot = selected_neighbors[idx];
      }
      if (level == 0) {
        writeAllEdgeProjectionRecords(new_internal_id);
      }
    }

    for (size_t idx = 0; idx < selected_neighbors.size(); idx++) {

      tableint neighbor_id = selected_neighbors[idx];
      std::unique_lock<std::mutex> lock(link_list_locks_[neighbor_id]);

      auto &bucket = pes_candidates_[neighbor_id];
      if (!bucket.empty()) {
#ifdef USE_SSE
        _mm_prefetch((char *)bucket.data(), _MM_HINT_T0);
#endif
      }

      linklistsizeint *neighbor_link_list;
      if (level == 0)
        neighbor_link_list = get_linklist0(neighbor_id);
      else
        neighbor_link_list = get_linklist(neighbor_id, level);

      size_t neighbor_degree = getListCount(neighbor_link_list);

      if (neighbor_degree > max_degree_at_level)
        throw std::runtime_error("Bad value of neighbor degree");
      if (neighbor_id == new_internal_id)
        throw std::runtime_error("Trying to connect an element to itself");
      if (level > element_levels_[neighbor_id])
        throw std::runtime_error(
            "Trying to make a link on a non-existent level");

      int *neighbor_links = (int *)(neighbor_link_list + 1);

      if (neighbor_degree < max_degree_at_level) {

        if (level == 0) {
          const int current_edge_blocks =
              std::max(1, edgeBlockCountForDegree(neighbor_degree));
          const int required_edge_blocks =
              std::max(1, edgeBlockCountForDegree(neighbor_degree + 1));
          if (required_edge_blocks > current_edge_blocks) {
            size_t new_size = initial_adjacency_record_bytes_ +
                              (required_edge_blocks - 1) * segment_size_;
            adjacency_records_[neighbor_id] =
                (char *)realloc(adjacency_records_[neighbor_id], new_size);
            neighbor_link_list = get_linklist0(neighbor_id);
            neighbor_links = (int *)(neighbor_link_list + 1);
          }
        }

        int *link_slot = neighbor_links + neighbor_degree;

        *link_slot = new_internal_id;
        setListCount(neighbor_link_list, neighbor_degree + 1);

        if (level == 0 && neighbor_degree + 1 == max_degree_at_level) {
          base_record_is_full_[neighbor_id] = true;
        }

        if (level == 0) {
          writeEdgeProjectionRecord(neighbor_id, neighbor_degree);
          deletePESCandidate(pes_candidates_, neighbor_id, new_internal_id);
        }
      } else {

        dist_t distance_to_new =
            computeStoredVectorDistance(new_internal_id, neighbor_id);

        std::priority_queue<std::pair<dist_t, tableint>,
                            std::vector<std::pair<dist_t, tableint>>,
                            CompareByFirst>
            candidates;
        candidates.emplace(distance_to_new, new_internal_id);

        for (size_t j = 0; j < neighbor_degree; j++) {
          int *link_slot = neighbor_links + j;

          dist_t candidate_distance =
              computeStoredVectorDistance(*link_slot, neighbor_id);

          candidates.emplace(candidate_distance, *link_slot);
        }

        getNeighborsByHeuristic2(candidates, max_degree_at_level);

        if (level > 0) {
          int retained_degree = 0;
          while (candidates.size() > 0) {
            int *link_slot = neighbor_links + retained_degree;
            *link_slot = candidates.top().second;
            candidates.pop();
            retained_degree++;
          }
          setListCount(neighbor_link_list, retained_degree);
        } else {
          int retained_degree = 0;
          std::vector<tableint> retained_neighbors;
          while (candidates.size() > 0) {
            tableint candidate_id = candidates.top().second;
            retained_neighbors.push_back(candidate_id);
            candidates.pop();
            retained_degree++;
          }
          int write_pos = 0;

          bool keep_new_neighbor = false;
          for (int i = 0; i < retained_degree; i++) {
            if (new_internal_id == retained_neighbors[i]) {
              keep_new_neighbor = true;
              break;
            }
          }

          for (int i = 0; i < neighbor_degree; i++) {
            int existing_neighbor_id = *(neighbor_links + i);
            for (int j = 0; j < retained_degree; j++) {
              if (existing_neighbor_id == retained_neighbors[j]) {
                if (write_pos == i) {
                  write_pos++;
                  break;
                } else {
                  *(neighbor_links + write_pos) = existing_neighbor_id;
                  char *index_link =
                      (char *)(neighbor_links + max_base_degree_);
                  for (int pair_id = 0; pair_id < projection_level_count_ / 2;
                       pair_id++) {
                    *getProjectionCodePairSlot(index_link, write_pos, pair_id) =
                        *getProjectionCodePairSlot(index_link, i, pair_id);
                  }

                  *getPredictionTerm1(index_link, write_pos) =
                      *getPredictionTerm1(index_link, i);
                  *getPredictionTerm2(index_link, write_pos) =
                      *getPredictionTerm2(index_link, i);
                  if (storesNeighborNormInEdges()) {
                    *getNeighborNorm(index_link, write_pos) =
                        *getNeighborNorm(index_link, i);
                  }
                  write_pos++;
                  break;
                }
              }
            }
          }
          if (keep_new_neighbor == true) {
            neighbor_links[write_pos] = new_internal_id;
            writeEdgeProjectionRecord(neighbor_id, write_pos);
            deletePESCandidate(pes_candidates_, neighbor_id, new_internal_id);
          }
          setListCount(neighbor_link_list, retained_degree);
        }
      }
    }

    return next_closest_entry_point;
  }

  std::mutex global;
  size_t ef_ = 0;

  void setEf(size_t ef) { ef_ = ef; }

  void setEfc(int efc) { ef_construction_ = efc; }

  void saveIndexFile() {
    std::ofstream output(infoPath_, std::ios::binary);
    std::streampos position;

    writeBinaryPOD(output, projection_width_);
    writeBinaryPOD(output, projection_level_count_);
    writeBinaryPOD(output, projection_subspace_dim_);

    writeBinaryPOD(output, max_elements_);
    writeBinaryPOD(output, cur_element_count_);
    writeBinaryPOD(output, size_data_per_element_);
    writeBinaryPOD(output, size_links_per_element_);
    writeBinaryPOD(output, label_offset_);
    writeBinaryPOD(output, offsetData_);
    writeBinaryPOD(output, maxlevel_);
    writeBinaryPOD(output, enterpoint_node_);
    writeBinaryPOD(output, max_upper_degree_);

    writeBinaryPOD(output, max_base_degree_);
    writeBinaryPOD(output, target_degree_);
    writeBinaryPOD(output, mult_);
    writeBinaryPOD(output, ef_construction_);
    writeBinaryPOD(output, vector_record_bytes_);

    writeBinaryPOD(output, loaded_index_bytes_);
    writeBinaryPOD(output, packed_index_bytes_);
    writeBinaryPOD(output, vector_dim_);
    writeBinaryPOD(output, padded_dim_);
    writeBinaryPOD(output, extended_dim_);
    writeBinaryPOD(output, max_entry_points_);

    int max_round = max_base_degree_ / 16;
    for (int i = 0; i < max_round; i++) {
      writeBinaryPOD(output, edge_group_offsets_[i]);
      writeBinaryPOD(output, packed_group_offsets_[i]);
      writeBinaryPOD(output, packed_group_record_bytes_[i]);
    }

    for (int i = 0; i < projection_level_count_; ++i) {
      for (int j = 0; j < projection_width_; ++j) {
        output.write(
            reinterpret_cast<char *>(projection_directions_[i][j].data()),
            projection_subspace_dim_ * sizeof(float));
      }
    }

    // output.write(adjacency_records_, loaded_index_bytes_);
    output.write((char *)entry_point_ids_.data(),
                 max_entry_points_ * sizeof(int));
    output.write(entry_point_vector_records_,
                 max_entry_points_ * vector_record_bytes_);

    writeBinaryPOD(output, has_mips_entry_table_);
    if (has_mips_entry_table_) {
      writeBinaryPOD(output, pif_projection_width_);
      writeBinaryPOD(output, pif_subspace_dim_);
      writeBinaryPOD(output, pif_entries_per_bucket_);
      const size_t permutation_size = pif_permutation_.size();
      const size_t direction_size = pif_projection_directions_.size();
      const size_t table_size = pif_bucket_top_entries_.size();
      writeBinaryPOD(output, permutation_size);
      writeBinaryPOD(output, direction_size);
      writeBinaryPOD(output, table_size);
      output.write(reinterpret_cast<const char *>(pif_permutation_.data()),
                   permutation_size * sizeof(int));
      output.write(
          reinterpret_cast<const char *>(pif_projection_directions_.data()),
          direction_size * sizeof(float));
      output.write(
          reinterpret_cast<const char *>(pif_bucket_top_entries_.data()),
          table_size * sizeof(PIFEntry));
    }
    writeBinaryPOD(output, max_query_top_k_);

    output.close();
  }

  void loadIndexFile(const char *path_index, SpaceInterface<dist_t> *s1,
                     SpaceInterface<dist_t> *s2, SpaceInterface<dist_t> *s3,
                     size_t max_elements_i = 0) {
    std::string folderPath(path_index);
    std::string fullPath;
    if (!folderPath.empty() &&
        (folderPath.back() == '/' || folderPath.back() == '\\')) {
      indexPath_ = folderPath + "index.bin";
      infoPath_ = folderPath + "info.bin";
    } else {
      indexPath_ = folderPath + "/index.bin";
      infoPath_ = folderPath + "/info.bin";
    }

    std::ifstream input(infoPath_, std::ios::binary);

    if (!input.is_open())
      throw std::runtime_error("Cannot open file");

    // get file size:
    input.seekg(0, input.end);
    std::streampos total_filesize = input.tellg();
    input.seekg(0, input.beg);

    readBinaryPOD(input, projection_width_);

    readBinaryPOD(input, projection_level_count_);
    readBinaryPOD(input, projection_subspace_dim_);

    readBinaryPOD(input, max_elements_);
    readBinaryPOD(input, cur_element_count_);

    size_t max_elements = max_elements_i;
    if (max_elements < cur_element_count_)
      max_elements = max_elements_;
    max_elements_ = max_elements;
    readBinaryPOD(input, size_data_per_element_);
    readBinaryPOD(input, size_links_per_element_);
    readBinaryPOD(input, label_offset_);
    readBinaryPOD(input, offsetData_);
    readBinaryPOD(input, maxlevel_);
    readBinaryPOD(input, enterpoint_node_);
    readBinaryPOD(input, max_upper_degree_);

    readBinaryPOD(input, max_base_degree_);
    readBinaryPOD(input, target_degree_);
    readBinaryPOD(input, mult_);
    readBinaryPOD(input, ef_construction_);
    readBinaryPOD(input, vector_record_bytes_);

    readBinaryPOD(input, loaded_index_bytes_);
    readBinaryPOD(input, packed_index_bytes_);
    readBinaryPOD(input, vector_dim_);
    readBinaryPOD(input, padded_dim_);
    readBinaryPOD(input, extended_dim_);
    readBinaryPOD(input, max_entry_points_);

    int max_round = max_base_degree_ / 16;
    edge_group_offsets_ = new tableint[max_round]();
    packed_group_offsets_ = new size_t[max_round]();
    packed_group_record_bytes_ = new size_t[max_round]();

    for (int i = 0; i < max_round; i++) {
      readBinaryPOD(input, edge_group_offsets_[i]);
      readBinaryPOD(input, packed_group_offsets_[i]);
      readBinaryPOD(input, packed_group_record_bytes_[i]);
    }

    query_projection_directions_ =
        new float[(size_t)(projection_level_count_)*projection_width_ *
                  projection_subspace_dim_];

    float *cur_pos = query_projection_directions_;
    for (size_t i = 0; i < projection_level_count_; ++i) {
      for (size_t j = 0; j < projection_width_; ++j) {
        input.read(reinterpret_cast<char *>(cur_pos),
                   projection_subspace_dim_ * sizeof(float));
        cur_pos += projection_subspace_dim_;
      }
    }

    data_size_ = s1->get_data_size();
    vector_distance_func_ = s1->get_dist_func();
    inner_product_func_ = s2->get_dist_func();
    vector_space_param_ = s1->get_dist_func_param();

    projection_distance_func_ = s3->get_dist_func();
    projection_distance_param_ = s3->get_dist_func_param();

    std::ifstream input2(indexPath_, std::ios::binary);
    loaded_index_memory_ = (char *)malloc(loaded_index_bytes_);
    if (loaded_index_memory_ == nullptr)
      throw std::runtime_error(
          "Not enough memory: loadIndexFile failed to allocate level0");
    input2.read(loaded_index_memory_, loaded_index_bytes_);
    input2.close();

    vector_records_ = loaded_index_memory_ + packed_index_bytes_;
    packed_edge_record_ptrs_.resize(cur_element_count_);
    for (int group = 0; group < max_round; ++group) {
      const tableint group_begin = edge_group_offsets_[group];
      const tableint group_end =
          (group + 1 < max_round) ? edge_group_offsets_[group + 1]
                                  : static_cast<tableint>(cur_element_count_);
      char *group_base = loaded_index_memory_ + packed_group_offsets_[group];
      for (tableint packed_id = group_begin; packed_id < group_end;
           ++packed_id) {
        packed_edge_record_ptrs_[packed_id] =
            group_base +
            (packed_id - group_begin) * packed_group_record_bytes_[group];
      }
    }

    std::vector<std::mutex>(max_elements).swap(link_list_locks_);

    visited_list_pool_ = new VisitedListPool(1, max_elements);
    ef_ = 10;

    entry_point_ids_.resize(max_entry_points_);
    input.read((char *)entry_point_ids_.data(),
               max_entry_points_ * sizeof(int));

    entry_point_vector_records_ =
        (char *)malloc(max_entry_points_ * vector_record_bytes_);
    input.read(entry_point_vector_records_,
               max_entry_points_ * vector_record_bytes_);

    has_mips_entry_table_ = false;
    auto optional_pos = input.tellg();
    if (optional_pos != std::streampos(-1) && optional_pos < total_filesize) {
      readBinaryPOD(input, has_mips_entry_table_);
      if (has_mips_entry_table_) {
        readBinaryPOD(input, pif_projection_width_);
        readBinaryPOD(input, pif_subspace_dim_);
        readBinaryPOD(input, pif_entries_per_bucket_);
        size_t permutation_size = 0;
        size_t direction_size = 0;
        size_t table_size = 0;
        readBinaryPOD(input, permutation_size);
        readBinaryPOD(input, direction_size);
        readBinaryPOD(input, table_size);
        pif_permutation_.resize(permutation_size);
        pif_projection_directions_.resize(direction_size);
        pif_bucket_top_entries_.resize(table_size);
        input.read(reinterpret_cast<char *>(pif_permutation_.data()),
                   permutation_size * sizeof(int));
        input.read(reinterpret_cast<char *>(pif_projection_directions_.data()),
                   direction_size * sizeof(float));
        input.read(reinterpret_cast<char *>(pif_bucket_top_entries_.data()),
                   table_size * sizeof(PIFEntry));
        const int signed_m = 2 * pif_projection_width_;
        const size_t expected_cols =
            static_cast<size_t>(signed_m) * signed_m * signed_m * signed_m;
        pif_bucket_tail_scores_.assign(expected_cols,
                                       -std::numeric_limits<float>::infinity());
        if (pif_entries_per_bucket_ > 0 &&
            table_size ==
                expected_cols * static_cast<size_t>(pif_entries_per_bucket_)) {
          for (size_t col = 0; col < expected_cols; ++col) {
            pif_bucket_tail_scores_[col] =
                pif_bucket_top_entries_[col * pif_entries_per_bucket_ +
                                        (pif_entries_per_bucket_ - 1)]
                    .value;
          }
        }
        std::vector<std::mutex>(expected_cols).swap(pif_bucket_locks_);
      }
    }
    max_query_top_k_ = max_entry_points_;
    optional_pos = input.tellg();
    if (optional_pos != std::streampos(-1) && optional_pos < total_filesize) {
      readBinaryPOD(input, max_query_top_k_);
    } else if (has_mips_entry_table_ && pif_entries_per_bucket_ > 0) {
      max_query_top_k_ =
          std::min<size_t>(max_entry_points_,
                           static_cast<size_t>(pif_entries_per_bucket_));
    }
    if (max_query_top_k_ == 0) {
      max_query_top_k_ = max_entry_points_;
    }
    input.close();
    return;
  }

  unsigned short int getListCount(linklistsizeint *ptr) const {
    return *((unsigned short int *)ptr);
  }

  void setListCount(linklistsizeint *ptr, unsigned short int size) const {
    *((unsigned short int *)(ptr)) = *((unsigned short int *)&size);
  }

  inline int packedEdgeGroupForDegree(size_t degree) const {
    int group = static_cast<int>(degree / 16);
    if (degree % 16 == 0)
      group--;
    return group;
  }

  inline int edgeBlockCountForDegree(size_t degree) const {
    return static_cast<int>((degree + 15) / 16);
  }

  float orthogonalizeResidualAgainstCenter(float *object_vector,
                                           float *center_vector,
                                           float *residual_vector,
                                           float object_norm,
                                           float center_norm) {
    float *scaled_center = new float[extended_dim_];
    for (int i = 0; i < extended_dim_; i++)
      scaled_center[i] = center_vector[i];

    float object_center_dot_over_object_norm =
        (float)(dot_product_avx512_extended(object_vector, scaled_center) /
                object_norm);
    float cosine_to_center =
        (float)(object_center_dot_over_object_norm / center_norm);
    float center_projection_scale =
        (float)(object_norm * cosine_to_center / center_norm);

    for (int i = 0; i < extended_dim_; i++) {
      scaled_center[i] = scaled_center[i] * center_projection_scale;
    }
    float object_center_dot = object_norm * object_center_dot_over_object_norm;

    for (int i = 0; i < extended_dim_; i++) {
      residual_vector[i] = object_vector[i] - scaled_center[i];
    }

    delete[] scaled_center;
    return object_center_dot;
  }

  inline void restoreVector(float *v, int id) const {
    float *norm_pointer = (float *)getNormByInternalId(id);
    float norm = *norm_pointer;
    norm_pointer += 1;
    float scale = *norm_pointer;
    norm_pointer += 1;
    int16_t *data_int16 = (int16_t *)norm_pointer;

    for (int i = 0; i < padded_dim_; i++)
      v[i] = 1.0f * data_int16[i] * norm / scale;
  }

  inline float computeStoredVectorDistance(int id1, int id2) const {
    float *norm_pointer1 = (float *)getNormByInternalId(id1);
    float norm1 = *norm_pointer1;
    norm_pointer1 += 1;
    float scale1 = *norm_pointer1;
    norm_pointer1 += 1;

    float *norm_pointer2 = (float *)getNormByInternalId(id2);
    float norm2 = *norm_pointer2;
    norm_pointer2 += 1;
    float scale2 = *norm_pointer2;
    norm_pointer2 += 1;

    float dist = dot_product_avx512_int16(
        (int16_t *)norm_pointer1, (int16_t *)norm_pointer2, scale1, scale2);
    if (isMipsMetric()) {
      return -norm1 * norm2 * dist;
    }
    dist = norm1 * norm1 + norm2 * norm2 - 2 * norm1 * norm2 * dist;

    return dist;
  }

  inline float calc_dist_float_int16(float *v, int id) const {
    float query_squared_norm = inner_product_func_(v, v, vector_space_param_);
    float *norm_pointer = (float *)getNormByInternalId(id);
    float norm = *norm_pointer;
    norm_pointer += 1;
    float scale = *norm_pointer;
    norm_pointer += 1;

    float inner_product =
        dot_product_avx512_f32_i16(v, (int16_t *)norm_pointer, scale);
    if (isMipsMetric()) {
      return -norm * inner_product;
    }
    float dist = query_squared_norm + norm * norm - 2 * norm * inner_product;
    return dist;
  }

#include "pag_build_pipeline.inc"
};
} // namespace paglib
