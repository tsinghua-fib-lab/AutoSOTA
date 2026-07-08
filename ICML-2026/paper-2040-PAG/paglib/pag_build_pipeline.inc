void writeEdgeProjectionRecord(int id, int pos) {
  int projection_code;
  float best_projection_abs;
  unsigned char best_projection_code;

  float *edge_residual = new float[extended_dim_];
  float *selected_projection_vector = new float[extended_dim_];
  unsigned char *projection_codes = new unsigned char[projection_level_count_];

  int *data = (int *)get_linklist0(id);

  size_t size = getListCount((linklistsizeint *)data);

  int *neighborIds = data + 1;

  float center_norm = *(float *)getNormByInternalId(id);
  float *center_data = new float[padded_dim_];
  restoreVector(center_data, id);

  float *extended_center = new float[extended_dim_];
  permute(extended_center, center_data);

  int a = *(neighborIds + pos);

  float obj_norm = *(float *)getNormByInternalId(a);
  float *obj_data = new float[padded_dim_];
  restoreVector(obj_data, a);

  float *extended_obj = new float[extended_dim_];
  permute(extended_obj, obj_data);

  for (int l = 0; l < extended_dim_; l++) {
    edge_residual[l] = extended_obj[l] - extended_center[l];
  }

  float object_center_dot = orthogonalizeResidualAgainstCenter(
      extended_obj, extended_center, edge_residual, obj_norm, center_norm);

  float projection_score_sum = 0;

  for (int k = 0; k < projection_level_count_; k++) {
    for (int j = 0; j < projection_width_; j++) {
      float sum = 0;
      for (int l = 0; l < projection_subspace_dim_; l++) {
        sum += edge_residual[k * projection_subspace_dim_ + l] *
               projection_directions_[k][j][l];
      }

      if (sum < 0) {
        sum = -1.0f * sum;
        projection_code = j + 8;
      } else {
        projection_code = j;
      }

      if (j == 0) {
        best_projection_abs = sum;
        best_projection_code = projection_code;
      } else {
        if (sum > best_projection_abs) {
          best_projection_abs = sum;
          best_projection_code = projection_code;
        }
      }
    }
    projection_codes[k] = best_projection_code;
    projection_score_sum += best_projection_abs;
  }

  for (int k = 0; k < projection_level_count_; k++) {
    for (int l = 0; l < projection_subspace_dim_; l++) {
      int s;
      float direction_sign;
      if (projection_codes[k] >= 8) {
        s = projection_codes[k] - 8;
        direction_sign = -1.0f;
      } else {
        s = projection_codes[k];
        direction_sign = 1.0f;
      }
      selected_projection_vector[k * projection_subspace_dim_ + l] =
          projection_directions_[k][s][l] * direction_sign;
    }
  }

  float center_projection_dot =
      dot_product_avx512_extended(selected_projection_vector, extended_center);
  float residual_squared_norm =
      dot_product_avx512_extended(edge_residual, edge_residual);

  float prediction_term2_value = residual_squared_norm / projection_score_sum;
  if (prediction_term2_value < 0 || !std::isfinite(prediction_term2_value)) {
    prediction_term2_value = 0;
  }

  float prediction_term1_value =
      (center_projection_dot -
       (projection_score_sum * object_center_dot / residual_squared_norm)) /
      center_norm;

  if (!std::isfinite(prediction_term1_value) ||
      prediction_term1_value > 32767 || prediction_term1_value < -32767) {
    prediction_term1_value = 0;
  }

  char *index_pos = (char *)(neighborIds + max_base_degree_);

  for (int pair_id = 0; pair_id < projection_level_count_ / 2; pair_id++) {
    setProjectionCodePair(index_pos, pos, pair_id,
                          projection_codes[2 * pair_id],
                          projection_codes[2 * pair_id + 1]);
  }

  *getPredictionTerm1(index_pos, pos) = encodeBFloat16(prediction_term1_value);

  *getPredictionTerm2(index_pos, pos) = encodeBFloat16(prediction_term2_value);

  if (storesNeighborNormInEdges()) {
    float neighbor_half_squared_norm =
        half_squared_norms_[*getExternalLabelPtr(neighborIds[pos])];
    *getNeighborNorm(index_pos, pos) =
        encodeBFloat16(neighbor_half_squared_norm);
  }

  delete[] extended_center;
  delete[] extended_obj;
  delete[] projection_codes;
  delete[] edge_residual;
  delete[] selected_projection_vector;

  delete[] obj_data;
  delete[] center_data;
}

void writeAllEdgeProjectionRecords(int id) {
  int projection_code;
  float best_projection_abs;
  unsigned char best_projection_code;

  float *edge_residual = new float[extended_dim_];
  float *selected_projection_vector = new float[extended_dim_];
  unsigned char *projection_codes = new unsigned char[projection_level_count_];

  int *data;
  data = (int *)get_linklist0(id);

  size_t size = getListCount((linklistsizeint *)data);

  int *neighborIds = data + 1;

  float center_norm = *(float *)getNormByInternalId(id);
  float *center_data = new float[padded_dim_];
  restoreVector(center_data, id);

  float *extended_center = new float[extended_dim_];
  permute(extended_center, center_data);

  float *extended_obj = new float[extended_dim_];
  float *obj_data = new float[padded_dim_];
  for (int i = 0; i < size; i++) {
    int a = *(neighborIds + i);

    float obj_norm = *(float *)getNormByInternalId(a);
    restoreVector(obj_data, a);

    permute(extended_obj, obj_data);

    for (int l = 0; l < extended_dim_; l++) {
      edge_residual[l] = extended_obj[l] - extended_center[l];
    }

    float object_center_dot = orthogonalizeResidualAgainstCenter(
        extended_obj, extended_center, edge_residual, obj_norm, center_norm);

    float projection_score_sum = 0;

    for (int k = 0; k < projection_level_count_; k++) {

      for (int j = 0; j < projection_width_; j++) {
        float sum = 0;
        for (int l = 0; l < projection_subspace_dim_; l++) {
          sum += edge_residual[k * projection_subspace_dim_ + l] *
                 projection_directions_[k][j][l];
        }

        if (sum < 0) {
          sum = -1.0f * sum;
          projection_code = j + 8;
        } else {
          projection_code = j;
        }

        if (j == 0) {
          best_projection_abs = sum;
          best_projection_code = projection_code;
        } else {
          if (sum > best_projection_abs) {
            best_projection_abs = sum;
            best_projection_code = projection_code;
          }
        }
      }
      projection_codes[k] = best_projection_code;
      projection_score_sum += best_projection_abs;
    }

    for (int k = 0; k < projection_level_count_; k++) {
      for (int l = 0; l < projection_subspace_dim_; l++) {
        int s;
        float direction_sign;
        if (projection_codes[k] >= 8) {
          s = projection_codes[k] - 8;
          direction_sign = -1.0f;
        } else {
          s = projection_codes[k];
          direction_sign = 1.0f;
        }

        selected_projection_vector[k * projection_subspace_dim_ + l] =
            projection_directions_[k][s][l] * direction_sign;
      }
    }

    float center_projection_dot = dot_product_avx512_extended(
        selected_projection_vector, extended_center);
    float residual_squared_norm =
        dot_product_avx512_extended(edge_residual, edge_residual);

    float prediction_term2_value = residual_squared_norm / projection_score_sum;
    if (prediction_term2_value < 0 || !std::isfinite(prediction_term2_value)) {
      prediction_term2_value = 0;
    }

    float prediction_term1_value =
        (center_projection_dot -
         (projection_score_sum * object_center_dot / residual_squared_norm)) /
        center_norm;

    if (!std::isfinite(prediction_term1_value) ||
        prediction_term1_value > 32767 || prediction_term1_value < -32767) {
      prediction_term1_value = 0;
    }

    char *index_pos = (char *)(neighborIds + max_base_degree_);

    for (int pair_id = 0; pair_id < projection_level_count_ / 2; pair_id++) {
      setProjectionCodePair(index_pos, i, pair_id,
                            projection_codes[2 * pair_id],
                            projection_codes[2 * pair_id + 1]);
    }

    *getPredictionTerm1(index_pos, i) = encodeBFloat16(prediction_term1_value);

    *getPredictionTerm2(index_pos, i) = encodeBFloat16(prediction_term2_value);

    if (storesNeighborNormInEdges()) {
      float neighbor_half_squared_norm =
          half_squared_norms_[*getExternalLabelPtr(neighborIds[i])];
      *getNeighborNorm(index_pos, i) =
          encodeBFloat16(neighbor_half_squared_norm);
    }
    //*getPredictionTerm1(index_pos, i) = prediction_term1_value;
    //*getPredictionTerm2(index_pos, i) = prediction_term2_value;
    //*getNeighborNorm(index_pos, i) =
    // half_squared_norms_[*getExternalLabelPtr(neighborIds[i])];
  }

  delete[] extended_center;
  delete[] extended_obj;
  delete[] projection_codes;
  delete[] edge_residual;
  delete[] selected_projection_vector;

  delete[] obj_data;
  delete[] center_data;
}

void insertPointImpl(const void *data_point, labeltype label, float *norm) {

  tableint new_internal_id = 0;
  {
    std::unique_lock<std::mutex> templock_curr(cur_element_count_guard_);

    if (cur_element_count_ >= max_elements_) {
      throw std::runtime_error(
          "The number of elements exceeds the specified limit");
    };

    new_internal_id = cur_element_count_;
    cur_element_count_++;
  }

  std::unique_lock<std::mutex> lock_el(link_list_locks_[new_internal_id]);
  //     templock.unlock();

  {
    std::unique_lock<std::mutex> label_lock(label_metadata_guard_);
    if (static_cast<size_t>(label) >= half_squared_norms_.size()) {
      half_squared_norms_.resize(static_cast<size_t>(label) + 1, 0.0f);
    }
    half_squared_norms_[static_cast<size_t>(label)] =
        storesNeighborNormInEdges() ? 0.5f * (*norm) * (*norm) : 0.0f;
  }

  memset(adjacency_records_[new_internal_id], 0,
         initial_adjacency_record_bytes_);

  float vec_normalized[padded_dim_];
  float *cur_vec = (float *)data_point;
  for (int j = 0; j < padded_dim_; j++)
    vec_normalized[j] = cur_vec[j] / (*norm);

  char *norm_pointer = getNormByInternalId(new_internal_id);
  memcpy(norm_pointer, norm, sizeof(float));
  norm_pointer += 4;
  float *scale_pointer = (float *)norm_pointer;
  *scale_pointer = 32767.0f / maxAbsValue(vec_normalized);
  norm_pointer += 4;
  int16_t *dst_vec = (int16_t *)norm_pointer;

  for (int j = 0; j < padded_dim_; j++)
    dst_vec[j] = encodeFloatToInt16(vec_normalized[j], *scale_pointer);

  memcpy(dst_vec + padded_dim_, &label, sizeof(labeltype));
  //         throw std::runtime_error("Not enough memory: insertPointImpl failed
  //         to allocate linklist");
  // }

  if (new_internal_id > 0) {
    std::priority_queue<std::pair<dist_t, tableint>,
                        std::vector<std::pair<dist_t, tableint>>,
                        CompareByFirst>
        top_candidates;
    if (new_internal_id < construction_beam_width_)
      top_candidates = searchInitialBuildLayer(new_internal_id, 0, data_point);
    else {
      top_candidates = searchIncrementalBuildLayer(new_internal_id, data_point);
    }

    mutuallyConnectNewElement(new_internal_id, top_candidates, 0);
  }
  if (isMipsMetric()) {
    const bool already_in_pif =
        static_cast<size_t>(new_internal_id) < pif_preloaded_internal_ids_.size() &&
        pif_preloaded_internal_ids_[static_cast<size_t>(new_internal_id)] != 0;
    if (!already_in_pif) {
      updatePIFTableForInsertedPoint(static_cast<const float *>(data_point),
                                     static_cast<int>(new_internal_id));
    }
  }
  if (track_online_inserted_entry_points_) {
    std::unique_lock<std::mutex> label_lock(label_metadata_guard_);
    label_to_internal_id_[label] = new_internal_id;
    online_inserted_entry_ids_.push_back(new_internal_id);
  }
  // else {
  //}
  // }
};

void completePESForNodeImpl(int center_id) {
  int *center_link_record = (int *)get_linklist0(center_id);
  size_t center_degree = getListCount((linklistsizeint *)center_link_record);
  pes_start_pos_[center_id] = (int)center_degree;

  int *center_neighbors = center_link_record + 1;
  const auto &candidate_bucket = pes_candidates_[center_id];

  if (candidate_bucket.empty() || center_degree == max_base_degree_)
    return;

  for (const auto &pending_edge : candidate_bucket) {
    int candidate_id = static_cast<int>(pending_edge.id);

    bool exists = false;
    for (int j = 0; j < center_degree; ++j) {
      if (center_neighbors[j] == candidate_id) {
        exists = true;
        break;
      }
    }

    if (!exists) {

      float candidate_distance_to_center =
          computeStoredVectorDistance(center_id, candidate_id);

      bool covered_by_existing_edge = false;

      _mm_prefetch((char *)(get_linklist0(center_neighbors[0])), _MM_HINT_T0);

      for (int l = 0; l < center_degree; l++) {
        if (l < center_degree - 1) {
          _mm_prefetch((char *)(getNormByInternalId(center_neighbors[l + 1])),
                       _MM_HINT_T0);
          _mm_prefetch((char *)(get_linklist0(center_neighbors[l + 1])),
                       _MM_HINT_T0);
        }

        float candidate_distance_to_neighbor =
            computeStoredVectorDistance(candidate_id, center_neighbors[l]);

        float center_distance_to_neighbor =
            computeStoredVectorDistance(center_id, center_neighbors[l]);
        if (candidate_distance_to_neighbor <= candidate_distance_to_center &&
            center_distance_to_neighbor <= candidate_distance_to_center) {
          int *neighbor_link_record = (int *)get_linklist0(center_neighbors[l]);
          size_t neighbor_degree =
              getListCount((linklistsizeint *)neighbor_link_record);
          int *second_hop_neighbors = neighbor_link_record + 1;

          for (int ll = 0; ll < neighbor_degree; ll++) {
            if (second_hop_neighbors[ll] == candidate_id) {
              covered_by_existing_edge = true;
              break;
            }
          }
        }
      }

      if (covered_by_existing_edge == false) {
        center_neighbors[center_degree++] = candidate_id;
      }
    }

    if (center_degree == max_base_degree_) {
      break;
    }
  }

  for (const auto &pending_edge : candidate_bucket) {
    if (center_degree % 16 == 0)
      break;
    int candidate_id = static_cast<int>(pending_edge.id);
    bool exists = false;
    for (int j = 0; j < center_degree; ++j) {
      if (center_neighbors[j] == candidate_id) {
        exists = true;
        break;
      }
    }

    if (!exists) {
      center_neighbors[center_degree++] = candidate_id;
    }
  }
  setListCount((linklistsizeint *)center_link_record,
               (unsigned short int)center_degree);
}

void reservePESProjectionStorageForNodeImpl(int center_id) {
  int *center_link_record = (int *)get_linklist0(center_id);
  size_t center_degree = getListCount((linklistsizeint *)center_link_record);

  if (edgeBlockCountForDegree(pes_start_pos_[center_id]) !=
      edgeBlockCountForDegree(center_degree)) {
    int required_edge_blocks = edgeBlockCountForDegree(center_degree);
    size_t new_size = initial_adjacency_record_bytes_ +
                      (required_edge_blocks - 1) * segment_size_;
    adjacency_records_[center_id] =
        (char *)realloc(adjacency_records_[center_id], new_size);
  }
}

void encodePESProjectionsForNodeImpl(int center_id) {
  int *center_link_record = (int *)get_linklist0(center_id);
  size_t center_degree = getListCount((linklistsizeint *)center_link_record);
  for (int j = pes_start_pos_[center_id]; j < center_degree; j++) {
    writeEdgeProjectionRecord(center_id, j);
  }
}

void packSearchLayoutImpl(int vecsize) {
  int packed_group_count = max_base_degree_ / 16;
  edge_group_counts_ = new tableint[packed_group_count]();

  tableint *position_in_group = new tableint[vecsize];
  tableint *packed_id_by_internal_id = new tableint[vecsize];

  int *group_by_internal_id = new int[vecsize];

  for (int i = 0; i < vecsize; i++) {
    tableint *data = get_linklist0(i);
    tableint size = *data;
    data += 1;

    int group_id = packedEdgeGroupForDegree(size);

    group_by_internal_id[i] = group_id; // from 0
    position_in_group[i] = edge_group_counts_[group_id];
    edge_group_counts_[group_id]++;
  }

  edge_group_offsets_ = new tableint[packed_group_count]();
  for (int i = 1; i < packed_group_count; i++) {
    edge_group_offsets_[i] =
        edge_group_offsets_[i - 1] + edge_group_counts_[i - 1];
  }

  packed_to_internal_id_ = new int[vecsize];
  for (int i = 0; i < vecsize; i++) {
    packed_id_by_internal_id[i] =
        position_in_group[i] + edge_group_offsets_[group_by_internal_id[i]];
    packed_to_internal_id_[packed_id_by_internal_id[i]] = i;
  }

  packed_group_record_bytes_ = new size_t[packed_group_count]();
  for (int i = 0; i < packed_group_count; i++) {
    const int prediction_term_count = options_.metric == PAGMetric::L2 ? 3 : 2;
    packed_group_record_bytes_[i] =
        sizeof(linklistsizeint) +
        (i + 1) * 16 *
            (prediction_term_count * sizeof(int16_t) + sizeof(tableint) +
             projection_level_count_ / 2);
  }
  packed_index_bytes_ = 0;
  for (int i = 0; i < packed_group_count; i++) {
    packed_index_bytes_ +=
        packed_group_record_bytes_[i] * edge_group_counts_[i];
  }

  loaded_index_bytes_ = packed_index_bytes_ + (vecsize_ * vector_record_bytes_);

#pragma omp parallel for schedule(dynamic)
  for (size_t i = 0; i < vecsize; i++) {
    tableint *data = get_linklist0(i);
    tableint size = *data;
    data += 1;

    for (int j = 0; j < size; j++) {
      tableint obj_id = *(data + j);
      *(data + j) = packed_id_by_internal_id[obj_id];
    }
  }

  packed_group_offsets_ = new size_t[packed_group_count]();
  for (int i = 1; i < packed_group_count; i++) {
    packed_group_offsets_[i] =
        packed_group_offsets_[i - 1] +
        (packed_group_record_bytes_[i - 1] * edge_group_counts_[i - 1]);
  }

  std::ofstream output(indexPath_, std::ios::binary);
  for (int i = 0; i < vecsize; i++) {
    int cur_id = packed_to_internal_id_[i];
    int group_id = group_by_internal_id[cur_id]; // 0
    char *index_src = (char *)get_linklist0(cur_id);
    tableint size = *get_linklist0(cur_id);
    int edge_block_count = edgeBlockCountForDegree(size);

    size_t packed_group_bytes = 4 + 4 * 16 * (group_id + 1);
    output.write(index_src, packed_group_bytes);
    index_src += 4 * (max_base_degree_ + 1);
    char *projection_record_src = index_src;
    const int prediction_term_count = options_.metric == PAGMetric::L2 ? 3 : 2;
    output.write(projection_record_src,
                 edge_block_count *
                     (8 * projection_level_count_ +
                      16 * (prediction_term_count * sizeof(int16_t))));
  }

  for (int i = 0; i < vecsize; i++) {
    int cur_id = packed_to_internal_id_[i];
    char *vec_src = (char *)getNormByInternalId(cur_id);
    output.write(vec_src, vector_record_bytes_);
  }

  output.close();

  for (int i = 0; i < max_entry_points_; i++) {
    entry_point_ids_[i] = packed_id_by_internal_id[entry_point_ids_[i]];
  }
  if (has_mips_entry_table_) {
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < pif_bucket_top_entries_.size(); ++i) {
      pif_bucket_top_entries_[i].index =
          packed_id_by_internal_id[pif_bucket_top_entries_[i].index];
    }
  }

  delete[] position_in_group;
  delete[] packed_id_by_internal_id;
  delete[] group_by_internal_id;
  delete[] packed_to_internal_id_;
  packed_to_internal_id_ = nullptr;
  delete[] base_record_is_full_;
  base_record_is_full_ = nullptr;

  label_offset_ = data_size_ + sizeof(float);
}

void selectEntryPointSeedsImpl(float *data_point) {
  const size_t indexed_count = cur_element_count_;
  if (indexed_count == 0) {
    throw std::runtime_error("Cannot select entry points for an empty index");
  }

  entry_point_ids_.clear();
  if (entry_point_vector_records_ != nullptr) {
    free(entry_point_vector_records_);
    entry_point_vector_records_ = nullptr;
  }

  int *fallback_seed_candidates = new int[max_entry_points_];
  entry_point_vector_records_ =
      (char *)malloc(max_entry_points_ * vector_record_bytes_);

  char *dst_record = entry_point_vector_records_;

  std::vector<unsigned int> sorted_candidate_ids(indexed_count);
  std::vector<PESCandidate> center_distances(indexed_count);
#pragma omp parallel for
  for (size_t j = 0; j < indexed_count; j++) {
    float candidate_distance;
    if (isMipsMetric()) {
      candidate_distance = -(*(float *)getNormByInternalId((int)j));
    } else {
      float *cur_vec = new float[padded_dim_];
      restoreVector(cur_vec, j);
      candidate_distance =
          vector_distance_func_(cur_vec, data_point, vector_space_param_);
      delete[] cur_vec;
    }
    center_distances[j] =
        PESCandidate{static_cast<unsigned int>(j), candidate_distance};
  }

  std::sort(center_distances.begin(), center_distances.end(),
            [](const PESCandidate &a, const PESCandidate &b) {
              return a.distance < b.distance;
            });

  for (size_t j = 0; j < indexed_count; j++) {
    sorted_candidate_ids[j] = center_distances[j].id;
  }

  int selected_seed_count = 0;
  int fallback_seed_count = 0;

  std::vector<std::vector<float>> selected_seed_directions;
  selected_seed_directions.reserve(max_entry_points_);

  for (const auto &seed_candidate_id : sorted_candidate_ids) {
    size_t candidate_id = seed_candidate_id;

    float *candidate_vector = new float[padded_dim_];
    restoreVector(candidate_vector, (int)candidate_id);

    std::vector<float> direction_from_center(padded_dim_);
    for (int d = 0; d < padded_dim_; d++)
      direction_from_center[d] = candidate_vector[d] - data_point[d];

    float direction_norm_squared =
        inner_product_func_(direction_from_center.data(),
                            direction_from_center.data(), vector_space_param_);
    float direction_norm = std::sqrt(direction_norm_squared);
    if (direction_norm < 1e-8f)
      continue;

    for (int d = 0; d < padded_dim_; d++)
      direction_from_center[d] /= direction_norm;

    bool sufficiently_separated = true;
    for (const auto &selected_direction : selected_seed_directions) {
      float dot = inner_product_func_(selected_direction.data(),
                                      direction_from_center.data(),
                                      vector_space_param_);
      if (dot >= 0.5f) { // ?60?
        sufficiently_separated = false;
        break;
      }
    }

    if (sufficiently_separated) {
      entry_point_ids_.emplace_back(candidate_id);
      char *src_record = getNormByInternalId(candidate_id);
      memcpy(dst_record, src_record, vector_record_bytes_);
      dst_record += vector_record_bytes_;
      selected_seed_count++;

      if (selected_seed_count >= max_entry_points_)
        break;

      selected_seed_directions.emplace_back(std::move(direction_from_center));
    } else {
      if (fallback_seed_count < max_entry_points_) {
        fallback_seed_candidates[fallback_seed_count] = candidate_id;
        fallback_seed_count++;
      }
    }
    delete[] candidate_vector;
  }

  if (selected_seed_count < max_entry_points_) {
    int fallback_id = 0;
    while (selected_seed_count < max_entry_points_ &&
           fallback_id < fallback_seed_count) {
      const int candidate_id = fallback_seed_candidates[fallback_id++];
      entry_point_ids_.emplace_back(candidate_id);
      char *src_record = getNormByInternalId(candidate_id);
      memcpy(dst_record, src_record, vector_record_bytes_);
      dst_record += vector_record_bytes_;
      selected_seed_count++;
    }
    size_t repeat_id = 0;
    while (selected_seed_count < max_entry_points_) {
      const int candidate_id =
          entry_point_ids_.empty() ? 0 : entry_point_ids_[repeat_id % entry_point_ids_.size()];
      entry_point_ids_.emplace_back(candidate_id);
      char *src_record = getNormByInternalId(candidate_id);
      memcpy(dst_record, src_record, vector_record_bytes_);
      dst_record += vector_record_bytes_;
      selected_seed_count++;
      repeat_id++;
    }
  }

  delete[] fallback_seed_candidates;
};

void initializeFallbackEntryPointSeedsImpl() {
  entry_point_ids_.clear();
  entry_point_vector_records_ =
      (char *)malloc(max_entry_points_ * vector_record_bytes_);
  if (entry_point_vector_records_ == nullptr) {
    throw std::runtime_error("initializeFallbackEntryPointSeedsImpl failed to "
                             "allocate seed records");
  }

  char *dst = entry_point_vector_records_;
  const int seed_count = std::min<int>(max_entry_points_, cur_element_count_);
  for (int i = 0; i < seed_count; ++i) {
    entry_point_ids_.emplace_back(i);
    std::memcpy(dst, getNormByInternalId(i), vector_record_bytes_);
    dst += vector_record_bytes_;
  }
  for (int i = seed_count; i < static_cast<int>(max_entry_points_); ++i) {
    entry_point_ids_.emplace_back(0);
    std::memcpy(dst, getNormByInternalId(0), vector_record_bytes_);
    dst += vector_record_bytes_;
  }
}
