template <int FIXED_LB>
inline __attribute__((always_inline)) void accumulate_pairlayout_block16_fixed(
    const unsigned char *&char_pointer, const __m128i nibble_mask,
    const float *table, __m512 &sum0, __m512 &sum1, __m512 &sum2,
    __m512 &sum3) const {
  if constexpr (FIXED_LB == 0) {
    return;
  } else if constexpr (FIXED_LB >= 4) {
    add_8level_scores_64bytes(char_pointer + 0, nibble_mask, table + 0, sum0,
                              sum1, sum2, sum3);
    add_8level_scores_64bytes(char_pointer + 64, nibble_mask, table + 128, sum0,
                              sum1, sum2, sum3);
    add_8level_scores_64bytes(char_pointer + 128, nibble_mask, table + 256,
                              sum0, sum1, sum2, sum3);
    add_8level_scores_64bytes(char_pointer + 192, nibble_mask, table + 384,
                              sum0, sum1, sum2, sum3);
    char_pointer += 256;
    accumulate_pairlayout_block16_fixed<FIXED_LB - 4>(
        char_pointer, nibble_mask, table + 512, sum0, sum1, sum2, sum3);
  } else if constexpr (FIXED_LB >= 2) {
    add_8level_scores_64bytes(char_pointer + 0, nibble_mask, table + 0, sum0,
                              sum1, sum2, sum3);
    add_8level_scores_64bytes(char_pointer + 64, nibble_mask, table + 128, sum0,
                              sum1, sum2, sum3);
    char_pointer += 128;
    accumulate_pairlayout_block16_fixed<FIXED_LB - 2>(
        char_pointer, nibble_mask, table + 256, sum0, sum1, sum2, sum3);
  } else {
    add_8level_scores_64bytes(char_pointer + 0, nibble_mask, table + 0, sum0,
                              sum1, sum2, sum3);
    char_pointer += 64;
    accumulate_pairlayout_block16_fixed<0>(char_pointer, nibble_mask,
                                           table + 128, sum0, sum1, sum2, sum3);
  }
}

template <bool MIPS = false>
static inline __attribute__((always_inline)) __m512
finalize_pairlayout_block16_two_term(const unsigned char *&char_pointer,
                                     __m512 sum, const __m512 v_center_ip) {
  const uint16_t *bf16_ptr = reinterpret_cast<const uint16_t *>(char_pointer);

  const __m512i bf16_01 =
      _mm512_loadu_si512(reinterpret_cast<const __m512i *>(bf16_ptr));
  bf16_ptr += 32;

  const __m256i bf16_0 = _mm512_castsi512_si256(bf16_01);
  const __m256i bf16_1 = _mm512_extracti64x4_epi64(bf16_01, 1);

  __m512i v0_i32 = _mm512_cvtepu16_epi32(bf16_0);
  v0_i32 = _mm512_slli_epi32(v0_i32, 16);
  const __m512 v_scale = _mm512_castsi512_ps(v0_i32);
  sum = _mm512_fmadd_ps(v_scale, v_center_ip, sum);

  __m512i v1_i32 = _mm512_cvtepu16_epi32(bf16_1);
  v1_i32 = _mm512_slli_epi32(v1_i32, 16);
  const __m512 v_mul = _mm512_castsi512_ps(v1_i32);

  if constexpr (MIPS) {
    sum = _mm512_mul_ps(sum, v_mul);
  } else {
    sum = _mm512_fmadd_ps(sum, v_mul, _mm512_set1_ps(0.5f));
  }
  char_pointer = reinterpret_cast<const unsigned char *>(bf16_ptr);
  return sum;
}

template <int FIXED_LB, bool COSINE = false, bool MIPS = false>
inline __attribute__((always_inline)) __m512 eval_pairlayout_block16_fixed_impl(
    const unsigned char *&char_pointer, const __m128i nibble_mask,
    const float *table, const __m512 v_center_inner_product) const {
  if constexpr (FIXED_LB == 1) {
    const __m512 lut00 = _mm512_load_ps(table + 0);
    const __m512 lut01 = _mm512_load_ps(table + 16);
    const __m512 lut02 = _mm512_load_ps(table + 32);
    const __m512 lut03 = _mm512_load_ps(table + 48);
    const __m512 lut04 = _mm512_load_ps(table + 64);
    const __m512 lut05 = _mm512_load_ps(table + 80);
    const __m512 lut06 = _mm512_load_ps(table + 96);
    const __m512 lut07 = _mm512_load_ps(table + 112);
    if constexpr (!COSINE && !MIPS) {
      return eval_pairlayout_block16_hoisted_lb1(
          char_pointer, nibble_mask, lut00, lut01, lut02, lut03, lut04, lut05,
          lut06, lut07, v_center_inner_product);
    } else {
      __m512 sum0 = _mm512_setzero_ps();
      __m512 sum1 = _mm512_setzero_ps();
      __m512 sum2 = _mm512_setzero_ps();
      __m512 sum3 = _mm512_setzero_ps();
      add_level_pair_scores_16bytes_hoisted(
          _mm_loadu_si128(reinterpret_cast<const __m128i *>(char_pointer + 0)),
          nibble_mask, lut00, lut01, sum0);
      add_level_pair_scores_16bytes_hoisted(
          _mm_loadu_si128(reinterpret_cast<const __m128i *>(char_pointer + 16)),
          nibble_mask, lut02, lut03, sum1);
      add_level_pair_scores_16bytes_hoisted(
          _mm_loadu_si128(reinterpret_cast<const __m128i *>(char_pointer + 32)),
          nibble_mask, lut04, lut05, sum2);
      add_level_pair_scores_16bytes_hoisted(
          _mm_loadu_si128(reinterpret_cast<const __m128i *>(char_pointer + 48)),
          nibble_mask, lut06, lut07, sum3);
      char_pointer += 64;
      const __m512 sum =
          _mm512_add_ps(_mm512_add_ps(sum0, sum1), _mm512_add_ps(sum2, sum3));
      return finalize_pairlayout_block16_two_term<MIPS>(
          char_pointer, sum, v_center_inner_product);
    }
  } else if constexpr (FIXED_LB == 2) {
    const __m512 lut00 = _mm512_load_ps(table + 0);
    const __m512 lut01 = _mm512_load_ps(table + 16);
    const __m512 lut02 = _mm512_load_ps(table + 32);
    const __m512 lut03 = _mm512_load_ps(table + 48);
    const __m512 lut04 = _mm512_load_ps(table + 64);
    const __m512 lut05 = _mm512_load_ps(table + 80);
    const __m512 lut06 = _mm512_load_ps(table + 96);
    const __m512 lut07 = _mm512_load_ps(table + 112);
    const __m512 lut08 = _mm512_load_ps(table + 128);
    const __m512 lut09 = _mm512_load_ps(table + 144);
    const __m512 lut10 = _mm512_load_ps(table + 160);
    const __m512 lut11 = _mm512_load_ps(table + 176);
    const __m512 lut12 = _mm512_load_ps(table + 192);
    const __m512 lut13 = _mm512_load_ps(table + 208);
    const __m512 lut14 = _mm512_load_ps(table + 224);
    const __m512 lut15 = _mm512_load_ps(table + 240);
    if constexpr (!COSINE && !MIPS) {
      return eval_pairlayout_block16_hoisted_lb2(
          char_pointer, nibble_mask, lut00, lut01, lut02, lut03, lut04, lut05,
          lut06, lut07, lut08, lut09, lut10, lut11, lut12, lut13, lut14, lut15,
          v_center_inner_product);
    } else {
      __m512 sum0 = _mm512_setzero_ps();
      __m512 sum1 = _mm512_setzero_ps();
      __m512 sum2 = _mm512_setzero_ps();
      __m512 sum3 = _mm512_setzero_ps();
      add_8level_scores_64bytes(char_pointer + 0, nibble_mask, table + 0,
                                sum0, sum1, sum2, sum3);
      add_8level_scores_64bytes(char_pointer + 64, nibble_mask, table + 128,
                                sum0, sum1, sum2, sum3);
      char_pointer += 128;
      const __m512 sum =
          _mm512_add_ps(_mm512_add_ps(sum0, sum1), _mm512_add_ps(sum2, sum3));
      return finalize_pairlayout_block16_two_term<MIPS>(
          char_pointer, sum, v_center_inner_product);
    }
  } else if constexpr (FIXED_LB == 4) {
    if constexpr (!COSINE && !MIPS) {
      return eval_pairlayout_block16_unrolled_lb4(
          char_pointer, nibble_mask, table, v_center_inner_product);
    } else {
      __m512 sum0 = _mm512_setzero_ps();
      __m512 sum1 = _mm512_setzero_ps();
      __m512 sum2 = _mm512_setzero_ps();
      __m512 sum3 = _mm512_setzero_ps();
      add_8level_scores_64bytes(char_pointer + 0, nibble_mask, table + 0,
                                sum0, sum1, sum2, sum3);
      add_8level_scores_64bytes(char_pointer + 64, nibble_mask, table + 128,
                                sum0, sum1, sum2, sum3);
      add_8level_scores_64bytes(char_pointer + 128, nibble_mask, table + 256,
                                sum0, sum1, sum2, sum3);
      add_8level_scores_64bytes(char_pointer + 192, nibble_mask, table + 384,
                                sum0, sum1, sum2, sum3);
      char_pointer += 256;
      const __m512 sum =
          _mm512_add_ps(_mm512_add_ps(sum0, sum1), _mm512_add_ps(sum2, sum3));
      return finalize_pairlayout_block16_two_term<MIPS>(
          char_pointer, sum, v_center_inner_product);
    }
  } else {
    if constexpr (!COSINE && !MIPS) {
      return eval_pairlayout_block16(char_pointer, nibble_mask, table, FIXED_LB,
                                     v_center_inner_product);
    } else {
      __m512 sum0 = _mm512_setzero_ps();
      __m512 sum1 = _mm512_setzero_ps();
      __m512 sum2 = _mm512_setzero_ps();
      __m512 sum3 = _mm512_setzero_ps();
      for (int rr2 = 0; rr2 < FIXED_LB; rr2++) {
        add_8level_scores_64bytes(char_pointer, nibble_mask, table + rr2 * 128,
                                  sum0, sum1, sum2, sum3);
        char_pointer += 64;
      }
      const __m512 sum =
          _mm512_add_ps(_mm512_add_ps(sum0, sum1), _mm512_add_ps(sum2, sum3));
      return finalize_pairlayout_block16_two_term<MIPS>(
          char_pointer, sum, v_center_inner_product);
    }
  }
}

template <int FIXED_LB, bool COSINE = false, bool MIPS = false>
void queryFixedLevelBlocksImpl(float *query_point, float *query_extended_point,
                          size_t K, std::vector<Neighbor> &final_result,
                          float *table, int step) const {

  float query_norm =
      (COSINE || MIPS) ? 1.0f : sqrt(dot_product_avx512(query_point, query_point));

  float normalized_query[padded_dim_];
  for (int i = 0; i < padded_dim_; i++)
    normalized_query[i] = (COSINE || MIPS) ? query_point[i] : query_point[i] / query_norm;

  float query_int16_scale = 32767.0f / maxAbsValue(normalized_query);

  int16_t query_int16[padded_dim_];
  for (int i = 0; i < padded_dim_; i++)
    query_int16[i] = encodeFloatToInt16(normalized_query[i], query_int16_scale);

  VisitedList *vl = visited_list_pool_->getFreeVisitedList();
  vl_type *visited_array = vl->mass;
  vl_type visited_array_tag = vl->curV;

  const int max_tfb_rounds = ef_ / step;
  const int working_set_capacity = step;
  const int ejected_ring_capacity = (max_tfb_rounds > 1) ? step : 0;
  const int rejected_ring_capacity = (max_tfb_rounds > 1) ? step : 0;
  const int max_round = (max_base_degree_ / 16);
  constexpr int level_blocks = FIXED_LB;
  const __m128i m128_4 = _mm_set1_epi8(0x0F);

  std::vector<Neighbor> working_set(working_set_capacity);
  std::vector<Neighbor> ejected_ring(ejected_ring_capacity);
  std::vector<Neighbor> rejected_ring(rejected_ring_capacity);
  std::vector<Neighbor> ejected_ring_buffer(ejected_ring_capacity);

  int ejected_head = 0;
  int rejected_head = 0;
  int ejected_size = 0;
  int rejected_size = 0;

  char *index_pointer = nullptr;

  const PIFEntry *mips_entry_seeds =
      MIPS ? selectPIFSeeds(query_point) : nullptr;
  char *seed_vector_ptr = entry_point_vector_records_;

  auto locate_index_ptr = [&](int candidate_id) -> char * {
    if (loaded_index_memory_ == nullptr) {
      return adjacency_records_[candidate_id];
    }
    int left = 0;
    int right = max_round - 1;
    while (left < right) {
      int mid = (left + right + 1) >> 1;
      if (candidate_id < static_cast<int>(edge_group_offsets_[mid])) {
        right = mid - 1;
      } else {
        left = mid;
      }
    }
    const int group = left;
    return loaded_index_memory_ + packed_group_offsets_[group] +
           (candidate_id - edge_group_offsets_[group]) *
               packed_group_record_bytes_[group];
  };

  auto merge_buffers = [&](int limit, int &out_size) {
    RefillTFBWorkingSet(working_set, limit, rejected_ring, rejected_head,
                        rejected_size, ejected_ring, ejected_head, ejected_size,
                        ejected_ring_buffer, out_size);
  };

  const int pif_seed_count =
      (MIPS && mips_entry_seeds != nullptr)
          ? step
          : 0;

  for (int i = 0; i < step; i++) {
    const bool use_pif_seed = i < pif_seed_count;
    int cur_id =
        use_pif_seed ? mips_entry_seeds[i].index : entry_point_ids_[i];
    if (i != step - 1) {
      const bool prefetch_pif_seed = (i + 1) < pif_seed_count;
      int next_id = prefetch_pif_seed ? mips_entry_seeds[i + 1].index
                                      : entry_point_ids_[i + 1];
      _mm_prefetch((char *)(visited_array + next_id), _MM_HINT_T0);
    }

    if constexpr (MIPS) {
      seed_vector_ptr = getNormByInternalIdQuery(cur_id);
    }
    float seed_norm = *(float *)seed_vector_ptr;
    seed_vector_ptr += 4;
    float seed_int16_scale = *(float *)seed_vector_ptr;
    seed_vector_ptr += 4;

    float seed_inner_product =
        query_norm *
        dot_product_avx512_int16(query_int16, (int16_t *)seed_vector_ptr,
                                 query_int16_scale, seed_int16_scale);
    float seed_distance;
    if constexpr (COSINE) {
      seed_distance = 0.5f - seed_inner_product;
    } else if constexpr (MIPS) {
      seed_distance = -seed_norm * seed_inner_product;
    } else {
      seed_distance = seed_norm * (seed_norm * 0.5f - seed_inner_product);
    }

    seed_vector_ptr += (padded_dim_ * sizeof(int16_t));
    index_pointer = locate_index_ptr(cur_id);

    working_set[i] = Neighbor(*(unsigned int *)seed_vector_ptr, seed_distance,
                              seed_inner_product, true, index_pointer);
    seed_vector_ptr += 4;
    visited_array[cur_id] = visited_array_tag;
  }

  std::sort(working_set.begin(), working_set.begin() + step);

  if constexpr (MIPS) {
    if (std::getenv("PAG_PIF_ONLY") != nullptr) {
      final_result.resize(K);
      const int result_count = std::min<int>((int)K, step);
      for (int result_id = 0; result_id < result_count; ++result_id) {
        final_result[result_id] = working_set[result_id];
      }
      for (size_t result_id = result_count; result_id < K; ++result_id) {
        final_result[result_id] =
            Neighbor(0, std::numeric_limits<float>::infinity(), 0.0f, false);
      }
      visited_list_pool_->releaseVisitedList(vl);
      return;
    }
  }

  float *projection_direction_ptr = query_projection_directions_;
  for (int j = 0; j < projection_level_count_; j++) {
    float *y = query_extended_point + j * projection_subspace_dim_;
    float *projection_table_block = table + j * (2 * projection_width_);
    for (int i = 0; i < projection_width_; i++) {
      projection_table_block[i] =
          -1 * projection_distance_func_((void *)y,
                                         (void *)projection_direction_ptr,
                                         projection_distance_param_);
      projection_direction_ptr += projection_subspace_dim_;
      projection_table_block[i + projection_width_] = -1 * projection_table_block[i];
    }
  }

	  auto filterNeighborsByPRT16 = [&](int *neighborIds, int size, float center_ip,
	                                    float lower_bound,
	                                    int *prt_passed_ids) -> int {
	    const int round = (size + 15) >> 4;
	    const int div = (size % 16 == 0 ? 16 : size % 16);
	    const __m512 v_center_ip = _mm512_set1_ps(center_ip);
	    const __m512 v_lower_bound = _mm512_set1_ps(lower_bound);

    int count = 0;
    const int *ids = neighborIds;
    const int projection_record_offset =
        loaded_index_memory_ == nullptr ? max_base_degree_ : 16 * round;
    const unsigned char *char_pointer = reinterpret_cast<const unsigned char *>(
        neighborIds + projection_record_offset);
	
	    for (int rr = 0; rr < round; rr++) {
	      const __m512 pred = eval_pairlayout_block16_fixed_impl<FIXED_LB, COSINE, MIPS>(
	          char_pointer, m128_4, table, v_center_ip);

      const __mmask16 valid_lane_mask = (rr == round - 1 && div != 16)
                                            ? ((__mmask16)((1u << div) - 1u))
                                            : (__mmask16)0xFFFF;

      const __mmask16 passed_mask =
          _mm512_cmp_ps_mask(pred, v_lower_bound, _CMP_LE_OS) & valid_lane_mask;

      const __m512i v_neighbor_ids =
          _mm512_loadu_si512(reinterpret_cast<const __m512i *>(ids));
      _mm512_mask_compressstoreu_epi32(prt_passed_ids + count, passed_mask,
                                       v_neighbor_ids);
      count += _mm_popcnt_u32((unsigned)passed_mask);

      if (rr == round - 2 && count > 0) {
        _mm_prefetch((char *)(visited_array + *prt_passed_ids), _MM_HINT_T0);
        _mm_prefetch(vector_records_ + (*prt_passed_ids) * vector_record_bytes_,
                     _MM_HINT_T0);
      }

      ids += 16;
    }

    return count;
  };

  int k = 0;
  int next_k = 0;
  int next_next_k = 0;
  int prt_passed_ids[max_base_degree_];

  while (k < working_set_capacity) {
    working_set[k].needs_expansion = false;

    if (k == next_k) {
      next_k = working_set_capacity;
      for (int ii = k + 1; ii < working_set_capacity; ii++) {
        if (working_set[ii].needs_expansion == true) {
          next_k = ii;
          _mm_prefetch((char *)(working_set[next_k].edge_record), _MM_HINT_T0);
          break;
        }
      }
    }

    int *data = (int *)working_set[k].edge_record;
    int size = *data;
    int *neighborIds = data + 1;

    const int count = filterNeighborsByPRT16(
        neighborIds, size, working_set[k].inner_product,
        working_set[working_set_capacity - 1].distance, prt_passed_ids);

    if (count == 0) {
      k = next_k;
      continue;
    }

    next_next_k = next_k;
    size = count;
    neighborIds = prt_passed_ids;

    for (size_t j = 0; j < (size_t)size; j++) {
      int candidate_id = neighborIds[j];
      int *prefetch_id_ptr1 = neighborIds + j + 1;
      int *prefetch_id_ptr2 = neighborIds + j + 2;

      _mm_prefetch((char *)(visited_array + *prefetch_id_ptr1), _MM_HINT_T0);
      _mm_prefetch(vector_records_ + (*prefetch_id_ptr1) * vector_record_bytes_,
                   _MM_HINT_T0);
      _mm_prefetch(vector_records_ + (*prefetch_id_ptr2) * vector_record_bytes_,
                   _MM_HINT_T0);

      if (!(visited_array[candidate_id] == visited_array_tag)) {
        visited_array[candidate_id] = visited_array_tag;

        float *vector_record_ptr =
            (float *)getNormByInternalIdQuery(candidate_id);

        float candidate_norm = *vector_record_ptr;
        vector_record_ptr += 1;

        float candidate_int16_scale = *vector_record_ptr;
        vector_record_ptr += 1;

        float candidate_inner_product =
            query_norm *
            dot_product_avx512_int16(query_int16, (int16_t *)vector_record_ptr,
                                     query_int16_scale, candidate_int16_scale);
        float candidate_distance;
        if constexpr (COSINE) {
          candidate_distance = 0.5f - candidate_inner_product;
        } else if constexpr (MIPS) {
          candidate_distance = -candidate_norm * candidate_inner_product;
        } else {
          candidate_distance =
              candidate_norm * (candidate_norm * 0.5f - candidate_inner_product);
        }

        if (candidate_distance >=
            working_set[working_set_capacity - 1].distance) {
          if (max_tfb_rounds <= 1) {
            continue;
          }
          index_pointer = locate_index_ptr(candidate_id);
          Neighbor rejected_candidate(
              *(int *)(vector_record_ptr + (padded_dim_ / 2)),
              candidate_distance, candidate_inner_product, true, index_pointer);
          PushTFBRing(rejected_ring, rejected_head, rejected_size,
                      rejected_ring_capacity, rejected_candidate);
          continue;
        }

        if (max_tfb_rounds > 1 &&
            working_set[working_set_capacity - 1].needs_expansion == true) {
          PushTFBRing(ejected_ring, ejected_head, ejected_size,
                      ejected_ring_capacity,
                      working_set[working_set_capacity - 1]);
        }

        index_pointer = locate_index_ptr(candidate_id);
        Neighbor accepted_candidate(
            *(int *)(vector_record_ptr + (padded_dim_ / 2)), candidate_distance,
            candidate_inner_product, true, index_pointer);
        int r = InsertIntoPool2(working_set.data(), working_set_capacity,
                                accepted_candidate);

        if (r <= next_next_k) {
          if (r <= next_k) {
            next_next_k = next_k + 1;
            next_k = r;
            _mm_prefetch((char *)(index_pointer), _MM_HINT_T0);
          } else {
            next_next_k = r;
          }
        }
      }
    }

    k = next_k;
    next_k = next_next_k;
  }

  final_result.resize(K);
  const int initial_result_count = std::min<int>((int)K, working_set_capacity);
  for (int result_id = 0; result_id < initial_result_count; ++result_id) {
    final_result[result_id] = working_set[result_id];
  }
  for (size_t result_id = initial_result_count; result_id < K; ++result_id) {
    final_result[result_id] =
        Neighbor(0, std::numeric_limits<float>::infinity(), 0.0f, false);
  }

  int working_set_size = 0;
  if (max_tfb_rounds > 1) {
    merge_buffers(step, working_set_size);
  }

  for (int seg = 1; seg < max_tfb_rounds - 1; seg++) {
    if (working_set_size < 1)
      break;

    k = 0;
    next_k = 0;
    next_next_k = 0;

    while (k < working_set_size) {
      working_set[k].needs_expansion = false;

      if (k == next_k) {
        next_k = working_set_size;
        for (int ii = k + 1; ii < working_set_size; ii++) {
          if (working_set[ii].needs_expansion == true) {
            next_k = ii;
            _mm_prefetch((char *)(working_set[next_k].edge_record),
                         _MM_HINT_T0);
            break;
          }
        }
      }

      int *data = (int *)working_set[k].edge_record;
      int size = *data;
      int *neighborIds = data + 1;

      const int count = filterNeighborsByPRT16(
          neighborIds, size, working_set[k].inner_product,
          working_set[working_set_size - 1].distance, prt_passed_ids);

      if (count == 0) {
        k = next_k;
        continue;
      }

      next_next_k = next_k;
      size = count;
      neighborIds = prt_passed_ids;

      for (size_t j = 0; j < (size_t)size; j++) {
        int candidate_id = neighborIds[j];
        int *prefetch_id_ptr1 = neighborIds + j + 1;
        int *prefetch_id_ptr2 = neighborIds + j + 2;
#ifdef USE_SSE
        _mm_prefetch((char *)(visited_array + *prefetch_id_ptr1), _MM_HINT_T0);
        _mm_prefetch(vector_records_ +
                         (*prefetch_id_ptr1) * vector_record_bytes_,
                     _MM_HINT_T0);
        _mm_prefetch(vector_records_ +
                         (*prefetch_id_ptr2) * vector_record_bytes_,
                     _MM_HINT_T0);
#endif
        if (!(visited_array[candidate_id] == visited_array_tag)) {
          visited_array[candidate_id] = visited_array_tag;

          float *vector_record_ptr =
              (float *)getNormByInternalIdQuery(candidate_id);

          float candidate_norm = *vector_record_ptr;
          vector_record_ptr += 1;

          float candidate_int16_scale = *vector_record_ptr;
          vector_record_ptr += 1;

          float candidate_inner_product =
              query_norm * dot_product_avx512_int16(
                               query_int16, (int16_t *)vector_record_ptr,
                               query_int16_scale, candidate_int16_scale);
          float candidate_distance;
          if constexpr (COSINE) {
            candidate_distance = 0.5f - candidate_inner_product;
          } else if constexpr (MIPS) {
            candidate_distance = -candidate_norm * candidate_inner_product;
          } else {
            candidate_distance =
                candidate_norm * (candidate_norm * 0.5f - candidate_inner_product);
          }

          if (candidate_distance >=
              working_set[working_set_size - 1].distance) {
            index_pointer = locate_index_ptr(candidate_id);
            Neighbor rejected_candidate(
                *(int *)(vector_record_ptr + (padded_dim_ / 2)),
                candidate_distance, candidate_inner_product, true,
                index_pointer);
            PushTFBRing(rejected_ring, rejected_head, rejected_size,
                        rejected_ring_capacity, rejected_candidate);
            continue;
          }

          if (working_set[working_set_size - 1].needs_expansion == true) {
            PushTFBRing(ejected_ring, ejected_head, ejected_size,
                        ejected_ring_capacity,
                        working_set[working_set_size - 1]);
          }

          index_pointer = locate_index_ptr(candidate_id);
          Neighbor accepted_candidate(
              *(int *)(vector_record_ptr + (padded_dim_ / 2)),
              candidate_distance, candidate_inner_product, true, index_pointer);
          int r = InsertIntoPool2(working_set.data(), working_set_size,
                                  accepted_candidate);

          if (r <= next_next_k) {
            if (r <= next_k) {
              next_next_k = next_k + 1;
              next_k = r;
              _mm_prefetch((char *)(index_pointer), _MM_HINT_T0);
            } else {
              next_next_k = r;
            }
          }
        }
      }

      k = next_k;
      next_k = next_next_k;
    }

    for (int j = 0; j < working_set_size; j++) {
      if (working_set[j].distance < final_result[K - 1].distance) {
        InsertIntoPool2(final_result.data(), K, working_set[j]);
      } else {
        break;
      }
    }

    merge_buffers(step, working_set_size);
  }

  if (max_tfb_rounds > 1) {
    k = 0;
    next_k = 0;
    next_next_k = 0;

    while (k < working_set_size) {
      working_set[k].needs_expansion = false;

      if (k == next_k) {
        next_k = working_set_size;
        for (int ii = k + 1; ii < working_set_size; ii++) {
          if (working_set[ii].needs_expansion == true) {
            next_k = ii;
            _mm_prefetch((char *)(working_set[next_k].edge_record),
                         _MM_HINT_T0);
            break;
          }
        }
      }

      int *data = (int *)working_set[k].edge_record;
      int size = *data;
      int *neighborIds = data + 1;

      const int count = filterNeighborsByPRT16(
          neighborIds, size, working_set[k].inner_product,
          working_set[working_set_size - 1].distance, prt_passed_ids);

      if (count == 0) {
        k = next_k;
        continue;
      }

      next_next_k = next_k;
      size = count;
      neighborIds = prt_passed_ids;

      for (size_t j = 0; j < (size_t)size; j++) {
        int candidate_id = neighborIds[j];
        int *prefetch_id_ptr1 = neighborIds + j + 1;
        int *prefetch_id_ptr2 = neighborIds + j + 2;
#ifdef USE_SSE
        _mm_prefetch((char *)(visited_array + *prefetch_id_ptr1), _MM_HINT_T0);
        _mm_prefetch(vector_records_ +
                         (*prefetch_id_ptr1) * vector_record_bytes_,
                     _MM_HINT_T0);
        _mm_prefetch(vector_records_ +
                         (*prefetch_id_ptr2) * vector_record_bytes_,
                     _MM_HINT_T0);
#endif
        if (!(visited_array[candidate_id] == visited_array_tag)) {
          visited_array[candidate_id] = visited_array_tag;

          float *vector_record_ptr =
              (float *)getNormByInternalIdQuery(candidate_id);

          float candidate_norm = *vector_record_ptr;
          vector_record_ptr += 1;

          float candidate_int16_scale = *vector_record_ptr;
          vector_record_ptr += 1;

          float candidate_inner_product =
              query_norm * dot_product_avx512_int16(
                               query_int16, (int16_t *)vector_record_ptr,
                               query_int16_scale, candidate_int16_scale);
          float candidate_distance;
          if constexpr (COSINE) {
            candidate_distance = 0.5f - candidate_inner_product;
          } else if constexpr (MIPS) {
            candidate_distance = -candidate_norm * candidate_inner_product;
          } else {
            candidate_distance =
                candidate_norm * (candidate_norm * 0.5f - candidate_inner_product);
          }

          if (candidate_distance >=
              working_set[working_set_size - 1].distance) {
            continue;
          }

          index_pointer = locate_index_ptr(candidate_id);
          Neighbor accepted_candidate(
              *(int *)(vector_record_ptr + (padded_dim_ / 2)),
              candidate_distance, candidate_inner_product, true, index_pointer);
          int r = InsertIntoPool2(working_set.data(), working_set_size,
                                  accepted_candidate);

          if (r <= next_next_k) {
            if (r <= next_k) {
              next_next_k = next_k + 1;
              next_k = r;
              _mm_prefetch((char *)(index_pointer), _MM_HINT_T0);
            } else {
              next_next_k = r;
            }
          }
        }
      }

      k = next_k;
      next_k = next_next_k;
    }

    for (int j = 0; j < working_set_size; j++) {
      if (working_set[j].distance < final_result[K - 1].distance) {
        InsertIntoPool2(final_result.data(), K, working_set[j]);
      } else {
        break;
      }
    }
  }

  visited_list_pool_->releaseVisitedList(vl);
}

void queryLevelBlocks1(float *query_point, float *query_extended_point, size_t K,
                   std::vector<Neighbor> &final_result, float *table,
                   int step) const {
  queryFixedLevelBlocksImpl<1>(query_point, query_extended_point, K, final_result,
                          table, step);
}

void queryLevelBlocks2(float *query_point, float *query_extended_point, size_t K,
                   std::vector<Neighbor> &final_result, float *table,
                   int step) const {
  queryFixedLevelBlocksImpl<2>(query_point, query_extended_point, K, final_result,
                          table, step);
}

void queryLevelBlocks4(float *query_point, float *query_extended_point, size_t K,
                   std::vector<Neighbor> &final_result, float *table,
                   int step) const {
  queryFixedLevelBlocksImpl<4>(query_point, query_extended_point, K, final_result,
                          table, step);
}
void queryRuntimeLevelBlocksImpl(float *query_point, float *query_extended_point,
                            size_t K, std::vector<Neighbor> &final_result,
                            float *table, int step) const {

  float query_norm = sqrt(dot_product_avx512(query_point, query_point));

  float normalized_query[padded_dim_];
  for (int i = 0; i < padded_dim_; i++)
    normalized_query[i] = query_point[i] / query_norm;

  float query_int16_scale = 32767.0f / maxAbsValue(normalized_query);

  int16_t query_int16[padded_dim_];
  for (int i = 0; i < padded_dim_; i++)
    query_int16[i] = encodeFloatToInt16(normalized_query[i], query_int16_scale);

  VisitedList *vl = visited_list_pool_->getFreeVisitedList();
  vl_type *visited_array = vl->mass;
  vl_type visited_array_tag = vl->curV;

  const int max_tfb_rounds = ef_ / step;
  const int working_set_capacity = step;
  const int ejected_ring_capacity = step;
  const int rejected_ring_capacity = step;
  const int max_round = (max_base_degree_ / 16);
  const int level_blocks = projection_level_count_ / 8;
  const __m128i m128_4 = _mm_set1_epi8(0x0F);

  std::vector<Neighbor> working_set(working_set_capacity);
  std::vector<Neighbor> ejected_ring(ejected_ring_capacity);
  std::vector<Neighbor> rejected_ring(rejected_ring_capacity);
  std::vector<Neighbor> ejected_ring_buffer(ejected_ring_capacity);

  int ejected_head = 0;
  int rejected_head = 0;
  int ejected_size = 0;
  int rejected_size = 0;

  int cur_round = 0;
  char *index_pointer = nullptr;

  int next_id = entry_point_ids_[0];
  char *seed_vector_ptr = entry_point_vector_records_;

  auto locate_index_ptr = [&](int candidate_id) -> char * {
    if (loaded_index_memory_ == nullptr) {
      return adjacency_records_[candidate_id];
    }
    int left = 0;
    int right = max_round - 1;
    while (left < right) {
      int mid = (left + right + 1) >> 1;
      if (candidate_id < static_cast<int>(edge_group_offsets_[mid])) {
        right = mid - 1;
      } else {
        left = mid;
      }
    }
    const int group = left;
    return loaded_index_memory_ + packed_group_offsets_[group] +
           (candidate_id - edge_group_offsets_[group]) *
               packed_group_record_bytes_[group];
  };

  auto merge_buffers = [&](int limit, int &out_size) {
    RefillTFBWorkingSet(working_set, limit, rejected_ring, rejected_head,
                        rejected_size, ejected_ring, ejected_head, ejected_size,
                        ejected_ring_buffer, out_size);
  };

  for (int i = 0; i < step; i++) {
    int cur_id = next_id;

    if (i != step - 1) {
      next_id = entry_point_ids_[i + 1];
      _mm_prefetch((char *)(visited_array + next_id), _MM_HINT_T0);
    }

    float seed_norm = *(float *)seed_vector_ptr;
    seed_vector_ptr += 4;
    float seed_int16_scale = *(float *)seed_vector_ptr;
    seed_vector_ptr += 4;

    float seed_inner_product =
        query_norm *
        dot_product_avx512_int16(query_int16, (int16_t *)seed_vector_ptr,
                                 query_int16_scale, seed_int16_scale);
    float seed_distance = seed_norm * (seed_norm * 0.5f - seed_inner_product);

    seed_vector_ptr += (padded_dim_ * sizeof(int16_t));
    index_pointer = locate_index_ptr(cur_id);

    working_set[i] = Neighbor(*(unsigned int *)seed_vector_ptr, seed_distance,
                              seed_inner_product, true, index_pointer);
    seed_vector_ptr += 4;
    visited_array[cur_id] = visited_array_tag;
  }

  std::sort(working_set.begin(), working_set.begin() + step);

  float *projection_direction_ptr = query_projection_directions_;
  for (int j = 0; j < projection_level_count_; j++) {
    float *y = query_extended_point + j * projection_subspace_dim_;
    float *projection_table_block = table + j * (2 * projection_width_);
    for (int i = 0; i < projection_width_; i++) {
      projection_table_block[i] =
          -1 * projection_distance_func_((void *)y,
                                         (void *)projection_direction_ptr,
                                         projection_distance_param_);
      projection_direction_ptr += projection_subspace_dim_;
      projection_table_block[i + projection_width_] = -1 * projection_table_block[i];
    }
  }

  const bool use_hoisted_lut_lb1 = (level_blocks == 1);
  const bool use_hoisted_lut_lb2 = (level_blocks == 2);

  __m512 lut00, lut01, lut02, lut03, lut04, lut05, lut06, lut07;
  __m512 lut08, lut09, lut10, lut11, lut12, lut13, lut14, lut15;

  if (use_hoisted_lut_lb1 || use_hoisted_lut_lb2) {
    lut00 = _mm512_load_ps(table + 0);
    lut01 = _mm512_load_ps(table + 16);
    lut02 = _mm512_load_ps(table + 32);
    lut03 = _mm512_load_ps(table + 48);
    lut04 = _mm512_load_ps(table + 64);
    lut05 = _mm512_load_ps(table + 80);
    lut06 = _mm512_load_ps(table + 96);
    lut07 = _mm512_load_ps(table + 112);

    if (use_hoisted_lut_lb2) {
      lut08 = _mm512_load_ps(table + 128);
      lut09 = _mm512_load_ps(table + 144);
      lut10 = _mm512_load_ps(table + 160);
      lut11 = _mm512_load_ps(table + 176);
      lut12 = _mm512_load_ps(table + 192);
      lut13 = _mm512_load_ps(table + 208);
      lut14 = _mm512_load_ps(table + 224);
      lut15 = _mm512_load_ps(table + 240);
    }
  }

  auto eval_pairlayout_block16_search =
      [&](const unsigned char *&char_pointer,
          const __m512 v_center_ip) -> __m512 {
    if (use_hoisted_lut_lb1) {
      return eval_pairlayout_block16_hoisted_lb1(
          char_pointer, m128_4, lut00, lut01, lut02, lut03, lut04, lut05, lut06,
          lut07, v_center_ip);
    }
    if (use_hoisted_lut_lb2) {
      return eval_pairlayout_block16_hoisted_lb2(
          char_pointer, m128_4, lut00, lut01, lut02, lut03, lut04, lut05, lut06,
          lut07, lut08, lut09, lut10, lut11, lut12, lut13, lut14, lut15,
          v_center_ip);
    }
    if (level_blocks == 4) {
      return eval_pairlayout_block16_unrolled_lb4(char_pointer, m128_4, table,
                                                  v_center_ip);
    }
    return eval_pairlayout_block16(char_pointer, m128_4, table, level_blocks,
                                   v_center_ip);
  };

  auto filterNeighborsByPRT16 = [&](int *neighborIds, int size, float center_ip,
                                    float lower_bound,
                                    int *prt_passed_ids) -> int {
    const int round = (size + 15) >> 4;
    const int div = (size % 16 == 0 ? 16 : size % 16);
    const __m512 v_center_ip = _mm512_set1_ps(center_ip);
    const __m512 v_lower_bound = _mm512_set1_ps(lower_bound);

    int count = 0;
    const int *ids = neighborIds;
    const int projection_record_offset =
        loaded_index_memory_ == nullptr ? max_base_degree_ : 16 * round;
    const unsigned char *char_pointer = reinterpret_cast<const unsigned char *>(
        neighborIds + projection_record_offset);

    for (int rr = 0; rr < round; rr++) {
      const __m512 pred =
          eval_pairlayout_block16_search(char_pointer, v_center_ip);

      const __mmask16 valid_lane_mask = (rr == round - 1 && div != 16)
                                            ? ((__mmask16)((1u << div) - 1u))
                                            : (__mmask16)0xFFFF;

      const __mmask16 passed_mask =
          _mm512_cmp_ps_mask(pred, v_lower_bound, _CMP_LE_OS) & valid_lane_mask;

      const __m512i v_neighbor_ids =
          _mm512_loadu_si512(reinterpret_cast<const __m512i *>(ids));
      _mm512_mask_compressstoreu_epi32(prt_passed_ids + count, passed_mask,
                                       v_neighbor_ids);
      count += _mm_popcnt_u32((unsigned)passed_mask);

      if (rr == round - 2 && count > 0) {
        _mm_prefetch((char *)(visited_array + *prt_passed_ids), _MM_HINT_T0);
        _mm_prefetch(vector_records_ + (*prt_passed_ids) * vector_record_bytes_,
                     _MM_HINT_T0);
      }

      ids += 16;
    }

    return count;
  };

  int k = 0;
  int next_k = 0;
  int next_next_k = 0;
  int prt_passed_ids[max_base_degree_];

  while (k < working_set_capacity) {
    working_set[k].needs_expansion = false;

    if (k == next_k) {
      next_k = working_set_capacity;
      for (int ii = k + 1; ii < working_set_capacity; ii++) {
        if (working_set[ii].needs_expansion == true) {
          next_k = ii;
          _mm_prefetch((char *)(working_set[next_k].edge_record), _MM_HINT_T0);
          break;
        }
      }
    }

    int *data = (int *)working_set[k].edge_record;
    int size = *data;
    int *neighborIds = data + 1;

    const int count = filterNeighborsByPRT16(
        neighborIds, size, working_set[k].inner_product,
        working_set[working_set_capacity - 1].distance, prt_passed_ids);

    if (count == 0) {
      k = next_k;
      continue;
    }

    next_next_k = next_k;
    size = count;
    neighborIds = prt_passed_ids;

    for (size_t j = 0; j < (size_t)size; j++) {
      int candidate_id = neighborIds[j];
      int *prefetch_id_ptr1 = neighborIds + j + 1;
      int *prefetch_id_ptr2 = neighborIds + j + 2;

      _mm_prefetch((char *)(visited_array + *prefetch_id_ptr1), _MM_HINT_T0);
      _mm_prefetch(vector_records_ + (*prefetch_id_ptr1) * vector_record_bytes_,
                   _MM_HINT_T0);
      _mm_prefetch(vector_records_ + (*prefetch_id_ptr2) * vector_record_bytes_,
                   _MM_HINT_T0);

      if (!(visited_array[candidate_id] == visited_array_tag)) {
        visited_array[candidate_id] = visited_array_tag;

        float *vector_record_ptr =
            (float *)getNormByInternalIdQuery(candidate_id);

        float candidate_norm = *vector_record_ptr;
        vector_record_ptr += 1;

        float candidate_int16_scale = *vector_record_ptr;
        vector_record_ptr += 1;

        float candidate_inner_product =
            query_norm *
            dot_product_avx512_int16(query_int16, (int16_t *)vector_record_ptr,
                                     query_int16_scale, candidate_int16_scale);
        float candidate_distance =
            candidate_norm * (candidate_norm * 0.5f - candidate_inner_product);

        if (candidate_distance >=
            working_set[working_set_capacity - 1].distance) {
          index_pointer = locate_index_ptr(candidate_id);
          Neighbor rejected_candidate(
              *(int *)(vector_record_ptr + (padded_dim_ / 2)),
              candidate_distance, candidate_inner_product, true, index_pointer);
          PushTFBRing(rejected_ring, rejected_head, rejected_size,
                      rejected_ring_capacity, rejected_candidate);
          continue;
        }

        if (working_set[working_set_capacity - 1].needs_expansion == true) {
          PushTFBRing(ejected_ring, ejected_head, ejected_size,
                      ejected_ring_capacity,
                      working_set[working_set_capacity - 1]);
        }

        index_pointer = locate_index_ptr(candidate_id);
        Neighbor accepted_candidate(
            *(int *)(vector_record_ptr + (padded_dim_ / 2)), candidate_distance,
            candidate_inner_product, true, index_pointer);
        int r = InsertIntoPool2(working_set.data(), working_set_capacity,
                                accepted_candidate);

        if (r <= next_next_k) {
          if (r <= next_k) {
            next_next_k = next_k + 1;
            next_k = r;
            _mm_prefetch((char *)(index_pointer), _MM_HINT_T0);
          } else {
            next_next_k = r;
          }
        }
      }
    }

    k = next_k;
    next_k = next_next_k;
  }

  final_result.resize(K);
  const int initial_result_count = std::min<int>((int)K, working_set_capacity);
  for (int result_id = 0; result_id < initial_result_count; ++result_id) {
    final_result[result_id] = working_set[result_id];
  }
  for (size_t result_id = initial_result_count; result_id < K; ++result_id) {
    final_result[result_id] =
        Neighbor(0, std::numeric_limits<float>::infinity(), 0.0f, false);
  }

  int working_set_size = 0;
  if (max_tfb_rounds > 1) {
    merge_buffers(step, working_set_size);
  }

  for (int seg = 1; seg < max_tfb_rounds - 1; seg++) {
    if (working_set_size < 1)
      break;

    k = 0;
    next_k = 0;
    next_next_k = 0;

    while (k < working_set_size) {
      working_set[k].needs_expansion = false;

      if (k == next_k) {
        next_k = working_set_size;
        for (int ii = k + 1; ii < working_set_size; ii++) {
          if (working_set[ii].needs_expansion == true) {
            next_k = ii;
            _mm_prefetch((char *)(working_set[next_k].edge_record),
                         _MM_HINT_T0);
            break;
          }
        }
      }

      int *data = (int *)working_set[k].edge_record;
      int size = *data;
      int *neighborIds = data + 1;

      const int count = filterNeighborsByPRT16(
          neighborIds, size, working_set[k].inner_product,
          working_set[working_set_size - 1].distance, prt_passed_ids);

      if (count == 0) {
        k = next_k;
        continue;
      }

      next_next_k = next_k;
      size = count;
      neighborIds = prt_passed_ids;

      for (size_t j = 0; j < (size_t)size; j++) {
        int candidate_id = neighborIds[j];
        int *prefetch_id_ptr1 = neighborIds + j + 1;
        int *prefetch_id_ptr2 = neighborIds + j + 2;
#ifdef USE_SSE
        _mm_prefetch((char *)(visited_array + *prefetch_id_ptr1), _MM_HINT_T0);
        _mm_prefetch(vector_records_ +
                         (*prefetch_id_ptr1) * vector_record_bytes_,
                     _MM_HINT_T0);
        _mm_prefetch(vector_records_ +
                         (*prefetch_id_ptr2) * vector_record_bytes_,
                     _MM_HINT_T0);
#endif
        if (!(visited_array[candidate_id] == visited_array_tag)) {
          visited_array[candidate_id] = visited_array_tag;

          float *vector_record_ptr =
              (float *)getNormByInternalIdQuery(candidate_id);

          float candidate_norm = *vector_record_ptr;
          vector_record_ptr += 1;

          float candidate_int16_scale = *vector_record_ptr;
          vector_record_ptr += 1;

          float candidate_inner_product =
              query_norm * dot_product_avx512_int16(
                               query_int16, (int16_t *)vector_record_ptr,
                               query_int16_scale, candidate_int16_scale);
          float candidate_distance = candidate_norm * (candidate_norm * 0.5f -
                                                       candidate_inner_product);

          if (candidate_distance >=
              working_set[working_set_size - 1].distance) {
            index_pointer = locate_index_ptr(candidate_id);
            Neighbor rejected_candidate(
                *(int *)(vector_record_ptr + (padded_dim_ / 2)),
                candidate_distance, candidate_inner_product, true,
                index_pointer);
            PushTFBRing(rejected_ring, rejected_head, rejected_size,
                        rejected_ring_capacity, rejected_candidate);
            continue;
          }

          if (working_set[working_set_size - 1].needs_expansion == true) {
            PushTFBRing(ejected_ring, ejected_head, ejected_size,
                        ejected_ring_capacity,
                        working_set[working_set_size - 1]);
          }

          index_pointer = locate_index_ptr(candidate_id);
          Neighbor accepted_candidate(
              *(int *)(vector_record_ptr + (padded_dim_ / 2)),
              candidate_distance, candidate_inner_product, true, index_pointer);
          int r = InsertIntoPool2(working_set.data(), working_set_size,
                                  accepted_candidate);

          if (r <= next_next_k) {
            if (r <= next_k) {
              next_next_k = next_k + 1;
              next_k = r;
              _mm_prefetch((char *)(index_pointer), _MM_HINT_T0);
            } else {
              next_next_k = r;
            }
          }
        }
      }

      k = next_k;
      next_k = next_next_k;
    }

    for (int j = 0; j < working_set_size; j++) {
      if (working_set[j].distance < final_result[K - 1].distance) {
        InsertIntoPool2(final_result.data(), K, working_set[j]);
      } else {
        break;
      }
    }

    merge_buffers(step, working_set_size);
  }

  if (max_tfb_rounds > 1) {
    k = 0;
    next_k = 0;
    next_next_k = 0;

    while (k < working_set_size) {
      working_set[k].needs_expansion = false;

      if (k == next_k) {
        next_k = working_set_size;
        for (int ii = k + 1; ii < working_set_size; ii++) {
          if (working_set[ii].needs_expansion == true) {
            next_k = ii;
            _mm_prefetch((char *)(working_set[next_k].edge_record),
                         _MM_HINT_T0);
            break;
          }
        }
      }

      int *data = (int *)working_set[k].edge_record;
      int size = *data;
      int *neighborIds = data + 1;

      const int count = filterNeighborsByPRT16(
          neighborIds, size, working_set[k].inner_product,
          working_set[working_set_size - 1].distance, prt_passed_ids);

      if (count == 0) {
        k = next_k;
        continue;
      }

      next_next_k = next_k;
      size = count;
      neighborIds = prt_passed_ids;

      for (size_t j = 0; j < (size_t)size; j++) {
        int candidate_id = neighborIds[j];
        int *prefetch_id_ptr1 = neighborIds + j + 1;
        int *prefetch_id_ptr2 = neighborIds + j + 2;
#ifdef USE_SSE
        _mm_prefetch((char *)(visited_array + *prefetch_id_ptr1), _MM_HINT_T0);
        _mm_prefetch(vector_records_ +
                         (*prefetch_id_ptr1) * vector_record_bytes_,
                     _MM_HINT_T0);
        _mm_prefetch(vector_records_ +
                         (*prefetch_id_ptr2) * vector_record_bytes_,
                     _MM_HINT_T0);
#endif
        if (!(visited_array[candidate_id] == visited_array_tag)) {
          visited_array[candidate_id] = visited_array_tag;

          float *vector_record_ptr =
              (float *)getNormByInternalIdQuery(candidate_id);

          float candidate_norm = *vector_record_ptr;
          vector_record_ptr += 1;

          float candidate_int16_scale = *vector_record_ptr;
          vector_record_ptr += 1;

          float candidate_inner_product =
              query_norm * dot_product_avx512_int16(
                               query_int16, (int16_t *)vector_record_ptr,
                               query_int16_scale, candidate_int16_scale);
          float candidate_distance = candidate_norm * (candidate_norm * 0.5f -
                                                       candidate_inner_product);

          if (candidate_distance >=
              working_set[working_set_size - 1].distance) {
            continue;
          }

          index_pointer = locate_index_ptr(candidate_id);
          Neighbor accepted_candidate(
              *(int *)(vector_record_ptr + (padded_dim_ / 2)),
              candidate_distance, candidate_inner_product, true, index_pointer);
          int r = InsertIntoPool2(working_set.data(), working_set_size,
                                  accepted_candidate);

          if (r <= next_next_k) {
            if (r <= next_k) {
              next_next_k = next_k + 1;
              next_k = r;
              _mm_prefetch((char *)(index_pointer), _MM_HINT_T0);
            } else {
              next_next_k = r;
            }
          }
        }
      }

      k = next_k;
      next_k = next_next_k;
    }

    for (int j = 0; j < working_set_size; j++) {
      if (working_set[j].distance < final_result[K - 1].distance) {
        InsertIntoPool2(final_result.data(), K, working_set[j]);
      } else {
        break;
      }
    }
  }

  visited_list_pool_->releaseVisitedList(vl);
}

void queryImpl(float *query_point, float *query_extended_point, size_t K,
               std::vector<Neighbor> &final_result, float *table,
               int step) const {
  if (step > static_cast<int>(max_entry_points_)) {
    throw std::runtime_error(
        "Requested TFB working set is larger than the index seed count; "
        "rebuild the index with a larger entry point capacity.");
  }

  const int level_blocks = projection_level_count_ >> 3;
  if (options_.metric == PAGMetric::Cosine) {
    switch (level_blocks) {
    case 1:
      queryFixedLevelBlocksImpl<1, true>(query_point, query_extended_point, K,
                                         final_result, table, step);
      return;
    case 2:
      queryFixedLevelBlocksImpl<2, true>(query_point, query_extended_point, K,
                                         final_result, table, step);
      return;
    case 3:
      queryFixedLevelBlocksImpl<3, true>(query_point, query_extended_point, K,
                                         final_result, table, step);
      return;
    case 4:
      queryFixedLevelBlocksImpl<4, true>(query_point, query_extended_point, K,
                                         final_result, table, step);
      return;
    case 5:
      queryFixedLevelBlocksImpl<5, true>(query_point, query_extended_point, K,
                                         final_result, table, step);
      return;
    case 6:
      queryFixedLevelBlocksImpl<6, true>(query_point, query_extended_point, K,
                                         final_result, table, step);
      return;
    case 7:
      queryFixedLevelBlocksImpl<7, true>(query_point, query_extended_point, K,
                                         final_result, table, step);
      return;
    case 8:
      queryFixedLevelBlocksImpl<8, true>(query_point, query_extended_point, K,
                                         final_result, table, step);
      return;
    case 9:
      queryFixedLevelBlocksImpl<9, true>(query_point, query_extended_point, K,
                                         final_result, table, step);
      return;
    case 10:
      queryFixedLevelBlocksImpl<10, true>(query_point, query_extended_point, K,
                                          final_result, table, step);
      return;
    case 11:
      queryFixedLevelBlocksImpl<11, true>(query_point, query_extended_point, K,
                                          final_result, table, step);
      return;
    case 12:
      queryFixedLevelBlocksImpl<12, true>(query_point, query_extended_point, K,
                                          final_result, table, step);
      return;
    case 13:
      queryFixedLevelBlocksImpl<13, true>(query_point, query_extended_point, K,
                                          final_result, table, step);
      return;
    case 14:
      queryFixedLevelBlocksImpl<14, true>(query_point, query_extended_point, K,
                                          final_result, table, step);
      return;
    case 15:
      queryFixedLevelBlocksImpl<15, true>(query_point, query_extended_point, K,
                                          final_result, table, step);
      return;
    case 16:
      queryFixedLevelBlocksImpl<16, true>(query_point, query_extended_point, K,
                                          final_result, table, step);
      return;
    default:
      break;
    }
  }
  if (options_.metric == PAGMetric::MaximumInnerProduct) {
    switch (level_blocks) {
    case 1:
      queryFixedLevelBlocksImpl<1, false, true>(query_point, query_extended_point, K,
                                                final_result, table, step);
      return;
    case 2:
      queryFixedLevelBlocksImpl<2, false, true>(query_point, query_extended_point, K,
                                                final_result, table, step);
      return;
    case 3:
      queryFixedLevelBlocksImpl<3, false, true>(query_point, query_extended_point, K,
                                                final_result, table, step);
      return;
    case 4:
      queryFixedLevelBlocksImpl<4, false, true>(query_point, query_extended_point, K,
                                                final_result, table, step);
      return;
    case 5:
      queryFixedLevelBlocksImpl<5, false, true>(query_point, query_extended_point, K,
                                                final_result, table, step);
      return;
    case 6:
      queryFixedLevelBlocksImpl<6, false, true>(query_point, query_extended_point, K,
                                                final_result, table, step);
      return;
    case 7:
      queryFixedLevelBlocksImpl<7, false, true>(query_point, query_extended_point, K,
                                                final_result, table, step);
      return;
    case 8:
      queryFixedLevelBlocksImpl<8, false, true>(query_point, query_extended_point, K,
                                                final_result, table, step);
      return;
    case 9:
      queryFixedLevelBlocksImpl<9, false, true>(query_point, query_extended_point, K,
                                                final_result, table, step);
      return;
    case 10:
      queryFixedLevelBlocksImpl<10, false, true>(query_point, query_extended_point, K,
                                                 final_result, table, step);
      return;
    case 11:
      queryFixedLevelBlocksImpl<11, false, true>(query_point, query_extended_point, K,
                                                 final_result, table, step);
      return;
    case 12:
      queryFixedLevelBlocksImpl<12, false, true>(query_point, query_extended_point, K,
                                                 final_result, table, step);
      return;
    case 13:
      queryFixedLevelBlocksImpl<13, false, true>(query_point, query_extended_point, K,
                                                 final_result, table, step);
      return;
    case 14:
      queryFixedLevelBlocksImpl<14, false, true>(query_point, query_extended_point, K,
                                                 final_result, table, step);
      return;
    case 15:
      queryFixedLevelBlocksImpl<15, false, true>(query_point, query_extended_point, K,
                                                 final_result, table, step);
      return;
    case 16:
      queryFixedLevelBlocksImpl<16, false, true>(query_point, query_extended_point, K,
                                                 final_result, table, step);
      return;
    default:
      throw std::runtime_error(
          "MIPS currently requires projection_levels/8 in [1, 16]");
    }
  }
  switch (level_blocks) {
  case 1:
    queryLevelBlocks1(query_point, query_extended_point, K, final_result, table,
                  step);
    return;
  case 2:
    queryLevelBlocks2(query_point, query_extended_point, K, final_result, table,
                  step);
    return;
  case 3:
    queryFixedLevelBlocksImpl<3>(query_point, query_extended_point, K, final_result,
                            table, step);
    return;
  case 4:
    queryLevelBlocks4(query_point, query_extended_point, K, final_result, table,
                  step);
    return;
  case 5:
    queryFixedLevelBlocksImpl<5>(query_point, query_extended_point, K, final_result,
                            table, step);
    return;
  case 6:
    queryFixedLevelBlocksImpl<6>(query_point, query_extended_point, K, final_result,
                            table, step);
    return;
  case 7:
    queryFixedLevelBlocksImpl<7>(query_point, query_extended_point, K, final_result,
                            table, step);
    return;
  case 8:
    queryFixedLevelBlocksImpl<8>(query_point, query_extended_point, K, final_result,
                            table, step);
    return;
  case 9:
    queryFixedLevelBlocksImpl<9>(query_point, query_extended_point, K, final_result,
                            table, step);
    return;
  case 10:
    queryFixedLevelBlocksImpl<10>(query_point, query_extended_point, K, final_result,
                             table, step);
    return;
  case 11:
    queryFixedLevelBlocksImpl<11>(query_point, query_extended_point, K, final_result,
                             table, step);
    return;
  case 12:
    queryFixedLevelBlocksImpl<12>(query_point, query_extended_point, K, final_result,
                             table, step);
    return;
  case 13:
    queryFixedLevelBlocksImpl<13>(query_point, query_extended_point, K, final_result,
                             table, step);
    return;
  case 14:
    queryFixedLevelBlocksImpl<14>(query_point, query_extended_point, K, final_result,
                             table, step);
    return;
  case 15:
    queryFixedLevelBlocksImpl<15>(query_point, query_extended_point, K, final_result,
                             table, step);
    return;
  case 16:
    queryFixedLevelBlocksImpl<16>(query_point, query_extended_point, K, final_result,
                             table, step);
    return;
  default:
    queryRuntimeLevelBlocksImpl(query_point, query_extended_point, K, final_result,
                           table, step);
    return;
  }
}
