#pragma once
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <immintrin.h>
#include <vector>

inline void write_low4(unsigned char *dst, unsigned char val) {
  *dst = (*dst & 0xF0) | (val & 0x0F);
}

inline void write_high4(unsigned char *dst, unsigned char val) {
  *dst = (*dst & 0x0F) | ((val & 0x0F) << 4);
}

inline unsigned char get_low4(unsigned char val) { return val & 0x0F; }

inline unsigned char get_high4(unsigned char val) { return (val >> 4) & 0x0F; }

static inline __attribute__((always_inline)) void
add_level_pair_scores_16bytes(const __m128i pair_bytes,
                              const __m128i nibble_mask, const float *t_ptr,
                              __m512 &sum) {
  const __m128i odd_idx = _mm_and_si128(pair_bytes, nibble_mask);
  const __m128i even_idx =
      _mm_and_si128(_mm_srli_epi64(pair_bytes, 4), nibble_mask);

  sum = _mm512_add_ps(sum, _mm512_permutexvar_ps(_mm512_cvtepu8_epi32(even_idx),
                                                 _mm512_load_ps(t_ptr + 0)));
  sum = _mm512_add_ps(sum, _mm512_permutexvar_ps(_mm512_cvtepu8_epi32(odd_idx),
                                                 _mm512_load_ps(t_ptr + 16)));
}

static inline __attribute__((always_inline)) void
add_8level_scores_64bytes(const unsigned char *ptr, const __m128i nibble_mask,
                          const float *t_ptr, __m512 &sum0, __m512 &sum1,
                          __m512 &sum2, __m512 &sum3) {
  add_level_pair_scores_16bytes(
      _mm_loadu_si128(reinterpret_cast<const __m128i *>(ptr + 0)), nibble_mask,
      t_ptr + 0, sum0);
  add_level_pair_scores_16bytes(
      _mm_loadu_si128(reinterpret_cast<const __m128i *>(ptr + 16)), nibble_mask,
      t_ptr + 32, sum1);
  add_level_pair_scores_16bytes(
      _mm_loadu_si128(reinterpret_cast<const __m128i *>(ptr + 32)), nibble_mask,
      t_ptr + 64, sum2);
  add_level_pair_scores_16bytes(
      _mm_loadu_si128(reinterpret_cast<const __m128i *>(ptr + 48)), nibble_mask,
      t_ptr + 96, sum3);
}

static inline __attribute__((always_inline)) __m512
eval_pairlayout_block16(const unsigned char *&char_pointer,
                        const __m128i nibble_mask, const float *table,
                        int level_blocks, const __m512 v_center_ip) {
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
  const __m512 v_scale = _mm512_castsi512_ps(v0_i32);

  sum = _mm512_fmadd_ps(v_scale, v_center_ip, sum);

  __m512i v1_i32 = _mm512_cvtepu16_epi32(bf16_1);
  v1_i32 = _mm512_slli_epi32(v1_i32, 16);
  const __m512 v_mul = _mm512_castsi512_ps(v1_i32);

  const __m256i bf16_2 =
      _mm256_loadu_si256(reinterpret_cast<const __m256i *>(bf16_ptr));
  bf16_ptr += 16;

  __m512i v2_i32 = _mm512_cvtepu16_epi32(bf16_2);
  v2_i32 = _mm512_slli_epi32(v2_i32, 16);
  const __m512 v_add = _mm512_castsi512_ps(v2_i32);

  sum = _mm512_fmadd_ps(sum, v_mul, v_add);
  char_pointer = reinterpret_cast<const unsigned char *>(bf16_ptr);
  return sum;
}

static inline __attribute__((always_inline)) void
add_level_pair_scores_16bytes_hoisted(const __m128i pair_bytes,
                                      const __m128i nibble_mask,
                                      const __m512 lut_even,
                                      const __m512 lut_odd, __m512 &sum) {
  const __m128i odd_idx = _mm_and_si128(pair_bytes, nibble_mask);
  const __m128i even_idx =
      _mm_and_si128(_mm_srli_epi64(pair_bytes, 4), nibble_mask);

  const __m512i even_idx32 = _mm512_cvtepu8_epi32(even_idx);
  const __m512i odd_idx32 = _mm512_cvtepu8_epi32(odd_idx);

  sum = _mm512_add_ps(sum, _mm512_permutexvar_ps(even_idx32, lut_even));
  sum = _mm512_add_ps(sum, _mm512_permutexvar_ps(odd_idx32, lut_odd));
}

static inline __attribute__((always_inline)) __m512
finalize_pairlayout_block16(const unsigned char *&char_pointer, __m512 sum,
                            const __m512 v_center_ip) {
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

  const __m256i bf16_2 =
      _mm256_loadu_si256(reinterpret_cast<const __m256i *>(bf16_ptr));
  bf16_ptr += 16;

  __m512i v2_i32 = _mm512_cvtepu16_epi32(bf16_2);
  v2_i32 = _mm512_slli_epi32(v2_i32, 16);
  const __m512 v_add = _mm512_castsi512_ps(v2_i32);

  sum = _mm512_fmadd_ps(sum, v_mul, v_add);
  char_pointer = reinterpret_cast<const unsigned char *>(bf16_ptr);
  return sum;
}

static inline __attribute__((always_inline)) __m512
eval_pairlayout_block16_hoisted_lb1(const unsigned char *&char_pointer,
                                    const __m128i nibble_mask,
                                    const __m512 lut00, const __m512 lut01,
                                    const __m512 lut02, const __m512 lut03,
                                    const __m512 lut04, const __m512 lut05,
                                    const __m512 lut06, const __m512 lut07,
                                    const __m512 v_center_ip) {
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
  return finalize_pairlayout_block16(char_pointer, sum, v_center_ip);
}

static inline __attribute__((always_inline)) __m512
eval_pairlayout_block16_hoisted_lb2(
    const unsigned char *&char_pointer, const __m128i nibble_mask,
    const __m512 lut00, const __m512 lut01, const __m512 lut02,
    const __m512 lut03, const __m512 lut04, const __m512 lut05,
    const __m512 lut06, const __m512 lut07, const __m512 lut08,
    const __m512 lut09, const __m512 lut10, const __m512 lut11,
    const __m512 lut12, const __m512 lut13, const __m512 lut14,
    const __m512 lut15, const __m512 v_center_ip) {
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

  add_level_pair_scores_16bytes_hoisted(
      _mm_loadu_si128(reinterpret_cast<const __m128i *>(char_pointer + 64)),
      nibble_mask, lut08, lut09, sum0);
  add_level_pair_scores_16bytes_hoisted(
      _mm_loadu_si128(reinterpret_cast<const __m128i *>(char_pointer + 80)),
      nibble_mask, lut10, lut11, sum1);
  add_level_pair_scores_16bytes_hoisted(
      _mm_loadu_si128(reinterpret_cast<const __m128i *>(char_pointer + 96)),
      nibble_mask, lut12, lut13, sum2);
  add_level_pair_scores_16bytes_hoisted(
      _mm_loadu_si128(reinterpret_cast<const __m128i *>(char_pointer + 112)),
      nibble_mask, lut14, lut15, sum3);
  char_pointer += 128;

  const __m512 sum =
      _mm512_add_ps(_mm512_add_ps(sum0, sum1), _mm512_add_ps(sum2, sum3));
  return finalize_pairlayout_block16(char_pointer, sum, v_center_ip);
}

static inline __attribute__((always_inline)) __m512
eval_pairlayout_block16_unrolled_lb4(const unsigned char *&char_pointer,
                                     const __m128i nibble_mask,
                                     const float *table,
                                     const __m512 v_center_ip) {
  __m512 sum0 = _mm512_setzero_ps();
  __m512 sum1 = _mm512_setzero_ps();
  __m512 sum2 = _mm512_setzero_ps();
  __m512 sum3 = _mm512_setzero_ps();

  add_8level_scores_64bytes(char_pointer + 0, nibble_mask, table + 0, sum0,
                            sum1, sum2, sum3);
  add_8level_scores_64bytes(char_pointer + 64, nibble_mask, table + 128, sum0,
                            sum1, sum2, sum3);
  add_8level_scores_64bytes(char_pointer + 128, nibble_mask, table + 256, sum0,
                            sum1, sum2, sum3);
  add_8level_scores_64bytes(char_pointer + 192, nibble_mask, table + 384, sum0,
                            sum1, sum2, sum3);
  char_pointer += 256;

  const __m512 sum =
      _mm512_add_ps(_mm512_add_ps(sum0, sum1), _mm512_add_ps(sum2, sum3));
  return finalize_pairlayout_block16(char_pointer, sum, v_center_ip);
}

static inline __attribute__((always_inline)) void
eval_pairlayout_block16_store_build(const unsigned char *&char_pointer,
                                    const __m128i nibble_mask,
                                    const float *table, int level_blocks,
                                    const __m512 v_center_ip, float *out_pred) {
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

  const __m256i bf16_0 =
      _mm256_loadu_si256(reinterpret_cast<const __m256i *>(bf16_ptr));
  bf16_ptr += 16;
  const __m256i bf16_1 =
      _mm256_loadu_si256(reinterpret_cast<const __m256i *>(bf16_ptr));
  bf16_ptr += 16;
  const __m256i bf16_2 =
      _mm256_loadu_si256(reinterpret_cast<const __m256i *>(bf16_ptr));
  bf16_ptr += 16;

  __m512i v0_i32 = _mm512_cvtepu16_epi32(bf16_0);
  v0_i32 = _mm512_slli_epi32(v0_i32, 16);
  const __m512 v_scale = _mm512_castsi512_ps(v0_i32);
  sum = _mm512_fmadd_ps(v_scale, v_center_ip, sum);

  __m512i v1_i32 = _mm512_cvtepu16_epi32(bf16_1);
  v1_i32 = _mm512_slli_epi32(v1_i32, 16);
  const __m512 v_mul = _mm512_castsi512_ps(v1_i32);

  __m512i v2_i32 = _mm512_cvtepu16_epi32(bf16_2);
  v2_i32 = _mm512_slli_epi32(v2_i32, 16);
  const __m512 v_add = _mm512_castsi512_ps(v2_i32);

  sum = _mm512_fmadd_ps(sum, v_mul, v_add);
  _mm512_store_ps(out_pred, sum);
  char_pointer = reinterpret_cast<const unsigned char *>(bf16_ptr);
}

static inline int InsertIntoPool(Neighbor *addr, unsigned K,
                                 const Neighbor &nn) {
  float d = nn.distance;

  if (d < addr[0].distance) {
    memmove(addr + 1, addr, K * sizeof(Neighbor));
    addr[0] = nn;
    return 0;
  }

  unsigned left = 0, right = K - 1;
  while (right - left > 1) {
    unsigned mid = (left + right) >> 1;
    if (addr[mid].distance > d)
      right = mid;
    else
      left = mid;
  }

  memmove(addr + right + 1, addr + right, (K - right) * sizeof(Neighbor));

  addr[right] = nn;

  return right;
}

static inline int InsertIntoPoolIndex(NeighborIndex *addr, unsigned K,
                                      const NeighborIndex &nn) {
  float d = nn.distance;
  unsigned idx = K;
  unsigned left = 0;
  unsigned right = K;

  while (left < right) {
    unsigned mid = left + (right - left) / 2;
    if (addr[mid].distance < d) {
      left = mid + 1;
    } else {
      right = mid;
    }
  }
  idx = left;

  size_t num_elements_to_move = (K - 1) - idx;

  if (num_elements_to_move > 0) {
    memmove(addr + idx + 1, addr + idx,
            num_elements_to_move * sizeof(NeighborIndex));
  }

  addr[idx] = nn;

  return idx;
}

static inline int InsertIntoPool2(Neighbor *addr, unsigned K,
                                  const Neighbor &nn) {
  float d = nn.distance;

  unsigned idx = K;
  unsigned left = 0;
  unsigned right = K;

  while (left < right) {
    unsigned mid = left + (right - left) / 2;
    if (addr[mid].distance < d) {
      left = mid + 1;
    } else {
      right = mid;
    }
  }
  idx = left;

  size_t num_elements_to_move = (K - 1) - idx;

  if (num_elements_to_move > 0) {
    memmove(addr + idx + 1, addr + idx,
            num_elements_to_move * sizeof(Neighbor));
  }

  addr[idx] = nn;

  return idx;
}

struct PESCandidate {
  unsigned int id;
  float distance;
};

const size_t MAX_NEIGHBORS = 64;

void insertPESCandidate(std::vector<std::vector<PESCandidate>> &pes_candidates,
                        size_t index, unsigned int id, float distance) {
  PESCandidate newNeighbor{id, distance};
  auto &bucket = pes_candidates[index];
  auto it = std::lower_bound(bucket.begin(), bucket.end(), newNeighbor,
                             [](const PESCandidate &a, const PESCandidate &b) {
                               return a.distance < b.distance;
                             });
  bucket.insert(it, newNeighbor);
  if (bucket.size() > MAX_NEIGHBORS) {
    bucket.pop_back();
  }
}

void deletePESCandidate(std::vector<std::vector<PESCandidate>> &pes_candidates,
                        size_t index, unsigned int id) {
  if (index >= pes_candidates.size())
    return;

  auto &bucket = pes_candidates[index];
  auto it = std::find_if(bucket.begin(), bucket.end(),
                         [id](const PESCandidate &n) { return n.id == id; });

  if (it != bucket.end()) {
    bucket.erase(it);
  }
}

template <typename Candidate>
static inline __attribute__((always_inline)) void
PushTFBRing(std::vector<Candidate> &ring, int &head, int &size, int capacity,
            const Candidate &candidate) {
  ring[head] = candidate;
  if (size < capacity) {
    size++;
  }
  head = (head + 1) % capacity;
}

template <typename Candidate>
static inline __attribute__((always_inline)) void
RefillTFBWorkingSet(std::vector<Candidate> &working_set, int working_capacity,
                    std::vector<Candidate> &rejected_ring, int &rejected_head,
                    int &rejected_size, std::vector<Candidate> &ejected_ring,
                    int &ejected_head, int &ejected_size,
                    std::vector<Candidate> &ejected_buffer, int &working_size) {
  std::sort(rejected_ring.begin(), rejected_ring.begin() + rejected_size);
  std::rotate(ejected_ring.begin(), ejected_ring.begin() + ejected_head,
              ejected_ring.begin() + ejected_size);
  std::reverse(ejected_ring.begin(), ejected_ring.begin() + ejected_size);

  int rejected_pos = 0;
  int ejected_pos = 0;
  int out_pos = 0;
  while (out_pos < working_capacity && ejected_pos < ejected_size &&
         rejected_pos < rejected_size) {
    if (ejected_ring[ejected_pos].distance <
        rejected_ring[rejected_pos].distance) {
      working_set[out_pos++] = ejected_ring[ejected_pos++];
    } else {
      working_set[out_pos++] = rejected_ring[rejected_pos++];
    }
  }
  while (out_pos < working_capacity && ejected_pos < ejected_size) {
    working_set[out_pos++] = ejected_ring[ejected_pos++];
  }
  while (out_pos < working_capacity && rejected_pos < rejected_size) {
    working_set[out_pos++] = rejected_ring[rejected_pos++];
  }
  working_size = out_pos;

  int buffer_size = 0;
  int ejected_tail = ejected_size - 1;
  int rejected_tail = rejected_size - 1;
  while (buffer_size < working_capacity && ejected_tail >= ejected_pos &&
         rejected_tail >= rejected_pos) {
    if (ejected_ring[ejected_tail].distance >
        rejected_ring[rejected_tail].distance) {
      ejected_buffer[buffer_size++] = ejected_ring[ejected_tail--];
    } else {
      ejected_buffer[buffer_size++] = rejected_ring[rejected_tail--];
    }
  }
  while (buffer_size < working_capacity && ejected_tail >= ejected_pos) {
    ejected_buffer[buffer_size++] = ejected_ring[ejected_tail--];
  }
  while (buffer_size < working_capacity && rejected_tail >= rejected_pos) {
    ejected_buffer[buffer_size++] = rejected_ring[rejected_tail--];
  }

  ejected_ring.swap(ejected_buffer);
  ejected_head = buffer_size % working_capacity;
  ejected_size = buffer_size;
  rejected_head = 0;
  rejected_size = 0;
}
