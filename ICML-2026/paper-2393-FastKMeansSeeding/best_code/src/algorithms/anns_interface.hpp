#pragma once

#include <vector>
#include <memory>
#include <limits>
#include <cmath>
#include <algorithm>
#include <random>

#include "external/hnswlib/hnswlib.h"

namespace fastcluster {

/* ============================================================
   Abstract ANNS Interface

   All ANNS methods must implement this interface for use in QKMEANS
   ============================================================ */
class ANNSInterface {
public:
    virtual ~ANNSInterface() = default;

    // Insert a new center point
    virtual void insert(size_t center_idx, const float* center_ptr) = 0;

    // Query for squared distance to nearest center
    virtual float query(const float* point) const = 0;

    // Get number of centers inserted
    virtual size_t num_centers() const = 0;

    // Get method name for logging
    virtual const char* name() const = 0;
};

/* ============================================================
   1. HNSW-based ANNS (using hnswlib)

   Pros: Excellent recall, fast queries
   Cons: Higher memory, slower insertion
   ============================================================ */
class HNSW_ANNS : public ANNSInterface {
public:
    HNSW_ANNS(size_t dim, size_t max_centers, size_t M = 16,
              size_t ef_construction = 200, size_t ef_search = 50)
        : dim_(dim), num_centers_(0) {

        space_ = std::make_unique<hnswlib::L2Space>(dim);
        index_ = std::make_unique<hnswlib::HierarchicalNSW<float>>(
            space_.get(), max_centers, M, ef_construction);
        index_->setEf(ef_search);
    }

    void insert(size_t center_idx, const float* center_ptr) override {
        index_->addPoint(center_ptr, num_centers_);
        center_indices_.push_back(center_idx);
        num_centers_++;
    }

    float query(const float* point) const override {
        if (num_centers_ == 0) {
            return std::numeric_limits<float>::max();
        }
        auto result = index_->searchKnn(point, 1);
        if (result.empty()) {
            return std::numeric_limits<float>::max();
        }
        return result.top().first;  // hnswlib returns squared L2 distance
    }

    size_t num_centers() const override { return num_centers_; }
    const char* name() const override { return "HNSW"; }

    void setEf(size_t ef) { index_->setEf(ef); }

private:
    size_t dim_;
    size_t num_centers_;
    std::unique_ptr<hnswlib::L2Space> space_;
    std::unique_ptr<hnswlib::HierarchicalNSW<float>> index_;
    std::vector<size_t> center_indices_;
};

/* ============================================================
   2. Brute Force ANNS (exact nearest neighbor)

   Pros: Exact results, simple
   Cons: O(k) per query
   ============================================================ */
class BruteForce_ANNS : public ANNSInterface {
public:
    BruteForce_ANNS(size_t dim, size_t max_centers)
        : dim_(dim) {
        centers_.reserve(max_centers * dim);
    }

    void insert(size_t center_idx, const float* center_ptr) override {
        for (size_t i = 0; i < dim_; ++i) {
            centers_.push_back(center_ptr[i]);
        }
        center_indices_.push_back(center_idx);
    }

    float query(const float* point) const override {
        size_t k = num_centers();
        if (k == 0) {
            return std::numeric_limits<float>::max();
        }

        float min_dist = std::numeric_limits<float>::max();
        for (size_t c = 0; c < k; ++c) {
            float dist = 0.0f;
            const float* center = &centers_[c * dim_];
            for (size_t j = 0; j < dim_; ++j) {
                float diff = point[j] - center[j];
                dist += diff * diff;
            }
            min_dist = std::min(min_dist, dist);
        }
        return min_dist;
    }

    size_t num_centers() const override { return center_indices_.size(); }
    const char* name() const override { return "BruteForce"; }

private:
    size_t dim_;
    std::vector<float> centers_;
    std::vector<size_t> center_indices_;
};

/* ============================================================
   3. Random Projection Tree ANNS (simplified Annoy-style)

   Uses multiple random projection trees for approximate NN search
   Pros: Good for medium dimensions, simple
   Cons: Lower recall than HNSW for high-dim
   ============================================================ */
class RPTree_ANNS : public ANNSInterface {
public:
    RPTree_ANNS(size_t dim, size_t max_centers, size_t num_trees = 8, uint64_t seed = 42)
        : dim_(dim), num_trees_(num_trees), rng_(seed) {

        centers_.reserve(max_centers * dim);

        // Generate random projection vectors for each tree
        std::normal_distribution<float> gauss(0.0f, 1.0f);
        projections_.resize(num_trees_ * dim);
        for (size_t i = 0; i < num_trees_ * dim; ++i) {
            projections_[i] = gauss(rng_);
        }

        // Initialize trees (simple sorted list for now)
        tree_indices_.resize(num_trees_);
        tree_projections_.resize(num_trees_);
    }

    void insert(size_t center_idx, const float* center_ptr) override {
        // Store center
        size_t c = num_centers();
        for (size_t i = 0; i < dim_; ++i) {
            centers_.push_back(center_ptr[i]);
        }
        center_indices_.push_back(center_idx);

        // Insert into each tree
        for (size_t t = 0; t < num_trees_; ++t) {
            float proj = project(center_ptr, t);

            // Insert maintaining sorted order
            auto& indices = tree_indices_[t];
            auto& projs = tree_projections_[t];

            auto it = std::lower_bound(projs.begin(), projs.end(), proj);
            size_t pos = it - projs.begin();

            indices.insert(indices.begin() + pos, c);
            projs.insert(projs.begin() + pos, proj);
        }
    }

    float query(const float* point) const override {
        size_t k = num_centers();
        if (k == 0) {
            return std::numeric_limits<float>::max();
        }

        // Number of candidates to check per tree
        size_t num_candidates = std::min(size_t(10), k);

        float min_dist = std::numeric_limits<float>::max();

        // Search each tree
        for (size_t t = 0; t < num_trees_; ++t) {
            float proj = project(point, t);

            const auto& projs = tree_projections_[t];
            const auto& indices = tree_indices_[t];

            // Binary search for closest projection
            auto it = std::lower_bound(projs.begin(), projs.end(), proj);
            size_t pos = it - projs.begin();

            // Check candidates around this position
            size_t start = (pos > num_candidates/2) ? pos - num_candidates/2 : 0;
            size_t end = std::min(start + num_candidates, k);

            for (size_t i = start; i < end; ++i) {
                size_t c = indices[i];
                float dist = compute_dist(point, c);
                min_dist = std::min(min_dist, dist);
            }
        }

        return min_dist;
    }

    size_t num_centers() const override { return center_indices_.size(); }
    const char* name() const override { return "RPTree"; }

private:
    float project(const float* point, size_t tree_idx) const {
        float sum = 0.0f;
        const float* proj = &projections_[tree_idx * dim_];
        for (size_t i = 0; i < dim_; ++i) {
            sum += point[i] * proj[i];
        }
        return sum;
    }

    float compute_dist(const float* point, size_t center_idx) const {
        float dist = 0.0f;
        const float* center = &centers_[center_idx * dim_];
        for (size_t j = 0; j < dim_; ++j) {
            float diff = point[j] - center[j];
            dist += diff * diff;
        }
        return dist;
    }

    size_t dim_;
    size_t num_trees_;
    std::mt19937_64 rng_;
    std::vector<float> centers_;
    std::vector<size_t> center_indices_;
    std::vector<float> projections_;
    std::vector<std::vector<size_t>> tree_indices_;
    std::vector<std::vector<float>> tree_projections_;
};

/* ============================================================
   4. LSH-based ANNS (E2LSH style for L2 distance)

   Uses random hyperplanes for hashing
   Pros: Theoretical guarantees, simple
   Cons: Lower recall than HNSW, needs tuning
   ============================================================ */
class LSH_ANNS : public ANNSInterface {
public:
    LSH_ANNS(size_t dim, size_t max_centers, size_t num_tables = 8,
             size_t num_hashes = 16, float w = 4.0f, uint64_t seed = 42)
        : dim_(dim), num_tables_(num_tables), num_hashes_(num_hashes), w_(w), rng_(seed) {

        centers_.reserve(max_centers * dim);

        // Generate hash functions for each table
        // h(x) = floor((a·x + b) / w) where a ~ N(0,1)^d, b ~ U(0,w)
        std::normal_distribution<float> gauss(0.0f, 1.0f);
        std::uniform_real_distribution<float> unif_b(0.0f, w);

        hash_vectors_.resize(num_tables_ * num_hashes_ * dim);
        hash_offsets_.resize(num_tables_ * num_hashes_);

        for (size_t t = 0; t < num_tables_; ++t) {
            for (size_t h = 0; h < num_hashes_; ++h) {
                size_t base = (t * num_hashes_ + h) * dim;
                for (size_t i = 0; i < dim; ++i) {
                    hash_vectors_[base + i] = gauss(rng_);
                }
                hash_offsets_[t * num_hashes_ + h] = unif_b(rng_);
            }
        }

        // Initialize hash tables
        hash_tables_.resize(num_tables_);
    }

    void insert(size_t center_idx, const float* center_ptr) override {
        size_t c = num_centers();
        for (size_t i = 0; i < dim_; ++i) {
            centers_.push_back(center_ptr[i]);
        }
        center_indices_.push_back(center_idx);

        // Insert into each hash table
        for (size_t t = 0; t < num_tables_; ++t) {
            size_t hash = compute_hash(center_ptr, t);
            hash_tables_[t][hash].push_back(c);
        }
    }

    float query(const float* point) const override {
        size_t k = num_centers();
        if (k == 0) {
            return std::numeric_limits<float>::max();
        }

        float min_dist = std::numeric_limits<float>::max();

        // Query each hash table
        for (size_t t = 0; t < num_tables_; ++t) {
            size_t hash = compute_hash(point, t);

            auto it = hash_tables_[t].find(hash);
            if (it != hash_tables_[t].end()) {
                for (size_t c : it->second) {
                    float dist = compute_dist(point, c);
                    min_dist = std::min(min_dist, dist);
                }
            }
        }

        // If no candidates found in hash buckets, fall back to checking all
        if (min_dist == std::numeric_limits<float>::max()) {
            for (size_t c = 0; c < k; ++c) {
                float dist = compute_dist(point, c);
                min_dist = std::min(min_dist, dist);
            }
        }

        return min_dist;
    }

    size_t num_centers() const override { return center_indices_.size(); }
    const char* name() const override { return "LSH"; }

private:
    size_t compute_hash(const float* point, size_t table_idx) const {
        // Concatenate hash values from each hash function
        size_t hash = 0;
        for (size_t h = 0; h < num_hashes_; ++h) {
            float proj = 0.0f;
            const float* a = &hash_vectors_[(table_idx * num_hashes_ + h) * dim_];
            for (size_t i = 0; i < dim_; ++i) {
                proj += point[i] * a[i];
            }
            proj += hash_offsets_[table_idx * num_hashes_ + h];
            int bucket = static_cast<int>(std::floor(proj / w_));

            // Combine into single hash
            hash = hash * 31 + static_cast<size_t>(bucket + 1000000);  // offset to handle negatives
        }
        return hash;
    }

    float compute_dist(const float* point, size_t center_idx) const {
        float dist = 0.0f;
        const float* center = &centers_[center_idx * dim_];
        for (size_t j = 0; j < dim_; ++j) {
            float diff = point[j] - center[j];
            dist += diff * diff;
        }
        return dist;
    }

    size_t dim_;
    size_t num_tables_;
    size_t num_hashes_;
    float w_;
    std::mt19937_64 rng_;
    std::vector<float> centers_;
    std::vector<size_t> center_indices_;
    std::vector<float> hash_vectors_;
    std::vector<float> hash_offsets_;
    std::vector<std::unordered_map<size_t, std::vector<size_t>>> hash_tables_;
};

/* ============================================================
   5. IVF (Inverted File Index) ANNS

   Simple partition-based method
   Pros: Fast with good partitioning
   Cons: Requires good initial partition, lower recall
   ============================================================ */
class IVF_ANNS : public ANNSInterface {
public:
    IVF_ANNS(size_t dim, size_t max_centers, size_t num_partitions = 16,
             size_t nprobe = 4, uint64_t seed = 42)
        : dim_(dim), num_partitions_(num_partitions), nprobe_(nprobe), rng_(seed) {

        centers_.reserve(max_centers * dim);

        // Initialize random partition centroids (will be updated as centers are added)
        std::normal_distribution<float> gauss(0.0f, 1.0f);
        partition_centroids_.resize(num_partitions_ * dim);
        for (size_t i = 0; i < num_partitions_ * dim; ++i) {
            partition_centroids_[i] = gauss(rng_);
        }

        partition_lists_.resize(num_partitions_);
    }

    void insert(size_t center_idx, const float* center_ptr) override {
        size_t c = num_centers();
        for (size_t i = 0; i < dim_; ++i) {
            centers_.push_back(center_ptr[i]);
        }
        center_indices_.push_back(center_idx);

        // Find nearest partition and insert
        size_t best_partition = find_nearest_partition(center_ptr);
        partition_lists_[best_partition].push_back(c);
    }

    float query(const float* point) const override {
        size_t k = num_centers();
        if (k == 0) {
            return std::numeric_limits<float>::max();
        }

        // Find nprobe nearest partitions
        std::vector<std::pair<float, size_t>> partition_dists(num_partitions_);
        for (size_t p = 0; p < num_partitions_; ++p) {
            float dist = 0.0f;
            const float* centroid = &partition_centroids_[p * dim_];
            for (size_t j = 0; j < dim_; ++j) {
                float diff = point[j] - centroid[j];
                dist += diff * diff;
            }
            partition_dists[p] = {dist, p};
        }

        std::partial_sort(partition_dists.begin(),
                         partition_dists.begin() + std::min(nprobe_, num_partitions_),
                         partition_dists.end());

        // Search nprobe nearest partitions
        float min_dist = std::numeric_limits<float>::max();
        for (size_t i = 0; i < std::min(nprobe_, num_partitions_); ++i) {
            size_t p = partition_dists[i].second;
            for (size_t c : partition_lists_[p]) {
                float dist = compute_dist(point, c);
                min_dist = std::min(min_dist, dist);
            }
        }

        return min_dist;
    }

    size_t num_centers() const override { return center_indices_.size(); }
    const char* name() const override { return "IVF"; }

private:
    size_t find_nearest_partition(const float* point) const {
        float min_dist = std::numeric_limits<float>::max();
        size_t best = 0;
        for (size_t p = 0; p < num_partitions_; ++p) {
            float dist = 0.0f;
            const float* centroid = &partition_centroids_[p * dim_];
            for (size_t j = 0; j < dim_; ++j) {
                float diff = point[j] - centroid[j];
                dist += diff * diff;
            }
            if (dist < min_dist) {
                min_dist = dist;
                best = p;
            }
        }
        return best;
    }

    float compute_dist(const float* point, size_t center_idx) const {
        float dist = 0.0f;
        const float* center = &centers_[center_idx * dim_];
        for (size_t j = 0; j < dim_; ++j) {
            float diff = point[j] - center[j];
            dist += diff * diff;
        }
        return dist;
    }

    size_t dim_;
    size_t num_partitions_;
    size_t nprobe_;
    std::mt19937_64 rng_;
    std::vector<float> centers_;
    std::vector<size_t> center_indices_;
    std::vector<float> partition_centroids_;
    std::vector<std::vector<size_t>> partition_lists_;
};

/* ============================================================
   ANNS Factory - Create ANNS instance by name
   ============================================================ */
enum class ANNSMethod {
    HNSW,
    BruteForce,
    RPTree,
    LSH,
    IVF
};

inline std::unique_ptr<ANNSInterface> create_anns(
    ANNSMethod method, size_t dim, size_t max_centers,
    size_t ef = 50, size_t M = 16, uint64_t seed = 42) {

    switch (method) {
        case ANNSMethod::HNSW:
            return std::make_unique<HNSW_ANNS>(dim, max_centers, M, 200, ef);
        case ANNSMethod::BruteForce:
            return std::make_unique<BruteForce_ANNS>(dim, max_centers);
        case ANNSMethod::RPTree:
            return std::make_unique<RPTree_ANNS>(dim, max_centers, 8, seed);
        case ANNSMethod::LSH:
            return std::make_unique<LSH_ANNS>(dim, max_centers, 8, 16, 4.0f, seed);
        case ANNSMethod::IVF:
            return std::make_unique<IVF_ANNS>(dim, max_centers, 16, 4, seed);
        default:
            return std::make_unique<HNSW_ANNS>(dim, max_centers, M, 200, ef);
    }
}

inline ANNSMethod anns_method_from_string(const std::string& name) {
    if (name == "hnsw" || name == "HNSW") return ANNSMethod::HNSW;
    if (name == "brute" || name == "BruteForce" || name == "bruteforce") return ANNSMethod::BruteForce;
    if (name == "rptree" || name == "RPTree") return ANNSMethod::RPTree;
    if (name == "lsh" || name == "LSH") return ANNSMethod::LSH;
    if (name == "ivf" || name == "IVF") return ANNSMethod::IVF;
    return ANNSMethod::HNSW;  // default
}

inline const char* anns_method_to_string(ANNSMethod method) {
    switch (method) {
        case ANNSMethod::HNSW: return "HNSW";
        case ANNSMethod::BruteForce: return "BruteForce";
        case ANNSMethod::RPTree: return "RPTree";
        case ANNSMethod::LSH: return "LSH";
        case ANNSMethod::IVF: return "IVF";
        default: return "HNSW";
    }
}

} // namespace fastcluster
