#pragma once

#include "src/core/dataset.hpp"
#include "src/algorithms/prone.hpp"  // For D2SegmentTree
#include "external/hnswlib/hnswlib.h"

#include <random>
#include <vector>
#include <cmath>
#include <cassert>
#include <limits>
#include <algorithm>
#include <numeric>
#include <memory>
#include <unordered_map>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace fastcluster {

/* ============================================================
   QKMEANS - Fast k-means++ seeding via rejection sampling

   Implements the algorithm from:
   "Faster k-means Seeding Under The Manifold Hypothesis"

   Key idea: Use rejection sampling with proposal distribution
   κ(x|C) = (||x||² + ||c₁||²) / (||X||²_F + n·||c₁||²)
   to approximate D² sampling efficiently.
   ============================================================ */

/* ------------------------------------------------------------
   HNSW-based Approximate Nearest Neighbor Structure
   Uses hnswlib for fast approximate nearest neighbor queries
   ------------------------------------------------------------ */
class QKMeansANNS {
public:
    QKMeansANNS(size_t dim, size_t max_centers, size_t M = 16, size_t ef_construction = 200)
        : dim_(dim), num_centers_(0) {

        // Initialize L2 space for squared Euclidean distance
        space_ = std::make_unique<hnswlib::L2Space>(dim);

        // Create HNSW index
        // M: number of connections per layer (higher = better quality, more memory)
        // ef_construction: size of dynamic candidate list during construction
        index_ = std::make_unique<hnswlib::HierarchicalNSW<float>>(
            space_.get(), max_centers, M, ef_construction);

        // Set ef for search (higher = better quality, slower)
        index_->setEf(50);
    }

    void insert(size_t center_idx, const float* center_ptr) {
        index_->addPoint(center_ptr, num_centers_);
        center_indices_.push_back(center_idx);
        num_centers_++;
    }

    // Returns squared distance to approximate nearest center
    float query(const float* point) const {
        if (num_centers_ == 0) {
            return std::numeric_limits<float>::max();
        }

        // Query for 1 nearest neighbor
        auto result = index_->searchKnn(point, 1);

        if (result.empty()) {
            return std::numeric_limits<float>::max();
        }

        // result is a priority queue of (distance, label) pairs
        // hnswlib returns squared L2 distance for L2Space
        return result.top().first;
    }

    size_t num_centers() const { return num_centers_; }

    // Set search quality parameter (higher = better quality, slower)
    void setEf(size_t ef) {
        index_->setEf(ef);
    }

private:
    size_t dim_;
    size_t num_centers_;
    std::unique_ptr<hnswlib::L2Space> space_;
    std::unique_ptr<hnswlib::HierarchicalNSW<float>> index_;
    std::vector<size_t> center_indices_;
};

/* ------------------------------------------------------------
   Main QKMEANS Class
   ------------------------------------------------------------ */
class QKMEANS {
public:
    explicit QKMEANS(uint64_t seed = 42) : rng_(seed) {}

    /* --------------------------------------------------------
       Main interface

       Parameters:
         X      - Input dataset
         k      - Number of centers to select
         ef     - HNSW search quality (higher = better, slower)
                  Typical values: 10-100
         m      - Chain length for rejection sampling
                  Higher = better approximation, slower
         use_jl - Apply Johnson-Lindenstrauss dim reduction
       -------------------------------------------------------- */
    Dataset run(const Dataset& X, size_t k, size_t ef = 50,
                size_t m = 100, bool use_jl = false,
                size_t M = 16, size_t ef_construction = 200) {
        assert(X.size() > 0);
        assert(k > 0 && k <= X.size());

        const size_t n = X.size();
        const size_t d = X.dim();

        // Optional JL dimensionality reduction
        Dataset X_jl;
        std::vector<float> jl_matrix;
        size_t d_work = d;

        if (use_jl && d > 10 * std::log(static_cast<float>(k))) {
            d_work = static_cast<size_t>(std::max(10.0, 10.0 * std::log(static_cast<double>(k))));
            jl_matrix = generate_jl_matrix(d, d_work);
            X_jl = apply_jl_transform(X, jl_matrix, d_work);
        }

        const Dataset& X_work = (use_jl && X_jl.size() > 0) ? X_jl : X;
        const size_t d_actual = X_work.dim();

        // Step 1: Preprocessing - compute squared norms
        PreprocessedData prep = preprocess(X_work);

        // Step 2: Initialize HNSW-based ANNS structure
        // M=16, ef_construction=200 are good defaults for quality
        QKMeansANNS anns(d_actual, k, M, ef_construction);
        anns.setEf(ef);

        // Step 3: Select first center as the max-norm point in a small random subset
        // O(subset_size * d) = negligible cost. High-norm c1 makes kappa more
        // uniform, enabling better exploration (Cost) without the acceptance rate
        // collapse of an extreme outlier (Time)
        std::uniform_int_distribution<size_t> unif_idx(0, n - 1);
        size_t subset_size = std::min(size_t(100), n);
        size_t c1_idx = unif_idx(rng_);
        float max_norm = prep.sq_norms[c1_idx];
        for (size_t i = 1; i < subset_size; ++i) {
            size_t idx = unif_idx(rng_);
            if (prep.sq_norms[idx] > max_norm) {
                max_norm = prep.sq_norms[idx];
                c1_idx = idx;
            }
        }
        float c1_norm_sq = prep.sq_norms[c1_idx];

        // Step 4: Build κ distribution using segment tree for O(log n) sampling
        // κ(x|C) = (||x||² + ||c₁||²) / (||X||²_F + n·||c₁||²)
        // Also precompute proposal_weights in same pass for efficiency
        std::vector<float> kappa_weights(n);
        std::vector<float> proposal_weights(n);
        for (size_t i = 0; i < n; ++i) {
            float norm_plus_c1 = prep.sq_norms[i] + c1_norm_sq;
            kappa_weights[i] = norm_plus_c1;  // Unnormalized (segment tree handles it)
            proposal_weights[i] = 2.0f * norm_plus_c1;
        }

        // Build segment tree for O(log n) sampling from κ distribution
        D2SegmentTree kappa_tree(n);
        kappa_tree.build(kappa_weights);

        // Store selected center indices
        std::vector<size_t> center_indices;
        center_indices.reserve(k);
        center_indices.push_back(c1_idx);

        // Insert first center into ANNS
        anns.insert(c1_idx, X_work.row_ptr(c1_idx));

        // Step 5: Sample remaining k-1 centers using rejection sampling
        std::uniform_real_distribution<float> unif01(0.0f, 1.0f);

        // Maximum iterations: m * ln(k) as per paper
        size_t max_iter = static_cast<size_t>(
            static_cast<float>(m) * std::log(static_cast<float>(k) + 1.0f)
        );
        max_iter = std::max(max_iter, size_t(10));

        for (size_t t = 1; t < k; ++t) {
            // Rejection sampling for center t
            size_t s = unif_idx(rng_);  // Fallback: uniform random sample

            // Candidate dedup cache: avoid redundant HNSW queries
            // when the same point is sampled multiple times from kappa
            std::unordered_map<size_t, float> query_cache;

            for (size_t iter = 0; iter < max_iter; ++iter) {
                // Sample x from κ(·|C) - O(log n) using segment tree
                float u = unif01(rng_) * kappa_tree.total();
                size_t x_idx = kappa_tree.sample(u);

                // Query ANNS for approximate distance to nearest center
                // (with cache to avoid redundant queries for re-sampled candidates)
                float dist_to_center;
                auto cache_it = query_cache.find(x_idx);
                if (cache_it != query_cache.end()) {
                    dist_to_center = cache_it->second;
                } else {
                    dist_to_center = anns.query(X_work.row_ptr(x_idx));
                    query_cache[x_idx] = dist_to_center;
                }

                // Compute acceptance probability
                // r = cost(x, C) / (2 * (||x||² + ||c₁||²))
                float r = dist_to_center / proposal_weights[x_idx];

                // Clamp r to [0, 1] for numerical stability
                r = std::min(r, 1.0f);

                // Rejection test
                if (unif01(rng_) <= r) {
                    s = x_idx;
                    break;  // Accepted!
                }
            }

            center_indices.push_back(s);
            anns.insert(s, X_work.row_ptr(s));
        }

        // Build output Dataset from original (non-JL) coordinates
        Dataset centers(k, d);
        for (size_t i = 0; i < k; ++i) {
            std::copy(
                X.row_ptr(center_indices[i]),
                X.row_ptr(center_indices[i]) + d,
                centers.row_ptr(i)
            );
        }

        return centers;
    }

private:
    std::mt19937_64 rng_;

    /* --------------------------------------------------------
       Preprocessed data for efficient sampling
       -------------------------------------------------------- */
    struct PreprocessedData {
        std::vector<float> sq_norms;   // ||x||² for each point
        float frobenius_sq;             // ||X||²_F = sum of all ||x||²
        size_t n;
    };

    /* --------------------------------------------------------
       Compute squared norms and Frobenius norm
       O(nD) time, parallelized
       -------------------------------------------------------- */
    PreprocessedData preprocess(const Dataset& X) {
        PreprocessedData prep;
        const size_t n = X.size();
        const size_t d = X.dim();

        prep.n = n;
        prep.sq_norms.resize(n);
        prep.frobenius_sq = 0.0f;

        // Compute ||x||² for each point in parallel
        float frob_sum = 0.0f;

        #pragma omp parallel for reduction(+:frob_sum) schedule(static)
        for (size_t i = 0; i < n; ++i) {
            float norm_sq = 0.0f;
            const float* x = X.row_ptr(i);
            for (size_t j = 0; j < d; ++j) {
                norm_sq += x[j] * x[j];
            }
            prep.sq_norms[i] = norm_sq;
            frob_sum += norm_sq;
        }

        prep.frobenius_sq = frob_sum;

        return prep;
    }

    /* --------------------------------------------------------
       Generate random JL projection matrix
       Uses Gaussian entries scaled by 1/sqrt(d_new)
       -------------------------------------------------------- */
    std::vector<float> generate_jl_matrix(size_t d_old, size_t d_new) {
        std::vector<float> matrix(d_old * d_new);
        std::normal_distribution<float> gauss(0.0f, 1.0f / std::sqrt(static_cast<float>(d_new)));

        for (size_t i = 0; i < d_old * d_new; ++i) {
            matrix[i] = gauss(rng_);
        }

        return matrix;
    }

    /* --------------------------------------------------------
       Apply JL transform: X_new = X @ JL_matrix
       O(n * d_old * d_new), parallelized
       -------------------------------------------------------- */
    Dataset apply_jl_transform(const Dataset& X, const std::vector<float>& jl_matrix,
                               size_t d_new) {
        const size_t n = X.size();
        const size_t d = X.dim();

        Dataset X_proj(n, d_new);

        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < n; ++i) {
            const float* x = X.row_ptr(i);
            float* x_proj = X_proj.row_ptr(i);

            for (size_t j = 0; j < d_new; ++j) {
                float sum = 0.0f;
                for (size_t l = 0; l < d; ++l) {
                    sum += x[l] * jl_matrix[l * d_new + j];
                }
                x_proj[j] = sum;
            }
        }

        return X_proj;
    }
};

} // namespace fastcluster
