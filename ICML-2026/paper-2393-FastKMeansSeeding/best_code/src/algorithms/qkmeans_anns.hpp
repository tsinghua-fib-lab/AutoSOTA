#pragma once

#include "src/core/dataset.hpp"
#include "src/algorithms/prone.hpp"  // For D2SegmentTree
#include "src/algorithms/anns_interface.hpp"

#include <random>
#include <vector>
#include <cmath>
#include <cassert>
#include <limits>
#include <algorithm>
#include <numeric>
#include <memory>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace fastcluster {

/* ============================================================
   QKMEANS_ANNS - QKMEANS with pluggable ANNS methods

   Same algorithm as QKMEANS but allows swapping the ANNS backend
   to study the effect of different approximate nearest neighbor methods.
   ============================================================ */
class QKMEANS_ANNS {
public:
    explicit QKMEANS_ANNS(uint64_t seed = 42) : rng_(seed) {}

    /* --------------------------------------------------------
       Main interface

       Parameters:
         X           - Input dataset
         k           - Number of centers to select
         anns_method - Which ANNS method to use
         ef          - Search quality parameter (for HNSW)
         M           - Graph connectivity (for HNSW)
       -------------------------------------------------------- */
    Dataset run(const Dataset& X, size_t k, ANNSMethod anns_method = ANNSMethod::HNSW,
                size_t ef = 50, size_t M = 16) {
        assert(X.size() > 0);
        assert(k > 0 && k <= X.size());

        const size_t n = X.size();
        const size_t d = X.dim();

        // Step 1: Preprocessing - compute squared norms
        PreprocessedData prep = preprocess(X);

        // Step 2: Initialize ANNS structure based on method
        auto anns = create_anns(anns_method, d, k, ef, M, rng_());

        // Step 3: Sample first center uniformly at random
        std::uniform_int_distribution<size_t> unif_idx(0, n - 1);
        size_t c1_idx = unif_idx(rng_);
        float c1_norm_sq = prep.sq_norms[c1_idx];

        // Step 4: Build κ distribution using segment tree for O(log n) sampling
        std::vector<float> kappa_weights(n);
        std::vector<float> proposal_weights(n);
        for (size_t i = 0; i < n; ++i) {
            float norm_plus_c1 = prep.sq_norms[i] + c1_norm_sq;
            kappa_weights[i] = norm_plus_c1;
            proposal_weights[i] = 2.0f * norm_plus_c1;
        }

        D2SegmentTree kappa_tree(n);
        kappa_tree.build(kappa_weights);

        // Store selected center indices
        std::vector<size_t> center_indices;
        center_indices.reserve(k);
        center_indices.push_back(c1_idx);

        // Insert first center into ANNS
        anns->insert(c1_idx, X.row_ptr(c1_idx));

        // Step 5: Sample remaining k-1 centers using rejection sampling
        std::uniform_real_distribution<float> unif01(0.0f, 1.0f);

        // Maximum iterations: m * ln(k) as per paper (using m=100 default)
        size_t m = 100;
        size_t max_iter = static_cast<size_t>(
            static_cast<float>(m) * std::log(static_cast<float>(k) + 1.0f)
        );
        max_iter = std::max(max_iter, size_t(10));

        for (size_t t = 1; t < k; ++t) {
            size_t s = unif_idx(rng_);  // Fallback

            for (size_t iter = 0; iter < max_iter; ++iter) {
                // Sample x from κ(·|C)
                float u = unif01(rng_) * kappa_tree.total();
                size_t x_idx = kappa_tree.sample(u);

                // Query ANNS for approximate distance to nearest center
                float dist_to_center = anns->query(X.row_ptr(x_idx));

                // Compute acceptance probability
                float r = dist_to_center / proposal_weights[x_idx];
                r = std::min(r, 1.0f);

                // Rejection test
                if (unif01(rng_) <= r) {
                    s = x_idx;
                    break;
                }
            }

            center_indices.push_back(s);
            anns->insert(s, X.row_ptr(s));
        }

        // Build output Dataset
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

    struct PreprocessedData {
        std::vector<float> sq_norms;
        float frobenius_sq;
        size_t n;
    };

    PreprocessedData preprocess(const Dataset& X) {
        PreprocessedData prep;
        const size_t n = X.size();
        const size_t d = X.dim();

        prep.n = n;
        prep.sq_norms.resize(n);
        prep.frobenius_sq = 0.0f;

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
};

} // namespace fastcluster
