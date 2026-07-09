#pragma once

#include "src/core/dataset.hpp"
#include "src/algorithms/prone.hpp"  // For D2SegmentTree

#include <vector>
#include <random>
#include <algorithm>
#include <cassert>
#include <limits>

namespace fastcluster {

/* ============================================================
   Standard K-Means++ Seeding (Arthur & Vassilvitskii 2007)
   O(nkd) time complexity
   ============================================================ */
class KMeansPP {
public:
    explicit KMeansPP(uint64_t seed = 42)
        : rng_(seed) {}

    Dataset run(const Dataset& X, size_t k) {
        const size_t n = X.size();
        const size_t d = X.dim();
        assert(k <= n);

        std::vector<size_t> center_indices;
        center_indices.reserve(k);

        // D^2 distances to nearest center
        std::vector<float> d2(n, std::numeric_limits<float>::infinity());

        // Step 1: Choose first center uniformly at random
        std::uniform_int_distribution<size_t> unif(0, n - 1);
        size_t first = unif(rng_);
        center_indices.push_back(first);

        // Update distances to first center
        const float* c0 = X.row_ptr(first);
        for (size_t i = 0; i < n; ++i) {
            d2[i] = Dataset::l2_sq(X.row_ptr(i), c0, d);
        }

        // Build segment tree for O(log n) sampling
        D2SegmentTree tree(n);
        tree.build(d2);

        std::uniform_real_distribution<float> unif01(0.0f, 1.0f);

        // Step 2: Choose remaining k-1 centers
        while (center_indices.size() < k) {
            // Sample proportional to D^2
            float Z = tree.total();
            float u = unif01(rng_) * Z;
            size_t new_center = tree.sample(u);

            center_indices.push_back(new_center);

            // Update D^2 distances
            const float* cn = X.row_ptr(new_center);
            for (size_t i = 0; i < n; ++i) {
                float dist = Dataset::l2_sq(X.row_ptr(i), cn, d);
                if (dist < d2[i]) {
                    d2[i] = dist;
                    tree.update(i, dist);
                }
            }
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
};

} // namespace fastcluster
