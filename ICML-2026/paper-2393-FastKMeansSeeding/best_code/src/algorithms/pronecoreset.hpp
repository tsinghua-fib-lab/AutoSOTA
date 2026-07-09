#pragma once

#include "src/core/dataset.hpp"
#include "src/algorithms/prone.hpp"

#include <vector>
#include <random>
#include <numeric>
#include <cassert>
#include <limits>
#include <cmath>

namespace fastcluster {

/* ============================================================
   Standard k-means++ seeding on a Dataset
   ============================================================ */
inline Dataset kmeanspp_seeding(
    const Dataset& X,
    size_t k,
    std::mt19937_64& rng
) {
    const size_t n = X.size();
    const size_t d = X.dim();
    assert(k <= n);

    Dataset centers(1, d);

    std::uniform_int_distribution<size_t> unif(0, n - 1);
    size_t first = unif(rng);

    std::copy(
        X.row_ptr(first),
        X.row_ptr(first) + d,
        centers.row_ptr(0)
    );

    std::vector<float> dist2(n);

    for (size_t i = 0; i < n; ++i) {
        dist2[i] = Dataset::l2_sq(
            X.row_ptr(i),
            centers.row_ptr(0),
            d
        );
    }

    for (size_t c = 1; c < k; ++c) {
        std::discrete_distribution<size_t> d2dist(
            dist2.begin(), dist2.end()
        );

        size_t idx = d2dist(rng);

        Dataset new_centers(c + 1, d);
        for (size_t j = 0; j < c; ++j) {
            std::copy(
                centers.row_ptr(j),
                centers.row_ptr(j) + d,
                new_centers.row_ptr(j)
            );
        }

        std::copy(
            X.row_ptr(idx),
            X.row_ptr(idx) + d,
            new_centers.row_ptr(c)
        );

        centers = std::move(new_centers);

        for (size_t i = 0; i < n; ++i) {
            float d2 = Dataset::l2_sq(
                X.row_ptr(i),
                centers.row_ptr(c),
                d
            );
            if (d2 < dist2[i])
                dist2[i] = d2;
        }
    }

    return centers;
}

/* ============================================================
   PRONECoreset
   ============================================================ */
class PRONECoreset {
public:
    explicit PRONECoreset(uint64_t seed = 42)
        : rng_(seed) {}

    Dataset run(const Dataset& X, size_t k, size_t m) {
        const size_t n = X.size();
        const size_t d = X.dim();
        // assert(k <= n);
        // assert(m >= k);

        /* --------------------------------------------
           Step 1: PRONE clustering
           -------------------------------------------- */
        PRONE prone(rng_());
        Dataset C0 = prone.run(X, k);

        /* --------------------------------------------
           Step 2: sensitivity computation
           s_i ∝ d(x_i, C0)^2
           -------------------------------------------- */
        std::vector<float> sens(n);
        float total = 0.0f;

        for (size_t i = 0; i < n; ++i) {
            float best = std::numeric_limits<float>::max();
            const float* xi = X.row_ptr(i);

            for (size_t c = 0; c < k; ++c) {
                float d2 = Dataset::l2_sq(
                    xi,
                    C0.row_ptr(c),
                    d
                );
                if (d2 < best)
                    best = d2;
            }

            sens[i] = best;
            total += best;
        }

        if (total == 0.0f) {
            // All points identical – fallback
            return kmeanspp_seeding(X, k, rng_);
        }

        for (float& s : sens)
            s /= total;

        /* --------------------------------------------
           Step 3: sample coreset
           -------------------------------------------- */
        std::discrete_distribution<size_t> sens_dist(
            sens.begin(), sens.end()
        );

        Dataset coreset(m, d);
        for (size_t i = 0; i < m; ++i) {
            size_t idx = sens_dist(rng_);
            std::copy(
                X.row_ptr(idx),
                X.row_ptr(idx) + d,
                coreset.row_ptr(i)
            );
        }

        /* --------------------------------------------
           Step 4: exact k-means++ on coreset
           -------------------------------------------- */
        return kmeanspp_seeding(coreset, k, rng_);
    }

private:
    std::mt19937_64 rng_;
};

} // namespace fastcluster
