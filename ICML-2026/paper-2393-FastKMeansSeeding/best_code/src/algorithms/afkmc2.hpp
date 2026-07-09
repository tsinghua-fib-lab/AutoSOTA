#pragma once

#include "src/core/dataset.hpp"
#include <random>
#include <vector>
#include <numeric>
#include <cassert>
#include <limits>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace fastcluster {

/* ============================================
   AFK-MC2 seeding (Assumption-Free K-MC2)
   Implements Algorithm 1 from Bachem et al. 2016
   ============================================ */
class AFKMC2 {
public:
    /* -----------------------------
       Default constructor
       ----------------------------- */
    explicit AFKMC2(uint64_t seed = 42)
        : rng_(seed) {}

    /* --------------------------------------------
       Run AFK-MC2 seeding
       Returns a Dataset containing k centers
       -------------------------------------------- */
    Dataset run(const Dataset& X, size_t k, size_t m) {
        assert(X.size() > 0);
        assert(k <= X.size());

        const size_t n = X.size();
        const size_t d = X.dim();

        /* =========================
           Step 1: sample c1 uniform
           ========================= */
        std::uniform_int_distribution<size_t> unif_idx(0, n - 1);
        size_t c1_idx = unif_idx(rng_);

        Dataset centers(1, d);
        std::copy(
            X.row_ptr(c1_idx),
            X.row_ptr(c1_idx) + d,
            centers.row_ptr(0)
        );

        /* =========================
           Step 2: compute proposal q
           q(x) = 1/2 * D(x,c1)^2 / sum + 1/(2n)
           ========================= */
        std::vector<float> q(n);
        float sum_d2 = 0.0f;

        // Sequential: AFKMC2 should not use parallelization
        for (size_t i = 0; i < n; ++i) {
            sum_d2 += Dataset::l2_sq(
                X.row_ptr(i),
                centers.row_ptr(0),
                d
            );
        }

        const float uniform_term = 0.5f / static_cast<float>(n);
        for (size_t i = 0; i < n; ++i) {
            float d2 = Dataset::l2_sq(
                X.row_ptr(i),
                centers.row_ptr(0),
                d
            );
            q[i] = 0.5f * (d2 / sum_d2) + uniform_term;
        }

        std::discrete_distribution<size_t> proposal(q.begin(), q.end());
        std::uniform_real_distribution<float> unif01(0.0f, 1.0f);

        /* =========================
           Main loop: add k-1 centers
           ========================= */
        for (size_t i = 2; i <= k; ++i) {

            size_t x_idx = proposal(rng_);
            float dx = min_d2_to_centers(X, x_idx, centers);

            for (size_t step = 1; step < m; ++step) {
                size_t y_idx = proposal(rng_);
                float dy = min_d2_to_centers(X, y_idx, centers);

                float accept_ratio =
                    (dy * q[x_idx]) / (dx * q[y_idx]);

                if (accept_ratio >= 1.0f ||
                    accept_ratio > unif01(rng_)) {
                    x_idx = y_idx;
                    dx = dy;
                }
            }

            centers_append(centers, X, x_idx);
        }

        return centers;
    }

private:
    std::mt19937_64 rng_;

    /* --------------------------------------------
       Compute min squared distance to current centers
       -------------------------------------------- */
    static float min_d2_to_centers(
        const Dataset& X,
        size_t idx,
        const Dataset& centers
    ) {
        float best = std::numeric_limits<float>::max();
        const float* x = X.row_ptr(idx);

        for (size_t c = 0; c < centers.size(); ++c) {
            float d2 = Dataset::l2_sq(
                x,
                centers.row_ptr(c),
                X.dim()
            );
            if (d2 < best) best = d2;
        }
        return best;
    }

    /* --------------------------------------------
       Append a new center to Dataset
       -------------------------------------------- */
    static void centers_append(
        Dataset& centers,
        const Dataset& X,
        size_t idx
    ) {
        size_t old_n = centers.size();
        size_t d = centers.dim();

        Dataset new_centers(old_n + 1, d);

        for (size_t i = 0; i < old_n; ++i) {
            std::copy(
                centers.row_ptr(i),
                centers.row_ptr(i) + d,
                new_centers.row_ptr(i)
            );
        }

        std::copy(
            X.row_ptr(idx),
            X.row_ptr(idx) + d,
            new_centers.row_ptr(old_n)
        );

        centers = std::move(new_centers);
    }
};

} // namespace fastcluster
