#pragma once

#include "src/core/dataset.hpp"

#include <vector>
#include <limits>
#include <cmath>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace fastcluster {

/* ============================================================
   Lloyd's Algorithm for K-Means Clustering
   ============================================================ */
class Lloyd {
public:
    struct Result {
        Dataset centers;
        std::vector<size_t> labels;
        float inertia;
        int n_iter;
        bool converged;
    };

    explicit Lloyd(int max_iter = 300, float tol = 1e-4)
        : max_iter_(max_iter), tol_(tol) {}

    Result run(const Dataset& X, const Dataset& init_centers) {
        const size_t n = X.size();
        const size_t d = X.dim();
        const size_t k = init_centers.size();

        Result result;
        result.labels.resize(n);
        result.n_iter = 0;
        result.converged = false;

        // Copy initial centers
        Dataset centers(k, d);
        for (size_t i = 0; i < k; ++i) {
            std::copy(
                init_centers.row_ptr(i),
                init_centers.row_ptr(i) + d,
                centers.row_ptr(i)
            );
        }

        std::vector<size_t> counts(k);
        Dataset new_centers(k, d);

        for (int iter = 0; iter < max_iter_; ++iter) {
            result.n_iter = iter + 1;

            // Assignment step: assign each point to nearest center
            float inertia = 0.0f;

            #pragma omp parallel for reduction(+:inertia) schedule(static)
            for (size_t i = 0; i < n; ++i) {
                const float* x = X.row_ptr(i);
                float best_dist = std::numeric_limits<float>::max();
                size_t best_c = 0;

                for (size_t c = 0; c < k; ++c) {
                    float dist = Dataset::l2_sq(x, centers.row_ptr(c), d);
                    if (dist < best_dist) {
                        best_dist = dist;
                        best_c = c;
                    }
                }

                result.labels[i] = best_c;
                inertia += best_dist;
            }
            result.inertia = inertia;

            // Update step: compute new centroids
            std::fill(counts.begin(), counts.end(), 0);
            for (size_t i = 0; i < k; ++i) {
                std::fill(new_centers.row_ptr(i), new_centers.row_ptr(i) + d, 0.0f);
            }

            for (size_t i = 0; i < n; ++i) {
                size_t c = result.labels[i];
                counts[c]++;
                const float* x = X.row_ptr(i);
                float* nc = new_centers.row_ptr(c);
                for (size_t j = 0; j < d; ++j) {
                    nc[j] += x[j];
                }
            }

            // Normalize and check convergence
            float max_shift = 0.0f;
            for (size_t c = 0; c < k; ++c) {
                float* nc = new_centers.row_ptr(c);
                float* oc = centers.row_ptr(c);

                if (counts[c] > 0) {
                    for (size_t j = 0; j < d; ++j) {
                        nc[j] /= static_cast<float>(counts[c]);
                    }
                } else {
                    // Empty cluster: keep old center
                    std::copy(oc, oc + d, nc);
                }

                // Compute shift
                float shift = Dataset::l2_sq(nc, oc, d);
                if (shift > max_shift) max_shift = shift;
            }

            // Copy new centers
            for (size_t c = 0; c < k; ++c) {
                std::copy(
                    new_centers.row_ptr(c),
                    new_centers.row_ptr(c) + d,
                    centers.row_ptr(c)
                );
            }

            // Check convergence
            if (std::sqrt(max_shift) < tol_) {
                result.converged = true;
                break;
            }
        }

        result.centers = std::move(centers);
        return result;
    }

private:
    int max_iter_;
    float tol_;
};

} // namespace fastcluster
