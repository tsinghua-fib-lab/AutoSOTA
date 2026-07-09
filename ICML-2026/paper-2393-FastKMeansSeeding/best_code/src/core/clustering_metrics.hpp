#pragma once

#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <limits>
#include <cassert>

namespace fastcluster {

/* ============================================================
   Clustering Quality Metrics
   ============================================================ */

/* ------------------------------------------------------------
   Hungarian Algorithm for optimal assignment
   Minimizes total cost in O(n^3)
   ------------------------------------------------------------ */
class Hungarian {
public:
    static std::vector<size_t> solve(const std::vector<std::vector<double>>& cost) {
        size_t n = cost.size();
        if (n == 0) return {};

        // For non-square matrices, pad with zeros
        size_t m = cost[0].size();
        size_t dim = std::max(n, m);

        std::vector<std::vector<double>> C(dim, std::vector<double>(dim, 0.0));
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < m; ++j) {
                C[i][j] = cost[i][j];
            }
        }

        // Hungarian algorithm implementation
        std::vector<double> u(dim + 1, 0), v(dim + 1, 0);
        std::vector<size_t> p(dim + 1, 0), way(dim + 1, 0);

        for (size_t i = 1; i <= dim; ++i) {
            p[0] = i;
            size_t j0 = 0;
            std::vector<double> minv(dim + 1, std::numeric_limits<double>::max());
            std::vector<bool> used(dim + 1, false);

            do {
                used[j0] = true;
                size_t i0 = p[j0], j1 = 0;
                double delta = std::numeric_limits<double>::max();

                for (size_t j = 1; j <= dim; ++j) {
                    if (!used[j]) {
                        double cur = C[i0 - 1][j - 1] - u[i0] - v[j];
                        if (cur < minv[j]) {
                            minv[j] = cur;
                            way[j] = j0;
                        }
                        if (minv[j] < delta) {
                            delta = minv[j];
                            j1 = j;
                        }
                    }
                }

                for (size_t j = 0; j <= dim; ++j) {
                    if (used[j]) {
                        u[p[j]] += delta;
                        v[j] -= delta;
                    } else {
                        minv[j] -= delta;
                    }
                }

                j0 = j1;
            } while (p[j0] != 0);

            do {
                size_t j1 = way[j0];
                p[j0] = p[j1];
                j0 = j1;
            } while (j0);
        }

        std::vector<size_t> assignment(n);
        for (size_t j = 1; j <= dim; ++j) {
            if (p[j] > 0 && p[j] <= n) {
                assignment[p[j] - 1] = j - 1;
            }
        }

        return assignment;
    }
};

/* ------------------------------------------------------------
   Clustering Accuracy via Hungarian Assignment
   ------------------------------------------------------------ */
inline double cluster_accuracy(
    const std::vector<size_t>& y_true,
    const std::vector<size_t>& y_pred
) {
    size_t n = y_true.size();
    assert(n == y_pred.size());

    // Find number of clusters/classes
    size_t n_true = *std::max_element(y_true.begin(), y_true.end()) + 1;
    size_t n_pred = *std::max_element(y_pred.begin(), y_pred.end()) + 1;
    size_t n_clusters = std::max(n_true, n_pred);

    // Build confusion matrix
    std::vector<std::vector<double>> confusion(n_clusters,
        std::vector<double>(n_clusters, 0.0));

    for (size_t i = 0; i < n; ++i) {
        confusion[y_pred[i]][y_true[i]] += 1.0;
    }

    // Convert to cost matrix (negative for maximization)
    for (size_t i = 0; i < n_clusters; ++i) {
        for (size_t j = 0; j < n_clusters; ++j) {
            confusion[i][j] = -confusion[i][j];
        }
    }

    // Solve assignment problem
    auto assignment = Hungarian::solve(confusion);

    // Count correct assignments
    double correct = 0.0;
    for (size_t i = 0; i < n; ++i) {
        if (assignment[y_pred[i]] == y_true[i]) {
            correct += 1.0;
        }
    }

    return correct / static_cast<double>(n);
}

/* ------------------------------------------------------------
   Normalized Mutual Information (NMI)
   ------------------------------------------------------------ */
inline double normalized_mutual_info(
    const std::vector<size_t>& y_true,
    const std::vector<size_t>& y_pred
) {
    size_t n = y_true.size();
    assert(n == y_pred.size());

    size_t n_true = *std::max_element(y_true.begin(), y_true.end()) + 1;
    size_t n_pred = *std::max_element(y_pred.begin(), y_pred.end()) + 1;

    // Contingency matrix
    std::vector<std::vector<double>> contingency(n_true,
        std::vector<double>(n_pred, 0.0));

    for (size_t i = 0; i < n; ++i) {
        contingency[y_true[i]][y_pred[i]] += 1.0;
    }

    // Marginals
    std::vector<double> a(n_true, 0.0), b(n_pred, 0.0);
    for (size_t i = 0; i < n_true; ++i) {
        for (size_t j = 0; j < n_pred; ++j) {
            a[i] += contingency[i][j];
            b[j] += contingency[i][j];
        }
    }

    double N = static_cast<double>(n);

    // Entropy of true labels
    double H_true = 0.0;
    for (size_t i = 0; i < n_true; ++i) {
        if (a[i] > 0) {
            double p = a[i] / N;
            H_true -= p * std::log(p);
        }
    }

    // Entropy of predicted labels
    double H_pred = 0.0;
    for (size_t j = 0; j < n_pred; ++j) {
        if (b[j] > 0) {
            double p = b[j] / N;
            H_pred -= p * std::log(p);
        }
    }

    // Mutual Information
    double MI = 0.0;
    for (size_t i = 0; i < n_true; ++i) {
        for (size_t j = 0; j < n_pred; ++j) {
            if (contingency[i][j] > 0 && a[i] > 0 && b[j] > 0) {
                double p_ij = contingency[i][j] / N;
                double p_i = a[i] / N;
                double p_j = b[j] / N;
                MI += p_ij * std::log(p_ij / (p_i * p_j));
            }
        }
    }

    // NMI (arithmetic mean normalization)
    if (H_true + H_pred < 1e-10) return 1.0;
    return 2.0 * MI / (H_true + H_pred);
}

/* ------------------------------------------------------------
   Adjusted Rand Index (ARI)
   ------------------------------------------------------------ */
inline double adjusted_rand_index(
    const std::vector<size_t>& y_true,
    const std::vector<size_t>& y_pred
) {
    size_t n = y_true.size();
    assert(n == y_pred.size());

    size_t n_true = *std::max_element(y_true.begin(), y_true.end()) + 1;
    size_t n_pred = *std::max_element(y_pred.begin(), y_pred.end()) + 1;

    // Contingency matrix
    std::vector<std::vector<double>> nij(n_true,
        std::vector<double>(n_pred, 0.0));

    for (size_t i = 0; i < n; ++i) {
        nij[y_true[i]][y_pred[i]] += 1.0;
    }

    // Row and column sums
    std::vector<double> a(n_true, 0.0), b(n_pred, 0.0);
    for (size_t i = 0; i < n_true; ++i) {
        for (size_t j = 0; j < n_pred; ++j) {
            a[i] += nij[i][j];
            b[j] += nij[i][j];
        }
    }

    // Binomial coefficient helper: n choose 2
    auto comb2 = [](double x) -> double {
        return x * (x - 1.0) / 2.0;
    };

    // Sum of C(n_ij, 2)
    double sum_nij = 0.0;
    for (size_t i = 0; i < n_true; ++i) {
        for (size_t j = 0; j < n_pred; ++j) {
            sum_nij += comb2(nij[i][j]);
        }
    }

    // Sum of C(a_i, 2)
    double sum_a = 0.0;
    for (size_t i = 0; i < n_true; ++i) {
        sum_a += comb2(a[i]);
    }

    // Sum of C(b_j, 2)
    double sum_b = 0.0;
    for (size_t j = 0; j < n_pred; ++j) {
        sum_b += comb2(b[j]);
    }

    double N = static_cast<double>(n);
    double comb_n = comb2(N);

    // ARI formula
    double expected = (sum_a * sum_b) / comb_n;
    double max_index = 0.5 * (sum_a + sum_b);
    double denominator = max_index - expected;

    if (std::abs(denominator) < 1e-10) return 1.0;
    return (sum_nij - expected) / denominator;
}

} // namespace fastcluster
