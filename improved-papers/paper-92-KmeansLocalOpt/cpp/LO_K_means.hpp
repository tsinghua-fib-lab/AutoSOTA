#pragma once

#include <bits/stdc++.h>
using namespace std;

/*
LO-K-means (C-LO, D-LO, Min-D-LO).
Key Assumptions:
- N > K (Number of samples > Number of clusters).
- All input data points (X) are unique.

The time complexity per iteration is O(NKD).

Example Usage:
LO_K_Means<double> kmeans(unique_data, sample_weights, K, "kmeans++", "squared", rand(), 1e-9);
auto [assignments, centers, loss] = kmeans.K_means();

Default settings: init="kmeans++", breg="squared".

Available Clustering Methods:
- K_means(): Standard K-means algorithm.
- C_LO_K_means(): Aims to ensure C-local optimality.
- D_LO_K_means(): Aims to ensure D-local optimality.
- Min_D_LO_K_means(): Aims to ensure D-local optimality; often faster in practice than D_LO_K_means.
- D_LO_P_X(): Peng and Xia (2005) algorithm, aims to ensure D-local optimality.
*/
template <typename T>
struct LO_K_Means {
    vector<vector<T>> X;  // Data matrix of shape (N, D)
    vector<T> weight;     // Sample weights of length N
    vector<vector<T>> init_centers;
    int N, D, K;  // Number of points, dimensions, Number of clusters
    int random_state, step_num;
    T eps, MAX_T;
    string breg;  // Type of Bregman divergence (squared, KL, Itakura)

    LO_K_Means(const vector<vector<T>>& data, const vector<T>& w, int cluster_num, const string& init_ = "kmeans++", const string& breg_ = "squared", int random_state_ = rand(), T eps_ = 1e-10)
        : X(data),
          weight(w),
          N((int)data.size()),
          D((int)data[0].size()),
          K(cluster_num),
          init_centers(cluster_num, vector<T>((int)data[0].size())),
          breg(breg_),
          random_state(random_state_),
          eps(eps_),
          MAX_T(numeric_limits<T>::infinity() / 2),
          step_num(0) {
        if (init_ == "kmeans++") kmeans_plus_plus_init();
        else if (init_ == "rand") kmeans_rand_init();
        else assert(false);
    }

    // k-means++ initialization
    void kmeans_plus_plus_init() {
        vector<vector<T>> centers(K);
        mt19937 rng(random_state);
        discrete_distribution<int> dist_first(weight.begin(), weight.end());
        int first_idx = dist_first(rng);
        init_centers[0] = X[first_idx];
        vector<T> closest_dist_sq(N, MAX_T);

        for (int c = 1; c < K; c++) {
            for (int i = 0; i < N; i++) {
                T dist = 0.0;
                for (int d = 0; d < D; d++) {
                    dist += (X[i][d] - init_centers[c - 1][d]) * (X[i][d] - init_centers[c - 1][d]);
                }
                closest_dist_sq[i] = min(closest_dist_sq[i], dist);
            }

            vector<T> probs(N, 0.0);
            for (int i = 0; i < N; ++i) {
                probs[i] = weight[i] * closest_dist_sq[i];
            }
            discrete_distribution<int> dist_next(probs.begin(), probs.end());
            init_centers[c] = X[dist_next(rng)];
        }
    }
    // Random initialization
    void kmeans_rand_init() {
        mt19937 rng(random_state);
        discrete_distribution<int> dist(weight.begin(), weight.end());
        for (int c = 0; c < K; c++) init_centers[c] = X[dist(rng)];
    }

    T phi(const vector<T>& a) {
        assert((int)a.size() == D);
        T ret = 0;
        for (int i = 0; i < D; i++) {
            assert(a[i] > eps);
            if (breg == "KL") ret += a[i] * log(a[i]);
            else if (breg == "Itakura") ret += -log(a[i]);
            else assert(false);
        }
        return ret;
    }

    T grad(const vector<T>& a, const vector<T>& b) {
        assert((int)a.size() == D);
        T ret = 0;
        for (int i = 0; i < D; i++) {
            assert(b[i] > eps);
            if (breg == "KL") ret += (a[i] - b[i]) * (log(b[i]) + 1);
            else if (breg == "Itakura") ret += -(a[i] - b[i]) / b[i];
            else assert(false);
        }
        return ret;
    }

    // Compute Bregman divergence between vectors a and b
    T bregman(const vector<T>& a, const vector<T>& b) {
        assert((int)a.size() == D);
        assert((int)b.size() == D);
        if (breg == "squared") {
            T ret = 0;
            for (int i = 0; i < D; i++) ret += (a[i] - b[i]) * (a[i] - b[i]);
            return ret;
        } else {
            return phi(a) - phi(b) - grad(a, b);
        }
    }

    // Compute new centers
    vector<vector<T>> weighted_mean(const vector<int>& assignment) {
        vector<vector<T>> avg(K, vector<T>(D));
        vector<T> total_weight(K, 0.0);
        for (int i = 0; i < N; i++) {
            for (int d = 0; d < D; d++) {
                avg[assignment[i]][d] += weight[i] * X[i][d];
            }
            total_weight[assignment[i]] += weight[i];
        }
        for (int i = 0; i < K; i++) {
            if (total_weight[i] > eps) {
                for (int d = 0; d < D; ++d) avg[i][d] /= total_weight[i];
            }
        }
        return avg;
    }

    // Compute clustering loss
    T clustering_loss(const vector<vector<T>>& centers, const vector<int>& assignment) {
        T ret = 0;
        for (int i = 0; i < N; i++) {
            ret += weight[i] * bregman(X[i], centers[assignment[i]]);
        }
        return ret;
    }

    // Standard K-means
    tuple<vector<int>, vector<vector<T>>, T> K_means() {
        vector<vector<T>> centers = init_centers;
        vector<int> assignment(N, 0);
        vector<int> old_assignment(N, -1);
        vector<vector<T>> distances(N, vector<T>(K, 0.0));
        step_num = 0;
        while (assignment != old_assignment) {
            step_num++;
            swap(old_assignment, assignment);
            for (int i = 0; i < N; i++) {
                for (int k = 0; k < K; k++) {
                    distances[i][k] = bregman(X[i], centers[k]);
                }
            }
            vector<T> cluster_weight(K, 0);
            for (int i = 0; i < N; ++i) {
                T best_dist = MAX_T;
                int best_k = 0;
                for (int k = 0; k < K; k++) {
                    if (distances[i][k] + eps < best_dist) {
                        best_dist = distances[i][k];
                        best_k = k;
                    }
                }
                assignment[i] = best_k;
                cluster_weight[assignment[i]] += weight[i];
            }
            int id = 0;
            for (int i = 0; i < K; i++) {
                if (cluster_weight[i] == 0) {
                    while (id < N && (cluster_weight[assignment[id]] <= weight[id])) id++;
                    cluster_weight[assignment[id]] -= weight[id];
                    assignment[id] = i;
                    cluster_weight[i] += weight[id];
                }
            }
            centers = weighted_mean(assignment);
        }
        T loss = clustering_loss(centers, assignment);
        return {assignment, centers, loss};
    }

    // C-LO-K-means
    tuple<vector<int>, vector<vector<T>>, T> C_LO_K_means() {
        vector<vector<T>> centers = init_centers;
        vector<int> assignment(N, 0);
        vector<int> old_assignment(N, -1);
        vector<vector<T>> distances(N, vector<T>(K));
        step_num = 0;
        while (assignment != old_assignment) {
            step_num++;
            swap(old_assignment, assignment);
            for (int i = 0; i < N; i++) {
                for (int k = 0; k < K; k++) {
                    distances[i][k] = bregman(X[i], centers[k]);
                }
            }
            vector<T> cluster_weight(K, 0);
            for (int i = 0; i < N; ++i) {
                T best_dist = MAX_T;
                int best_k = 0;
                for (int k = 0; k < K; k++) {
                    if (distances[i][k] + eps < best_dist) {
                        best_dist = distances[i][k];
                        best_k = k;
                    }
                }
                assignment[i] = best_k;
                cluster_weight[assignment[i]] += weight[i];
            }
            int id = 0;
            for (int i = 0; i < K; i++) {
                if (cluster_weight[i] == 0) {
                    while (id < N && (cluster_weight[assignment[id]] <= weight[id])) id++;
                    cluster_weight[assignment[id]] -= weight[id];
                    assignment[id] = i;
                    cluster_weight[i] += weight[id];
                }
            }
            // Function 1
            if (assignment == old_assignment) {
                bool flag = false;
                for (int i = 0; i < N; i++) {
                    vector<int> min_indices;
                    T curr_val = distances[i][assignment[i]];
                    for (int j = 0; j < K; j++) {
                        if (fabsl(distances[i][j] - curr_val) < eps) {
                            min_indices.push_back(j);
                        }
                    }
                    if (min_indices.size() >= 2) {
                        assignment[i] = min_indices[1];
                        break;
                    }
                }
            }
            centers = weighted_mean(assignment);
        }
        T loss = clustering_loss(centers, assignment);
        return {assignment, centers, loss};
    }

    // D-LO-K-means
    tuple<vector<int>, vector<vector<T>>, T> D_LO_K_means() {
        vector<vector<T>> centers = init_centers;
        vector<int> assignment(N, 0);
        vector<int> old_assignment(N, -1);
        vector<vector<T>> distances(N, vector<T>(K));
        step_num = 0;
        for (int i = 0; i < N; i++) {
            for (int k = 0; k < K; k++) {
                distances[i][k] = bregman(X[i], centers[k]);
            }
        }
        while (assignment != old_assignment) {
            step_num++;
            swap(old_assignment, assignment);
            vector<T> cluster_weight(K, 0);
            for (int i = 0; i < N; ++i) {
                T best_dist = MAX_T;
                int best_k = 0;
                for (int k = 0; k < K; k++) {
                    if (distances[i][k] + eps < best_dist) {
                        best_dist = distances[i][k];
                        best_k = k;
                    }
                }
                assignment[i] = best_k;
                cluster_weight[assignment[i]] += weight[i];
            }
            int id = 0;
            for (int i = 0; i < K; i++) {
                if (cluster_weight[i] == 0) {
                    while (id < N && (cluster_weight[assignment[id]] <= weight[id])) id++;
                    cluster_weight[assignment[id]] -= weight[id];
                    assignment[id] = i;
                    cluster_weight[i] += weight[id];
                }
            }

            centers = weighted_mean(assignment);
            for (int i = 0; i < N; i++) {
                for (int k = 0; k < K; k++) {
                    distances[i][k] = bregman(X[i], centers[k]);
                }
            }

            // Function 2
            if (assignment == old_assignment) {
                bool flag = false;
                for (int i = 0; i < N; i++) {
                    int a = assignment[i];
                    if (cluster_weight[a] <= weight[i]) continue;
                    vector<T> new_center_a(D), new_center_j(D);
                    for (int d = 0; d < D; d++) new_center_a[d] = centers[a][d] + weight[i] * (centers[a][d] - X[i][d]) / (cluster_weight[a] - weight[i]);
                    for (int j = 0; j < K; j++) {
                        if (assignment[i] == j) continue;
                        for (int d = 0; d < D; d++) new_center_j[d] = centers[j][d] + weight[i] * (X[i][d] - centers[j][d]) / (cluster_weight[j] + weight[i]);

                        T diff = weight[i] * (distances[i][j] - distances[i][a]) -
                                 ((cluster_weight[j] + weight[i]) * bregman(new_center_j, centers[j]) + (cluster_weight[a] - weight[i]) * bregman(new_center_a, centers[a]));

                        if (diff + eps < 0.0) {
                            assignment[i] = j;
                            for (int d = 0; d < D; d++) {
                                centers[a] = new_center_a;
                                centers[j] = new_center_j;
                            }
                            for (int n = 0; n < N; n++) {
                                distances[n][a] = bregman(X[n], centers[a]);
                                distances[n][j] = bregman(X[n], centers[j]);
                            }
                            flag = true;
                            break;
                        }
                    }
                    if (flag) break;
                }
            }
        }
        T loss = clustering_loss(centers, assignment);
        return {assignment, centers, loss};
    }

    // Min-D-LO-K-means
    tuple<vector<int>, vector<vector<T>>, T> Min_D_LO_K_means() {
        vector<vector<T>> centers = init_centers;
        vector<int> assignment(N, 0);
        vector<int> old_assignment(N, -1);
        vector<vector<T>> distances(N, vector<T>(K));
        step_num = 0;
        for (int i = 0; i < N; i++) {
            for (int k = 0; k < K; k++) {
                distances[i][k] = bregman(X[i], centers[k]);
            }
        }
        while (assignment != old_assignment) {
            step_num++;
            swap(old_assignment, assignment);
            vector<T> cluster_weight(K, 0);
            for (int i = 0; i < N; ++i) {
                T best_dist = MAX_T;
                int best_k = 0;
                for (int k = 0; k < K; k++) {
                    if (distances[i][k] + eps < best_dist) {
                        best_dist = distances[i][k];
                        best_k = k;
                    }
                }
                assignment[i] = best_k;
                cluster_weight[assignment[i]] += weight[i];
            }
            int id = 0;
            for (int i = 0; i < K; i++) {
                if (cluster_weight[i] == 0) {
                    while (id < N && (cluster_weight[assignment[id]] <= weight[id])) id++;
                    cluster_weight[assignment[id]] -= weight[id];
                    assignment[id] = i;
                    cluster_weight[i] += weight[id];
                }
            }

            centers = weighted_mean(assignment);
            for (int i = 0; i < N; i++) {
                for (int k = 0; k < K; k++) {
                    distances[i][k] = bregman(X[i], centers[k]);
                }
            }

            // Function 3
            if (assignment == old_assignment) {
                T best_diff = MAX_T;
                int best_i = 0, best_j = 0;
                for (int i = 0; i < N; i++) {
                    int a = assignment[i];
                    if (cluster_weight[a] <= weight[i]) continue;
                    vector<T> new_center_a(D), new_center_j(D);
                    for (int d = 0; d < D; d++) new_center_a[d] = centers[a][d] + weight[i] * (centers[a][d] - X[i][d]) / (cluster_weight[a] - weight[i]);
                    for (int j = 0; j < K; j++) {
                        if (assignment[i] == j) continue;
                        for (int d = 0; d < D; d++) new_center_j[d] = centers[j][d] + weight[i] * (X[i][d] - centers[j][d]) / (cluster_weight[j] + weight[i]);

                        T diff = weight[i] * (distances[i][j] - distances[i][a]) -
                                 ((cluster_weight[j] + weight[i]) * bregman(new_center_j, centers[j]) + (cluster_weight[a] - weight[i]) * bregman(new_center_a, centers[a]));

                        if (diff + eps < best_diff) {
                            best_diff = diff;
                            best_i = i;
                            best_j = j;
                        }
                    }
                }
                if (best_diff + eps < 0.0) {
                    int best_a = assignment[best_i];
                    for (int d = 0; d < D; d++) {
                        centers[best_a][d] = centers[best_a][d] + weight[best_i] * (centers[best_a][d] - X[best_i][d]) / (cluster_weight[best_a] - weight[best_i]);
                        centers[best_j][d] = centers[best_j][d] + weight[best_i] * (X[best_i][d] - centers[best_j][d]) / (cluster_weight[best_j] + weight[best_i]);
                    }
                    for (int n = 0; n < N; n++) {
                        distances[n][best_a] = bregman(X[n], centers[best_a]);
                        distances[n][best_j] = bregman(X[n], centers[best_j]);
                    }
                    assignment[best_i] = best_j;
                }
            }
        }
        T loss = clustering_loss(centers, assignment);
        return {assignment, centers, loss};
    }

    // Peng, J. and Xia, Y. "A Cutting Algorithm for the Minimum Sum-of-Squared Error Clustering." In Proceedings of the 2005 SIAM International Conference on Data Mining, pp. 150–160. SIAM, 2005.
    tuple<vector<int>, vector<vector<T>>, T> D_LO_P_X() {
        assert(breg == "squared");
        vector<vector<T>> centers = init_centers;
        vector<int> assignment(N, 0);
        vector<vector<T>> distances(N, vector<T>(K, 0.0));
        for (int i = 0; i < N; i++) {
            for (int k = 0; k < K; k++) {
                distances[i][k] = bregman(X[i], centers[k]);
            }
        }
        vector<T> cluster_weight(K, 0);
        for (int i = 0; i < N; ++i) {
            T best_dist = MAX_T;
            int best_k = 0;
            for (int k = 0; k < K; k++) {
                if (distances[i][k] + eps < best_dist) {
                    best_dist = distances[i][k];
                    best_k = k;
                }
            }
            assignment[i] = best_k;
            cluster_weight[assignment[i]] += weight[i];
        }
        step_num = 0;
        while (1) {
            step_num++;
            centers = weighted_mean(assignment);
            for (int i = 0; i < N; i++) {
                for (int k = 0; k < K; k++) {
                    distances[i][k] = bregman(X[i], centers[k]);
                }
            }
            bool flag = false;
            for (int i = 0; i < N; i++) {
                int a = assignment[i];
                if (cluster_weight[a] <= weight[i]) continue;
                for (int j = 0; j < K; j++) {
                    if (assignment[i] == j) continue;

                    // The diff can be expressed by the following equation (in squared Euclidean distance)
                    T diff = weight[i] * (distances[i][j] - distances[i][a]) -
                             (weight[i] * weight[i] * distances[i][j] / (cluster_weight[j] + weight[i]) + weight[i] * weight[i] * distances[i][a] / (cluster_weight[a] - weight[i]));

                    if (diff + eps < 0.0) {
                        assignment[i] = j;
                        cluster_weight[a] -= weight[i];
                        cluster_weight[j] += weight[i];
                        flag = true;
                        break;
                    }
                }
                if (flag) break;
            }
            if (!flag) break;
        }
        T loss = clustering_loss(centers, assignment);
        return {assignment, centers, loss};
    }
};