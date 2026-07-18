#ifndef TSP_FUNC_H
#define TSP_FUNC_H

#include <algorithm>
#include <iostream>
#include <limits>
#include <random>
#include <cassert>
#include <tuple>
#include <stdexcept>
#include <queue>

typedef std::vector<uint_fast16_t> Tour;
typedef std::vector<std::vector<double> > DoubleMatrix;
typedef std::vector<std::vector<uint_fast16_t> > IntMatrix;
typedef std::pair<uint_fast32_t, uint_fast16_t> IntPair;
typedef std::priority_queue<IntPair, std::vector<IntPair>, std::greater<IntPair> > PQ;

#define DEFAULT_TIME_LIMIT 2.0
#define EXACT_SOLUTION_THRESHOLD 20
#define EMPTY_SET 0

const long long kInf = 1e17;
const uint_fast16_t NEIGH_LIMIT = 20;
const static int GEO_SHUFFLE_WIDTH = 30;
const double CLOCKS_PER_SEC_INV = 1.0 / CLOCKS_PER_SEC;

// Set-up RNG
std::random_device rd;
std::mt19937 gen(rd());

Tour find_tour(const IntMatrix dists, const float duration_seconds);
const IntMatrix get_nearest_neighbour_matrix(const IntMatrix& dists);
Tour local_search(const IntMatrix dists, const clock_t& clock_begin, float duration_seconds);
void print_tour(const Tour& tour);
void two_opt(uint_fast16_t n, Tour& tour, const IntMatrix& dists, const IntMatrix& nn_matrix, uint_fast16_t* position, const clock_t& clock_begin, float duration_seconds);
Tour double_bridge_move(uint_fast16_t n, Tour& tour);
void reverse_tour(uint_fast16_t n, Tour& tour, uint_fast16_t start, uint_fast16_t end, uint_fast16_t* position);
void shuffle_tour(uint_fast16_t n, Tour& tour);
Tour solve_tsp_exact(const IntMatrix dists);
void three_opt(uint_fast16_t n, Tour& tour, const IntMatrix& dists, const IntMatrix& nn_matrix, uint_fast16_t* position, const clock_t& clock_begin, float duration_seconds);
void swap_adj_seg(uint_fast16_t n, Tour& tour, uint_fast16_t* position, uint_fast16_t A, uint_fast16_t B, uint_fast16_t C, uint_fast16_t D);
Tour greedy_nearest_neighbor(const IntMatrix& dist);
uint_fast32_t compute_cost(uint_fast16_t n, const Tour& tour, const IntMatrix& dists);
uint_fast32_t compute_cost_wrapper(const Tour& tour, const IntMatrix& dists);


Tour find_tour(const IntMatrix dists, const float duration_seconds = DEFAULT_TIME_LIMIT) {
    clock_t clock_begin = clock();
    if (duration_seconds <= 0.0) throw std::invalid_argument("Duration must be strictly positive");

    uint_fast16_t n = dists.size();
    if (n < 2) throw std::invalid_argument("Need at least 2 nodes");
    uint_fast16_t m = dists[0].size();
    if (m != n) throw std::invalid_argument(
        "Distance matrix must be square but got ("
        + std::to_string(n) + ", " + std::to_string(m)
        + ")"
    );

    // dist = dists;
    // if (n <= EXACT_SOLUTION_THRESHOLD) return solve_tsp_exact(dists);

    // Solve it
    return local_search(dists, clock_begin, duration_seconds);
}

inline bool is_time_exceeded(clock_t begin, float duration_seconds) {
    double elapsed_time = (double) (clock() - begin) * CLOCKS_PER_SEC_INV;
    return elapsed_time > duration_seconds;
}

// Fills up nearest neighbour matrix where for each node i, we compute the nearest
// neighbour.
const IntMatrix get_nearest_neighbour_matrix(const IntMatrix& dists) {
    uint_fast16_t n = dists.size();
    IntMatrix nn_matrix;
    nn_matrix.resize(n);

    uint_fast16_t size = std::min(NEIGH_LIMIT, n);
    for (uint_fast16_t i = 0; i < n; ++i) {
        nn_matrix[i] = Tour(size);
    }

    PQ pq;
    uint_fast16_t j = 0;
    uint_fast16_t d, v;
    for (uint_fast16_t i = 0; i < n; ++i) {
        pq = PQ();
        for (j = 0; j < n; ++j) {
            // we don't want a node to become it's own nearest neighbour
            if (i == j) continue;
            pq.push({dists[i][j], j});
        }

        j = 0;
        while (!pq.empty()) {
            if (j >= NEIGH_LIMIT) break;

            std::tie(d, v) = pq.top();
            pq.pop();

            nn_matrix[i][j] = v;
            j++;
        }
    }
    return nn_matrix;
}

// Implementation of bottom-up Held-Karp's algorithm, solving TSP in
// O(2^n * n^2) time.
Tour solve_tsp_exact(const IntMatrix dists) {
    int n = dists.size();
    const int num_combinations = 1 << n;
    const int max_mask = num_combinations - 1;

    // Vector is used instead because of seg fault
    std::vector<std::vector<long long> > dp(num_combinations);
    for (int i = 0; i < num_combinations; ++i) {
        dp[i] = std::vector<long long>(n);
        for (int j = 0; j < n; ++j) {
            dp[i][j] = kInf;
        }
    }

    // Fix node 0 as source for TSP tour
    for (int k = 1; k < n; ++k) {
        dp[1 << k][k] = (long long) dists[0][k];
    }

    // Iterate through all subsets of vertices/nodes
    for (int S = EMPTY_SET; S < num_combinations; ++S) {
        if (S & (1 << 0)) continue; // Don't care about subsets with source node

        // Each k nodes take turns to be dest!
        for (int k = 0; k < n; ++k) {
            if (!(S & (1 << k))) continue;

            for (int m = 0; m < n; ++m) {
                if ((m == k) || (!(S & (1 << m)))) continue;
                dp[S][k] = std::min(
                    dp[S][k],
                    dp[S & ~(1 << k)][m] + (long long) dists[m][k]
                );
            }
        }
    }

    // Reconstruct the path from dp array
    long long ans = kInf;
    Tour ans_path;
    int chosen_node = 0;

    ans_path.push_back(chosen_node);
    int mask = 1 << chosen_node;

    // Get the remaining n - 1 nodes
    for (int i = 0; i < n - 1; ++i) {
        ans = kInf;
        for (int k = 0; k < n; ++k) {
            // Only go through all potential dest not inside tour
            if (mask & (1 << k)) continue;
            int d = dp[max_mask & ~(mask)][k] + dists[k][ans_path[ans_path.size() - 1]];
            if (d < ans) {
                ans = d;
                chosen_node = k;
            }
        }

        ans_path.push_back(chosen_node);
        mask |= 1 << chosen_node;
    }

    return ans_path;
}

Tour double_bridge_move(uint_fast16_t n, Tour& tour) {
    Tour new_tour;
    new_tour.reserve(n);

    if (n < 8) {
        new_tour = tour;
        shuffle(new_tour.begin(), new_tour.end(), rd);
        return new_tour;
    }

    std::uniform_int_distribution<uint_fast16_t> random_offset(1, n / 4);
    uint_fast16_t A = random_offset(rd);
    uint_fast16_t B = A + random_offset(rd);
    uint_fast16_t C = B + random_offset(rd);
    copy(tour.begin(), tour.begin() + A, back_inserter(new_tour));
    copy(tour.begin() + C, tour.end(), back_inserter(new_tour));
    copy(tour.begin() + B, tour.begin() + C, back_inserter(new_tour));
    copy(tour.begin() + A, tour.begin() + B, back_inserter(new_tour));
    return new_tour;
}

// The below method is copy-pasted from estan/tsp
// See LICENSE file for more details
void reverse_tour(uint_fast16_t n, Tour& tour, uint_fast16_t start, uint_fast16_t end, uint_fast16_t* position) {
    uint_fast16_t num_swaps = (((start <= end ? end - start : (end + n) - start) + 1) / 2);
    uint_fast16_t i = start;
    uint_fast16_t j = end;
    while (num_swaps--) {
        std::swap(tour[i], tour[j]);

        position[tour[i]] = i;
        position[tour[j]] = j;

        i = (i + 1) % n;
        j = ((j + n) - 1) % n;
    }
}

Tour greedy_nearest_neighbor(const IntMatrix& dist) {
    uint_fast16_t n = dist.size();
    bool *used = new bool[n];
    for (uint_fast16_t i = 0; i < n; ++i) {
        used[i] = false;
    }

    Tour tour(n);
    tour[0] = 0;
    used[0] = true;
    uint_fast16_t j;
    for (uint_fast16_t i = 1; i < n; ++i) {
        int best_idx = -1;
        for (j = 1; j < n; ++j) {
            if (used[j]) continue;
            if (best_idx == -1 || dist[tour[i - 1]][j] < dist[tour[i - 1]][best_idx]) {
                best_idx = j;
            }
        }
        tour[i] = best_idx;
        used[best_idx] = true;
    }
    delete[] used;
    return tour;
}


// Does local search over the problem until the time limit is reached
Tour local_search(const IntMatrix dists, const clock_t& clock_begin, float duration_seconds) {
    uint_fast16_t n = dists.size();
    // Start of with the greedy solution (in problem specs), then improve from there via SLS
    IntMatrix nn_matrix = get_nearest_neighbour_matrix(dists);
    Tour tour = greedy_nearest_neighbor(dists);

    // Compute cost for this tour, takes O(n)
    uint_fast32_t cost = compute_cost(n, tour, dists);

    // Perform 2-opt and 3 -top on tour
    Tour bestTour = tour;
    uint_fast32_t bestCost = cost;

    // Find finding our first local maxima, hope we get a pretty good one
    uint_fast16_t* position = new uint_fast16_t[n];
    uint_fast16_t i;
    uint_fast16_t num_iter_no_improvement = 0;
    bool no_early_stopping = false;

    // Diversification
    while (
        !is_time_exceeded(clock_begin, duration_seconds)
        && (no_early_stopping || num_iter_no_improvement < 400)
    ) {
        tour = double_bridge_move(n, tour);
        shuffle_tour(n, tour);

        // Next, perform subsidiary local search
        for (i = 0; i < n; ++i) {
            position[tour[i]] = i;
        }

        // Intensification
        two_opt(n, tour, dists, nn_matrix, position, clock_begin, duration_seconds);
        three_opt(n, tour, dists, nn_matrix, position, clock_begin, duration_seconds);
        cost = compute_cost(n, tour, dists);

        if (cost <= bestCost) {
            bestCost = cost;
            bestTour = tour;
            num_iter_no_improvement = 0;
        } else {
            tour = bestTour;
            num_iter_no_improvement++;
        }
    }
    return bestTour;
}

// Prints a given tour (linear fashion/order)
void print_tour(const Tour& tour) {
    for (int v : tour) {
        std::cout << v << "\n";
    }
}

// Performs a 2-opt move on a given tour, where we select 2 distinct edges based on
// nearest neighbour heuristic (and take the first improving perturbative neighbour)
// Terminates when there are no improving neighbours or when time limit exeeded.
void two_opt(uint_fast16_t n, Tour& tour, const IntMatrix& dists, const IntMatrix& nn_matrix, uint_fast16_t* position, const clock_t& clock_begin, float duration_seconds) {
    uint_fast16_t u1, u2, u3, u4;

    uint_fast16_t idx1 = 0;
    uint_fast16_t tempPos;
    uint_fast16_t tempIdx;
    bool improving_tsp = true; // Exit when no improving neighbours
    uint_fast16_t N;

    int d0, d1;

    // Exit SLS if no improving TSP (neighbours), i.e, we can't optimise the TSP any further
    while (!is_time_exceeded(clock_begin, duration_seconds) && improving_tsp) {
        improving_tsp = false;

        // Reset variables
        N = n;

        while (N--) {
            u1 = tour[idx1];
            tempIdx = (idx1 + 1) % n;
            u2 = tour[tempIdx];

            for (int idx2 : nn_matrix[u1]) {
                u3 = idx2;

                d0 = dists[u1][u2];
                d1 = dists[u1][u3];

                // All subsequent neighbours of u1 guaranteed to have longer distances
                if (d0 <= d1) break;

                tempPos = position[idx2];
                u4 = tour[(tempPos + 1) % n];

                if (u1 == u4 || u2 == u3) continue;

                // Using technique for fast w(p') computation from lecture slides
                if (d0 + dists[u3][u4] > d1 + dists[u2][u4]) {
                    reverse_tour(n, tour, tempIdx, tempPos, position);
                    improving_tsp = true;

                    break;
                }
            }

            idx1 = tempIdx; // Increment index by 1 (mod n)
        }
    }
}

// This randomly selects a subtour and shuffles it
// modifies tour in place
void shuffle_tour(uint_fast16_t n, Tour& tour) {
    if (n <= GEO_SHUFFLE_WIDTH) {
        shuffle(tour.begin(), tour.end(), rd);
        return;
    }
    std::uniform_int_distribution<size_t> random_offset(0, n - GEO_SHUFFLE_WIDTH); // [a, b] Inclusive
    int left = random_offset(rd);
    shuffle(tour.begin() + left, tour.begin() + left + GEO_SHUFFLE_WIDTH, rd); // [a, b) exclusive
    return;
}

// Performs a 3-opt move on a given tour, where we select 3 distinct edges based on
// nearest neighbour heuristic (and take the first improving perturbative neighbour)
void three_opt(uint_fast16_t n, Tour& tour, const IntMatrix& dists, const IntMatrix& nn_matrix, uint_fast16_t* position, const clock_t& clock_begin, float duration_seconds) {
    uint_fast16_t u1, u2, u3, u4, u5, u6;
    uint_fast16_t u1Idx, u2Idx, u3Idx, u4Idx, u5Idx, u6Idx;

    bool improving_tsp = true; // Exit when no improving neighbours
    while (!is_time_exceeded(clock_begin, duration_seconds) && improving_tsp) {
        improving_tsp = false;

        for (uint_fast16_t idx1 = 0; idx1 < n; ++idx1) {

            u1Idx = idx1;
            u2Idx = (idx1 + 1) % n;
            u1 = tour[u1Idx];
            u2 = tour[u2Idx];

            for (auto idx2 : nn_matrix[u2]) {
                u3 = idx2;
                u3Idx = position[u3];

                // All subsequent neighbours of guaranteed to have longer distances
                if (dists[u1][u2] <= dists[u2][u3]) break;

                if (u3 == u1) continue;

                u4Idx = (position[u3] + 1) % n;
                u4 = tour[u4Idx];

                if (u4Idx != u1Idx) {
                    for (auto idx3 : nn_matrix[u4]) {
                        u5 = idx3;
                        u5Idx = position[u5];

                        if (u5 == u3 || u5 == u1 || u5 == u2) continue;

                        if (!((u5Idx < u3Idx && u3Idx < u2Idx) ||
                            (u3Idx < u2Idx && u2Idx < u5Idx) ||
                            (u2Idx < u5Idx && u5Idx < u3Idx))) continue;

                        // All subsequent neighbours of guaranteed to have longer distances
                        if (dists[u3][u4] <= ((dists[u2][u3] - dists[u1][u2]) + dists[u4][u5])) break;

                        u6Idx = (position[u5] + 1) % n;
                        u6 = tour[u6Idx];

                        if (((((dists[u2][u3] - dists[u1][u2]) + dists[u4][u5]) - dists[u3][u4]) + dists[u1][u6]) < (dists[u5][u6])) {
                            improving_tsp = true;

                            swap_adj_seg(n, tour, position, u4Idx, u1Idx, u2Idx, u5Idx);

                            goto nextU1U2;
                        }
                    }
                }

                u4Idx = ((position[u3] - 1) + n) % n;
                u4 = tour[u4Idx];

                for (auto idx3 : nn_matrix[u4]) {
                    u5 = idx3;
                    u5Idx = position[u5];

                    if (u5 == u3 || u5 == u1 || u5 == u2) continue;

                    if (!((u5Idx < u3Idx && u3Idx < u2Idx) ||
                        (u3Idx < u2Idx && u2Idx < u5Idx) ||
                        (u2Idx < u5Idx && u5Idx < u3Idx))) continue;

                    // All subsequent neighbours of guaranteed to have longer distances
                    if (dists[u3][u4] <= ((dists[u2][u3] - dists[u1][u2]) + dists[u4][u5])) break;

                    u6Idx = (position[u5] + 1) % n;
                    u6 = tour[u6Idx];

                    if (((((dists[u2][u3] - dists[u1][u2]) + dists[u4][u5]) - dists[u3][u4]) + dists[u1][u6]) < (dists[u5][u6])) {
                        improving_tsp = true;

                        reverse_tour(n, tour, u6Idx, u4Idx, position);
                        reverse_tour(n, tour, u3Idx, u1Idx, position);

                        goto nextU1U2;
                    }
                }
            }
            nextU1U2: continue;
        }
    }
}

void swap_adj_seg(uint_fast16_t n, Tour& tour, uint_fast16_t* position, uint_fast16_t A, uint_fast16_t B, uint_fast16_t C, uint_fast16_t D) {
    Tour temp;

    uint_fast16_t cur = C;
    while (cur != D) {
        temp.push_back(tour[cur]);
        cur = (cur + 1) % n;
    }
    temp.push_back(tour[cur]); // temp contains segment [C .. D]

    cur = A;
    while (cur != B) {
        temp.push_back(tour[cur]);
        cur = (cur + 1) % n;
    }
    temp.push_back(tour[cur]); // temp contains segment [C .. D, A .. B]

    for (uint_fast16_t i = 0; i < temp.size(); ++i) {
        tour[(A + i) % n] = temp[i];
        position[temp[i]] = (A + i) % n;
    }
}


bool is_valid_tour(int n, const Tour tour) {
    if (tour.size() != n) return false;
    std::vector<bool> visited(n, false);
    for (int idx : tour) {
        if (idx >= n || visited[idx]) return false;
        visited[idx] = true;
    }
    return true;
}

// Computes the cost of a given tour, takes O(n)
uint_fast32_t compute_cost(uint_fast16_t n, const Tour& tour, const IntMatrix& dists) {
    uint_fast32_t cost = 0;
    uint_fast16_t prev_idx = tour[0];
    uint_fast16_t cur_idx;
    for (uint_fast16_t i = 1; i < n; ++i) {
        cur_idx = tour[i];
        cost += dists[prev_idx][cur_idx];
        prev_idx = cur_idx;
    }

    cost += dists[prev_idx][tour[0]]; // Close the loop
    return cost;
}

uint_fast32_t compute_cost_wrapper(const Tour& tour, const IntMatrix& dists) {
    return compute_cost(dists.size(), tour, dists);
}

double score(int cost, int opt_cost, int naive_cost) {
    // Returns a double in the range 0..1, where 0 denotes a solution that
    // is worse than or equivalent to the naive solution and 1 denotes a
    // solution that is equivalent to the optimal solution
    if (cost < opt_cost) throw std::invalid_argument(
        "Received cost < opt_cost, which cannot be true: "
        + std::to_string(cost) + " < " + std::to_string(opt_cost)
    );

    if (naive_cost == opt_cost) return cost == opt_cost ? 1.0 : 0.0;
    double actual_cost_diff = (double) cost - opt_cost;
    double naive_cost_diff = (double) naive_cost - opt_cost;
    double relative_diff = actual_cost_diff / naive_cost_diff;
    return std::pow(0.02, std::max(relative_diff, 0.0));
}

double score_tour(const Tour& tour, const IntMatrix& dists, int opt_cost) {
    uint_fast32_t cost = compute_cost(dists.size(), tour, dists);
    Tour greedy_tour = greedy_nearest_neighbor(dists);
    uint_fast32_t greedy_cost = compute_cost(dists.size(), greedy_tour, dists);
    return score(cost, opt_cost, greedy_cost);
}

#endif // TSP_FUNC_H
