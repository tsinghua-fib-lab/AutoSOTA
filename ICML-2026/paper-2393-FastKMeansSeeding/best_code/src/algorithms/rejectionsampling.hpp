#pragma once

#include "src/core/dataset.hpp"

#include <vector>
#include <random>
#include <algorithm>
#include <cassert>
#include <limits>
#include <cmath>
#include <numeric>
#include <map>
#include <set>
#include <cstdint>

namespace rejectionsampling {

using std::vector;
using std::map;
using std::pair;
using std::set;

/* ============================================================
   RandomHandler - Global random number generator
   (matching fast_k_means_2020/random_handler.h)
   ============================================================ */
class RandomHandler {
public:
    static std::mt19937_64& eng() {
        static std::mt19937_64 engine(42);
        return engine;
    }

    static void seed(uint64_t s) {
        eng().seed(s);
    }
};

/* ============================================================
   PreProcessInputPoints - Preprocessing before tree construction
   (matching fast_k_means_2020/preprocess_input_points.cc)
   ============================================================ */
class PreProcessInputPoints {
public:
    // Scales the input points to integers by multiplying by scaling_factor
    static vector<vector<int>> ScaleToIntSpace(
        const vector<vector<double>>& input_point_double,
        double scaling_factor
    ) {
        if (input_point_double.empty()) return {};

        size_t n = input_point_double.size();
        size_t d = input_point_double[0].size();

        vector<vector<int>> input_points(n, vector<int>(d));

        for (size_t j = 0; j < d; j++) {
            for (size_t i = 0; i < n; i++) {
                input_points[i][j] = static_cast<int>(input_point_double[i][j] * scaling_factor);
            }
        }
        return input_points;
    }

    // Shifts the minimum coordinate to zero for each dimension
    static void ShiftToDimensionsZero(vector<vector<int>>* input_points) {
        if (input_points->empty()) return;

        size_t n = input_points->size();
        size_t d = (*input_points)[0].size();

        for (size_t j = 0; j < d; j++) {
            int min_coordinate = std::numeric_limits<int>::max();
            for (size_t i = 0; i < n; i++) {
                min_coordinate = std::min(min_coordinate, (*input_points)[i][j]);
            }
            for (size_t i = 0; i < n; i++) {
                (*input_points)[i][j] -= min_coordinate;
            }
        }
    }

    // Adds a random value to all points (doesn't affect objective)
    static void RandomShiftSpace(vector<vector<int>>* input_points) {
        if (input_points->empty()) return;

        size_t n = input_points->size();
        size_t d = (*input_points)[0].size();

        for (size_t j = 0; j < d; j++) {
            int max_coordinate = 0;
            for (size_t i = 0; i < n; i++) {
                max_coordinate = std::max(max_coordinate, (*input_points)[i][j]);
            }
            uint64_t shift = RandomHandler::eng()() % std::max(1, max_coordinate);
            for (size_t i = 0; i < n; i++) {
                (*input_points)[i][j] += static_cast<int>(shift);
            }
        }
    }
};

/* ============================================================
   TreeEmbedding - Hierarchical tree structure
   (matching fast_k_means_2020/tree_embedding.cc)
   ============================================================ */
class TreeEmbedding {
public:
    // Mapping of space coordinates to node ids at each level
    vector<map<vector<int>, int>> space_id;

    // Mapping of node ids to space coordinates at each level
    vector<map<int, vector<int>>> id_space;

    // Children of each node
    vector<vector<int>> children;

    // Height of the tree
    int height = 0;

    // Root node id
    int root = 0;

    // First unused id for new nodes
    int first_unused_id = 0;

    // Number of points in each node
    vector<int> number_points;

    // List of point indices in the subtree of each node
    vector<vector<int>> points_in_node;

    void BuildTree(const vector<vector<int>>& input_points) {
        if (input_points.empty()) return;

        // Initialize first level
        id_space.push_back(map<int, vector<int>>());
        space_id.push_back(map<vector<int>, int>());

        // Constructing the first layer of the tree
        for (size_t i = 0; i < input_points.size(); i++) {
            if (space_id[height].find(input_points[i]) == space_id[height].end()) {
                id_space[height][first_unused_id] = input_points[i];
                space_id[height][input_points[i]] = first_unused_id++;
                number_points.push_back(0);
                children.push_back(vector<int>());
                points_in_node.push_back(vector<int>());
            }
            int node_id = space_id[height][input_points[i]];
            number_points[node_id]++;
            points_in_node[node_id].push_back(static_cast<int>(i));
        }

        // Build higher levels until we reach a single root
        while (space_id[height].size() > 1) {
            id_space.push_back(map<int, vector<int>>());
            space_id.push_back(map<vector<int>, int>());

            for (const auto& e : space_id[height]) {
                vector<int> e_space = e.first;
                int e_int = e.second;

                // Divide coordinates by 2 to get parent cell
                for (size_t i = 0; i < e_space.size(); i++) {
                    e_space[i] /= 2;
                }

                if (space_id[height + 1].find(e_space) == space_id[height + 1].end()) {
                    id_space[height + 1][first_unused_id] = e_space;
                    space_id[height + 1][e_space] = first_unused_id++;
                    number_points.push_back(0);
                    children.push_back(vector<int>());
                    points_in_node.push_back(vector<int>());
                }

                int current_id = space_id[height + 1][e_space];
                number_points[current_id] += number_points[e_int];
                children[current_id].push_back(e_int);

                for (int point : points_in_node[e_int]) {
                    points_in_node[current_id].push_back(point);
                }
            }
            height++;
        }

        root = space_id[height++].begin()->second;
    }
};

/* ============================================================
   SingleTreeClustering - Single tree for distance computation
   (matching fast_k_means_2020/single_tree_clustering.cc)
   ============================================================ */
class SingleTreeClustering {
public:
    // Closest open center for each point
    vector<int> closets_open_center;

    void InitializeTree(const vector<vector<double>>& input, double scaling_factor) {
        // Preprocessing the input
        input_ = PreProcessInputPoints::ScaleToIntSpace(input, scaling_factor);
        PreProcessInputPoints::ShiftToDimensionsZero(&input_);
        PreProcessInputPoints::RandomShiftSpace(&input_);

        // Embedding the input to a tree
        tree_.BuildTree(input_);

        // Initialize closest centers to -1 (no center)
        closets_open_center = vector<int>(input_.size(), -1);
    }

    // Returns points and their new distances if center is opened
    vector<pair<int, uint64_t>> ComputeCostAndOpen(int center, bool open_center) {
        vector<pair<int, uint64_t>> updated_distances;
        set<int> updated_nodes;

        vector<int> center_coordinate = input_[center];

        for (int i = 0; i < tree_.height; i++) {
            // Find the node containing this center at level i
            auto it = tree_.space_id[i].find(center_coordinate);
            if (it == tree_.space_id[i].end()) break;

            int node = it->second;

            // If this node already has an open center, stop
            // (points at higher levels already have closer centers)
            if (has_open_center_[node]) break;

            if (open_center) {
                has_open_center_[node] = true;
            }

            // Update all points in this node
            for (int point : tree_.points_in_node[node]) {
                if (updated_nodes.find(point) == updated_nodes.end()) {
                    // Distance at level i is 2^(2i) for k-means (squared)
                    uint64_t dist = static_cast<uint64_t>(1) << (2 * i);
                    updated_distances.push_back(pair<int, uint64_t>(point, dist));

                    if (open_center) {
                        closets_open_center[point] = center;
                    }
                    updated_nodes.insert(point);
                }
            }

            // Move to parent level by dividing coordinates by 2
            for (size_t j = 0; j < center_coordinate.size(); j++) {
                center_coordinate[j] /= 2;
            }
        }

        return updated_distances;
    }

private:
    TreeEmbedding tree_;
    vector<vector<int>> input_;
    map<int, bool> has_open_center_;
};

/* ============================================================
   MultiTreeClustering - Multiple trees with binary tree sampling
   (matching fast_k_means_2020/multi_tree_clustering.cc)
   ============================================================ */
class MultiTreeClustering {
public:
    // Closest open center for each point
    vector<int> closets_open_center;

    // Distance to closest center for each point
    vector<uint64_t> distance_to_center;

    void InitializeTree(const vector<vector<double>>& input,
                        int number_of_trees,
                        double scaling_factor) {
        single_trees_ = vector<SingleTreeClustering>(number_of_trees);
        for (int i = 0; i < number_of_trees; i++) {
            single_trees_[i].InitializeTree(input, scaling_factor);
        }

        number_of_points_ = static_cast<int>(input.size());
        closets_open_center = vector<int>(number_of_points_);
        distance_to_center = vector<uint64_t>(number_of_points_,
                                               std::numeric_limits<uint64_t>::max());

        // Find binary tree boundary (next power of 2)
        binary_tree_boundary_ = 1;
        while (binary_tree_boundary_ < number_of_points_) {
            binary_tree_boundary_ *= 2;
        }

        // Initialize binary tree values to max
        binary_tree_value_ = vector<uint64_t>(2 * binary_tree_boundary_,
                                               std::numeric_limits<uint64_t>::max());
    }

    // Compute improvement from opening center, optionally open it
    uint64_t ComputeCostAndOpen(int center, bool open_center) {
        // Store old costs in case we don't actually open
        map<int, uint64_t> old_costs;
        uint64_t improvement = 0;

        // If this is just a query and no centers are open yet, return 0
        if (!open_center && distance_to_center[0] == std::numeric_limits<uint64_t>::max()) {
            return improvement;
        }

        // Get updates from all trees
        for (size_t i = 0; i < single_trees_.size(); i++) {
            for (auto& update : single_trees_[i].ComputeCostAndOpen(center, open_center)) {
                if (update.second < distance_to_center[update.first]) {
                    // Save old cost if not already saved
                    if (old_costs.find(update.first) == old_costs.end()) {
                        old_costs[update.first] = distance_to_center[update.first];
                    }

                    // Compute improvement
                    if (distance_to_center[update.first] != std::numeric_limits<uint64_t>::max()) {
                        improvement += distance_to_center[update.first] - update.second;
                    }

                    distance_to_center[update.first] = update.second;

                    if (open_center) {
                        closets_open_center[update.first] = center;
                        UpdateDistance(update, 0, binary_tree_boundary_, 1);
                    }
                }
            }
        }

        // If not actually opening, restore old costs
        if (!open_center) {
            for (auto& update : old_costs) {
                distance_to_center[update.first] = update.second;
            }
        }

        return improvement;
    }

    // Sample a point proportional to D^2
    int SampleAPoint() {
        // First point is sampled uniformly
        if (binary_tree_value_[1] == std::numeric_limits<uint64_t>::max()) {
            return static_cast<int>(RandomHandler::eng()() % number_of_points_);
        }

        uint64_t chosen_prob = RandomHandler::eng()() % binary_tree_value_[1];
        return SampleAPointRecurse(chosen_prob, 0, binary_tree_boundary_, 1);
    }

private:
    vector<SingleTreeClustering> single_trees_;
    vector<uint64_t> binary_tree_value_;
    int binary_tree_boundary_ = 0;
    int number_of_points_ = 0;

    void UpdateDistance(pair<int, uint64_t> update, int left, int right, int binary_tree_id) {
        // Leaf node
        if (left + 1 >= right) {
            binary_tree_value_[binary_tree_id] = update.second;
            return;
        }

        int middle = (left + right) / 2;
        if (update.first < middle) {
            UpdateDistance(update, left, middle, binary_tree_id * 2);
        } else {
            UpdateDistance(update, middle, right, binary_tree_id * 2 + 1);
        }

        // Update internal node
        uint64_t left_val = binary_tree_value_[binary_tree_id * 2];
        uint64_t right_val = binary_tree_value_[binary_tree_id * 2 + 1];

        if (left_val != std::numeric_limits<uint64_t>::max() &&
            right_val != std::numeric_limits<uint64_t>::max()) {
            binary_tree_value_[binary_tree_id] = left_val + right_val;
        } else {
            binary_tree_value_[binary_tree_id] = std::min(left_val, right_val);
        }
    }

    int SampleAPointRecurse(uint64_t chosen_prob, int left, int right, int binary_tree_id) {
        if (left + 1 >= right) return left;

        int middle = (left + right) / 2;
        uint64_t left_val = binary_tree_value_[binary_tree_id * 2];

        if (chosen_prob < left_val) {
            return SampleAPointRecurse(chosen_prob, left, middle, binary_tree_id * 2);
        }
        return SampleAPointRecurse(chosen_prob - left_val, middle, right, binary_tree_id * 2 + 1);
    }
};

/* ============================================================
   LSHDataStructure - Locality Sensitive Hashing
   (matching fast_k_means_2020/lsh.cc)
   Based on p-stable distributions (Datar, Immorlica, Indyk, Mirrokni)
   ============================================================ */
class LSHDataStructure {
public:
    LSHDataStructure(int bucket_size, int nb_bins, int dimension)
        : nb_bins_(nb_bins), r_(bucket_size) {

        std::normal_distribution<double> distrib(0.0, 1.0);

        for (int i = 0; i < nb_bins_; i++) {
            int offset = static_cast<int>(RandomHandler::eng()() % r_);
            vector<double> coordinates;
            coordinates.reserve(dimension);

            for (int j = 0; j < dimension; j++) {
                coordinates.push_back(distrib(RandomHandler::eng()));
            }

            projectors_.push_back(pair<int, vector<double>>(offset, coordinates));
            bins_collection_.push_back(map<int, vector<int>>());
        }
    }

    // Insert a point with given ID
    void InsertPoint(int id, const vector<double>& coordinates) {
        points_[id] = coordinates;

        vector<int> proj = Project(coordinates);
        points_to_bins_[id] = proj;

        for (int i = 0; i < nb_bins_; i++) {
            auto it = bins_collection_[i].find(proj[i]);
            if (it != bins_collection_[i].end()) {
                it->second.push_back(id);
            } else {
                vector<int> new_bin;
                new_bin.push_back(id);
                bins_collection_[i][proj[i]] = new_bin;
            }
        }
    }

    // Query for approximate nearest neighbor distance
    double QueryPoint(const vector<double>& coordinates, int running_time) {
        if (points_.empty()) {
            return std::numeric_limits<double>::max();
        }

        vector<int> proj = Project(coordinates);
        int nb_comparisons = 0;

        // Initialize with first point
        int id = points_.begin()->first;
        double min_dist = SqrDist(points_.begin()->second, coordinates);

        for (int i = 0; i < nb_bins_; i++) {
            auto it = bins_collection_[i].find(proj[i]);
            if (it == bins_collection_[i].end()) continue;

            for (size_t j = 0; j < it->second.size(); j++) {
                auto p = points_.find(it->second[j]);
                double d = SqrDist(coordinates, p->second);

                if (d < min_dist) {
                    min_dist = d;
                    id = p->first;
                    nb_comparisons++;
                }

                if (nb_comparisons > running_time) return min_dist;
            }
        }

        return min_dist;
    }

private:
    int nb_bins_;
    int r_;
    map<int, vector<double>> points_;
    vector<pair<int, vector<double>>> projectors_;
    vector<map<int, vector<int>>> bins_collection_;
    map<int, vector<int>> points_to_bins_;

    double SqrDist(const vector<double>& p1, const vector<double>& p2) {
        double d = 0;
        for (size_t i = 0; i < p1.size(); i++) {
            d += (p1[i] - p2[i]) * (p1[i] - p2[i]);
        }
        return d;
    }

    vector<int> Project(const vector<double>& coordinates) {
        vector<int> projections;

        for (int i = 0; i < nb_bins_; i++) {
            int b = projectors_[i].first;
            double c = 0;
            for (size_t j = 0; j < projectors_[i].second.size(); j++) {
                c += projectors_[i].second[j] * coordinates[j];
            }
            projections.push_back(static_cast<int>((c + b) / r_));
        }

        return projections;
    }
};

/* ============================================================
   RejectionSamplingLSH - Main algorithm
   (matching fast_k_means_2020/rejection_sampling_lsh.cc)
   ============================================================ */
class RejectionSamplingLSH {
public:
    vector<int> centers;

    void RunAlgorithm(const vector<vector<double>>& input,
                      int k,
                      int number_of_trees,
                      double scaling_factor,
                      int number_greedy_rounds,
                      double boosting_prob_factor) {

        int d = static_cast<int>(input[0].size());

        // Initialize LSH
        int size_lsh = 15;
        LSHDataStructure lsh(10, size_lsh, d);

        // Initialize multi-tree clustering
        multi_trees_.InitializeTree(input, number_of_trees, scaling_factor);

        double max_prob = 0.0;

        while (centers.size() < static_cast<size_t>(k)) {
            pair<int, uint64_t> best_center_and_improvement(0, 0);
            int number_sampled = 0;

            while (number_sampled < number_greedy_rounds) {
                // Sample a candidate center
                int next_center = multi_trees_.SampleAPoint();

                double prob = 1.0;
                if (!centers.empty()) {
                    // Compute acceptance probability using LSH distance estimate
                    double lsh_dist = lsh.QueryPoint(input[next_center], size_lsh);
                    double tree_dist = static_cast<double>(multi_trees_.distance_to_center[next_center]);

                    // Avoid division by zero
                    if (tree_dist > 0) {
                        prob = lsh_dist / (tree_dist * d / (scaling_factor * scaling_factor));
                        prob *= boosting_prob_factor;
                        max_prob = std::max(prob, max_prob);
                    }
                }

                // Rejection test
                double rand_val = static_cast<double>(RandomHandler::eng()()) /
                                  static_cast<double>(std::numeric_limits<uint64_t>::max());
                if (rand_val > prob) {
                    continue;  // Rejected
                }

                // Accepted - compute improvement
                uint64_t improvement = multi_trees_.ComputeCostAndOpen(next_center, false);

                if (improvement >= best_center_and_improvement.second) {
                    best_center_and_improvement.first = next_center;
                    best_center_and_improvement.second = improvement;
                }

                number_sampled++;
            }

            // Open the best center
            multi_trees_.ComputeCostAndOpen(best_center_and_improvement.first, true);
            centers.push_back(best_center_and_improvement.first);
            lsh.InsertPoint(best_center_and_improvement.first,
                           input[best_center_and_improvement.first]);
        }
    }

    vector<int> GetAssignment() {
        return multi_trees_.closets_open_center;
    }

private:
    MultiTreeClustering multi_trees_;
};

/* ============================================================
   RejectionSamplingKMeansPP - Wrapper class matching our interface
   ============================================================ */
class RejectionSamplingKMeansPP {
public:
    explicit RejectionSamplingKMeansPP(uint64_t seed = 42) {
        RandomHandler::seed(seed);
    }

    fastcluster::Dataset run(const fastcluster::Dataset& X, size_t k) {
        const size_t n = X.size();
        const size_t d = X.dim();

        // Convert Dataset to vector<vector<double>>
        vector<vector<double>> input(n, vector<double>(d));
        for (size_t i = 0; i < n; i++) {
            const float* row = X.row_ptr(i);
            for (size_t j = 0; j < d; j++) {
                input[i][j] = static_cast<double>(row[j]);
            }
        }

        // Parameters (matching fast_k_means_main.cc)
        int number_of_trees = 4;
        double scaling_factor = 1.0;
        int number_greedy_rounds = 1;
        double boosting_prob_factor = std::sqrt(static_cast<double>(d));

        // Run rejection sampling algorithm
        RejectionSamplingLSH algo;
        algo.RunAlgorithm(input, static_cast<int>(k), number_of_trees,
                          scaling_factor, number_greedy_rounds, boosting_prob_factor);

        // Convert center indices to Dataset
        fastcluster::Dataset centers(k, d);
        for (size_t i = 0; i < k; i++) {
            int center_idx = algo.centers[i];
            std::copy(X.row_ptr(center_idx), X.row_ptr(center_idx) + d, centers.row_ptr(i));
        }

        return centers;
    }

private:
    std::mt19937_64 rng_;
};

} // namespace rejectionsampling
