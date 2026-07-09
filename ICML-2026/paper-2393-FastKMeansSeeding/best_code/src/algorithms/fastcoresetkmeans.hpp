#pragma once

#include "src/core/dataset.hpp"

#include <vector>
#include <random>
#include <algorithm>
#include <cassert>
#include <limits>
#include <cmath>
#include <numeric>
#include <unordered_map>
#include <stack>
#include <memory>

namespace fastcluster {

/* ============================================================
   CubeHST Node - Matching Python implementation
   Each node stores:
   - points (with index as first column)
   - weights
   - cell_path (list of ancestor node pointers)
   - children
   ============================================================ */
struct CubeHSTNode {
    std::vector<std::pair<size_t, std::vector<float>>> points;  // (index, coordinates)
    std::vector<float> weights;
    size_t depth = 0;
    size_t max_depth = 0;
    std::vector<CubeHSTNode*> cell_path;  // path from root to this node's parent
    bool is_root = false;
    bool marked = false;

    CubeHSTNode* parent = nullptr;
    std::vector<std::unique_ptr<CubeHSTNode>> children;

    std::vector<float> center;
    float edge_length = 0.0f;
    float diam = 0.0f;
    size_t n = 0;
    size_t d = 0;

    bool is_leaf() const { return children.empty(); }
    size_t size() const { return points.size(); }

    void mark() {
        marked = true;
    }
};

/* ============================================================
   CubeHST - Hierarchically Separated Tree (matching Python)
   ============================================================ */
class CubeHST {
public:
    explicit CubeHST(uint64_t seed = 42, size_t max_depth = 50) : rng_(seed), max_tree_depth_(max_depth) {}

    void build(const Dataset& X, const std::vector<float>& weights) {
        n_ = X.size();
        d_ = X.dim();

        // Create root node
        root_ = std::make_unique<CubeHSTNode>();
        root_->is_root = true;
        root_->depth = 0;

        // Store points with index as first "column" (matching Python)
        root_->points.resize(n_);
        root_->weights = weights;
        for (size_t i = 0; i < n_; ++i) {
            root_->points[i].first = i;
            root_->points[i].second.resize(d_);
            const float* p = X.row_ptr(i);
            for (size_t j = 0; j < d_; ++j) {
                root_->points[i].second[j] = p[j];
            }
        }

        root_->n = n_;
        root_->d = d_;

        // Get spread (max distance along any axis)
        root_->edge_length = get_spread(root_.get());

        // Random shift
        random_shift(root_.get());

        // Set center to (edge_length, edge_length, ..., edge_length)
        root_->center.assign(d_, root_->edge_length);

        // Diameter
        root_->diam = root_->edge_length * std::sqrt(static_cast<float>(d_)) * 2.0f;

        // Build tree recursively
        fit_tree(root_.get());
    }

    CubeHSTNode* root() { return root_.get(); }

    // Point to cell dictionary
    std::unordered_map<size_t, CubeHSTNode*>& ptc_dict() { return ptc_dict_; }

private:
    std::mt19937_64 rng_;
    size_t n_ = 0, d_ = 0;
    size_t max_tree_depth_ = 50;
    std::unique_ptr<CubeHSTNode> root_;
    std::unordered_map<size_t, CubeHSTNode*> ptc_dict_;

    float get_spread(CubeHSTNode* node) {
        if (node->points.empty()) return 1e-6f;

        std::vector<float> mins(d_, std::numeric_limits<float>::max());
        std::vector<float> maxs(d_, std::numeric_limits<float>::lowest());

        for (const auto& pt : node->points) {
            for (size_t j = 0; j < d_; ++j) {
                mins[j] = std::min(mins[j], pt.second[j]);
                maxs[j] = std::max(maxs[j], pt.second[j]);
            }
        }

        float max_spread = 0.0f;
        for (size_t j = 0; j < d_; ++j) {
            max_spread = std::max(max_spread, maxs[j] - mins[j]);
        }
        return std::max(max_spread, 1e-6f);
    }

    void random_shift(CubeHSTNode* node) {
        // Move pointset to all-positive values
        std::vector<float> mins(d_, std::numeric_limits<float>::max());
        for (const auto& pt : node->points) {
            for (size_t j = 0; j < d_; ++j) {
                mins[j] = std::min(mins[j], pt.second[j]);
            }
        }
        for (auto& pt : node->points) {
            for (size_t j = 0; j < d_; ++j) {
                pt.second[j] -= mins[j];
            }
        }

        // Apply random shift in [0, edge_length]
        std::uniform_real_distribution<float> unif(0.0f, node->edge_length);
        std::vector<float> shift(d_);
        for (size_t j = 0; j < d_; ++j) {
            shift[j] = unif(rng_);
        }
        for (auto& pt : node->points) {
            for (size_t j = 0; j < d_; ++j) {
                pt.second[j] += shift[j];
            }
        }
    }

    void fit_tree(CubeHSTNode* node) {
        // Base case: single point
        if (node->size() == 1) {
            node->max_depth = node->depth;
            ptc_dict_[node->points[0].first] = node;
            return;
        }

        // Base case: max depth reached - treat as leaf with multiple points
        if (node->depth >= max_tree_depth_) {
            node->max_depth = node->depth;
            for (const auto& pt : node->points) {
                ptc_dict_[pt.first] = node;
            }
            return;
        }

        // Compare each point to center, get direction vector in {-1, 1}^d
        // Use a map from direction signature to points
        std::unordered_map<std::string, std::vector<size_t>> direction_map;

        for (size_t i = 0; i < node->points.size(); ++i) {
            std::string dir_sig;
            for (size_t j = 0; j < d_; ++j) {
                dir_sig += (node->points[i].second[j] > node->center[j]) ? '1' : '0';
            }
            direction_map[dir_sig].push_back(i);
        }

        // new_cell_path = node.cell_path + [node]
        std::vector<CubeHSTNode*> new_cell_path = node->cell_path;
        new_cell_path.push_back(node);

        for (auto& kv : direction_map) {
            const std::string& dir_sig = kv.first;
            const std::vector<size_t>& point_indices = kv.second;

            auto child = std::make_unique<CubeHSTNode>();
            child->is_root = false;
            child->depth = node->depth + 1;
            child->cell_path = new_cell_path;
            child->edge_length = node->edge_length / 2.0f;
            child->d = d_;

            // Compute new center
            child->center.resize(d_);
            for (size_t j = 0; j < d_; ++j) {
                float dir = (dir_sig[j] == '1') ? 1.0f : -1.0f;
                child->center[j] = node->center[j] + dir * child->edge_length;
            }

            // Copy points and weights to child
            child->points.reserve(point_indices.size());
            child->weights.reserve(point_indices.size());
            for (size_t idx : point_indices) {
                child->points.push_back(node->points[idx]);
                child->weights.push_back(node->weights[idx]);
            }
            child->n = child->points.size();

            // Recurse
            fit_tree(child.get());

            // Set parent
            child->parent = node;

            // Update max_depth
            if (child->max_depth > node->max_depth) {
                node->max_depth = child->max_depth;
            }

            node->children.push_back(std::move(child));
        }
    }
};

/* ============================================================
   MultiHST - Multiple HSTs (matching Python)
   ============================================================ */
class MultiHST {
public:
    explicit MultiHST(size_t num_trees = 3, uint64_t seed = 42, size_t max_depth = 50)
        : num_trees_(num_trees), rng_(seed), max_depth_(max_depth) {}

    void build(const Dataset& X, const std::vector<float>& weights) {
        trees_.clear();
        trees_.reserve(num_trees_);

        for (size_t t = 0; t < num_trees_; ++t) {
            auto tree = std::make_unique<CubeHST>(rng_(), max_depth_);
            tree->build(X, weights);
            trees_.push_back(std::move(tree));
        }
    }

    size_t num_trees() const { return num_trees_; }

    CubeHSTNode* root(size_t i) { return trees_[i]->root(); }
    std::unordered_map<size_t, CubeHSTNode*>& ptc_dict(size_t i) { return trees_[i]->ptc_dict(); }

private:
    size_t num_trees_;
    std::mt19937_64 rng_;
    size_t max_depth_;
    std::vector<std::unique_ptr<CubeHST>> trees_;
};

/* ============================================================
   SampleTree - Binary tree for weighted sampling (matching Python)
   Uses cost propagation, not segment tree
   ============================================================ */
struct SampleTreeNode {
    std::vector<size_t> inds;
    float cost = -1.0f;  // -1 means not set yet
    SampleTreeNode* parent = nullptr;
    std::unique_ptr<SampleTreeNode> left_child;
    std::unique_ptr<SampleTreeNode> right_child;

    bool is_leaf() const { return inds.size() == 1; }
    bool has_parent() const { return parent != nullptr; }
};

class SampleTree {
public:
    void build(size_t n) {
        n_ = n;
        std::vector<size_t> inds(n);
        std::iota(inds.begin(), inds.end(), 0);
        root_ = create_sample_tree(inds);
    }

    SampleTreeNode* root() { return root_.get(); }

    // Point to cell dictionary
    std::unordered_map<size_t, SampleTreeNode*>& ptc_dict() { return ptc_dict_; }

    // Sample from tree
    size_t sample(std::mt19937_64& rng) {
        return multi_tree_sample(root_.get(), rng);
    }

    // Update cost (matching Python update_sample_tree)
    static void update_cost(SampleTreeNode* node, float cost_update) {
        // If we have not set this node's cost yet, set it to the current value
        if (node->cost == -1.0f) {
            node->cost = cost_update;
            if (node->has_parent()) {
                update_cost(node->parent, cost_update);
            }
        }
        // Otherwise, we have set this node's cost and want to update it
        else {
            if (node->is_leaf()) {
                float cost_delta = node->cost - cost_update;
                node->cost = cost_update;
                if (node->has_parent()) {
                    update_cost(node->parent, -1.0f * cost_delta);
                }
            }
            else {
                node->cost += cost_update;
                if (node->has_parent()) {
                    update_cost(node->parent, cost_update);
                }
            }
        }
    }

private:
    size_t n_ = 0;
    std::unique_ptr<SampleTreeNode> root_;
    std::unordered_map<size_t, SampleTreeNode*> ptc_dict_;

    std::unique_ptr<SampleTreeNode> create_sample_tree(const std::vector<size_t>& inds) {
        auto node = std::make_unique<SampleTreeNode>();
        node->inds = inds;

        if (inds.size() == 1) {
            ptc_dict_[inds[0]] = node.get();
            return node;
        }

        size_t split = inds.size() / 2;
        std::vector<size_t> left_inds(inds.begin(), inds.begin() + split);
        std::vector<size_t> right_inds(inds.begin() + split, inds.end());

        node->left_child = create_sample_tree(left_inds);
        node->right_child = create_sample_tree(right_inds);
        node->left_child->parent = node.get();
        node->right_child->parent = node.get();

        return node;
    }

    size_t multi_tree_sample(SampleTreeNode* node, std::mt19937_64& rng) {
        if (node->is_leaf()) {
            return node->inds[0];
        }

        float left_cost = node->left_child->cost;
        float right_cost = node->right_child->cost;

        // Handle uninitialized costs
        if (left_cost < 0) left_cost = 0;
        if (right_cost < 0) right_cost = 0;

        float total = left_cost + right_cost;
        if (total <= 0) {
            // Uniform random
            std::uniform_int_distribution<size_t> unif(0, n_ - 1);
            return unif(rng);
        }

        float left_prob = left_cost / total;
        std::uniform_real_distribution<float> unif01(0.0f, 1.0f);
        if (unif01(rng) < left_prob) {
            return multi_tree_sample(node->left_child.get(), rng);
        }
        return multi_tree_sample(node->right_child.get(), rng);
    }
};

/* ============================================================
   tree_dist function (matching Python)
   ============================================================ */
inline float tree_dist_func(float diam, size_t curr_depth, size_t max_depth) {
    return 4.0f * diam * (std::pow(0.5f, static_cast<float>(curr_depth)) -
                          std::pow(0.5f, static_cast<float>(max_depth)));
}

/* ============================================================
   mark_nodes - Walk up from leaf to find top unmarked node
   (matching Python)
   ============================================================ */
inline CubeHSTNode* mark_nodes(CubeHSTNode* node) {
    node->mark();
    if (node->parent == nullptr) {
        return node;
    }
    if (node->parent->marked) {
        return node;
    }
    return mark_nodes(node->parent);
}

/* ============================================================
   set_all_dists - Update distances for all points in subtree
   (matching Python exactly)
   ============================================================ */
inline void set_all_dists(
    SampleTree& sample_tree,
    std::vector<size_t>& labels,
    CubeHSTNode* curr_node,
    CubeHSTNode* center_node,
    size_t c,
    CubeHSTNode* root,
    int norm,
    float dist = 0.0f,
    size_t depth = 0
) {
    if (curr_node->is_leaf()) {
        // Handle leaf nodes (may have multiple points if max depth was reached)
        for (size_t i = 0; i < curr_node->points.size(); ++i) {
            size_t curr_point = curr_node->points[i].first;
            SampleTreeNode* sample_tree_node = sample_tree.ptc_dict()[curr_point];

            // Cost update needs to account for point weight
            float cost_update = dist * curr_node->weights[i];
            if (cost_update < sample_tree_node->cost || sample_tree_node->cost == -1.0f) {
                SampleTree::update_cost(sample_tree_node, cost_update);
                labels[curr_point] = c;
            }
        }
    }

    for (auto& child : curr_node->children) {
        float new_dist = dist;

        // If the center and the current point are in the same subtree, update their distance
        if (depth < center_node->cell_path.size() &&
            center_node->cell_path[depth] == child->cell_path[depth]) {
            float td = tree_dist_func(root->diam, depth, root->max_depth);
            new_dist = std::pow(td, static_cast<float>(norm));
        }
        if (child.get() == center_node) {
            new_dist = 0.0f;
        }

        set_all_dists(
            sample_tree,
            labels,
            child.get(),
            center_node,
            c,
            root,
            norm,
            new_dist,
            depth + 1
        );
    }
}

/* ============================================================
   multi_tree_open - Open a new center in all trees
   (matching Python)
   ============================================================ */
inline void multi_tree_open(
    MultiHST& multi_hst,
    size_t c,
    SampleTree& sample_tree,
    std::vector<size_t>& labels,
    int norm
) {
    for (size_t i = 0; i < multi_hst.num_trees(); ++i) {
        CubeHSTNode* root = multi_hst.root(i);
        auto& hst_ptc_dict = multi_hst.ptc_dict(i);

        CubeHSTNode* leaf = hst_ptc_dict[c];
        CubeHSTNode* top_unmarked = mark_nodes(leaf);
        set_all_dists(sample_tree, labels, top_unmarked, leaf, c, root, norm);
    }
}

/* ============================================================
   FastCoresetKMeansPP - Main algorithm class
   ============================================================ */
class FastCoresetKMeansPP {
public:
    explicit FastCoresetKMeansPP(uint64_t seed = 42)
        : rng_(seed) {}

    Dataset run(const Dataset& X, size_t k, size_t coreset_size) {
        const size_t n = X.size();
        const size_t d = X.dim();
        const int norm = 2;  // k-means uses norm=2

        assert(k <= n);
        assert(coreset_size >= k);

        /* ============================
           Phase 1: Fast K-Means++ via Multi-HST
           (matching Python fast_cluster_pp)
           ============================ */

        // Weights are all 1.0 for input points
        std::vector<float> weights(n, 1.0f);

        // Build Multi-HST (norm+1 = 3 trees for k-means)
        MultiHST multi_hst(norm + 1, rng_());
        multi_hst.build(X, weights);

        // Build sample tree
        SampleTree sample_tree;
        sample_tree.build(n);

        // Labels array (-1 means unassigned, but we use size_t so use SIZE_MAX)
        std::vector<size_t> labels(n, SIZE_MAX);

        // Centers
        std::vector<size_t> centers;
        centers.reserve(k);

        // Main loop: sample 2*k centers (matching Python)
        std::uniform_int_distribution<size_t> unif_idx(0, n - 1);
        for (size_t i = 0; i < 2 * k; ++i) {
            size_t c;
            if (centers.empty()) {
                c = unif_idx(rng_);
            } else {
                c = sample_tree.sample(rng_);
            }

            multi_tree_open(multi_hst, c, sample_tree, labels, norm);
            centers.push_back(c);
        }

        // Get costs from sample tree
        std::vector<float> costs(n);
        for (size_t i = 0; i < n; ++i) {
            SampleTreeNode* node = sample_tree.ptc_dict()[i];
            costs[i] = (node->cost >= 0) ? node->cost : 0.0f;
        }

        /* ============================
           Phase 2: Sensitivity Sampling
           (matching Python bound_sensitivities)
           ============================ */

        // Get cost per center
        std::vector<float> cost_per_center(centers.size(), 0.0f);
        for (size_t i = 0; i < centers.size(); ++i) {
            size_t center = centers[i];
            for (size_t j = 0; j < n; ++j) {
                if (labels[j] == center) {
                    cost_per_center[i] += costs[j];
                }
            }
        }

        // Compute sensitivities
        const float alpha = 10.0f;
        std::vector<float> sensitivities(n, 0.0f);

        for (size_t i = 0; i < centers.size(); ++i) {
            size_t center = centers[i];
            std::vector<size_t> points_in_cluster;
            for (size_t j = 0; j < n; ++j) {
                if (labels[j] == center) {
                    points_in_cluster.push_back(j);
                }
            }

            for (size_t j : points_in_cluster) {
                if (cost_per_center[i] > 0) {
                    sensitivities[j] = costs[j] / cost_per_center[i];
                }
                sensitivities[j] *= alpha;
                if (!points_in_cluster.empty()) {
                    sensitivities[j] += 1.0f / static_cast<float>(points_in_cluster.size());
                }
            }
        }

        // Normalize sensitivities
        float total_sens = 0.0f;
        for (float s : sensitivities) total_sens += s;
        if (total_sens > 0) {
            for (float& s : sensitivities) s /= total_sens;
        } else {
            for (float& s : sensitivities) s = 1.0f / static_cast<float>(n);
        }

        // Sample coreset
        std::discrete_distribution<size_t> sens_dist(sensitivities.begin(), sensitivities.end());

        Dataset coreset(coreset_size, d);
        std::vector<float> coreset_weights(coreset_size);

        for (size_t i = 0; i < coreset_size; ++i) {
            size_t idx = sens_dist(rng_);
            std::copy(X.row_ptr(idx), X.row_ptr(idx) + d, coreset.row_ptr(i));
            coreset_weights[i] = 1.0f / (static_cast<float>(coreset_size) * sensitivities[idx]);
        }

        // Normalize weights
        float weight_sum = 0.0f;
        for (float w : coreset_weights) weight_sum += w;
        if (weight_sum > 0) {
            for (float& w : coreset_weights) w *= static_cast<float>(n) / weight_sum;
        }

        /* ============================
           Phase 3: Weighted K-Means++ on Coreset
           ============================ */
        return weighted_kmeanspp(coreset, coreset_weights, k);
    }

private:
    std::mt19937_64 rng_;

    Dataset weighted_kmeanspp(
        const Dataset& coreset,
        const std::vector<float>& weights,
        size_t k
    ) {
        const size_t m = coreset.size();
        const size_t d = coreset.dim();

        Dataset centers(k, d);
        std::vector<float> dist2(m, std::numeric_limits<float>::max());

        std::discrete_distribution<size_t> weight_dist(weights.begin(), weights.end());
        size_t c1 = weight_dist(rng_);
        std::copy(coreset.row_ptr(c1), coreset.row_ptr(c1) + d, centers.row_ptr(0));

        for (size_t i = 0; i < m; ++i) {
            dist2[i] = Dataset::l2_sq(coreset.row_ptr(i), centers.row_ptr(0), d);
        }

        for (size_t c = 1; c < k; ++c) {
            std::vector<float> weighted_dist2(m);
            for (size_t i = 0; i < m; ++i) {
                weighted_dist2[i] = dist2[i] * weights[i];
            }

            std::discrete_distribution<size_t> d2_dist(weighted_dist2.begin(), weighted_dist2.end());
            size_t idx = d2_dist(rng_);

            std::copy(coreset.row_ptr(idx), coreset.row_ptr(idx) + d, centers.row_ptr(c));

            for (size_t i = 0; i < m; ++i) {
                float d2 = Dataset::l2_sq(coreset.row_ptr(i), centers.row_ptr(c), d);
                if (d2 < dist2[i]) {
                    dist2[i] = d2;
                }
            }
        }

        return centers;
    }
};

} // namespace fastcluster
