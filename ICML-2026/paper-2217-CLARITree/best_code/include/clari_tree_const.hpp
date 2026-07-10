#pragma once

#include <Eigen/Dense>
#include <cstddef>
#include <set>
#include <string>
#include <vector>

/*
Minimal public interface for constant-leaf trees with
both greedy splits (GreedyConst) and CLARITree splits (CLARITreeConst).
Implementation is in clari_tree_const.cpp
*/

class ConstNode {
public:
    ConstNode* left;                 // left child, if not leaf
    ConstNode* right;                // right child, if not leaf
    bool is_leaf;               // is leaf flag
    std::size_t n_instances;    // number of samples routed to this node
    double obj;                 // objective value at this node
    double threshold;           // split threshold
    int    feature_idx;         // split feature index
    double prediction;          // constant prediction at leaf

    ConstNode();
    ~ConstNode();

    // Rule-of-three
    ConstNode& operator=(const ConstNode& other);
    ConstNode(const ConstNode& other);

    // Pretty-print
    std::string print_tree(int indentation = 0);
};

class GreedyConst {
public:
    // Data (copied on fit)
    Eigen::MatrixXd X;
    Eigen::VectorXd y;

    bool  verbose;
    int   depth;                // max depth
    unsigned long int n;        // #samples
    unsigned long int m;        // #features
    double lambda;              // user penalty in [0,1]
    double scaled_lambda;       // dataset-specific penalty-per-leaf
    int n_thresholds;           // maximum number of thresholds per continuous feature
    std::string thresholds_strategy; // threshold generation strategy
    std::vector<int> continuous_idx_;  // indices in X (excluding intercept col 0)
    std::vector<int> binary_idx_;      // indices in X (excluding intercept col 0)
    std::vector<int> categorical_idx_; // optional, provided by caller (excluding intercept col 0)
    int min_leaf_node_size;      // requested minimum samples per leaf; <= 0 means 1
    double y_mean_ = 0.0; // mean of y (for centering)
    bool has_intercept_ = false; // whether X has intercept column at 0
    ConstNode*  root;                // root pointer

    explicit GreedyConst(int depth, double lambda = 0.0, int n_thresholds = 1, bool verbose = true, int min_leaf_node_size = 1);
    GreedyConst(int depth, double lambda, int n_thresholds, const std::string& thresholds_strategy, bool verbose = true, int min_leaf_node_size = 1);
    virtual ~GreedyConst();

    // Rule-of-three
    GreedyConst& operator=(const GreedyConst& other);
    GreedyConst(const GreedyConst& other);

    // Fit and predict
    double fit(Eigen::MatrixXd X, Eigen::VectorXd y, const std::vector<int>& categorical_idx = {});
    Eigen::VectorXd predict(Eigen::MatrixXd X);
    double predict_row(Eigen::VectorXd x);

    // Debug
    std::string print_tree();
    std::vector<std::vector<double>> get_traversed_thresholds() const;
    std::string print_traversed_thresholds() const;
    std::vector<std::vector<double>> get_threshold_pool() const;
    std::string print_threshold_pool() const;
    std::size_t n_leaves() const;

    static bool is_binary_column(const Eigen::VectorXd& col);
    void detect_feature_types();

private:
    static std::size_t count_leaves(const ConstNode* n);

protected:
    // Train-time helpers
    void fit_coefficients(ConstNode* node, Eigen::MatrixXd X, Eigen::VectorXd y);
    void resolve_min_leaf_node_size();
    bool children_respect_min_leaf_size(std::size_t left_count, std::size_t right_count) const;
    void reset_traversed_thresholds();
    void record_traversed_threshold(unsigned long int feature_idx, double threshold);
    void build_threshold_pool(const std::vector<std::vector<unsigned long int>>& sorted_indices);

    // Greedy recursive builder (virtual so CLARITreeConst can override)
    virtual double recursive_fit(std::vector<std::vector<unsigned long int>>& sorted_indices,
                                 int n, double y_sum, double y_sum_sq,
                                 ConstNode* node, int depth_remaining);

    // Sum-of-squares loss from mean
    static double loss(int n, double sum, double sum_sq);

    std::vector<std::set<double>> traversed_thresholds_;
    std::vector<std::vector<double>> threshold_pool_;
    std::size_t resolved_min_leaf_node_size_ = 1;
};

// CLARITreeConst: special case of CLARITree for constant-leaf trees
class CLARITreeConst : public GreedyConst {
public:
    explicit CLARITreeConst(int depth, double lambda = 0.0, int n_thresholds = 1, bool verbose = true, int min_leaf_node_size = 1);
    CLARITreeConst(int depth, double lambda, int n_thresholds, const std::string& thresholds_strategy, bool verbose = true, int min_leaf_node_size = 1);

protected:
    double recursive_fit(std::vector<std::vector<unsigned long int>>& sorted_indices,
                         int n, double y_sum, double y_sum_sq,
                         ConstNode* node, int depth_remaining) override;
};
