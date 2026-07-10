#ifndef CLARI_TREE_HPP
#define CLARI_TREE_HPP
#pragma once

#include <string>
#include <vector>
#include <set>
#include <Eigen/Dense>
#include <cstddef>
#include <limits>

// Depth is an int across the tree API.
using Depth = int;

//
// ========== Node ==========
// Represents one node in the regression tree
//
class Node {
public:
    Node* left;                   // left child (nullptr if leaf)
    Node* right;                  // right child (nullptr if leaf)
    bool is_leaf;                 // flag for leaf node
    std::size_t n_instances;      // number of samples routed to this node
    double obj;                   // objective value at this node
    double threshold;             // threshold used for splitting
    int feature_idx;              // feature index used for splitting
    Eigen::VectorXd coefficients; // ridge regression coefficients

    Node();
    ~Node();

    Node& operator=(const Node& other);   // copy assignment
    Node(const Node& other);              // copy constructor

    // std::string print_tree(int indentation = 0); 
    std::string print_tree(
        int indentation = 0,
        const std::vector<int>& continuous_idx = {}
    ) const; // print subtree
};

//
// ========== Greedy ==========
// Greedy regression tree with ridge regression in each leaf
//
class Greedy {
public:
    Eigen::MatrixXd X;   // feature matrix
    Eigen::VectorXd y;   // target vector
    double kappa;        // ridge regularization parameter
    bool verbose;        // verbosity flag
    Depth depth;         // maximum depth
    unsigned long int n; // number of samples
    unsigned long int m; // number of features
    double lambda;       // penalty on number of leaf nodes
    double scaled_kappa; // scaled by n
    double scaled_lambda;// scaled by TSS
    int n_thresholds;    // maximum number of thresholds per continuous feature
    std::string thresholds_strategy; // threshold generation strategy
    std::vector<int> continuous_idx_;  // indices in X (excluding intercept col 0)
    std::vector<int> binary_idx_;      // indices in X (excluding intercept col 0)
    std::vector<int> categorical_idx_; // optional, provided by caller (excluding intercept col 0)
    Eigen::MatrixXd X_reg_;   // [n x p_reg] = [1 | continuous features]
    int p_reg_ = 0;           // = 1 + continuous_idx_.size()
    int p_split_ = 0;         // = X.cols() (original with intercept)
    Eigen::VectorXd x_mean_;  // mean of continuous features (for standardization)
    Eigen::VectorXd x_std_;   // std of continuous features (for standardization)
    double y_mean_ = 0.0;     // mean of y (for centering)
    int min_leaf_node_size;   // requested minimum samples per leaf; <= 0 means auto
    double refine_kappa_factor; // factor to multiply kappa for final leaf coefficient refinement (1.0 = no refinement)
    Node* root;          // root node

    Greedy(double kappa, Depth depth, double lambda = 0.0, int n_thresholds = 1, bool verbose = true, int min_leaf_node_size = 0);
    Greedy(double kappa, Depth depth, double lambda, int n_thresholds, const std::string& thresholds_strategy, bool verbose = true, int min_leaf_node_size = 0);
    virtual ~Greedy();

    Greedy& operator=(const Greedy& other);
    Greedy(const Greedy& other);

    double fit(Eigen::MatrixXd X, Eigen::VectorXd y, const std::vector<int>& categorical_idx = {}); // fit tree, return objective
    void fit_coefficients(Node* node, Eigen::MatrixXd X, Eigen::VectorXd y);

    virtual double recursive_fit(
        std::vector<std::vector<unsigned long int>>& sorted_indices,
        Eigen::LLT<Eigen::MatrixXd>& llt,
        Eigen::VectorXd& b,
        double y_sum_sq,
        Node* node,
        Depth depth_remaining
    );

    // double loss(Eigen::MatrixXd L, Eigen::VectorXd b, double y_sum_sq);
    double loss(const Eigen::LLT<Eigen::MatrixXd>& llt,
                const Eigen::VectorXd& b,
                double y_sum_sq);
    Eigen::VectorXd predict(Eigen::MatrixXd X);
    double predict_row(Eigen::VectorXd x);

    std::string print_tree();
    std::vector<std::vector<double>> get_traversed_thresholds() const;
    std::string print_traversed_thresholds() const;
    std::vector<std::vector<double>> get_threshold_pool() const;
    std::string print_threshold_pool() const;
    std::size_t n_leaves() const;

    // --- NEW helpers---
    static bool is_binary_column(const Eigen::VectorXd& col);
    void detect_feature_types(); // fills *_idx_, sets p_reg_, p_split_

    // Row in regression space [1 | x_cont]
    Eigen::RowVectorXd reg_row(int i) const;

    // Full recompute (regression view) for a subset of rows
    std::tuple<Eigen::LLT<Eigen::MatrixXd>, Eigen::VectorXd, double>
    recompute_stats_from_rows(const std::vector<int>& rows);

protected:
    void resolve_min_leaf_node_size();
    bool children_respect_min_leaf_size(std::size_t left_count, std::size_t right_count) const;
    void reset_traversed_thresholds();
    void record_traversed_threshold(unsigned long int feature_idx, double threshold);
    void build_threshold_pool(const std::vector<std::vector<unsigned long int>>& sorted_indices);
    std::vector<std::set<double>> traversed_thresholds_;
    std::vector<std::vector<double>> threshold_pool_;
    std::size_t resolved_min_leaf_node_size_ = 1;

private:
    static std::size_t count_leaves(const Node* n);
    double recursive_fit_simplified_candidate(
        std::vector<std::vector<unsigned long int>> sorted_indices,
        Eigen::LLT<Eigen::MatrixXd> llt,
        Eigen::VectorXd b,
        double y_sum_sq,
        Node* node,
        Depth depth_remaining
    );

  
};

//
// ========== CLARITree ==========
// Main CLARITree algorithm with recursive splitting strategy
//
class CLARITree : public Greedy {
public:
    CLARITree(double kappa, Depth depth, double lambda = 0.0, int n_thresholds = 1, bool verbose = true, int min_leaf_node_size = 0);
    CLARITree(double kappa, Depth depth, double lambda, int n_thresholds, const std::string& thresholds_strategy, bool verbose = true, int min_leaf_node_size = 0);

    double recursive_fit(
        std::vector<std::vector<unsigned long int>>& sorted_indices,
        Eigen::LLT<Eigen::MatrixXd>& llt,
        Eigen::VectorXd& b,
        double y_sum_sq,
        Node* node,
        Depth depth_remaining
    ) override;
};

#endif // CLARI_TREE_HPP
