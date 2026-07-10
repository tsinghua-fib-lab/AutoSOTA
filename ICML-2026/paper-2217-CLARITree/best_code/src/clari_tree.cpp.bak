#include <iostream>
#include <iomanip>

#include <algorithm>
#include <vector>
#include <array>
#include <numeric>
#include <set>
#include <chrono>
#include <functional>
#include <sstream>
#include <cmath>

#include <Eigen/Dense>

#include "clari_tree.hpp"

using namespace Eigen;
using namespace Eigen::indexing;
using namespace std;

namespace {
    bool has_intercept_col(const Eigen::MatrixXd& X) {
        if (X.cols() == 0 || X.rows() == 0) {
            return false;
        }
        const double tol = 1e-12;
        return (X.col(0).array() - 1.0).abs().maxCoeff() <= tol;
    }

    void ensure_intercept_inplace(Eigen::MatrixXd& X) {
        if (has_intercept_col(X)) {
            return;
        }
        Eigen::MatrixXd X1(X.rows(), X.cols() + 1);
        X1.col(0).setOnes();
        X1.rightCols(X.cols()) = X;
        X.swap(X1);
    }

    bool is_standardized_cols(const Eigen::MatrixXd& X,
                            const std::vector<int>& cols,
                            double mean_tol,
                            double std_tol) {
        if (cols.empty()) return true;
        const int n = static_cast<int>(X.rows());
        if (n == 0) return true;
        for (int idx : cols) {
            const Eigen::VectorXd col = X.col(idx);
            const double mean = col.mean();
            const double var = (col.array() - mean).square().mean();
            const double std = std::sqrt(var);
            if (std < 1e-12) {
                // Constant column: treat as already standardized.
                continue;
            }
            if (std::abs(mean) > mean_tol || std::abs(std - 1.0) > std_tol) {
                return false;
            }
        }
        return true;
    }

    std::size_t node_instance_count_from_sorted_indices(
        const std::vector<std::vector<unsigned long int>>& sorted_indices,
        unsigned long int m) {
        for (unsigned long int feature = 1; feature < m; ++feature) {
            if (!sorted_indices[feature].empty()) {
                return sorted_indices[feature].size();
            }
        }
        return 0;
    }

    std::vector<double> collect_sorted_feature_values(
        const Eigen::MatrixXd& X,
        const std::vector<unsigned long int>& order,
        unsigned long int feature) {
        std::vector<double> values;
        values.reserve(order.size());
        for (unsigned long int row : order) {
            values.push_back(X((int)row, feature));
        }
        return values;
    }

    std::vector<double> deduplicate_sorted_values(const std::vector<double>& values) {
        std::vector<double> deduped;
        deduped.reserve(values.size());
        for (double value : values) {
            if (deduped.empty() || value != deduped.back()) {
                deduped.push_back(value);
            }
        }
        return deduped;
    }

    std::vector<double> adjacent_midpoints(const std::vector<double>& unique_values) {
        std::vector<double> midpoints;
        if (unique_values.size() < 2) {
            return midpoints;
        }
        midpoints.reserve(unique_values.size() - 1);
        for (std::size_t i = 0; i + 1 < unique_values.size(); ++i) {
            midpoints.push_back((unique_values[i] + unique_values[i + 1]) / 2.0);
        }
        return midpoints;
    }

    std::vector<double> make_uniform_thresholds(double min_value, double max_value, int max_thresholds) {
        if (max_thresholds <= 0 || min_value >= max_value) {
            return {};
        }

        std::vector<double> thresholds;
        thresholds.reserve(max_thresholds);
        const double step = (max_value - min_value) / static_cast<double>(max_thresholds);
        for (int k = 0; k < max_thresholds; ++k) {
            const double left = min_value + step * static_cast<double>(k);
            const double right = min_value + step * static_cast<double>(k + 1);
            thresholds.push_back((left + right) / 2.0);
        }
        return deduplicate_sorted_values(thresholds);
    }

    double empirical_quantile(const std::vector<double>& sorted_values, double probability) {
        if (sorted_values.empty()) {
            throw std::runtime_error("Cannot compute quantiles for an empty feature.");
        }
        if (sorted_values.size() == 1) {
            return sorted_values.front();
        }

        probability = std::clamp(probability, 0.0, 1.0);
        const double position = probability * static_cast<double>(sorted_values.size() - 1);
        const std::size_t lower_idx = static_cast<std::size_t>(std::floor(position));
        const std::size_t upper_idx = static_cast<std::size_t>(std::ceil(position));
        if (lower_idx == upper_idx) {
            return sorted_values[lower_idx];
        }

        const double weight = position - static_cast<double>(lower_idx);
        return sorted_values[lower_idx] +
               weight * (sorted_values[upper_idx] - sorted_values[lower_idx]);
    }

    std::vector<double> make_quantile_thresholds(const std::vector<double>& sorted_values, int max_thresholds) {
        if (max_thresholds <= 0 || sorted_values.empty()) {
            return {};
        }

        std::vector<double> thresholds;
        thresholds.reserve(max_thresholds);
        for (int k = 1; k <= max_thresholds; ++k) {
            const double probability = static_cast<double>(k) / static_cast<double>(max_thresholds + 1);
            thresholds.push_back(empirical_quantile(sorted_values, probability));
        }
        return deduplicate_sorted_values(thresholds);
    }

    std::vector<double> make_thresholds(
        const std::vector<double>& sorted_values,
        int max_thresholds,
        const std::string& strategy) {
        if (max_thresholds <= 0 || sorted_values.empty() || sorted_values.front() == sorted_values.back()) {
            return {};
        }

        const std::vector<double> unique_values = deduplicate_sorted_values(sorted_values);
        if (unique_values.size() <= static_cast<std::size_t>(max_thresholds + 1)) {
            return adjacent_midpoints(unique_values);
        }
        if (strategy == "uniform") {
            return make_uniform_thresholds(sorted_values.front(), sorted_values.back(), max_thresholds);
        }
        if (strategy == "quantile") {
            return make_quantile_thresholds(sorted_values, max_thresholds);
        }
        throw runtime_error("Unknown thresholds_strategy: " + strategy);
    }
}  // namespace

// Node 
// constructor
Node::Node()
    : left(nullptr), right(nullptr), is_leaf(true), n_instances(0), obj(0), threshold(0),
      feature_idx(0), coefficients(Eigen::VectorXd::Zero(1)) {}

// destructor
Node::~Node()
{
    delete left;
    delete right;
}

// assignment operator
Node &Node::operator=(const Node &other)
{
    if (this != &other)
    {
        // Clean up existing resources
        delete left;
        delete right;

        // Copy data from the other node
        left = other.left ? new Node(*other.left) : nullptr;
        right = other.right ? new Node(*other.right) : nullptr;
        is_leaf = other.is_leaf;
        n_instances = other.n_instances;
        obj = other.obj;
        threshold = other.threshold;
        feature_idx = other.feature_idx;
        coefficients = other.coefficients;
    }
    return *this;
}

// copy constructor
Node::Node(const Node &other)
{
    left = other.left ? new Node(*other.left) : nullptr;
    right = other.right ? new Node(*other.right) : nullptr;
    is_leaf = other.is_leaf;
    n_instances = other.n_instances;
    obj = other.obj;
    threshold = other.threshold;
    feature_idx = other.feature_idx;
    coefficients = other.coefficients;
}

// print the tree structure
std::string Node::print_tree(int indentation, const std::vector<int>& continuous_idx) const
{
    std::string indent(indentation, ' ');
    std::ostringstream oss;
    oss.setf(std::ios::fixed);
    oss << std::setprecision(6);
    if (is_leaf)
    {
        oss << indent << "Ridge leaf; obj = "
            << obj << "\n";
        if (!continuous_idx.empty() && coefficients.size() > 0)
        {
            oss << indent << "Continuous features: ";
            for (std::size_t k = 0; k < continuous_idx.size(); ++k)
            {
                if (k > 0) oss << ", ";
                oss << "x_" << continuous_idx[k];
            }
            oss << "\n";
            oss << indent << "Fit: " << coefficients(0);
            for (std::size_t k = 0; k < continuous_idx.size(); ++k)
            {
                int coef_idx = 1 + static_cast<int>(k);
                if (coef_idx >= coefficients.size())
                {
                    break;
                }
                double bj = coefficients(coef_idx);
                if (bj >= 0.0)
                {
                    oss << " + " << bj;
                }
                else
                {
                    oss << " - " << std::abs(bj);
                }
                oss << "*x_" << continuous_idx[k];
            }
            oss << "\n";
        }
        else
        {
            if (coefficients.size() > 0)
            {
                oss << indent << "Fit: " << coefficients(0) << "\n";
            }
            else
            {
                oss << indent << "No coefficients available.\n";
            }
        }
        return oss.str();
    }
    const std::size_t left_instances = left ? left->n_instances : 0;
    oss << indent << "If feature " << feature_idx
        << " <= " << threshold
        << " (n=" << left_instances << "):\n";
    if (left)
    {
        oss << left->print_tree(indentation + 2, continuous_idx);
    }
    const std::size_t right_instances = right ? right->n_instances : 0;
    oss << indent << "Else (n=" << right_instances << "):\n";
    if (right)
    {
        oss << right->print_tree(indentation + 2, continuous_idx);
    }
    return oss.str();
}

// Greedy
// constructor
Greedy::Greedy(double kappa, Depth depth, double lambda, int n_thresholds, bool verbose, int min_leaf_node_size)
    : Greedy(kappa, depth, lambda, n_thresholds, "quantile", verbose, min_leaf_node_size) {}

Greedy::Greedy(double kappa, Depth depth, double lambda, int n_thresholds, const std::string& thresholds_strategy, bool verbose, int min_leaf_node_size)
    : kappa(kappa),
      verbose(verbose),
      depth(depth),
      lambda(lambda),
      n_thresholds(n_thresholds),
      thresholds_strategy(thresholds_strategy),
      min_leaf_node_size(min_leaf_node_size),
      root(new Node()) {}

// destructor
Greedy::~Greedy()
{
    delete root;
}

// assignment operator
Greedy &Greedy::operator=(const Greedy &other)
{
    if (this != &other)
    {
        // Clean up existing resources
        delete root;

        // Copy data from the other tree
        X = other.X;
        y = other.y;
        kappa = other.kappa;
        scaled_kappa = other.scaled_kappa;
        verbose = other.verbose;
        depth = other.depth;
        lambda = other.lambda;
        scaled_lambda = other.scaled_lambda;
        n = other.n;
        m = other.m;
        min_leaf_node_size = other.min_leaf_node_size;
        root = other.root ? new Node(*other.root) : nullptr;
        n_thresholds = other.n_thresholds;
        thresholds_strategy = other.thresholds_strategy;
        continuous_idx_ = other.continuous_idx_;
        binary_idx_ = other.binary_idx_;
        categorical_idx_ = other.categorical_idx_;
        X_reg_ = other.X_reg_;
        p_reg_ = other.p_reg_;
        p_split_ = other.p_split_;
        x_mean_ = other.x_mean_;
        x_std_ = other.x_std_;
        y_mean_ = other.y_mean_;
        traversed_thresholds_ = other.traversed_thresholds_;
        threshold_pool_ = other.threshold_pool_;
        resolved_min_leaf_node_size_ = other.resolved_min_leaf_node_size_;
    }
    return *this;
}

// copy constructor
Greedy::Greedy(const Greedy &other)
    : X(other.X), y(other.y), kappa(other.kappa), scaled_kappa(other.scaled_kappa),
      verbose(other.verbose), depth(other.depth), n(other.n), m(other.m),
      lambda(other.lambda), scaled_lambda(other.scaled_lambda),
      n_thresholds(other.n_thresholds), thresholds_strategy(other.thresholds_strategy),
      min_leaf_node_size(other.min_leaf_node_size),
      continuous_idx_(other.continuous_idx_),
      binary_idx_(other.binary_idx_),
      categorical_idx_(other.categorical_idx_),
      X_reg_(other.X_reg_),
      p_reg_(other.p_reg_),
      p_split_(other.p_split_),
      x_mean_(other.x_mean_), x_std_(other.x_std_), y_mean_(other.y_mean_),
      traversed_thresholds_(other.traversed_thresholds_),
      threshold_pool_(other.threshold_pool_),
      resolved_min_leaf_node_size_(other.resolved_min_leaf_node_size_)
{
    root = other.root ? new Node(*other.root) : nullptr;
}

void Greedy::resolve_min_leaf_node_size() {
    const std::size_t default_min_leaf_node_size =
        std::max<std::size_t>(1, 5 * static_cast<std::size_t>(continuous_idx_.size()));
    resolved_min_leaf_node_size_ = min_leaf_node_size > 0
        ? static_cast<std::size_t>(min_leaf_node_size)
        : default_min_leaf_node_size;
}

bool Greedy::children_respect_min_leaf_size(std::size_t left_count, std::size_t right_count) const {
    return left_count >= resolved_min_leaf_node_size_ &&
           right_count >= resolved_min_leaf_node_size_;
}

void Greedy::reset_traversed_thresholds() {
    traversed_thresholds_.clear();
    traversed_thresholds_.resize(this->m);
}

void Greedy::record_traversed_threshold(unsigned long int feature_idx, double threshold) {
    if (feature_idx >= traversed_thresholds_.size()) {
        return;
    }
    traversed_thresholds_[feature_idx].insert(threshold);
}

void Greedy::build_threshold_pool(const std::vector<std::vector<unsigned long int>>& sorted_indices) {
    threshold_pool_.clear();
    threshold_pool_.resize(this->m);

    for (unsigned long int feature = 1; feature < this->m; ++feature) {
        const bool is_bin = std::find(binary_idx_.begin(), binary_idx_.end(), (int)feature) != binary_idx_.end();
        const bool is_cont = std::find(continuous_idx_.begin(), continuous_idx_.end(), (int)feature) != continuous_idx_.end();
        if (is_bin) {
            threshold_pool_[feature].push_back(0.5);
            continue;
        }
        if (!is_cont) {
            continue;
        }
        const auto& order = sorted_indices[feature];
        if (order.size() < 2) {
            continue;
        }

        const std::vector<double> sorted_values = collect_sorted_feature_values(this->X, order, feature);
        threshold_pool_[feature] = make_thresholds(sorted_values, n_thresholds, thresholds_strategy);
    }
}

double Greedy::fit(MatrixXd X, VectorXd y, const std::vector<int>& categorical_idx)
{
    ensure_intercept_inplace(X);
    this->X = X;
    this->n = X.rows();
    this->m = X.cols();
    reset_traversed_thresholds();
    this->categorical_idx_.clear();
    this->categorical_idx_.reserve(categorical_idx.size());
    for (int j_raw : categorical_idx) {
        this->categorical_idx_.push_back(j_raw + 1);  // shift for intercept
    }
    // detect feature types (based on original X)
    detect_feature_types();
    resolve_min_leaf_node_size();
    // scale by n
    this->scaled_lambda = this->n * this->lambda;
    // scale by TSS
    double mean_y = y.mean();
    double tss = (y.array() - mean_y).matrix().squaredNorm();
    this->scaled_lambda = this->lambda * tss;
    this->scaled_kappa = this->n * this->kappa;
    // Standardize continuous X and center y if needed
    const double mean_tol = 1e-6;
    const double std_tol = 1e-3;
    const bool x_is_standardized = is_standardized_cols(this->X, continuous_idx_, mean_tol, std_tol);
    const bool y_is_centered = std::abs(mean_y) <= mean_tol;
    const int cont_n = static_cast<int>(continuous_idx_.size());
    x_mean_.resize(cont_n);
    x_std_.resize(cont_n);
    for (int k = 0; k < cont_n; ++k) {
        const int j = continuous_idx_[k];
        const Eigen::VectorXd col = this->X.col(j);
        const double mu = col.mean();
        const double var = (col.array() - mu).square().mean();
        double sd = std::sqrt(var);
        if (sd < 1e-12) {
            sd = 1.0;
        }
        if (x_is_standardized) {
            x_mean_(k) = 0.0;
            x_std_(k) = 1.0;
        } else {
            x_mean_(k) = mu;
            x_std_(k) = sd;
        }
    }
    if (y_is_centered) {
        y_mean_ = 0.0;
        this->y = y;
    } else {
        y_mean_ = mean_y;
        this->y = y.array() - y_mean_;
    }
    // Build X_reg_ = [1 | standardized continuous columns]
    p_reg_ = 1 + cont_n;
    X_reg_.resize(this->n, p_reg_);
    X_reg_.col(0) = Eigen::VectorXd::Ones(this->n);
    for (int k = 0; k < cont_n; ++k) {
        const int j = continuous_idx_[k];
        X_reg_.col(1 + k) = (this->X.col(j).array() - x_mean_(k)) / x_std_(k);
    }
    if (p_reg_ <= 1) {
        throw std::runtime_error("Provide continuous columns (no non-binary, non-categorical numeric features found).");
    }

    // Root Cholesky built on X_reg_, not on full X
    MatrixXd gram = X_reg_.transpose() * X_reg_ + scaled_kappa * MatrixXd::Identity(p_reg_, p_reg_);
    gram(0,0) -= scaled_kappa - 1e-12;
    LLT<MatrixXd> lltOfA(gram);
    VectorXd b = X_reg_.transpose() * this->y;
    double y_sum_sq = this->y.squaredNorm();
    double parent_loss = Greedy::loss(lltOfA, b, y_sum_sq);

    delete this->root; // delete old root if exists
    this->root = new Node();
    this->root->obj = parent_loss + this->scaled_lambda;

    // get sorted feature indices
    // that is, for each feature, we want a row vector of indices sorted by feature value
    vector<vector<unsigned long int>> sorted_indices(this->m, vector<unsigned long int>(this->n));
    for (unsigned long int feature = 1; feature < this->m; feature++)
    {
        vector<unsigned long int> indices(this->n);
        iota(indices.begin(), indices.end(), 0); // fill with 0, 1, ..., n-1
        sort(indices.begin(), indices.end(), [&](unsigned long int a, unsigned long int b)
             { return X(a, feature) < X(b, feature); });
        sorted_indices[feature] = indices;
    }
    // Build fit-time global threshold pool (shared across all nodes).
    build_threshold_pool(sorted_indices);

    // learn partitioning structure and resulting loss
    double objective = recursive_fit(sorted_indices, lltOfA, b, y_sum_sq, this->root, this->depth);
    // Now learn the coefficient vectors for each of the nodes.
    fit_coefficients(this->root, this->X, this->y);

    return objective;
}

void Greedy::fit_coefficients(Node *node, MatrixXd X, VectorXd y)
{
    /*
    Fit the coefficients for the linear regression at this node. Only called after the tree structure is fully learned.
    */
    if (node->is_leaf)
    {
        // Build local X_reg from provided rows
        MatrixXd Xloc(X.rows(), p_reg_);
        Xloc.col(0).setOnes();
        for (int k = 0; k < (int)continuous_idx_.size(); ++k) {
            Xloc.col(1 + k) = (X.col(continuous_idx_[k]).array() - x_mean_(k)) / x_std_(k);
        }

        // fit ridge regression
        MatrixXd gram = Xloc.transpose() * Xloc + scaled_kappa * MatrixXd::Identity(Xloc.cols(), Xloc.cols());
        gram(0, 0) -= scaled_kappa - 1e-12;
        // node->coefficients = gram.inverse() * X.transpose() * y;
        LLT<MatrixXd> llt(gram);
        VectorXd beta_std = llt.solve(Xloc.transpose() * y);
        // check that coefficients lead to same sum sq error + lambda penalty as currently recorded in node->obj
        VectorXd yhat = Xloc * beta_std;
        double reg = 1e-12 * beta_std(0) * beta_std(0) + this->scaled_kappa * beta_std.tail(beta_std.size() - 1).squaredNorm(); // delete the penalty of intercept
        
        // Convert coefficients back to original scale for prediction/printing
        VectorXd beta_orig = beta_std;
        double intercept = y_mean_ + beta_std(0);
        for (int k = 0; k < (int)continuous_idx_.size(); ++k) {
            const double mu = x_mean_(k);
            const double sd = x_std_(k);
            const double bj = beta_std(1 + k);
            intercept -= bj * mu / sd;
            beta_orig(1 + k) = bj / sd;
        }
        beta_orig(0) = intercept;
        node->coefficients = beta_orig;

        // Debug check
        // double new_loss = (y - yhat).squaredNorm() + reg;
        // // double new_loss = (y - yhat).squaredNorm() + this->scaled_kappa * node->coefficients.squaredNorm();
        // double new_obj = new_loss + this->scaled_lambda;
        // double tol = 1e-10 * std::max(1.0, std::abs(node->obj));
        // if (abs(new_obj - node->obj) > tol)
        // {
        //     double diff = std::abs(new_obj - node->obj);
        //     cerr << "Warning: Fit coefficients in at least one leaf do not match recorded loss."
        //          << "Diff = " << diff << endl;
        // }
    }
    else
    {
        // Match the split convention used during search and prediction.
        vector<unsigned long int> left_indices;
        vector<unsigned long int> right_indices;
        for (unsigned long int i = 0; i < X.rows(); i++)
        {
            if (X(i, node->feature_idx) <= node->threshold)
            {
                left_indices.push_back(i);
            }
            else
            {
                right_indices.push_back(i);
            }
        }
        // fit coefficients for children
        fit_coefficients(node->left, X(left_indices, Eigen::all), y(left_indices));
        fit_coefficients(node->right, X(right_indices, Eigen::all), y(right_indices));
    }
    return;
}

double Greedy::recursive_fit(vector<vector<unsigned long int>>& sorted_indices, LLT<MatrixXd>& llt, VectorXd& b, double y_sum_sq, Node *node, Depth depth_remaining)
{
    /*
    Finds a greedy tree from the current node, stopping if depth is 0 or if no valid split is found.
    Returns the objective of the greedy tree from this node
    Assumes current node has loss value filled in with its loss if it were a leaf.
    */
    node->n_instances = node_instance_count_from_sorted_indices(sorted_indices, this->m);
    if (depth_remaining == 0 ||
        node->obj <= 2 * this->scaled_lambda ||
        node->n_instances < 2 * resolved_min_leaf_node_size_)
    {
        node->is_leaf = true;
        return node->obj;
    }

    // find best split
    bool split_flag = false;
    unsigned long int best_feature = 0;
    double best_threshold = 0.0;
    // double min_obj = node->obj; // start with parent loss
    double min_obj = std::numeric_limits<double>::infinity();
    LLT<MatrixXd> best_llt_right;
    LLT<MatrixXd> best_llt_left;
    VectorXd best_b_left;
    VectorXd best_b_right;
    vector<int> best_indices_left;
    double best_left_obj, best_right_obj;
    double best_y_sum_sq_left, best_y_sum_sq_right;

    for (unsigned long int feature = 1; feature < this->m; feature++)
    {   
        bool is_bin = std::find(binary_idx_.begin(), binary_idx_.end(), (int)feature) != binary_idx_.end();
        bool is_cont = std::find(continuous_idx_.begin(), continuous_idx_.end(), (int)feature) != continuous_idx_.end();

        // --- 1) Handle binary feature (0/1) ---------------------------------
        if (is_bin)
        {
            std::vector<int> left_rows, right_rows;
            left_rows.reserve(sorted_indices[feature].size());
            right_rows.reserve(sorted_indices[feature].size());

            // Split by 0/1 using only current node's samples
            for (unsigned long int row : sorted_indices[feature])
            {
                if (this->X((int)row, feature) <= 0.5)
                    left_rows.push_back((int)row);
                else
                    right_rows.push_back((int)row);
            }
            if (!children_respect_min_leaf_size(left_rows.size(), right_rows.size()))
                continue; // not splittable
            record_traversed_threshold(feature, 0.5);

            auto [lltL, bL, yssL] = recompute_stats_from_rows(left_rows);
            auto [lltR, bR, yssR] = recompute_stats_from_rows(right_rows);

            double left_obj = loss(lltL, bL, yssL) + this->scaled_lambda;
            double right_obj = loss(lltR, bR, yssR) + this->scaled_lambda;

            if (left_obj + right_obj < min_obj)
            {
                split_flag = true;
                min_obj = left_obj + right_obj;
                best_feature = feature;
                best_threshold = 0.5;
                best_llt_left = lltL;
                best_llt_right = lltR;
                best_b_left = bL;
                best_b_right = bR;
                best_left_obj = left_obj;
                best_right_obj = right_obj;
                best_indices_left.assign(left_rows.begin(), left_rows.end());
                best_y_sum_sq_left = yssL;
                best_y_sum_sq_right = yssR;
            }
            continue; // done with binary feature
        }

        // --- 2) Skip non-continuous (categorical not one-hot) ---------------
        if (!is_cont)
            continue;
        if (feature >= threshold_pool_.size())
            continue;
        const auto& feature_pool = threshold_pool_[feature];
        if (feature_pool.empty())
            continue;
        MatrixXd gram_left = this->scaled_kappa * MatrixXd::Identity(p_reg_, p_reg_);
        gram_left(0, 0) = 1e-12;
        VectorXd b_left = VectorXd::Zero(p_reg_);
        LLT<MatrixXd> llt_left(gram_left);
        double y_sum_sq_left = 0; // sum of squares of y values in left child
        vector<int> left_indices = {};
        VectorXd b_right = b;             // copy parent b
        LLT<MatrixXd> llt_right = llt;    // use parent llt
        double y_sum_sq_right = y_sum_sq; // copy parent y sum squared
        std::size_t pool_idx = 0;
        for (unsigned long int feature_idx = 0; feature_idx < sorted_indices[feature].size(); feature_idx++)
        {
            int row = sorted_indices[feature][feature_idx]; // get the row index for this feature & threshold
            b_left += reg_row(row).transpose() * this->y(row);
            llt_left.rankUpdate(reg_row(row), 1);
            y_sum_sq_left += this->y(row) * this->y(row);
            left_indices.push_back(row);

            b_right -= reg_row(row).transpose() * this->y(row);
            llt_right.rankUpdate(reg_row(row), -1);
            y_sum_sq_right -= this->y(row) * this->y(row);
            if (feature_idx == sorted_indices[feature].size() - 1)
            {
                // if this is the last feature index, we can't split further
                continue;
            }
            const double current_value = this->X(row, feature);
            const double next_value = this->X(sorted_indices[feature][feature_idx + 1], feature);
            if (current_value == next_value)
            {
                continue; // skip if this is not a valid split point
            }
            const std::size_t left_count = left_indices.size();
            const std::size_t right_count = sorted_indices[feature].size() - left_count;
            if (!children_respect_min_leaf_size(left_count, right_count))
            {
                continue;
            }

            while (pool_idx < feature_pool.size() && feature_pool[pool_idx] < current_value) {
                ++pool_idx;
            }

            std::size_t eval_idx = pool_idx;
            while (eval_idx < feature_pool.size() && feature_pool[eval_idx] < next_value) {
                const double candidate_threshold = feature_pool[eval_idx];
                record_traversed_threshold(feature, candidate_threshold);

                // double left_obj = loss(llt_left.matrixL(), b_left, y_sum_sq_left) + this->scaled_lambda;
                // double right_obj = loss(llt_right.matrixL(), b_right, y_sum_sq_right) + this->scaled_lambda;
                double left_obj = loss(llt_left, b_left, y_sum_sq_left) + this->scaled_lambda;
                double right_obj = loss(llt_right, b_right, y_sum_sq_right) + this->scaled_lambda;
                if (left_obj + right_obj < min_obj)
                {
                    split_flag = true;
                    min_obj = left_obj + right_obj;
                    best_feature = feature;
                    best_threshold = candidate_threshold;
                    best_llt_left = llt_left;
                    best_llt_right = llt_right;
                    best_b_left = b_left;
                    best_b_right = b_right;
                    best_left_obj = left_obj;
                    best_right_obj = right_obj;
                    best_indices_left = left_indices;
                    best_y_sum_sq_left = y_sum_sq_left;
                    best_y_sum_sq_right = y_sum_sq_right;
                }
                ++eval_idx;
            }
            pool_idx = eval_idx;
        }
    }

    if (!split_flag)
    {
        node->is_leaf = true; // no valid split found, this is a leaf
        return node->obj;     // return the loss at this node
    }
    // if the depth remaining is 1, we have find the best feature and don't need to sort indices for children
    if (depth_remaining == 1)
    {
        if (best_left_obj + best_right_obj >= node->obj)
        {
            node->is_leaf = true;
            return node->obj;
        }
        node->is_leaf = false;
        node->feature_idx = best_feature;
        node->threshold = best_threshold;

        node->left = new Node();
        node->right = new Node();
        node->left->obj = best_left_obj;
        node->right->obj = best_right_obj;
        const std::size_t left_count = static_cast<std::size_t>(best_indices_left.size());
        const std::size_t right_count = (node->n_instances >= left_count)
            ? (node->n_instances - left_count)
            : 0;
        node->left->n_instances = left_count;
        node->right->n_instances = right_count;

        node->obj = best_left_obj + best_right_obj;
        return node->obj;
    }

    // if valid split found, create children nodes to see if it's worth continuing to split
    node->left = new Node();
    node->left->obj = best_left_obj; // set left loss to minimum loss found
    node->right = new Node();
    node->right->obj = best_right_obj; // set right loss to minimum loss found

    // update sorted indices for children
    set<unsigned long int> left_indices_set(best_indices_left.begin(), best_indices_left.end());
    vector<vector<unsigned long int>> sorted_indices_left(this->m, vector<unsigned long int>());
    vector<vector<unsigned long int>> sorted_indices_right(this->m, vector<unsigned long int>());
    for (unsigned long int feature = 1; feature < this->m; feature++)
    {
        // for each feature, emplace back for left or right child based on which it's in
        for (unsigned long int idx : sorted_indices[feature])
        {
            if (left_indices_set.find(idx) != left_indices_set.end())
            {
                sorted_indices_left[feature].push_back(idx);
            }
            else
            {
                sorted_indices_right[feature].push_back(idx);
            }
        }
    }

    // now replace losses based on greedy completions
    double final_left_objective = Greedy::recursive_fit(sorted_indices_left, best_llt_left, best_b_left, best_y_sum_sq_left, node->left, depth_remaining - 1);
    double final_right_objective = Greedy::recursive_fit(sorted_indices_right, best_llt_right, best_b_right, best_y_sum_sq_right, node->right, depth_remaining - 1);

    if (final_left_objective + final_right_objective < node->obj)
    {
        node->is_leaf = false;
        node->feature_idx = best_feature;
        node->threshold = best_threshold;
        node->obj = final_left_objective + final_right_objective;
        return final_left_objective + final_right_objective; // return the objective at this node
    }
    else
    {
        delete node->left;    // delete left child if it was created
        delete node->right;   // delete right child if it was created
        node->left = nullptr;
        node->right = nullptr;
        node->is_leaf = true; // no valid split found, this is a leaf
        return node->obj;     // return the objective at this node
    }
}

/*
Loss using cholesky decomposition
Parameters:
- L, the lower triangular form of ch decomp for XTX + kappa(I)
- y_sum_sq, sum of squares of y values
- b, equal to X^Ty
\|y\|^2 - \|L^{-1} b\|^2
*/
double Greedy::loss(const LLT<MatrixXd>& llt,
                    const VectorXd& b,
                    double y_sum_sq)
{
    VectorXd z = llt.matrixL().solve(b);
    return y_sum_sq - z.squaredNorm();
}


VectorXd Greedy::predict(MatrixXd X)
{
    /*
    Predicts the output for the input matrix X using the fitted tree.
    Returns a vector of predictions.
    */
    ensure_intercept_inplace(X);
    if (X.cols() != this->m) {
        throw runtime_error("predict: X has unexpected number of columns (intercept is column 0).");
    }
    VectorXd predictions(X.rows());
    for (int i = 0; i < X.rows(); i++)
    {
        predictions(i) = predict_row(X.row(i));
    }
    return predictions;
}

double Greedy::predict_row(VectorXd x)
{
    if (x.size() == this->m - 1) {
        VectorXd x1(this->m);
        x1(0) = 1.0;
        x1.tail(this->m - 1) = x;
        x.swap(x1);
    } else if (x.size() != this->m) {
        throw runtime_error("predict_row: x has unexpected length (intercept is column 0).");
    }
    // Traverse the tree to make a prediction
    Node *current = this->root;
    while (current != nullptr)
    {
        if (current->is_leaf)
        {
            // return current->coefficients.transpose() * x;
            // assemble z = [1, x_cont]^T
            VectorXd z(p_reg_);
            z(0) = 1.0;
            for (int k = 0; k < (int)continuous_idx_.size(); ++k) {
                z(1 + k) = x(continuous_idx_[k]);
            }
            return current->coefficients.dot(z);
        }
        // Decide whether to go left or right
        if (x(current->feature_idx) <= current->threshold)
        {
            current = current->left;
        }
        else
        {
            current = current->right;
        }
    }
    // If we reach here, something went wrong
    throw runtime_error("Invalid tree structure");
}


std::string Greedy::print_tree()
{
    if (this->root == nullptr)
    {
        return "No tree currently fit!";
    }

    return this->root->print_tree(0, this->continuous_idx_);
}

std::size_t Greedy::count_leaves(const Node* n) {
    if (!n) return 0;
    if (n->is_leaf) return 1;
    return count_leaves(n->left) + count_leaves(n->right);
}

std::vector<std::vector<double>> Greedy::get_traversed_thresholds() const {
    std::vector<std::vector<double>> out(traversed_thresholds_.size());
    for (std::size_t j = 0; j < traversed_thresholds_.size(); ++j) {
        out[j].assign(traversed_thresholds_[j].begin(), traversed_thresholds_[j].end());
    }
    return out;
}

std::string Greedy::print_traversed_thresholds() const {
    std::ostringstream oss;
    oss.setf(std::ios::fixed);
    oss << std::setprecision(6);

    const auto all_thresholds = get_traversed_thresholds();
    if (all_thresholds.empty()) {
        return "No threshold traversal recorded. Fit the model first.\n";
    }
    for (std::size_t feature = 1; feature < all_thresholds.size(); ++feature) {
        oss << "feature " << feature << ": [";
        for (std::size_t i = 0; i < all_thresholds[feature].size(); ++i) {
            if (i > 0) {
                oss << ", ";
            }
            oss << all_thresholds[feature][i];
        }
        oss << "]\n";
    }
    return oss.str();
}

std::vector<std::vector<double>> Greedy::get_threshold_pool() const {
    return threshold_pool_;
}

std::string Greedy::print_threshold_pool() const {
    std::ostringstream oss;
    oss.setf(std::ios::fixed);
    oss << std::setprecision(6);

    if (threshold_pool_.empty()) {
        return "No threshold pool recorded. Fit the model first.\n";
    }
    for (std::size_t feature = 1; feature < threshold_pool_.size(); ++feature) {
        oss << "feature " << feature << ": [";
        for (std::size_t i = 0; i < threshold_pool_[feature].size(); ++i) {
            if (i > 0) {
                oss << ", ";
            }
            oss << threshold_pool_[feature][i];
        }
        oss << "]\n";
    }
    return oss.str();
}

std::size_t Greedy::n_leaves() const {
    return count_leaves(this->root);
}

bool Greedy::is_binary_column(const Eigen::VectorXd& col) {
    for (int i = 0; i < col.size(); ++i) {
        double v = col(i);
        if (!(v == 0.0 || v == 1.0)) return false;
    }
    return true;
}

void Greedy::detect_feature_types() {
    const int P = this->X.cols();     // includes intercept at 0
    p_split_ = P;

    // Mark which columns are categorical (we expect categorical_idx_ is already shifted in fit)
    std::vector<bool> is_categorical(P, false);
    for (int j : categorical_idx_) {
        if (j > 0 && j < P)            // ignore intercept (0) and out-of-range
            is_categorical[j] = true;
    }

    // scan features (skip intercept at 0)
    binary_idx_.clear();
    continuous_idx_.clear();
    for (int j = 1; j < P; ++j) {
        Eigen::VectorXd col = this->X.col(j);
        if (is_binary_column(col)) {
            binary_idx_.push_back(j);
        } else if (!is_categorical[j]) {
            continuous_idx_.push_back(j);
        } // else: multi-class categorical (not one-hot), will be skipped for splitting
    }

    // build X_reg_ later after standardization stats are known
    p_reg_ = 1 + (int)continuous_idx_.size();
}

Eigen::RowVectorXd Greedy::reg_row(int i) const {
    Eigen::RowVectorXd r(p_reg_);
    r(0) = 1.0; // intercept
    for (int k = 0; k < (int)continuous_idx_.size(); ++k) {
        r(1 + k) = (X(i, continuous_idx_[k]) - x_mean_(k)) / x_std_(k);
    }
    return r;
}


auto Greedy::recompute_stats_from_rows(const std::vector<int>& rows)
    -> std::tuple<Eigen::LLT<Eigen::MatrixXd, Eigen::Lower>, Eigen::VectorXd, double>
{
    MatrixXd gram = this->scaled_kappa * MatrixXd::Identity(p_reg_, p_reg_);
    gram(0,0) = 1e-12;
    VectorXd bb = VectorXd::Zero(p_reg_);
    double yss = 0.0;
    for (int r : rows) {
        auto xr = reg_row(r);
        bb.noalias()   += xr.transpose() * this->y(r);
        gram.noalias() += xr.transpose() * xr;
        yss += this->y(r) * this->y(r);
    }
    LLT<MatrixXd> lltmp(gram);
    return {lltmp, bb, yss};
}

// constructor
CLARITree::CLARITree(double kappa, Depth depth, double lambda, int n_thresholds, bool verbose, int min_leaf_node_size)
    : Greedy(kappa, depth, lambda, n_thresholds, verbose, min_leaf_node_size) {}

CLARITree::CLARITree(double kappa, Depth depth, double lambda, int n_thresholds, const std::string& thresholds_strategy, bool verbose, int min_leaf_node_size)
    : Greedy(kappa, depth, lambda, n_thresholds, thresholds_strategy, verbose, min_leaf_node_size) {}

double CLARITree::recursive_fit(vector<vector<unsigned long int>>& sorted_indices, LLT<MatrixXd>& llt, VectorXd& b, double y_sum_sq, Node *node, Depth depth_remaining)
{
    /*
    For every possible single step split, compute the loss using greedy,
    then pick the split locally the best with that heuristic.
    Then, replace those greedy calls with another CLARITree call
    */
    node->n_instances = node_instance_count_from_sorted_indices(sorted_indices, this->m);
    if (depth_remaining == 0 ||
        node->obj <= 2 * this->scaled_lambda ||
        node->n_instances < 2 * resolved_min_leaf_node_size_)
    {
        node->is_leaf = true;
        return node->obj; // return the objective at this node
    }

    // find best split
    bool split_flag = false;
    unsigned long int best_feature = 0;
    double best_threshold = 0.0;
    // double min_obj = node->obj; // start with parent loss
    double min_obj = std::numeric_limits<double>::infinity();
    LLT<MatrixXd> best_llt_right;
    LLT<MatrixXd> best_llt_left;
    VectorXd best_b_left;
    VectorXd best_b_right;
    vector<unsigned long int> best_indices_left;
    double best_left_leaf_obj, best_right_leaf_obj;
    double best_y_sum_sq_left, best_y_sum_sq_right;

    for (unsigned long int feature = 1; feature < this->m; feature++)
    {   
        bool is_bin = std::find(binary_idx_.begin(), binary_idx_.end(), (int)feature) != binary_idx_.end();
        bool is_cont = std::find(continuous_idx_.begin(), continuous_idx_.end(), (int)feature) != continuous_idx_.end();

        // --- 1) Handle binary feature (0/1) ---------------------------------
        if (is_bin)
        {
            std::vector<int> left_rows, right_rows;
            left_rows.reserve(sorted_indices[feature].size());
            right_rows.reserve(sorted_indices[feature].size());

            // Split by 0/1 using only current node's samples
            for (unsigned long int row : sorted_indices[feature])
            {
                if (this->X((int)row, feature) <= 0.5)
                    left_rows.push_back((int)row);
                else
                    right_rows.push_back((int)row);
            }
            if (!children_respect_min_leaf_size(left_rows.size(), right_rows.size()))
                continue; // not splittable
            record_traversed_threshold(feature, 0.5);

            auto [lltL, bL, yssL] = recompute_stats_from_rows(left_rows);
            auto [lltR, bR, yssR] = recompute_stats_from_rows(right_rows);

            double left_obj = loss(lltL, bL, yssL) + this->scaled_lambda;
            double right_obj = loss(lltR, bR, yssR) + this->scaled_lambda;

            if (depth_remaining == 1)
            {
                if (left_obj + right_obj < min_obj)
                {
                    split_flag = true;
                    min_obj = left_obj + right_obj;
                    best_feature = feature;
                    best_threshold = 0.5;
                    best_llt_left = lltL;
                    best_llt_right = lltR;
                    best_b_left = bL;
                    best_b_right = bR;
                    best_left_leaf_obj = left_obj;
                    best_right_leaf_obj = right_obj;
                    best_indices_left.assign(left_rows.begin(), left_rows.end());
                    best_y_sum_sq_left = yssL;
                    best_y_sum_sq_right = yssR;
                }
            }
            else
            {
                set<unsigned long int> left_indices_set(left_rows.begin(), left_rows.end());

                delete node->left;
                delete node->right;
                node->left = nullptr;
                node->right = nullptr;
                node->left = new Node();
                node->left->obj = left_obj;
                node->right = new Node();
                node->right->obj = right_obj;

                vector<vector<unsigned long int>> sorted_indices_left(this->m, vector<unsigned long int>());
                vector<vector<unsigned long int>> sorted_indices_right(this->m, vector<unsigned long int>());
                for (unsigned long int f2 = 1; f2 < this->m; f2++)
                {
                    for (unsigned long int idx : sorted_indices[f2])
                    {
                        if (left_indices_set.find(idx) != left_indices_set.end())
                        {
                            sorted_indices_left[f2].push_back(idx);
                        }
                        else
                        {
                            sorted_indices_right[f2].push_back(idx);
                        }
                    }
                }

                double greedy_completion_left_obj =
                    Greedy::recursive_fit(sorted_indices_left, lltL, bL, yssL, node->left, depth_remaining - 1);
                double greedy_completion_right_obj =
                    Greedy::recursive_fit(sorted_indices_right, lltR, bR, yssR, node->right, depth_remaining - 1);

                if (greedy_completion_left_obj + greedy_completion_right_obj < min_obj)
                {
                    split_flag = true;
                    min_obj = greedy_completion_left_obj + greedy_completion_right_obj;
                    best_feature = feature;
                    best_threshold = 0.5;
                    best_llt_left = lltL;
                    best_llt_right = lltR;
                    best_b_left = bL;
                    best_b_right = bR;
                    best_left_leaf_obj = left_obj;
                    best_right_leaf_obj = right_obj;
                    best_indices_left.assign(left_rows.begin(), left_rows.end());
                    best_y_sum_sq_left = yssL;
                    best_y_sum_sq_right = yssR;
                }
            }
            continue; // done with binary feature
        }

        // --- 2) Skip non-continuous (categorical not one-hot) ---------------
        if (!is_cont)
            continue;
        if (feature >= threshold_pool_.size())
            continue;
        const auto& feature_pool = threshold_pool_[feature];
        if (feature_pool.empty())
            continue;

        MatrixXd gram_left = this->scaled_kappa * MatrixXd::Identity(p_reg_, p_reg_);
        gram_left(0, 0) = 1e-12;
        VectorXd b_left = VectorXd::Zero(p_reg_);
        LLT<MatrixXd> llt_left(gram_left);
        double y_sum_sq_left = 0; // sum of squares of y values in left child
        vector<unsigned long int> left_indices = {};
        VectorXd b_right = b;             // copy parent b
        LLT<MatrixXd> llt_right = llt;    // use parent llt
        double y_sum_sq_right = y_sum_sq; // copy parent y sum squared
        std::size_t pool_idx = 0;
        for (unsigned long int feature_idx = 0; feature_idx < sorted_indices[feature].size(); feature_idx++)
        {
            int row = sorted_indices[feature][feature_idx]; // get the row index for this feature & threshold
            b_left += reg_row(row).transpose() * this->y(row);
            llt_left.rankUpdate(reg_row(row), 1);
            y_sum_sq_left += this->y(row) * this->y(row);
            left_indices.push_back(row);

            b_right -= reg_row(row).transpose() * this->y(row);
            llt_right.rankUpdate(reg_row(row), -1);
            y_sum_sq_right -= this->y(row) * this->y(row);
            if (feature_idx == sorted_indices[feature].size() - 1)
            {
                // if this is the last feature index, we can't split further
                continue;
            }
            const double current_value = this->X(row, feature);
            const double next_value = this->X(sorted_indices[feature][feature_idx + 1], feature);
            if (current_value == next_value)
            {
                // skip if this is not a valid split point
                continue;
            }
            const std::size_t left_count = left_indices.size();
            const std::size_t right_count = sorted_indices[feature].size() - left_count;
            if (!children_respect_min_leaf_size(left_count, right_count))
            {
                continue;
            }

            while (pool_idx < feature_pool.size() && feature_pool[pool_idx] < current_value) {
                ++pool_idx;
            }
            std::size_t eval_idx = pool_idx;
            while (eval_idx < feature_pool.size() && feature_pool[eval_idx] < next_value) {
                const double candidate_threshold = feature_pool[eval_idx];
                record_traversed_threshold(feature, candidate_threshold);
                // loss estimate based on greedy completion
                double left_obj = loss(llt_left, b_left, y_sum_sq_left) + this->scaled_lambda;
                double right_obj = loss(llt_right, b_right, y_sum_sq_right) + this->scaled_lambda;

                if (depth_remaining == 1)
                {
                    // if remaining depth=1 one can find the solution directly
                    if (left_obj + right_obj < min_obj)
                    {
                        split_flag = true;
                        min_obj = left_obj + right_obj;
                        best_feature = feature;
                        best_threshold = candidate_threshold;
                        best_llt_left = llt_left;
                        best_llt_right = llt_right;
                        best_b_left = b_left;
                        best_b_right = b_right;
                        best_left_leaf_obj = left_obj;
                        best_right_leaf_obj = right_obj;
                        best_indices_left = left_indices;
                        best_y_sum_sq_left = y_sum_sq_left;
                        best_y_sum_sq_right = y_sum_sq_right;
                    }
                }
                else
                {
                    // compute artifacts needed for greedy call.
                    set<unsigned long int> left_indices_set(left_indices.begin(), left_indices.end());

                    delete node->left;  // delete left child if it was created
                    delete node->right; // delete right child if it was created
                    node->left = nullptr;
                    node->right = nullptr;
                    node->left = new Node();
                    node->left->obj = left_obj; // set left obj to minimum obj found
                    node->right = new Node();
                    node->right->obj = right_obj; // set right obj to minimum obj found

                    // update sorted indices for children
                    vector<vector<unsigned long int>> sorted_indices_left(this->m, vector<unsigned long int>());
                    vector<vector<unsigned long int>> sorted_indices_right(this->m, vector<unsigned long int>());
                    for (unsigned long int feature = 1; feature < this->m; feature++)
                    {
                        // for each feature, emplace back for left or right child based on which it's in
                        for (unsigned long int idx : sorted_indices[feature])
                        {
                            if (left_indices_set.find(idx) != left_indices_set.end())
                            {
                                sorted_indices_left[feature].push_back(idx);
                            }
                            else
                            {
                                sorted_indices_right[feature].push_back(idx);
                            }
                        }
                    }

                    double greedy_completion_left_obj = Greedy::recursive_fit(sorted_indices_left, llt_left, b_left, y_sum_sq_left, node->left, depth_remaining - 1);
                    double greedy_completion_right_obj = Greedy::recursive_fit(sorted_indices_right, llt_right, b_right, y_sum_sq_right, node->right, depth_remaining - 1);
                    // lookahead tree in other papers, will this be much worse than us?
                    // double greedy_completion_left_obj = Greedy::recursive_fit(sorted_indices_left, llt_left, b_left, y_sum_sq_left, node->left, 1);
                    // double greedy_completion_right_obj = Greedy::recursive_fit(sorted_indices_right, llt_right, b_right, y_sum_sq_right, node->right, 1);

                    if (greedy_completion_left_obj + greedy_completion_right_obj < min_obj)
                    {
                        split_flag = true;
                        min_obj = greedy_completion_left_obj + greedy_completion_right_obj;
                        best_feature = feature;
                        best_threshold = candidate_threshold;
                        best_llt_left = llt_left;
                        best_llt_right = llt_right;
                        best_b_left = b_left;
                        best_b_right = b_right;
                        best_left_leaf_obj = left_obj;
                        best_right_leaf_obj = right_obj;
                        best_indices_left = left_indices;
                        best_y_sum_sq_left = y_sum_sq_left;
                        best_y_sum_sq_right = y_sum_sq_right;
                    }
                }
                ++eval_idx;
            }
            pool_idx = eval_idx;
        }
    }

    if (!split_flag)
    {
        node->is_leaf = true; // no valid split found, this is a leaf
        return node->obj;     // return the loss at this node
    }

    if (depth_remaining == 1)
    {   
        if (min_obj >= node->obj)
        {
            node->is_leaf = true;
            return node->obj;
        }
        node->is_leaf = false;
        node->feature_idx = best_feature;
        node->threshold = best_threshold;

        node->left = new Node();
        node->right = new Node();
        node->left->obj = best_left_leaf_obj;
        node->right->obj = best_right_leaf_obj;
        const std::size_t left_count = static_cast<std::size_t>(best_indices_left.size());
        const std::size_t right_count = (node->n_instances >= left_count)
            ? (node->n_instances - left_count)
            : 0;
        node->left->n_instances = left_count;
        node->right->n_instances = right_count;

        node->obj = node->left->obj + node->right->obj;
        return node->obj;
    }

    // if valid split found, create children nodes to see if it's worth continuing to split
    delete node->left;  // delete left child if it was created
    delete node->right; // delete right child if it was created
    node->left = nullptr;
    node->right = nullptr;
    node->left = new Node();
    node->left->obj = best_left_leaf_obj; // set left obj to minimum leaf obj found
    node->right = new Node();
    node->right->obj = best_right_leaf_obj; // set right obj to minimum leaf obj found

    // update sorted indices for children
    set<unsigned long int> left_indices_set(best_indices_left.begin(), best_indices_left.end());
    vector<vector<unsigned long int>> sorted_indices_left(this->m, vector<unsigned long int>());
    vector<vector<unsigned long int>> sorted_indices_right(this->m, vector<unsigned long int>());
    for (unsigned long int feature = 1; feature < this->m; feature++)
    {
        // for each feature, emplace back for left or right child based on which it's in
        for (unsigned long int idx : sorted_indices[feature])
        {
            if (left_indices_set.find(idx) != left_indices_set.end())
            {
                sorted_indices_left[feature].push_back(idx);
            }
            else
            {
                sorted_indices_right[feature].push_back(idx);
            }
        }
    }

    // now replace losses based on recursive completions (using CLARITree's approach, not the full greedy Greedy)
    double final_left_obj = recursive_fit(sorted_indices_left, best_llt_left, best_b_left, best_y_sum_sq_left, node->left, depth_remaining - 1);
    double final_right_obj = recursive_fit(sorted_indices_right, best_llt_right, best_b_right, best_y_sum_sq_right, node->right, depth_remaining - 1);

    if (final_left_obj + final_right_obj < node->obj)
    {
        node->is_leaf = false; // this node is not a leaf, we found a valid split
        node->feature_idx = best_feature;
        node->threshold = best_threshold;
        node->obj = final_left_obj + final_right_obj;
        return final_left_obj + final_right_obj; // return the obj at this node
    }
    else
    {
        delete node->left;    // delete left child if it was created
        delete node->right;   // delete right child if it was created
        node->left = nullptr;
        node->right = nullptr;
        node->is_leaf = true; // no valid split found, this is a leaf
        return node->obj;     // return the obj at this node
    }
}
