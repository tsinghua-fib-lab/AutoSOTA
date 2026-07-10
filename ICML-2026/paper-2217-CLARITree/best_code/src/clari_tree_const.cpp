#include <iostream>
#include <iomanip>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <limits>
#include <numeric>
#include <set>
#include <sstream>
#include <vector>

#include <Eigen/Dense>

#include "clari_tree_const.hpp"

using namespace Eigen;
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

ConstNode::ConstNode()
    : left(nullptr), right(nullptr), is_leaf(true), n_instances(0), obj(0), threshold(0),
      feature_idx(0), prediction(0.0) {}

ConstNode::~ConstNode()
{
    delete left;
    delete right;
}

ConstNode& ConstNode::operator=(const ConstNode& other) {
    if (this != &other) {
        delete left;
        delete right;

        left = other.left ? new ConstNode(*other.left) : nullptr;
        right = other.right ? new ConstNode(*other.right) : nullptr;
        is_leaf = other.is_leaf;
        n_instances = other.n_instances;
        obj = other.obj;
        threshold = other.threshold;
        feature_idx = other.feature_idx;
        prediction = other.prediction;
    }
    return *this;
}

ConstNode::ConstNode(const ConstNode& other) {
    left = other.left ? new ConstNode(*other.left) : nullptr;
    right = other.right ? new ConstNode(*other.right) : nullptr;
    is_leaf = other.is_leaf;
    n_instances = other.n_instances;
    obj = other.obj;
    threshold = other.threshold;
    feature_idx = other.feature_idx;
    prediction = other.prediction;
}

std::string ConstNode::print_tree(int indentation) {
    string indent(indentation, ' ');
    std::ostringstream oss;
    oss.setf(std::ios::fixed);
    oss << std::setprecision(6);
    if (is_leaf) {
        oss << indent << "Leaf const = " << prediction << "; objective " << obj << "\n";
        return oss.str();
    }

    oss << indent << "If feature " << feature_idx << " <= " << threshold << ":\n";
    if (left) {
        oss << left->print_tree(indentation + 2);
    }
    oss << indent << "Else:\n";
    if (right) {
        oss << right->print_tree(indentation + 2);
    }
    return oss.str();
}

GreedyConst::GreedyConst(int depth, double lambda, int n_thresholds, bool verbose, int min_leaf_node_size)
    : GreedyConst(depth, lambda, n_thresholds, "quantile", verbose, min_leaf_node_size) {}

GreedyConst::GreedyConst(int depth, double lambda, int n_thresholds, const std::string& thresholds_strategy, bool verbose, int min_leaf_node_size)
    : verbose(verbose),
      depth(depth),
      n(0),
      m(0),
      lambda(lambda),
      scaled_lambda(0.0),
      n_thresholds(n_thresholds),
      thresholds_strategy(thresholds_strategy),
      min_leaf_node_size(min_leaf_node_size),
      y_mean_(0.0),
      has_intercept_(false),
      root(new ConstNode()) {}

GreedyConst::~GreedyConst() {
    delete root;
}

GreedyConst& GreedyConst::operator=(const GreedyConst& other) {
    if (this != &other) {
        delete root;

        X = other.X;
        y = other.y;
        verbose = other.verbose;
        depth = other.depth;
        lambda = other.lambda;
        scaled_lambda = other.scaled_lambda;
        n = other.n;
        m = other.m;
        n_thresholds = other.n_thresholds;
        thresholds_strategy = other.thresholds_strategy;
        continuous_idx_ = other.continuous_idx_;
        binary_idx_ = other.binary_idx_;
        categorical_idx_ = other.categorical_idx_;
        min_leaf_node_size = other.min_leaf_node_size;
        y_mean_ = other.y_mean_;
        has_intercept_ = other.has_intercept_;
        traversed_thresholds_ = other.traversed_thresholds_;
        threshold_pool_ = other.threshold_pool_;
        resolved_min_leaf_node_size_ = other.resolved_min_leaf_node_size_;
        root = other.root ? new ConstNode(*other.root) : nullptr;
    }
    return *this;
}

GreedyConst::GreedyConst(const GreedyConst& other)
    : X(other.X),
      y(other.y),
      verbose(other.verbose),
      depth(other.depth),
      n(other.n),
      m(other.m),
      lambda(other.lambda),
      scaled_lambda(other.scaled_lambda),
      n_thresholds(other.n_thresholds),
      thresholds_strategy(other.thresholds_strategy),
      continuous_idx_(other.continuous_idx_),
      binary_idx_(other.binary_idx_),
      categorical_idx_(other.categorical_idx_),
      min_leaf_node_size(other.min_leaf_node_size),
      y_mean_(other.y_mean_),
      has_intercept_(other.has_intercept_),
      root(other.root ? new ConstNode(*other.root) : nullptr),
      traversed_thresholds_(other.traversed_thresholds_),
      threshold_pool_(other.threshold_pool_),
      resolved_min_leaf_node_size_(other.resolved_min_leaf_node_size_) {}

void GreedyConst::resolve_min_leaf_node_size() {
    resolved_min_leaf_node_size_ = min_leaf_node_size > 0
        ? static_cast<std::size_t>(min_leaf_node_size)
        : 1;
}

bool GreedyConst::children_respect_min_leaf_size(std::size_t left_count, std::size_t right_count) const {
    return left_count >= resolved_min_leaf_node_size_ &&
           right_count >= resolved_min_leaf_node_size_;
}

void GreedyConst::reset_traversed_thresholds() {
    traversed_thresholds_.clear();
    traversed_thresholds_.resize(this->m);
}

void GreedyConst::record_traversed_threshold(unsigned long int feature_idx, double threshold) {
    if (feature_idx >= traversed_thresholds_.size()) {
        return;
    }
    traversed_thresholds_[feature_idx].insert(threshold);
}

bool GreedyConst::is_binary_column(const Eigen::VectorXd& col) {
    for (int i = 0; i < col.size(); ++i) {
        const double v = col(i);
        if (!(v == 0.0 || v == 1.0)) {
            return false;
        }
    }
    return true;
}

void GreedyConst::detect_feature_types() {
    const int P = this->X.cols();

    std::vector<bool> is_categorical(P, false);
    for (int j : categorical_idx_) {
        if (j > 0 && j < P) {
            is_categorical[j] = true;
        }
    }

    binary_idx_.clear();
    continuous_idx_.clear();
    for (int j = 1; j < P; ++j) {
        Eigen::VectorXd col = this->X.col(j);
        if (is_binary_column(col)) {
            binary_idx_.push_back(j);
        } else if (!is_categorical[j]) {
            continuous_idx_.push_back(j);
        }
    }
}

void GreedyConst::build_threshold_pool(
    const std::vector<std::vector<unsigned long int>>& sorted_indices) {
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

double GreedyConst::fit(MatrixXd X, VectorXd y, const std::vector<int>& categorical_idx) {
    ensure_intercept_inplace(X);
    this->X = X;
    this->n = X.rows();
    this->m = X.cols();
    this->has_intercept_ = has_intercept_col(this->X);
    reset_traversed_thresholds();

    this->categorical_idx_.clear();
    this->categorical_idx_.reserve(categorical_idx.size());
    for (int j_raw : categorical_idx) {
        this->categorical_idx_.push_back(j_raw + 1);
    }
    detect_feature_types();
    resolve_min_leaf_node_size();

    const double mean_y = y.mean();
    const double mean_tol = 1e-6;
    if (std::abs(mean_y) <= mean_tol) {
        this->y_mean_ = 0.0;
        this->y = y;
    } else {
        this->y_mean_ = mean_y;
        this->y = y.array() - this->y_mean_;
    }

    double tss = (y.array() - mean_y).matrix().squaredNorm();
    this->scaled_lambda = this->lambda * tss;

    double y_sum = this->y.sum();
    double y_sum_sq = this->y.squaredNorm();
    double parent_loss = GreedyConst::loss(static_cast<int>(this->n), y_sum, y_sum_sq);

    delete this->root;
    this->root = new ConstNode();
    this->root->obj = parent_loss + this->scaled_lambda;

    vector<vector<unsigned long int>> sorted_indices(this->m, vector<unsigned long int>(this->n));
    for (unsigned long int feature = 1; feature < this->m; feature++) {
        vector<unsigned long int> indices(this->n);
        iota(indices.begin(), indices.end(), 0);
        sort(indices.begin(), indices.end(), [&](unsigned long int a, unsigned long int b) {
            return this->X(a, feature) < this->X(b, feature);
        });
        sorted_indices[feature] = indices;
    }
    build_threshold_pool(sorted_indices);

    double objective = recursive_fit(
        sorted_indices,
        static_cast<int>(this->n),
        y_sum,
        y_sum_sq,
        this->root,
        this->depth);
    fit_coefficients(this->root, this->X, this->y);

    return objective;
}

void GreedyConst::fit_coefficients(ConstNode* node, MatrixXd X, VectorXd y) {
    if (node->is_leaf) {
        node->prediction = y.size() > 0 ? y.sum() / y.size() + this->y_mean_ : this->y_mean_;
        return;
    }

    vector<unsigned long int> left_indices;
    vector<unsigned long int> right_indices;
    for (unsigned long int i = 0; i < X.rows(); i++) {
        if (X(i, node->feature_idx) <= node->threshold) {
            left_indices.push_back(i);
        } else {
            right_indices.push_back(i);
        }
    }
    fit_coefficients(node->left, X(left_indices, Eigen::all), y(left_indices));
    fit_coefficients(node->right, X(right_indices, Eigen::all), y(right_indices));
}

double GreedyConst::recursive_fit(vector<vector<unsigned long int>>& sorted_indices, int n, double y_sum, double y_sum_sq, ConstNode* node, int depth_remaining) {
    node->n_instances = node_instance_count_from_sorted_indices(sorted_indices, this->m);
    if (depth_remaining == 0 ||
        node->obj <= 2 * this->scaled_lambda ||
        static_cast<std::size_t>(n) < 2 * resolved_min_leaf_node_size_) {
        node->is_leaf = true;
        return node->obj;
    }

    bool split_flag = false;
    unsigned long int best_feature = 0;
    double best_threshold = 0.0;
    double min_obj = std::numeric_limits<double>::infinity();
    vector<unsigned long int> best_indices_left;
    double best_left_obj = 0.0, best_right_obj = 0.0;
    double best_y_sum_sq_left = 0.0, best_y_sum_sq_right = 0.0;
    double best_y_sum_left = 0.0, best_y_sum_right = 0.0;
    int best_n_left = 0, best_n_right = 0;

    for (unsigned long int feature = 1; feature < this->m; feature++) {
        const bool is_bin = std::find(binary_idx_.begin(), binary_idx_.end(), (int)feature) != binary_idx_.end();
        const bool is_cont = std::find(continuous_idx_.begin(), continuous_idx_.end(), (int)feature) != continuous_idx_.end();

        if (is_bin) {
            vector<unsigned long int> left_rows;
            vector<unsigned long int> right_rows;
            left_rows.reserve(sorted_indices[feature].size());
            right_rows.reserve(sorted_indices[feature].size());
            double yss_left = 0.0, yss_right = 0.0;
            double sum_left = 0.0, sum_right = 0.0;

            for (unsigned long int row : sorted_indices[feature]) {
                if (this->X((int)row, feature) <= 0.5) {
                    left_rows.push_back(row);
                    sum_left += this->y(row);
                    yss_left += this->y(row) * this->y(row);
                } else {
                    right_rows.push_back(row);
                    sum_right += this->y(row);
                    yss_right += this->y(row) * this->y(row);
                }
            }
            if (!children_respect_min_leaf_size(left_rows.size(), right_rows.size())) {
                continue;
            }
            record_traversed_threshold(feature, 0.5);

            const int n_left = static_cast<int>(left_rows.size());
            const int n_right = static_cast<int>(right_rows.size());
            double left_obj = loss(n_left, sum_left, yss_left) + this->scaled_lambda;
            double right_obj = loss(n_right, sum_right, yss_right) + this->scaled_lambda;

            if (left_obj + right_obj < min_obj) {
                split_flag = true;
                min_obj = left_obj + right_obj;
                best_feature = feature;
                best_threshold = 0.5;
                best_left_obj = left_obj;
                best_right_obj = right_obj;
                best_indices_left = left_rows;
                best_y_sum_sq_left = yss_left;
                best_y_sum_sq_right = yss_right;
                best_y_sum_left = sum_left;
                best_y_sum_right = sum_right;
                best_n_left = n_left;
                best_n_right = n_right;
            }
            continue;
        }

        if (!is_cont) {
            continue;
        }
        if (feature >= threshold_pool_.size()) {
            continue;
        }
        const auto& feature_pool = threshold_pool_[feature];
        if (feature_pool.empty()) {
            continue;
        }

        double y_sum_sq_left = 0.0;
        double y_sum_left = 0.0;
        int n_left = 0;
        vector<unsigned long int> left_indices;

        double y_sum_sq_right = y_sum_sq;
        double y_sum_right = y_sum;
        int n_right = n;
        std::size_t pool_idx = 0;

        for (unsigned long int feature_idx = 0; feature_idx < sorted_indices[feature].size(); feature_idx++) {
            const unsigned long int row = sorted_indices[feature][feature_idx];
            y_sum_sq_left += this->y(row) * this->y(row);
            y_sum_left += this->y(row);
            left_indices.push_back(row);
            n_left++;

            y_sum_sq_right -= this->y(row) * this->y(row);
            y_sum_right -= this->y(row);
            n_right--;

            if (feature_idx == sorted_indices[feature].size() - 1) {
                continue;
            }

            const double current_value = this->X(row, feature);
            const double next_value = this->X(sorted_indices[feature][feature_idx + 1], feature);
            if (current_value == next_value) {
                continue;
            }
            if (!children_respect_min_leaf_size(left_indices.size(), static_cast<std::size_t>(n_right))) {
                continue;
            }

            while (pool_idx < feature_pool.size() && feature_pool[pool_idx] < current_value) {
                ++pool_idx;
            }

            std::size_t eval_idx = pool_idx;
            while (eval_idx < feature_pool.size() && feature_pool[eval_idx] < next_value) {
                const double candidate_threshold = feature_pool[eval_idx];
                record_traversed_threshold(feature, candidate_threshold);

                double left_obj = loss(n_left, y_sum_left, y_sum_sq_left) + this->scaled_lambda;
                double right_obj = loss(n_right, y_sum_right, y_sum_sq_right) + this->scaled_lambda;
                if (left_obj + right_obj < min_obj) {
                    split_flag = true;
                    min_obj = left_obj + right_obj;
                    best_feature = feature;
                    best_threshold = candidate_threshold;
                    best_left_obj = left_obj;
                    best_right_obj = right_obj;
                    best_indices_left = left_indices;
                    best_y_sum_sq_left = y_sum_sq_left;
                    best_y_sum_sq_right = y_sum_sq_right;
                    best_y_sum_left = y_sum_left;
                    best_y_sum_right = y_sum_right;
                    best_n_left = n_left;
                    best_n_right = n_right;
                }
                ++eval_idx;
            }
            pool_idx = eval_idx;
        }
    }

    if (!split_flag) {
        node->is_leaf = true;
        return node->obj;
    }

    if (depth_remaining == 1) {
        if (best_left_obj + best_right_obj >= node->obj) {
            node->is_leaf = true;
            return node->obj;
        }
        node->is_leaf = false;
        node->feature_idx = best_feature;
        node->threshold = best_threshold;

        delete node->left;
        delete node->right;
        node->left = new ConstNode();
        node->right = new ConstNode();
        node->left->obj = best_left_obj;
        node->right->obj = best_right_obj;
        node->left->n_instances = static_cast<std::size_t>(best_n_left);
        node->right->n_instances = static_cast<std::size_t>(best_n_right);

        node->obj = best_left_obj + best_right_obj;
        return node->obj;
    }

    delete node->left;
    delete node->right;
    node->left = new ConstNode();
    node->left->obj = best_left_obj;
    node->right = new ConstNode();
    node->right->obj = best_right_obj;

    set<unsigned long int> left_indices_set(best_indices_left.begin(), best_indices_left.end());
    vector<vector<unsigned long int>> sorted_indices_left(this->m, vector<unsigned long int>());
    vector<vector<unsigned long int>> sorted_indices_right(this->m, vector<unsigned long int>());
    for (unsigned long int feature = 1; feature < this->m; feature++) {
        for (unsigned long int idx : sorted_indices[feature]) {
            if (left_indices_set.find(idx) != left_indices_set.end()) {
                sorted_indices_left[feature].push_back(idx);
            } else {
                sorted_indices_right[feature].push_back(idx);
            }
        }
    }

    double final_left_objective = GreedyConst::recursive_fit(
        sorted_indices_left,
        best_n_left,
        best_y_sum_left,
        best_y_sum_sq_left,
        node->left,
        depth_remaining - 1);
    double final_right_objective = GreedyConst::recursive_fit(
        sorted_indices_right,
        best_n_right,
        best_y_sum_right,
        best_y_sum_sq_right,
        node->right,
        depth_remaining - 1);

    if (final_left_objective + final_right_objective < node->obj) {
        node->is_leaf = false;
        node->feature_idx = best_feature;
        node->threshold = best_threshold;
        node->obj = final_left_objective + final_right_objective;
        return final_left_objective + final_right_objective;
    }

    delete node->left;
    delete node->right;
    node->left = nullptr;
    node->right = nullptr;
    node->is_leaf = true;
    return node->obj;
}

double GreedyConst::loss(int n, double sum, double sum_sq) {
    if (n <= 0) {
        return std::numeric_limits<double>::infinity();
    }
    return sum_sq - sum * sum / n;
}

VectorXd GreedyConst::predict(MatrixXd X) {
    ensure_intercept_inplace(X);
    if (X.cols() != this->m) {
        throw runtime_error("predict: X has unexpected number of columns (intercept is column 0).");
    }
    VectorXd predictions(X.rows());
    for (int i = 0; i < X.rows(); i++) {
        predictions(i) = predict_row(X.row(i));
    }
    return predictions;
}

double GreedyConst::predict_row(VectorXd x) {
    if (x.size() == this->m - 1) {
        VectorXd x1(this->m);
        x1(0) = 1.0;
        x1.tail(this->m - 1) = x;
        x.swap(x1);
    } else if (x.size() != this->m) {
        throw runtime_error("predict_row: x has unexpected length (intercept is column 0).");
    }

    ConstNode* current = this->root;
    while (current != nullptr) {
        if (current->is_leaf) {
            return current->prediction;
        }
        if (x(current->feature_idx) <= current->threshold) {
            current = current->left;
        } else {
            current = current->right;
        }
    }
    throw runtime_error("Invalid tree structure");
}

string GreedyConst::print_tree() {
    if (this->root == nullptr) {
        return "No tree currently fit!";
    }
    return this->root->print_tree();
}

std::vector<std::vector<double>> GreedyConst::get_traversed_thresholds() const {
    std::vector<std::vector<double>> out(traversed_thresholds_.size());
    for (std::size_t feature = 0; feature < traversed_thresholds_.size(); ++feature) {
        out[feature].assign(traversed_thresholds_[feature].begin(), traversed_thresholds_[feature].end());
    }
    return out;
}

std::string GreedyConst::print_traversed_thresholds() const {
    std::ostringstream oss;
    oss.setf(std::ios::fixed);
    oss << std::setprecision(6);
    const auto all_thresholds = get_traversed_thresholds();
    for (std::size_t feature = 1; feature < all_thresholds.size(); ++feature) {
        oss << "Feature " << feature << ":";
        for (double threshold : all_thresholds[feature]) {
            oss << " " << threshold;
        }
        oss << "\n";
    }
    return oss.str();
}

std::vector<std::vector<double>> GreedyConst::get_threshold_pool() const {
    return threshold_pool_;
}

std::string GreedyConst::print_threshold_pool() const {
    std::ostringstream oss;
    oss.setf(std::ios::fixed);
    oss << std::setprecision(6);
    for (std::size_t feature = 1; feature < threshold_pool_.size(); ++feature) {
        oss << "Feature " << feature << ":";
        for (double threshold : threshold_pool_[feature]) {
            oss << " " << threshold;
        }
        oss << "\n";
    }
    return oss.str();
}

std::size_t GreedyConst::count_leaves(const ConstNode* n) {
    if (!n) return 0;
    if (n->is_leaf) return 1;
    return count_leaves(n->left) + count_leaves(n->right);
}

std::size_t GreedyConst::n_leaves() const {
    return count_leaves(this->root);
}

CLARITreeConst::CLARITreeConst(int depth, double lambda, int n_thresholds, bool verbose, int min_leaf_node_size)
    : GreedyConst(depth, lambda, n_thresholds, verbose, min_leaf_node_size) {}

CLARITreeConst::CLARITreeConst(int depth, double lambda, int n_thresholds, const std::string& thresholds_strategy, bool verbose, int min_leaf_node_size)
    : GreedyConst(depth, lambda, n_thresholds, thresholds_strategy, verbose, min_leaf_node_size) {}

double CLARITreeConst::recursive_fit(vector<vector<unsigned long int>>& sorted_indices, int n, double y_sum, double y_sum_sq, ConstNode* node, int depth_remaining) {
    node->n_instances = node_instance_count_from_sorted_indices(sorted_indices, this->m);
    if (depth_remaining == 0 ||
        node->obj <= 2 * this->scaled_lambda ||
        static_cast<std::size_t>(n) < 2 * resolved_min_leaf_node_size_) {
        node->is_leaf = true;
        return node->obj;
    }

    bool split_flag = false;
    unsigned long int best_feature = 0;
    double best_threshold = 0.0;
    double min_obj = std::numeric_limits<double>::infinity();
    vector<unsigned long int> best_indices_left;
    double best_left_leaf_obj = 0.0, best_right_leaf_obj = 0.0;
    double best_y_sum_sq_left = 0.0, best_y_sum_sq_right = 0.0;
    double best_y_sum_left = 0.0, best_y_sum_right = 0.0;
    int best_n_left = 0, best_n_right = 0;

    for (unsigned long int feature = 1; feature < this->m; feature++) {
        const bool is_bin = std::find(binary_idx_.begin(), binary_idx_.end(), (int)feature) != binary_idx_.end();
        const bool is_cont = std::find(continuous_idx_.begin(), continuous_idx_.end(), (int)feature) != continuous_idx_.end();

        if (is_bin) {
            vector<unsigned long int> left_rows;
            vector<unsigned long int> right_rows;
            left_rows.reserve(sorted_indices[feature].size());
            right_rows.reserve(sorted_indices[feature].size());
            double yss_left = 0.0, yss_right = 0.0;
            double sum_left = 0.0, sum_right = 0.0;

            for (unsigned long int row : sorted_indices[feature]) {
                if (this->X((int)row, feature) <= 0.5) {
                    left_rows.push_back(row);
                    sum_left += this->y(row);
                    yss_left += this->y(row) * this->y(row);
                } else {
                    right_rows.push_back(row);
                    sum_right += this->y(row);
                    yss_right += this->y(row) * this->y(row);
                }
            }
            if (!children_respect_min_leaf_size(left_rows.size(), right_rows.size())) {
                continue;
            }
            record_traversed_threshold(feature, 0.5);

            const int n_left = static_cast<int>(left_rows.size());
            const int n_right = static_cast<int>(right_rows.size());
            double left_obj = loss(n_left, sum_left, yss_left) + this->scaled_lambda;
            double right_obj = loss(n_right, sum_right, yss_right) + this->scaled_lambda;

            if (depth_remaining == 1) {
                if (left_obj + right_obj < min_obj) {
                    split_flag = true;
                    min_obj = left_obj + right_obj;
                    best_feature = feature;
                    best_threshold = 0.5;
                    best_left_leaf_obj = left_obj;
                    best_right_leaf_obj = right_obj;
                    best_indices_left = left_rows;
                    best_y_sum_sq_left = yss_left;
                    best_y_sum_sq_right = yss_right;
                    best_y_sum_left = sum_left;
                    best_y_sum_right = sum_right;
                    best_n_left = n_left;
                    best_n_right = n_right;
                }
            } else {
                set<unsigned long int> left_indices_set(left_rows.begin(), left_rows.end());

                delete node->left;
                delete node->right;
                node->left = nullptr;
                node->right = nullptr;
                node->left = new ConstNode();
                node->left->obj = left_obj;
                node->right = new ConstNode();
                node->right->obj = right_obj;

                vector<vector<unsigned long int>> sorted_indices_left(this->m, vector<unsigned long int>());
                vector<vector<unsigned long int>> sorted_indices_right(this->m, vector<unsigned long int>());
                for (unsigned long int f2 = 1; f2 < this->m; f2++) {
                    for (unsigned long int idx : sorted_indices[f2]) {
                        if (left_indices_set.find(idx) != left_indices_set.end()) {
                            sorted_indices_left[f2].push_back(idx);
                        } else {
                            sorted_indices_right[f2].push_back(idx);
                        }
                    }
                }

                double greedy_completion_left_obj =
                    GreedyConst::recursive_fit(sorted_indices_left, n_left, sum_left, yss_left, node->left, depth_remaining - 1);
                double greedy_completion_right_obj =
                    GreedyConst::recursive_fit(sorted_indices_right, n_right, sum_right, yss_right, node->right, depth_remaining - 1);

                if (greedy_completion_left_obj + greedy_completion_right_obj < min_obj) {
                    split_flag = true;
                    min_obj = greedy_completion_left_obj + greedy_completion_right_obj;
                    best_feature = feature;
                    best_threshold = 0.5;
                    best_left_leaf_obj = left_obj;
                    best_right_leaf_obj = right_obj;
                    best_indices_left = left_rows;
                    best_y_sum_sq_left = yss_left;
                    best_y_sum_sq_right = yss_right;
                    best_y_sum_left = sum_left;
                    best_y_sum_right = sum_right;
                    best_n_left = n_left;
                    best_n_right = n_right;
                }
            }
            continue;
        }

        if (!is_cont) {
            continue;
        }
        if (feature >= threshold_pool_.size()) {
            continue;
        }
        const auto& feature_pool = threshold_pool_[feature];
        if (feature_pool.empty()) {
            continue;
        }

        double y_sum_sq_left = 0.0;
        double y_sum_left = 0.0;
        int n_left = 0;
        vector<unsigned long int> left_indices;

        double y_sum_sq_right = y_sum_sq;
        double y_sum_right = y_sum;
        int n_right = n;
        std::size_t pool_idx = 0;

        for (unsigned long int feature_idx = 0; feature_idx < sorted_indices[feature].size(); feature_idx++) {
            const unsigned long int row = sorted_indices[feature][feature_idx];
            y_sum_sq_left += this->y(row) * this->y(row);
            y_sum_left += this->y(row);
            left_indices.push_back(row);
            n_left++;

            y_sum_sq_right -= this->y(row) * this->y(row);
            y_sum_right -= this->y(row);
            n_right--;

            if (feature_idx == sorted_indices[feature].size() - 1) {
                continue;
            }

            const double current_value = this->X(row, feature);
            const double next_value = this->X(sorted_indices[feature][feature_idx + 1], feature);
            if (current_value == next_value) {
                continue;
            }
            if (!children_respect_min_leaf_size(left_indices.size(), static_cast<std::size_t>(n_right))) {
                continue;
            }

            while (pool_idx < feature_pool.size() && feature_pool[pool_idx] < current_value) {
                ++pool_idx;
            }

            std::size_t eval_idx = pool_idx;
            while (eval_idx < feature_pool.size() && feature_pool[eval_idx] < next_value) {
                const double candidate_threshold = feature_pool[eval_idx];
                record_traversed_threshold(feature, candidate_threshold);

                double left_obj = loss(n_left, y_sum_left, y_sum_sq_left) + this->scaled_lambda;
                double right_obj = loss(n_right, y_sum_right, y_sum_sq_right) + this->scaled_lambda;

                if (depth_remaining == 1) {
                    if (left_obj + right_obj < min_obj) {
                        split_flag = true;
                        min_obj = left_obj + right_obj;
                        best_feature = feature;
                        best_threshold = candidate_threshold;
                        best_left_leaf_obj = left_obj;
                        best_right_leaf_obj = right_obj;
                        best_indices_left = left_indices;
                        best_y_sum_sq_left = y_sum_sq_left;
                        best_y_sum_sq_right = y_sum_sq_right;
                        best_y_sum_left = y_sum_left;
                        best_y_sum_right = y_sum_right;
                        best_n_left = n_left;
                        best_n_right = n_right;
                    }
                } else {
                    set<unsigned long int> left_indices_set(left_indices.begin(), left_indices.end());

                    delete node->left;
                    delete node->right;
                    node->left = nullptr;
                    node->right = nullptr;
                    node->left = new ConstNode();
                    node->left->obj = left_obj;
                    node->right = new ConstNode();
                    node->right->obj = right_obj;

                    vector<vector<unsigned long int>> sorted_indices_left(this->m, vector<unsigned long int>());
                    vector<vector<unsigned long int>> sorted_indices_right(this->m, vector<unsigned long int>());
                    for (unsigned long int f2 = 1; f2 < this->m; f2++) {
                        for (unsigned long int idx : sorted_indices[f2]) {
                            if (left_indices_set.find(idx) != left_indices_set.end()) {
                                sorted_indices_left[f2].push_back(idx);
                            } else {
                                sorted_indices_right[f2].push_back(idx);
                            }
                        }
                    }

                    double greedy_completion_left_obj =
                        GreedyConst::recursive_fit(sorted_indices_left, n_left, y_sum_left, y_sum_sq_left, node->left, depth_remaining - 1);
                    double greedy_completion_right_obj =
                        GreedyConst::recursive_fit(sorted_indices_right, n_right, y_sum_right, y_sum_sq_right, node->right, depth_remaining - 1);

                    if (greedy_completion_left_obj + greedy_completion_right_obj < min_obj) {
                        split_flag = true;
                        min_obj = greedy_completion_left_obj + greedy_completion_right_obj;
                        best_feature = feature;
                        best_threshold = candidate_threshold;
                        best_left_leaf_obj = left_obj;
                        best_right_leaf_obj = right_obj;
                        best_indices_left = left_indices;
                        best_y_sum_sq_left = y_sum_sq_left;
                        best_y_sum_sq_right = y_sum_sq_right;
                        best_y_sum_left = y_sum_left;
                        best_y_sum_right = y_sum_right;
                        best_n_left = n_left;
                        best_n_right = n_right;
                    }
                }
                ++eval_idx;
            }
            pool_idx = eval_idx;
        }
    }

    if (!split_flag) {
        delete node->left;
        delete node->right;
        node->left = nullptr;
        node->right = nullptr;
        node->is_leaf = true;
        return node->obj;
    }

    if (depth_remaining == 1) {
        if (min_obj >= node->obj) {
            node->is_leaf = true;
            return node->obj;
        }
        node->is_leaf = false;
        node->feature_idx = best_feature;
        node->threshold = best_threshold;

        delete node->left;
        delete node->right;
        node->left = new ConstNode();
        node->right = new ConstNode();
        node->left->obj = best_left_leaf_obj;
        node->right->obj = best_right_leaf_obj;
        node->left->n_instances = static_cast<std::size_t>(best_n_left);
        node->right->n_instances = static_cast<std::size_t>(best_n_right);

        node->obj = node->left->obj + node->right->obj;
        return node->obj;
    }

    delete node->left;
    delete node->right;
    node->left = nullptr;
    node->right = nullptr;
    node->left = new ConstNode();
    node->left->obj = best_left_leaf_obj;
    node->right = new ConstNode();
    node->right->obj = best_right_leaf_obj;

    set<unsigned long int> left_indices_set(best_indices_left.begin(), best_indices_left.end());
    vector<vector<unsigned long int>> sorted_indices_left(this->m, vector<unsigned long int>());
    vector<vector<unsigned long int>> sorted_indices_right(this->m, vector<unsigned long int>());
    for (unsigned long int feature = 1; feature < this->m; feature++) {
        for (unsigned long int idx : sorted_indices[feature]) {
            if (left_indices_set.find(idx) != left_indices_set.end()) {
                sorted_indices_left[feature].push_back(idx);
            } else {
                sorted_indices_right[feature].push_back(idx);
            }
        }
    }

    double final_left_obj = recursive_fit(
        sorted_indices_left,
        best_n_left,
        best_y_sum_left,
        best_y_sum_sq_left,
        node->left,
        depth_remaining - 1);
    double final_right_obj = recursive_fit(
        sorted_indices_right,
        best_n_right,
        best_y_sum_right,
        best_y_sum_sq_right,
        node->right,
        depth_remaining - 1);

    if (final_left_obj + final_right_obj < node->obj) {
        node->is_leaf = false;
        node->feature_idx = best_feature;
        node->threshold = best_threshold;
        node->obj = final_left_obj + final_right_obj;
        return final_left_obj + final_right_obj;
    }

    delete node->left;
    delete node->right;
    node->left = nullptr;
    node->right = nullptr;
    node->is_leaf = true;
    return node->obj;
}
