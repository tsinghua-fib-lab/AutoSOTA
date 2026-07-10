#include "clari_tree.hpp"
#include <iostream>
#include <chrono>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <random>
#include <algorithm>

using namespace Eigen;
using namespace std;

namespace {
    string trim(string s) {
        const auto first = s.find_first_not_of(" \t\r\n");
        if (first == string::npos) {
            return "";
        }
        const auto last = s.find_last_not_of(" \t\r\n");
        return s.substr(first, last - first + 1);
    }

    bool parse_double(const string& text, double& value) {
        const string cell = trim(text);
        if (cell.empty()) {
            return false;
        }
        try {
            size_t consumed = 0;
            value = stod(cell, &consumed);
            return consumed == cell.size();
        } catch (const invalid_argument&) {
            return false;
        } catch (const out_of_range&) {
            return false;
        }
    }

    bool parse_csv_row(const string& line, vector<double>& row, char delimiter = ',') {
        row.clear();
        stringstream ss(line);
        string cell;
        while (getline(ss, cell, delimiter)) {
            double value = 0.0;
            if (!parse_double(cell, value)) {
                return false;
            }
            row.push_back(value);
        }
        return !row.empty();
    }

    bool filename_uses_target_first(const string& filename) {
        return filename.find("targetfirst") != string::npos;
    }

    bool read_csv(const string& filename, MatrixXd& X, VectorXd& y) {
        ifstream file(filename);
        if (!file.is_open()) {
            cerr << "Error: Could not open CSV file: " << filename << endl;
            return false;
        }

        vector<vector<double>> data;
        string line;
        int line_number = 0;
        bool skipped_header = false;
        while (getline(file, line)) {
            ++line_number;
            if (trim(line).empty()) {
                continue;
            }
            vector<double> row;
            if (!parse_csv_row(line, row)) {
                if (data.empty() && !skipped_header) {
                    skipped_header = true;
                    continue;
                }
                cerr << "Error: Could not parse numeric CSV row " << line_number << endl;
                return false;
            }
            data.push_back(row);
        }

        if (data.empty()) {
            cerr << "Error: No numeric rows found in CSV file." << endl;
            return false;
        }
        const int rows = static_cast<int>(data.size());
        const int cols = static_cast<int>(data.front().size());
        if (cols < 2) {
            cerr << "Error: CSV must contain at least one feature column and one target column." << endl;
            return false;
        }
        for (int i = 0; i < rows; ++i) {
            if (static_cast<int>(data[i].size()) != cols) {
                cerr << "Error: Inconsistent number of columns in numeric row " << (i + 1) << endl;
                return false;
            }
        }

        const bool target_first = filename_uses_target_first(filename);
        X.resize(rows, cols - 1);
        y.resize(rows);
        for (int i = 0; i < rows; ++i) {
            if (target_first) {
                y(i) = data[i][0];
                for (int j = 1; j < cols; ++j) {
                    X(i, j - 1) = data[i][j];
                }
            } else {
                for (int j = 0; j < cols - 1; ++j) {
                    X(i, j) = data[i][j];
                }
                y(i) = data[i][cols - 1];
            }
        }
        return true;
    }

    double mse(const VectorXd& y_true, const VectorXd& y_pred) {
        return (y_true - y_pred).squaredNorm() / y_true.size();
    }

    double r2(const VectorXd& y_true, const VectorXd& y_pred) {
        const double ss_res = (y_true - y_pred).squaredNorm();
        const double mean_y = y_true.mean();
        const double ss_tot = (y_true.array() - mean_y).matrix().squaredNorm();
        return ss_tot == 0.0 ? 1.0 : 1.0 - ss_res / ss_tot;
    }
}

int main(int argc, char* argv[]){
    if (argc >= 5 && argc <= 7) {
        const string filename = argv[1];
        const Depth depth = std::stoi(argv[2]);
        const double lambda = std::stod(argv[3]);
        const double kappa = std::stod(argv[4]);
        const int n_thresholds = argc >= 6 ? std::stoi(argv[5]) : 1;
        const std::string thresholds_strategy = argc >= 7 ? argv[6] : "quantile";

        MatrixXd X;
        VectorXd y;
        if (!read_csv(filename, X, y)) {
            return 1;
        }

        cout << "Loaded CSV: " << filename << " (" << X.rows() << " rows, "
             << X.cols() << " features)" << endl;

        int n = X.rows();
        int n_train = static_cast<int>(0.8 * n);

        std::vector<int> indices(n);
        std::iota(indices.begin(), indices.end(), 0);

        std::mt19937 rng(42); 
        std::shuffle(indices.begin(), indices.end(), rng);
        MatrixXd X_train(n_train, X.cols());
        VectorXd y_train(n_train);

        MatrixXd X_test(n - n_train, X.cols());
        VectorXd y_test(n - n_train);

        for (int i = 0; i < n_train; ++i) {
            X_train.row(i) = X.row(indices[i]);
            y_train(i) = y(indices[i]);
        }

        for (int i = n_train; i < n; ++i) {
            X_test.row(i - n_train) = X.row(indices[i]);
            y_test(i - n_train) = y(indices[i]);
        }
        auto start = chrono::high_resolution_clock::now();

        Greedy greedy_tree = Greedy(kappa, depth, lambda, n_thresholds, thresholds_strategy, false);
        double greedy_loss = greedy_tree.fit(X_train, y_train);

        VectorXd greedy_pred_train = greedy_tree.predict(X_train);
        VectorXd greedy_pred_test  = greedy_tree.predict(X_test);

        auto end = chrono::high_resolution_clock::now();
        chrono::duration<double> elapsed = end - start;

        cout << "Greedy loss found: " << greedy_loss
             << " in " << elapsed.count() << " seconds" << endl;

        cout << "Greedy TRAIN MSE: " << mse(y_train, greedy_pred_train) << endl;
        cout << "Greedy TRAIN R^2: " << r2(y_train, greedy_pred_train) << endl;

        cout << "Greedy TEST MSE: " << mse(y_test, greedy_pred_test) << endl;
        cout << "Greedy TEST R^2: " << r2(y_test, greedy_pred_test) << endl;
        start = chrono::high_resolution_clock::now();

        CLARITree clari_tree = CLARITree(kappa, depth, lambda, n_thresholds, thresholds_strategy, false);
        double clari_loss = clari_tree.fit(X_train, y_train);

        VectorXd clari_pred_train = clari_tree.predict(X_train);
        VectorXd clari_pred_test  = clari_tree.predict(X_test);

        end = chrono::high_resolution_clock::now();
        elapsed = end - start;

        cout << "CLARITree loss found: " << clari_loss
             << " in " << elapsed.count() << " seconds" << endl;

        cout << "CLARITree TRAIN MSE: " << mse(y_train, clari_pred_train) << endl;
        cout << "CLARITree TRAIN R^2: " << r2(y_train, clari_pred_train) << endl;

        cout << "CLARITree TEST MSE: " << mse(y_test, clari_pred_test) << endl;
        cout << "CLARITree TEST R^2: " << r2(y_test, clari_pred_test) << endl;

    } else {
        cout << "Usage: " << argv[0]
             << " <csv_file> <depth> <lambda> <kappa> [n_thresholds] [thresholds_strategy]" << endl;
        return 1;
    }

    return 0;
}