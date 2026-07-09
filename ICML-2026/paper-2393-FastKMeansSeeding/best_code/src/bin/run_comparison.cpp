/**
 * Run all seeding algorithms on a dataset for comparison.
 *
 * Usage: ./bin/run_comparison config.json
 *
 * Runs: k-means++, AFKMC2, PRONE, PRONECoreset, FastCoreset,
 *       RejectionSamplingLSH, and QKMEANS
 */

#include <iostream>
#include <fstream>
#include <chrono>
#include <vector>
#include <string>
#include <iomanip>
#include <algorithm>

#include "src/core/dataset.hpp"
#include "src/algorithms/kmeanspp.hpp"
#include "src/algorithms/afkmc2.hpp"
#include "src/algorithms/prone.hpp"
#include "src/algorithms/pronecoreset.hpp"
#include "src/algorithms/fastcoresetkmeans.hpp"
#include "src/algorithms/rejectionsampling.hpp"
#include "src/algorithms/qkmeans.hpp"
#include "external/nlohmann/json.hpp"

using json = nlohmann::json;
using namespace fastcluster;

inline double elapsed_ms(
    const std::chrono::high_resolution_clock::time_point& start,
    const std::chrono::high_resolution_clock::time_point& end
) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: ./bin/run_comparison config.json\n";
        return 1;
    }

    // Load config
    std::ifstream cfg_file(argv[1]);
    if (!cfg_file) {
        std::cerr << "Error: Failed to open config file\n";
        return 1;
    }

    json cfg;
    cfg_file >> cfg;

    // Parse config
    std::string name = cfg.value("name", cfg.value("dataset_name", "dataset"));
    std::string data_path = cfg.value("data_path", cfg.value("dataset_path", ""));
    std::vector<size_t> k_values = cfg["k_values"].get<std::vector<size_t>>();
    int num_runs = cfg.value("num_runs", 5);
    std::string output_csv = cfg.value("output_csv", "results/comparison.csv");

    // Algorithm params
    size_t m = cfg.value("m", 100);
    double alpha = cfg.value("alpha", 0.01);
    size_t ef = cfg.value("ef", 50);
    size_t hnsw_M = cfg.value("M", 16);
    size_t hnsw_ef_construction = cfg.value("ef_construction", 200);

    // Load dataset
    Dataset X;
    X.fromTxt(data_path);
    const size_t n = X.size();
    const size_t d = X.dim();

    std::cout << "=== Algorithm Comparison ===\n"
              << "Dataset: " << name << " (n=" << n << ", d=" << d << ")\n"
              << "k values: ";
    for (auto k : k_values) std::cout << k << " ";
    std::cout << "\nRuns: " << num_runs << ", m=" << m << ", alpha=" << alpha << ", ef=" << ef << ", M=" << hnsw_M << ", ef_construction=" << hnsw_ef_construction << "\n\n";

    // Open CSV
    std::ofstream csv(output_csv);
    csv << "dataset,method,k,seeding_cost,seeding_time_ms\n";

    // Header
    std::cout << std::setw(20) << "Method"
              << std::setw(8) << "k"
              << std::setw(16) << "Cost"
              << std::setw(12) << "Time(ms)" << "\n"
              << std::string(56, '-') << "\n";

    for (size_t k : k_values) {
        if (k > n) continue;

        std::cout << "\n--- k=" << k << " ---\n";

        // K-Means++
        {
            double total_cost = 0, total_time = 0;
            for (int run = 0; run < num_runs; ++run) {
                KMeansPP alg(42 + run);
                auto t0 = std::chrono::high_resolution_clock::now();
                Dataset centers = alg.run(X, k);
                auto t1 = std::chrono::high_resolution_clock::now();
                total_time += elapsed_ms(t0, t1);
                total_cost += X.clustering_cost(centers);
            }
            double avg_cost = total_cost / num_runs;
            double avg_time = total_time / num_runs;
            csv << name << ",kmeanspp," << k << "," << avg_cost << "," << avg_time << "\n";
            std::cout << std::setw(20) << "kmeanspp" << std::setw(8) << k
                      << std::setw(16) << std::fixed << std::setprecision(1) << avg_cost
                      << std::setw(12) << std::setprecision(2) << avg_time << "\n";
        }

        // AFKMC2
        {
            double total_cost = 0, total_time = 0;
            for (int run = 0; run < num_runs; ++run) {
                AFKMC2 alg(42 + run);
                auto t0 = std::chrono::high_resolution_clock::now();
                Dataset centers = alg.run(X, k, m);
                auto t1 = std::chrono::high_resolution_clock::now();
                total_time += elapsed_ms(t0, t1);
                total_cost += X.clustering_cost(centers);
            }
            double avg_cost = total_cost / num_runs;
            double avg_time = total_time / num_runs;
            csv << name << ",afkmc2," << k << "," << avg_cost << "," << avg_time << "\n";
            std::cout << std::setw(20) << "afkmc2" << std::setw(8) << k
                      << std::setw(16) << avg_cost << std::setw(12) << avg_time << "\n";
        }

        // PRONE
        {
            double total_cost = 0, total_time = 0;
            for (int run = 0; run < num_runs; ++run) {
                PRONE alg(42 + run);
                auto t0 = std::chrono::high_resolution_clock::now();
                Dataset centers = alg.run(X, k);
                auto t1 = std::chrono::high_resolution_clock::now();
                total_time += elapsed_ms(t0, t1);
                total_cost += X.clustering_cost(centers);
            }
            double avg_cost = total_cost / num_runs;
            double avg_time = total_time / num_runs;
            csv << name << ",prone," << k << "," << avg_cost << "," << avg_time << "\n";
            std::cout << std::setw(20) << "prone" << std::setw(8) << k
                      << std::setw(16) << avg_cost << std::setw(12) << avg_time << "\n";
        }

        // PRONECoreset
        {
            size_t coreset_size = std::max(k, static_cast<size_t>(alpha * n));
            double total_cost = 0, total_time = 0;
            for (int run = 0; run < num_runs; ++run) {
                PRONECoreset alg(42 + run);
                auto t0 = std::chrono::high_resolution_clock::now();
                Dataset centers = alg.run(X, k, coreset_size);
                auto t1 = std::chrono::high_resolution_clock::now();
                total_time += elapsed_ms(t0, t1);
                total_cost += X.clustering_cost(centers);
            }
            double avg_cost = total_cost / num_runs;
            double avg_time = total_time / num_runs;
            csv << name << ",pronecoreset," << k << "," << avg_cost << "," << avg_time << "\n";
            std::cout << std::setw(20) << "pronecoreset" << std::setw(8) << k
                      << std::setw(16) << avg_cost << std::setw(12) << avg_time << "\n";
        }

        // FastCoreset
        {
            size_t coreset_size = std::max(k, static_cast<size_t>(alpha * n));
            double total_cost = 0, total_time = 0;
            for (int run = 0; run < num_runs; ++run) {
                FastCoresetKMeansPP alg(42 + run);
                auto t0 = std::chrono::high_resolution_clock::now();
                Dataset centers = alg.run(X, k, coreset_size);
                auto t1 = std::chrono::high_resolution_clock::now();
                total_time += elapsed_ms(t0, t1);
                total_cost += X.clustering_cost(centers);
            }
            double avg_cost = total_cost / num_runs;
            double avg_time = total_time / num_runs;
            csv << name << ",fastcoreset," << k << "," << avg_cost << "," << avg_time << "\n";
            std::cout << std::setw(20) << "fastcoreset" << std::setw(8) << k
                      << std::setw(16) << avg_cost << std::setw(12) << avg_time << "\n";
        }

        // RejectionSamplingLSH
        {
            double total_cost = 0, total_time = 0;
            for (int run = 0; run < num_runs; ++run) {
                rejectionsampling::RejectionSamplingKMeansPP alg;
                rejectionsampling::RandomHandler::seed(42 + run);
                auto t0 = std::chrono::high_resolution_clock::now();
                Dataset centers = alg.run(X, k);
                auto t1 = std::chrono::high_resolution_clock::now();
                total_time += elapsed_ms(t0, t1);
                total_cost += X.clustering_cost(centers);
            }
            double avg_cost = total_cost / num_runs;
            double avg_time = total_time / num_runs;
            csv << name << ",rejectionlsh," << k << "," << avg_cost << "," << avg_time << "\n";
            std::cout << std::setw(20) << "rejectionlsh" << std::setw(8) << k
                      << std::setw(16) << avg_cost << std::setw(12) << avg_time << "\n";
        }

        // QKMEANS
        {
            double total_cost = 0, total_time = 0;
            for (int run = 0; run < num_runs; ++run) {
                QKMEANS alg(42 + run);
                auto t0 = std::chrono::high_resolution_clock::now();
                Dataset centers = alg.run(X, k, ef, m, false, hnsw_M, hnsw_ef_construction);
                auto t1 = std::chrono::high_resolution_clock::now();
                total_time += elapsed_ms(t0, t1);
                total_cost += X.clustering_cost(centers);
            }
            double avg_cost = total_cost / num_runs;
            double avg_time = total_time / num_runs;
            csv << name << ",qkmeans," << k << "," << avg_cost << "," << avg_time << "\n";
            std::cout << std::setw(20) << "qkmeans" << std::setw(8) << k
                      << std::setw(16) << avg_cost << std::setw(12) << avg_time << "\n";
        }

        csv.flush();
    }

    csv.close();
    std::cout << "\nResults saved to: " << output_csv << "\n";

    return 0;
}
