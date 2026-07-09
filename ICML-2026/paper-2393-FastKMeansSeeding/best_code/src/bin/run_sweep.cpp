/**
 * Hyperparameter sweep for all seeding algorithms.
 *
 * Usage: ./bin/run_sweep config.json
 *
 * Sweeps over:
 *   - m values for AFKMC2, QKMEANS
 *   - ef values for QKMEANS
 *   - alpha values for PRONECoreset, FastCoreset
 */

#include <iostream>
#include <fstream>
#include <chrono>
#include <vector>
#include <string>
#include <iomanip>
#include <sstream>

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
        std::cerr << "Usage: ./bin/run_sweep config.json\n";
        return 1;
    }

    std::ifstream cfg_file(argv[1]);
    if (!cfg_file) {
        std::cerr << "Error: Failed to open config file\n";
        return 1;
    }

    json cfg;
    cfg_file >> cfg;

    // Parse config
    std::string data_path = cfg["data_path"];
    std::string name = cfg.value("name", "dataset");
    std::vector<size_t> k_values = cfg["k_values"].get<std::vector<size_t>>();
    int num_runs = cfg.value("num_runs", 5);
    std::string output_csv = cfg.value("output_csv", "results/sweep.csv");

    // Sweep values
    std::vector<size_t> m_values = cfg.value("m_values", std::vector<size_t>{50, 100, 200, 500});
    std::vector<size_t> ef_values = cfg.value("ef_values", std::vector<size_t>{10, 25, 50, 100});
    std::vector<double> alpha_values = cfg.value("alpha_values", std::vector<double>{0.005, 0.01, 0.02, 0.05});

    // Load dataset
    Dataset X;
    X.fromTxt(data_path);
    const size_t n = X.size();
    const size_t d = X.dim();

    std::cout << "=== Hyperparameter Sweep ===\n"
              << "Dataset: " << name << " (n=" << n << ", d=" << d << ")\n"
              << "k values: ";
    for (auto k : k_values) std::cout << k << " ";
    std::cout << "\nRuns per config: " << num_runs << "\n\n";

    // Open CSV
    std::ofstream csv(output_csv);
    csv << "dataset,method,k,hyperparam,hyperparam_value,seeding_cost,seeding_time_ms\n";

    // Header
    std::cout << std::setw(18) << "Method"
              << std::setw(8) << "k"
              << std::setw(14) << "Param"
              << std::setw(14) << "Cost"
              << std::setw(12) << "Time(ms)" << "\n"
              << std::string(66, '-') << "\n";

    for (size_t k : k_values) {
        if (k > n) continue;

        std::cout << "\n=== k=" << k << " ===\n";

        // K-Means++ (baseline)
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

            csv << name << ",kmeanspp," << k << ",none,0," << avg_cost << "," << avg_time << "\n";
            std::cout << std::setw(18) << "kmeanspp" << std::setw(8) << k
                      << std::setw(14) << "-"
                      << std::setw(14) << std::fixed << std::setprecision(1) << avg_cost
                      << std::setw(12) << std::setprecision(2) << avg_time << "\n";
        }

        // PRONE (no params)
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

            csv << name << ",prone," << k << ",none,0," << avg_cost << "," << avg_time << "\n";
            std::cout << std::setw(18) << "prone" << std::setw(8) << k
                      << std::setw(14) << "-"
                      << std::setw(14) << avg_cost << std::setw(12) << avg_time << "\n";
        }

        // AFKMC2 - sweep m
        for (size_t m : m_values) {
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

            csv << name << ",afkmc2," << k << ",m," << m << "," << avg_cost << "," << avg_time << "\n";
            std::cout << std::setw(18) << "afkmc2" << std::setw(8) << k
                      << std::setw(14) << ("m=" + std::to_string(m))
                      << std::setw(14) << avg_cost << std::setw(12) << avg_time << "\n";
        }

        // PRONECoreset - sweep alpha
        for (double alpha : alpha_values) {
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

            csv << name << ",pronecoreset," << k << ",alpha," << alpha << "," << avg_cost << "," << avg_time << "\n";
            std::ostringstream param;
            param << "a=" << alpha;
            std::cout << std::setw(18) << "pronecoreset" << std::setw(8) << k
                      << std::setw(14) << param.str()
                      << std::setw(14) << avg_cost << std::setw(12) << avg_time << "\n";
        }

        // FastCoreset - sweep alpha
        for (double alpha : alpha_values) {
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

            csv << name << ",fastcoreset," << k << ",alpha," << alpha << "," << avg_cost << "," << avg_time << "\n";
            std::ostringstream param;
            param << "a=" << alpha;
            std::cout << std::setw(18) << "fastcoreset" << std::setw(8) << k
                      << std::setw(14) << param.str()
                      << std::setw(14) << avg_cost << std::setw(12) << avg_time << "\n";
        }

        // RejectionSamplingLSH (no params)
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

            csv << name << ",rejectionlsh," << k << ",none,0," << avg_cost << "," << avg_time << "\n";
            std::cout << std::setw(18) << "rejectionlsh" << std::setw(8) << k
                      << std::setw(14) << "-"
                      << std::setw(14) << avg_cost << std::setw(12) << avg_time << "\n";
        }

        // QKMEANS - sweep ef and m
        for (size_t ef : ef_values) {
            for (size_t m : m_values) {
                double total_cost = 0, total_time = 0;
                for (int run = 0; run < num_runs; ++run) {
                    QKMEANS alg(42 + run);
                    auto t0 = std::chrono::high_resolution_clock::now();
                    Dataset centers = alg.run(X, k, ef, m, false);
                    auto t1 = std::chrono::high_resolution_clock::now();
                    total_time += elapsed_ms(t0, t1);
                    total_cost += X.clustering_cost(centers);
                }
                double avg_cost = total_cost / num_runs;
                double avg_time = total_time / num_runs;

                csv << name << ",qkmeans," << k << ",ef_m," << ef << "_" << m << ","
                    << avg_cost << "," << avg_time << "\n";
                std::ostringstream param;
                param << "ef=" << ef << ",m=" << m;
                std::cout << std::setw(18) << "qkmeans" << std::setw(8) << k
                          << std::setw(14) << param.str()
                          << std::setw(14) << avg_cost << std::setw(12) << avg_time << "\n";
            }
        }

        csv.flush();
    }

    csv.close();
    std::cout << "\nResults saved to: " << output_csv << "\n";

    return 0;
}
