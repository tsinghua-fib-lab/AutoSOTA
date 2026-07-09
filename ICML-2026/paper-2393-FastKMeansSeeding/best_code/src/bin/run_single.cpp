/**
 * Unified single-algorithm benchmark runner.
 *
 * Usage: ./bin/run_single <algorithm> config.json
 *
 * Algorithms: kmeanspp, afkmc2, prone, pronecoreset, fastcoreset,
 *             rejectionlsh, qkmeans, qkmeans_anns
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
#include "src/algorithms/qkmeans_anns.hpp"
#include "external/nlohmann/json.hpp"

using json = nlohmann::json;
using namespace fastcluster;

inline double elapsed_ms(
    const std::chrono::high_resolution_clock::time_point& start,
    const std::chrono::high_resolution_clock::time_point& end
) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

void print_usage() {
    std::cerr << "Usage: ./bin/run_single <algorithm> config.json\n\n"
              << "Algorithms:\n"
              << "  kmeanspp      - Standard k-means++ (baseline)\n"
              << "  afkmc2        - AFK-MC2 (MCMC-based)\n"
              << "  prone         - PRONE (1D projection + segment tree)\n"
              << "  pronecoreset  - PRONE + coreset\n"
              << "  fastcoreset   - HST-based fast coreset\n"
              << "  rejectionlsh  - Rejection sampling with LSH\n"
              << "  qkmeans       - QKMEANS (HNSW + rejection sampling)\n"
              << "  qkmeans_anns  - QKMEANS with pluggable ANNS\n";
}

int main(int argc, char** argv) {
    if (argc != 3) {
        print_usage();
        return 1;
    }

    std::string algorithm = argv[1];
    std::string config_path = argv[2];

    // Validate algorithm
    std::vector<std::string> valid_algos = {
        "kmeanspp", "afkmc2", "prone", "pronecoreset",
        "fastcoreset", "rejectionlsh", "qkmeans", "qkmeans_anns"
    };
    if (std::find(valid_algos.begin(), valid_algos.end(), algorithm) == valid_algos.end()) {
        std::cerr << "Error: Unknown algorithm '" << algorithm << "'\n\n";
        print_usage();
        return 1;
    }

    // Load config
    std::ifstream cfg_file(config_path);
    if (!cfg_file) {
        std::cerr << "Error: Failed to open config file: " << config_path << "\n";
        return 1;
    }

    json cfg;
    cfg_file >> cfg;

    // Parse common config
    std::string name = cfg.value("name", "dataset");
    std::string data_path = cfg["data_path"];
    std::vector<size_t> k_values = cfg["k_values"].get<std::vector<size_t>>();
    int num_runs = cfg.value("num_runs", 5);
    std::string output_csv = cfg.value("output_csv", "results/" + algorithm + ".csv");

    // Algorithm-specific params
    std::vector<size_t> m_values = cfg.value("m_values", std::vector<size_t>{100});
    std::vector<size_t> ef_values = cfg.value("ef_values", std::vector<size_t>{50});
    std::vector<double> alpha_values = cfg.value("alpha_values", std::vector<double>{0.01});
    std::string anns_method = cfg.value("anns_method", "hnsw");

    // Load dataset
    Dataset X;
    X.fromTxt(data_path);
    const size_t n = X.size();
    const size_t d = X.dim();

    std::cout << "=== " << algorithm << " Benchmark ===\n"
              << "Dataset: " << name << " (n=" << n << ", d=" << d << ")\n"
              << "k values: ";
    for (auto k : k_values) std::cout << k << " ";
    std::cout << "\nRuns per config: " << num_runs << "\n\n";

    // Open CSV
    std::ofstream csv(output_csv);

    // Write header based on algorithm
    if (algorithm == "qkmeans" || algorithm == "qkmeans_anns") {
        csv << "dataset,method,k,ef,m,seeding_cost,seeding_time_ms\n";
    } else if (algorithm == "afkmc2") {
        csv << "dataset,method,k,m,seeding_cost,seeding_time_ms\n";
    } else if (algorithm == "pronecoreset" || algorithm == "fastcoreset") {
        csv << "dataset,method,k,alpha,seeding_cost,seeding_time_ms\n";
    } else {
        csv << "dataset,method,k,seeding_cost,seeding_time_ms\n";
    }

    // Run benchmarks
    for (size_t k : k_values) {
        if (k > n) continue;

        if (algorithm == "kmeanspp") {
            double total_cost = 0, total_time = 0;
            for (int run = 0; run < num_runs; ++run) {
                KMeansPP alg(42 + run);
                auto t0 = std::chrono::high_resolution_clock::now();
                Dataset centers = alg.run(X, k);
                auto t1 = std::chrono::high_resolution_clock::now();
                total_time += elapsed_ms(t0, t1);
                total_cost += X.clustering_cost(centers);
            }
            csv << name << ",kmeanspp," << k << ","
                << total_cost/num_runs << "," << total_time/num_runs << "\n";
            std::cout << "k=" << k << " cost=" << total_cost/num_runs
                      << " time=" << total_time/num_runs << "ms\n";
        }
        else if (algorithm == "afkmc2") {
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
                csv << name << ",afkmc2," << k << "," << m << ","
                    << total_cost/num_runs << "," << total_time/num_runs << "\n";
                std::cout << "k=" << k << " m=" << m << " cost=" << total_cost/num_runs
                          << " time=" << total_time/num_runs << "ms\n";
            }
        }
        else if (algorithm == "prone") {
            double total_cost = 0, total_time = 0;
            for (int run = 0; run < num_runs; ++run) {
                PRONE alg(42 + run);
                auto t0 = std::chrono::high_resolution_clock::now();
                Dataset centers = alg.run(X, k);
                auto t1 = std::chrono::high_resolution_clock::now();
                total_time += elapsed_ms(t0, t1);
                total_cost += X.clustering_cost(centers);
            }
            csv << name << ",prone," << k << ","
                << total_cost/num_runs << "," << total_time/num_runs << "\n";
            std::cout << "k=" << k << " cost=" << total_cost/num_runs
                      << " time=" << total_time/num_runs << "ms\n";
        }
        else if (algorithm == "pronecoreset") {
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
                csv << name << ",pronecoreset," << k << "," << alpha << ","
                    << total_cost/num_runs << "," << total_time/num_runs << "\n";
                std::cout << "k=" << k << " alpha=" << alpha << " cost=" << total_cost/num_runs
                          << " time=" << total_time/num_runs << "ms\n";
            }
        }
        else if (algorithm == "fastcoreset") {
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
                csv << name << ",fastcoreset," << k << "," << alpha << ","
                    << total_cost/num_runs << "," << total_time/num_runs << "\n";
                std::cout << "k=" << k << " alpha=" << alpha << " cost=" << total_cost/num_runs
                          << " time=" << total_time/num_runs << "ms\n";
            }
        }
        else if (algorithm == "rejectionlsh") {
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
            csv << name << ",rejectionlsh," << k << ","
                << total_cost/num_runs << "," << total_time/num_runs << "\n";
            std::cout << "k=" << k << " cost=" << total_cost/num_runs
                      << " time=" << total_time/num_runs << "ms\n";
        }
        else if (algorithm == "qkmeans") {
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
                    csv << name << ",qkmeans," << k << "," << ef << "," << m << ","
                        << total_cost/num_runs << "," << total_time/num_runs << "\n";
                    std::cout << "k=" << k << " ef=" << ef << " m=" << m
                              << " cost=" << total_cost/num_runs
                              << " time=" << total_time/num_runs << "ms\n";
                }
            }
        }
        else if (algorithm == "qkmeans_anns") {
            ANNSMethod method = anns_method_from_string(anns_method);

            for (size_t ef : ef_values) {
                double total_cost = 0, total_time = 0;
                for (int run = 0; run < num_runs; ++run) {
                    QKMEANS_ANNS alg(42 + run);
                    auto t0 = std::chrono::high_resolution_clock::now();
                    Dataset centers = alg.run(X, k, method, ef, 16);
                    auto t1 = std::chrono::high_resolution_clock::now();
                    total_time += elapsed_ms(t0, t1);
                    total_cost += X.clustering_cost(centers);
                }
                csv << name << ",qkmeans_anns," << k << "," << ef << ",100,"
                    << total_cost/num_runs << "," << total_time/num_runs << "\n";
                std::cout << "k=" << k << " ef=" << ef
                          << " cost=" << total_cost/num_runs
                          << " time=" << total_time/num_runs << "ms\n";
            }
        }

        csv.flush();
    }

    csv.close();
    std::cout << "\nResults saved to: " << output_csv << "\n";

    return 0;
}
