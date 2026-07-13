#include "env/sailing_env.h"

#include "algorithms/est/est_decision_node.h"
#include "algorithms/ments/dents/dents_decision_node.h"
#include "algorithms/ments/dents/dents_manager.h"
#include "algorithms/ments/ments_decision_node.h"
#include "algorithms/ments/ments_manager.h"
#include "algorithms/ments/rents/rents_decision_node.h"
#include "algorithms/ments/tents/tents_decision_node.h"
#include "algorithms/varde/varde_decision_node.h"
#include "algorithms/varde/varde_manager.h"
#include "algorithms/uct/uct_decision_node.h"
#include "algorithms/uct/uct_manager.h"

#include "mc_eval.h"
#include "mcts.h"
#include "mcts_env_context.h"

#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

using namespace std;

namespace {
    struct Candidate {
        string algo;
        string config;
        function<pair<shared_ptr<mcts::MctsManager>, shared_ptr<mcts::MctsDNode>>(
            shared_ptr<mcts::MctsEnv> env,
            shared_ptr<const mcts::State> init_state,
            int max_depth,
            int seed)> make;
    };

    struct EvalStats {
        double mean;
        double stddev;
    };

    static EvalStats evaluate_tree(
        shared_ptr<const mcts::MctsEnv> env,
        shared_ptr<const mcts::MctsDNode> root,
        int horizon,
        int rollouts,
        int threads,
        int seed)
    {
        mcts::RandManager eval_rng(seed);
        mcts::EvalPolicy policy(root, env, eval_rng);
        mcts::MCEvaluator evaluator(env, policy, horizon, eval_rng);
        evaluator.run_rollouts(rollouts, threads);
        return {evaluator.get_mean_return(), evaluator.get_stddev_return()};
    }

    static vector<Candidate> build_candidates() {
        vector<Candidate> cands;

        // UCT grid over bias and epsilon.
        for (double bias : {mcts::UctManagerArgs::USE_AUTO_BIAS, 2.0, 5.0, 10.0}) {
            for (double eps : {0.1, 0.5, 1.0}) {
                string cfg = string("bias=") + (bias == mcts::UctManagerArgs::USE_AUTO_BIAS ? "auto" : to_string(bias)) +
                             ",eps=" + to_string(eps);
                cands.push_back({"UCT", cfg, [bias, eps](auto env, auto init_state, int max_depth, int seed) {
                    mcts::UctManagerArgs args(env);
                    args.max_depth = max_depth;
                    args.mcts_mode = false;
                    args.bias = bias;
                    args.epsilon_exploration = eps;
                    args.seed = seed;
                    auto mgr = make_shared<mcts::UctManager>(args);
                    auto root = make_shared<mcts::UctDNode>(mgr, init_state, 0, 0);
                    return make_pair(static_pointer_cast<mcts::MctsManager>(mgr), static_pointer_cast<mcts::MctsDNode>(root));
                }});
            }
        }

        // MENTS temp/epsilon sweeps.
        for (double temp : {0.1, 1.0, 2.0, 3.0, 5.0, 10.0}) {
            for (double eps : {0.25, 0.5, 0.75}) {
                string cfg = "temp=" + to_string(temp) + ",eps=" + to_string(eps);

                cands.push_back({"MENTS", cfg, [temp, eps](auto env, auto init_state, int max_depth, int seed) {
                    mcts::MentsManagerArgs args(env);
                    args.max_depth = max_depth;
                    args.mcts_mode = false;
                    args.temp = temp;
                    args.epsilon = eps;
                    args.seed = seed;
                    auto mgr = make_shared<mcts::MentsManager>(args);
                    auto root = make_shared<mcts::MentsDNode>(mgr, init_state, 0, 0);
                    return make_pair(static_pointer_cast<mcts::MctsManager>(mgr), static_pointer_cast<mcts::MctsDNode>(root));
                }});
            }
        }

        // RENTS temp/epsilon sweeps
        for (double temp : {0.1, 1.0, 2.0, 3.0, 5.0, 10.0}) {
            for (double eps : {0.25, 0.5, 0.75}) {
                string cfg = "temp=" + to_string(temp) + ",eps=" + to_string(eps);

                cands.push_back({"RENTS", cfg, [temp, eps](auto env, auto init_state, int max_depth, int seed) {
                    mcts::MentsManagerArgs args(env);
                    args.max_depth = max_depth;
                    args.mcts_mode = false;
                    args.temp = temp;
                    args.epsilon = eps;
                    args.seed = seed;
                    auto mgr = make_shared<mcts::MentsManager>(args);
                    auto root = make_shared<mcts::RentsDNode>(mgr, init_state, 0, 0);
                    return make_pair(static_pointer_cast<mcts::MctsManager>(mgr), static_pointer_cast<mcts::MctsDNode>(root));
                }});
            }
        }

        // TENTS temp/epsilon sweeps
        for (double temp : {0.1, 1.0, 2.0, 3.0, 5.0, 10.0}) {
            for (double eps : {0.25, 0.5, 0.75}) {
                string cfg = "temp=" + to_string(temp) + ",eps=" + to_string(eps);

                cands.push_back({"TENTS", cfg, [temp, eps](auto env, auto init_state, int max_depth, int seed) {
                    mcts::MentsManagerArgs args(env);
                    args.max_depth = max_depth;
                    args.mcts_mode = false;
                    args.temp = temp;
                    args.epsilon = eps;
                    args.seed = seed;
                    auto mgr = make_shared<mcts::MentsManager>(args);
                    auto root = make_shared<mcts::TentsDNode>(mgr, init_state, 0, 0);
                    return make_pair(static_pointer_cast<mcts::MctsManager>(mgr), static_pointer_cast<mcts::MctsDNode>(root));
                }});
            }
        }

        // DENTS temp/epsilon sweeps
        for (double temp : {0.1, 1.0, 2.0, 3.0, 5.0, 10.0}) {
            for (double eps : {0.25, 0.5, 0.75}) {
                string cfg = "temp=" + to_string(temp) + ",eps=" + to_string(eps);

                cands.push_back({"DENTS", cfg, [temp, eps](auto env, auto init_state, int max_depth, int seed) {
                    mcts::DentsManagerArgs args(env);
                    args.max_depth = max_depth;
                    args.mcts_mode = false;
                    args.temp = temp;
                    args.epsilon = eps;
                    args.seed = seed;
                    args.use_dp_value = true;
                    auto mgr = make_shared<mcts::DentsManager>(args);
                    auto root = make_shared<mcts::DentsDNode>(mgr, init_state, 0, 0);
                    return make_pair(static_pointer_cast<mcts::MctsManager>(mgr), static_pointer_cast<mcts::MctsDNode>(root));
                }});
            }
        }

        // BTS temp/epsilon sweeps
        for (double temp : {0.1, 1.0, 2.0, 3.0, 5.0, 10.0}) {
            for (double eps : {0.25, 0.5, 0.75}) {
                string cfg = "temp=" + to_string(temp) + ",eps=" + to_string(eps);

                cands.push_back({"BTS", cfg, [temp, eps](auto env, auto init_state, int max_depth, int seed) {
                    mcts::DentsManagerArgs args(env);
                    args.max_depth = max_depth;
                    args.mcts_mode = false;
                    args.temp = temp;
                    args.epsilon = eps;
                    args.seed = seed;
                    auto mgr = make_shared<mcts::DentsManager>(args);
                    auto root = make_shared<mcts::EstDNode>(mgr, init_state, 0, 0);
                    return make_pair(static_pointer_cast<mcts::MctsManager>(mgr), static_pointer_cast<mcts::MctsDNode>(root));
                }});
            }
        }

        // VarDE temp/variance_floor sweeps.
        for (double temp : {0.1, 1.0, 2.0, 3.0, 5.0, 10.0}) {
            for (double variance_floor : {1.0, 10.0, 100.0, 1000.0}) {
                for (bool use_dp_value : {true, false}) {
                    string cfg = "temp=" + to_string(temp) + ",variance_floor=" + to_string(variance_floor)
                        + ",value_src=" + string(use_dp_value ? "dp" : "mean");
                    cands.push_back({"VarDE", cfg, [temp, variance_floor, use_dp_value](auto env, auto init_state, int max_depth, int seed) {
                        mcts::RvipManagerArgs args(env);
                        args.max_depth = max_depth;
                        args.mcts_mode = false;
                        args.temp = temp;
                        args.variance_floor = variance_floor;
                        args.use_dp_value = use_dp_value;
                        args.seed = seed;
                        auto mgr = make_shared<mcts::RvipManager>(args);
                        auto root = make_shared<mcts::RvipDNode>(mgr, init_state, 0, 0);
                        return make_pair(static_pointer_cast<mcts::MctsManager>(mgr), static_pointer_cast<mcts::MctsDNode>(root));
                    }});
                }
            }
        }

        return cands;
    }
}

int main(int argc, char** argv) {
    (void)argc; (void)argv;

    const int grid_size = 6;
    const int init_wind_dir = 7;
    const double wind_stay_prob = 0.6;
    const double wind_turn_prob = 0.2;
    const double base_cost = 0.1;
    const double angle_cost_scale = 1.0;
    const double goal_reward = 0.0;
    const int horizon = 50;
    const int total_trials = 10000;
    const int eval_rollouts = 200;
    const int runs = 50;
    const int threads = 4;
    const int base_seed = 4242;

    auto env = make_shared<mcts::exp::SailingEnv>(
        grid_size, init_wind_dir, wind_stay_prob, wind_turn_prob, base_cost, angle_cost_scale, goal_reward);
    auto init_state = env->get_initial_state_itfc();
    auto candidates = build_candidates();

    ofstream out("tune_sailing.csv", ios::out | ios::trunc);
    out << "env,algorithm,config,run,mc_mean,mc_stddev\n";

    struct Aggregate { double sum = 0.0; int count = 0; };
    unordered_map<string, unordered_map<string, Aggregate>> agg;

    cout << "[tune] Sailing, grid_size=" << grid_size << ", horizon=" << horizon << ", trials=" << total_trials << "\n";

    for (const auto& cand : candidates) {
        for (int run = 0; run < runs; ++run) {
            int seed = base_seed + 1000 * run + (int)hash<string>{}(cand.algo + cand.config) % 997;
            auto [mgr, root] = cand.make(env, init_state, horizon, seed);
            auto pool = make_unique<mcts::MctsPool>(mgr, root, threads);
            pool->run_trials(total_trials, numeric_limits<double>::max(), true);

            auto stats = evaluate_tree(env, root, horizon, eval_rollouts, threads, seed + 777);
            out << "sailing," << cand.algo << "," << cand.config << "," << run
                << "," << stats.mean << "," << stats.stddev << "\n";

            auto& a = agg[cand.algo][cand.config];
            a.sum += stats.mean;
            a.count += 1;
        }
        cout << "  done " << cand.algo << " cfg=" << cand.config << "\n";
    }

    cout << "\nBest configs (avg over runs per config):\n";
    cout << fixed << setprecision(4);
    for (const auto& [algo, cfg_map] : agg) {
        double best_avg = -numeric_limits<double>::infinity();
        string best_cfg;
        for (const auto& [cfg, a] : cfg_map) {
            double avg = (a.count > 0) ? (a.sum / a.count) : -numeric_limits<double>::infinity();
            if (avg > best_avg) {
                best_avg = avg;
                best_cfg = cfg;
            }
        }
        cout << "  " << algo << " -> avg=" << best_avg << " cfg=" << best_cfg << "\n";
    }

    cout << "Results written to tune_sailing.csv\n";
    return 0;
}
