#include "env/taxi_env.h"

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

        cands.push_back({"UCT", [](auto env, auto init_state, int max_depth, int seed) {
            mcts::UctManagerArgs args(env);
            args.max_depth = max_depth;
            args.mcts_mode = false;
            args.bias = 5.0;
            args.epsilon_exploration = 0.1;
            args.seed = seed;
            auto mgr = make_shared<mcts::UctManager>(args);
            auto root = make_shared<mcts::UctDNode>(mgr, init_state, 0, 0);
            return make_pair(static_pointer_cast<mcts::MctsManager>(mgr), static_pointer_cast<mcts::MctsDNode>(root));
        }});

        cands.push_back({"MENTS", [](auto env, auto init_state, int max_depth, int seed) {
            mcts::MentsManagerArgs args(env);
            args.max_depth = max_depth;
            args.mcts_mode = false;
            args.temp = 0.1;
            args.epsilon = 0.75;
            args.seed = seed;
            auto mgr = make_shared<mcts::MentsManager>(args);
            auto root = make_shared<mcts::MentsDNode>(mgr, init_state, 0, 0);
            return make_pair(static_pointer_cast<mcts::MctsManager>(mgr), static_pointer_cast<mcts::MctsDNode>(root));
        }});

        cands.push_back({"RENTS", [](auto env, auto init_state, int max_depth, int seed) {
            mcts::MentsManagerArgs args(env);
            args.max_depth = max_depth;
            args.mcts_mode = false;
            args.temp = 0.1;
            args.epsilon = 0.75;
            args.seed = seed;
            auto mgr = make_shared<mcts::MentsManager>(args);
            auto root = make_shared<mcts::RentsDNode>(mgr, init_state, 0, 0);
            return make_pair(static_pointer_cast<mcts::MctsManager>(mgr), static_pointer_cast<mcts::MctsDNode>(root));
        }});

        cands.push_back({"TENTS", [](auto env, auto init_state, int max_depth, int seed) {
            mcts::MentsManagerArgs args(env);
            args.max_depth = max_depth;
            args.mcts_mode = false;
            args.temp = 3.0;
            args.epsilon = 0.75;
            args.seed = seed;
            auto mgr = make_shared<mcts::MentsManager>(args);
            auto root = make_shared<mcts::TentsDNode>(mgr, init_state, 0, 0);
            return make_pair(static_pointer_cast<mcts::MctsManager>(mgr), static_pointer_cast<mcts::MctsDNode>(root));
        }});

        cands.push_back({"DENTS", [](auto env, auto init_state, int max_depth, int seed) {
            mcts::DentsManagerArgs args(env);
            args.max_depth = max_depth;
            args.mcts_mode = false;
            args.temp = 3.0;
            args.epsilon = 0.75;
            args.seed = seed;
            args.use_dp_value = true;
            auto mgr = make_shared<mcts::DentsManager>(args);
            auto root = make_shared<mcts::DentsDNode>(mgr, init_state, 0, 0);
            return make_pair(static_pointer_cast<mcts::MctsManager>(mgr), static_pointer_cast<mcts::MctsDNode>(root));
        }});

        cands.push_back({"BTS", [](auto env, auto init_state, int max_depth, int seed) {
            mcts::DentsManagerArgs args(env);
            args.max_depth = max_depth;
            args.mcts_mode = false;
            args.temp = 5.0;
            args.epsilon = 0.75;
            args.seed = seed;
            auto mgr = make_shared<mcts::DentsManager>(args);
            auto root = make_shared<mcts::EstDNode>(mgr, init_state, 0, 0);
            return make_pair(static_pointer_cast<mcts::MctsManager>(mgr), static_pointer_cast<mcts::MctsDNode>(root));
        }});

        cands.push_back({"VarDE", [](auto env, auto init_state, int max_depth, int seed) {
            mcts::RvipManagerArgs args(env);
            args.max_depth = max_depth;
            args.mcts_mode = false;
            args.temp = 3.0;
            args.variance_floor = 10.0;
            args.seed = seed;
            args.use_dp_value = false;
            auto mgr = make_shared<mcts::RvipManager>(args);
            auto root = make_shared<mcts::RvipDNode>(mgr, init_state, 0, 0);
            return make_pair(static_pointer_cast<mcts::MctsManager>(mgr), static_pointer_cast<mcts::MctsDNode>(root));
        }});

        return cands;
    }
}

int main(int argc, char** argv) {
    (void)argc; (void)argv;

    const int horizon = 20;
    const int eval_rollouts = 200;
    const int runs = 100;
    const int threads = 4;
    const int base_seed = 4242;
    const int initial_state = 2;
    const bool is_raining = true;

    // Trial counts to evaluate at
    vector<int> trial_counts;
    for (int i = 1; i <= 30; ++i) {
        trial_counts.push_back(i * 1000);
    }

    auto env = make_shared<mcts::exp::TaxiEnv>(initial_state, is_raining);
    auto init_state = env->get_initial_state_itfc();
    auto candidates = build_candidates();

    ofstream out("results_taxi.csv", ios::out | ios::trunc);
    out << "env,algorithm,run,trial,mc_mean,mc_stddev\n";

    cout << "[exp] Taxi, horizon=" << horizon
         << ", init_state=" << initial_state << ", raining=" << (is_raining ? 1 : 0) << "\n";
    cout << "  Algorithms: " << candidates.size() << ", Runs: " << runs << ", Trial counts: " << trial_counts.size() << "\n";

    for (const auto& cand : candidates) {
        cout << "\nRunning " << cand.algo << "...\n";

        for (int run = 0; run < runs; ++run) {
            int seed = base_seed + 1000 * run + (int)hash<string>{}(cand.algo) % 997;

            // Create a fresh tree for this run
            auto [mgr, root] = cand.make(env, init_state, horizon, seed);
            auto pool = make_unique<mcts::MctsPool>(mgr, root, threads);

            // Run trials and evaluate at each checkpoint
            int trials_so_far = 0;
            for (int target_trials : trial_counts) {
                int trials_to_add = target_trials - trials_so_far;
                if (trials_to_add > 0) {
                    pool->run_trials(trials_to_add, numeric_limits<double>::max(), true);
                    trials_so_far = target_trials;
                }

                auto stats = evaluate_tree(env, root, horizon, eval_rollouts, threads, seed + 777);
                out << "taxi," << cand.algo << "," << run
                    << "," << target_trials << "," << stats.mean << "," << stats.stddev << "\n";
            }

            cout << "  " << cand.algo << " run " << (run + 1) << "/" << runs << " done\n";
        }
    }

    cout << "\nResults written to results_taxi.csv\n";
    return 0;
}
