#include "env/synthetic_tree_env.h"

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

#include <algorithm>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

using namespace std;

namespace {
    static bool is_dist_string(const std::string& s) {
        std::string v = s;
        std::transform(v.begin(), v.end(), v.begin(), ::tolower);
        return (v == "gaussian" || v == "normal" || v == "bernoulli" || v == "bern");
    }

    static mcts::exp::RewardDistribution parse_reward_dist(const std::string& s) {
        std::string v = s;
        std::transform(v.begin(), v.end(), v.begin(), ::tolower);
        if (v == "bernoulli" || v == "bern") return mcts::exp::RewardDistribution::Bernoulli;
        return mcts::exp::RewardDistribution::Gaussian;
    }

    static bool is_csv_path(const std::string& s) {
        std::string v = s;
        std::transform(v.begin(), v.end(), v.begin(), ::tolower);
        return v.size() >= 4 && v.rfind(".csv") == v.size() - 4;
    }

    static std::string trim_copy(const std::string& s) {
        const std::string whitespace = " \t\r\n";
        const auto start = s.find_first_not_of(whitespace);
        if (start == std::string::npos) return "";
        const auto end = s.find_last_not_of(whitespace);
        return s.substr(start, end - start + 1);
    }

    static void load_reward_csv(
        const std::string& path,
        std::vector<double>& means,
        std::vector<double>& stds)
    {
        std::ifstream in(path);
        if (!in.is_open()) {
            throw std::runtime_error("Failed to open reward CSV: " + path);
        }
        std::string line;
        while (std::getline(in, line)) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            std::string trimmed = trim_copy(line);
            if (trimmed.empty()) continue;
            if (!trimmed.empty() && trimmed[0] == '#') continue;

            std::stringstream ss(trimmed);
            std::string col1;
            std::string col2;
            if (!std::getline(ss, col1, ',')) continue;
            std::getline(ss, col2);
            col1 = trim_copy(col1);
            col2 = trim_copy(col2);
            if (col1.empty()) continue;

            try {
                double mean = std::stod(col1);
                double stdv = 0.0;
                if (!col2.empty()) {
                    try {
                        stdv = std::stod(col2);
                    } catch (...) {
                        stdv = 0.0;
                    }
                }
                means.push_back(mean);
                stds.push_back(stdv);
            } catch (...) {
                continue;
            }
        }
        if (means.empty()) {
            throw std::runtime_error("Reward CSV has no numeric rows: " + path);
        }
    }

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
            args.temp = 1.0;
            args.epsilon = 0.25;
            args.seed = seed;
            auto mgr = make_shared<mcts::MentsManager>(args);
            auto root = make_shared<mcts::MentsDNode>(mgr, init_state, 0, 0);
            return make_pair(static_pointer_cast<mcts::MctsManager>(mgr), static_pointer_cast<mcts::MctsDNode>(root));
        }});

        cands.push_back({"RENTS", [](auto env, auto init_state, int max_depth, int seed) {
            mcts::MentsManagerArgs args(env);
            args.max_depth = max_depth;
            args.mcts_mode = false;
            args.temp = 1.0;
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
            args.temp = 5.0;
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
            args.temp = 1.0;
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
            args.epsilon = 0.25;
            args.seed = seed;
            auto mgr = make_shared<mcts::DentsManager>(args);
            auto root = make_shared<mcts::EstDNode>(mgr, init_state, 0, 0);
            return make_pair(static_pointer_cast<mcts::MctsManager>(mgr), static_pointer_cast<mcts::MctsDNode>(root));
        }});

        cands.push_back({"VarDE", [](auto env, auto init_state, int max_depth, int seed) {
            mcts::RvipManagerArgs args(env);
            args.max_depth = max_depth;
            args.mcts_mode = false;
            args.temp = 1.0;
            args.variance_floor = 1.0;
            args.seed = seed;
            args.use_dp_value = true;
            auto mgr = make_shared<mcts::RvipManager>(args);
            auto root = make_shared<mcts::RvipDNode>(mgr, init_state, 0, 0);
            return make_pair(static_pointer_cast<mcts::MctsManager>(mgr), static_pointer_cast<mcts::MctsDNode>(root));
        }});

        return cands;
    }
}

int main(int argc, char** argv) {
    // CLI: run_stree [reward_csv] [reward_dist]

    const int depth = 5;
    const int branching_factor = 3;
    const double gamma = 1.0;
    const double action_noise = 0.2;
    const int reward_seed = 4242;
    auto reward_dist = mcts::exp::RewardDistribution::Gaussian;
    std::vector<double> reward_mean_values;
    std::vector<double> reward_std_values;
    std::string reward_csv = "stree_rewards.csv";
    int argi = 1;
    if (argi < argc && is_csv_path(argv[argi])) {
        reward_csv = argv[argi++];
    }
    load_reward_csv(reward_csv, reward_mean_values, reward_std_values);
    if (argi < argc && is_dist_string(argv[argi])) {
        reward_dist = parse_reward_dist(argv[argi]);
    }
    const int eval_rollouts = 1000;
    const int runs = 100;
    const int threads = 4;
    const int base_seed = 4242;

    vector<int> trial_counts;
    for (int i = 1; i <= 100; ++i) {
        trial_counts.push_back(i * 50);
    }

    auto env = make_shared<mcts::exp::SyntheticTreeEnv>(
        branching_factor,
        depth,
        gamma,
        action_noise,
        reward_seed,
        reward_dist,
        reward_mean_values,
        reward_std_values);
    auto init_state = env->get_initial_state_itfc();
    auto candidates = build_candidates();

    ofstream out("results_stree.csv", ios::out | ios::trunc);
    out << "env,algorithm,run,trial,mc_mean,mc_stddev\n";

    cout << "[exp] STree, depth=" << depth
         << ", branching=" << branching_factor
         << ", action_noise=" << action_noise
         << ", reward_csv=" << (reward_csv.empty() ? "<none>" : reward_csv)
         << ", reward_dist=" << (reward_dist == mcts::exp::RewardDistribution::Bernoulli ? "bernoulli" : "gaussian")
         << "\n";
    cout << "  Algorithms: " << candidates.size()
         << ", Runs: " << runs
         << ", Trial counts: " << trial_counts.size() << "\n";

    for (const auto& cand : candidates) {
        cout << "\nRunning " << cand.algo << "...\n";

        for (int run = 0; run < runs; ++run) {
            int seed = base_seed + 1000 * run + (int)hash<string>{}(cand.algo) % 997;

            auto [mgr, root] = cand.make(env, init_state, depth, seed);
            auto pool = make_unique<mcts::MctsPool>(mgr, root, threads);

            int trials_so_far = 0;
            for (int target_trials : trial_counts) {
                int trials_to_add = target_trials - trials_so_far;
                if (trials_to_add > 0) {
                    pool->run_trials(trials_to_add, numeric_limits<double>::max(), true);
                    trials_so_far = target_trials;
                }

                auto stats = evaluate_tree(env, root, depth, eval_rollouts, threads, seed + 777);
                out << "stree," << cand.algo << "," << run
                    << "," << target_trials << "," << stats.mean << "," << stats.stddev << "\n";
            }

            cout << "  " << cand.algo << " run " << (run + 1) << "/" << runs << " done\n";
        }
    }

    cout << "\nResults written to results_stree.csv\n";
    return 0;
}
