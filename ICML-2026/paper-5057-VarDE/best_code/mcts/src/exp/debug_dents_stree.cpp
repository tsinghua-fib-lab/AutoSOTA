#include "env/synthetic_tree_env.h"

#include "algorithms/ments/dents/dents_decision_node.h"
#include "algorithms/ments/dents/dents_manager.h"

#include "mc_eval.h"
#include "mcts.h"

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <queue>
#include <sstream>
#include <set>
#include <string>
#include <stdexcept>
#include <unordered_set>
#include <tuple>
#include <vector>

using namespace std;
using mcts::DentsCNode;
using mcts::DentsDNode;
using mcts::DentsManager;

static bool is_dist_string(const std::string& s) {
    std::string v = s;
    std::transform(v.begin(), v.end(), v.begin(), ::tolower);
    return (v == "gaussian" || v == "normal" || v == "bernoulli" || v == "bern");
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

static mcts::exp::RewardDistribution parse_reward_dist(const std::string& s) {
    std::string v = s;
    std::transform(v.begin(), v.end(), v.begin(), ::tolower);
    if (v == "bernoulli" || v == "bern") return mcts::exp::RewardDistribution::Bernoulli;
    return mcts::exp::RewardDistribution::Gaussian;
}

// Debug-friendly DENTS nodes exposing internal stats.
class DentsDNodeDbg;

class DentsCNodeDbg : public DentsCNode {
    public:
        using DentsCNode::DentsCNode;

        int visits() const { return num_visits; }
        int backups() const { return mcts::MentsCNode::num_backups; }
        double dp() const { return dp_value; }
        double avg() const { return avg_return; }
        double soft() const { return soft_value; }
        double entropy() const { return subtree_entropy; }

        std::vector<std::pair<std::shared_ptr<const mcts::State>, std::shared_ptr<DentsDNodeDbg>>> child_nodes() const {
            std::vector<std::pair<std::shared_ptr<const mcts::State>, std::shared_ptr<DentsDNodeDbg>>> out;
            for (const auto& kv : children) {
                auto obsv = std::static_pointer_cast<const mcts::State>(kv.first);
                auto child = std::static_pointer_cast<DentsDNodeDbg>(kv.second);
                out.emplace_back(obsv, child);
            }
            return out;
        }

    protected:
        shared_ptr<DentsDNode> create_child_node_helper(shared_ptr<const mcts::State> observation,
                                                        shared_ptr<const mcts::State> next_state = nullptr) const {
            auto child = make_shared<DentsDNodeDbg>(
                static_pointer_cast<DentsManager>(mcts_manager),
                observation,
                decision_depth + 1,
                decision_timestep + 1,
                static_pointer_cast<const DentsCNodeDbg>(shared_from_this()));
            return static_pointer_cast<DentsDNode>(child);
        }

        shared_ptr<mcts::MctsDNode> create_child_node_helper_itfc(
            shared_ptr<const mcts::Observation> observation,
            shared_ptr<const mcts::State> next_state = nullptr) const override
        {
            auto obsv_state = static_pointer_cast<const mcts::State>(observation);
            auto child = create_child_node_helper(obsv_state, next_state);
            return static_pointer_cast<mcts::MctsDNode>(child);
        }
};

class DentsDNodeDbg : public DentsDNode {
    public:
        using DentsDNode::DentsDNode;

        int visits() const { return num_visits; }
        int backups() const { return mcts::MentsDNode::num_backups; }
        double dp() const { return dp_value; }
        double avg() const { return avg_return; }
        double soft() const { return soft_value; }
        double entropy() const { return subtree_entropy; }

        std::shared_ptr<const mcts::State> get_state_ptr() const { return state; }
        int depth() const { return decision_depth; }
        std::vector<std::pair<std::shared_ptr<const mcts::Action>, std::shared_ptr<DentsCNodeDbg>>> child_cnodes() const {
            std::vector<std::pair<std::shared_ptr<const mcts::Action>, std::shared_ptr<DentsCNodeDbg>>> out;
            if (actions == nullptr) return out;
            for (const auto& a : *actions) {
                if (!has_child_node(a)) continue;
                auto c = static_pointer_cast<DentsCNodeDbg>(get_child_node(a));
                out.emplace_back(a, c);
            }
            return out;
        }

        vector<tuple<string,int,double,double,double,double>> child_snapshot() {
            vector<tuple<string,int,double,double,double,double>> rows;
            lock_all_children();
            for (const auto& a : *actions) {
                if (!has_child_node(a)) continue;
                auto c = static_pointer_cast<DentsCNodeDbg>(get_child_node(a));
                rows.emplace_back(
                    a->get_pretty_print_string(),
                    c->visits(),
                    c->dp(),
                    c->avg(),
                    c->soft(),
                    c->entropy());
            }
            unlock_all_children();
            return rows;
        }

    protected:
        shared_ptr<mcts::MctsCNode> create_child_node_helper_itfc(
            shared_ptr<const mcts::Action> action) const override
        {
            auto child = make_shared<DentsCNodeDbg>(
                static_pointer_cast<DentsManager>(mcts_manager),
                state,
                action,
                decision_depth,
                decision_timestep,
                static_pointer_cast<const DentsDNodeDbg>(shared_from_this()));
            return static_pointer_cast<mcts::MctsCNode>(child);
        }
};

static pair<double, double> eval_policy(
    shared_ptr<const mcts::MctsEnv> env,
    shared_ptr<const mcts::MctsDNode> root,
    int horizon,
    int rollouts,
    int threads,
    int seed)
{
    mcts::RandManager rng(seed);
    mcts::EvalPolicy policy(root, env, rng);
    mcts::MCEvaluator evaluator(env, policy, horizon, rng);
    evaluator.run_rollouts(rollouts, threads);
    return {evaluator.get_mean_return(), evaluator.get_stddev_return()};
}

static void print_policy_dump(shared_ptr<DentsDNodeDbg> root) {
    try {
        cout << "\n[DENTS Policy Dump]\n";
        std::queue<std::shared_ptr<DentsDNodeDbg>> q;
        auto state_key = [](const std::shared_ptr<const mcts::State>& s) {
            auto t2 = std::dynamic_pointer_cast<const mcts::IntPairState>(s);
            if (t2) return t2->state;
            return std::make_pair(-9999, -9999);
        };
        struct PairHash {
            size_t operator()(const std::pair<int,int>& pr) const {
                size_t h1 = std::hash<int>{}(pr.first);
                size_t h2 = std::hash<int>{}(pr.second);
                return h1 ^ (h2 + 0x9e3779b9 + (h1 << 6) + (h1 >> 2));
            }
        };
        std::unordered_set<std::pair<int,int>, PairHash> seen;

        q.push(root);
        seen.insert(state_key(root->get_state_ptr()));

        mcts::MctsEnvContext ctx;
        size_t node_count = 0;
        int max_depth_seen = 0;
        const size_t MAX_NODES = 10000;

        while (!q.empty() && node_count < MAX_NODES) {
            try {
                auto d = q.front();
                q.pop();
                if (!d) continue;

                auto state = d->get_state_ptr();
                string state_str = state ? state->get_pretty_print_string() : "<null>";

                max_depth_seen = std::max(max_depth_seen, d->depth());
                node_count++;

                std::shared_ptr<const mcts::Action> rec_action = nullptr;
                string rec_str = "<none>";

                bool is_terminal = false;
                {
                    d->lock_all_children();
                    auto test_children = d->child_cnodes();
                    d->unlock_all_children();
                    is_terminal = test_children.empty();
                }

                if (!is_terminal) {
                    try {
                        rec_action = d->recommend_action_itfc(ctx);
                        rec_str = rec_action ? rec_action->get_pretty_print_string() : "<none>";
                    } catch (...) {
                        rec_str = "<error>";
                    }
                } else {
                    rec_str = "<terminal>";
                }

                double rec_val = numeric_limits<double>::quiet_NaN();
                bool has_val = false;
                if (rec_action && d->has_child_node(rec_action)) {
                    try {
                        auto c = std::static_pointer_cast<DentsCNodeDbg>(d->get_child_node(rec_action));
                        if (c) {
                            rec_val = c->avg();
                            has_val = true;
                        }
                    } catch (...) {}
                }

                cout << "state=" << state_str
                     << " depth=" << d->depth()
                     << " visits=" << d->visits()
                     << " backups=" << d->backups()
                     << " dp=" << d->dp()
                     << " rec=" << rec_str;
                if (has_val) cout << " q=" << rec_val;
                cout << " avg=" << d->avg()
                     << " soft=" << d->soft()
                     << " ent=" << d->entropy()
                     << "\n";

                d->lock_all_children();
                std::vector<std::pair<std::shared_ptr<const mcts::Action>, std::shared_ptr<DentsCNodeDbg>>> cchildren;
                try {
                    cchildren = d->child_cnodes();
                } catch (...) {}
                d->unlock_all_children();

                for (const auto& [action, cnode] : cchildren) {
                    (void)action;
                    if (!cnode) continue;

                    cnode->lock_all_children();
                    std::vector<std::pair<std::shared_ptr<const mcts::State>, std::shared_ptr<DentsDNodeDbg>>> dchildren;
                    try {
                        dchildren = cnode->child_nodes();
                    } catch (...) {}
                    cnode->unlock_all_children();

                    for (const auto& [obsv, dchild] : dchildren) {
                        auto key = state_key(obsv);
                        if (dchild && seen.insert(key).second) {
                            q.push(dchild);
                        }
                    }
                }
            } catch (const std::exception& e) {
                cout << "[ERROR] Exception: " << e.what() << "\n";
                break;
            } catch (...) {
                cout << "[ERROR] Unknown exception\n";
                break;
            }
        }

        if (node_count >= MAX_NODES) {
            cout << "[WARNING] Reached max node limit (" << MAX_NODES << "), stopping early\n";
        }

        cout << "[DENTS Policy Dump End] nodes=" << node_count << " max_depth=" << max_depth_seen << "\n";
    } catch (const std::exception& e) {
        cout << "[DENTS Policy Dump ERROR] " << e.what() << "\n";
    } catch (...) {
        cout << "[DENTS Policy Dump ERROR] Unknown exception\n";
    }
}

static void print_tree_statistics(shared_ptr<DentsManager> mgr, shared_ptr<DentsDNodeDbg> root) {
    cout << "\n[DENTS Tree Statistics]\n";

    try {
        std::map<int, int> depth_counts;
        depth_counts[0] = 1;

        std::queue<std::shared_ptr<DentsDNodeDbg>> q;
        std::set<void*> seen_ptrs;

        q.push(root);
        seen_ptrs.insert(root.get());

        int total = 1;
        int max_d = 0;

        while (!q.empty()) {
            auto d = q.front();
            q.pop();

            max_d = std::max(max_d, d->depth());

            d->lock_all_children();
            auto cchildren = d->child_cnodes();
            d->unlock_all_children();

            for (const auto& [action, cnode] : cchildren) {
                (void)action;
                if (!cnode) continue;

                cnode->lock_all_children();
                auto dchildren = cnode->child_nodes();
                cnode->unlock_all_children();

                for (const auto& [obsv, dchild] : dchildren) {
                    if (!dchild) continue;
                    void* dptr = dchild.get();
                    if (seen_ptrs.find(dptr) == seen_ptrs.end()) {
                        seen_ptrs.insert(dptr);
                        depth_counts[dchild->depth()]++;
                        total++;
                        q.push(dchild);
                    }
                }
            }
        }

        cout << "Total nodes: " << total << "\n";
        cout << "Max depth: " << max_d << "\n";
        cout << "Nodes by depth:\n";
        for (const auto& [d, cnt] : depth_counts) {
            cout << "  depth=" << d << " count=" << cnt << "\n";
        }
    } catch (const std::exception& e) {
        cout << "Exception: " << e.what() << "\n";
    } catch (...) {
        cout << "Unknown exception\n";
    }
}

int main(int argc, char** argv) {
    // CLI: debug_dents_stree <temp> <trials> <depth> <branching> <action_noise> <rollouts>
    //                      [gamma] [reward_seed] [reward_csv] [reward_dist]
    // reward_dist: gaussian|normal|bernoulli|bern (default: gaussian)
    double temp = 1.0;
    int total_trials = 10000;
    int depth = 5;
    int branching_factor = 3;
    double action_noise = 0.2;
    int rollouts = 1000;
    double gamma = 1.0;
    int reward_seed = 4242;
    mcts::exp::RewardDistribution reward_dist = mcts::exp::RewardDistribution::Gaussian;
    bool resample_rewards_each_rollout = true;
    std::vector<double> reward_mean_values;
    std::vector<double> reward_std_values;
    std::string reward_csv = "stree_rewards.csv";
    int threads = 4;
    int seed = 4242;

    int argi = 1;
    if (argi < argc) temp = atof(argv[argi++]);
    if (argi < argc) total_trials = atoi(argv[argi++]);
    if (argi < argc) depth = atoi(argv[argi++]);
    if (argi < argc) branching_factor = atoi(argv[argi++]);
    if (argi < argc) action_noise = atof(argv[argi++]);
    if (argi < argc) rollouts = atoi(argv[argi++]);
    if (argi < argc) gamma = atof(argv[argi++]);
    if (argi < argc) reward_seed = atoi(argv[argi++]);

    if (argi < argc && is_csv_path(argv[argi])) {
        reward_csv = argv[argi++];
        load_reward_csv(reward_csv, reward_mean_values, reward_std_values);
    }

    if (argi < argc && is_dist_string(argv[argi])) {
        reward_dist = parse_reward_dist(argv[argi]);
    }

    auto env = make_shared<mcts::exp::SyntheticTreeEnv>(
        branching_factor,
        depth,
        gamma,
        action_noise,
        reward_seed,
        reward_dist,
        reward_mean_values,
        reward_std_values,
        0.0,
        1.0,
        resample_rewards_each_rollout);
    auto init_state = env->get_initial_state_itfc();

    mcts::DentsManagerArgs args(env);
    args.max_depth = depth;
    args.mcts_mode = false;
    args.temp = temp;
    args.seed = seed;
    auto mgr = make_shared<DentsManager>(args);
    auto root = make_shared<DentsDNodeDbg>(mgr, init_state, 0, 0);

    auto pool = make_unique<mcts::MctsPool>(mgr, root, threads);

    cout << "[DENTS dbg STree] starting"
         << " temp=" << temp
         << " trials=" << total_trials
         << " depth=" << depth
         << " branching=" << branching_factor
         << " action_noise=" << action_noise
         << " rollouts=" << rollouts
         << " gamma=" << gamma
         << " reward_seed=" << reward_seed
         << " reward_csv=" << (reward_csv.empty() ? "<none>" : reward_csv)
         << " resample=" << (resample_rewards_each_rollout ? 1 : 0)
         << " reward_dist=" << (reward_dist == mcts::exp::RewardDistribution::Bernoulli ? "bernoulli" : "gaussian")
         << "\n";

    vector<int> checkpoints = {};
    for (int i = 1; i <= 20; i++) {
        checkpoints.push_back(i * (total_trials / 20));
    }
    int trials_done = 0;
    cout << fixed << setprecision(4);
    for (int cp : checkpoints) {
        int delta = cp - trials_done;
        if (delta > 0) {
            pool->run_trials(delta, numeric_limits<double>::max(), true);
            trials_done = cp;
        }

        auto stats = eval_policy(env, root, depth, rollouts, threads, seed + 888 + cp);
        auto dbg_root = root;
        auto rows = dbg_root->child_snapshot();
        sort(rows.begin(), rows.end(), [](const auto& a, const auto& b) {
            return get<1>(a) > get<1>(b);
        });

        cout << "[DENTS dbg] temp=" << temp
             << " trials=" << trials_done
             << " mean=" << stats.first
             << " std=" << stats.second
             << " root_children=" << rows.size() << "\n";

        int top = 0;
        for (const auto& r : rows) {
            if (top++ >= 5) break;
            cout << "    action=" << get<0>(r)
                 << " visits=" << get<1>(r)
                 << " dp=" << get<2>(r)
                 << " avg=" << get<3>(r)
                 << " soft=" << get<4>(r)
                 << " ent=" << get<5>(r) << "\n";
        }
    }

    mcts::MctsEnvContext ctx;
    auto recommended_action = root->recommend_action_itfc(ctx);
    cout << "\n[DENTS Final Policy] Recommended action: "
         << recommended_action->get_pretty_print_string() << "\n";

    print_policy_dump(root);
    print_tree_statistics(mgr, root);

    return 0;
}
