#include "env/sailing_env.h"

#include "algorithms/varde/varde_decision_node.h"
#include "algorithms/varde/varde_manager.h"

#include "mc_eval.h"
#include "mcts.h"

#include <algorithm>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <queue>
#include <string>
#include <unordered_set>
#include <tuple>
#include <vector>
#include <map>
#include <set>

using namespace std;
using mcts::RvipCNode;
using mcts::RvipDNode;
using mcts::RvipManager;

// Debug-friendly VarDE nodes exposing internal stats.
class RvipDNodeDbg;

class RvipCNodeDbg : public RvipCNode {
    public:
        using RvipCNode::RvipCNode;

        int visits() const { return num_visits; }
        int backups() const { return num_backups; }
        double dp() const { return dp_value; }
        double mean() const { return avg_return; }
        double variance() const { return get_variance(); }

        std::vector<std::pair<std::shared_ptr<const mcts::State>, std::shared_ptr<RvipDNodeDbg>>> child_nodes() const {
            std::vector<std::pair<std::shared_ptr<const mcts::State>, std::shared_ptr<RvipDNodeDbg>>> out;
            for (const auto& kv : children) {
                auto obsv = std::static_pointer_cast<const mcts::State>(kv.first);
                auto child = std::static_pointer_cast<RvipDNodeDbg>(kv.second);
                out.emplace_back(obsv, child);
            }
            return out;
        }

    protected:
        shared_ptr<RvipDNode> create_child_node_helper(shared_ptr<const mcts::State> observation,
                                                        shared_ptr<const mcts::State> next_state = nullptr) const {
            auto child = make_shared<RvipDNodeDbg>(
                static_pointer_cast<RvipManager>(mcts_manager),
                observation,
                decision_depth + 1,
                decision_timestep + 1,
                static_pointer_cast<const RvipCNodeDbg>(shared_from_this()));
            return static_pointer_cast<RvipDNode>(child);
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

class RvipDNodeDbg : public RvipDNode {
    public:
        using RvipDNode::RvipDNode;

        int visits() const { return num_visits; }
        int backups() const { return num_backups; }
        double dp() const { return dp_value; }
        double mean() const { return avg_return; }

        vector<tuple<string,int,double,double,double,double>> child_snapshot() {
            vector<tuple<string,int,double,double,double,double>> rows;
            for (const auto& a : *actions) {
                if (!has_child_node(a)) continue;
                auto c = static_pointer_cast<RvipCNodeDbg>(get_child_node(a));
                double influence = compute_influence_weight(a);
                rows.emplace_back(
                    a->get_pretty_print_string(),
                    c->visits(),
                    c->dp(),
                    c->mean(),
                    c->variance(),
                    influence);
            }
            return rows;
        }

        std::shared_ptr<const mcts::State> get_state_ptr() const { return state; }
        int depth() const { return decision_depth; }
        std::vector<std::pair<std::shared_ptr<const mcts::Action>, std::shared_ptr<RvipCNodeDbg>>> child_cnodes() const {
            std::vector<std::pair<std::shared_ptr<const mcts::Action>, std::shared_ptr<RvipCNodeDbg>>> out;
            if (actions == nullptr) return out;
            for (const auto& a : *actions) {
                if (!has_child_node(a)) continue;
                auto c = static_pointer_cast<RvipCNodeDbg>(get_child_node(a));
                out.emplace_back(a, c);
            }
            return out;
        }

    protected:
        shared_ptr<mcts::MctsCNode> create_child_node_helper_itfc(
            shared_ptr<const mcts::Action> action) const override
        {
            auto child = make_shared<RvipCNodeDbg>(
                static_pointer_cast<RvipManager>(mcts_manager),
                state,
                action,
                decision_depth,
                decision_timestep,
                static_pointer_cast<const RvipDNodeDbg>(shared_from_this()));
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

static void print_policy_dump(shared_ptr<RvipDNodeDbg> root) {
    try {
        cout << "\n[VarDE Policy Dump]\n";
        std::queue<std::shared_ptr<RvipDNodeDbg>> q;
        struct TupleHash {
            size_t operator()(const std::tuple<int,int,int>& tpl) const {
                auto [a,b,c] = tpl;
                size_t h1 = std::hash<int>{}(a);
                size_t h2 = std::hash<int>{}(b);
                size_t h3 = std::hash<int>{}(c);
                return ((h1 * 1315423911u) ^ (h2 + 2654435761u)) ^ h3;
            }
        };
        std::unordered_set<std::tuple<int,int,int>, TupleHash> seen;

        q.push(root);
        mcts::MctsEnvContext ctx;
        size_t node_count = 0;
        int max_depth_seen = 0;
        const size_t MAX_NODES = 10000;

        while (!q.empty() && node_count < MAX_NODES) {
            auto d = q.front();
            q.pop();
            if (!d) continue;

            auto state = d->get_state_ptr();
            string state_str = state ? state->get_pretty_print_string() : "<null>";
            max_depth_seen = std::max(max_depth_seen, d->depth());
            node_count++;

            std::shared_ptr<const mcts::Action> rec_action = nullptr;
            string rec_str = "<none>";
            bool is_terminal = d->child_cnodes().empty();
            
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

            cout << "state=" << state_str << " depth=" << d->depth() << " visits=" << d->visits()
                 << " backups=" << d->backups() << " dp=" << d->dp() << " rec=" << rec_str << "\n";

            for (const auto& [action, cnode] : d->child_cnodes()) {
                for (const auto& [obsv, dchild] : cnode->child_nodes()) {
                    if (dchild) q.push(dchild);
                }
            }
        }

        if (node_count >= MAX_NODES) {
            cout << "[WARNING] Reached max node limit (" << MAX_NODES << ")\n";
        }
        cout << "[VarDE Policy Dump End] nodes=" << node_count << " max_depth=" << max_depth_seen << "\n";
    } catch (const std::exception& e) {
        cout << "[VarDE Policy Dump ERROR] " << e.what() << "\n";
    }
}

static void print_tree_statistics(shared_ptr<RvipManager> mgr, shared_ptr<RvipDNodeDbg> root) {
    cout << "\n[VarDE Tree Statistics]\n";
    try {
        std::map<int, int> depth_counts;
        depth_counts[0] = 1;
        std::queue<std::shared_ptr<RvipDNodeDbg>> q;
        std::set<void*> seen_ptrs;
        q.push(root);
        seen_ptrs.insert(root.get());
        int total = 1;
        int max_d = 0;
        
        while (!q.empty()) {
            auto d = q.front();
            q.pop();
            max_d = std::max(max_d, d->depth());
            
            for (const auto& [action, cnode] : d->child_cnodes()) {
                if (!cnode) continue;
                for (const auto& [obsv, dchild] : cnode->child_nodes()) {
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
    }
}

int main(int argc, char** argv) {
    // CLI: debug_varde_sailing <temp> <trials> <horizon> <rollouts> <variance_floor> [value_source]
    // value_source: 'dp'|'mean' (default: 'dp')
    ios::sync_with_stdio(false);
    cout << unitbuf;
    
    double temp = 1.0;
    int total_trials = 20000;
    int horizon = 50;
    int rollouts = 200;
    double variance_floor = 1.0;
    int threads = 4;
    int seed = 4242;
    int grid_size = 6;
    double wind_stay_prob = 0.6;
    double wind_turn_prob = 0.2;
    double base_cost = 0.1;
    double angle_cost_scale = 1.0;    
    double goal_reward = 0.0;
    if (argc > 1) temp = atof(argv[1]);
    if (argc > 2) total_trials = atoi(argv[2]);
    if (argc > 3) horizon = atoi(argv[3]);
    if (argc > 4) rollouts = atoi(argv[4]);
    if (argc > 5) variance_floor = atof(argv[5]);
    
    bool use_dp_value = false;
    if (argc > 6) {
        std::string vsrc = argv[6];
        std::transform(vsrc.begin(), vsrc.end(), vsrc.begin(), ::tolower);
        if (vsrc == "mean" || vsrc == "0" || vsrc == "false") use_dp_value = false;
        else use_dp_value = true;
    }

    auto env = make_shared<mcts::exp::SailingEnv>(
        grid_size, 7, wind_stay_prob, wind_turn_prob, base_cost, angle_cost_scale, goal_reward);
    auto init_state = env->get_initial_state_itfc();

    mcts::RvipManagerArgs args(env);
    args.max_depth = horizon;
    args.mcts_mode = false;
    args.temp = temp;
    args.variance_floor = variance_floor;
    args.seed = seed;
    args.use_dp_value = use_dp_value;
    auto mgr = make_shared<RvipManager>(args);
    auto root = make_shared<RvipDNodeDbg>(mgr, init_state, 0, 0);

    auto pool = make_unique<mcts::MctsPool>(mgr, root, threads);

    cout << "[VarDE dbg Sailing] starting" 
        << " temp=" << temp 
        << " var_floor=" << variance_floor 
        << " trials=" << total_trials 
        << " horizon=" << horizon 
        << " rollouts=" << rollouts 
        << " threads=" << threads 
        << " value_src=" << (use_dp_value ? "dp" : "mean")
        << "\n";

    vector<int> checkpoints;
    for (int i = 1; i <= 20; ++i) {
        checkpoints.push_back(i * 1000);
    }
    int trials_done = 0;
    cout << fixed << setprecision(4);
    for (int cp : checkpoints) {
        int delta = cp - trials_done;
        if (delta > 0) {
            try {
                pool->run_trials(delta, numeric_limits<double>::max(), true);
                trials_done = cp;
            } catch (const exception& e) {
                cout << "[ERROR] Exception during run_trials: " << e.what() << "\n";
                return 1;
            }
        }

        auto stats = eval_policy(env, root, horizon, rollouts, threads, seed + 888 + cp);
        auto dbg_root = root;
        auto rows = dbg_root->child_snapshot();
        sort(rows.begin(), rows.end(), [](const auto& a, const auto& b) {
            return get<1>(a) > get<1>(b);
        });

        cout << "[VarDE dbg] temp=" << temp
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
                 << " mean=" << get<3>(r)
                 << " var=" << get<4>(r)
                 << " influence=" << get<5>(r) << "\n";
        }
    }

    mcts::MctsEnvContext ctx;
    try {
        auto recommended_action = root->recommend_action_itfc(ctx);
        cout << "\n[VarDE Final Policy] Recommended action: " 
             << recommended_action->get_pretty_print_string() << "\n";
    } catch (const std::exception& e) {
        cout << "\n[VarDE Final Policy] Error: " << e.what() << "\n";
    }

    //print_policy_dump(root);
    //print_tree_statistics(mgr, root);
    
    // Explicitly clean up the pool and force exit
    pool.reset();
    cout.flush();
    exit(0);
}
