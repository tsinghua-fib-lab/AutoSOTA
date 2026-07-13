#include "algorithms/uct/uct_decision_node.h"

#include "helper_templates.h"

#include <cmath>
#include <float.h>
#include <sstream>
#include <vector>

using namespace std; 

namespace mcts {
    /**
     * Constructor, inits members. 
     * 
     * If we have a heuristic function, then initialises 'num_visits' and 'avg_return' according to the heuristic in 
     * the manager. If we have a prior function, then
     */
    UctDNode::UctDNode(
        shared_ptr<UctManager> mcts_manager,
        shared_ptr<const State> state,
        int decision_depth,
        int decision_timestep,
        shared_ptr<const UctCNode> parent) :
            MctsDNode(
                static_pointer_cast<MctsManager>(mcts_manager),
                state,
                decision_depth,
                decision_timestep,
                static_pointer_cast<const MctsCNode>(parent)),
            num_backups(0),
            avg_return(0.0),
            actions(mcts_manager->mcts_env->get_valid_actions_itfc(state)),
            policy_prior() 
    {   
        if (mcts_manager->heuristic_fn != nullptr) {
            num_visits = mcts_manager->heuristic_psuedo_trials;
            num_backups = mcts_manager->heuristic_psuedo_trials;
            avg_return = heuristic_value;
        }

        if (mcts_manager->prior_fn != nullptr) {
            policy_prior = mcts_manager->prior_fn(state, mcts_manager->mcts_env);
        }
    }
    
    /**
     * Helper function for checking if we have a prior or not. 
     * 
     * Code more readable with 'has_prior()' rather than checking against a nullptr.
     */
    bool UctDNode::has_prior() const {
        UctManager& manager = *static_pointer_cast<UctManager>(mcts_manager);
        return manager.prior_fn != nullptr;
    }

    /**
     * Visit just needs to call base implmentation in MctsDNode: increments num_visits
     */
    void UctDNode::visit(MctsEnvContext& ctx) {
        MctsDNode::visit_itfc(ctx);
    }

    /**
     * Computes the ucb term used in action selection. I.e. sqrt(log N(s) / N(s,a)).
     */
    double UctDNode::compute_ucb_term(int num_visits, int child_visits) const {
        double num_visits_d = (num_visits > 0) ? (double)num_visits : 1.0;
        double child_visits_d = (child_visits > 0) ? (double)child_visits : 1.0;
        return sqrt(log(num_visits_d) / child_visits_d);
    }

    /**
     * Helper to compute ucb values
     * 
     * Iterates through all possible actions, and compute ucb values for them. This function assumes that we want a 
     * value for every single action.
     * 
     * Additionally uses the adaptive bias from PROST.
     * 
     * The value computed is of the form:
     *      q_value + bias * ucb_term 
     * 
     * If we have a policy prior, then we use the ucb value of form:
     *      q_value + prior(action) * bias * ucb_term
     * 
     * If we are playing a 2 player game, then we assume at opponent nodes that the policy prior is computed to 
     * minimise the value, so we use the value of form (where opp_coeff is -1 or 1):
     *      opp_coeff * q_value + prior(action) * bias * ucb_term
     * 
     * TODO: Consider fine grained locking if want to optimise. Probably don't need bias and values to be held super 
     *      consistent throughout function.
     */
    void UctDNode::fill_ucb_values(unordered_map<shared_ptr<const Action>,double>& ucb_values, MctsEnvContext& ctx) const {
        shared_ptr<UctManager> manager = static_pointer_cast<UctManager>(mcts_manager);
        double opp_coeff = is_opponent() ? -1.0 : 1.0;

        // Lock all children
        lock_all_children();

        // Compute adaptive bias if using
        double bias = manager->bias; 
        if (bias == UctManager::USE_AUTO_BIAS) {
            bias = UctManager::AUTO_BIAS_MIN_BIAS;
            for (shared_ptr<const Action> action : *actions) {
                if (!has_child_node(action)) continue;
                double child_abs_val = abs(get_child_node(action)->avg_return);
                if (child_abs_val > bias) bias = child_abs_val;
            }
        }

        // Compute usb values
        for (shared_ptr<const Action> action : *actions) {
            double action_ucb_value = 0.0;

            int child_visits = (has_child_node(action)) ? get_child_node(action)->num_visits : 0;
            action_ucb_value += compute_ucb_term(num_visits, child_visits);
            action_ucb_value *= bias;
            if (has_prior()) {
                action_ucb_value *= policy_prior->at(action);
            }
            
            if (has_child_node(action)) {
                action_ucb_value += opp_coeff * get_child_node(action)->avg_return;
            }

            ucb_values[action] = action_ucb_value;
        }  

        // unlock all children
        unlock_all_children();      
    }

    /**
     * Selects an action according to the ucb algorithm, creating child nodes as necessary.
     * 
     * If we have a policy prior, just go ahead and compute a ucb value for all actions, and pick the best. 
     * 
     * Otherwise we do standard UCB, by 'pulling each arm' (action) once first.
     */
    shared_ptr<const Action> UctDNode::select_action_ucb(MctsEnvContext& ctx) {
        // Pull uninitialised arms if needed
        if (!has_prior()) {
            vector<shared_ptr<const Action>> actions_yet_to_try;
            for (shared_ptr<const Action> action : *actions) {
                if (!has_child_node(action)) {
                    actions_yet_to_try.push_back(action);
                }
            }

            if (actions_yet_to_try.size() > 0) {
                int indx = mcts_manager->get_rand_int(0,actions_yet_to_try.size());
                shared_ptr<const Action> action = actions_yet_to_try[indx];
                create_child_node(action);
                return action;
            }
        }
        
        // Compute ucb values and return action with max value
        unordered_map<shared_ptr<const Action>,double> ucb_values;
        fill_ucb_values(ucb_values, ctx);
        shared_ptr<const Action> result_action = helper::get_max_key_break_ties_randomly(ucb_values, *mcts_manager);

        // Remember to create the child node if it doesnt exist!
        if (!has_child_node(result_action)) {
            create_child_node(result_action);
        }
        return result_action;
    }

    /**
     * Selects a (uniformly) random action, creating the child if it doesn't yet exist.
     */
    shared_ptr<const Action> UctDNode::select_action_random() {
        int index = mcts_manager->get_rand_int(0, actions->size());
        shared_ptr<const Action> action = actions->at(index);
        if (!has_child_node(action)) {
            create_child_node(action);
        }
        return action;
    }
    
    /**
     * Select action function.
     * 
     * Decides randomly if we need to do epsilon exploration, and appropriately calls the ucb (or random) select action
     * function for if we didn't (or did) want to explore this trial.
     */
    shared_ptr<const Action> UctDNode::select_action(MctsEnvContext& ctx) {
        shared_ptr<UctManager> manager = static_pointer_cast<UctManager>(mcts_manager);
        if (manager->epsilon_exploration > 0.0) {
            if (manager->get_rand_uniform() < manager->epsilon_exploration) {
                return select_action_random();
            }
        }
        return select_action_ucb(ctx);
    }

    /**
     * Recommends the action corresponding to the best child node avg_return. Breaks ties randomly.
     * 
     * If acting as the opponent, we recommend the minimum value by multiplying by -1.0.
     */
    shared_ptr<const Action> UctDNode::recommend_action_best_empirical() const {
        double opp_coeff = is_opponent() ? -1.0 : 1.0;
        unordered_map<shared_ptr<const Action>, double> action_values;

        for (shared_ptr<const Action> action : *actions) {
            if (!has_child_node(action)) continue;
            action_values[action] = opp_coeff * get_child_node(action)->avg_return;
        }

        // If no children, best we can do is select a random action to recommend
        if (action_values.size() == 0u) {
            int index = mcts_manager->get_rand_int(0, actions->size());
            return actions->at(index);
        }

        return helper::get_max_key_break_ties_randomly(action_values, *mcts_manager);
    }

    /**
     * Recommends the action corresponding to the child most visited. Breaks ties randomly.
     */
    shared_ptr<const Action> UctDNode::recommend_action_most_visited() const {
        unordered_map<shared_ptr<const Action>, int> visit_counts;

        for (shared_ptr<const Action> action : *actions) {
            if (!has_child_node(action)) continue;
            visit_counts[action] = get_child_node(action)->num_visits;
        }

        // If no children, best we can do is select a random action to recommend
        if (visit_counts.size() == 0u) {
            int index = mcts_manager->get_rand_int(0, actions->size());
            return actions->at(index);
        }

        return helper::get_max_key_break_ties_randomly(visit_counts, *mcts_manager);
    }

    /**
     * Recommend action function.
     * 
     * Recommends an action based on the options provided by UctManager.
     */
    shared_ptr<const Action> UctDNode::recommend_action(MctsEnvContext& ctx) const {
        UctManager& manager = (UctManager&) *mcts_manager;
        if (manager.recommend_most_visited) {
            return recommend_action_most_visited();
        }
        return recommend_action_best_empirical();
    }

    /**
     * Computes running average.
     */
    void UctDNode::backup_average_return(const double trial_return_after_node) {
        num_backups++;
        avg_return += (trial_return_after_node - avg_return) / (double) num_backups;
    }

    /**
     * Calls the running average return backup function.
     */
    void UctDNode::backup(
        const vector<double>& trial_rewards_before_node, 
        const vector<double>& trial_rewards_after_node, 
        const double trial_cumulative_return_after_node, 
        const double trial_cumulative_return,
        MctsEnvContext& ctx) 
    {
        backup_average_return(trial_cumulative_return_after_node);
    }
    
    /**
     * Make a child
     */
    shared_ptr<UctCNode> UctDNode::create_child_node_helper(shared_ptr<const Action> action) const {
        return make_shared<UctCNode>(
            static_pointer_cast<UctManager>(mcts_manager), 
            state, 
            action, 
            decision_depth, 
            decision_timestep, 
            static_pointer_cast<const UctDNode>(shared_from_this()));
    }

    /**
     * Pretty print val = print current avg_return in node
     */
    string UctDNode::get_pretty_print_val() const {
        stringstream ss;
        ss << avg_return;
        return ss.str();
    }
}

/**
 * Boilerplate function definitions.
 * All this code basically calls the corresponding base implementation function, with approprtiate casts before/after.
 */
namespace mcts {
    shared_ptr<UctCNode> UctDNode::create_child_node(shared_ptr<const Action> action) {
        shared_ptr<MctsCNode> new_child = MctsDNode::create_child_node_itfc(action);
        return static_pointer_cast<UctCNode>(new_child);
    }

    bool UctDNode::has_child_node(shared_ptr<const Action> action) const {
        return MctsDNode::has_child_node_itfc(static_pointer_cast<const Action>(action));
    }
    shared_ptr<UctCNode> UctDNode::get_child_node(shared_ptr<const Action> action) const {
        shared_ptr<const Action> act_itfc = static_pointer_cast<const Action>(action);
        shared_ptr<MctsCNode> new_child = MctsDNode::get_child_node_itfc(act_itfc);
        return static_pointer_cast<UctCNode>(new_child);
    }
}

/**
 * Boilerplate MctsDNode interface implementation. Copied from mcts_decision_node_template.h.
 */
namespace mcts {
    void UctDNode::visit_itfc(MctsEnvContext& ctx) {
        visit(ctx);
    }

    shared_ptr<const Action> UctDNode::select_action_itfc(MctsEnvContext& ctx) {
        return select_action(ctx);
    }

    shared_ptr<const Action> UctDNode::recommend_action_itfc(MctsEnvContext& ctx) const {
        return recommend_action(ctx);
    }

    void UctDNode::backup_itfc(
        const vector<double>& trial_rewards_before_node, 
        const vector<double>& trial_rewards_after_node, 
        const double trial_cumulative_return_after_node, 
        const double trial_cumulative_return,
        MctsEnvContext& ctx) 
    {
        backup(
            trial_rewards_before_node, 
            trial_rewards_after_node, 
            trial_cumulative_return_after_node, 
            trial_cumulative_return, 
            ctx);
    }

    shared_ptr<MctsCNode> UctDNode::create_child_node_helper_itfc(shared_ptr<const Action> action) const {
        shared_ptr<UctCNode> child_node = create_child_node_helper(action);
        return static_pointer_cast<MctsCNode>(child_node);
    }
}
