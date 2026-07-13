#include "algorithms/ments/ments_decision_node.h"

#include "helper_templates.h"

#include <cmath>
#include <limits>


#include <iostream>
using namespace std; 
    
// Epsilon to be used as a minimum prob, if lower than this just set to zero
static double EPS = 1e-16;
static double LOG_MIN_ARG = 1e-32;
static double LOG_MAX_ARG = 1e32;
static double MIN_LOG_WEIGHT = -32.0;
static double MAX_LOG_WEIGHT = 32.0;

namespace mcts {
    MentsDNode::MentsDNode(
        shared_ptr<MentsManager> mcts_manager,
        shared_ptr<const State> state,
        int decision_depth,
        int decision_timestep,
        shared_ptr<const MentsCNode> parent) :
            MctsDNode(
                static_pointer_cast<MctsManager>(mcts_manager),
                state,
                decision_depth,
                decision_timestep,
                static_pointer_cast<const MctsCNode>(parent)),
            num_backups(0),
            soft_value(0.0),
            actions(mcts_manager->mcts_env->get_valid_actions_itfc(state)),
            policy_prior(),
            psuedo_q_value_offset(0.0)
    {
        if (mcts_manager->heuristic_fn != nullptr) {
            soft_value = heuristic_value;
        }

        if (mcts_manager->prior_fn != nullptr) {
            policy_prior = mcts_manager->prior_fn(state, mcts_manager->mcts_env);

            if (mcts_manager->shift_pseudo_q_values) {
                double mean_log_weight = 0.0;
                double i = 1.0;
                for (pair<shared_ptr<const Action>,double> pr : *policy_prior) {
                    double weight = pr.second;
                    double log_weight = MIN_LOG_WEIGHT;
                    if (weight >= LOG_MAX_ARG) {
                        log_weight = MAX_LOG_WEIGHT;
                    } else if (weight > LOG_MIN_ARG) {
                        log_weight = log(weight);
                    }
                    mean_log_weight *= (i-1.0) / i;
                    mean_log_weight += log_weight / i;
                }
                psuedo_q_value_offset = mcts_manager->psuedo_q_value_offset - mean_log_weight;
            }
        }
    }
    
    /**
     * Helper function for checking if we have a prior or not. 
     */
    bool MentsDNode::has_prior() const {
        MentsManager& manager = (MentsManager&) *mcts_manager;
        return manager.prior_fn != nullptr;
    }

    /**
     * Get the temperature to use
     */
    double MentsDNode::get_temp() const {
        MentsManager& manager = (MentsManager&) *mcts_manager;
        if (manager.temp_decay_fn == nullptr) return manager.temp;

        double visits_scale = manager.temp_decay_visits_scale;
        if (is_root_node() && manager.temp_decay_root_node_visits_scale > 0.0) {
            visits_scale = manager.temp_decay_root_node_visits_scale;
        }
        return compute_decayed_temp(
            manager.temp_decay_fn, manager.temp, manager.temp_decay_min_temp, num_visits, visits_scale);

    }
    
    /**
     * Default visit is typically fine.
     * 
     * The main nuance is making sure that 'num_backups' is updated for leaf nodes, where Mcts will only ever call 
     * visit on these nodes. If this node is a leaf, then a backup should essentially be a no-op. However, for the 
     * soft_backup in chance nodes to work, the number of backups needs to be updated, even at leaf nodes.
     */
    void MentsDNode::visit(MctsEnvContext& ctx) {
        MctsDNode::visit_itfc(ctx);
        if (is_leaf()) {
            num_backups++;
        }
    }

    /**
     * Gets the q_value to use for a child
     * 
     * Handle numerical instability for the prior_prob=0 case
     * log
     */
    double MentsDNode::get_soft_q_value(std::shared_ptr<const Action> action, double opp_coeff) const {
        if (has_child_node(action)) {
            MentsCNode& child = (MentsCNode&) *get_child_node(action);
            return child.soft_value * opp_coeff;
        } 

        if (has_prior()) {
            double weight = policy_prior->at(action);
            double log_weight = MIN_LOG_WEIGHT;
            if (weight >= LOG_MAX_ARG) {
                log_weight = MAX_LOG_WEIGHT;
            } else if (weight > LOG_MIN_ARG) {
                log_weight = log(weight);
            }
            return log_weight + psuedo_q_value_offset;
        } 

        MentsManager& manager = (MentsManager&) *mcts_manager;
        return manager.default_q_value * opp_coeff;
    }

    /**
     * Compute action weights.
     * 
     * Performs the following:
     *  - gets the effective temperature (that includes a coefficient for if this node is acting as an opponent)
     *  - computes the normalisation term (see below)
     *  - computes the action weights
     * 
     * The normalisation term is needed for numerical stability. Even for relatively small values of x, exp(x) can 
     * result in overflow. So to prevent both overflow and underflow, the maximum weight is normalised to a value of 
     * 1. 
     * 
     * Note that because this normalisation is equivalent to multiplying all terms, this normalisation still results in 
     * a distribution that has the same shape (exp(y_i) == (exp(y_i-max_y)exp(max_y)) in maths terms). Thus if we just 
     * want to use the distribution, we can ignore the normalisation term.
     * 
     * The normalisation term is still returned via reference for when we do need to know the exact values of these 
     * weights (which is the case in backups).
     * 
     */
    void MentsDNode::compute_action_weights(
        ActionDistr& action_weights, 
        double& sum_action_weights, 
        double& normalisation_term, 
        MctsEnvContext& context) const
    {
        // get temp
        double opp_coeff = is_opponent() ? -1.0 : 1.0;
        double temp = get_temp();

        // compute normalisation term
        normalisation_term = numeric_limits<double>::lowest();
        for (shared_ptr<const Action> action : *actions) {
            double q_value_over_temp = get_soft_q_value(action,opp_coeff) / temp;
            if (normalisation_term < q_value_over_temp) {
                normalisation_term = q_value_over_temp;
            }
        }

        // compute action weights
        sum_action_weights = 0.0;
        for (shared_ptr<const Action> action : *actions) {
            double soft_q_value = get_soft_q_value(action,opp_coeff);
            double action_weight = exp((soft_q_value/temp) - normalisation_term);
            action_weights[action] = action_weight;
            sum_action_weights += action_weight;
        }
    }

    /**
     * Computes action distribution.
     * 
     * Lambda is the probability that we uniformly sample an action, rather than using the boltzmann action weights.
     * In this implementation it is equal to epsilon / log(num_visits + 1)
     * 
     * The first call computes a (unormalised) boltzmann distribution over the actions, using child soft values. This 
     * boltzmann distribution is then normalised and interpolated with a uniform distribution, according to the value 
     * of lambda above.
     * 
     * Additionally, protects the computation of action weights by grabbing the children's locks.
     */
    void MentsDNode::compute_action_distribution(
        ActionDistr& action_distr, 
        MctsEnvContext& context) const 
    {  
        // compute boltzmann weights
        double sum_weights;
        double _normalisation_term;
        lock_all_children();
        compute_action_weights(action_distr, sum_weights, _normalisation_term, context);
        unlock_all_children();

        // compute lambda
        MentsManager& manager = (MentsManager&) *mcts_manager;
        double epsilon = manager.epsilon;
        if (is_root_node() && manager.root_node_epsilon > 0.0) epsilon = manager.root_node_epsilon;
        double lambda = epsilon / log(num_visits+1);
        if (lambda > manager.max_explore_prob) {
            lambda = manager.max_explore_prob;
        }

        // normalise and interpolate masses with uniform masses
        double num_actions = actions->size();
        double uniform_distr_mass = 1.0 / num_actions;
        vector<shared_ptr<const Action>> near_zero_prob_actions;
        for (shared_ptr<const Action> action : *actions) {
            action_distr[action] *= (1.0 - lambda) / sum_weights;
            if (manager.prior_policy_search_weight > 0.0) {
                double lambda_tilde = manager.prior_policy_search_weight / log(num_visits+3);
                action_distr[action] *= (1.0 - lambda_tilde);
                action_distr[action] += (1.0 - lambda) * lambda_tilde * policy_prior->at(action);
            }
            action_distr[action] += lambda * uniform_distr_mass;
            if (action_distr[action] < EPS) {
                near_zero_prob_actions.push_back(action);
            }
        }

        // Remove close to zero probabilities (never going to sample + leads to numerical ick later)
        for (shared_ptr<const Action> action : near_zero_prob_actions) {
            action_distr.erase(action);
        }
    }

    /**
     * Implements selct action for ments
     * 
     * - Computes the action distribution.
     * - Samples an action
     * - Creates the node if it doesn't exist already
     */
    shared_ptr<const Action> MentsDNode::select_action_ments(MctsEnvContext& ctx) {
        ActionDistr action_distr;
        compute_action_distribution(action_distr, ctx);
        shared_ptr<const Action> selected_action = helper::sample_from_distribution(action_distr, *mcts_manager);
        if (!has_child_node(selected_action)) {
            create_child_node(selected_action);
        }
        return selected_action;
    }

    /**
     * Calls the ments implementation of select action
     */
    shared_ptr<const Action> MentsDNode::select_action(MctsEnvContext& ctx) {
        return select_action_ments(ctx);
    }

    /**
     * Builds a map of actions to q-values for actions that do and do not meet the 'recommend_visit_threshold'. 
     * And then recommends the max from the thresholded map, and from the unthresholded map if the thresholded one is 
     * empty.
     */
    shared_ptr<const Action> MentsDNode::recommend_action_best_soft_value() const {
        MentsManager& manager = *static_pointer_cast<MentsManager>(mcts_manager);
        double opp_coeff = is_opponent() ? -1.0 : 1.0;
        unordered_map<shared_ptr<const Action>, double> soft_values_thresholded;
        unordered_map<shared_ptr<const Action>, double> soft_values;

        for (shared_ptr<const Action> action : *actions) {
            double q_value = get_soft_q_value(action, opp_coeff);
            if (has_child_node(action) && get_child_node(action)->num_visits >= manager.recommend_visit_threshold) {
                soft_values_thresholded[action] = q_value;
            } else {
                soft_values[action] = q_value;
            }
        }

        if (soft_values_thresholded.size() > 0) {
            return helper::get_max_key_break_ties_randomly(soft_values_thresholded, manager);
        }
        return helper::get_max_key_break_ties_randomly(soft_values, manager);
    }

    /**
     * Recommends the action corresponding to the child most visited. Breaks ties randomly.
     */
    shared_ptr<const Action> MentsDNode::recommend_action_most_visited() const {
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
     * Recommends an action based on the options provided by MentsManager.
     */
    shared_ptr<const Action> MentsDNode::recommend_action(MctsEnvContext& ctx) const {
        MentsManager& manager = (MentsManager&) *mcts_manager;
        if (manager.recommend_most_visited) {
            return recommend_action_most_visited();
        }
        return recommend_action_best_soft_value();
    }
    
    /**
     * Implements a soft backup for ments.
     * 
     * I.e. perform V(s) = temp * log(sum(exp(Q(s,a)/temp))). Noting that the sum term is directly returned by 
     * compute_action_weights
     * 
     * In two player games, we maintain an invariant that values are always computed with respect to the main player. 
     * So for an opponent, the weights used the negative of child values, and we need to 'undo' that. (Which is why 
     * we have the additional opp_coeff term).
     * 
     * Additionally, we need to add the normalisation term from 'compute_action_weights' at the end. 
     * 
     * Also don't forget to increment num_backups
     */
    void MentsDNode::backup_soft(MctsEnvContext& ctx) {
        num_backups++;

        ActionDistr action_weights;
        double sum_weights;
        double normalisation_term;
        lock_all_children();
        compute_action_weights(action_weights, sum_weights, normalisation_term, ctx);
        unlock_all_children();

        double opp_coeff = is_opponent() ? -1.0 : 1.0;
        double temp = get_temp();
        soft_value = opp_coeff * temp * (log(sum_weights) + normalisation_term);
    }

    /**
     * Calls the ments implementation of backup, performing soft backup
     */
    void MentsDNode::backup(
        const vector<double>& trial_rewards_before_node, 
        const vector<double>& trial_rewards_after_node, 
        const double trial_cumulative_return_after_node, 
        const double trial_cumulative_return,
        MctsEnvContext& ctx) 
    {
        backup_soft(ctx);
    }

    /**
     * Make child node
     */
    shared_ptr<MentsCNode> MentsDNode::create_child_node_helper(shared_ptr<const Action> action) const {
        return make_shared<MentsCNode>(
            static_pointer_cast<MentsManager>(mcts_manager), 
            state, 
            action, 
            decision_depth, 
            decision_timestep, 
            static_pointer_cast<const MentsDNode>(shared_from_this()));
    }

    /**
     * Return string of the soft value
     */
    string MentsDNode::get_pretty_print_val() const {
        stringstream ss;
        ss << soft_value << "(temp:" << get_temp() << ")";
        return ss.str();
    }
}



/**
 * Boilerplate function definitions.
 * All this code basically calls the corresponding base implementation function, with approprtiate casts before/after.
 */
namespace mcts {
    shared_ptr<MentsCNode> MentsDNode::create_child_node(shared_ptr<const Action> action) {
        shared_ptr<const Action> act_itfc = static_pointer_cast<const Action>(action);
        shared_ptr<MctsCNode> new_child = MctsDNode::create_child_node_itfc(act_itfc);
        return static_pointer_cast<MentsCNode>(new_child);
    }

    bool MentsDNode::has_child_node(shared_ptr<const Action> action) const {
        return MctsDNode::has_child_node_itfc(static_pointer_cast<const Action>(action));
    }

    shared_ptr<MentsCNode> MentsDNode::get_child_node(shared_ptr<const Action> action) const {
        shared_ptr<const Action> act_itfc = static_pointer_cast<const Action>(action);
        shared_ptr<MctsCNode> new_child = MctsDNode::get_child_node_itfc(act_itfc);
        return static_pointer_cast<MentsCNode>(new_child);
    }
}



/**
 * Boilerplate MctsDNode interface implementation. Copied from mcts_decision_node_template.h.
 */
namespace mcts {
    void MentsDNode::visit_itfc(MctsEnvContext& ctx) {
        MctsEnvContext& ctx_itfc = (MctsEnvContext&) ctx;
        visit(ctx_itfc);
    }

    shared_ptr<const Action> MentsDNode::select_action_itfc(MctsEnvContext& ctx) {
        MctsEnvContext& ctx_itfc = (MctsEnvContext&) ctx;
        shared_ptr<const Action> action = select_action(ctx_itfc);
        return static_pointer_cast<const Action>(action);
    }

    shared_ptr<const Action> MentsDNode::recommend_action_itfc(MctsEnvContext& ctx) const {
        MctsEnvContext& ctx_itfc = (MctsEnvContext&) ctx;
        shared_ptr<const Action> action = recommend_action(ctx_itfc);
        return static_pointer_cast<const Action>(action);
    }

    void MentsDNode::backup_itfc(
        const vector<double>& trial_rewards_before_node, 
        const vector<double>& trial_rewards_after_node, 
        const double trial_cumulative_return_after_node, 
        const double trial_cumulative_return,
        MctsEnvContext& ctx) 
    {
        MctsEnvContext& ctx_itfc = (MctsEnvContext&) ctx;
        backup(
            trial_rewards_before_node, 
            trial_rewards_after_node, 
            trial_cumulative_return_after_node, 
            trial_cumulative_return, 
            ctx_itfc);
    }

    shared_ptr<MctsCNode> MentsDNode::create_child_node_helper_itfc(shared_ptr<const Action> action) const {
        shared_ptr<const Action> act_itfc = static_pointer_cast<const Action>(action);
        shared_ptr<MentsCNode> child_node = create_child_node_helper(act_itfc);
        return static_pointer_cast<MctsCNode>(child_node);
    }
}
