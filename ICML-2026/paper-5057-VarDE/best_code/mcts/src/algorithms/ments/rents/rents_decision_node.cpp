#include "algorithms/ments/rents/rents_decision_node.h"

#include "helper_templates.h"

#include <cmath>
#include <limits>
#include <sstream>

using namespace std; 

static double EPS = 1e-16;

namespace mcts {
    RentsDNode::RentsDNode(
        shared_ptr<MentsManager> mcts_manager,
        shared_ptr<const State> state,
        int decision_depth,
        int decision_timestep,
        shared_ptr<const RentsCNode> parent) :
            MentsDNode(
                mcts_manager,
                state,
                decision_depth,
                decision_timestep,
                static_pointer_cast<const MentsCNode>(parent))
    {
        stringstream ss_n;
        ss_n << "d_" << decision_depth;
        _node_distr_key = ss_n.str();

        if (decision_depth > 0) {
            stringstream ss_p;
            ss_p << "d_" << decision_depth-1;
            _parent_distr_key = ss_p.str();
        }
    }

    /**
     * Gets the action distribution for a parent node 
     * Or just null pointer if we're the root node
    */
    shared_ptr<ActionDistr> RentsDNode::get_parent_distr_from_context(MctsEnvContext& ctx) const {
        if (decision_depth < 1) return nullptr;
        return ctx.get_value_ptr<ActionDistr>(_parent_distr_key);
    }

    /**
     * Puts the action distribution for this node into the Mcts env context
    */
    void RentsDNode::put_node_distr_in_context(shared_ptr<ActionDistr> action_distr, MctsEnvContext& ctx) const {
        ctx.put_value(_node_distr_key, action_distr);
    }

    /**
    * Get prob from parent distribution (handling boundary cases at the root node and when parent didn't have the 
    * action passed in as an option).
    * 
    * Just returns the value stored in the distribution.
    * If the action is not in the distribution return 0.0
    * If the distribution is nullptr (then we are root node) and return 1.0 so can compute normal ments distr.
    */
    double RentsDNode::get_parent_action_prob(
        shared_ptr<ActionDistr> parent_distr, shared_ptr<const Action> action) const 
    {
        if (parent_distr == nullptr) return 1.0;
        if (parent_distr->find(action) == parent_distr->end()) return 0.0;
        return parent_distr->at(action);
    }

    /**
     * Compute action weights.
     * 
     * Computes the distribution from the paper 
     * Paper: http://proceedings.mlr.press/v139/dam21a/dam21a.pdf
     * 
     * Multiplies weights from parent decision node into weights for actions at this node.
     */
    void RentsDNode::compute_action_weights(
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

        // Get parent distribution
        shared_ptr<ActionDistr> parent_distr = get_parent_distr_from_context(context);

        // compute action weights
        sum_action_weights = 0.0;
        for (shared_ptr<const Action> action : *actions) {
            double soft_q_value = get_soft_q_value(action,opp_coeff);
            double action_weight = exp((soft_q_value/temp) - normalisation_term);
            action_weight *= get_parent_action_prob(parent_distr, action);
            action_weights[action] = action_weight;
            sum_action_weights += action_weight;
        }
        
        // If all action weights extremely small, then just make it uniform random, for numerical stability
        if (sum_action_weights < EPS) {
            double uniform_weight = 1.0 / actions->size();
            for (shared_ptr<const Action> action : *actions) {
                action_weights[action] = uniform_weight;
            }
            sum_action_weights = 1.0;
        }
    }

    /**
     * Implements selct action for rents
     * 
     * - Computes the action distribution.
     * - Stores the distribution in the context
     * - Samples an action
     * - Creates the node if it doesn't exist already
     */
    shared_ptr<const Action> RentsDNode::select_action_rents(MctsEnvContext& ctx) {
        shared_ptr<ActionDistr> action_distr = make_shared<ActionDistr>();
        compute_action_distribution(*action_distr, ctx);
        put_node_distr_in_context(action_distr, ctx);
        shared_ptr<const Action> selected_action = helper::sample_from_distribution(*action_distr, *mcts_manager);
        if (!has_child_node(selected_action)) {
            create_child_node(selected_action);
        }
        return selected_action;
    }

    /**
     * Calls the rents implementation of select action
     */
    shared_ptr<const Action> RentsDNode::select_action(MctsEnvContext& ctx) {
        return select_action_rents(ctx);
    }

    /**
     * Make child node
     */
    shared_ptr<RentsCNode> RentsDNode::create_child_node_helper(shared_ptr<const Action> action) const {
        return make_shared<RentsCNode>(
            static_pointer_cast<MentsManager>(mcts_manager), 
            state, 
            action, 
            decision_depth, 
            decision_timestep, 
            static_pointer_cast<const RentsDNode>(shared_from_this()));
    }
}



/**
 * Boilerplate MctsDNode interface implementation. Copied from mcts_decision_node_template.h.
 */
namespace mcts {
    shared_ptr<MctsCNode> RentsDNode::create_child_node_helper_itfc(shared_ptr<const Action> action) const {
        shared_ptr<const Action> act_itfc = static_pointer_cast<const Action>(action);
        shared_ptr<RentsCNode> child_node = create_child_node_helper(act_itfc);
        return static_pointer_cast<MctsCNode>(child_node);
    }
}