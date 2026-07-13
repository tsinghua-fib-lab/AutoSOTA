#include "algorithms/est/est_decision_node.h"

using namespace std; 

namespace mcts {
    /**
     * Constructor
    */
    EstDNode::EstDNode(
        shared_ptr<DentsManager> mcts_manager,
        shared_ptr<const State> state,
        int decision_depth,
        int decision_timestep,
        shared_ptr<const EstCNode> parent) :
            DentsDNode(
                mcts_manager,
                state,
                decision_depth,
                decision_timestep,
                static_pointer_cast<const DentsCNode>(parent))
    {
    }

    /**
     * Gets the q_value to use for a child, calls ments version when there is not a child node
     */
    double EstDNode::get_soft_q_value(std::shared_ptr<const Action> action, double opp_coeff) const {
        if (has_child_node(action)) {
            EstCNode& child = (EstCNode&) *get_child_node(action);
            DentsManager& manager = (DentsManager&) *mcts_manager;
            double val_estimate = manager.use_dp_value ? child.dp_value : child.avg_return;
            return val_estimate * opp_coeff;
        } 

        return MentsDNode::get_soft_q_value(action, opp_coeff);
    }

    /**
     * Calls both the ments soft backup and dp backup
     * 
     * Recall that the dp backup needs to be passed the type of the child nodes (so can keep dp logic in dp node)
     */
    void EstDNode::backup(
        const vector<double>& trial_rewards_before_node, 
        const vector<double>& trial_rewards_after_node, 
        const double trial_cumulative_return_after_node, 
        const double trial_cumulative_return,
        MctsEnvContext& ctx) 
    {
        MentsDNode::num_backups++;

        // value backup
        DentsManager& manager = (DentsManager&) *mcts_manager;
        if (manager.use_dp_value) {
            backup_dp<DentsCNode>(children, is_opponent());
        } else {
            backup_emp(trial_cumulative_return_after_node);
        }
    }

    /**
     * Return string of the soft value
     */
    string EstDNode::get_pretty_print_val() const {
        DentsManager& manager = (DentsManager&) *mcts_manager;
        stringstream ss;
        ss << (manager.use_dp_value ? dp_value : avg_return);
        return ss.str();
    }

    /**
     * Make child node
     */
    shared_ptr<EstCNode> EstDNode::create_child_node_helper(shared_ptr<const Action> action) const {
        return make_shared<EstCNode>(
            static_pointer_cast<DentsManager>(MctsDNode::mcts_manager), 
            state, 
            action, 
            decision_depth, 
            decision_timestep, 
            static_pointer_cast<const EstDNode>(shared_from_this()));
    }
}

/**
 * Boilerplate MctsDNode interface implementation. Copied from mcts_decision_node_template.h.
 */
namespace mcts {
    shared_ptr<MctsCNode> EstDNode::create_child_node_helper_itfc(shared_ptr<const Action> action) const {
        shared_ptr<const Action> act_itfc = static_pointer_cast<const Action>(action);
        shared_ptr<EstCNode> child_node = create_child_node_helper(act_itfc);
        return static_pointer_cast<MctsCNode>(child_node);
    }
}