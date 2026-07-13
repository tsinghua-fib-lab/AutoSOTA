#include "algorithms/est/est_chance_node.h"

using namespace std;

namespace mcts {
    /**
     * Constructor
     */
    EstCNode::EstCNode(
        shared_ptr<DentsManager> mcts_manager,
        shared_ptr<const State> state,
        shared_ptr<const Action> action,
        int decision_depth,
        int decision_timestep,
        shared_ptr<const EstDNode> parent) :
            DentsCNode(
                mcts_manager,
                state,
                action,
                decision_depth,
                decision_timestep,
                static_pointer_cast<const DentsDNode>(parent))
    {
    }

    /**
     * Calls ments soft backup and dp backup
     * 
     * Recall that the dp backup needs to be passed the type of the child nodes (so can keep dp logic in dp node)
     */
    void EstCNode::backup(
        const std::vector<double>& trial_rewards_before_node, 
        const std::vector<double>& trial_rewards_after_node, 
        const double trial_cumulative_return_after_node, 
        const double trial_cumulative_return,
        MctsEnvContext& ctx)
    {   
        MentsCNode::num_backups++;

        // value backup
        DentsManager& manager = (DentsManager&) *mcts_manager;
        if (manager.use_dp_value) {
            backup_dp<EstDNode>(children, local_reward, is_opponent());
        } else {
            backup_emp(trial_cumulative_return_after_node);
        }
    }

    /**
     * Make child node
     */
    shared_ptr<EstDNode> EstCNode::create_child_node_helper(
        shared_ptr<const State> observation, shared_ptr<const State> next_state) const 
    {  
        shared_ptr<const State> mdp_next_state = static_pointer_cast<const State>(observation);
        return make_shared<EstDNode>(
            static_pointer_cast<DentsManager>(MctsCNode::mcts_manager), 
            mdp_next_state,
            decision_depth+1, 
            decision_timestep+1, 
            static_pointer_cast<const EstCNode>(shared_from_this()));
    }

    /**
     * Return string of the soft value
     */
    string EstCNode::get_pretty_print_val() const {
        DentsManager& manager = (DentsManager&) *mcts_manager;
        stringstream ss;
        ss << (manager.use_dp_value ? dp_value : avg_return);
        return ss.str();
    }
}

/**
 * Boilerplate MctsCNode interface implementation. Copied from mcts_chance_node_template.h.
 */
namespace mcts {
    shared_ptr<MctsDNode> EstCNode::create_child_node_helper_itfc(
        shared_ptr<const Observation> observation, shared_ptr<const State> next_state) const 
    {
        shared_ptr<const State> obsv_itfc = static_pointer_cast<const State>(observation);
        shared_ptr<const State> next_state_itfc = static_pointer_cast<const State>(next_state);
        shared_ptr<EstDNode> child_node = create_child_node_helper(obsv_itfc, next_state_itfc);
        return static_pointer_cast<MctsDNode>(child_node);
    }
}