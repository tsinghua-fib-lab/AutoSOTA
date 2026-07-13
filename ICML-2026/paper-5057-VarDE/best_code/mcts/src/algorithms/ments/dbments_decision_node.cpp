#include "algorithms/ments/dbments_decision_node.h"

using namespace std; 

namespace mcts {
    DBMentsDNode::DBMentsDNode(
        shared_ptr<MentsManager> mcts_manager,
        shared_ptr<const State> state,
        int decision_depth,
        int decision_timestep,
        shared_ptr<const DBMentsCNode> parent) :
            MentsDNode(
                mcts_manager,
                state,
                decision_depth,
                decision_timestep,
                static_pointer_cast<const MentsCNode>(parent)),
            DPDNode(heuristic_value)
    {
    }

    /**
     * Call both ments and dp visit functions
     */
    void DBMentsDNode::visit(MctsEnvContext& ctx) {
        MentsDNode::visit(ctx);
        DPDNode::visit_dp(is_leaf());
    }

    /**
     * Calls the dpdnode implementation of recommend action
     */
    shared_ptr<const Action> DBMentsDNode::recommend_action_best_dp_value() const {
        if (children.size() == 0u) {
            int index = mcts_manager->get_rand_int(0, actions->size());
            return actions->at(index);
        }

        MentsManager& manager = (MentsManager&) *MctsDNode::mcts_manager;
        return DPDNode::recommend_action_best_dp_value<DBMentsCNode>(
            children, *mcts_manager, manager.recommend_visit_threshold, is_opponent());
    }

    /**
     * Implements recommend action to call best dp value
    */
    shared_ptr<const Action> DBMentsDNode::recommend_action(MctsEnvContext& ctx) const {
        MentsManager& manager = (MentsManager&) *mcts_manager;
        if (manager.recommend_most_visited) {
            return recommend_action_most_visited();
        }
        return recommend_action_best_dp_value();
    }

    /**
     * Calls both the ments soft backup and dp backup
     * 
     * Recall that the dp backup needs to be passed the type of the child nodes (so can keep dp logic in dp node)
     */
    void DBMentsDNode::backup(
        const vector<double>& trial_rewards_before_node, 
        const vector<double>& trial_rewards_after_node, 
        const double trial_cumulative_return_after_node, 
        const double trial_cumulative_return,
        MctsEnvContext& ctx) 
    {
        backup_soft(ctx);
        backup_dp<DBMentsCNode>(children, is_opponent());
    }

    /**
     * Return string of the soft value
     */
    string DBMentsDNode::get_pretty_print_val() const {
        stringstream ss;
        ss << dp_value << "(temp:" << get_temp() << ",soft_val:" << soft_value << ")";
        return ss.str();
    }

    /**
     * Make child node
     */
    shared_ptr<DBMentsCNode> DBMentsDNode::create_child_node_helper(shared_ptr<const Action> action) const {
        return make_shared<DBMentsCNode>(
            static_pointer_cast<MentsManager>(MctsDNode::mcts_manager), 
            state, 
            action, 
            decision_depth, 
            decision_timestep, 
            static_pointer_cast<const DBMentsDNode>(shared_from_this()));
    }
}

/**
 * Boilerplate MctsDNode interface implementation. Copied from mcts_decision_node_template.h.
 */
namespace mcts {
    shared_ptr<MctsCNode> DBMentsDNode::create_child_node_helper_itfc(shared_ptr<const Action> action) const {
        shared_ptr<const Action> act_itfc = static_pointer_cast<const Action>(action);
        shared_ptr<DBMentsCNode> child_node = create_child_node_helper(act_itfc);
        return static_pointer_cast<MctsCNode>(child_node);
    }
}