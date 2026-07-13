#include "algorithms/ments/dents/dents_decision_node.h"

#include "algorithms/common/decaying_temp.h"

using namespace std; 

namespace mcts {
    DentsDNode::DentsDNode(
        shared_ptr<DentsManager> mcts_manager,
        shared_ptr<const State> state,
        int decision_depth,
        int decision_timestep,
        shared_ptr<const DentsCNode> parent) :
            DBMentsDNode(
                static_pointer_cast<MentsManager>(mcts_manager),
                state,
                decision_depth,
                decision_timestep,
                static_pointer_cast<const DBMentsCNode>(parent)),
            EntDNode(),
            EmpNode(1, heuristic_value)
    {
        // Init entropy
        ActionDistr action_distr;
        MctsEnvContext ctx;
        compute_action_distribution(action_distr, ctx);
        backup_ent<DentsCNode>(children, action_distr, is_opponent());
    }

    /**
     * Get decayed temp
     */
    double DentsDNode::get_value_temp() const {
        DentsManager& manager = (DentsManager&) *mcts_manager;
        if (manager.value_temp_decay_fn == nullptr) return manager.value_temp_init;

        double visits_scale = manager.value_temp_decay_visits_scale;
        if (is_root_node() && manager.value_temp_decay_root_node_visits_scale > 0.0) {
            visits_scale = manager.value_temp_decay_root_node_visits_scale;
        }
        return compute_decayed_temp(
            manager.value_temp_decay_fn, 
            manager.value_temp_init, 
            manager.value_temp_decay_min_temp, 
            num_visits, 
            visits_scale);
    }

    /**
     * Gets the soft q value of a child node (as considered by this current node).
     * 
     * These values are of the form V + temp_decayed * H. Note that the 'temp_decayed' used is the decayed temperature 
     * for *this* node, not the child node, and the soft value returned != child.soft_value.
     * 
     * Other cases are suitably handled by the implementation in MentsDNode, so just call that
    */
    double DentsDNode::get_soft_q_value(std::shared_ptr<const Action> action, double opp_coeff) const {
        if (!has_child_node(action)) {
            return MentsDNode::get_soft_q_value(action, opp_coeff);
        }

        DentsManager& manager = (DentsManager&) *mcts_manager;
        DentsCNode& child = (DentsCNode&) *get_child_node(action);
        double val_estimate = child.dp_value;
        if (!manager.use_dp_value) val_estimate = child.avg_return;
        return opp_coeff * (val_estimate + get_value_temp() * child.subtree_entropy);
    }

    /**
     * Calls the empnode implementation of recommend action
     */
    shared_ptr<const Action> DentsDNode::recommend_action_best_empirical_value() const {
        if (children.size() == 0u) {
            int index = mcts_manager->get_rand_int(0, actions->size());
            return actions->at(index);
        }

        DentsManager& manager = (DentsManager&) *MctsDNode::mcts_manager;
        return EmpNode::recommend_action_best_emp_value<DentsCNode>(
            children, *mcts_manager, manager.recommend_visit_threshold, is_opponent());
    }

    /**
     * Implements recommend action to call best dp value
    */
    shared_ptr<const Action> DentsDNode::recommend_action(MctsEnvContext& ctx) const {
        DentsManager& manager = (DentsManager&) *mcts_manager;
        if (manager.recommend_most_visited) {
            return recommend_action_most_visited();
        }
        if (manager.use_dp_value) {
            return recommend_action_best_dp_value();
        }
        return recommend_action_best_empirical_value();
    }

    /**
     * Calls both the entropy backup and dp backup
     * 
     * Recall that the dp backup needs to be passed the type of the child nodes (so can keep dp logic in dp node)
     */
    void DentsDNode::backup(
        const vector<double>& trial_rewards_before_node, 
        const vector<double>& trial_rewards_after_node, 
        const double trial_cumulative_return_after_node, 
        const double trial_cumulative_return,
        MctsEnvContext& ctx) 
    {
        MentsDNode::num_backups++;

        // entropy backup
        ActionDistr action_distr;
        compute_action_distribution(action_distr, ctx);
        backup_ent<DentsCNode>(children, action_distr, is_opponent());

        // value backup
        double val_estimate;
        DentsManager& manager = (DentsManager&) *mcts_manager;
        if (manager.use_dp_value) {
            backup_dp<DentsCNode>(children, is_opponent());
            val_estimate = dp_value;
        } else {
            backup_emp(trial_cumulative_return_after_node);
            val_estimate = avg_return;
        }
    
        // update local soft_value so that value is sensible / for pretty printing
        soft_value = val_estimate + get_value_temp() * subtree_entropy;
    }

    /**
     * Make child node
     */
    shared_ptr<DentsCNode> DentsDNode::create_child_node_helper(shared_ptr<const Action> action) const {
        return make_shared<DentsCNode>(
            static_pointer_cast<DentsManager>(MctsDNode::mcts_manager), 
            state, 
            action, 
            decision_depth, 
            decision_timestep, 
            static_pointer_cast<const DentsDNode>(shared_from_this()));
    }

    /**
     * Return string with all of the relevant values in this node
     */
    string DentsDNode::get_pretty_print_val() const {
        DentsManager& manager = (DentsManager&) *mcts_manager;
        double val_estimate = manager.use_dp_value ? dp_value : avg_return; 

        stringstream ss;
        ss << val_estimate << "(temp:" << get_temp() << ",entrpy:" << subtree_entropy << ",val_temp:" << get_value_temp() 
            << ",soft_val:" << soft_value << ")";
        return ss.str();
    }
}

/**
 * Boilerplate MctsDNode interface implementation. Copied from mcts_decision_node_template.h.
 */
namespace mcts {
    shared_ptr<MctsCNode> DentsDNode::create_child_node_helper_itfc(shared_ptr<const Action> action) const {
        shared_ptr<const Action> act_itfc = static_pointer_cast<const Action>(action);
        shared_ptr<DentsCNode> child_node = create_child_node_helper(act_itfc);
        return static_pointer_cast<MctsCNode>(child_node);
    }
}
