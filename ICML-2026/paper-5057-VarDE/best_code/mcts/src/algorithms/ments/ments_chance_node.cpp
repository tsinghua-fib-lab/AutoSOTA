#include "algorithms/ments/ments_chance_node.h"

#include "helper_templates.h"

using namespace std;

namespace mcts {
    /**
     * Constructor
     */
    MentsCNode::MentsCNode(
        shared_ptr<MentsManager> mcts_manager,
        shared_ptr<const State> state,
        shared_ptr<const Action> action,
        int decision_depth,
        int decision_timestep,
        shared_ptr<const MentsDNode> parent) :
            MctsCNode(
                static_pointer_cast<MctsManager>(mcts_manager),
                state,
                action,
                decision_depth,
                decision_timestep,
                static_pointer_cast<const MctsDNode>(parent)),
            num_backups(0),
            soft_value(mcts_manager->default_q_value),
            local_reward(mcts_manager->mcts_env->get_reward_itfc(state,action)),
            next_state_distr(mcts_manager->mcts_env->get_transition_distribution_itfc(state,action)) 
    {
    }

    /**
     * Visit just needs to increment num_visits.
     */
    void MentsCNode::visit(MctsEnvContext& ctx) {
        MctsCNode::visit_itfc(ctx);
    }

    /**
     * Implementation of sample_observation, that uses the sample from distribution helper function.
     */
    shared_ptr<const State> MentsCNode::sample_observation_random() {
        shared_ptr<const State> sampled_state = helper::sample_from_distribution(*next_state_distr, *mcts_manager);
        if (!has_child_node(sampled_state)) {
            create_child_node(sampled_state);
        }
        return sampled_state;
    }
    
    /**
     * Sample observation calls sample_observation_random.
     */
    shared_ptr<const State> MentsCNode::sample_observation(MctsEnvContext& ctx) {
        return sample_observation_random();
    }


    /**
     * Ments soft backup averages the value of children nodes, and adds the immediate reward for R(s,a).
     * 
     * Remember to increment num backups at the end. We sum the child backups because in concurrent envionrments they 
     * might be greater than the number of backups at this node.
     * 
     * Implemented as a running average to avoid using two loops. Feels like it should be more efficient, but would be 
     * nice to check at some point if really optimising.
     * 
     * Made 'sum_child_backups' a double to force computations to all be floating point rather than integer (otherwise 
     * get an integer div).
     * 
     * Additional note on concurrency fun. It is possible and valid to currently have a child with zero backups. 
     * Consider if we have another trial that also searches this node, it made a new child, but hasn't backed it up 
     * yet. Hence it's necessary to include the line "if (child.num_backups == 0) continue;" to avoid a division by 
     * zero causing NaNs.
     */
    void MentsCNode::backup_soft() {
        num_backups++;

        soft_value = 0.0;
        double sum_child_backups = 0.0;
        lock_all_children();
        for (pair<shared_ptr<const Observation>,shared_ptr<MctsDNode>> pr : children) {
            MentsDNode& child = (MentsDNode&) *pr.second;
            if (child.num_backups == 0) continue;
            sum_child_backups += child.num_backups;
            soft_value *= (sum_child_backups - child.num_backups) / sum_child_backups;
            soft_value += child.num_backups * child.soft_value / sum_child_backups; 
        }
        unlock_all_children();
        soft_value += local_reward; // +R(s,a)
    }

    /**
     * Calls ments soft backup
     */
    void MentsCNode::backup(
        const std::vector<double>& trial_rewards_before_node, 
        const std::vector<double>& trial_rewards_after_node, 
        const double trial_cumulative_return_after_node, 
        const double trial_cumulative_return,
        MctsEnvContext& ctx)
    {   
        backup_soft();
    }

    /**
     * Make child node
     */
    shared_ptr<MentsDNode> MentsCNode::create_child_node_helper(
        shared_ptr<const State> observation, shared_ptr<const State> next_state) const 
    {  
        shared_ptr<const State> mdp_next_state = static_pointer_cast<const State>(observation);
        return make_shared<MentsDNode>(
            static_pointer_cast<MentsManager>(mcts_manager), 
            mdp_next_state,
            decision_depth+1, 
            decision_timestep+1, 
            static_pointer_cast<const MentsCNode>(shared_from_this()));
    }

    /**
     * Return string of the soft value
     */
    string MentsCNode::get_pretty_print_val() const {
        stringstream ss;
        ss << soft_value;
        return ss.str();
    }
}

/**
 * Boilerplate function definitions.
 * All this code basically calls the corresponding base implementation function, with approprtiate casts before/after.
 */
namespace mcts {
    shared_ptr<MentsDNode> MentsCNode::create_child_node(shared_ptr<const State> observation, shared_ptr<const State> next_state) {
        shared_ptr<const Observation> obsv_itfc = static_pointer_cast<const Observation>(observation);
        shared_ptr<const State> next_state_itfc = static_pointer_cast<const State>(next_state);
        shared_ptr<MctsDNode> new_child = MctsCNode::create_child_node_itfc(obsv_itfc, next_state_itfc);
        return static_pointer_cast<MentsDNode>(new_child);
    }

    bool MentsCNode::has_child_node(std::shared_ptr<const State> observation) const {
        return MctsCNode::has_child_node_itfc(static_pointer_cast<const Observation>(observation));
    }
    
    shared_ptr<MentsDNode> MentsCNode::get_child_node(shared_ptr<const State> observation) const {
        shared_ptr<const Observation> obsv_itfc = static_pointer_cast<const Observation>(observation);
        shared_ptr<MctsDNode> new_child = MctsCNode::get_child_node_itfc(obsv_itfc);
        return static_pointer_cast<MentsDNode>(new_child);
    }
}

/**
 * Boilerplate MctsCNode interface implementation. Copied from mcts_chance_node_template.h.
 */
namespace mcts {
    void MentsCNode::visit_itfc(MctsEnvContext& ctx) {
        MctsEnvContext& ctx_itfc = (MctsEnvContext&) ctx;
        visit(ctx_itfc);
    }

    shared_ptr<const Observation> MentsCNode::sample_observation_itfc(MctsEnvContext& ctx) {
        MctsEnvContext& ctx_itfc = (MctsEnvContext&) ctx;
        shared_ptr<const State> obsv = sample_observation(ctx_itfc);
        return static_pointer_cast<const Observation>(obsv);
    }

    void MentsCNode::backup_itfc(
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

    shared_ptr<MctsDNode> MentsCNode::create_child_node_helper_itfc(
        shared_ptr<const Observation> observation, shared_ptr<const State> next_state) const 
    {
        shared_ptr<const State> obsv_itfc = static_pointer_cast<const State>(observation);
        shared_ptr<const State> next_state_itfc = static_pointer_cast<const State>(next_state);
        shared_ptr<MentsDNode> child_node = create_child_node_helper(obsv_itfc, next_state_itfc);
        return static_pointer_cast<MctsDNode>(child_node);
    }
}