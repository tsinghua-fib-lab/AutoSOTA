#include "algorithms/uct/uct_chance_node.h"

#include "helper_templates.h"

using namespace std; 

namespace mcts {
    /**
     * Construct Uct Chance node. Use mcts_manager to initialise values with heuristic if necessary.
     */
    UctCNode::UctCNode(
        shared_ptr<UctManager> mcts_manager,
        shared_ptr<const State> state,
        shared_ptr<const Action> action,
        int decision_depth,
        int decision_timestep,
        shared_ptr<const UctDNode> parent) :
            MctsCNode(
                static_pointer_cast<MctsManager>(mcts_manager),
                state,
                action,
                decision_depth,
                decision_timestep,
                static_pointer_cast<const MctsDNode>(parent)),
            num_backups(0),
            avg_return(0.0),
            next_state_distr(mcts_manager->mcts_env->get_transition_distribution_itfc(state,action))
    {  
    }

    /**
     * Visit just needs to increment num_visits.
     */
    void UctCNode::visit(MctsEnvContext& ctx) {
        MctsCNode::visit_itfc(ctx);
    }

    /**
     * Implementation of sample_observation, that uses the sample from distribution helper function.
     */
    shared_ptr<const State> UctCNode::sample_observation_random() {
        shared_ptr<const State> sampled_state = helper::sample_from_distribution(*next_state_distr, *mcts_manager);
        if (!has_child_node(sampled_state)) {
            create_child_node(sampled_state);
        }
        return sampled_state;
    }
    
    /**
     * Sample observation calls sample_observation_random.
     */
    shared_ptr<const State> UctCNode::sample_observation(MctsEnvContext& ctx) {
        return sample_observation_random();
    }

    /**
     * Computes running average.
     */
    void UctCNode::backup_average_return(const double trial_return_after_node) {
        num_backups++;
        avg_return += (trial_return_after_node - avg_return) / (double) num_backups;
    }

    /**
     * Calls the running average return backup function.
     */
    void UctCNode::backup(
        const vector<double>& trial_rewards_before_node, 
        const vector<double>& trial_rewards_after_node, 
        const double trial_cumulative_return_after_node, 
        const double trial_cumulative_return,
        MctsEnvContext& ctx) 
    {
        backup_average_return(trial_cumulative_return_after_node);
    }

    /**
     * Make a new UctDNode on the heap, with correct arguments for a child node.
     */
    shared_ptr<UctDNode> UctCNode::create_child_node_helper(shared_ptr<const State> observation) const
    {
        shared_ptr<const State> next_state = static_pointer_cast<const State>(observation);
        return make_shared<UctDNode>(
            static_pointer_cast<UctManager>(mcts_manager), 
            next_state,
            decision_depth+1, 
            decision_timestep+1, 
            static_pointer_cast<const UctCNode>(shared_from_this()));
    }

    /**
     * Pretty print val = print current avg_return in node
     */
    string UctCNode::get_pretty_print_val() const {
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
    shared_ptr<UctDNode> UctCNode::create_child_node(shared_ptr<const State> observation) 
    {
        shared_ptr<const Observation> obsv_itfc = static_pointer_cast<const Observation>(observation);
        shared_ptr<MctsDNode> new_child = MctsCNode::create_child_node_itfc(obsv_itfc);
        return static_pointer_cast<UctDNode>(new_child);
    }

    bool UctCNode::has_child_node(std::shared_ptr<const State> observation) const {
        return MctsCNode::has_child_node_itfc(static_pointer_cast<const Observation>(observation));
    }
    
    shared_ptr<UctDNode> UctCNode::get_child_node(shared_ptr<const State> observation) const {
        shared_ptr<const Observation> obsv_itfc = static_pointer_cast<const Observation>(observation);
        shared_ptr<MctsDNode> new_child = MctsCNode::get_child_node_itfc(obsv_itfc);
        return static_pointer_cast<UctDNode>(new_child);
    }
}

/**
 * Boilerplate MctsCNode interface implementation. Copied from mcts_chance_node_template.h.
 */
namespace mcts {
    void UctCNode::visit_itfc(MctsEnvContext& ctx) {
        MctsEnvContext& ctx_itfc = (MctsEnvContext&) ctx;
        visit(ctx_itfc);
    }

    shared_ptr<const Observation> UctCNode::sample_observation_itfc(MctsEnvContext& ctx) {
        MctsEnvContext& ctx_itfc = (MctsEnvContext&) ctx;
        shared_ptr<const State> obsv = sample_observation(ctx_itfc);
        return static_pointer_cast<const Observation>(obsv);
    }

    void UctCNode::backup_itfc(
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

    shared_ptr<MctsDNode> UctCNode::create_child_node_helper_itfc(
        shared_ptr<const Observation> observation, shared_ptr<const State> next_state) const 
    {
        shared_ptr<const State> obsv_itfc = static_pointer_cast<const State>(observation);
        shared_ptr<UctDNode> child_node = create_child_node_helper(obsv_itfc);
        return static_pointer_cast<MctsDNode>(child_node);
    }
}