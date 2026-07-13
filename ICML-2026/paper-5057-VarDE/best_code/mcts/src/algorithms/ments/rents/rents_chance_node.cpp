#include "algorithms/ments/rents/rents_chance_node.h"

using namespace std;

namespace mcts {
    /**
     * Constructor
     */
    RentsCNode::RentsCNode(
        shared_ptr<MentsManager> mcts_manager,
        shared_ptr<const State> state,
        shared_ptr<const Action> action,
        int decision_depth,
        int decision_timestep,
        shared_ptr<const RentsDNode> parent) :
            MentsCNode(
                mcts_manager,
                state,
                action,
                decision_depth,
                decision_timestep,
                static_pointer_cast<const MentsDNode>(parent))
    {
    }

    /**
     * Make child node
     */
    shared_ptr<RentsDNode> RentsCNode::create_child_node_helper(
        shared_ptr<const State> observation, shared_ptr<const State> next_state) const 
    {  
        shared_ptr<const State> mdp_next_state = static_pointer_cast<const State>(observation);
        return make_shared<RentsDNode>(
            static_pointer_cast<MentsManager>(mcts_manager), 
            mdp_next_state,
            decision_depth+1, 
            decision_timestep+1, 
            static_pointer_cast<const RentsCNode>(shared_from_this()));
    }
}

/**
 * Boilerplate MctsCNode interface implementation. Copied from mcts_chance_node_template.h.
 */
namespace mcts {
    shared_ptr<MctsDNode> RentsCNode::create_child_node_helper_itfc(
        shared_ptr<const Observation> observation, shared_ptr<const State> next_state) const 
    {
        shared_ptr<const State> obsv_itfc = static_pointer_cast<const State>(observation);
        shared_ptr<const State> next_state_itfc = static_pointer_cast<const State>(next_state);
        shared_ptr<RentsDNode> child_node = create_child_node_helper(obsv_itfc, next_state_itfc);
        return static_pointer_cast<MctsDNode>(child_node);
    }
}