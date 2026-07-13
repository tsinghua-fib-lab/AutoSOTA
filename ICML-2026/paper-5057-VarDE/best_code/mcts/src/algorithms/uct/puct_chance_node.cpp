#include "algorithms/uct/puct_chance_node.h"

using namespace std; 

namespace mcts {
    /**
     * Construct Puct Chance node
     */
    PuctCNode::PuctCNode(
        shared_ptr<PuctManager> mcts_manager,
        shared_ptr<const State> state,
        shared_ptr<const Action> action,
        int decision_depth,
        int decision_timestep,
        shared_ptr<const PuctDNode> parent) :
            UctCNode(mcts_manager, state, action, decision_depth, decision_timestep, parent)
    {  
    }

    /**
     * Make a new PuctDNode on the heap, with correct arguments for a child node.
     */
    shared_ptr<PuctDNode> PuctCNode::create_child_node_helper(shared_ptr<const State> observation) const
    {
        shared_ptr<const State> next_state = static_pointer_cast<const State>(observation);
        return make_shared<PuctDNode>(
            static_pointer_cast<PuctManager>(mcts_manager), 
            next_state,
            decision_depth+1, 
            decision_timestep+1, 
            static_pointer_cast<const PuctCNode>(shared_from_this()));
    }

    /**
     * Make create_child functions make a Puct child rather than a UctCNode or MctsCNode
     */
    shared_ptr<MctsDNode> PuctCNode::create_child_node_helper_itfc(
        shared_ptr<const Observation> observation, shared_ptr<const State> next_state) const 
    {
        shared_ptr<const State> obsv_itfc = static_pointer_cast<const State>(observation);
        shared_ptr<PuctDNode> child_node = create_child_node_helper(obsv_itfc);
        return static_pointer_cast<MctsDNode>(child_node);
    }
}