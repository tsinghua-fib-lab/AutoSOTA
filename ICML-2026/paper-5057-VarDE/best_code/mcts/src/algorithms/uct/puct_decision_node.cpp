#include "algorithms/uct/puct_decision_node.h"

#include <cmath>

using namespace std; 

namespace mcts {
    /**
     * Construct Puct Decision node.
     */
    PuctDNode::PuctDNode(
        shared_ptr<PuctManager> mcts_manager,
        shared_ptr<const State> state,
        int decision_depth,
        int decision_timestep,
        shared_ptr<const PuctCNode> parent) :
            UctDNode(mcts_manager, state, decision_depth, decision_timestep, parent)
    {
    }

    /**
     * Computes the ucb term used in action selection. I.e. sqrt(log N(s) / N(s,a)).
     */
    double PuctDNode::compute_ucb_term(int num_visits, int child_visits) const {
        shared_ptr<PuctManager> manager = static_pointer_cast<PuctManager>(mcts_manager);
        double num_visits_d = (num_visits > 0) ? (double)num_visits : 1.0;
        double child_visits_d = (child_visits > 0) ? (double)child_visits : 1.0;
        if (manager->puct_power != 0.5) {
            return pow(num_visits_d, manager->puct_power) / child_visits_d;
        }
        return sqrt(num_visits_d) / child_visits_d;
    }
    
    /**
     * Make a new PuctCNode on the heap, with correct arguments for a child node.
     */
    shared_ptr<PuctCNode> PuctDNode::create_child_node_helper(shared_ptr<const Action> action) const {
        return make_shared<PuctCNode>(
            static_pointer_cast<PuctManager>(mcts_manager), 
            state, 
            action, 
            decision_depth, 
            decision_timestep, 
            static_pointer_cast<const PuctDNode>(shared_from_this()));
    }

    /**
     * Make create_child functions make a Puct child rather than a UctCNode or MctsCNode
     */
    shared_ptr<MctsCNode> PuctDNode::create_child_node_helper_itfc(shared_ptr<const Action> action) const {
        shared_ptr<PuctCNode> child_node = create_child_node_helper(action);
        return static_pointer_cast<MctsCNode>(child_node);
    }
}