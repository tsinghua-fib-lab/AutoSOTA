#pragma once

#include "algorithms/uct/puct_chance_node.h"
#include "algorithms/uct/uct_decision_node.h"
#include "algorithms/uct/puct_manager.h"

#include <memory>

namespace mcts {
    // forward declare corresponding PuctCNode class
    class PuctCNode;

    /**
     * Implementation of PUCT (decision nodes) in Mcts schema. 
     * 
     * Puct decision nodes only need to do the following things differently to uct:
     * - compute a different form of the ucb term,
     * - assure that children are created as PuctCNode's rather than UctCNodes.
     */
    class PuctDNode : public UctDNode {
        friend PuctCNode;

        protected:

        public: 
            /**
             * Constructor
             */
            PuctDNode(
                std::shared_ptr<PuctManager> mcts_manager,
                std::shared_ptr<const State> state,
                int decision_depth,
                int decision_timestep,
                std::shared_ptr<const PuctCNode> parent=nullptr); 

            virtual ~PuctDNode() = default;

        protected:
            /**
             * Implements the updated form of the ucb term for puct.
             */
            virtual double compute_ucb_term(int num_visits, int child_visits) const;

            /**
             * Helper to make a PuctCNode child object.
             */
            std::shared_ptr<PuctCNode> create_child_node_helper(std::shared_ptr<const Action> action) const;

        public:
            /**
             * Override create_child_node_helper_itfc, so that the MctsDNode create_child_node_itfc function can 
             * create the correct PuctCNode using the above version of create_child_node
             */
            virtual std::shared_ptr<MctsCNode> create_child_node_helper_itfc(std::shared_ptr<const Action> action) const;
    };
}
