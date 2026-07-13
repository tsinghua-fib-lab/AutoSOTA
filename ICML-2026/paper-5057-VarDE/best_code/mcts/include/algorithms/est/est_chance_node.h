#pragma once

#include "algorithms/est/est_decision_node.h"
#include "algorithms/ments/dents/dents_manager.h"
#include "mcts_types.h"

#include "algorithms/ments/dents/dents_chance_node.h"
#include "algorithms/ments/dents/dents_decision_node.h"
#include "mcts_env.h"
#include "mcts_env_context.h"
#include "mcts_manager.h"

#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>

namespace mcts {
    // forward declare corresponding EstDNode class
    class EstDNode;
#
    /**
     * Implementation of Boltzmann search Mcts
     * 
     */
    class EstCNode : public DentsCNode {
        // Allow EstDNode access to private members
        friend EstDNode;

        protected:

        public: 
            /**
             * Constructor
             */
            EstCNode(
                std::shared_ptr<DentsManager> mcts_manager,
                std::shared_ptr<const State> state,
                std::shared_ptr<const Action> action,
                int decision_depth,
                int decision_timestep,
                std::shared_ptr<const EstDNode> parent=nullptr);

            virtual ~EstCNode() = default;

            /**
             * Calls both the entropy backup and dp backup from DPCNode
             */
            virtual void backup(
                const std::vector<double>& trial_rewards_before_node, 
                const std::vector<double>& trial_rewards_after_node, 
                const double trial_cumulative_return_after_node, 
                const double trial_cumulative_return,
                MctsEnvContext& ctx);

        protected:
            /**
             * Helper to make a EstDNode child object.
             */
            std::shared_ptr<EstDNode> create_child_node_helper(
                std::shared_ptr<const State> observation, std::shared_ptr<const State> next_state=nullptr) const;

            /**
             * Override the pretty print to print out both the dp value and the soft value
             */
            virtual std::string get_pretty_print_val() const;
            
        public:
            /**
             * Override create_child_node_helper_itfc still, so that the MctsCNode create_child_node_itfc function can 
             * create the correct EstDNode using the above version of create_child_node
             */
            virtual std::shared_ptr<MctsDNode> create_child_node_helper_itfc(
                std::shared_ptr<const Observation> observation, std::shared_ptr<const State> next_state=nullptr) const;
    };
}