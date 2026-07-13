#pragma once

#include "algorithms/est/est_chance_node.h"
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
    class EstCNode;

    /**
     * Implementation of Boltzmann search Mcts, built ontop of DBMents because that's already coded up
     * 
     */
    class EstDNode : public DentsDNode {
        // Allow IDBMentsDNode access to private members
        friend EstCNode;

        public:  
            /**
             * Constructor
             */
            EstDNode(
                std::shared_ptr<DentsManager> mcts_manager,
                std::shared_ptr<const State> state,
                int decision_depth,
                int decision_timestep,
                std::shared_ptr<const EstCNode> parent=nullptr); 

            virtual ~EstDNode() = default;          

            /**
             * To implement Est given an implementation of ments, we just need to swap out the soft q value to 
             * return the dp backup value
             * 
             * Args:
             *      action: 
             *          The action to get the corresponding q value for
             *      opponent_coeff: 
             *          A value of -1.0 or 1.0 for if we are acting as the opponent in a two player game or not 
             *          respectively
             */
            virtual double get_soft_q_value(std::shared_ptr<const Action> action, double opponent_coeff) const;
            
            /**
             * Calls both the entropy backup and dp backup from DPDNode
             */
            virtual void backup(
                const std::vector<double>& trial_rewards_before_node, 
                const std::vector<double>& trial_rewards_after_node, 
                const double trial_cumulative_return_after_node, 
                const double trial_cumulative_return,
                MctsEnvContext& ctx);

        protected:
            /**
             * Helper to make a EstCNode child object.
             */
            std::shared_ptr<EstCNode> create_child_node_helper(std::shared_ptr<const Action> action) const;

            /**
             * Override the pretty print to print out the soft value, dp value, entropy and temperature
             */
            virtual std::string get_pretty_print_val() const;
        
        public:
            /**
             * Override create_child_node_helper_itfc still, so that the MctsDNode create_child_node_itfc function can 
             * create the correct EstCNode using the above version of create_child_node
             */
            virtual std::shared_ptr<MctsCNode> create_child_node_helper_itfc(
                std::shared_ptr<const Action> action) const;
    };
}