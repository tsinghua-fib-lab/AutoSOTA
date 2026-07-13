#pragma once

#include "algorithms/ments/dents/dents_chance_node.h"
#include "algorithms/ments/dents/dents_manager.h"
#include "mcts_types.h"

#include "algorithms/ments/dbments_chance_node.h"
#include "algorithms/ments/dbments_decision_node.h"
#include "algorithms/common/emp_node.h"
#include "algorithms/common/ent_chance_node.h"
#include "algorithms/common/ent_decision_node.h"
#include "mcts_env.h"
#include "mcts_env_context.h"
#include "mcts_manager.h"

#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>

namespace mcts {
    // forward declare corresponding DentsDNode class
    class DentsCNode;

    /**
     * An implementation of I(mproved)DB-DENTS in the Mcts schema
     * 
     * This implementation adds entropy backups to DB-MENTS to compute a soft value using a decayed entropy. Decayed 
     * temperature is used as a coefficient of entropy when computing soft values, search temperature is used in the 
     * energy based policy.
     */
    class DentsDNode : public DBMentsDNode, public EntDNode, public EmpNode {
        // Allow MentsDNode access to private members
        friend DentsCNode;

        public:  
            /**
             * Constructor
             */
            DentsDNode(
                std::shared_ptr<DentsManager> mcts_manager,
                std::shared_ptr<const State> state,
                int decision_depth,
                int decision_timestep,
                std::shared_ptr<const DentsCNode> parent=nullptr); 

            virtual ~DentsDNode() = default;
            
            /**
             * Helper to get the temperature that should be used for computing the soft value to use in ments functions.
             * I.e. The 'value_temp' refers to the temperature coefficient of entropy when computing (soft) values
             */
            virtual double get_value_temp() const;            
            
            /**
             * Helper to get the soft q-value of an action. Taking into account for if we are acting as an opponent.
             * 
             * This computes the value of V + alpha_decayed * H, for each child, where V is the dp value of the child,
             * H is the subtree entropy of the child and alpha_decayed is the decayed temperature.
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
             * Uses the DPDNode to recommend an action according to the DP values.
             * 
             * Recommends a random action if this node has zero children.
             */
            virtual std::shared_ptr<const Action> recommend_action_best_empirical_value() const;
            
            /**
             * Implements the Mcts recommend_action function for the node
             * 
             * Args:
             *      ctx: A context for if a recommendation also requires a context
             * 
             * Returns:
             *      The recommended action
             */
            virtual std::shared_ptr<const Action> recommend_action(MctsEnvContext& ctx) const;   
            
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
             * Helper to make a DentsCNode child object.
             */
            std::shared_ptr<DentsCNode> create_child_node_helper(std::shared_ptr<const Action> action) const;

            /**
             * Override the pretty print to print out the soft value, dp value, entropy and temperature
             */
            virtual std::string get_pretty_print_val() const;
        
        public:
            /**
             * Override create_child_node_helper_itfc still, so that the MctsDNode create_child_node_itfc function can 
             * create the correct DentsCNode using the above version of create_child_node
             */
            virtual std::shared_ptr<MctsCNode> create_child_node_helper_itfc(
                std::shared_ptr<const Action> action) const;
    };
}