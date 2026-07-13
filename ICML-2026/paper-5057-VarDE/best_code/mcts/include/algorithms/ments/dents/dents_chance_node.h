#pragma once

#include "algorithms/ments/dents/dents_decision_node.h"
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
    class DentsDNode;
    
    /**
     * An implementation of I(mproved)DB-DENTS in the Mcts schema
     * 
     * The implementation only needs to change backup functions to use dp backups too.
     */
    class DentsCNode : public DBMentsCNode, public EntCNode, public EmpNode {
        // Allow DentsDNode access to private members
        friend DentsDNode;

        public: 
            /**
             * Constructor
             */
            DentsCNode(
                std::shared_ptr<DentsManager> mcts_manager,
                std::shared_ptr<const State> state,
                std::shared_ptr<const Action> action,
                int decision_depth,
                int decision_timestep,
                std::shared_ptr<const DentsDNode> parent=nullptr);

            virtual ~DentsCNode() = default;
            
            /**
             * Helper to get the temperature that should be used for computing the soft value to use in ments functions.
             * I.e. The 'value_temp' refers to the temperature coefficient of entropy when computing (soft) values
             */
            virtual double get_value_temp() const;   

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
             * Helper to make a DentsDNode child object.
             */
            std::shared_ptr<DentsDNode> create_child_node_helper(
                std::shared_ptr<const State> observation, std::shared_ptr<const State> next_state=nullptr) const;

            /**
             * Override the pretty print to print out both the dp value and the soft value
             */
            virtual std::string get_pretty_print_val() const;
            
        public:
            /**
             * Override create_child_node_helper_itfc still, so that the MctsCNode create_child_node_itfc function can 
             * create the correct DentsDNode using the above version of create_child_node
             */
            virtual std::shared_ptr<MctsDNode> create_child_node_helper_itfc(
                std::shared_ptr<const Observation> observation, std::shared_ptr<const State> next_state=nullptr) const;
    };
}