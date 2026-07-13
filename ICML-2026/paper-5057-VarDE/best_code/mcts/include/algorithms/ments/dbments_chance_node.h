#pragma once

#include "algorithms/ments/dbments_decision_node.h"
#include "algorithms/ments/ments_manager.h"
#include "mcts_types.h"

#include "algorithms/ments/ments_chance_node.h"
#include "algorithms/ments/ments_decision_node.h"
#include "algorithms/common/dp_chance_node.h"
#include "algorithms/common/dp_decision_node.h"
#include "mcts_env.h"
#include "mcts_env_context.h"
#include "mcts_manager.h"

#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>

namespace mcts {
    // forward declare corresponding DBMentsDNode class
    class DBMentsDNode;
    
    /**
     * An implementation of DB-MENTS in the Mcts schema
     * 
     * The implementation only needs to change backup functions to use dp backups too.
     */
    class DBMentsCNode : public MentsCNode, public DPCNode {
        // Allow DBMentsDNode access to private members
        friend DBMentsDNode;

        protected:

        public: 
            /**
             * Constructor
             */
            DBMentsCNode(
                std::shared_ptr<MentsManager> mcts_manager,
                std::shared_ptr<const State> state,
                std::shared_ptr<const Action> action,
                int decision_depth,
                int decision_timestep,
                std::shared_ptr<const DBMentsDNode> parent=nullptr);

            virtual ~DBMentsCNode() = default;
            
            /**
             * Calls both the soft backup from MentsCNode and dp backup from DPCNode
             */
            virtual void backup(
                const std::vector<double>& trial_rewards_before_node, 
                const std::vector<double>& trial_rewards_after_node, 
                const double trial_cumulative_return_after_node, 
                const double trial_cumulative_return,
                MctsEnvContext& ctx);

        protected:
            /**
             * Helper to make a DBMentsDNode child object.
             */
            std::shared_ptr<DBMentsDNode> create_child_node_helper(
                std::shared_ptr<const State> observation, std::shared_ptr<const State> next_state=nullptr) const;

            /**
             * Override the pretty print to print out both the dp value and the soft value
             */
            virtual std::string get_pretty_print_val() const;
            
        public:
            /**
             * Override create_child_node_helper_itfc still, so that the MctsCNode create_child_node_itfc function can 
             * create the correct DBMentsDNode using the above version of create_child_node
             */
            virtual std::shared_ptr<MctsDNode> create_child_node_helper_itfc(
                std::shared_ptr<const Observation> observation, std::shared_ptr<const State> next_state=nullptr) const;
    };
}