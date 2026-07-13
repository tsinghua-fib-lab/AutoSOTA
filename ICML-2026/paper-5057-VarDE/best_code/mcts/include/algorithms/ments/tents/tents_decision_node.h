#pragma once

#include "algorithms/ments/tents/tents_chance_node.h"
#include "algorithms/ments/ments_manager.h"
#include "mcts_types.h"

#include "algorithms/ments/ments_decision_node.h"
#include "mcts_env.h"
#include "mcts_env_context.h"
#include "mcts_manager.h"

#include <map>
#include <memory>
#include <unordered_map>
#include <vector>

namespace mcts {
    // forward declare corresponding TentsCNode class
    class TentsCNode;

    /**
     * An implementation of TENTS in the Mcts schema
     * 
     * Paper: http://proceedings.mlr.press/v139/dam21a/dam21a.pdf
     * 
     * Maps are maintained to map between the values of 'soft_q_value/temp' (where soft_q_value is the soft value of 
     * a child) to prevent having to sort everytime the node is visited. So action selection may use slightly outdated 
     * q values in a multi-threaded setting, but many threads probably outweighs this consistency cost.
     * 
     * N.B. For convenience the code for tents frequently uses 'q_value' to mean the value of Q(s,a)/temp, trying to 
     * avoid being verbose.
     * 
     * TODO: consider changing to be more verbose but clearer/less confusing. It is consistent in this file but not 
     * clear what the value of the variable is when just read the name
     * 
     * Member variables:
     *      qval_to_act: 
     *          A multi-map from qvalue/temp values to actions (note that this is sorted from smallest to largest, so 
     *          the reverse iterator will iterate from largest to smallest)
     *      act_to_qval:
     *          An unordered_map from action to qvalue/temp values, which is used for tents computations, and useful 
     *          for looking up items in the qval_to_act multimap.
     *      _selected_action_key:
     *          A key to use for storing the selected action in MctsEnvContexts 
     */
    class TentsDNode : public MentsDNode {
        // Allow TentsCNode access to private members
        friend TentsCNode;

        /**
         * Core TentsDNode implementation.
         */
        protected:
            std::multimap<double,std::shared_ptr<const Action>> qval_to_act;
            std::unordered_map<std::shared_ptr<const Action>,double> act_to_qval;
            std::string _selected_action_key;

            /**
             * Returns the value of Q(s,a)/temp, which is frequently used in tents.
             * 
             * Assumes that we already hold locks for all of the children.
             * 
             * Args:
             *      action: The action that we want a value for
             * 
             * Returns:
             *      The soft value corresponding to the child at 'action', divided by the temperature
             * 
            */
            double get_soft_q_value_over_temp(std::shared_ptr<const Action> action) const;

            /**
             * Updates the 'qval_to_act' and 'act_to_qval' maps.
             * 
             * Assumes that we already hold locks for all of the children.
             * 
             * Args:
             *      action: The action to be updated in the maps
             *      new_q_value: The new q_value (over temp) to be updated in the maps
            */
            void update_maps(std::shared_ptr<const Action> action, double new_q_value);

            /**
             * Computes the sparse action set for this node.
             * 
             * Assumes that we already hold locks for all of the children.
             * See paper for definition of sparse_action_set. http://proceedings.mlr.press/v139/dam21a/dam21a.pdf
            */
            std::unique_ptr<std::vector<std::shared_ptr<const Action>>> get_sparse_action_set() const;

            /**
             * Computes the spmax at this node. 
             * 
             * Assumes that we already hold locks for all of the children.
             * See paper for definition of spmax. http://proceedings.mlr.press/v139/dam21a/dam21a.pdf
             * 
             * Returns:
             *      The spmax
            */
            double spmax() const;

            /**
             * Computes the weights for each action for the Tents action selection.
             * 
             * (This excludes any probability mass from epsilon exploration).
             * 
             * Assumes that we already hold locks for all of the children.
             * 
             * Args:
             *      action_weights: 
             *          An ActionDistr to be filled with weights for each action
             *      sun_action_weights:
             *          A double reference to be filled with the sum of all the weights in 'action_weights'
             *      normalisation_term:
             *          Not necessary for tents
             *      context:
             *          A Mcts env context
             */
            virtual void compute_action_weights(
                ActionDistr& action_weights, 
                double& sum_action_weights, 
                double& normalisation_term, 
                MctsEnvContext& context) const;

            /**
             * Updates the tents maps for the backup, using the selected action stored in ctx
             * 
             * Args:
             *      ctx: A Mcts env context
            */
           void backup_update_map(MctsEnvContext& ctx);

            /**
             * Implements the tents backup. I.e. soft_value = temp * spmax()
             * 
             * Args:
             *      ctx: A Mcts env context
             */
            void backup_tents(MctsEnvContext& ctx);





        /**
         * Core MctsDNode implementation functions.
         */
        public:  
            /**
             * Constructor
             */
            TentsDNode(
                std::shared_ptr<MentsManager> mcts_manager,
                std::shared_ptr<const State> state,
                int decision_depth,
                int decision_timestep,
                std::shared_ptr<const TentsCNode> parent=nullptr); 

            virtual ~TentsDNode() = default;
            
            /**
             * Implements the Mcts select_action function for the node
             * 
             * Just calls ments backup, but stores the action selected in the context for use in the backup
             * 
             * Args:
             *      ctx: A context provided to all Mcts functions throughout a trial to pass intermediate/transient info
             * 
             * Returns:
             *      The selected action
             */
            virtual std::shared_ptr<const Action> select_action(MctsEnvContext& ctx);
            
            /**
             * Implements the Mcts backup function for the node
             * 
             * Calls backup_tents
             * 
             * Args:
             *      trial_rewards_before_node: unused
             *      trial_rewards_after_node: unused
             *      trial_cumulative_return_after_node: unused
             *      trial_cumulative_return: unused
             *      ctx: A Mcts env context
             */
            virtual void backup(
                const std::vector<double>& trial_rewards_before_node, 
                const std::vector<double>& trial_rewards_after_node, 
                const double trial_cumulative_return_after_node, 
                const double trial_cumulative_return,
                MctsEnvContext& ctx);

        protected:
            /**
             * A helper function that makes a child node object on the heap and returns it. 
             * 
             * The 'create_child_node' boilerplate function uses this function to make a new child, add it to the 
             * children map (or bypass making the node using the transposition table if using). The function is marked 
             * const to enforce that we don't accidently try to duplicate logic surrounding adding children and 
             * interacting with the transposition table.
             * 
             * Args:
             *      action: An action to create a child node for
             * 
             * Returns:
             *      A pointer to a new MentsCNode object
             */
            std::shared_ptr<TentsCNode> create_child_node_helper(std::shared_ptr<const Action> action) const;



        /**
         * MctsDNode interface function definitions, used by Mcts subroutines to interact with this node. Copied from 
         * mcts_decision_node.h. 
         * 
         * Boilerplate definitions are provided in mcts_decision_node_template.h, that wrap above functions in pointer 
         * casts.
         */
        public:
            virtual std::shared_ptr<MctsCNode> create_child_node_helper_itfc(
                std::shared_ptr<const Action> action) const;
    };
}
