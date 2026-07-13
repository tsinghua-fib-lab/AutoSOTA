#pragma once

#include "algorithms/ments/ments_decision_node.h"
#include "algorithms/ments/ments_manager.h"
#include "mcts_types.h"

#include "mcts_chance_node.h"
#include "mcts_decision_node.h"
#include "mcts_env.h"
#include "mcts_env_context.h"
#include "mcts_manager.h"

#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>

namespace mcts {
    // forward declare corresponding MentsDNode class
    class MentsDNode;
    
    /**
     * Implementation of Ments chance nodes in the Mcts schema. (Just randomly samples outcomes).
     * 
     * Note that the cached reward implies that this implementation can only be used with R(s,a) style rewards, 
     * and not with R(s,a,s') style rewards.
     * 
     * Member variables:
     *      soft_value: 
     *          The soft value at this node
     *      num_backups: 
     *          The number of times this node has been backed up
     *      local_reward: 
     *          A cached value of the environments reward R(state,action), for this nodes state, action pair. (Mostly 
     *          for the case where it is non-trivial to compute the reward, so don't recompute).
     *      next_state_distr: 
     *          A cached StateDistribution, representing the distribution over possible next states
     */
    class MentsCNode : public MctsCNode {
        // Allow MentsDNode access to private members
        friend MentsDNode;

        /**
         * Core MentsCNode implementation.
         */
        protected:
            int num_backups;
            double soft_value;
            double local_reward;
            std::shared_ptr<StateDistr> next_state_distr;

            /**
             * Handles the Mcts sample_observation function by randomly sampling.
             * 
             * Returns:
             *      The sampled next state
             */
            std::shared_ptr<const State> sample_observation_random();

            /**
             * Implements a soft backup for ments. (Average of child nodes values + immediate reward R(s,a))
             * 
             * I.e. Q(s,a) = R(s,a) + sum_{s'} [N(s')V(s') / N(s,a)]
             */
            void backup_soft();



        /**
         * Core MctsCNode implementation functions.
         */
        public: 
            /**
             * Constructor
             */
            MentsCNode(
                std::shared_ptr<MentsManager> mcts_manager,
                std::shared_ptr<const State> state,
                std::shared_ptr<const Action> action,
                int decision_depth,
                int decision_timestep,
                std::shared_ptr<const MentsDNode> parent=nullptr);

            /**
             * Implements the Mcts visit function for the node
             * 
             * Args:
             *      ctx: A context provided to all Mcts functions throughout a trial to pass intermediate/transient info
             */
            void visit(MctsEnvContext& ctx);
            
            /**
             * Implements the Mcts sample_observation function for the node
             * 
             * Args:
             *      ctx: A context provided to all Mcts functions throughout a trial to pass intermediate/transient info
             * 
             * Returns:
             *      The sampled observation
             */
            std::shared_ptr<const State> sample_observation(MctsEnvContext& ctx);
            
            /**
             * Implements the Mcts backup function for the node
             * 
             * Args:
             *      trial_rewards_before_node: 
             *          A list of rewards recieved (at each timestep) on the trial prior to reaching this node.
             *      trial_rewards_after_node:
             *          A list of rewards recieved (at each timestep) on the trial after reaching this node. This list 
             *          includes the reward from R(state,action) that would have been recieved from taking an action 
             *          from this node.
             *      trial_cumulative_return_after_node:
             *          Sum of rewards in the 'trial_rewards_after_node' list
             *      trial_cumulative_return:
             *          Sum of rewards in both of the 'trial_rewards_after_node' and 'trial_rewards_before_node' lists
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
             *      observation: The observation object leading to the child node
             *      next_state: The next state to construct the child node with
             * 
             * Returns:
             *      A pointer to a new MentsDNode object
             */
            std::shared_ptr<MentsDNode> create_child_node_helper(
                std::shared_ptr<const State> observation, std::shared_ptr<const State> next_state=nullptr) const;

            /**
             * Returns a string representation of the value of this node currently. Used for pretty printing.
             * 
             * Returns:
             *      A string representing the value of this node
             */
            virtual std::string get_pretty_print_val() const;



        /**
         * Boilerplate function definitions. 
         * 
         * Functionality implemented in mcts_decision_node.h, but it's useful to have wrappers to avoid needing to 
         * use pointer casts frequently.
         * 
         * Boilerplate implementations provided in mcts_chance_node_template.h
         */
        public:
            /**
             * Mark destructor as virtual.
             */
            virtual ~MentsCNode() = default;

            /**
             * Creates a child node, handles the internal management of the creation and returns a pointer to it.
             * 
             * This funciton is a wrapper for the create_child_node_itfc function definted in mcts_decision_node.cpp, 
             * and handles the casting required to use it.
             * 
             * - If the child already exists in children, it returns a pointer to that child.
             * - (If using transposition table) If the child already exists in the transposition table, but not in 
             *      children, it adds the child to children and then returns a pointer to it.
             * - If the child hasn't been created before, it makes the child (using 'create_child_node_helper'), and 
             *      inserts it appropriately into children (and the transposition table if relevant).
             * 
             * Args:
             *      observation: The observation object leading to the child node
             *      next_state: The next state to construct the child node with
             * 
             * Returns:
             *      A pointer to a new child chance node
             */
            std::shared_ptr<MentsDNode> create_child_node(
                std::shared_ptr<const State> observation, std::shared_ptr<const State> next_state=nullptr);

            /**
             * If this node has a child object corresponding to 'observation'.
             * 
             * Args:
             *      observation: An observation to check if we have a child for
             * 
             * Returns:
             *      true if we have a child corresponding to 'observation'
             */
            bool has_child_node(std::shared_ptr<const State> observation) const;

            /**
             * Retrieves a child node from the children map.
             * 
             * If a child doesn't exist for the observation, an exception will be thrown.
             * 
             * Args:
             *      observation: The observation to get the corresponding child of
             * 
             * Returns:
             *      A pointer to the child node corresponding to 'observation'
             */
            std::shared_ptr<MentsDNode> get_child_node(std::shared_ptr<const State> observation) const;



        /**
         * MctsCNode interface function definitions, used by Mcts subroutines to interact with this node. Copied from 
         * mcts_chance_node.h. 
         * 
         * Boilerplate definitions are provided in mcts_chance_node_template.h, that wrap above functions in pointer 
         * casts.
         */
        public:
            virtual void visit_itfc(MctsEnvContext& ctx);
            virtual std::shared_ptr<const Observation> sample_observation_itfc(MctsEnvContext& ctx);
            virtual void backup_itfc(
                const std::vector<double>& trial_rewards_before_node, 
                const std::vector<double>& trial_rewards_after_node, 
                const double trial_cumulative_return_after_node, 
                const double trial_cumulative_return,
                MctsEnvContext& ctx);

            virtual std::shared_ptr<MctsDNode> create_child_node_helper_itfc(
                std::shared_ptr<const Observation> observation, std::shared_ptr<const State> next_state=nullptr) const;
            // virtual std::shared_ptr<MctsDNode> create_child_node_itfc(
            //    std::shared_ptr<const Observation> observation, std::shared_ptr<const State> next_state=nullptr) final;
                


        /**
         * Implemented in mcts_chance_node.{h,cpp}
         */
        // public:
        //     bool is_two_player_game() const;
        //     bool is_opponent() const;
        //     int get_num_children() const;

        //     bool has_child_node_itfc(std::shared_ptr<const Observation> observation) const;
        //     std::shared_ptr<MctsDNode> get_child_node_itfc(std::shared_ptr<const Observation> observation) const;

        //     std::string get_pretty_print_string(int depth) const;

        // private:
        //     void get_pretty_print_string_helper(std::stringstream& ss, int depth, int num_tabs) const;
    };
}