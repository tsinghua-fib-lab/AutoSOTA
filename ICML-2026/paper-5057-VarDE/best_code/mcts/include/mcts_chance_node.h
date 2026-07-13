#pragma once

#include "mcts_decision_node.h"
#include "mcts_manager.h"

#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>

namespace mcts {
    // forward declare
    class MctsDNode;

    // CNodeMap type is lengthy, so typedef
    typedef std::unordered_map<std::shared_ptr<const Observation>,std::shared_ptr<MctsDNode>> DNodeChildMap;
    
    /**
     * An abstract base class for Chance Nodes.
     * 
     * This class provides some base implementations that can be useful across different Mcts algorithms. Including 
     * a transposition table implementation and pretty print functions for debugging.
     * 
     * Member variables:
     *      node_lock: 
     *          A mutex that is used to protect this entire node.
     *      mcts_manager: 
     *          A MctsManager object that stores the 'global' information about how the Mcts algorithm should operate,
     *          so that an implementation can provide multiple modes of operation. Additionally stores the 
     *          transposition tables
     *      state:
     *          The state associated with this node
     *      action:
     *          The action associated with this node
     *      decision_depth:
     *          The decision depth of the node in this tree
     *      decision_timestep:
     *          The timestep corresponding to the current state in the larger planning problem. This is necessary 
     *          when the Mcts algorithm is used at each timestep to make a decision. For example, this is necessary in 
     *          a two player game to decide who's turn it is
     *      num_visits:
     *          The number of times the node has been visited (had the 'visit' function called)
     *      parent:
     *          A pointer to this nodes parent node. nullptr if this node is the root node
     *      children:
     *          A map from Action objects to child MctsCNode objects
     */
    class MctsCNode : public std::enable_shared_from_this<MctsCNode> {
        // Allow MctsDNode access to private members
        friend MctsDNode;

        protected:
            std::mutex node_lock;

            std::shared_ptr<MctsManager> mcts_manager;
            std::shared_ptr<const State> state;
            std::shared_ptr<const Action> action;
            int decision_depth;
            int decision_timestep;
            std::weak_ptr<const MctsDNode> parent;

            int num_visits;
            DNodeChildMap children;

        public: 
            /**
             * Default constructor.
             * 
             * Initialises the attributes of the class.
             */
            MctsCNode(
                std::shared_ptr<MctsManager> mcts_manager,
                std::shared_ptr<const State> state,
                std::shared_ptr<const Action> action,
                int decision_depth,
                int decision_timestep,
                std::shared_ptr<const MctsDNode> parent=nullptr);

            /**
             * Mark destructor as virtual for subclassing.
             */
            virtual ~MctsCNode() = default;

            /**
             * Aquires the lock for this node.
             */
            void lock();

            /**
             * Releases the lock for this node.
             */
            void unlock();

            /**
             * Gets a reference to the lock for this node (so can use in a lock_guard for example)
             */
            std::mutex& get_lock();

            /**
             * Helper function to lock all children nodes.
             */
            void lock_all_children() const;

            /**
             * Helper function to unlock all children nodes.
             */
            void unlock_all_children() const;

            /**
             * Mcts visit function.
             * 
             * Called everytime the Mcts routine selects this node.
             * 
             * Args:
             *      ctx: object holding context information for the current trial   
             */
            virtual void visit_itfc(MctsEnvContext& ctx);

            /**
             * The sample observation function. Selects an observation to explore from this node.
             * 
             * Used int he selection phase of the Mcts routine to select observations at this node.
             * 
             * Args:
             *      ctx: object holding context information for the current trial 
             *      
             * Returns:
             *      The sampled observation
             */
            virtual std::shared_ptr<const Observation> sample_observation_itfc(MctsEnvContext& ctx) = 0;

            /**
             * Mcts backup function.
             * 
             * Updates the information in this node in the backup phase of the Mcts routine.
             * 
             * Args:
             *      trial_rewards_before_node: 
             *          A list of rewards recieved (at each timestep) on the trial prior to reaching this node.
             *      trial_rewards_after_node:
             *          A list of rewards recieved (at each timestep) on the trial after reaching this node. This list 
             *          includes the reward from R(state,action) that would have been recieved from taking the action 
             *          in this node.
             *      trial_cumulative_return_after_node:
             *          Sum of rewards in the 'trial_rewards_after_node' list
             *      trial_cumulative_return:
             *          Sum of rewards in both of the 'trial_rewards_after_node' and 'trial_rewards_before_node' lists
             */
            virtual void backup_itfc(
                const std::vector<double>& trial_rewards_before_node, 
                const std::vector<double>& trial_rewards_after_node, 
                const double trial_cumulative_return_after_node, 
                const double trial_cumulative_return,
                MctsEnvContext& ctx) = 0;

            /**
             * Creates a child node and inserts it in the unordered_map 'children'.
             * 
             * This virtual final method means that this implementation cannot be overriden. This is to protect the 
             * logic surrounding the transposition table, which is found in the 'mcts_manager' object. It will perform 
             * the following logic:
             *      - if not using transposition table:
             *          - make child node using 'create_child_node_helper' and insert in children map
             *      - if using transposition table:
             *          - check transposition table for child node, if it exists, adds to children map and returns
             *          - otherwise creates the child node, and inserts it into the children map and transposition table
             * 
             * Args:
             *      observation: The observation object leading to the child node
             *      next_state: The next state to construct the child node with
             * 
             * Returns:
             *      A pointer to the created child node
             */
            virtual std::shared_ptr<MctsDNode> create_child_node_itfc(
                std::shared_ptr<const Observation> observation, std::shared_ptr<const State> next_state=nullptr) final;

        protected:
            /**
             * Helper for constructing a child node. Should create and instance and return a pointer to it.
             * 
             * Args:
             *      observation: The observation object leading to the child node
             *      next_state: The next state to construct the child node with
             * 
             * Returns:
             *      A pointer to a newly created child node on the heap
             */
            virtual std::shared_ptr<MctsDNode> create_child_node_helper_itfc(
                std::shared_ptr<const Observation> observation, 
                std::shared_ptr<const State> next_state=nullptr) const = 0;

            /**
             * Helper for pretty printing. Should return some string representing the current 'value' of this node.
             * 
             * Returns:
             *      string representing the current value of this node
             */
            virtual std::string get_pretty_print_val() const = 0;

        public:
            /**
             * Returns if this node is planning for a two player game.
             * 
             * Returns:
             *      True if currently planning for a two player game
             */
            bool is_two_player_game() const;

            /**
             * Returns if this node is planning as the opponent in a two player game.
             * 
             * If not a two player game, this will always return false.
             * 
             * Returns:
             *      If this node is planning as the opponent in a two player game
             */
            bool is_opponent() const;

            /**
             * Helper function to get number of children this node currently has.
             * 
             * Virtual so can be mocked in tests.
             * 
             * Returns:
             *      Number of children in 'children' map
             */
            virtual int get_num_children() const;

            /**
             * Helper function to check if node has a child for the given observation
             * 
             * Args:
             *      observation: The observation to check if we have a child for
             * 
             * Returns:
             *      Returns true if node has a child corresponding to 'observation'
             */
            bool has_child_node_itfc(std::shared_ptr<const Observation> observation) const;

            /**
             * Returns a pointer to a child of this node.
             * 
             * Virtual so can be mocked in tests.
             * 
             * Args:
             *      observation: The observation that we want the corresponding child node for.
             * 
             * Returns:
             *      A pointer to the child decision node.
             */
            virtual std::shared_ptr<MctsDNode> get_child_node_itfc(
                std::shared_ptr<const Observation> observation) const;

            /**
             * Pretty prints the tree to a string.
             * 
             * Args:
             *      depth: To what (decision) depth do we want to print to?
             * 
             * Returns:
             *      A string that is a pretty representation of the top part of the tree, rooted at this node
             */
            std::string get_pretty_print_string(int depth) const;

        private:
            /**
             * A helper function that actually implements 'get_pretty_pring_string' above.
             */
            void get_pretty_print_string_helper(std::stringstream& ss, int depth, int num_tabs) const;
    };
}