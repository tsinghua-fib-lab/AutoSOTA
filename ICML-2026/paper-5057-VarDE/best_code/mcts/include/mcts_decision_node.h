#pragma once

#include "mcts_chance_node.h"
#include "mcts_env.h"
#include "mcts_manager.h"

#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace mcts {
    // forward declare
    class MctsCNode;
    class MctsLogger;
    class MctsPool;

    // CNodeMap type is lengthy, so typedef
    typedef std::unordered_map<std::shared_ptr<const Action>,std::shared_ptr<MctsCNode>> CNodeChildMap;

    /**
     * An abstract base class for Decision Node.
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
     *          The state associated with this node, which we want to make a decision for (what is the best action)
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
     *      heuristic_value:
     *          The heuristic value of this decision node
     */
    class MctsDNode : public std::enable_shared_from_this<MctsDNode> {
        // Allow MctsCNode, Logger and Pool access to private members
        friend MctsCNode;
        friend MctsLogger;
        friend MctsPool;

        protected:
            std::mutex node_lock;

            std::shared_ptr<MctsManager> mcts_manager;
            std::shared_ptr<const State> state;
            int decision_depth;
            int decision_timestep;
            std::weak_ptr<const MctsCNode> parent;

            int num_visits;
            CNodeChildMap children;

            double heuristic_value;

        public: 
            /**
             * Constructor.
             * 
             * Initialises the attributes of the class.
             */
            MctsDNode(
                std::shared_ptr<MctsManager> mcts_manager,
                std::shared_ptr<const State> state,
                int decision_depth,
                int decision_timestep,
                std::shared_ptr<const MctsCNode> parent=nullptr); 

            /**
             * Mark destructor as virtual for subclassing.
             */
            virtual ~MctsDNode() = default;

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
             * Mcts select action function. Selects an action to explore fromt this node
             * 
             * Used in the selection phase of the Mcts routine to select actions at this node.
             * 
             * Args:
             *      ctx: object holding context information for the current trial 
             *      
             * Returns:
             *      The selected action
             */
            virtual std::shared_ptr<const Action> select_action_itfc(MctsEnvContext& ctx) = 0;

            /**
             * Recommends an action from this node.
             * 
             * Recommends what this node considers to be the best action to take from its current state.
             * 
             * Args:
             *      ctx: object holding context information for the current trial 
             * 
             * Returns:
             *      The recommended action
             */
            virtual std::shared_ptr<const Action> recommend_action_itfc(MctsEnvContext& ctx) const = 0;

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
             *          includes the reward from R(state,action) that would have been recieved from taking an action 
             *          from this node.
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
             * Returns if the node is a sink node in the environment.
             * 
             * Used to decide if this node is (and always will be) a leaf of the tree (it has no possible nodes that 
             * can be expanded).
             * 
             * Returns:
             *      If this node corresponds to a 'sink state' in the environment
             */
            virtual bool is_sink() const;

            /**
             * Returns if the node is a leaf node.
             * 
             * A node can be a leaf in one of two ways. Firstly if it is a sink state in the environment, or, secondly 
             * if it is at the maximum decision depth for the tree search.
             * 
             * Returns:
             *      If this node is a leaf node for the tree search. 
             */
            bool is_leaf() const;

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
             *      action: The action to create a child node for 
             * 
             * Returns:
             *      A pointer to the created child node
             */
            virtual std::shared_ptr<MctsCNode> create_child_node_itfc(std::shared_ptr<const Action> action) final;

        protected:
            /**
             * Helper for constructing a child node. Should create and instance and return a pointer to it.
             * 
             * Args:
             *      action: The action to create a child node for 
             * 
             * Returns:
             *      A pointer to a newly created child node on the heap
             */
            virtual std::shared_ptr<MctsCNode> create_child_node_helper_itfc(
                std::shared_ptr<const Action> action) const = 0;

            /**
             * Helper for pretty printing. Should return some string representing the current 'value' of this node.
             * 
             * Returns:
             *      string representing the current value of this node
             */
            virtual std::string get_pretty_print_val() const = 0;

        public:
            /**
             * Returns if this node is the root node of the tree.
             * 
             * Returns:
             *      True if current node is the root node
             */
            bool is_root_node() const;

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
             * Virtual so it can be mocked in tests.
             * 
             * Returns:
             *      If this node is planning as the opponent in a two player game
             */
            virtual bool is_opponent() const;

            /**
             * Gets the number of times that the node has been visited.
             * 
             * Returns:
             *      The number of times this node has been visited.
            */
            int get_num_visits() const;  

            /**
             * Helper function to get number of children this node currently has.
             * 
             * Virtual so it can be mocked in tests.
             * 
             * Returns:
             *      Number of children in 'children' map
             */
            virtual int get_num_children() const;

            /**
             * Helper function to check if node has a child for the given action
             * 
             * Args:
             *      action: The action to check if we have a child for
             * 
             * Returns:
             *      Returns true if node has a child corresponding to 'action'
             */
            bool has_child_node_itfc(std::shared_ptr<const Action> action) const;

            /**
             * Returns a pointer to a child of this node.
             * 
             * Virtual so it can be mocked in tests.
             * 
             * Args:
             *      action: The action that we want the corresponding child node for.
             * 
             * Returns:
             *      A pointer to the child chance node.
             */
            virtual std::shared_ptr<MctsCNode> get_child_node_itfc(std::shared_ptr<const Action> action) const;

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

            /**
             * Loads a tree from a given filename.
             * 
             * Args:
             *      filename: The filename to look for a tree file at
             * 
             * Returns:
             *      A MctsDNode which is the root node of a Mcts tree
             */
            static std::shared_ptr<MctsDNode> load(std::string& filename);

            /**
             * Saves the tree to a given filename.
             * 
             * Args:
             *      filename: The filename to save this tree as an object to
             * 
             * Returns:
             *      True if saving was successful.
             */
            bool save(std::string& filename) const;

        private:
            /**
             * A helper function that actually implements 'get_pretty_pring_string' above.
             */
            void get_pretty_print_string_helper(std::stringstream& ss, int depth, int num_tabs) const;
    };
}