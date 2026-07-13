/**
 * Template for MctsDNode subclasses, because it involves some boilerplate code that will generally look the same.
 * 
 * To use the template, copy the relevant sections into your .h and .cpp files, and make the following find and replace
 * operations:
 *      _DNode -> YourDNodeClass                
 *      _CNode -> YourCNodeClass                
 *      _Manager -> YourMctsManagerClass            (your subclass of MctsManager containing params for your alg)
 *      _Context -> YourMctsEnvContextClassName     (MctsEnvContext unless you know you need something more)
 *      _S -> YourStateClass                        (MctsState unless you want to restrict types of state that can be used)
 *      _A -> YourActionClass                       (MctsAction unless you want to restrict types of action that can be used)
 *      _O -> YourObservationClass                  (MctsState if fully observable, MctsObservation if partially observable
 *                                                   unless you want to restrict the types of observation that can be used)
 * 
 * Finally, complete all of the TODO comments inline.
 * 
 * Note that much of the boilerplate code can be deleted if you don't actually use it. I think the only strictly 
 * necessary functions are 'create_child_node_helper' and 'create_child_node_helper_itfc'. See Puct implementation for 
 * an example of being able to delete much of the code and rely on subclass code.
 */

/**
 * -----------------------------------
 * .h template - copy into .h file
 * -----------------------------------
 */

#pragma once

#include "mcts_chance_node.h"
#include "mcts_decision_node.h"
#include "mcts_env.h"
#include "mcts_env_context.h"
#include "mcts_manager.h"

#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>

// TODO: any other includes needed (e.g. the .h file for YourMctsManagerClassName)

namespace mcts {
    // TODO: delete these forward declarations (added to stop IDEs showing compile errors).
    class _S;
    class _A;
    class _O;
    class _Manager;
    class _Context;

    // forward declare corresponding _DNode class
    class _DNode;
    
    /**
     * TODO: abstract base class for CNodes, list of attributes.
     * TODO: write around parent = the parent that CONSTRUCTED this node
     * TODO: create a superclass MctsNode, that shoves all of the shared functionality of the decision and chance nodes together
     */
    class _CNode : public MctsCNode {
        // Allow MctsDNode access to private members
        friend _DNode;

        /**
         * Core _CNode implementation.
         */
        protected:
            /**
             * TODO: add your member variables here
             * TODO: add any additional member functions here 
             * (Change access modifiers as needed)
             */



        /**
         * Core MctsCNode implementation functions.
         */
        public: 
            /**
             * Constructor
             */
            _CNode(
                std::shared_ptr<_Manager> mcts_manager,
                std::shared_ptr<const _S> state,
                std::shared_ptr<const _A> action,
                int decision_depth,
                int decision_timestep,
                std::shared_ptr<const _DNode> parent=nullptr);

            /**
             * Implements the Mcts visit function for the node
             * 
             * Args:
             *      ctx: A context provided to all Mcts functions throughout a trial to pass intermediate/transient info
             */
            void visit(_Context& ctx);
            
            /**
             * Implements the Mcts sample_observation function for the node
             * 
             * Args:
             *      ctx: A context provided to all Mcts functions throughout a trial to pass intermediate/transient info
             * 
             * Returns:
             *      The sampled observation
             */
            std::shared_ptr<const _O> sample_observation(_Context& ctx);
            
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
            void backup(
                const std::vector<double>& trial_rewards_before_node, 
                const std::vector<double>& trial_rewards_after_node, 
                const double trial_cumulative_return_after_node, 
                const double trial_cumulative_return,
                _Context& ctx);

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
             *      A pointer to a new _DNode object
             */
            std::shared_ptr<_DNode> create_child_node_helper(
                std::shared_ptr<const _O> observation, std::shared_ptr<const _S> next_state=nullptr) const;

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
            virtual ~_CNode() = default;

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
            std::shared_ptr<_DNode> create_child_node(
                std::shared_ptr<const _O> observation, std::shared_ptr<const _S> next_state=nullptr);

            /**
             * If this node has a child object corresponding to 'observation'.
             * 
             * Args:
             *      observation: An observation to check if we have a child for
             * 
             * Returns:
             *      true if we have a child corresponding to 'observation'
             */
            bool has_child_node(std::shared_ptr<const _O> observation) const;

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
            std::shared_ptr<_DNode> get_child_node(std::shared_ptr<const _O> observation) const;



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






/**
 * -----------------------------------
 * .cpp template - copy into .cpp file
 * -----------------------------------
 */

// TODO: add include for your header file

using namespace std; 

/**
 * TODO: implement your class here.
 */
namespace mcts {
    _CNode::_CNode(
        shared_ptr<_Manager> mcts_manager,
        shared_ptr<const _S> state,
        shared_ptr<const _A> action,
        int decision_depth,
        int decision_timestep,
        shared_ptr<const _DNode> parent) :
            MctsCNode(
                static_pointer_cast<MctsManager>(mcts_manager),
                static_pointer_cast<const State>(state),
                static_pointer_cast<const Action>(action),
                decision_depth,
                decision_timestep,
                static_pointer_cast<const MctsDNode>(parent)) {}

    void _CNode::visit(_Context& ctx) {
        num_visits += 1;
    }

    shared_ptr<const _O> _CNode::sample_observation(_Context& ctx) {
        return nullptr;
    }

    void _CNode::backup(
        const std::vector<double>& trial_rewards_before_node, 
        const std::vector<double>& trial_rewards_after_node, 
        const double trial_cumulative_return_after_node, 
        const double trial_cumulative_return,
        _Context& ctx)
    {   
    }

    /**
     * POMDP TODO: make use of next_state parameter
     * N.B. for MDP, next_state=nullptr
     */ 
    shared_ptr<_DNode> _CNode::create_child_node_helper(
        shared_ptr<const _O> observation, shared_ptr<const _S> next_state) const 
    {  
        shared_ptr<const _S> mdp_next_state = static_pointer_cast<const _S>(observation);
        return make_shared<_DNode>(
            mcts_manager, 
            mdp_next_state,
            decision_depth+1, 
            decision_timestep+1, 
            static_pointer_cast<const _CNode>(shared_from_this()));
    }

    string _CNode::get_pretty_print_val() const {
        return "";
    }
}

/**
 * Boilerplate function definitions.
 * All this code basically calls the corresponding base implementation function, with approprtiate casts before/after.
 */
namespace mcts {
    shared_ptr<_DNode> _CNode::create_child_node(shared_ptr<const _O> observation, shared_ptr<const _S> next_state) {
        shared_ptr<const Observation> obsv_itfc = static_pointer_cast<const Observation>(observation);
        shared_ptr<const State> next_state_itfc = static_pointer_cast<const State>(next_state);
        shared_ptr<MctsDNode> new_child = MctsCNode::create_child_node_itfc(obsv_itfc, next_state_itfc);
        return static_pointer_cast<_DNode>(new_child);
    }

    bool _CNode::has_child_node(std::shared_ptr<const _O> observation) const {
        return MctsCNode::has_child_node_itfc(static_pointer_cast<const Observation>(observation));
    }
    
    shared_ptr<_DNode> _CNode::get_child_node(shared_ptr<const _O> observation) const {
        shared_ptr<const Observation> obsv_itfc = static_pointer_cast<const Observation>(observation);
        shared_ptr<MctsDNode> new_child = MctsCNode::get_child_node_itfc(obsv_itfc);
        return static_pointer_cast<_DNode>(new_child);
    }
}

/**
 * Boilerplate MctsCNode interface implementation. Copied from mcts_chance_node_template.h.
 */
namespace mcts {
    void _CNode::visit_itfc(MctsEnvContext& ctx) {
        _Context& ctx_itfc = (_Context&) ctx;
        visit(ctx_itfc);
    }

    shared_ptr<const Observation> _CNode::sample_observation_itfc(MctsEnvContext& ctx) {
        _Context& ctx_itfc = (_Context&) ctx;
        shared_ptr<const _O> obsv = sample_observation(ctx_itfc);
        return static_pointer_cast<const Observation>(obsv);
    }

    void _CNode::backup_itfc(
        const vector<double>& trial_rewards_before_node, 
        const vector<double>& trial_rewards_after_node, 
        const double trial_cumulative_return_after_node, 
        const double trial_cumulative_return,
        MctsEnvContext& ctx) 
    {
        _Context& ctx_itfc = (_Context&) ctx;
        backup(
            trial_rewards_before_node, 
            trial_rewards_after_node, 
            trial_cumulative_return_after_node, 
            trial_cumulative_return, 
            ctx_itfc);
    }

    shared_ptr<MctsDNode> _CNode::create_child_node_helper_itfc(
        shared_ptr<const Observation> observation, shared_ptr<const State> next_state) const 
    {
        shared_ptr<const _O> obsv_itfc = static_pointer_cast<const _O>(observation);
        shared_ptr<const _S> next_state_itfc = static_pointer_cast<const _S>(next_state);
        shared_ptr<_DNode> child_node = create_child_node_helper(obsv_itfc, next_state_itfc);
        return static_pointer_cast<MctsDNode>(child_node);
    }
}