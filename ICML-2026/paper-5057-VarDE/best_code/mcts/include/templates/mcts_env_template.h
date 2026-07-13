/**
 * Template for MctsEnv subclasses, because it involves some boilerplate code that will generally look the same.
 * 
 * To use the template, copy the relevant sections into your .h and .cpp files, and make the following find and replace
 * operations:
 *      _Env -> YourEnvClass
 *      _Context -> YourMctsEnvContextClassName     (MctsEnvContext unless you know you need something more)
 *      _S -> YourStateClass                        (your custom state class, or predefined state class from mcts_types.h)
 *      _A -> YourActionClass                       (your custom action class, or predefined action class from mcts_types.h)
 *      _O -> YourObservationClass                  (if fully observable, then == YourStateClass)
 * 
 * Finally, complete all of the TODO comments inline.
 */

/**
 * -----------------------------------
 * .h template - copy into .h file
 * -----------------------------------
 */

#pragma once

#include "mcts_env.h"
#include "mcts_env_context.h"
#include "mcts_manager.h"
#include "mcts_types.h"

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

// TODO: any other includes needed (e.g. the .h file for YourMctsManagerClassName)

namespace mcts{
    // TODO: delete these forward declarations (added to stop IDEs showing compile errors).
    class _Context;
    class _S;
    class _A;
    class _O;

    // TODO: delete these if not novel State/Action/Observation classes
    typedef std::unordered_map<std::shared_ptr<const _S>,double> _SDistr;
    typedef std::vector<std::shared_ptr<const _A>> _AVector;
    typedef std::unordered_map<std::shared_ptr<const _O>,double> _ODistr;

    /** 
     * TODO: write your docstring
     */
    class _Env : public MctsEnv {

        /**
         * Core _Env implementaion.
         */
        protected:
            /**
             * TODO: add your member variables here
             * TODO: add any additional member functinos here
             * (Change access modifiers as needed)
             */



        /**
         * Core MctsEnv implementation functinos.
         */
        public:
            /**
             * Constructor
             */
            _Env(bool is_fully_observable);

            /**
             * Mark destructor as virtual for subclassing.
             */
            virtual ~_Env() = default;

            /**
             * Returns the initial state for the environment.
             * 
             * Returns:
             *      Initial state for this environment instance
             */
            std::shared_ptr<const _S> get_initial_state() const;

            /**
             * Returns if a state is a sink state.
             * 
             * Args:
             *      state: The state to be checked if it is a sink state
             * 
             * Returns:
             *      True if 'state' is a sink state and false otherwise
             */
            bool is_sink_state(std::shared_ptr<const _S> state) const;

            /**
             * Returns a list of actions that are valid in a given state.
             * 
             * Args:
             *      state: The state that we want a list of available actions from
             * 
             * Returns:
             *      Returns a list of actions available from 'state'
             */
            std::shared_ptr<_AVector> get_valid_actions(std::shared_ptr<const _S> state) const;

            /**
             * Returns a distribution over successor states from a state action pair.
             * 
             * Given a state and action returns a distribution of possible successor states. The probability 
             * distribution is returned in the form of a map, where the keys are of the State type, and the values are 
             * doubles, which sum to one.
             * 
             * Args:
             *      state: The state to get a transition distribution from
             *      action: The action to get a transition distribution for
             * 
             * Returns:
             *      Returns a successor state distribution from taking 'action' in state 'state'.
             */
            std::shared_ptr<_SDistr> get_transition_distribution(
                std::shared_ptr<const _S> state, std::shared_ptr<const _A> action) const;

            /**
             * Samples an successor state when taking an action from a state.
             * 
             * Given a state, action pair, samples a possible successor state that can arrise.
             * 
             * Args:
             *      state: The state to sample an observation from
             *      action: The action taken to sample an observation for
             *      rand_manager: A RandManager ref to access the random number sampling interface
             * 
             * Returns:
             *      Returns an successor state sampled from taking 'action' from 'state'
             */
            std::shared_ptr<const _S> sample_transition_distribution(
                std::shared_ptr<const _S> state, 
                std::shared_ptr<const _A> action, 
                RandManager& rand_manager) const;
            
            /**
             * Returns the reward for a given state, action, observation tuple.
             * 
             * Commonly the reward is written as a function of just the state and action pair. But we provide the 
             * option to depend on the observation too. 
             * 
             * Args:
             *      state: The current state to get a reward for
             *      action: The action taken to get a reward for
             *      observation: 
             *          The (optional) observation sampled from the state, action pair that can optionally be used as 
             *          part of the reward function.
             * 
             * Returns:
             *      The reward for taking 'action' from 'state' (and sampling 'observation')
             */
            double get_reward(
                std::shared_ptr<const _S> state, 
                std::shared_ptr<const _A> action, 
                std::shared_ptr<const _O> observation=nullptr) const;



        /**
         * Boilerplate functinos (defined in mcts_env_template.{h,cpp}) using the default implementations provided by 
         * mcts_env.{h,cpp}. 
         * 
         * TODO: decide if need to override any of these (yes if want a partially observable environment, or want a 
         * custom MctsEnvContext object).
         */
        public:
            /**
             * Returns a distribution over observations from a (next) state, action pair.
             * 
             * Given a state and action returns a distribution of possible observations. The probability 
             * distribution is returned in the form of a map, where the keys are of the Observation type, and the 
             * values are doubles, which sum to one.
             * 
             * A default implementation is provided for full observable environments, where observation == next state.
             * 
             * Args:
             *      action: The action to get an observation distribution for
             *      next_state: The state (arriving in)  to get an observation distribution from
             * 
             * Returns:
             *      Returns a distribution over observations from taking 'action' in state 'state'.
             */
            virtual std::shared_ptr<_ODistr> get_observation_distribution(
                std::shared_ptr<const _A> action, std::shared_ptr<const _S> next_state) const;

            /**
             * Samples an observation when arriving in a (next) state after taking an action.
             * 
             * Given a state-action pair, samples a possible sobservation.
             * 
             * A default implementation is provided for full observable environments, where observation == next state.
             * 
             * Args:
             *      action: The action taken to sample an observation for
             *      next_state: The state (arriving in)  to sample an observation for
             *      rand_manager: A RandManager ref to access the random number sampling interface
             * 
             * Returns:
             *      Returns an observation sampled from taking 'action' that arived in 'next_state'
             */
            virtual std::shared_ptr<const _O> sample_observation_distribution(
                std::shared_ptr<const _A> action, 
                std::shared_ptr<const _S> next_state, 
                RandManager& rand_manager) const;

            /**
             * Samples a context that can be used to store information throughout a single trial.
             * 
             * Sometimes it is useful to place each trial in some sort of context, or a context can be used to cache 
             * information that doesn't need to be stored in the tree search permenantly, but is useful computationally. 
             * This function generates a context to be used. Most of the time it will be something like an empty map. 
             * 
             * Args:
             *      state: The initial state
             * 
             * Returns:
             *      A MctsEnvContext object, that will be passed to the Mcts functions for a single trial, used to 
             *      provide some context or space for caching.
             */
            virtual std::shared_ptr<_Context> sample_context(std::shared_ptr<const _S> state) const;



        /**
         * MctsEnv interface function definitions. Boilerplate implementations provided from mcts_env_template.h
         */
        public:
            virtual std::shared_ptr<const State> get_initial_state_itfc() const;
            virtual bool is_sink_state_itfc(std::shared_ptr<const State> state) const;
            virtual std::shared_ptr<ActionVector> get_valid_actions_itfc(std::shared_ptr<const State> state) const;
            virtual std::shared_ptr<StateDistr> get_transition_distribution_itfc(
                std::shared_ptr<const State> state, std::shared_ptr<const Action> action) const;
            virtual std::shared_ptr<const State> sample_transition_distribution_itfc(
                std::shared_ptr<const State> state, 
                std::shared_ptr<const Action> action, 
                 RandManager& rand_manager) const;
            virtual std::shared_ptr<ObservationDistr> get_observation_distribution_itfc(
                std::shared_ptr<const Action> action, std::shared_ptr<const State> next_state) const;
            virtual std::shared_ptr<const Observation> sample_observation_distribution_itfc(
                std::shared_ptr<const Action> action, 
                std::shared_ptr<const State> next_state, 
                 RandManager& rand_manager) const;
            virtual double get_reward_itfc(
                std::shared_ptr<const State> state, 
                std::shared_ptr<const Action> action, 
                std::shared_ptr<const Observation> observation=nullptr) const;
            virtual std::shared_ptr<MctsEnvContext> sample_context_itfc(std::shared_ptr<const State> state) const;
        
        /**
         * Implemented in mcts_env.{h,cpp}
         */
        // public:
        //     bool is_fully_observable();
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
    _Env::_Env(bool is_fully_observable) : MctsEnv(is_fully_observable) {

    }

    shared_ptr<const _S> _Env::get_initial_state() const {
        return nullptr;
    }

    bool _Env::is_sink_state(shared_ptr<const _S> state) const {
        return false;
    }

    shared_ptr<_AVector> _Env::get_valid_actions(shared_ptr<const _S> state) const {
        return nullptr;
    }

    shared_ptr<_SDistr> _Env::get_transition_distribution(
        shared_ptr<const _S> state, shared_ptr<const _A> action) const 
    {
        return nullptr;
    }

    shared_ptr<const _S> _Env::sample_transition_distribution(
        shared_ptr<const _S> state, shared_ptr<const _A> action, RandManager& rand_manager) const 
    {
        return nullptr;
    }

    double _Env::get_reward(
        shared_ptr<const _S> state, 
        shared_ptr<const _A> action, 
        shared_ptr<const _O> observation) const 
    {
        return 0.0;
    }
}



/**
 * Boilerplate defined functions. Copied from mcts_env_template.h.
 * 
 * TODO: decide if need to write a custom version of these depending on if need partial observability or if need 
 * custom contexts.
 */
namespace mcts {
    shared_ptr<_ODistr> _Env::get_observation_distribution(
        shared_ptr<const _A> action, shared_ptr<const _S> next_state) const 
    {
        shared_ptr<const Action> act_itfc = static_pointer_cast<const Action>(action);
        shared_ptr<const State> next_state_itfc = static_pointer_cast<const State>(next_state);
        shared_ptr<ObservationDistr> distr_itfc = MctsEnv::get_observation_distribution_itfc(
            act_itfc, next_state_itfc);
        shared_ptr<_ODistr> distr;
        for (pair<const shared_ptr<const Observation>,double> pr : *distr_itfc) {
            shared_ptr<const _O> obsv = static_pointer_cast<const _O>(pr.first);
            distr->insert_or_assign(obsv, pr.second);
        }
        return distr;
    }

    shared_ptr<const _O> _Env::sample_observation_distribution(
        shared_ptr<const _A> action, 
        shared_ptr<const _S> next_state, 
        RandManager& rand_manager) const 
    {
        shared_ptr<const Action> act_itfc = static_pointer_cast<const Action>(action);
        shared_ptr<const State> next_state_itfc = static_pointer_cast<const State>(next_state);
        shared_ptr<const Observation> obsv_itfc = MctsEnv::sample_observation_distribution_itfc(
            act_itfc, next_state_itfc, rand_manager);
        return static_pointer_cast<const _O>(obsv_itfc);
    }

    shared_ptr<_Context> _Env::sample_context(shared_ptr<const _S> state) const
    {
        shared_ptr<const State> state_itfc = static_pointer_cast<const State>(state);
        shared_ptr<MctsEnvContext> context = MctsEnv::sample_context_itfc(state_itfc);
        return static_pointer_cast<_Context>(context);
    }
}



/**
 * Boilerplate MctsEnv interface implementation. Copied from mcts_env_template.h.
 * All this code basically calls the corresponding implementation function, with approprtiate casts before/after.
 */
namespace mcts {
    
    shared_ptr<const State> _Env::get_initial_state_itfc() const {
        shared_ptr<const _S> init_state = get_initial_state();
        return static_pointer_cast<const State>(init_state);
    }

    bool _Env::is_sink_state_itfc(shared_ptr<const State> state) const {
        shared_ptr<const _S> state_itfc = static_pointer_cast<const _S>(state);
        return is_sink_state(state_itfc);
    }

    shared_ptr<ActionVector> _Env::get_valid_actions_itfc(shared_ptr<const State> state) const {
        shared_ptr<const _S> state_itfc = static_pointer_cast<const _S>(state);
        shared_ptr<vector<shared_ptr<const _A>>> valid_actions_itfc = get_valid_actions(state_itfc);

        shared_ptr<ActionVector> valid_actions = make_shared<ActionVector>();
        for (shared_ptr<const _A> act : *valid_actions_itfc) {
            valid_actions->push_back(static_pointer_cast<const Action>(act));
        }
        return valid_actions;
    }

    shared_ptr<StateDistr> _Env::get_transition_distribution_itfc(
        shared_ptr<const State> state, shared_ptr<const Action> action) const 
    {
        shared_ptr<const _S> state_itfc = static_pointer_cast<const _S>(state);
        shared_ptr<const _A> action_itfc = static_pointer_cast<const _A>(action);
        shared_ptr<_SDistr> distr_itfc = get_transition_distribution(state_itfc, action_itfc);
        
        shared_ptr<StateDistr> distr = make_shared<StateDistr>(); 
        for (pair<shared_ptr<const _S>,double> key_val_pair : *distr_itfc) {
            shared_ptr<const State> obsv = static_pointer_cast<const State>(key_val_pair.first);
            double prob = key_val_pair.second;
            distr->insert_or_assign(obsv, prob);
        }
        return distr;
    }

    shared_ptr<const State> _Env::sample_transition_distribution_itfc(
       shared_ptr<const State> state, shared_ptr<const Action> action, RandManager& rand_manager) const 
    {
        shared_ptr<const _S> state_itfc = static_pointer_cast<const _S>(state);
        shared_ptr<const _A> action_itfc = static_pointer_cast<const _A>(action);
        shared_ptr<const _S> obsv = sample_transition_distribution(state_itfc, action_itfc, rand_manager);
        return static_pointer_cast<const State>(obsv);
    }

    shared_ptr<ObservationDistr> _Env::get_observation_distribution_itfc(
        shared_ptr<const Action> action, shared_ptr<const State> next_state) const
    {
        shared_ptr<const _A> act_itfc = static_pointer_cast<const _A>(action);
        shared_ptr<const _S> next_state_itfc = static_pointer_cast<const _S>(next_state);
        shared_ptr<_ODistr> distr_itfc = get_observation_distribution(
            act_itfc, next_state_itfc);
        shared_ptr<ObservationDistr> distr;
        for (pair<const shared_ptr<const _O>,double> pr : *distr_itfc) {
            shared_ptr<const Observation> obsv = static_pointer_cast<const Observation>(pr.first);
            distr->insert_or_assign(obsv, pr.second);
        }
        return distr;
    }     

    shared_ptr<const Observation> _Env::sample_observation_distribution_itfc(
        shared_ptr<const Action> action, 
        shared_ptr<const State> next_state,
         RandManager& rand_manager) const
    {
        shared_ptr<const _A> act_itfc = static_pointer_cast<const _A>(action);
        shared_ptr<const _S> next_state_itfc = static_pointer_cast<const _S>(next_state);
        shared_ptr<const _O> obsv_itfc = sample_observation_distribution(
            act_itfc, next_state_itfc, rand_manager);
        return static_pointer_cast<const Observation>(obsv_itfc);
    }

    double _Env::get_reward_itfc(
        shared_ptr<const State> state, 
        shared_ptr<const Action> action, 
        shared_ptr<const Observation> observation) const
    {
        shared_ptr<const _S> state_itfc = static_pointer_cast<const _S>(state);
        shared_ptr<const _A> action_itfc = static_pointer_cast<const _A>(action);
        shared_ptr<const _O> obsv_itfc = static_pointer_cast<const _O>(observation);
        return get_reward(state_itfc, action_itfc, obsv_itfc);
    }

    shared_ptr<MctsEnvContext> _Env::sample_context_itfc(shared_ptr<const State> state) const
    {
        shared_ptr<const _S> state_itfc = static_pointer_cast<const _S>(state);
        shared_ptr<_Context> context = sample_context(state_itfc);
        return static_pointer_cast<MctsEnvContext>(context);
    }
}