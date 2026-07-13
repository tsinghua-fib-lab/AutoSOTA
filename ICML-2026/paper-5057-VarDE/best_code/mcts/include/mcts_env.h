#pragma once

#include "mcts_env_context.h"
#include "mcts_manager.h"
#include "mcts_types.h"

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace mcts {
    // Forward declare
    class RandManager;
    
    /**
     * An abstract class for representing an environment.
     * 
     * Defines a set of functions that should be general enough to handle a range of environments for planning, like 
     * MDPs and POMDPs. 
     * 
     * Additionally, any subclasses that wish to plan using a transposition table should implement the std::hash and 
     * std::equal_to functions. 
     * 
     * Uses the State, Action and Observation objects from mcts_types.h, as base types for the following:
     *      State: Objects representing the current state of the world
     *      Action: Objects representing the actions that an agent can take
     *      Observation: 
     *          Objects representing the outcomes that can occur from taking actions in the environment. For something 
     *          like an MDP, we would probably want Observation's to be the successor states (which would require 
     *          Observation == State subtypes)
     * 
     * Member variables:
     *      _is_fully_observable: 
     *          A boolean representing if the environment is fully observable or not. If true, then it should be safe 
     *          to assume that the default implementations of 'get_observation_distribution_itfc' and 
     *          'sample_observation_distribution_itfc' are used (the ones that just cast the state object into an 
     *          observation object)
     */
    class MctsEnv {
        protected:
            bool _is_fully_observable;

        public:
            /**
             * Constructor
             */
            MctsEnv(bool is_fully_observable);

            /**
             * Mark destructor virtual in case class is inherited from
             */
            virtual ~MctsEnv() = default;

            /**
             * Getter for checking if environment is fully observable or not
             */
            bool is_fully_observable();

            /**
             * Returns the initial state for the environment.
             * 
             * Returns:
             *      Initial state for this environment instance
             */
            virtual std::shared_ptr<const State> get_initial_state_itfc() const = 0;

            /**
             * Returns if a state is a sink state.
             * 
             * Args:
             *      state: The state to be checked if it is a sink state
             * 
             * Returns:
             *      True if 'state' is a sink state and false otherwise
             */
            virtual bool is_sink_state_itfc(std::shared_ptr<const State> state) const = 0;

            /**
             * Returns a list of actions that are valid in a given state.
             * 
             * Args:
             *      state: The state that we want a list of available actions from
             * 
             * Returns:
             *      Returns a list of actions available from 'state'
             */
            virtual std::shared_ptr<ActionVector> get_valid_actions_itfc(
                std::shared_ptr<const State> state) const = 0;

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
            virtual std::shared_ptr<StateDistr> get_transition_distribution_itfc(
                std::shared_ptr<const State> state, std::shared_ptr<const Action> action) const = 0;

            /**
             * Samples an successor state when taking an action from a state.
             * 
             * Given a state, action pair, samples a possible successor state that can arrise.
             * 
             * Args:
             *      state: The state to sample an observation from
             *      action: The action taken to sample an observation for
             *      rand_manager: A pointer to a RandManager to access the random number sampling interface
             * 
             * Returns:
             *      Returns an successor state sampled from taking 'action' from 'state'
             */
            virtual std::shared_ptr<const State> sample_transition_distribution_itfc(
                std::shared_ptr<const State> state, 
                std::shared_ptr<const Action> action, 
                RandManager& rand_manager) const = 0;

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
            virtual std::shared_ptr<ObservationDistr> get_observation_distribution_itfc(
                std::shared_ptr<const Action> action, std::shared_ptr<const State> next_state) const;

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
             *      rand_manager: A pointer to a RandManager to access the random number sampling interface
             * 
             * Returns:
             *      Returns an observation sampled from taking 'action' that arived in 'next_state'
             */
            virtual std::shared_ptr<const Observation> sample_observation_distribution_itfc(
                std::shared_ptr<const Action> action, 
                std::shared_ptr<const State> next_state, 
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
            virtual double get_reward_itfc(
                std::shared_ptr<const State> state, 
                std::shared_ptr<const Action> action, 
                std::shared_ptr<const Observation> observation=nullptr) const = 0;

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
            virtual std::shared_ptr<MctsEnvContext> sample_context_itfc(std::shared_ptr<const State> state) const;
    };
}