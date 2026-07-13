#include "mcts_env.h"

#include "helper_templates.h"

#include <stdexcept>

using namespace std;

namespace mcts {
    /**
     * Constructor
     */
    MctsEnv::MctsEnv(bool is_fully_observable) : _is_fully_observable(is_fully_observable) {}

    /**
     * Is fully observable getter
     */
    bool MctsEnv::is_fully_observable() {
        return _is_fully_observable;
    }

    /**
     * Default implmentation of 'get_observation_distribution_itfc'.
     * 
     * Just casts the next_state into an observation object, and returns it as a delta distribution.
     */
    shared_ptr<ObservationDistr> MctsEnv::get_observation_distribution_itfc(
        shared_ptr<const Action> action, shared_ptr<const State> next_state) const 
    {
        if (_is_fully_observable) {
            shared_ptr<ObservationDistr> distr = make_shared<ObservationDistr>();
            shared_ptr<const Observation> obsv = static_pointer_cast<const Observation>(next_state);
            distr->insert_or_assign(obsv, 1.0);
            return distr;
        }
        throw runtime_error("Attempting to use non-existant default get_observation_distribution for POMDP envs");
    }

    /**
     * Default implmentation of 'sample_observation_distribution_itfc'.
     * 
     * Just casts the next_state into an observation object, and returns it.
     * 
     * If not fully observable, samples.
     */
    shared_ptr<const Observation> MctsEnv::sample_observation_distribution_itfc(
        shared_ptr<const Action> action, 
        shared_ptr<const State> next_state, 
        RandManager& rand_manager) const 
    {
        if (_is_fully_observable) {
            return static_pointer_cast<const Observation>(next_state);
        }
        shared_ptr<ObservationDistr> distr = get_observation_distribution_itfc(action, next_state);
        return helper::sample_from_distribution(*distr, rand_manager);
    }

    /**
     * Default implementation of 'sample_context'
     * 
     * Returns an (default constructed) MctsEnvContext, which is really just a wrapper around an empty map. It's useful 
     * to return this type so we can subclass it, rather than forcing Mcts algorithms to use a specific map for a 
     * context.
     */
    shared_ptr<MctsEnvContext> MctsEnv::sample_context_itfc(shared_ptr<const State> state) const {
        return make_shared<MctsEnvContext>();
    }
} 