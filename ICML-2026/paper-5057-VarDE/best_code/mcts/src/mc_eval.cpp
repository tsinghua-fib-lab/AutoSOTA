#include "mc_eval.h"

#include <cmath>
#include <thread>

using namespace std;

/**
 * Eval policy implementation
*/
namespace mcts {
    EvalPolicy::EvalPolicy(
        shared_ptr<const MctsDNode> root_node, shared_ptr<const MctsEnv> mcts_env, RandManager& rand_manager) :
            root_node(root_node), cur_node(root_node), mcts_env(mcts_env), rand_manager(rand_manager) {}

    EvalPolicy::EvalPolicy(const EvalPolicy& policy) :
        root_node(policy.root_node), 
        cur_node(policy.root_node), 
        mcts_env(policy.mcts_env), 
        rand_manager(policy.rand_manager) {}
    
    /**
     * Resets cur_node back to root node.
    */
    void EvalPolicy::reset() {
        cur_node = root_node;
    }

    /**
     * Gets a uniform random action.
    */
    shared_ptr<const Action> EvalPolicy::get_random_action(shared_ptr<const State> state) {
        shared_ptr<ActionVector> actions = mcts_env->get_valid_actions_itfc(state);
        int indx = rand_manager.get_rand_int(0, actions->size());
        return actions->at(indx);
    }

    /**
     * Gets the best recommendation from the current node.
    */
    shared_ptr<const Action> EvalPolicy::get_action(shared_ptr<const State> state, MctsEnvContext& context) {
        if (cur_node == nullptr) return get_random_action(state);
        return cur_node->recommend_action_itfc(context);
    }

    /**
     * Updates 'cur_node' for the last step taken in a trial.
    */
    void EvalPolicy::update_step(shared_ptr<const Action> action, shared_ptr<const Observation> obsv) {
        if (cur_node == nullptr) return;
        if (!cur_node->has_child_node_itfc(action)) {
            cur_node = nullptr;
            return;
        }
        MctsCNode& chance_node = *cur_node->get_child_node_itfc(action);
        if (!chance_node.has_child_node_itfc(obsv)) {
            cur_node = nullptr;
            return;
        }
        cur_node = chance_node.get_child_node_itfc(obsv);
    }
}

/**
 * MC Eval implementation
*/
namespace mcts {
    MCEvaluator::MCEvaluator(
        shared_ptr<const MctsEnv> mcts_env, 
        EvalPolicy& policy, 
        int max_trial_length, 
        RandManager& rand_manager) :
            mcts_env(mcts_env), 
            policy(policy), 
            max_trial_length(max_trial_length), 
            sampled_returns(), 
            rand_manager(rand_manager) {}

    /**
     * Runs a single rollout and stores the result in 'sampled_returns'.
    */
    void MCEvaluator::run_rollout(EvalPolicy& thread_policy) {
        // Reset
        thread_policy.reset();

        // Bookkeeping
        int num_actions_taken = 0;
        double sample_return = 0.0;
        shared_ptr<const State> state = mcts_env->get_initial_state_itfc();
        auto context_ptr = mcts_env->sample_context_itfc(state);
        MctsEnvContext& context = *context_ptr;

        // Run trial
        while (num_actions_taken < max_trial_length && !mcts_env->is_sink_state_itfc(state)) {
            shared_ptr<const Action> action = thread_policy.get_action(state, context);
            shared_ptr<const State> next_state = mcts_env->sample_transition_distribution_itfc(
                state, action, rand_manager);
            shared_ptr<const Observation> obsv = mcts_env->sample_observation_distribution_itfc(
                action, next_state, rand_manager);
            
            sample_return += mcts_env->get_reward_itfc(state, action, obsv);

            thread_policy.update_step(action, obsv);
            state = next_state;
            num_actions_taken++;
        }

        // store
        lock_guard lg(lock);
        sampled_returns.push_back(sample_return);
    }
    
    /**
     * Called as a thread. Runs this threads portion of 'total_rollouts' many rollouts. To make coding simple as we 
     * know exactly how many rollouts to perform ahead of time in this case, this thread will just be allocated all of 
     * the rollouts numbered == thread_id mod num_threads.
    */
    void MCEvaluator::thread_run_rollouts(
        int total_rollouts, int thread_id, int num_threads, unique_ptr<EvalPolicy> thread_policy) 
    {
        for (int i=thread_id; i < total_rollouts; i+=num_threads) {
            run_rollout(*thread_policy);
        }
    }

    /**
     * Runs 'num_rollouts' using 'num_threads'. Just sets each thread up, starts it running and then waits for them. 
     * Note that a pointer to a copy constructed policy is passed to each thread for it to use. (So each thread can 
     * assume that it has it's own thread_policy, copied from 'policy')
    */
    void MCEvaluator::run_rollouts(int num_rollouts, int num_threads) {
        // spawn
        vector<thread> threads;
        for (int i=0; i<num_threads; i++) {
            threads.push_back(thread(
                &MCEvaluator::thread_run_rollouts, 
                this, 
                num_rollouts, 
                i, 
                num_threads, 
                make_unique<EvalPolicy>(policy)));
        }

        // wait
        for (int i=0; i<num_threads; i++) {
            threads[i].join();
        }

    }

    /**
     * Returns the mean return of 'sampled_returns'
    */
    double MCEvaluator::get_mean_return() {
        double weight = 1.0 / sampled_returns.size();
        double mean = 0.0;
        for (double val : sampled_returns) {
            mean += weight * val;
        }
        return mean;
    }

    /**
    * Returns the stddev of 'sampled_returns'
    */
    double MCEvaluator::get_stddev_return() {
        double mean = get_mean_return();
        double weight = 1.0 / (sampled_returns.size() - 1.0);
        double variance = 0.0;
        for (double val : sampled_returns) {
            variance += weight * pow(val - mean, 2.0);
        }
        return sqrt(variance);
    }
}
