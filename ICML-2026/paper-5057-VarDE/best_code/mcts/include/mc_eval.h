#pragma once

#include "mcts_chance_node.h"
#include "mcts_decision_node.h"
#include "mcts_env.h"
#include "mcts_env_context.h"
#include "mcts_manager.h"
#include "mcts_types.h"

#include <memory>
#include <mutex>
#include <vector>

namespace mcts {
    // Forward declare
    class MCEvaluator;

    /**
     * Converts a Mcts tree into a complete policy that can be used for evaluation.
     * 
     * While performing a trial will trace the trial in the tree and use the recommendations given by decision nodes.
     * When a trial reaches a state that there isn't a corresponding tree node, uniform random policy is used.
     * 
     * N.B. Currently only works for fully observable environments.
     * 
     * Member variables:
     *      root_node: 
     *          The root node of the Mcts tree to use
     *      cur_node: 
     *          The current decision node to use to make recommendations in the policy (iff null, use uniform)
     *      mcts_env: 
     *          The environment for when random actions need to be sampled
     *      rand_manager: 
     *          Manager for rng
    */
    class EvalPolicy {
        friend MCEvaluator;

        protected:
            std::shared_ptr<const MctsDNode> root_node;
            std::shared_ptr<const MctsDNode> cur_node;
            std::shared_ptr<const MctsEnv> mcts_env;
            RandManager& rand_manager;

        public:
            EvalPolicy(
                std::shared_ptr<const MctsDNode> root_node, 
                std::shared_ptr<const MctsEnv> mcts_env, 
                RandManager& rand_manager);

            EvalPolicy(const EvalPolicy& policy);

            /**
             * Resets cur_node back to root node.
            */
            void reset();

            /**
             * Gets a uniform random action.
            */
            std::shared_ptr<const Action> get_random_action(std::shared_ptr<const State> state);

            /**
             * Gets the best recommendation from the current node.
            */
            std::shared_ptr<const Action> get_action(std::shared_ptr<const State> state, MctsEnvContext& context);

            /**
             * Updates 'cur_node' for the last step taken in a trial.
            */
           void update_step(std::shared_ptr<const Action> action, std::shared_ptr<const Observation> obsv);
    };

    /**
     * MC Evaluator
     * 
     * N.B. Currently only works for fully observable environments.
     * N.B.B. We store references in the class, and the caller needs to assure that they still exists. (I.e. usually 
     * we're going to make an MCEvaluator, call run rollouts, and then throw it away).
     * 
     * Member variables:
     *      mcts_env: 
     *          The env that we want to evaluate in
     *      policy: 
     *          The policy to evaluate
     *      max_trial_length: 
     *          The maximum trial length to use in MC evaluations
     *      sampled_returns: 
     *          A list of sampled returns 
     *      rand_manager: 
     *          Manager for rng
     *      lock: 
     *          A lock to protect access to class variables (i.e. sampled_returns in 'run_rollout')
    */
    class MCEvaluator {
        protected:
            std::shared_ptr<const MctsEnv> mcts_env;
            EvalPolicy& policy;
            int max_trial_length;
            std::vector<double> sampled_returns;
            RandManager& rand_manager;
            std::mutex lock;

            /**
             * Runs a single rollout and stores the result in 'sampled_returns'.
            */
            void run_rollout(EvalPolicy& thread_policy);

            /**
             * Runs rollouts as a worker thread
            */
            void thread_run_rollouts(
                int total_rollouts, int thread_id, int num_threads, std::unique_ptr<EvalPolicy> thread_policy);



        public:
            MCEvaluator(
                std::shared_ptr<const MctsEnv> mcts_env,
                EvalPolicy& eval_policy,
                int max_trial_length,
                RandManager& rand_manager);

            /**
             * Run 'num_rollout' many rollouts to gather stats. Does so by spawning 'num_threads' many threads and 
             * setting them off to run rollouts using the 'thread_run_rollouts' function.
             * 
             * Assumes that the tree is static during this call, so does not lock the nodes of the tree.
            */
            void run_rollouts(int num_rollouts, int num_threads);

            /**
             * Returns the mean return of 'sampled_returns'
            */
           double get_mean_return();

           /**
            * Returns the stddev of 'sampled_returns'
           */
          double get_stddev_return();
    };
}