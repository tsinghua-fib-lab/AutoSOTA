#pragma once

#include "mcts_decision_node.h"
#include "mcts_env_context.h"
#include "mcts_logger.h"
#include "mcts_manager.h"

#include <chrono>
#include <condition_variable>
#include <limits>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>


namespace mcts {
    /**
     * A class encapsulating all of the logic required to run a Mcts routine.
     * 
     * At a higher level, this class creats a pool of threads that call 'run_trial' until they are signalled to stop. 
     * The threads are signaled to stop when the maximum duration has passed, or the desirect number of trials have 
     * been performed.
     * 
     * All functions are marked as virtual so that they can be mocked for unit testing.
     * 
     * Member variables:
     *      workers: 
     *          A vector of threads (thread pool) that run the worker_fn routine.
     *      work_left_cv: 
     *          A condition variable used to coordinate when workers are working.
     *      work_left_lock: 
     *          A mutex for 'work_left_lock', protecting variables used to decide when there is work.
     *      logging_lock: 
     *          A mutex protecting any logging performed by Mcts.
     *      thread_pool_alive: 
     *          A boolean stating if the workers thread pool is running. Set to false at destruction.
     *      num_threads: 
     *          The number of threads used in the workers thread pool.
     *      num_trials: 
     *          The number of trials the pool is currently trying to run in total.
     *      start_time: 
     *          The start time of a 'run_trials' call.
     *      max_run_time: 
     *          The maximum duration we allow Mcts to run for)not including finishing off the current trials).
     *      trials_remaining: 
     *          The number of trials the workers pool needs to run to have completed 'num_trials' trials.
     *      num_threads_working: 
     *          The number of threads currently working
     *      trials_completed: 
     *          The number of trials completed for 
     *      mcts_manager: 
     *          The MctsManager to use in the Mcts planning routine
     *      root_node: 
     *          The MctsDNode root node that currently want to plan for
     */
    class MctsPool {
        protected:   
            // multithreading variables (also protected by can_work_lock)
            std::vector<std::thread> workers;
            std::condition_variable_any work_left_cv;
            std::mutex work_left_lock;
            std::mutex logging_lock;
            bool thread_pool_alive;

            // constant after init
            int num_threads;

            // protected by can_work_lock - variables only updated on 'run_trials' call 
            int num_trials;
            std::chrono::time_point<std::chrono::system_clock> start_time;
            std::chrono::duration<double> max_run_time;

            // protected by can_work_lock - variables related to if should run more trials + updated by workers
            int trials_remaining;
            int num_threads_working;

            // protected by logging_lock - variables to do with logging
            int trials_completed;
            std::shared_ptr<MctsLogger> logger;

            // Manager and root node specifying the flavour of Mcts to run (the problem and algorithm)
            std::shared_ptr<MctsManager> mcts_manager;
            std::shared_ptr<MctsDNode> root_node;

        public:
            /**
             * Constructs the MctsPool with 'num_threads' worker threads.
             * 
             * Args:
             *      manager: The MctsManager to use for this instance of Mcts
             *      root_node: 
             *          The root node to run Mcts on. The algorithm is specified by the subclass of MctsDNode used. If 
             *          NULL, then a default root node construction is attempted using the initial state from 
             *          mcts_manager->mcts_env.
             *      num_threads: The number of worker threads to spawn
             */
            MctsPool(
                std::shared_ptr<MctsManager> mcts_manager=nullptr, 
                std::shared_ptr<MctsDNode> root_node=nullptr, 
                int num_threads=1,
                std::shared_ptr<MctsLogger> logger=nullptr);

            /**
             * Destructor. Required to allow the thread pool to exit gracefully.
             */
            virtual ~MctsPool();

            /**
             * Setter for new search environment, so thread pool can be reused
            */
            void set_new_env(
                std::shared_ptr<MctsManager> new_mcts_manager, 
                std::shared_ptr<MctsDNode> new_root_node,
                std::shared_ptr<MctsLogger> new_logger=nullptr);

            /**
             * Returns a boolean for if the workers need to do more work to complete a 'run_trials' call.
             * 
             * Calls to this function should be protected using work_left_lock.
             */
            virtual bool work_left();

        protected:
            /**
             * Checks if a worker should continue their selection phase or if it is time to end.
             * 
             * Selection phase ends when a leaf node (in the mcts_env) or max depth is hit, or when running in mcts_mode, 
             * once any new nodes have been added.
             * 
             * Args:
             *      cur_node: The most recent node reached in the selection phase
             *      new_decision_node_created_this_trial: If a new decision node has been created this trial
             * 
             * Returns:
             *      If the selection phase should be ended.
             */
            virtual bool should_continue_selection_phase(
                std::shared_ptr<MctsDNode> cur_node, bool new_decision_node_created_this_trial);

            /**
             * Runs the selection phase of a trial, called by worker threads.
             * 
             * Args:
             *      nodes_to_backup: 
             *          A list to be filled with pairs of (MctsDNode, MctsCNode), that should have 'backup' called on them in the 
             *          backup phase
             *      rewards:
             *          A list to be filled with rewards obtained during this selection phase (including the heuristic 
             *          value of a frontier node (which is not visited))
             *      context:
             *          The MctsContext for this trial
             * 
             * Returns:
             *      Nothing, 'nodes_to_backup' and 'rewards' are filled by this function as 'return_values'.
             */
            void run_selection_phase(
                std::vector<std::pair<std::shared_ptr<MctsDNode>,std::shared_ptr<MctsCNode>>>& nodes_to_backup, 
                std::vector<double>& rewards, 
                MctsEnvContext& context);

            /**
             * Runs the backup phase of a trial, called by worker threads.
             * 
             * Args:
             *      nodes_to_backup: 
             *          A list of pairs of (MctsDNode, MctsCNode), to call backup on (created by selection phase)
             *      rewards:
             *          A list of rewards obtained during the selection phase that may be used by backups
             *      context:
             *          The MctsContext for this trial
             */
            void run_backup_phase(
                std::vector<std::pair<std::shared_ptr<MctsDNode>,std::shared_ptr<MctsCNode>>>& nodes_to_backup, 
                std::vector<double>& rewards, 
                MctsEnvContext& context);

            /**
             * Performs a single Mcts trial. Called by worker_fn.
             * 
             * Args:
             *      trials_remaining: 
             *          The number of trials remaining at the time of calling (not including the one about to be run by 
             *          this function)
             */
            virtual void run_mcts_trial(int trials_remaining);

            /**
            * Checks if we need to perform logging, and if so, will write a log to the logger
            */
            void try_log();


            /**
             * The worker thread thnuk.
             * 
             * Waits for work, and calls 'run_mcts_trial' until there is no more work to do.
             */
            virtual void worker_fn();

        public:
            /**
             * Waits on the MctsPool to finish a 'run_trials' call. Returns when the workers have finished.
             */
            virtual void join();

            /**
             * Specifys how many trails/how long to run trials of Mcts for.
             * 
             * Default values essentially runs the trials indefinitely.
             * 
             * Args:
             *      max_trials: The maximum number of Mcts trials to run
             *      max_time: The maximum (human time) to run Mcts trials for
             *      blocking: If this call is blocking and will only return when the Mcts trials have finished
             */
            virtual void run_trials(
                int max_trials=std::numeric_limits<int>::max(), 
                double max_time=std::numeric_limits<double>::max(), 
                bool blocking=true);            
    };
}
