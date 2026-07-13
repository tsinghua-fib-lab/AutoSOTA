#pragma once

#include "mcts_decision_node.h"

#include <chrono>
#include <limits>
#include <memory>
#include <ostream>
#include <vector>

namespace mcts {

    /**
     * Base class for entries in logger
     * 
     * Member variables:
     *      runtime: 
     *          The total runtime for this datapoint
     *      num_trials: 
     *          The number of times the root node has been visited (trials started)
     */
    struct LoggerEntry {
        std::chrono::duration<double> runtime;
        int num_visits;

        // Construcor
        LoggerEntry(std::chrono::duration<double> runtime, int num_visits);

        // Virtual destructor
        virtual ~LoggerEntry() = default;

        /**
         * Writes a header for an ostream, so we would know what the entries correspond to in a csv
         * 
         * Args:
         *      os: An output stream
         */
        virtual void write_header_to_ostream(std::ostream& os);

        /**
         * Writes this entry to an ostream
         * 
         * Args:
         *      os: An output stream
         */
        virtual void write_to_ostream(std::ostream& os);
    };

    /**
     * Abstract logger class
     * 
     * Member variables:
     *      entries: 
     *          A vector of LoggerEntry objects, each representing a datatype
     *      prior_runtime: 
     *          The amount of runtime logger from previous calls to MctsPool run_trials
     *      start_time: 
     *          The time at the start of the last run_trials call in MctsPool
     *      trials_completed: 
     *          The number of trials completed (number of times 'trial_completed' called)
     *      trials_delta: 
     *          Indicates 'log' should be called every 'trials_delta' completed trials.
     *      last_log_num_trials:
     *          The value of 'num_trials' the last time 'log' was (should have been) called
     *      runtime_delta: 
     *          Indicates 'log' should be called every 'runtime_delta' seconds. Default is max value which means never 
     *          log because of runtime.
     *      next_log_runtime_threshold: 
     *          The next runtime duration that we should log at 
     */
    class MctsLogger {
        protected:
            std::vector<LoggerEntry> entries;
            std::chrono::duration<double> prior_runtime;
            std::chrono::time_point<std::chrono::system_clock> start_time;

            int trials_completed;
            int trials_delta;
            int last_log_num_trials;
            std::chrono::duration<double> runtime_delta;
            std::chrono::duration<double> next_log_runtime_threshold;

        public:
            MctsLogger();

            virtual ~MctsLogger() = default;

            /**
             * Setter for trials delta
             */
            void set_trials_delta(int delta);

            /**
             * Setter for runtime delta
             */
            void set_runtime_delta(double delta);

            /**
             * Gets the size of the logger
             */
            int size() const;

            /**
             * Adds an entry to 'entries' that represents an origin point
             */
            virtual void add_origin_entry();

            /**
             * Call this at the beginning of a run_trials call to set start time 
             */
            void reset_start_time();

            /**
             * Helper to get the current runtime
             * 
             * Returns:
             *      The total runtime used in the Mcts routine so far
             */
            std::chrono::duration<double> get_current_total_runtime();

            /**
             * Call when a trial is completed to increment trials completed.
            */
           void trial_completed();

            /**
             * Checks if it is time to call log
             * 
             * Note that if the deltas are at their default values, the rhs of the comparisons will always be a max 
             * value and the checks will never pass
             * 
             * Uses the current time and current value of 'trials_completed' to determine if it is time to log.
             * 
             * Returns:
             *      If 'log' should be called for this current trial
             */
            bool should_log();

            /**
             * Adds an logger entry based off the current 
             * 
             * Assumes that the lock for the node has already been 
             * 
             * Args:
             *      node: A (root) node to log information about
             */
            virtual void log(std::shared_ptr<MctsDNode> node);

            /**
             * Call this when all trials have been run
             */
            void update_prior_runtime();

            /**
             * Write logger to an ostream
             * 
             * Args:
             *      os: The output stream to write to
             */
            void write_to_ostream(std::ostream& os);
    };
}