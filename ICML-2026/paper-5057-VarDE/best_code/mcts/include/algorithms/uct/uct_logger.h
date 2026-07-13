#pragma once

#include "algorithms/uct/uct_decision_node.h"
#include "mcts_logger.h"

#include <chrono>
#include <limits>
#include <memory>
#include <ostream>
#include <vector>

namespace mcts {
    /**
     * Uct logger point
     * 
     * Member variables:
     *      runtime: 
     *          The total runtime for this datapoint
     *      num_trials: 
     *          The number of times the root node has been visited (trials started)
     *      num_backups: 
     *          The number of backups completed at root node (trials completed)
     *      avg_return: 
     *          Average return at the root node for uct
     */
    struct UctLoggerEntry : public LoggerEntry {
        int num_backups;
        double avg_return;

        // Construcor
        UctLoggerEntry(std::chrono::duration<double> runtime, int num_trials, int num_backups, double avg_return);

        // Virtual destructor
        virtual ~UctLoggerEntry() = default;

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
     * Implementation of logger for UCT algorithms
     * 
     * Just needs to override the origin entry and log functions.
     */
    class UctLogger : public MctsLogger {
        public:
            UctLogger();

            virtual ~UctLogger() = default;

            /**
             * Adds an entry to 'entries' that represents an origin point
             */
            virtual void add_origin_entry();

            /**
             * Adds an logger entry based off the current 
             * 
             * Assumes that the lock for the node has already been 
             * 
             * Args:
             *      node: A (root) node to log information about
             */
            virtual void log(std::shared_ptr<MctsDNode> node);
    };
}