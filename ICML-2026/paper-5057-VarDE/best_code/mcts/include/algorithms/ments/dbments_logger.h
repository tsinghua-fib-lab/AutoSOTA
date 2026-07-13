#pragma once

#include "algorithms/ments/dbments_decision_node.h"
#include "algorithms/ments/ments_logger.h"

#include <chrono>
#include <limits>
#include <memory>
#include <ostream>
#include <vector>

namespace mcts {
    /**
     * DBMents logger point
     * 
     * Member variables:
     *      runtime: The total runtime for this datapoint
     *      num_trials: The number of times the root node has been visited (trials started)
     *      num_backups: The number of backups completed at root node (trials completed)
     *      soft_value: The ments soft value at the root node
     *      dp_value: The dp value at the root node
     */
    struct DBMentsLoggerEntry : public MentsLoggerEntry {
        double dp_value;

        // Construcor
        DBMentsLoggerEntry(
            std::chrono::duration<double> runtime, int num_trials, int num_backups, double soft_value, double dp_value);

        // Virtual destructor
        virtual ~DBMentsLoggerEntry() = default;

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
     * Implementation of logger for DBMents algorithms
     * 
     * Just needs to override the origin entry and log functions.
     */
    class DBMentsLogger : public MentsLogger {
        public:
            DBMentsLogger();

            virtual ~DBMentsLogger() = default;

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