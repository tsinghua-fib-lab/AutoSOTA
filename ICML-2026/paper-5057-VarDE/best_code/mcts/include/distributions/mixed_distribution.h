#pragma once

#include "distributions/distribution.h"

#include "mcts_manager.h"

#include <unordered_map>




namespace mcts {
    /**
     * Typedef long types
    */
    template<typename T>
    using MixedDistributionDistr = std::unordered_map<std::shared_ptr<Distribution<T>>,double>;
    /**
     * A mixed distribution
     * 
     * This represents a mixed distribution over other distributions.
     * 
     * Assumes that the weights of the given distribution sum to 1.0.
     * 
     * Member variables:
     *      distr: 
     *          A dictionary mapping from shared_ptr<Distribution> to doubles, defining the categorical distribution 
     *          over other distributions.
     *      sum_weights: 
     *          A double, keeping track of the sum of all of the probability weights
    */
    template <typename T>
    class MixedDistribution : public Distribution<T> {

        /**
         * Member variables
        */
        protected:
            std::shared_ptr<MixedDistributionDistr<T>> distr;
            double sum_weights;
        
        public:
            /**
             * Constructor
            */
            MixedDistribution(std::shared_ptr<MixedDistributionDistr<T>> distr);

            /**
             * Samples a random T from the mixed distribution
            */
            virtual T sample(RandManager& rand_manager);
        
    };
}

#include "distributions/mixed_distribution.cc"