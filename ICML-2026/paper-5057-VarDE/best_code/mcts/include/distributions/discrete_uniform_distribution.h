#pragma once

#include "distributions/distribution.h"

#include "mcts_manager.h"

#include <memory>
#include <vector>


namespace mcts {
    /**
     * A discrete uniform distribution
     * 
     * Member variables:
     *      keys: A vector of objects that we want to sample over
    */
    template <typename T>
    class DiscreteUniformDistribution : public Distribution<T> {

        protected:
            std::shared_ptr<std::vector<T>> keys;

        public:
            DiscreteUniformDistribution(std::shared_ptr<std::vector<T>> keys);

            /**
             * Samples a random T uniformly randomly
            */
            virtual T sample(RandManager& rand_manager);;
    };
}

#include "distributions/discrete_uniform_distribution.cc"