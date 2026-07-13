#pragma once

#include "mcts_manager.h"

namespace mcts {
    /**
     * An abstract class for defining the interface a distribution object should provide. (I.e. we can sample from it)
    */
    template <typename T>
    class Distribution {
        public:
            /**
             * Samples an object of type T from the distribution and returns it.
            */
            virtual T sample(RandManager& rand_manager) = 0;
    };
}