#include "distributions/discrete_uniform_distribution.h"


namespace mcts {
    using namespace std;
    /**
     * Constructor
    */
    template <typename T>
    DiscreteUniformDistribution<T>::DiscreteUniformDistribution(shared_ptr<vector<T>> keys) : keys(keys) {};
    /**
     * Sampling from discrete uniform just needs a random integer
    */
    template <typename T>
    T DiscreteUniformDistribution<T>::sample(RandManager& rand_manager) {
        int index = rand_manager.get_rand_int(0, keys->size());
        return keys->at(index);
    };
}