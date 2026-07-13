#include "distributions/mixed_distribution.h"

#include "helper_templates.h"


namespace mcts {
    using namespace std;

    /**
     * Constructor 
    */
    template <typename T>
    MixedDistribution<T>::MixedDistribution(
        shared_ptr<MixedDistributionDistr<T>> distr) :
            distr(distr)
    {
    }

    /**
     * Sample from the mixed distribution. First samples from the categorical distribution over the distributions, and 
     * then samples from that distribution.
     * 
     * For now assume that we are mixing a small number of distributions, so that sampling naively is probably faster 
     * than adding the overhead of alias table logic. If thats ever necessary, we can use a CatagoricalDistribution.
    */
    template <typename T>
    T MixedDistribution<T>::sample(RandManager& rand_manager) {
        shared_ptr<Distribution<T>> sampled_distr = helper::sample_from_distribution(*distr, rand_manager);
        return sampled_distr->sample(rand_manager);
    };
}