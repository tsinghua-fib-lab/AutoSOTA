#include "algorithms/common/decaying_temp.h"

static double CONST_E = exp(1.0);
static double CONST_SIGMOID_NUMERATOR = 1.0 + exp(-5.0);

namespace mcts {

    /**
     * Compute decayed temperature to use, given a decay function and search params
    */
    double compute_decayed_temp(
        TempDecayFnPtr f, double init_temp, double min_temp, int num_visits, double visits_scale) 
    {
        double temp = init_temp * f(visits_scale * num_visits);
        if (temp < min_temp) return min_temp;
        return temp;   
    }

    /**
     * Inverse square root temp decay function
     * f(m) = 1/sqrt(1+m)
    */
    double decayed_temp_inv_sqrt(double scaled_visits) {
        return 1.0 / sqrt(1.0 + scaled_visits);
    }

    /**
     * Inverse log temp decay function
     * f(m) = 1/log(e + m)
    */
    double decayed_temp_inv_log(double scaled_visits) {
        return 1.0 / log(CONST_E + scaled_visits);
    }

    /**
        * Sigmoid temp decay function
        * f(m) = (1+exp(-5)) / (1+exp(m-5))
    */
    double decayed_temp_sigmoid(double scaled_visits) {
        return CONST_SIGMOID_NUMERATOR / (1.0 + exp(scaled_visits - 5.0)); 
    }
}