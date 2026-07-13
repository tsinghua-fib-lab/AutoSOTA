#pragma once

#include <cmath>

namespace mcts {
    /**
     * Typedef for temperature decay function
     * 
     * A valid temp decay function, f, should satisfy:
     * 1. f(0) = 1.0
     * 2. f(m) -> 0.0, as m -> infty
    */
    double _DummyTempDecayFn(double scaled_visits);
    typedef decltype(&_DummyTempDecayFn) TempDecayFnPtr;

    /**
     * Compute the decayed temperature
    */
    double compute_decayed_temp(
        TempDecayFnPtr f, double init_temp, double min_temp, int num_visits, double visits_scale);

    /**
     * Inverse square root temp decay function
     * f(m) = 1/sqrt(1+m)
    */
    double decayed_temp_inv_sqrt(double scaled_visits);

    /**
     * Inverse log temp decay function
     * f(m) = 1/log(e + m)
    */
    double decayed_temp_inv_log(double scaled_visits);

    /**
        * Sigmoid temp decay function
        * f(m) = (1+exp(-5)) / (1+exp(m-5))
    */
    double decayed_temp_sigmoid(double scaled_visits);
}