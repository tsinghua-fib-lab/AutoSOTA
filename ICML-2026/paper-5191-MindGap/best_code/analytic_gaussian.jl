using SpecialFunctions: erf

# This Julia implementation follows the Apache-2.0-licensed reference code:
# https://github.com/BorjaBalle/analytic-gaussian-mechanism/blob/master/agm-example.py
#
# This baseline calibrates Gaussian noise exactly through the analytic Gaussian
# mechanism test of Balle and Wang (ICML 2018). It is kept separate from the
# mixture mechanisms because it has a closed one-dimensional calibration
# routine and does not need a shift-grid privacy check.

normal_cdf(t) = 0.5 * (1.0 + erf(float(t) / sqrt(2.0)))

analytic_case_a(epsilon, s) =
    normal_cdf(sqrt(epsilon * s)) -
    exp(epsilon) * normal_cdf(-sqrt(epsilon * (s + 2.0)))

analytic_case_b(epsilon, s) =
    normal_cdf(-sqrt(epsilon * s)) -
    exp(epsilon) * normal_cdf(-sqrt(epsilon * (s + 2.0)))

function bracket_by_doubling(predicate_stop, lower, upper)
    while !predicate_stop(upper)
        lower = upper
        upper = 2.0 * lower
    end
    return lower, upper
end

function binary_search_until(predicate_stop, move_left, lower, upper)
    midpoint = lower + (upper - lower) / 2.0
    while !predicate_stop(midpoint)
        if move_left(midpoint)
            upper = midpoint
        else
            lower = midpoint
        end
        midpoint = lower + (upper - lower) / 2.0
    end
    return midpoint
end

function calibrate_analytic_gaussian(epsilon, delta, sensitivity; tol = 1e-12)
    delta_threshold = analytic_case_a(epsilon, 0.0)

    if delta == delta_threshold
        alpha = 1.0
    else
        if delta > delta_threshold
            bracket_condition = s -> analytic_case_a(epsilon, s) >= delta
            delta_at = s -> analytic_case_a(epsilon, s)
            move_left = s -> delta_at(s) > delta
            alpha_at = s -> sqrt(1.0 + s / 2.0) - sqrt(s / 2.0)
        else
            bracket_condition = s -> analytic_case_b(epsilon, s) <= delta
            delta_at = s -> analytic_case_b(epsilon, s)
            move_left = s -> delta_at(s) < delta
            alpha_at = s -> sqrt(1.0 + s / 2.0) + sqrt(s / 2.0)
        end

        stop_search = s -> abs(delta_at(s) - delta) <= tol
        lower, upper = bracket_by_doubling(bracket_condition, 0.0, 1.0)
        solution = binary_search_until(stop_search, move_left, lower, upper)
        alpha = alpha_at(solution)
    end

    return alpha * sensitivity / sqrt(2.0 * epsilon)
end
