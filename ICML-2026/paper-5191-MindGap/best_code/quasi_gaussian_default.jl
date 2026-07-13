# Quasi-Gaussian calibration for the default numerical regime.
#
# The mechanism combines a central Gaussian term and symmetric shifted tail
# terms. The utility quantities below are closed-form moments of that density;
# only sigma calibration requires scalar root searches.

const _quasi_default_normal = Normal(0, 1)

struct QuasiGaussianCache
    sigma::Float64
    inverse_sigma::Float64
    inverse_sigma_squared::Float64
    exp_epsilon::Float64
    cdf_sensitivity::Float64
    inverse_normalizer::Float64
end

function quasi_cache(sigma::Real)
    sigma_value = Float64(sigma)
    inverse_sigma = 1 / sigma_value
    inverse_sigma_squared = inverse_sigma^2
    exp_epsilon = exp(epsilon)
    cdf_sensitivity = cdf(_quasi_default_normal, Delta * inverse_sigma)
    inverse_normalizer = 1 / (
        sqrt(2 * pi) * sigma_value * (exp_epsilon + 2 * cdf_sensitivity)
    )
    return QuasiGaussianCache(
        sigma_value,
        inverse_sigma,
        inverse_sigma_squared,
        exp_epsilon,
        cdf_sensitivity,
        inverse_normalizer,
    )
end

@inline function quasi_pdf(x::Real, cache::QuasiGaussianCache)
    central_term = cache.exp_epsilon *
                   exp(-x^2 * 0.5 * cache.inverse_sigma_squared)
    shifted_term = exp(
        -(abs(x) - Delta)^2 * 0.5 * cache.inverse_sigma_squared,
    )
    return (central_term + shifted_term) * cache.inverse_normalizer
end

quasi_pdf(x, sigma) = quasi_pdf(x, quasi_cache(sigma))

function quasi_cdf(x, sigma)
    cache = quasi_cache(sigma)
    denominator = cache.exp_epsilon + 2 * cache.cdf_sensitivity
    if x < 0
        numerator = cache.exp_epsilon *
                    cdf(_quasi_default_normal, x * cache.inverse_sigma) +
                    cdf(_quasi_default_normal, (x + Delta) * cache.inverse_sigma)
    else
        numerator = cache.exp_epsilon *
                    cdf(_quasi_default_normal, x * cache.inverse_sigma) +
                    cdf(_quasi_default_normal, (x - Delta) * cache.inverse_sigma) +
                    cache.cdf_sensitivity -
                    cdf(_quasi_default_normal, -Delta * cache.inverse_sigma)
    end
    return numerator / denominator
end

function sample_quasi_gaussian(sigma)
    cache = quasi_cache(sigma)
    central_probability = cache.exp_epsilon /
                          (cache.exp_epsilon + 2 * cache.cdf_sensitivity)
    if rand() < central_probability
        return rand(_quasi_default_normal) * cache.sigma
    end

    probability = rand()
    sign = rand((-1, 1))
    magnitude = Delta + cache.sigma * quantile(
        _quasi_default_normal,
        cdf(_quasi_default_normal, -Delta * cache.inverse_sigma) +
        probability * cache.cdf_sensitivity,
    )
    return sign * magnitude
end

function quasi_l1_noise(sigma)
    cache = quasi_cache(sigma)
    numerator = sqrt(2 / pi) * cache.sigma * (
        cache.exp_epsilon +
        exp(-(Delta^2) * 0.5 * cache.inverse_sigma_squared)
    ) + 2 * Delta * cache.cdf_sensitivity
    denominator = cache.exp_epsilon + 2 * cache.cdf_sensitivity
    return numerator / denominator
end

function quasi_l2_squared_noise(sigma)
    cache = quasi_cache(sigma)
    shifted_second_moment = cache.cdf_sensitivity * (cache.sigma^2 + Delta^2) +
                            (cache.sigma * Delta / sqrt(2 * pi)) *
                            exp(-(Delta^2) * 0.5 * cache.inverse_sigma_squared)
    numerator = cache.exp_epsilon * cache.sigma^2 + 2 * shifted_second_moment
    denominator = cache.exp_epsilon + 2 * cache.cdf_sensitivity
    return numerator / denominator
end

# Equation defining the first candidate sigma in the quasi-Gaussian
# calibration derivation.
function quasi_first_sigma_equation(sigma)
    inverse_sigma = 1 / sigma
    first_cdf = cdf(
        _quasi_default_normal,
        -epsilon * sigma / Delta - Delta * inverse_sigma,
    )
    second_cdf = cdf(
        _quasi_default_normal,
        -epsilon * sigma / Delta + Delta * inverse_sigma,
    )
    sensitivity_cdf = cdf(_quasi_default_normal, Delta * inverse_sigma)
    return exp(2 * epsilon) * first_cdf -
           second_cdf +
           (exp(epsilon) + 2 * sensitivity_cdf) * delta
end

function quasi_pdf_minimizer(sigma)
    cache = quasi_cache(sigma)
    minimum_location = Delta
    root_term = sqrt(max(Delta^2 - 4 * cache.sigma^2, 0))
    candidate_limit = (Delta + root_term) / 2
    condition = -cache.exp_epsilon +
                ((Delta - candidate_limit) / candidate_limit) *
                exp(
                    (2 * candidate_limit * Delta - Delta^2) *
                    0.5 * cache.inverse_sigma_squared,
                )

    if Delta^2 > 4 * cache.sigma^2 && condition > 0
        candidate = Optim.minimizer(
            optimize(x -> quasi_pdf(x, cache), Delta / 2, candidate_limit, GoldenSection()),
        )
        quasi_pdf(candidate, cache) < quasi_pdf(minimum_location, cache) &&
            (minimum_location = candidate)
    end
    return minimum_location, quasi_pdf(minimum_location, cache)
end

function quasi_pdf_maximizer(sigma)
    cache = quasi_cache(sigma)
    root_term = sqrt(max(Delta^2 - 4 * cache.sigma^2, 0))
    search_limit = Delta^2 <= 4 * cache.sigma^2 ?
                   Delta / 2 : (Delta - root_term) / 2
    maximum_location = Optim.minimizer(
        optimize(x -> -quasi_pdf(x, cache), 0, search_limit, GoldenSection()),
    )
    return maximum_location, quasi_pdf(maximum_location, cache)
end

function quasi_density_ratio_equation(sigma)
    maximum_density = quasi_pdf_maximizer(sigma)[2]
    minimum_density = quasi_pdf_minimizer(sigma)[2]
    minimum_density == 0 && return Inf
    return maximum_density / minimum_density - exp(epsilon)
end

function calibrate_quasi_sigma()
    first_sigma = 0.0
    if exp(epsilon) + 2 < 1 / delta
        upper = sqrt(2 * (epsilon - log(delta))) * Delta / epsilon
        first_sigma = find_zero(quasi_first_sigma_equation, (1e-8, upper), Bisection())
    end

    upper = sqrt(2 * log(1.25 / delta)) * Delta / epsilon
    while quasi_density_ratio_equation(upper) > 0
        upper *= 2
    end
    second_sigma = find_zero(quasi_density_ratio_equation, (1e-8, upper), Bisection())

    # The calibration imposes both scalar conditions; using the larger
    # candidate is the feasible noise scale for the mechanism.
    return max(first_sigma, second_sigma)
end
