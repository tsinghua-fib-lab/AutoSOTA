# Multi-Gaussian calibration for epsilon <= 0.5 and delta <= 1e-3.
#
# This branch keeps tighter quadrature/root tolerances for the most demanding
# privacy regime and uses a bounded shift scan.

const _multi_low_normal = Normal(0, 1)

struct MultiGaussianLowEpsilonCache
    epsilon::Float64
    Delta::Float64
    K::Int
    weights::Vector{Float64}
    normalized_weights::Vector{Float64}
    centers::Vector{Float64}
    weight_sum::Float64
    inverse_normalizer_without_sigma::Float64
    exp_epsilon::Float64
end

const _multi_low_cache = Ref{Union{Nothing,MultiGaussianLowEpsilonCache}}(nothing)

function refresh_multi_low_cache!()
    mixture_indices = collect(-K:K)
    weights = exp.(-abs.(mixture_indices) .* epsilon)
    normalized_weights = weights ./ sum(weights)
    retained = findall(normalized_weights .>= 1e-15)
    isempty(retained) && (retained = [argmax(normalized_weights)])
    mixture_indices = mixture_indices[retained]
    weights = weights[retained]
    weight_sum = sum(weights)
    normalized_weights = weights ./ weight_sum

    _multi_low_cache[] = MultiGaussianLowEpsilonCache(
        Float64(epsilon),
        Float64(Delta),
        Int(K),
        weights,
        normalized_weights,
        mixture_indices .* Delta,
        weight_sum,
        1 / (sqrt(2 * pi) * weight_sum),
        exp(epsilon),
    )
end

function multi_low_cache()
    cache = _multi_low_cache[]
    if cache === nothing ||
       cache.epsilon != epsilon ||
       cache.Delta != Delta ||
       cache.K != K
        refresh_multi_low_cache!()
        cache = _multi_low_cache[]
    end
    return cache::MultiGaussianLowEpsilonCache
end

function multi_cdf(x, sigma)
    cache = multi_low_cache()
    total = 0.0
    @inbounds @simd for index in eachindex(cache.weights)
        total += cache.weights[index] *
                 cdf(_multi_low_normal, (x - cache.centers[index]) / sigma)
    end
    return total / cache.weight_sum
end

function sample_multi_gaussian(sigma; number = 1)
    cache = multi_low_cache()
    component = sample(eachindex(cache.centers), Weights(cache.normalized_weights))
    return rand(_multi_low_normal, number) .* sigma .+ cache.centers[component]
end

function multi_pdf(x, sigma)
    cache = multi_low_cache()
    inverse_two_sigma_squared = 1 / (2 * sigma^2)
    total = 0.0
    @inbounds @simd for index in eachindex(cache.weights)
        total += cache.weights[index] *
                 exp(-(x - cache.centers[index])^2 * inverse_two_sigma_squared)
    end
    return total / (sqrt(2 * pi) * sigma * cache.weight_sum)
end

function multi_l1_noise(sigma)
    cache = multi_low_cache()
    total = 0.0
    @inbounds @simd for index in eachindex(cache.weights)
        center = cache.centers[index]
        total += cache.weights[index] * (
            2 * sigma^2 * exp(-(center^2) / (2 * sigma^2)) +
            center * sqrt(2 * pi) * sigma *
            (1 - 2 * cdf(_multi_low_normal, -center / sigma))
        )
    end
    return total / (sqrt(2 * pi) * sigma * cache.weight_sum)
end

function multi_l2_squared_noise(sigma)
    cache = multi_low_cache()
    total = 0.0
    @inbounds @simd for index in eachindex(cache.weights)
        total += cache.weights[index] * (sigma^2 + cache.centers[index]^2)
    end
    return total / cache.weight_sum
end

function multi_low_privacy_integrand(x, shift, sigma, cache)
    inverse_two_sigma_squared = 1 / (2 * sigma^2)
    density_numerator = 0.0
    shifted_density_numerator = 0.0
    @inbounds @simd for index in eachindex(cache.weights)
        weight = cache.weights[index]
        center = cache.centers[index]
        density_numerator += weight *
                             exp(-(x - center)^2 * inverse_two_sigma_squared)
        shifted_density_numerator += weight *
                                     exp(-(x + shift - center)^2 * inverse_two_sigma_squared)
    end
    violation = cache.exp_epsilon * density_numerator - shifted_density_numerator
    return violation < 0 ?
           violation * (cache.inverse_normalizer_without_sigma / sigma) : 0.0
end

function multi_privacy_shortfall(
    sigma;
    eta_override = 1e-6,
    minimum_steps = 500,
    maximum_steps = 500,
    tail_multiplier = 12.0,
    quadrature_atol = 1e-12,
    quadrature_rtol = 1e-7,
)
    cache = multi_low_cache()
    eta_local = isnothing(eta_override) ? eta : eta_override

    # A checked shift is accepted against (1 - eta) * delta, rather than the
    # full delta budget. The withheld portion supplies the numerical margin
    # associated with discretizing shifts in [0, Delta].
    raw_spacing = sqrt(2 * pi) * sigma * delta * eta_local
    ideal_steps = ceil(Int, Delta / max(raw_spacing, eps()))
    steps = clamp(ideal_steps, minimum_steps, maximum_steps)
    shift_spacing = Delta / steps
    smallest_shortfall = Inf

    # The maximum number of evaluated shifts is exposed in the signature so
    # the numerical resolution can be adjusted without changing the mechanism.
    for step in steps:-1:0
        shift = step * shift_spacing
        integration_limit = tail_multiplier * sigma + cache.K * Delta + shift
        shortfall = quadgk(
            x -> multi_low_privacy_integrand(x, shift, sigma, cache),
            -integration_limit,
            integration_limit;
            atol = quadrature_atol,
            rtol = quadrature_rtol,
        )[1] + (1 - eta_local) * delta
        if isnan(shortfall) || shortfall < 0
            return -1.0
        end
        smallest_shortfall = min(smallest_shortfall, shortfall)
    end
    return smallest_shortfall
end

function calibrate_multi_sigma(
    ;
    eta_override = 1e-6,
    minimum_steps = 500,
    maximum_steps = 500,
    tail_multiplier = 12.0,
    quadrature_atol = 1e-12,
    quadrature_rtol = 1e-7,
)
    refresh_multi_low_cache!()
    eta_local = isnothing(eta_override) ? eta : eta_override
    cached_shortfalls = Dict{Float64,Float64}()
    shortfall = sigma -> get!(cached_shortfalls, Float64(sigma)) do
        multi_privacy_shortfall(
            sigma;
            eta_override = eta_local,
            minimum_steps = minimum_steps,
            maximum_steps = maximum_steps,
            tail_multiplier = tail_multiplier,
            quadrature_atol = quadrature_atol,
            quadrature_rtol = quadrature_rtol,
        )
    end

    upper = sqrt(2 * log(1.25 / ((1 - eta_local) * delta))) * Delta / epsilon
    lower = 1.0
    while shortfall(lower) > 0
        lower /= 2
    end

    attempts = 0
    while shortfall(upper) < 0 && attempts < 30
        upper *= 2
        attempts += 1
    end
    shortfall(upper) < 0 && return upper

    return find_zero(shortfall, (lower, upper), method = Brent(), xatol = 1e-7, atol = 1e-7)
end
