# Multi-Gaussian calibration for the default numerical regime.
#
# The density is a finite Gaussian mixture with centers k * Delta and weights
# proportional to exp(-abs(k) * epsilon). For a proposed sigma, the privacy
# check scans shifts in [0, Delta] and verifies the approximate-DP inequality
# by numerical quadrature.

const _multi_default_normal = Normal(0, 1)

struct MultiGaussianDefaultCache
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

const _multi_default_cache = Ref{Union{Nothing,MultiGaussianDefaultCache}}(nothing)

function refresh_multi_default_cache!()
    mixture_indices = collect(-K:K)
    weights = exp.(-abs.(mixture_indices) .* epsilon)
    normalized_weights = weights ./ sum(weights)

    # Terms below machine-relevant mass only add quadrature work. The cutoff
    # is far below the privacy tolerances used here.
    retained = findall(normalized_weights .>= 1e-15)
    isempty(retained) && (retained = [argmax(normalized_weights)])
    mixture_indices = mixture_indices[retained]
    weights = weights[retained]
    weight_sum = sum(weights)
    normalized_weights = weights ./ weight_sum

    _multi_default_cache[] = MultiGaussianDefaultCache(
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

function multi_default_cache()
    cache = _multi_default_cache[]
    if cache === nothing ||
       cache.epsilon != epsilon ||
       cache.Delta != Delta ||
       cache.K != K
        refresh_multi_default_cache!()
        cache = _multi_default_cache[]
    end
    return cache::MultiGaussianDefaultCache
end

function epsilon_scaled_step_cap(base_steps::Int, current_epsilon::Float64)
    # Larger epsilon concentrates mixture weights nearer zero, allowing a
    # smaller cap on the shift scan.
    scaled_steps = round(Int, base_steps / (1 + current_epsilon))
    return clamp(scaled_steps, 1000, base_steps)
end

function epsilon_scaled_tail(base_tail::Float64, current_epsilon::Float64)
    # The same concentration reduces the useful quadrature tail length.
    return max(6.0, base_tail / (1 + 0.5 * current_epsilon))
end

function multi_cdf(x, sigma)
    cache = multi_default_cache()
    total = 0.0
    @inbounds @simd for index in eachindex(cache.weights)
        total += cache.weights[index] *
                 cdf(_multi_default_normal, (x - cache.centers[index]) / sigma)
    end
    return total / cache.weight_sum
end

function sample_multi_gaussian(sigma; number = 1)
    cache = multi_default_cache()
    component = sample(eachindex(cache.centers), Weights(cache.normalized_weights))
    return rand(_multi_default_normal, number) .* sigma .+ cache.centers[component]
end

function multi_pdf(x, sigma)
    cache = multi_default_cache()
    inverse_two_sigma_squared = 1 / (2 * sigma^2)
    total = 0.0
    @inbounds @simd for index in eachindex(cache.weights)
        total += cache.weights[index] *
                 exp(-(x - cache.centers[index])^2 * inverse_two_sigma_squared)
    end
    return total / (sqrt(2 * pi) * sigma * cache.weight_sum)
end

function multi_l1_noise(sigma)
    cache = multi_default_cache()
    total = 0.0
    @inbounds @simd for index in eachindex(cache.weights)
        center = cache.centers[index]
        total += cache.weights[index] * (
            2 * sigma^2 * exp(-(center^2) / (2 * sigma^2)) +
            center * sqrt(2 * pi) * sigma *
            (1 - 2 * cdf(_multi_default_normal, -center / sigma))
        )
    end
    return total / (sqrt(2 * pi) * sigma * cache.weight_sum)
end

function multi_l2_squared_noise(sigma)
    cache = multi_default_cache()
    total = 0.0
    @inbounds @simd for index in eachindex(cache.weights)
        total += cache.weights[index] * (sigma^2 + cache.centers[index]^2)
    end
    return total / cache.weight_sum
end

function multi_default_privacy_integrand(x, shift, sigma, cache)
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
    eta_override = nothing,
    base_max_steps = 2000,
    base_tail_multiplier = 12.0,
    quadrature_atol = nothing,
    quadrature_rtol = 1e-6,
)
    cache = multi_default_cache()
    eta_local = isnothing(eta_override) ? eta : eta_override

    # The integral below measures only the violating (negative) part of
    # exp(epsilon) * p(x) - p(x + shift). A nonnegative shortfall certifies
    # the checked shift using (1 - eta) * delta, leaving eta * delta as the
    # numerical reserve used to set the shift-grid resolution.
    raw_spacing = sqrt(2 * pi) * sigma * delta * eta_local
    ideal_steps = ceil(Int, Delta / max(raw_spacing, eps()))
    max_steps = epsilon_scaled_step_cap(base_max_steps, Float64(epsilon))
    steps = clamp(ideal_steps, 400, max_steps)
    shift_spacing = Delta / steps

    atol = isnothing(quadrature_atol) ? max(delta * 1e-3, 1e-12) : quadrature_atol
    tail_multiplier = epsilon_scaled_tail(base_tail_multiplier, Float64(epsilon))
    smallest_shortfall = Inf

    # The capped number of shifts is part of the numerical specification and
    # should be revalidated if modified.
    for step in steps:-1:0
        shift = step * shift_spacing
        integration_limit = tail_multiplier * sigma + cache.K * Delta + shift
        shortfall = try
            quadgk(
                x -> multi_default_privacy_integrand(x, shift, sigma, cache),
                -integration_limit,
                integration_limit;
                atol = atol,
                rtol = quadrature_rtol,
            )[1] + (1 - eta_local) * delta
        catch
            return -1.0
        end

        if isnan(shortfall) || shortfall < 0
            return -1.0
        end
        smallest_shortfall = min(smallest_shortfall, shortfall)
    end
    return smallest_shortfall
end

function calibrate_multi_sigma(
    ;
    eta_override = nothing,
    base_max_steps = 2000,
    base_tail_multiplier = 12.0,
    quadrature_atol = nothing,
    quadrature_rtol = 1e-6,
)
    refresh_multi_default_cache!()
    eta_local = isnothing(eta_override) ? eta : eta_override

    # Brent's method can request an identical sigma more than once. Memoizing
    # this deterministic check removes repeated quadrature without changing
    # any acceptance decision.
    cached_shortfalls = Dict{Float64,Float64}()
    shortfall = sigma -> get!(cached_shortfalls, Float64(sigma)) do
        multi_privacy_shortfall(
            sigma;
            eta_override = eta_local,
            base_max_steps = base_max_steps,
            base_tail_multiplier = base_tail_multiplier,
            quadrature_atol = quadrature_atol,
            quadrature_rtol = quadrature_rtol,
        )
    end

    lower = 1.0
    while shortfall(lower) > 0
        lower /= 2
    end

    upper = sqrt(2 * log(1.25 / ((1 - eta_local) * delta))) * Delta / epsilon
    attempts = 0
    while shortfall(upper) < 0 && attempts < 30
        upper *= 2
        attempts += 1
    end
    shortfall(upper) < 0 && return upper

    return find_zero(shortfall, (lower, upper), method = Brent(), xatol = 1e-6, atol = 1e-6)
end
