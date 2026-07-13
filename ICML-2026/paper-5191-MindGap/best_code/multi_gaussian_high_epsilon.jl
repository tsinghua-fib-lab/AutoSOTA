# Multi-Gaussian calibration for epsilon >= 5.
#
# At high epsilon, mixture terms span many orders of magnitude. This branch
# evaluates mixture numerators in the log domain and integrates only windows
# surrounding mixture centers and their shifted counterparts.

const _multi_high_normal = Normal(0, 1)

struct MultiGaussianHighEpsilonCache
    epsilon::Float64
    Delta::Float64
    K::Int
    log_weights::Vector{Float64}
    centers::Vector{Float64}
    weight_sum::Float64
    inverse_normalizer_without_sigma::Float64
end

const _multi_high_cache = Ref{Union{Nothing,MultiGaussianHighEpsilonCache}}(nothing)

function refresh_multi_high_cache!()
    mixture_indices = collect(-K:K)
    log_weights = -abs.(mixture_indices) .* epsilon
    largest_log_weight = maximum(log_weights)
    weight_sum = exp(
        largest_log_weight + log(sum(exp.(log_weights .- largest_log_weight))),
    )
    _multi_high_cache[] = MultiGaussianHighEpsilonCache(
        Float64(epsilon),
        Float64(Delta),
        Int(K),
        log_weights,
        mixture_indices .* Delta,
        weight_sum,
        1 / (sqrt(2 * pi) * weight_sum),
    )
end

function multi_high_cache()
    cache = _multi_high_cache[]
    if cache === nothing ||
       cache.epsilon != epsilon ||
       cache.Delta != Delta ||
       cache.K != K
        refresh_multi_high_cache!()
        cache = _multi_high_cache[]
    end
    return cache::MultiGaussianHighEpsilonCache
end

function log_mixture_numerator(log_weights, x, centers, inverse_two_sigma_squared)
    maximum_term = -Inf
    @inbounds for index in eachindex(log_weights)
        term = log_weights[index] -
               (x - centers[index])^2 * inverse_two_sigma_squared
        maximum_term = max(maximum_term, term)
    end
    maximum_term == -Inf && return -Inf

    shifted_sum = 0.0
    @inbounds for index in eachindex(log_weights)
        term = log_weights[index] -
               (x - centers[index])^2 * inverse_two_sigma_squared
        shifted_sum += exp(term - maximum_term)
    end
    return maximum_term + log(shifted_sum)
end

function multi_l2_squared_noise(sigma)
    cache = multi_high_cache()
    weights = exp.(cache.log_weights)
    total = sum(
        weights[index] * (sigma^2 + cache.centers[index]^2)
        for index in eachindex(weights)
    )
    return total / cache.weight_sum
end

function multi_l1_noise(sigma)
    cache = multi_high_cache()
    weights = exp.(cache.log_weights)
    total = 0.0
    @inbounds for index in eachindex(weights)
        center = cache.centers[index]
        total += weights[index] * (
            2 * sigma^2 * exp(-(center^2) / (2 * sigma^2)) +
            center * sqrt(2 * pi) * sigma *
            (1 - 2 * cdf(_multi_high_normal, -center / sigma))
        )
    end
    return total / (sqrt(2 * pi) * sigma * cache.weight_sum)
end

function multi_high_privacy_integrand(x, shift, sigma, cache)
    inverse_two_sigma_squared = 1 / (2 * sigma^2)
    log_density = log_mixture_numerator(
        cache.log_weights,
        x,
        cache.centers,
        inverse_two_sigma_squared,
    )
    log_shifted_density = log_mixture_numerator(
        cache.log_weights,
        x + shift,
        cache.centers,
        inverse_two_sigma_squared,
    )

    log_scaled_density = cache.epsilon + log_density
    log_scaled_density >= log_shifted_density && return 0.0

    # Compute exp(log_scaled_density) - exp(log_shifted_density) without
    # subtracting two nearly equal small floating-point numbers.
    relative_log_difference = log_scaled_density - log_shifted_density
    violating_difference = exp(log_shifted_density) * expm1(relative_log_difference)
    return violating_difference * (cache.inverse_normalizer_without_sigma / sigma)
end

function merge_integration_windows(hotspots, half_width)
    intervals = Vector{Tuple{Float64,Float64}}()
    isempty(hotspots) && return intervals

    start_point = hotspots[1] - half_width
    end_point = hotspots[1] + half_width
    for index in 2:length(hotspots)
        candidate_start = hotspots[index] - half_width
        candidate_end = hotspots[index] + half_width
        if candidate_start <= end_point
            end_point = max(end_point, candidate_end)
        else
            push!(intervals, (start_point, end_point))
            start_point, end_point = candidate_start, candidate_end
        end
    end
    push!(intervals, (start_point, end_point))
    return intervals
end

function multi_privacy_shortfall(
    sigma;
    eta_override = nothing,
    maximum_steps = 300,
    quadrature_atol = 1e-12,
)
    cache = multi_high_cache()
    eta_local = isnothing(eta_override) ? eta : eta_override
    raw_spacing = sqrt(2 * pi) * sigma * delta * eta_local
    ideal_steps = ceil(Int, Delta / max(raw_spacing, eps()))
    steps = clamp(ideal_steps, 50, maximum_steps)
    shift_spacing = Delta / steps
    window_half_width = 8.0 * sigma
    smallest_shortfall = Inf

    for step in 0:steps
        shift = step == steps ? Delta : step * shift_spacing
        hotspots = Float64[]
        @inbounds for center in cache.centers
            push!(hotspots, center)
            push!(hotspots, center - shift)
        end
        sort!(hotspots)

        violating_integral = 0.0
        for (lower, upper) in merge_integration_windows(hotspots, window_half_width)
            value, _ = quadgk(
                x -> multi_high_privacy_integrand(x, shift, sigma, cache),
                lower,
                upper;
                atol = quadrature_atol,
            )
            violating_integral += value
        end

        shortfall = violating_integral + (1 - eta_local) * delta
        if isnan(shortfall) || shortfall < 0
            return -1.0
        end
        smallest_shortfall = min(smallest_shortfall, shortfall)
    end
    return smallest_shortfall
end

function calibrate_multi_sigma(; eta_override = nothing, maximum_steps = 300, quadrature_atol = 1e-12)
    refresh_multi_high_cache!()
    eta_local = isnothing(eta_override) ? eta : eta_override
    cached_shortfalls = Dict{Float64,Float64}()
    shortfall = sigma -> get!(cached_shortfalls, Float64(sigma)) do
        multi_privacy_shortfall(
            sigma;
            eta_override = eta_local,
            maximum_steps = maximum_steps,
            quadrature_atol = quadrature_atol,
        )
    end

    upper = 1.0
    while shortfall(upper) < 0 && upper < 100.0
        upper *= 2
    end
    lower = 1e-7
    while shortfall(lower) > 0 && lower > 1e-15
        lower /= 2
    end
    return find_zero(shortfall, (lower, upper), method = Brent(), xatol = 1e-9, atol = 1e-9)
end
