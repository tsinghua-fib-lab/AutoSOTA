using Distributions, StatsBase, Roots, QuadGK

# Public entry point for the multi-Gaussian mechanism.
#
# The three included files implement the same mechanism but use numerical
# routines tuned for different privacy regimes. Keeping selection here means a
# reader never has to choose an implementation file manually.
for parameter in (:epsilon, :delta, :Delta, :K, :eta)
    isdefined(@__MODULE__, parameter) ||
        error("Define epsilon, delta, sensitivity/Delta, K, and eta before including multi.jl.")
end

if epsilon >= 5.0
    const MULTI_GAUSSIAN_REGIME = "high epsilon (epsilon >= 5)"
    include(joinpath(@__DIR__, "multi_gaussian_high_epsilon.jl"))
elseif epsilon <= 0.5 && delta <= 1e-3
    const MULTI_GAUSSIAN_REGIME = "low epsilon and small delta (epsilon <= 0.5, delta <= 1e-3)"
    include(joinpath(@__DIR__, "multi_gaussian_low_epsilon_small_delta.jl"))
else
    const MULTI_GAUSSIAN_REGIME = "default numerical regime"
    include(joinpath(@__DIR__, "multi_gaussian_default.jl"))
end

function multi_gaussian_metrics()
    sigma = calibrate_multi_sigma()
    l2_squared = multi_l2_squared_noise(sigma)
    return (
        sigma = sigma,
        l1_noise = multi_l1_noise(sigma),
        l2_noise = sqrt(l2_squared),
        l2_squared_noise = l2_squared,
        regime = MULTI_GAUSSIAN_REGIME,
    )
end
