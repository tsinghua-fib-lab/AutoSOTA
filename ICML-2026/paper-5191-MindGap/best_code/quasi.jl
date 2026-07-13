using Distributions, Optim, Roots

# Public entry point for the quasi-Gaussian mechanism. The low-epsilon,
# small-delta branch evaluates the same calibration equations with tighter root
# tolerances, where the boundary is most sensitive numerically.
for parameter in (:epsilon, :delta, :Delta)
    isdefined(@__MODULE__, parameter) ||
        error("Define epsilon, delta, and sensitivity/Delta before including quasi.jl.")
end

if epsilon <= 0.5 && delta <= 1e-3
    const QUASI_GAUSSIAN_REGIME = "low epsilon and small delta (epsilon <= 0.5, delta <= 1e-3)"
    include(joinpath(@__DIR__, "quasi_gaussian_low_epsilon_small_delta.jl"))
else
    const QUASI_GAUSSIAN_REGIME = "default numerical regime"
    include(joinpath(@__DIR__, "quasi_gaussian_default.jl"))
end

function quasi_gaussian_metrics()
    sigma = calibrate_quasi_sigma()
    l2_squared = quasi_l2_squared_noise(sigma)
    return (
        sigma = sigma,
        l1_noise = quasi_l1_noise(sigma),
        l2_noise = sqrt(l2_squared),
        l2_squared_noise = l2_squared,
        regime = QUASI_GAUSSIAN_REGIME,
    )
end
