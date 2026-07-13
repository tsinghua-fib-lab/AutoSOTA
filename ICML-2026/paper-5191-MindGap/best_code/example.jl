using Printf

# ---------------------------------------------------------------------------
# Inputs: edit only these values for a new privacy-calibration instance.
# ---------------------------------------------------------------------------
epsilon = 2.0
delta = 0.01
sensitivity = 1.0

# Model settings used in the paper experiments.
# Delta is the notation used in the mechanism derivations for sensitivity.
Delta = sensitivity
K = 10 #change between 1-20
eta = 0.01

include(joinpath(@__DIR__, "multi.jl"))
include(joinpath(@__DIR__, "quasi.jl"))
include(joinpath(@__DIR__, "analytic_gaussian.jl"))

multi_result = multi_gaussian_metrics()
quasi_result = quasi_gaussian_metrics()

analytic_sigma = calibrate_analytic_gaussian(epsilon, delta, sensitivity)
analytic_l2_squared = analytic_sigma^2
analytic_result = (
    sigma = analytic_sigma,
    l1_noise = analytic_sigma * sqrt(2 / pi),
    l2_noise = sqrt(analytic_l2_squared),
    l2_squared_noise = analytic_l2_squared,
    regime = "analytic Gaussian baseline",
)

# Retain a structured value for users who include this file from another script.
results = (
    multi_gaussian = multi_result,
    quasi_gaussian = quasi_result,
    analytic_gaussian = analytic_result,
)

println("Privacy parameters")
@printf("  epsilon:     %.6g\n", epsilon)
@printf("  delta:       %.6g\n", delta)
@printf("  sensitivity: %.6g\n\n", sensitivity)

println("Utility quantities for scalar additive noise Z")
println("  L1 noise         = E[|Z|] (amplitude)")
println("  L2 noise         = sqrt(E[Z^2])")
println("  L2 squared noise = E[Z^2] (power)")
println()

function print_result(name, result)
    println(name, " [", result.regime, "]")
    @printf("  sigma:            %.8f\n", result.sigma)
    @printf("  L1 noise:         %.8f\n", result.l1_noise)
    @printf("  L2 noise:         %.8f\n", result.l2_noise)
    @printf("  L2 squared noise: %.8f\n\n", result.l2_squared_noise)
end

print_result("Multi Gaussian", multi_result)
print_result("Quasi Gaussian", quasi_result)
print_result("Analytic Gaussian", analytic_result)
