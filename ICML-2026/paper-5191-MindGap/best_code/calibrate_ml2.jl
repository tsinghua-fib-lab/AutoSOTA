using Printf

epsilon = 1.0
delta = 1e-3
Delta = 1.0
K = 10
eta = 0.01

cd("/repo")
include("multi.jl")
include("quasi.jl")
include("analytic_gaussian.jl")

N = 336
Delta_val = 2.0
delta_global = 1.0 / N^2

# zCDP approach sigma
log_inv_delta = log(1.0 / delta_global)
y = (-2*sqrt(log_inv_delta) + sqrt(4*log_inv_delta + 4*5.0)) / 2
rho_total = y^2
rho_per = rho_total / 200
sigma_zcdp = Delta_val / sqrt(2 * rho_per)
@printf "sigma_zcdp = %.6f\n" sigma_zcdp

# Now: M-G with sigma_zcdp and per-update eps=0.324
# The M-G mixture uses epsilon in its weights
# For a given (sigma, epsilon, Delta, K), the M-G provides some (eps_priv, delta_priv) DP guarantee
# We want to compute the L1 noise of this mixture distribution
# 
# The L1 noise function multi_l1_noise(sigma) uses the current global epsilon for weights
# So we need to set the global epsilon, refresh the cache, and compute L1

global epsilon, delta, Delta
epsilon = 0.324266  # per-update epsilon from zCDP
delta = delta_global
Delta = Delta_val

# Compute L1 of M-G with sigma_zcdp and epsilon=0.324 in the mixture
# We need to force-recompute the cache for these parameters
sigma_test = sigma_zcdp  # 29.96

# The L1 noise function uses multi_default_cache() which depends on global epsilon, Delta, K
# Set global epsilon for the mixture weights
l1_zcdp_mg = multi_l1_noise(sigma_test)
@printf "M-G (eps_mix=%.4f, sigma=%.4f): L1=%.6f\n" epsilon sigma_test l1_zcdp_mg

# Compare with AG
l1_zcdp_ag = sigma_test * sqrt(2/pi)
@printf "A-G (sigma=%.4f): L1=%.6f\n" sigma_test l1_zcdp_ag

# Try different approach: calibrate all mechanisms for ONE-SHOT (eps_global, delta_global)
@printf "\n=== One-shot calibration for GLOBAL privacy (eps=5, delta=%.2e) ===\n" delta_global

epsilon = 5.0
delta = delta_global
Delta = Delta_val
K = 10
eta = 0.01

# AG
sigma_AG_global = calibrate_analytic_gaussian(5.0, delta_global, Delta_val)
@printf "AG global: sigma=%.4f, L1=%.4f\n" sigma_AG_global sigma_AG_global*sqrt(2/pi)

# MG
refresh_multi_default_cache!()
sigma_MG_global = calibrate_multi_sigma()
l1_MG_global = multi_l1_noise(sigma_MG_global)
@printf "MG global: sigma=%.4f, L1=%.4f\n" sigma_MG_global l1_MG_global

# QG
sigma_QG_global = nothing
try
    global epsilon, delta, Delta
    epsilon = 5.0
    delta = delta_global
    Delta = Delta_val
    sigma_QG_global = calibrate_quasi_sigma()
    l1_QG_global = quasi_l1_noise(sigma_QG_global)
    @printf "QG global: sigma=%.4f, L1=%.4f\n" sigma_QG_global l1_QG_global
catch e
    @printf "QG global: ERROR %s\n" e
end

# After 200 compositions with zCDP:
@printf "\n=== After 200 updates (zCDP composition) ===\n"
rho_per_AG = Delta_val^2 / (2*sigma_AG_global^2)
rho_tot_AG = 200 * rho_per_AG
eps_comp_AG = rho_tot_AG + 2*sqrt(rho_tot_AG * log_inv_delta)
@printf "AG: composed eps=%.2f (target 5.0)\n" eps_comp_AG

rho_per_MG = Delta_val^2 / (2*sigma_MG_global^2)
rho_tot_MG = 200 * rho_per_MG
eps_comp_MG = rho_tot_MG + 2*sqrt(rho_tot_MG * log_inv_delta)
@printf "MG: composed eps=%.2f (target 5.0)\n" eps_comp_MG
