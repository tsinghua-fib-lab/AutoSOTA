using Printf

# Define required globals BEFORE including the mechanism files
epsilon = 1.0
delta = 1e-3
Delta = 1.0
K = 10
eta = 0.01

cd("/repo")
include("multi.jl")
include("quasi.jl")
include("analytic_gaussian.jl")

# ML experiment parameters
N = 336
Delta_val = 2.0
eps_global = 5.0
delta_global = 1.0 / N^2
T_val = 100
P_val = 2
total_updates = T_val * P_val

# zCDP: solve for rho_total
log_inv_delta = log(1.0 / delta_global)
y = (-2*sqrt(log_inv_delta) + sqrt(4*log_inv_delta + 4*eps_global)) / 2
rho_total = y^2
rho_per = rho_total / total_updates
sigma_zcdp = Delta_val / sqrt(2 * rho_per)

@printf "=== ML Experiment Noise Calibration ===\n"
@printf "N=%d, d=%d, T=%d, P=%d\n" N 8 T_val P_val
@printf "Global: eps=%.1f, delta=%.6e\n" eps_global delta_global
@printf "zCDP: rho_total=%.6f, rho_per=%.8f, sigma=%.6f\n\n" rho_total rho_per sigma_zcdp

# A-G L1 noise
l1_AG = sigma_zcdp * sqrt(2/pi)
@printf "A-G: sigma=%.4f, L1=%.6f\n" sigma_zcdp l1_AG

# Calibrate M-G for various epsilon values
@printf "\n--- Multi-Gaussian calibration (Delta=%.1f, delta=%.2e, K=%d, eta=%.2f) ---\n" Delta_val delta_global K eta
for eps_val in [0.1, 0.2, 0.324266, 0.5, 1.0, 2.0, 5.0]
    global epsilon, delta, Delta
    epsilon = eps_val
    delta = delta_global
    Delta = Delta_val
    
    try
        sigma_MG = calibrate_multi_sigma()
        l1_MG = multi_l1_noise(sigma_MG)
        l2_MG = sqrt(multi_l2_squared_noise(sigma_MG))
        @printf "eps=%.4f: sigma=%.6f, L1=%.6f, L2=%.6f\n" eps_val sigma_MG l1_MG l2_MG
    catch e
        @printf "eps=%.4f: ERROR %s\n" eps_val e
    end
end

# Q-G calibration
@printf "\n--- Quasi-Gaussian calibration (Delta=%.1f, delta=%.2e) ---\n" Delta_val delta_global
for eps_val in [0.1, 0.2, 0.324266, 0.5, 1.0, 2.0, 5.0]
    global epsilon, delta, Delta
    epsilon = eps_val
    delta = delta_global
    Delta = Delta_val
    
    try
        sigma_QG = calibrate_quasi_sigma()
        l1_QG = quasi_l1_noise(sigma_QG)
        l2_QG = sqrt(quasi_l2_squared_noise(sigma_QG))
        @printf "eps=%.4f: sigma=%.6f, L1=%.6f, L2=%.6f\n" eps_val sigma_QG l1_QG l2_QG
    catch e
        @printf "eps=%.4f: ERROR %s\n" eps_val e
    end
end
