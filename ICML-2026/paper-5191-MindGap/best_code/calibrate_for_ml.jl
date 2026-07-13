using Printf
using JSON

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
eps_global = 5.0
delta_global = 1.0 / N^2
T_val = 100
P_val = 2
total_updates = T_val * P_val

# zCDP: solve for per-update privacy
log_inv_delta = log(1.0 / delta_global)
y = (-2*sqrt(log_inv_delta) + sqrt(4*log_inv_delta + 4*eps_global)) / 2
rho_total = y^2
rho_per = rho_total / total_updates
eps_per = rho_per + 2 * sqrt(rho_per * log_inv_delta)

# Approach 1: zCDP (used for A-G)
sigma_AG_zcdp = Delta_val / sqrt(2 * rho_per)

# Approach 2: One-shot per-update calibration (used for M-G and Q-G)
global epsilon, delta, Delta
epsilon = eps_per
delta = delta_global
Delta = Delta_val

# M-G
sigma_MG = calibrate_multi_sigma()
l1_MG = multi_l1_noise(sigma_MG)
l2_MG = sqrt(multi_l2_squared_noise(sigma_MG))

# Q-G
sigma_QG = calibrate_quasi_sigma()
l1_QG = quasi_l1_noise(sigma_QG)
l2_QG = sqrt(quasi_l2_squared_noise(sigma_QG))

# A-G (direct one-shot calibration for comparison)
sigma_AG_direct = calibrate_analytic_gaussian(eps_per, delta_global, Delta_val)
l1_AG = sigma_AG_direct * sqrt(2/pi)

results = Dict(
    "N" => N,
    "d" => 8,
    "T" => T_val,
    "P" => P_val,
    "total_updates" => total_updates,
    "eps_global" => eps_global,
    "delta_global" => delta_global,
    "rho_per" => rho_per,
    "eps_per" => eps_per,
    "delta_per" => delta_global,
    "Delta" => Delta_val,
    "AG" => Dict("sigma" => sigma_AG_direct, "l1" => l1_AG, "method" => "one_shot"),
    "MG" => Dict("sigma" => sigma_MG, "l1" => l1_MG, "l2" => l2_MG, "K" => K, "eta" => eta, "method" => "one_shot"),
    "QG" => Dict("sigma" => sigma_QG, "l1" => l1_QG, "l2" => l2_QG, "method" => "one_shot"),
)

println(JSON.json(results))
