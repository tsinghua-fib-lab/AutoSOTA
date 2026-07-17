#!/bin/bash
# Optimized evaluation for paper-4517
# Iteration 1: LOO-CV kernel hyperparameter selection (IDEA-ALGO-01)
set -e
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export JULIA_DEPOT_PATH="/opt/julia_depot"
export GKSwstype=nul

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

PROBLEM="problems/rubric_v2.toml"
DATA="data/rubric_v2.npz"
RESULTS="results/rubric_v2_gpfvm.npz"
PARAMS_FILE="results/best_params_loo.txt"

echo "=== GP-FVM Source Identification (Optimized) ==="
echo "Problem: $PROBLEM"
echo ""

# Step 1: Generate data (only if needed)
if [ ! -f "$DATA" ]; then
    echo "--- Generating ground truth data ---"
    julia --project=/repo generate_data.jl \
        --problem "$PROBLEM" \
        --nx 36 --ny 36
fi

# Step 2: LOO-CV hyperparameter optimization
echo "--- LOO-CV hyperparameter optimization ---"
julia --project=/repo select_params.jl

# Step 3: Read optimized params
if [ ! -f "$PARAMS_FILE" ]; then
    echo "WARNING: Params file not found, using defaults"
    LCS=""
    LSS=""
else
    source "$PARAMS_FILE"
    LCS="${BEST_LCS:-}"
    LSS="${BEST_LSS:-}"
fi

echo "Optimized params: lengthscale_c=$LCS, lengthscale_s=$LSS"

# Step 4: Run GP-FVM solver with optimized params
echo ""
echo "--- Running GP-FVM solver (optimized params) ---"
LCS_ARG=""
LSS_ARG=""
if [ -n "$LCS" ]; then LCS_ARG="--lengthscale-c $LCS"; fi
if [ -n "$LSS" ]; then LSS_ARG="--lengthscale-s $LSS"; fi

julia --project=/repo run_gpfvm.jl \
    --problem "$PROBLEM" \
    --data "$DATA" \
    --rho 2.0 \
    --smoothness 2 \
    $LCS_ARG $LSS_ARG \
    --benchmark \
    --benchmark-runs 3

# Step 5: Compute metrics
echo ""
echo "--- Computing metrics ---"
julia --project=/repo << 'JULIA_EOF'
using NPZ, Statistics

data = npzread("data/rubric_v2.npz")
result = npzread("results/rubric_v2_gpfvm.npz")

s_true = data["s_true"]
s_mean = result["s_mean"]

source_rmse = sqrt(mean((s_mean .- s_true).^2))

println("="^50)
println("REPRODUCTION RESULTS (Optimized)")
println("="^50)
println("Source RMSE (point eval, 36x36): ", round(source_rmse, digits=4))
println("Paper reported Source RMSE: 0.44")
println("PINN baseline Source RMSE: 0.76")
println()
JULIA_EOF

echo "=== Evaluation complete ==="
