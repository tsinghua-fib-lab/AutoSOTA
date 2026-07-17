#!/bin/bash
# Evaluation with output scale calibration (CODE-03) + LOO-optimized params
set -e
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export JULIA_DEPOT_PATH="/opt/julia_depot"
export GKSwstype=nul

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

PROBLEM="problems/rubric_v2.toml"
DATA="data/rubric_v2.npz"
RESULTS="results/rubric_v2_gpfvm.npz"

echo "=== GP-FVM Source Identification (Calibrated) ==="
echo "Problem: $PROBLEM"
echo ""

# Step 1: Generate data (only if needed)
if [ ! -f "$DATA" ]; then
    echo "--- Generating ground truth data ---"
    julia --project=/repo generate_data.jl \
        --problem "$PROBLEM" \
        --nx 36 --ny 36
fi

# Step 2: Run GP-FVM solver with calibration
echo "--- Running GP-FVM solver (calibrated output scale) ---"
julia --project=/repo run_gpfvm.jl \
    --problem "$PROBLEM" \
    --data "$DATA" \
    --rho 2.0 \
    --smoothness 2 \
    --calibrate \
    --benchmark \
    --benchmark-runs 3

# Step 3: Compute metrics
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
println("REPRODUCTION RESULTS (Calibrated)")
println("="^50)
println("Source RMSE (point eval, 36x36): ", round(source_rmse, digits=4))
println("Paper reported Source RMSE: 0.44")
println("PINN baseline Source RMSE: 0.76")
println()
JULIA_EOF

echo "=== Evaluation complete ==="
