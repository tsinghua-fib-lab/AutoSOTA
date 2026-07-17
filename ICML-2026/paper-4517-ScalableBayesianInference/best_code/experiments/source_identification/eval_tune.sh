#!/bin/bash
# Targeted parameter tuning: shorter lengthscales + output_scale
set -e
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export JULIA_DEPOT_PATH="/opt/julia_depot"
export GKSwstype=nul

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

PROBLEM="problems/rubric_v2.toml"
DATA="data/rubric_v2.npz"

# Generate data if needed
if [ ! -f "$DATA" ]; then
    julia --project=/repo generate_data.jl --problem "$PROBLEM" --nx 36 --ny 36
fi

run_one() {
    local label="$1" os="$2" lc="$3" ls="$4" rho="$5"
    echo "--- $label ---"
    julia --project=/repo run_gpfvm.jl \
        --problem "$PROBLEM" --data "$DATA" \
        --rho "$rho" --smoothness 2 \
        --lengthscale-c "$lc" --lengthscale-s "$ls" \
        --output-scale "$os" \
        --quiet 2>&1 | tail -5
    rmse=$(julia --project=/repo -e '
        using NPZ; data=npzread("data/rubric_v2.npz"); result=npzread("results/rubric_v2_gpfvm.npz")
        println(round(sqrt(mean((result["s_mean"].-data["s_true"]).^2)), digits=5))
    ' 2>/dev/null)
    echo "  RMSE=$rmse"
    echo "$label | os=$os lc=$lc ls=$ls rho=$rho | RMSE=$rmse" >> results/tuning_results.txt
}

mkdir -p results
echo "# Tuning results $(date)" > results/tuning_results.txt

# Test shorter lengthscales (better source resolution)
# Combined with different output scales and rho values

# Baseline check
run_one "baseline" "1.0" "0.1429" "0.1429" "2.0"

# Shorter lengthscales
run_one "short-ls"  "1.0" "0.10" "0.10" "2.0"
run_one "vshort-ls" "1.0" "0.08" "0.08" "2.0"
run_one "vshort-ls-os05" "0.5" "0.08" "0.08" "2.0"
run_one "short-ls-os05" "0.5" "0.10" "0.10" "2.0"

# Different rho for concentration
run_one "short-rho3" "1.0" "0.10" "0.10" "3.0"

# Longer source lengthscale but shorter concentration
run_one "asym-1" "1.0" "0.08" "0.15" "2.0"
run_one "asym-2" "1.0" "0.10" "0.18" "2.0"

echo ""
echo "=== Results ==="
cat results/tuning_results.txt
