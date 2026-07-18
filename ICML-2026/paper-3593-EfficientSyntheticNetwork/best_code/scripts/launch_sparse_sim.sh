#!/bin/bash
# Launch sparse-simulation runs for GRAN + EDGE across r ∈ {2,3,4} × seeds 0..19.
#
# Assumes:
#   - data/sparse_sim/r=<r>/seed=<s>{.pkl,_A.npy} already generated
#     (run data/generate_sparse_sim.py first)
#   - baselines/{gran,edge}/src/ populated with the upstream clones
#
# Two run modes:
#   * local (default): GNU parallel locally, one job at a time per GPU
#   * SLURM           : set MODE=slurm, submits each (method, r, seed) as a
#                       job. Use SLURM_CONSTRAINT="<feature1|feature2>" to
#                       restrict to PyTorch-compatible GPU hardware (the
#                       stock wheel supports up to sm_90; map to your
#                       cluster's feature labels).

set -euo pipefail
RELEASE_ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$RELEASE_ROOT"

MODE=${MODE:-local}
CONCURRENCY=${CONCURRENCY:-4}
SLURM_CONSTRAINT=${SLURM_CONSTRAINT:-""}

run_one_local() {
  local method=$1 r=$2 seed=$3
  local out="runs/$method/r=$r/seed=$seed"
  if [[ -d "$out/samples" && $(ls "$out/samples"/*.npy 2>/dev/null | wc -l) -ge 200 ]]; then
    echo "skip $method r=$r seed=$seed (already 200 samples)"
    return
  fi
  python baselines/$method/run.py \
      --config baselines/$method/configs/sparse_sim.yaml \
      --r $r --seed $seed --output "$out"
}

run_one_slurm() {
  local method=$1 r=$2 seed=$3
  local out="runs/$method/r=$r/seed=$seed"
  local time="2:00:00"; local mem="64G"
  [[ "$method" == "edge" ]] && time="4:00:00"
  mkdir -p runs/logs
  local constraint_arg=""
  [[ -n "$SLURM_CONSTRAINT" ]] && constraint_arg="--constraint=$SLURM_CONSTRAINT"
  sbatch --job-name="${method}_r${r}_s${seed}" \
         --partition=gpu --gpus=1 --cpus-per-task=4 --mem=$mem \
         --time=$time $constraint_arg \
         --output="runs/logs/${method}_r${r}_s${seed}_%j.out" \
         --error="runs/logs/${method}_r${r}_s${seed}_%j.err" \
         --wrap "set -e; cd $RELEASE_ROOT; python baselines/$method/run.py --config baselines/$method/configs/sparse_sim.yaml --r $r --seed $seed --output $out"
}

export -f run_one_local

if [[ "$MODE" == "slurm" ]]; then
  for method in gran edge; do
    for r in 2 3 4; do
      for s in $(seq 0 19); do
        run_one_slurm "$method" "$r" "$s"
      done
    done
  done
elif command -v parallel >/dev/null; then
  parallel -j "$CONCURRENCY" run_one_local ::: gran edge ::: 2 3 4 ::: $(seq 0 19)
else
  for method in gran edge; do
    for r in 2 3 4; do
      for s in $(seq 0 19); do
        run_one_local "$method" "$r" "$s"
      done
    done
  done
fi
