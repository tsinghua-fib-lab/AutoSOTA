#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
EXP_DIR="$REPO_ROOT/experiments/section_4_2_dynamics"
RUNNER="run_model_reproduce_4_2.py"
PYTHON_BIN="${SECTION_4_2_PYTHON:-python3}"
PROJECT="${WANDB_PROJECT:-}"
ENTITY="${WANDB_ENTITY:-}"
NUM_EPOCH="${SECTION_4_2_NUM_EPOCH:-1000}"
SEEDS="${SECTION_4_2_SEEDS:-0 1 2 3 4}"
SLURM_SUBMIT=""
DRY_RUN=0
SYMMETRY_FILTER="all"
FAMILY_FILTER="all"

usage() {
  cat <<'EOF'
Usage: bash scripts/reproduce_4_2.sh [options] [target ...]

Targets:
  all              Run all Section 4.2 dynamics experiments.
  group            Run group-convolution cases.
  steerable        Run steerable cases.
  translation      Run Translation cases.
  rotation         Run Rotation cases.
  scale            Run Scale cases.

Options:
  --slurm-submit FILE   Submit each run through FILE with sbatch.
  --dry-run             Print commands without executing them.
  --num-epoch N         Override training epochs. Default: SECTION_4_2_NUM_EPOCH or 1000.
  --seeds "..."         Override seeds. Default: SECTION_4_2_SEEDS or "0 1 2 3 4".
  --symmetry NAME       all, translation, rotation, or scale.
  --model NAME          all, group, or steerable.
  --wandb-project NAME  Enable W&B logging for this project.
  --wandb-entity NAME   Optional W&B entity. Used only when W&B logging is enabled.
  -h, --help            Show this help.

Examples:
  bash scripts/reproduce_4_2.sh translation group
  bash scripts/reproduce_4_2.sh --symmetry rotation --model steerable
  bash scripts/reproduce_4_2.sh scale

Translation has no steerable model in this setup, so `translation steerable`
is rejected.

Environment:
  SECTION_4_2_PYTHON   Python executable. Default: python3.
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --slurm-submit)
      if [ "$#" -lt 2 ]; then
        echo "--slurm-submit requires a file argument" >&2
        exit 2
      fi
      SLURM_SUBMIT="$2"
      shift 2
      ;;
    --slurm-submit=*)
      SLURM_SUBMIT="${1#*=}"
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --num-epoch)
      if [ "$#" -lt 2 ]; then
        echo "--num-epoch requires a value" >&2
        exit 2
      fi
      NUM_EPOCH="$2"
      shift 2
      ;;
    --num-epoch=*)
      NUM_EPOCH="${1#*=}"
      shift
      ;;
    --seeds)
      if [ "$#" -lt 2 ]; then
        echo "--seeds requires a quoted seed list" >&2
        exit 2
      fi
      SEEDS="$2"
      shift 2
      ;;
    --seeds=*)
      SEEDS="${1#*=}"
      shift
      ;;
    --symmetry)
      if [ "$#" -lt 2 ]; then
        echo "--symmetry requires a value: all, translation, rotation, or scale" >&2
        exit 2
      fi
      SYMMETRY_FILTER="$2"
      shift 2
      ;;
    --symmetry=*)
      SYMMETRY_FILTER="${1#*=}"
      shift
      ;;
    --model|--family)
      if [ "$#" -lt 2 ]; then
        echo "$1 requires a value: all, group, or steerable" >&2
        exit 2
      fi
      FAMILY_FILTER="$2"
      shift 2
      ;;
    --model=*|--family=*)
      FAMILY_FILTER="${1#*=}"
      shift
      ;;
    --wandb-project)
      if [ "$#" -lt 2 ]; then
        echo "--wandb-project requires a value" >&2
        exit 2
      fi
      PROJECT="$2"
      shift 2
      ;;
    --wandb-project=*)
      PROJECT="${1#*=}"
      shift
      ;;
    --wandb-entity)
      if [ "$#" -lt 2 ]; then
        echo "--wandb-entity requires a value" >&2
        exit 2
      fi
      ENTITY="$2"
      shift 2
      ;;
    --wandb-entity=*)
      ENTITY="${1#*=}"
      shift
      ;;
    -h|--help|help)
      usage
      exit 0
      ;;
    *)
      break
      ;;
  esac
done

if [ -n "$SLURM_SUBMIT" ] && [[ "$SLURM_SUBMIT" != /* ]]; then
  SLURM_SUBMIT="$REPO_ROOT/$SLURM_SUBMIT"
fi

cd "$EXP_DIR"

run_cmd() {
  if [ "$DRY_RUN" -eq 1 ]; then
    if [ -n "$SLURM_SUBMIT" ]; then
      printf 'sbatch %q' "$SLURM_SUBMIT"
      printf ' %q' "$@"
      printf '\n'
    else
      printf '%q' "$1"
      shift
      printf ' %q' "$@"
      printf '\n'
    fi
    return
  fi

  if [ -n "$SLURM_SUBMIT" ]; then
    sbatch "$SLURM_SUBMIT" "$@"
  else
    "$@"
  fi
}

run_model() {
  local seed="$1"
  local symmetry="$2"
  local regulariser="$3"
  local use_steer="$4"
  local hidden_dim="$5"
  local num_layers="$6"
  local out_length="$7"
  local num_banks="$8"
  local num_scales="$9"
  local alpha="${10}"
  local batch_size="${11}"
  local learning_rate="${12}"
  local decay_rate="${13}"
  local cmd=(
    "$PYTHON_BIN" "$RUNNER"
    --dataset=PhiFlow \
    --relaxed_symmetry="$symmetry" \
    --regulariser="$regulariser" \
    --use_steer="$use_steer" \
    --hidden_dim="$hidden_dim" \
    --num_layers="$num_layers" \
    --out_length="$out_length" \
    --num_banks="$num_banks" \
    --num_scales="$num_scales" \
    --alpha="$alpha" \
    --batch_size="$batch_size" \
    --learning_rate="$learning_rate" \
    --decay_rate="$decay_rate" \
    --num_epoch="$NUM_EPOCH" \
    --seed="$seed"
  )

  if [ -n "$PROJECT" ]; then
    cmd+=(--project="$PROJECT")
  fi

  if [ -n "$ENTITY" ]; then
    cmd+=(--entity="$ENTITY")
  fi

  run_cmd "${cmd[@]}"
}

normalize_filters() {
  case "${SYMMETRY_FILTER,,}" in
    all)
      SYMMETRY_FILTER="all"
      ;;
    translation|trans)
      SYMMETRY_FILTER="Translation"
      ;;
    rotation|rot)
      SYMMETRY_FILTER="Rotation"
      ;;
    scale|scaling)
      SYMMETRY_FILTER="Scale"
      ;;
    *)
      echo "Unknown Section 4.2 symmetry: $SYMMETRY_FILTER" >&2
      usage >&2
      exit 2
      ;;
  esac

  case "${FAMILY_FILTER,,}" in
    all)
      FAMILY_FILTER="all"
      ;;
    group|group-conv|groupconv)
      FAMILY_FILTER="group"
      ;;
    steerable|steer|steerability)
      FAMILY_FILTER="steerable"
      ;;
    *)
      echo "Unknown Section 4.2 model family: $FAMILY_FILTER" >&2
      usage >&2
      exit 2
      ;;
  esac

  if [ "$SYMMETRY_FILTER" = "Translation" ] && [ "$FAMILY_FILTER" = "steerable" ]; then
    echo "Invalid Section 4.2 selection: Translation has no steerable model. Use 'translation group'." >&2
    exit 2
  fi
}

matches_selection() {
  local symmetry="$1"
  local family="$2"

  if [ "$SYMMETRY_FILTER" != "all" ] && [ "$SYMMETRY_FILTER" != "$symmetry" ]; then
    return 1
  fi

  if [ "$FAMILY_FILTER" != "all" ] && [ "$FAMILY_FILTER" != "$family" ]; then
    return 1
  fi

  return 0
}

run_case() {
  local symmetry="$1"
  local family="$2"
  local seed

  matches_selection "$symmetry" "$family" || return 0

  for seed in $SEEDS; do
    case "$symmetry:$family" in
      Translation:group)
        run_model "$seed" Translation projection false 128 5 6 2 2 10 8 0.001 0.95
        ;;
      Rotation:steerable)
        run_model "$seed" Rotation projection true 92 5 6 2 2 0.001 16 0.001 0.95
        ;;
      Scale:steerable)
        run_model "$seed" Scale projection true 64 5 6 2 2 0.0001 8 0.0001 0.95
        ;;
      Rotation:group)
        run_model "$seed" Rotation projection false 128 3 6 4 2 0.01 8 0.0048562 0.95
        ;;
      Scale:group)
        run_model "$seed" Scale projection false 128 6 6 3 1 0.000001 8 0.007743 0.95
        ;;
      *)
        echo "Unsupported Section 4.2 case: $symmetry $family" >&2
        exit 2
        ;;
    esac
  done
}

run_selected() {
  normalize_filters

  run_case Translation group
  run_case Rotation steerable
  run_case Scale steerable
  run_case Rotation group
  run_case Scale group
}

run_target() {
  case "$1" in
    all)
      SYMMETRY_FILTER="all"
      FAMILY_FILTER="all"
      ;;
    group)
      FAMILY_FILTER="group"
      ;;
    steerable)
      FAMILY_FILTER="steerable"
      ;;
    translation|trans)
      SYMMETRY_FILTER="Translation"
      ;;
    rotation|rot)
      SYMMETRY_FILTER="Rotation"
      ;;
    scale|scaling)
      SYMMETRY_FILTER="Scale"
      ;;
    -h|--help|help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown Section 4.2 target: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
}

for target in "$@"; do
  run_target "$target"
done

run_selected
