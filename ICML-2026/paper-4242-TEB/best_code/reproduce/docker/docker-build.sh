#!/bin/bash
# Build the eb-vllm reproduction Docker image
#
# Usage (from repo root):
#   bash reproduce/docker/docker-build.sh          # full build
#   bash reproduce/docker/docker-build.sh --exp    # rebuild experiment layer only (fast)
#
# The build has two stages:
#   1. vllm-openai: full vLLM build from source (slow, ~30min first time)
#   2. eb-vllm:icml2026: add reproduce/ scripts and deps (fast, ~1min)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

BASE_IMAGE="vllm-openai"
EXP_IMAGE="eb-vllm:icml2026"
EXP_ONLY=false

for arg in "$@"; do
    case "$arg" in
        --exp|--exp-only) EXP_ONLY=true ;;
        --base-image=*) BASE_IMAGE="${arg#*=}" ;;
        *) echo "Unknown arg: $arg"; exit 1 ;;
    esac
done

# Step 1: Build vLLM base image
if [ "$EXP_ONLY" = false ]; then
    echo "========================================="
    echo "Step 1/2: Building vLLM base image ..."
    echo "========================================="
    docker build \
        -f docker/Dockerfile \
        --target vllm-openai \
        -t "$BASE_IMAGE" \
        .
else
    # Verify base image exists
    if ! docker image inspect "$BASE_IMAGE" >/dev/null 2>&1; then
        echo "Error: base image '$BASE_IMAGE' not found. Run without --exp first."
        exit 1
    fi
    echo "Skipping base image build (using existing '$BASE_IMAGE')"
fi

# Step 2: Build experiment image
echo ""
echo "========================================="
echo "Step 2/2: Building experiment image ..."
echo "========================================="
docker build \
    -f reproduce/docker/Dockerfile \
    --build-arg BASE_IMAGE="$BASE_IMAGE" \
    -t "$EXP_IMAGE" \
    .

echo ""
echo "========================================="
echo "Build complete: $EXP_IMAGE"
echo "========================================="
echo ""
echo "Run example:"
echo "  docker run --gpus '\"device=4\"' --rm -it \\"
echo "    -v ~/.cache/huggingface:/root/.cache/huggingface \\"
echo "    $EXP_IMAGE bash"
echo ""
echo "Then inside container:"
echo '  SCHEDULERS="v1,eb,ebplus" \'
echo '  CONCURRENCY_PHASES="32:4000" \'
echo '  bash reproduce/eb_plus/non_stationary/run_concurrency_shift.sh 0'
