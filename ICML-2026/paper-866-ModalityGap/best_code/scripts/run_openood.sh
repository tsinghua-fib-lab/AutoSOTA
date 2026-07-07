#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"

DATASET="cifar10"
if [[ $# -gt 0 && "$1" != --* ]]; then
    DATASET="$1"
    shift
fi

case "${DATASET}" in
    cifar10)
        CONFIG="${PROJECT_ROOT}/configs/streaming_ood_cifar10.yaml"
        ;;
    cifar100)
        CONFIG="${PROJECT_ROOT}/configs/streaming_ood_cifar100.yaml"
        ;;
    *)
        echo "Usage: bash scripts/run_openood.sh [cifar10|cifar100] --data_root /path/to/data"
        exit 1
        ;;
esac

cd "${PROJECT_ROOT}"

python "${PROJECT_ROOT}/main.py" \
    --config "${CONFIG}" \
    "$@"
