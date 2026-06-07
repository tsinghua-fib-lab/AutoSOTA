#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${ROOT_DIR}"
OUTPUT_ROOT="${PROJECT_DIR}/output"
RESULT_ROOT="${PROJECT_DIR}/render_results"
PYTHON_BIN="${PYTHON_BIN:-python}"

mkdir -p "${RESULT_ROOT}"

for exp_dir in "${OUTPUT_ROOT}"/*/*/; do
    if [[ ! -d "${exp_dir}" ]]; then
        continue
    fi

    exp_name="$(basename "${exp_dir}")"
    item_name="$(basename "$(dirname "${exp_dir}")")"
    ckpt_path="$(ls "${exp_dir}"/chkpnt*.pth 2>/dev/null | sort -V | tail -n 1 || true)"

    if [[ -z "${ckpt_path}" ]]; then
        echo "Skipping ${exp_name}: no checkpoint found." >&2
        continue
    fi

    iteration="$(basename "${ckpt_path}" | sed -E 's/[^0-9]*([0-9]+)\.pth/\1/')"
    out_dir="${RESULT_ROOT}/${item_name}"
    mkdir -p "${out_dir}"

    "${PYTHON_BIN}" scripts/render_for_eval.py \
        --checkpoint "${ckpt_path}" \
        --final_iterations "${iteration}" \
        --output_dir "${out_dir}" \
        --subset both \
        --eval

    log_src="${exp_dir%/}/visualize_log.csv"
    if [[ -f "${log_src}" ]]; then
        cp "${log_src}" "${out_dir}/"
    else
        echo "Warning: visualize_log.csv not found in ${exp_name}." >&2
    fi

    echo "Rendered ${exp_name} -> ${out_dir}"

done