#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

make all
mkdir -p logs

base_compute=2.56e16
base_data=2.4e7

while read -r N r dpr_ckpt spare_ckpt; do
  weibull_shape=0.78
  case "$N" in
    200) allreduce_time=2; weibull_scale=51979 ;;
    600) allreduce_time=6; weibull_scale=155937 ;;
    1000) allreduce_time=10; weibull_scale=259894 ;;
    *) echo "Unknown N=$N for allreduce-time" >&2; exit 1 ;;
  esac

  dpr_compute=$(awk -v base="$base_compute" -v r="$r" 'BEGIN { printf "%.10g", base*r }')
  dpr_data=$(awk -v base="$base_data" -v r="$r" 'BEGIN { printf "%.10g", base*r }')

  dpr_log="logs/run_3_dpr_weibull_N${N}_r${r}.log"
  spare_log="logs/run_3_spare_weibull_N${N}_r${r}.log"
  exp_rate=$(awk -v n="$N" 'BEGIN { printf "%.12g", 1/(300*n) }')
  dpr_exp_log="logs/run_3_dpr_exponential_N${N}_r${r}.log"
  spare_exp_log="logs/run_3_spare_exponential_N${N}_r${r}.log"

  /usr/bin/time ./bin/dpr --steps=10000 \
    --compute="${dpr_compute}" \
    --allreduce-time="${allreduce_time}" \
    --allreduce-fail-scale=0.5 \
    --model=2e13 \
    --data="${dpr_data}" \
    --ckpt="${dpr_ckpt}" \
    --workers="${N}" \
    --fail-dist=weibull \
    --weibull-shape="${weibull_shape}" \
    --weibull-scale="${weibull_scale}" \
    --recover=3600 \
    --partial-recover-time=0.1 \
    --compute-jitter=0.05 \
    --seed=0 \
    --replicate_level="${r}" \
    2>&1 | tee "${dpr_log}"

  /usr/bin/time ./bin/spare --steps=10000 \
    --compute="${base_compute}" \
    --allreduce-time="${allreduce_time}" \
    --allreduce-fail-scale=0.5 \
    --model=2e13 \
    --data="${base_data}" \
    --ckpt="${spare_ckpt}" \
    --workers="${N}" \
    --fail-dist=weibull \
    --weibull-shape="${weibull_shape}" \
    --weibull-scale="${weibull_scale}" \
    --recover=3600 \
    --partial-recover-time=0.1 \
    --compute-jitter=0.05 \
    --seed=0 \
    --replicate_level="${r}" \
    2>&1 | tee "${spare_log}"

  /usr/bin/time ./bin/dpr --steps=10000 \
    --compute="${dpr_compute}" \
    --allreduce-time="${allreduce_time}" \
    --allreduce-fail-scale=0.5 \
    --model=2e13 \
    --data="${dpr_data}" \
    --ckpt="${dpr_ckpt}" \
    --workers="${N}" \
    --fail-dist=exponential \
    --exp-rate="${exp_rate}" \
    --recover=3600 \
    --partial-recover-time=0.1 \
    --compute-jitter=0.05 \
    --seed=0 \
    --replicate_level="${r}" \
    2>&1 | tee "${dpr_exp_log}"

  /usr/bin/time ./bin/spare --steps=10000 \
    --compute="${base_compute}" \
    --allreduce-time="${allreduce_time}" \
    --allreduce-fail-scale=0.5 \
    --model=2e13 \
    --data="${base_data}" \
    --ckpt="${spare_ckpt}" \
    --workers="${N}" \
    --fail-dist=exponential \
    --exp-rate="${exp_rate}" \
    --recover=3600 \
    --partial-recover-time=0.1 \
    --compute-jitter=0.05 \
    --seed=0 \
    --replicate_level="${r}" \
    2>&1 | tee "${spare_exp_log}"
done << 'EOF'
200 2 8 8
200 3 6 10
200 4 6 12
200 5 5 13
200 6 5 14
200 7 4 15
200 8 4 15
200 9 3 15
200 10 3 15
200 11 3 15
200 12 3 15
600 2 9 9
600 3 8 13
600 4 8 16
600 5 7 18
600 6 7 20
600 7 6 22
600 8 6 23
600 9 5 24
600 10 5 25
600 11 5 25
600 12 4 25
600 13 4 25
600 14 4 25
600 15 4 25
600 16 4 25
600 17 3 25
600 18 3 25
600 19 3 25
600 20 3 25
1000 2 9 10
1000 3 10 15
1000 4 9 19
1000 5 9 22
1000 6 8 25
1000 7 8 27
1000 8 7 29
1000 9 7 30
1000 10 6 31
1000 11 6 32
1000 12 5 32
1000 13 5 32
1000 14 5 32
1000 15 5 32
1000 16 4 32
1000 17 4 32
1000 18 4 32
1000 19 4 32
1000 20 4 32
1000 21 4 32
1000 22 3 32
1000 23 3 32
1000 24 3 32
1000 25 3 32
1000 26 3 32
EOF
