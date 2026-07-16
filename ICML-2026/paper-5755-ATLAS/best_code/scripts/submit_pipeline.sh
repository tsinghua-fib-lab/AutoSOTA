#!/bin/bash
# Submit the full ATLAS pipeline to SGE for a single benchmark.
#
# Usage (run from repo root):
#   bash scripts/submit_pipeline.sh <benchmark> [queue]
#
# benchmark: arc | gsm8k | hellaswag | truthfulqa | winogrande
# queue:     long (default) | hpc

set -euo pipefail

BENCHMARK=${1:?Usage: $0 <benchmark> [queue]}
QUEUE=${2:-long}
LOGDIR="logs/${BENCHMARK}"
mkdir -p "$LOGDIR"

# ── Benchmark chunk config (must match 01_fit_irt.r / 02_wle_scoring.r) ──────
case "$BENCHMARK" in
  arc)        CHUNKS="106 211 316 421 526 631 736 840" ;;
  gsm8k)      CHUNKS="110 219 328 437 546 655 764 873 982 1091 1200 1307" ;;  # seq 110 1200 109 + 1307
  hellaswag)  CHUNKS=$(python3 -c "print(' '.join(map(str, list(range(113,5502,112))+[5608])))" ) ;;
  truthfulqa) CHUNKS="105 209 313 418 523 628" ;;
  winogrande) CHUNKS="105 209 313 417 521 626 731 836 941 1046" ;;
  *) echo "Unknown benchmark: $BENCHMARK"; exit 1 ;;
esac

# ── Step 1: fit IRT per chunk (parallel) ──────────────────────────────────────
STEP1_IDS=""
for CHUNK in $CHUNKS; do
  JID=$(qsub -terse \
    -N "irt_${BENCHMARK}_${CHUNK}" \
    -q "$QUEUE" \
    -l h_vmem=8G \
    -l h_rt=04:00:00 \
    -o "${LOGDIR}/step1_${CHUNK}.log" \
    -j y \
    << EOF
#!/bin/bash
cd /store01/nchawla/pli9/ATLAS
module load R
Rscript scripts/01_fit_irt.r --benchmark=${BENCHMARK} --chunk_end=${CHUNK}
EOF
  )
  echo "Step 1 chunk ${CHUNK}: job ${JID}"
  STEP1_IDS="${STEP1_IDS:+${STEP1_IDS},}${JID}"
done

# ── Step 2: mean/sigma linking + WLE (depends on all step-1 jobs) ─────────────
STEP2_ID=$(qsub -terse \
  -N "wle_${BENCHMARK}" \
  -q "$QUEUE" \
  -hold_jid "$STEP1_IDS" \
  -l h_vmem=16G \
  -l h_rt=04:00:00 \
  -o "${LOGDIR}/step2.log" \
  -j y \
  << EOF
#!/bin/bash
cd /store01/nchawla/pli9/ATLAS
module load R
Rscript scripts/02_wle_scoring.r --benchmark=${BENCHMARK}
EOF
)
echo "Step 2 WLE: job ${STEP2_ID}"

# ── Step 3: ATLAS CAT (depends on step 2) ─────────────────────────────────────
STEP3_IDS=""
for SE in 0.1 0.2 0.3; do
  STEP3_ID=$(qsub -terse \
    -N "atlas_${BENCHMARK}_${SE}" \
    -q "$QUEUE" \
    -hold_jid "$STEP2_ID" \
    -pe smp 8 \
    -l h_vmem=4G \
    -l h_rt=06:00:00 \
    -o "${LOGDIR}/step3_se${SE}.log" \
    -j y \
    << EOF
#!/bin/bash
cd /store01/nchawla/pli9/ATLAS
module load R
Rscript scripts/03_atlas_cat.r --benchmark_name=${BENCHMARK} --se_theta_stop=${SE} --n_cores=8
EOF
  )
  echo "Step 3 ATLAS se=${SE}: job ${STEP3_ID}"
  STEP3_IDS="${STEP3_IDS:+${STEP3_IDS},}${STEP3_ID}"
done

# ── Step 4: p-IRT accuracy (depends on all step-3 jobs) ──────────────────────
STEP4_IDS=""
for SE in 0.1 0.2 0.3; do
  STEP4_ID=$(qsub -terse \
    -N "pirt_${BENCHMARK}_${SE}" \
    -q "$QUEUE" \
    -hold_jid "$STEP3_IDS" \
    -l h_vmem=8G \
    -l h_rt=01:00:00 \
    -o "${LOGDIR}/step4_se${SE}.log" \
    -j y \
    << EOF
#!/bin/bash
cd /store01/nchawla/pli9/ATLAS
module load R
Rscript scripts/04_pirt_accuracy.r --benchmark=${BENCHMARK} --se_theta_stop=${SE}
EOF
  )
  echo "Step 4 p-IRT se=${SE}: job ${STEP4_ID}"
  STEP4_IDS="${STEP4_IDS:+${STEP4_IDS},}${STEP4_ID}"
done

# ── Step 5: actual accuracy (depends on step-2 response matrix) ───────────────
STEP5_ID=$(qsub -terse \
  -N "acc_${BENCHMARK}" \
  -q "$QUEUE" \
  -hold_jid "$STEP2_ID" \
  -l h_vmem=8G \
  -l h_rt=01:00:00 \
  -o "${LOGDIR}/step5.log" \
  -j y \
  << EOF
#!/bin/bash
cd /store01/nchawla/pli9/ATLAS
module load R
Rscript scripts/05_compute_actual_acc.r --benchmark=${BENCHMARK}
EOF
)
echo "Step 5 actual acc: job ${STEP5_ID}"

# ── Analysis: compare p-IRT vs actual (depends on step 4+5) ──────────────────
COMPARE_IDS=""
for SE in 0.1 0.2 0.3; do
  CMP_ID=$(qsub -terse \
    -N "cmp_${BENCHMARK}_${SE}" \
    -q "$QUEUE" \
    -hold_jid "${STEP4_IDS},${STEP5_ID}" \
    -l h_vmem=4G \
    -l h_rt=00:30:00 \
    -o "${LOGDIR}/compare_se${SE}.log" \
    -j y \
    << EOF
#!/bin/bash
cd /store01/nchawla/pli9/ATLAS
module load R
Rscript scripts/analysis/compare_pirt_actual.r --benchmark=${BENCHMARK} --se_theta_stop=${SE}
EOF
  )
  echo "Analysis compare SE=${SE}: job ${CMP_ID}"
  COMPARE_IDS="${COMPARE_IDS:+${COMPARE_IDS},}${CMP_ID}"
done

echo ""
echo "Pipeline submitted for benchmark=${BENCHMARK} on queue=${QUEUE}"
echo "Monitor with: qstat -u \$USER"
echo "Logs in: ${LOGDIR}/"
echo ""
echo "NOTE: After ALL benchmarks are done, run manually:"
echo "  python3 scripts/analysis/summarize_pirt_for_all_method.py"
echo "  python3 scripts/analysis/summarize_theta_for_all_method.py"
