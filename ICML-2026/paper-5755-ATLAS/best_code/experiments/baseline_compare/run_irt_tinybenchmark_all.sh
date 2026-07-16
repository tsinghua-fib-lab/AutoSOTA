#!/bin/bash
# Submit IRT calibration jobs for all tinybenchmark datasets

BENCHMARKS=("truthfulqa" "winogrande" "hellaswag" "gsm8k" "arc")

# Run from ATLAS repo root

echo "Submitting IRT calibration jobs for all benchmarks..."
echo ""

for BENCHMARK in "${BENCHMARKS[@]}"; do
    echo "Submitting job for: $BENCHMARK"
    qsub -v BENCHMARK=$BENCHMARK run_irt_tinybenchmark.job
    echo ""
done

echo "All jobs submitted!"
echo "Check job status with: qstat -u $USER"
echo "Results will be saved to: baseline_compare/tinybenchmark_calibration/"
