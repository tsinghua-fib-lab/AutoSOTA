#!/bin/bash
# Submit script for benchmark evaluation
# Usage: bash scripts/eval_benchmark.slurm <BENCHMARK>
# Example: bash scripts/eval_benchmark.slurm MathVista_MINI

BENCHMARK=$1

if [ -z "$BENCHMARK" ]; then
    echo "Error: BENCHMARK not provided"
    echo "Usage: bash scripts/eval_benchmark.slurm <BENCHMARK>"
    echo "Available: RealWorldQA, ChartQA_TEST, OCRBench, DocVQA_VAL, BLINK, MathVista_MINI"
    exit 1
fi

LOG_DIR="logs/eval_${BENCHMARK}_$(date +%Y%m%d_%H%M%S)"
mkdir -p $LOG_DIR

echo "Submitting jobs for benchmark: $BENCHMARK"
echo "Log directory: $LOG_DIR"

# Submit all 9 jobs
sbatch --export=BENCHMARK=$BENCHMARK,STRATEGY=no_recompute,CHUNK_K=0,LOG_DIR=$LOG_DIR scripts/eval_single.slurm
sbatch --export=BENCHMARK=$BENCHMARK,STRATEGY=no_recompute,CHUNK_K=2,LOG_DIR=$LOG_DIR scripts/eval_single.slurm
sbatch --export=BENCHMARK=$BENCHMARK,STRATEGY=lego,CHUNK_K=2,LOG_DIR=$LOG_DIR scripts/eval_single.slurm
sbatch --export=BENCHMARK=$BENCHMARK,STRATEGY=cacheblend,CHUNK_K=2,LOG_DIR=$LOG_DIR scripts/eval_single.slurm
sbatch --export=BENCHMARK=$BENCHMARK,STRATEGY=guided_recompute,CHUNK_K=2,LOG_DIR=$LOG_DIR scripts/eval_single.slurm
sbatch --export=BENCHMARK=$BENCHMARK,STRATEGY=no_recompute,CHUNK_K=4,LOG_DIR=$LOG_DIR scripts/eval_single.slurm
sbatch --export=BENCHMARK=$BENCHMARK,STRATEGY=lego,CHUNK_K=4,LOG_DIR=$LOG_DIR scripts/eval_single.slurm
sbatch --export=BENCHMARK=$BENCHMARK,STRATEGY=cacheblend,CHUNK_K=4,LOG_DIR=$LOG_DIR scripts/eval_single.slurm
sbatch --export=BENCHMARK=$BENCHMARK,STRATEGY=guided_recompute,CHUNK_K=4,LOG_DIR=$LOG_DIR scripts/eval_single.slurm

echo "Submitted 9 jobs. Check with: squeue -u $USER"
