#!/bin/bash -l

#$ -P ovla            # Specify the SCC project name you want to use
#$ -l h_rt=18:00:00   # Specify the hard time limit for the job
#$ -N prop-converted-ft           # Give job a name
#$ -j y               # Merge the error and output streams into a single file
#$ -l gpus=1
#$ -l gpu_type=L40S
#$ -pe omp 4  # cores

source /projectnb/ovla/ilker/miniconda3/bin/activate base
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# Echo commands as they are executed
set -x

# Target directory
MODEL_PATH=converted-models/prop-ft5
# Source directory (a converted model)
RESUME_MODEL_PATH=converted-models/prop

cp "$0" $MODEL_PATH.sh

python -m autoregltl.main --model-path=$MODEL_PATH --seed=42 train-ted --epochs=5 --val-max-samples=1000 --d-embed-enc=132 --num-heads=6 --d-ff=512 --num-layers=6 --tree-pos-enc --dec-pe=rope --embed-scaling=sqrtd --feature-normalization=l2 --loss-fct=adacos --cross-attn=per --no-enc-agg --no-dec-agg --batch-size=256 --grad-acc-steps=4 --eval-batch-size=256 --resume-from=$RESUME_MODEL_PATH --eval-steps=100

BEAM_SIZE=3
SAMPLES=10000

python3 -m autoregltl.main --model-path=$MODEL_PATH eval-ted --beam-size=$BEAM_SIZE --max-samples=$SAMPLES
python3 -m autoregltl.main --model-path=$MODEL_PATH eval-ted --ds-name=ltl-35-10ap --beam-size=$BEAM_SIZE --max-samples=$SAMPLES
python -m autoregltl.eval2da $MODEL_PATH

echo "JOB DONE!"
