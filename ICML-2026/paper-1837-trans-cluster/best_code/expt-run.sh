#!/bin/bash

ENC=(
    "onehot"
    "none"
)
NSAMPLES=$1
NDIMS=$2
NCLUSTERS=$3
LRATE=$4
LITEMP=$5
NSTEPS=$6
TRIAL=$7
ENVNAME=$8

DISTS="normal"
SCALE=0.1
DQKV=1
ATTACT="softmax"
ATTITEMP=1.0
BSZ=32
STEPS_EVAL=50
VAL_NBATCHES=10
LRDECAY=0.5
PATIENCE=5
ODIR="runs"
LOSSACT="softmax"

NAME="N${NSAMPLES}-D${NDIMS}-K${NCLUSTERS}-LR${LRATE}-LI${LITEMP}-T${NSTEPS}-${TRIAL}"
echo $NAME

echo "Activating conda environment ${ENVNAME}..."
eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"
conda activate ${ENVNAME}

SEED=$(sed "${TRIAL}q;d" random_seeds)
echo "RUNNING REP ${TRIAL}/10 WITH SEED ${SEED} ..."

for e in ${ENC[@]}; do
  echo "--------------------------------------------------"
  echo " Loss activation ${LOSSACT} with encoding ${e}"
  echo "--------------------------------------------------"
  cmd="python train.py \
    --nlb ${NSAMPLES} --nub ${NSAMPLES} --nclusters ${NCLUSTERS} --ndims ${NDIMS} \
    --train_dists ${DISTS} --scale ${SCALE} --em --es --sdb \
    --scratch ${e}  --dqkv ${DQKV} \
    --attn_act ${ATTACT} --attn_itemp ${ATTITEMP} \
    --loss_act ${LOSSACT} --loss_itemp ${LITEMP} \
    --seed ${SEED} --bsz ${BSZ} --init_lr ${LRATE} \
    --lr_decay ${LRDECAY} --patience ${PATIENCE} \
    --nsteps ${NSTEPS} --nsteps_per_eval ${STEPS_EVAL} \
    --val_nbatches ${VAL_NBATCHES} --odir ${ODIR}"
  echo $cmd
  # $cmd
done

echo "REP ${TRIAL}/10 completed"
echo "Deactivating conda environment"
conda deactivate
