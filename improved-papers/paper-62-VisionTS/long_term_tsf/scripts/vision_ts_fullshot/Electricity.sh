export CUDA_VISIBLE_DEVICES=3
export CKPT_DIR="../ckpt/"
export VM_ARCH="mae_base"
export CONTEXT_LEN=1152
export PERIODICITY=24
export ALIGN_CONST=0.4
export NORM_CONST=0.4

for PRED_LEN in 96 192 336 720; do
    python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --model VisionTS \
    --root_path ./dataset/electricity/ \
    --data_path electricity.csv \
    --save_dir save/ECL_$PRED_LEN \
    --model_id VisionTS_ECL_$PRED_LEN \
    --data custom \
    --features M \
    --train_epochs 1 \
    --vm_arch $VM_ARCH \
    --vm_ckpt $CKPT_DIR \
    --seq_len $CONTEXT_LEN \
    --periodicity $PERIODICITY \
    --pred_len $PRED_LEN \
    --norm_const $NORM_CONST \
    --align_const $ALIGN_CONST
done;