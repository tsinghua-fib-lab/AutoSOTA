export CUDA_VISIBLE_DEVICES=0

export OMP_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=16
export MKL_NUM_THREADS=16
export VECLIB_MAXIMUM_THREADS=16
export NUMEXPR_NUM_THREADS=16

model=MICN
anomaly_ratio=1
KF_confidence=0.90
epoch=3


for file in ../../../dataset/UCR/*_UCR_*.txt; do

  filename=$(basename "$file")

  model_id=$(echo "$filename" | cut -d'_' -f1)_UCR

  python -u ../../../run.py \
    --task_name anomaly_detection \
    --is_training 1 \
    --root_path ../../../dataset/UCR \
    --data_path "$filename" \
    --model_id "$model_id" \
    --model $model \
    --data UCR \
    --features M \
    --seq_len 128 \
    --pred_len 0 \
    --d_model 128 \
    --d_ff 128 \
    --e_layers 3 \
    --enc_in 1 \
    --c_out 1 \
    --top_k 3 \
    --anomaly_ratio $anomaly_ratio \
    --batch_size 128 \
    --train_epochs $epoch \
    --use_KalmanSmoothing \
      --KF_confidence $KF_confidence \
    --use_Gaussian_regularization \


  python -u ../../../run.py \
    --task_name anomaly_detection \
    --is_training 1 \
    --root_path ../../../dataset/UCR \
    --data_path "$filename" \
    --model_id "$model_id" \
    --model $model \
    --data UCR \
    --features M \
    --seq_len 128 \
    --pred_len 0 \
    --d_model 128 \
    --d_ff 128 \
    --e_layers 3 \
    --enc_in 1 \
    --c_out 1 \
    --top_k 3 \
    --anomaly_ratio $anomaly_ratio \
    --batch_size 128 \
    --train_epochs $epoch \

  # python -u ../../../run.py \
  # --task_name anomaly_detection \
  # --is_training 0 \
  # --root_path ../../../dataset/UCR \
  # --data_path "$filename" \
  # --model_id "$model_id" \
  # --model $model \
  # --data UCR \
  # --features M \
  # --seq_len 128 \
  # --pred_len 0 \
  # --d_model 128 \
  # --d_ff 128 \
  # --e_layers 3 \
  # --enc_in 1 \
  # --c_out 1 \
  # --top_k 3 \
  # --anomaly_ratio $anomaly_ratio \
  # --batch_size 128 \
  # --train_epochs $epoch \
  # --use_Gaussian_regularization \
  #   --alpha_GRLoss $alpha_GRLoss \
  #   --GRfilter_size $GRfilter_size

  # python -u ../../../run.py \
  # --task_name anomaly_detection \
  # --is_training 0 \
  # --root_path ../../../dataset/UCR \
  # --data_path "$filename" \
  # --model_id "$model_id" \
  # --model $model \
  # --data UCR \
  # --features M \
  # --seq_len 128 \
  # --pred_len 0 \
  # --d_model 128 \
  # --d_ff 128 \
  # --e_layers 3 \
  # --enc_in 1 \
  # --c_out 1 \
  # --top_k 3 \
  # --anomaly_ratio $anomaly_ratio \
  # --batch_size 128 \
  # --train_epochs $epoch \
  # --use_KalmanSmoothing \
  #   --anomaly_QR_ratio $anomaly_QR_ratio \
  #   --KF_confidence $KF_confidence \


done

