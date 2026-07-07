

export OMP_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=16
export MKL_NUM_THREADS=16
export VECLIB_MAXIMUM_THREADS=16
export NUMEXPR_NUM_THREADS=16

model=TimeMixer
data=SMAP
datapath='../../../dataset/SMAP'
seq_len=192
channels=1
anomaly_ratio=1
KF_confidence=0.90
epoch=3
down_sampling_layers=1
down_sampling_window=2

python -u ../../../run.py \
  --task_name anomaly_detection \
  --is_training 1 \
  --root_path $datapath \
  --model_id $data \
  --model $model \
  --data $data \
  --features M \
  --seq_len $seq_len \
  --pred_len 0 \
  --d_model 128 \
  --d_ff 128 \
  --e_layers 4 \
  --d_layers 1 \
  --factor 3 \
  --down_sampling_layers $down_sampling_layers \
  --down_sampling_method avg \
  --down_sampling_window $down_sampling_window \
  --enc_in $channels \
  --dec_in $channels \
  --c_out $channels \
  --anomaly_ratio $anomaly_ratio \
  --batch_size 128 \
  --train_epochs $epoch \
  --use_KalmanSmoothing \
     --KF_confidence $KF_confidence \
  --use_Gaussian_regularization \



python -u ../../../run.py \
  --task_name anomaly_detection \
  --is_training 1 \
  --root_path $datapath \
  --model_id $data \
  --model $model \
  --data $data \
  --features M \
  --seq_len $seq_len \
  --pred_len 0 \
  --d_model 128 \
  --d_ff 128 \
  --e_layers 4 \
  --d_layers 1 \
  --factor 3 \
  --down_sampling_layers $down_sampling_layers \
  --down_sampling_method avg \
  --down_sampling_window $down_sampling_window \
  --enc_in $channels \
  --dec_in $channels \
  --c_out $channels \
  --anomaly_ratio $anomaly_ratio \
  --batch_size 128 \
  --train_epochs $epoch

python -u ../../../run.py \
  --task_name anomaly_detection \
  --is_training 0 \
  --root_path $datapath \
  --model_id $data \
  --model $model \
  --data $data \
  --features M \
  --seq_len $seq_len \
  --pred_len 0 \
  --d_model 128 \
  --d_ff 128 \
  --e_layers 4 \
  --d_layers 1 \
  --factor 3 \
  --down_sampling_layers $down_sampling_layers \
  --down_sampling_method avg \
  --down_sampling_window $down_sampling_window \
  --enc_in $channels \
  --dec_in $channels \
  --c_out $channels \
  --anomaly_ratio $anomaly_ratio \
  --batch_size 128 \
  --use_Gaussian_regularization \


python -u ../../../run.py \
  --task_name anomaly_detection \
  --is_training 1 \
  --root_path $datapath \
  --model_id $data \
  --model $model \
  --data $data \
  --features M \
  --seq_len $seq_len \
  --pred_len 0 \
  --d_model 128 \
  --d_ff 128 \
  --e_layers 4 \
  --d_layers 1 \
  --factor 3 \
  --down_sampling_layers $down_sampling_layers \
  --down_sampling_method avg \
  --down_sampling_window $down_sampling_window \
  --enc_in $channels \
  --dec_in $channels \
  --c_out $channels \
  --anomaly_ratio $anomaly_ratio \
  --batch_size 128 \
  --train_epochs $epoch \
  --use_KalmanSmoothing \
    --anomaly_QR_ratio $anomaly_QR_ratio \
    --KF_confidence $KF_confidence 