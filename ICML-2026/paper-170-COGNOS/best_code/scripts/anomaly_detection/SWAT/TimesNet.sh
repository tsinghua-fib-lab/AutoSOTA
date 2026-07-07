

export OMP_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=16
export MKL_NUM_THREADS=16
export VECLIB_MAXIMUM_THREADS=16
export NUMEXPR_NUM_THREADS=16

model=TimesNet
data=SWAT
datapath='../../../dataset/SWaT'
seq_len=128
channels=51
anomaly_ratio=0.5
KF_confidence=0.90
epoch=3

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
  --d_model 64 \
  --d_ff 64 \
  --e_layers 2 \
  --enc_in $channels \
  --c_out $channels \
  --top_k 5 \
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
  --d_model 64 \
  --d_ff 64 \
  --e_layers 2 \
  --enc_in $channels \
  --c_out $channels \
  --top_k 5 \
  --anomaly_ratio $anomaly_ratio \
  --batch_size 128 \
  --train_epochs $epoch \


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
  --d_model 64 \
  --d_ff 64 \
  --e_layers 2 \
  --enc_in $channels \
  --c_out $channels \
  --top_k 5 \
  --anomaly_ratio $anomaly_ratio \
  --batch_size 128 \
  --train_epochs $epoch \
  --use_Gaussian_regularization \


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
  --d_model 64 \
  --d_ff 64 \
  --e_layers 2 \
  --enc_in $channels \
  --c_out $channels \
  --top_k 5 \
  --anomaly_ratio $anomaly_ratio \
  --batch_size 128 \
  --train_epochs $epoch \
  --use_KalmanSmoothing \
     --KF_confidence $KF_confidence \
