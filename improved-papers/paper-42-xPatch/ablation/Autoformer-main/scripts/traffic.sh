if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

seq_len=96
model_name=Autoformer
model_id_name=traffic

for pred_len in 96 192 336 720
do
  python -u run.py \
    --is_training 1 \
    --root_path ../dataset/ \
    --data_path traffic.csv \
    --model_id traffic_$pred_len \
    --model Autoformer \
    --data custom \
    --features M \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --label_len 48 \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 862 \
    --dec_in 862 \
    --c_out 862 \
    --des 'Exp' \
    --itr 1 \
    --train_epochs 3 >logs/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log
done