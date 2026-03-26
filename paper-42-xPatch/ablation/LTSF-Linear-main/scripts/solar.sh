if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

seq_len=96
model_name=DLinear
model_id_name=solar

for pred_len in 96 192 336 720
do
  python -u run.py \
    --is_training 1 \
    --root_path ../dataset/ \
    --data_path solar.txt \
    --model_id solar_96_$pred_len \
    --model $model_name \
    --data Solar \
    --features M \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --enc_in 137 \
    --des 'Exp' \
    --batch_size 16 \
    --itr 1 >logs/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log
done