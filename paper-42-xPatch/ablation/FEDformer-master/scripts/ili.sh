if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

seq_len=36
model_name=FEDformer
model_id_name=national_illness

for pred_len in 24 36 48 60
do
  python -u run.py \
    --is_training 1 \
    --root_path ../dataset/ \
    --data_path national_illness.csv \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --label_len 18 \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --des 'Exp' \
    --itr 1 >logs/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log
done