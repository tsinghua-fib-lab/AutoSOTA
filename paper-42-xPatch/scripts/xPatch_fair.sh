ma_type=ema
alpha=0.3
beta=0.3

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/"$ma_type ]; then
    mkdir ./logs/$ma_type
fi

model_name=xPatch
seq_len=512

for model_name in xPatch
do
for pred_len in 96 192 336 720
do
  python -u run.py \
    --is_training 1 \
    --root_path ./dataset/ \
    --data_path ETTh1.csv \
    --model_id ETTh1_$pred_len'_'$ma_type \
    --model $model_name \
    --data ETTh1 \
    --features M \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --enc_in 7 \
    --des 'Exp' \
    --itr 1 \
    --batch_size 128 \
    --learning_rate 0.0005 \
    --lradj 'type3'\
    --train_epochs 20 \
    --patience 5 \
    --ma_type $ma_type \
    --alpha $alpha \
    --beta $beta > logs/$ma_type/$model_name'_ETTh1_'$seq_len'_'$pred_len.log

  python -u run.py \
    --is_training 1 \
    --root_path ./dataset/ \
    --data_path ETTh2.csv \
    --model_id ETTh2_$pred_len'_'$ma_type \
    --model $model_name \
    --data ETTh2 \
    --features M \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --enc_in 7 \
    --des 'Exp' \
    --itr 1 \
    --batch_size 32 \
    --learning_rate 0.0001 \
    --lradj 'type3'\
    --train_epochs 20 \
    --patience 5 \
    --ma_type $ma_type \
    --alpha $alpha \
    --beta $beta > logs/$ma_type/$model_name'_ETTh2_'$seq_len'_'$pred_len.log

  python -u run.py \
    --is_training 1 \
    --root_path ./dataset/ \
    --data_path ETTm1.csv \
    --model_id ETTm1_$pred_len'_'$ma_type \
    --model $model_name \
    --data ETTm1 \
    --features M \
    --seq_len 336 \
    --pred_len $pred_len \
    --enc_in 7 \
    --des 'Exp' \
    --itr 1 \
    --batch_size 256 \
    --learning_rate 0.0001 \
    --lradj 'type3'\
    --train_epochs 20 \
    --patience 5 \
    --ma_type $ma_type \
    --alpha $alpha \
    --beta $beta > logs/$ma_type/$model_name'_ETTm1_336_'$pred_len.log

  python -u run.py \
    --is_training 1 \
    --root_path ./dataset/ \
    --data_path ETTm2.csv \
    --model_id ETTm2_$pred_len'_'$ma_type \
    --model $model_name \
    --data ETTm2 \
    --features M \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --enc_in 7 \
    --des 'Exp' \
    --itr 1 \
    --batch_size 256 \
    --learning_rate 0.0005 \
    --lradj 'type3'\
    --train_epochs 20 \
    --patience 5 \
    --ma_type $ma_type \
    --alpha $alpha \
    --beta $beta > logs/$ma_type/$model_name'_ETTm2_'$seq_len'_'$pred_len.log

  python -u run.py \
    --is_training 1 \
    --root_path ./dataset/ \
    --data_path weather.csv \
    --model_id weather_$pred_len'_'$ma_type \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len 720 \
    --pred_len $pred_len \
    --enc_in 21 \
    --des 'Exp' \
    --itr 1 \
    --batch_size 256 \
    --learning_rate 0.0001 \
    --lradj 'sigmoid'\
    --ma_type $ma_type \
    --alpha $alpha \
    --beta $beta > logs/$ma_type/$model_name'_weather_720_'$pred_len.log

  python -u run.py \
    --is_training 1 \
    --root_path ./dataset/ \
    --data_path traffic.csv \
    --model_id traffic_$pred_len'_'$ma_type \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len 720 \
    --pred_len $pred_len \
    --enc_in 862 \
    --des 'Exp' \
    --itr 1 \
    --batch_size 32 \
    --learning_rate 0.001 \
    --lradj 'sigmoid'\
    --ma_type $ma_type \
    --alpha $alpha \
    --beta $beta > logs/$ma_type/$model_name'_traffic_720_'$pred_len.log 

  python -u run.py \
    --is_training 1 \
    --root_path ./dataset/ \
    --data_path electricity.csv \
    --model_id electricity_$pred_len'_'$ma_type \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len 720 \
    --pred_len $pred_len \
    --enc_in 321 \
    --des 'Exp' \
    --itr 1 \
    --batch_size 64 \
    --learning_rate 0.0005 \
    --lradj 'sigmoid'\
    --ma_type $ma_type \
    --alpha $alpha \
    --beta $beta > logs/$ma_type/$model_name'_electricity_720_'$pred_len.log

  python -u run.py \
    --is_training 1 \
    --root_path ./dataset/ \
    --data_path solar.txt \
    --model_id solar_$pred_len'_'$ma_type \
    --model $model_name \
    --data Solar \
    --features M \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --enc_in 137 \
    --des 'Exp' \
    --itr 1 \
    --batch_size 128 \
    --learning_rate 0.0001 \
    --lradj 'sigmoid'\
    --ma_type $ma_type \
    --alpha $alpha \
    --beta $beta > logs/$ma_type/$model_name'_solar_'$seq_len'_'$pred_len.log
done
done

seq_len=96
ma_type=reg

for model_name in xPatch
do
for pred_len in 96 192 336 720
do
  python -u run.py \
    --is_training 1 \
    --root_path ./dataset/ \
    --data_path exchange_rate.csv \
    --model_id exchange_$pred_len'_'$ma_type \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --enc_in 8 \
    --des 'Exp' \
    --itr 1 \
    --batch_size 32 \
    --learning_rate 0.0001 \
    --lradj 'type1'\
    --train_epochs 10 \
    --patience 3 \
    --ma_type $ma_type \
    --alpha $alpha \
    --beta $beta > logs/$ma_type/$model_name'_exchange_'$seq_len'_'$pred_len.log
done
done

seq_len=104
ma_type=reg

for model_name in xPatch
do 
for pred_len in 24 36 48 60
do
  python -u run.py \
    --is_training 1 \
    --root_path ./dataset/ \
    --data_path national_illness.csv \
    --model_id ili_$pred_len'_'$ma_type \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --enc_in 7 \
    --des 'Exp' \
    --itr 1 \
    --batch_size 32 \
    --learning_rate 0.01 \
    --lradj 'type1'\
    --patch_len 8 \
    --stride 4 \
    --train_epochs 10 \
    --patience 3 \
    --ma_type $ma_type \
    --alpha $alpha \
    --beta $beta > logs/$ma_type/$model_name'_ili_'$seq_len'_'$pred_len.log
done
done