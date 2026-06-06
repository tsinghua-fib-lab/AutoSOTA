export CUDA_VISIBLE_DEVICES=0

seq_len=104
model=CALF


for pred_len in 24 36 48 60
do
for percent in 100
do

python main.py \
    --root_path ./datasets/illness/ \
    --data_path national_illness.csv \
    --model_id illness_$model'_'$gpt_layer'_'$seq_len'_'$pred_len'_'$percent \
    --data custom \
    --seq_len $seq_len \
    --label_len 18 \
    --pred_len $pred_len \
    --batch_size 16 \
    --learning_rate 0.0001 \
    --train_epochs 10 \
    --decay_fac 0.75 \
    --d_model 768 \
    --n_heads 4 \
    --d_ff 768 \
    --freq 0 \
    --patch_size 24 \
    --stride 2 \
    --all 1 \
    --percent $percent \
    --gpt_layer 6 \
    --itr 3 \
    --model $model \
    --is_gpt 1
done
done

