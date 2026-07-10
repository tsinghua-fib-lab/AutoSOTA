model_name=Transformer
attention_type=Prime
use_prime=0

if [ $attention_type == "Prime" ]; then
    use_prime=1
else
    use_prime=0
fi

python run.py \
    --is_training 1 \
    --model_id ${attention_type}_${model_name}_ETTh2 \
    --model ${model_name} \
    --data ETTh2 \
    --root_path ../dataset/ETT-small/ \
    --data_path ETTh2.csv \
    --features M \
    --batch_size 128 \
    --seq_len 96 \
    --pred_len 96 \
    --d_model 128 \
    --num_heads 8 \
    --dropout 0.1 \
    --num_layers 2 \
    --epochs 10 \
    --patience 3 \
    --learning_rate 0.0001 \
    --use_norm 1 \
    --use_prime ${use_prime} \
    --learnable_diagonal 1 \
    --filter_type 2 \
    --idrop 0.4 \
    --fredf_loss 0

python run.py \
    --is_training 1 \
    --model_id ${attention_type}_${model_name}_ETTh2 \
    --model ${model_name} \
    --data ETTh2 \
    --root_path ../dataset/ETT-small/ \
    --data_path ETTh2.csv \
    --features M \
    --batch_size 128 \
    --seq_len 96 \
    --pred_len 192 \
    --d_model 128 \
    --num_heads 8 \
    --dropout 0.1 \
    --num_layers 2 \
    --epochs 10 \
    --patience 3 \
    --learning_rate 0.0001 \
    --use_norm 1 \
    --use_prime ${use_prime} \
    --learnable_diagonal 1 \
    --filter_type 1 \
    --idrop 0.8 \
    --fredf_loss 0

python run.py \
    --is_training 1 \
    --model_id ${attention_type}_${model_name}_ETTh2 \
    --model ${model_name} \
    --data ETTh2 \
    --root_path ../dataset/ETT-small/ \
    --data_path ETTh2.csv \
    --features M \
    --batch_size 128 \
    --seq_len 96 \
    --pred_len 336 \
    --d_model 128 \
    --num_heads 8 \
    --dropout 0.1 \
    --num_layers 2 \
    --epochs 10 \
    --patience 3 \
    --learning_rate 0.0001 \
    --use_norm 1 \
    --use_prime ${use_prime} \
    --learnable_diagonal 1 \
    --filter_type 1 \
    --idrop 0.8 \
    --fredf_loss 0

python run.py \
    --is_training 1 \
    --model_id ${attention_type}_${model_name}_ETTh2 \
    --model ${model_name} \
    --data ETTh2 \
    --root_path ../dataset/ETT-small/ \
    --data_path ETTh2.csv \
    --features M \
    --batch_size 128 \
    --seq_len 96 \
    --pred_len 720 \
    --d_model 128 \
    --num_heads 8 \
    --dropout 0.1 \
    --num_layers 2 \
    --epochs 10 \
    --patience 3 \
    --learning_rate 0.0001 \
    --use_norm 1 \
    --use_prime ${use_prime} \
    --learnable_diagonal 1 \
    --filter_type 1 \
    --idrop 0.8 \
    --fredf_loss 0