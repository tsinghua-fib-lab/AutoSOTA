model_name=Transformer
attention_type=Prime
use_prime=0

if [ $attention_type == "Prime" ]; then
    use_prime=1
fi

python run.py \
    --is_training 1 \
    --model_id ${attention_type}_${model_name}_Weather\
    --model ${model_name} \
    --data custom \
    --root_path ../dataset/weather/ \
    --data_path weather.csv \
    --features M \
    --batch_size 128 \
    --seq_len 96 \
    --pred_len 96 \
    --d_model 128 \
    --num_heads 8 \
    --dropout 0.4 \
    --num_layers 3 \
    --epochs 10 \
    --patience 3 \
    --learning_rate 0.0005 \
    --use_norm 1 \
    --use_prime ${use_prime} \
    --learnable_diagonal 1 \
    --filter_type 1 \
    --idrop 0.0 \
    --fredf_loss 0 \
    --save_pred 0

python run.py \
    --is_training 1 \
    --model_id ${attention_type}_${model_name}_Weather\
    --model ${model_name} \
    --data custom \
    --root_path ../dataset/weather/ \
    --data_path weather.csv \
    --features M \
    --batch_size 128 \
    --seq_len 96 \
    --pred_len 192 \
    --d_model 128 \
    --num_heads 8 \
    --dropout 0.4 \
    --num_layers 3 \
    --epochs 10 \
    --patience 3 \
    --learning_rate 0.0005 \
    --use_norm 1 \
    --use_prime ${use_prime} \
    --learnable_diagonal 1 \
    --filter_type 1 \
    --idrop 0.0 \
    --fredf_loss 0 \
    --save_pred 0

python run.py \
    --is_training 1 \
    --model_id ${attention_type}_${model_name}_Weather\
    --model ${model_name} \
    --data custom \
    --root_path ../dataset/weather/ \
    --data_path weather.csv \
    --features M \
    --batch_size 128 \
    --seq_len 96 \
    --pred_len 336 \
    --d_model 128 \
    --num_heads 8 \
    --dropout 0.4 \
    --num_layers 3 \
    --epochs 10 \
    --patience 3 \
    --learning_rate 0.0005 \
    --use_norm 1 \
    --use_prime ${use_prime} \
    --learnable_diagonal 1 \
    --filter_type 1 \
    --idrop 0.0 \
    --fredf_loss 0 \
    --save_pred 0

python run.py \
    --is_training 1 \
    --model_id ${attention_type}_${model_name}_Weather\
    --model ${model_name} \
    --data custom \
    --root_path ../dataset/weather/ \
    --data_path weather.csv \
    --features M \
    --batch_size 128 \
    --seq_len 96 \
    --pred_len 720 \
    --d_model 128 \
    --num_heads 8 \
    --dropout 0.4 \
    --num_layers 3 \
    --epochs 10 \
    --patience 3 \
    --learning_rate 0.0005 \
    --use_norm 1 \
    --use_prime ${use_prime} \
    --learnable_diagonal 1 \
    --filter_type 1 \
    --idrop 0.0 \
    --fredf_loss 0 \
    --save_pred 0