export CUDA_VISIBLE_DEVICES=0

model_name=SiGMA
seq_len=96
e_layers=1
d_model=16
d_ff=32
learning_rate=0.01

pred_lens=(96 192 336 720)

for pred_len in ${pred_lens[@]}; do
	python -u run.py \	
		--task_name long_term_forecast \
		--is_training 1 \
		--root_path ./dataset/electricity/ \
		--data_path electricity.csv \
		--model_id ECL_${seq_len}_${pred_len} \
		--model ${model_name} \
		--data custom \
		--features M \
		--seq_len ${seq_len} \
		--label_len 0 \
		--pred_len ${pred_len} \
		--e_layers ${e_layers} \
		--enc_in 321 \
		--c_out 321 \
		--d_model ${d_model} \
		--d_ff ${d_ff} \
		--learning_rate ${learning_rate} \
		--scale_independence 0 \
		--feature_transformation 1 
done

