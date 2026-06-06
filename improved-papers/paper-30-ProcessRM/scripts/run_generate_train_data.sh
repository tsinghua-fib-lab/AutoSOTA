CUDA_VISIBLE_DEVICES=0 python generate_train_data.py \
 --seed 22 \
 --model_dir \
 --data_dir \
 --save_dir \
 --start 0 \
 --end  \
 --generate \

CUDA_VISIBLE_DEVICES=0 python generate_train_data.py \
 --seed 22 \
 --synthetic_data_dir \
 --save_dir \
 --start 0 \
 --end  \
 --get_label \
 --use_binary_pro \