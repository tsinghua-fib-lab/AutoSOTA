# llama-2-7b-w2g64
CUDA_VISIBLE_DEVICES=0 python -m model_transfer.real_to_fake \
--model path/to/original/quantized/model \
--save_dir path/to/new/model \
--wbits 2 \
--group_size 64