# llama-2-7b-w2g64
CUDA_VISIBLE_DEVICES=0 python -m model_transfer.fp32_to_16 \
--model path/to/original/quantized/model \
--save_dir path/to/new/model \
--target_type fp16 \
--wbits 2 \
--group_size 64 \
--eval_ppl