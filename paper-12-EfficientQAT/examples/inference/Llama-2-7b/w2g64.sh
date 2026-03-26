CUDA_VISIBLE_DEVICES=0 python main_block_ap.py \
--resume_quant path/to/Llama-2-7b-w2g64 \
--net Llama-2 \
--wbits 2 \
--group_size 64 \
--output_dir ./output/inference_results/ \
--eval_ppl \
--eval_tasks  piqa,arc_easy,arc_challenge,hellaswag,winogrande
