CUDA_VISIBLE_DEVICES=0 python main_block_ap.py \
--model path/to/Llama-2-7b \
--net Llama-2 \
--wbits 16 \
--output_dir ./output/inference_results/ \
--eval_ppl \
--eval_tasks  piqa,arc_easy,arc_challenge,hellaswag,winogrande
