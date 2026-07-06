# For efficiency, we set batch_size=2 for the small MLLM.
# Larger MLLMs use batch_size=1 by default due to GPU memory constraints.
# The same batch-size setting is used for DOUBT and all reproduced baselines.
# To run on other datasets, simply change the value of the --benchmark argument.

python doubt.py --lvlm Qwen2-VL-2B-Instruct --benchmark LLaVABench --llm Qwen2.5-3B-Instruct  --inference_temp 0.1 --sampling_temp 0.5 --sampling_time 10 --batch_size 2;
python doubt.py --lvlm Qwen2-VL-7B-Instruct --benchmark LLaVABench --llm Qwen2.5-3B-Instruct  --inference_temp 0.1 --sampling_temp 0.5 --sampling_time 10 --batch_size 1;
python doubt.py --lvlm Qwen2-VL-72B-Instruct --benchmark LLaVABench --llm Qwen2.5-3B-Instruct  --inference_temp 0.1 --sampling_temp 0.5 --sampling_time 10 --batch_size 1;
python doubt.py --lvlm InternVL2-1B --benchmark LLaVABench --llm Qwen2.5-3B-Instruct  --inference_temp 0.1 --sampling_temp 0.5 --sampling_time 10 --batch_size 1;
python doubt.py --lvlm InternVL2-8B --benchmark LLaVABench --llm Qwen2.5-3B-Instruct  --inference_temp 0.1 --sampling_temp 0.5 --sampling_time 10 --batch_size 1;
python doubt.py --lvlm InternVL2-26B --benchmark LLaVABench --llm Qwen2.5-3B-Instruct  --inference_temp 0.1 --sampling_temp 0.5 --sampling_time 10 --batch_size 1;
python doubt.py --lvlm llava-1.5-7b-hf --benchmark LLaVABench --llm Qwen2.5-3B-Instruct  --inference_temp 0.1 --sampling_temp 0.5 --sampling_time 10 --batch_size 1;
python doubt.py --lvlm llava-1.5-13b-hf --benchmark LLaVABench --llm Qwen2.5-3B-Instruct  --inference_temp 0.1 --sampling_temp 0.5 --sampling_time 10 --batch_size 1;
python doubt.py --lvlm llava-v1.6-mistral-7b-hf --benchmark LLaVABench --llm Qwen2.5-3B-Instruct  --inference_temp 0.1 --sampling_temp 0.5 --sampling_time 10 --batch_size 1;
python doubt.py --lvlm llava-v1.6-vicuna-13b-hf --benchmark LLaVABench --llm Qwen2.5-3B-Instruct  --inference_temp 0.1 --sampling_temp 0.5 --sampling_time 10 --batch_size 1;