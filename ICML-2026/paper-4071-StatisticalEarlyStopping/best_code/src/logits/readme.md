We adapt the code from [iie-ycx/DEER](https://github.com/iie-ycx/DEER/blob/main/vllm-deer.py) and [chicosirius/think-or-not](https://github.com/chicosirius/think-or-not/blob/main/ThinkorNot/adaptive_think.py) to implement the two different measures of "confidence" described in the respective papers.

To calculate the confidence scores, we use the following commands:

```bash
python -m src.deer --model_name <MODEL_NAME> --data_name <DATA_NAME>
```

Because the beam search part can be time-consuming, we implemented a batch processing version:

```bash
python -m src.deer_batch --model_name <MODEL_NAME> --data_name <DATA_NAME>
```

To calibrate and test with the Maxwise framework, run the following command:

```bash
python -m src.logits.cal_and_test --model_name "<MODEL_NAME>" --data_name "<DATA_NAME>"
```

You may find the following loop scripts useful to run the code on multiple models and datasets:

```bash
models=("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
    "Qwen/QwQ-32B"
    "Qwen/Qwen3-8B"
    "Qwen/Qwen3-14B"
    "Qwen/Qwen3-32B"
    "nvidia/AceReason-Nemotron-1.1-7B"
    "nvidia/AceReason-Nemotron-14B"
    "XiaomiMiMo/MiMo-7B-RL-0530"
    "Skywork/Skywork-OR1-7B"
    "Skywork/Skywork-OR1-32B")
data_names=("mc" "umwp" "mip" "mmlu")

for model in "${models[@]}"; do
    for data_name in "${data_names[@]}"; do
        python -m src.logits.cal_and_test --model_name "$model" --data_name "$data_name"
    done
done
```