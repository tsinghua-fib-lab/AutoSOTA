# Probing Based Baseline

We train a logistic regression classifier with ridge penalty on this dataset to predict whether the question is ill-posed based on the token-level activations. Our code is implemented based on the description in the following paper: [Answering the Unanswerable Is to Err Knowingly: Analyzing and Mitigating Abstention Failures in Large Reasoning Models](https://arxiv.org/abs/2508.18760). At the time of this implementation, the authors have not released the code, so we implement it ourselves.

To train the linear probing model, run the following command:

```bash
python -m src.probe.train --model_name \"{model}\"
```
This will produce the `probe/{model}/probe_model_layer{layer_number}.pkl`, which stores the trained linear probing model.

To calibrate and test the model, run the following command:

```bash
python -m src.probe.cal_and_test --model_name \"{model}\" --data_name \"{data_name}\"
```
This will produce the `probe{model}/{data_name}_probe_stopping_results.jsonl` files, which the test results, respectively.


## Partial Least Squares (PLS)

In the paper, we also described using Partial Least Squares (PLS) to investigate whether there could be a lower dimensional representation of the token-level activations that could be a "cleaner" representation of uncertainty. To use PLS, run the following command to train the PLS model:

```bash
python -m src.probe.pls --model_name \"{model}\"
```
This will produce the `probe/{model}/pls_model_layer{layer_number}.pkl`, which stores the trained PLS model.

To calibrate and test the model, run the following command:

```bash
python -m src.probe.cal_and_test --model_name \"{model}\" --data_name \"{data_name}\" --classifier_type pls
```

# Running the Code

You may simply run the following commands to train and test the probing models:

```bash
models=("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B" "Qwen/Qwen3-8B" "Qwen/Qwen3-14B")
data_names=("mc" "umwp" "mip" "mmlu")

# Train models
for model in "${models[@]}"; do
    python -m src.probe.train --model_name "$model"
done

# Calibrate and test models
for model in "${models[@]}"; do
    for data_name in "${data_names[@]}"; do
        python -m src.probe.cal_and_test --model_name "$model" --data_name "$data_name"
    done
done
```


For PLS, run the following commands:

```bash
models=("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B" "Qwen/Qwen3-8B" "Qwen/Qwen3-14B")
data_names=("mc" "umwp" "mip" "mmlu")

# Train models
for model in "${models[@]}"; do
    python -m src.probe.pls --model_name "$model"
done

# Calibrate and test models
for model in "${models[@]}"; do
    for data_name in "${data_names[@]}"; do
        python -m src.probe.cal_and_test --model_name "$model" --data_name "$data_name" --classifier_type pls
    done
done
```