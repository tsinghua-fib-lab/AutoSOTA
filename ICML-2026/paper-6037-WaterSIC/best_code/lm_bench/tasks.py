import lm_eval
import datasets

from lm_bench.lm import My_LM
from parallel.config import no_q_config
from parallel.start import start


def process_results(results):
    metric_order = ["acc_norm,none", "acc,none"]
    result = {}
    for key, value in results["results"].items():
        for metric in metric_order:
            if metric in value:
                result[key] = value[metric]
                break
        if key not in result:
            print(f"Warning: cannot find the metric for {key}")
    return result


def evaluate_on_tasks(model, tokenizer, device, tasks):
    batch_size = model.params.max_batch_size
    lm_obj = My_LM(model=model, tokenizer=tokenizer, batch_size=batch_size, device=device)
    results = lm_eval.simple_evaluate(
        model=lm_obj,
        tasks=tasks,
        num_fewshot=0)
    return process_results(results)


if __name__ == "__main__":
    datasets.config.HF_DATASETS_TRUST_REMOTE_CODE = True
    tasks = ["arc_easy", "arc_challenge", "hellaswag", "lambada_openai", "winogrande", "piqa", "social_iqa", "openbookqa"]

    ckpt_path = "/home/semyon/quant-bucket/Llama-3-8B"
    override_params = {
        "max_batch_size": 128,
        "max_seq_len": 512
    }
    model, tokenizer = start(ckpt_path, False, no_q_config, override_params=override_params)
    device = "cuda"

    lm_obj = My_LM(model=model, tokenizer=tokenizer, batch_size=128, device=device)
    result = evaluate_on_tasks(model, tokenizer, device, tasks)
    print(result)
