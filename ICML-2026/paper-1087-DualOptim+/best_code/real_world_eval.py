import csv
import json
import os
import shutil
import subprocess
import warnings

import hydra
import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

from metrics import get_all_evals, get_dataloader, get_eval_results
from utils import get_model_identifiers_from_yaml, set_random_seed

warnings.filterwarnings("ignore")


def summary_results(eval_dir):
    for dirpath, dirnames, filenames in os.walk(eval_dir):
        for file in filenames:
            if file.endswith("json") and "results" in file:
                results_path = os.path.join(dirpath, file)
    results = json.load(open(results_path, "r"))["results"]
    results_dict = {
        "ARC-C": results["arc_challenge"]["acc_norm,none"],
        "MMLU": results["mmlu"]["acc,none"],
        "TruthfulQA(mc1)": results["truthfulqa_mc1"]["acc,none"],
        "TriviaQA": results["triviaqa"]["exact_match,remove_whitespace"],
        "GSM8k": results["gsm8k"]["exact_match,flexible-extract"],
    }
    with open(os.path.join(eval_dir, "../downstream_task_results.txt"), "w") as txtfile:
        for key, value in results_dict.items():
            txtfile.write(f"{key}: {value}\n")
    save_file = os.path.join(eval_dir, "../downstream_task_results.csv")
    with open(save_file, "a") as f:
        w = csv.DictWriter(f, results_dict.keys())
        w.writeheader()
        w.writerow(results_dict)

    return results_dict


import subprocess
import os


def general_eval(
        cfg,
        model_name,
        task_list=None,
        output_dir=".",
):
    command = "accelerate"
    tasks = ",".join(task_list)

    if cfg.use_LoRA:
        model_args = f"pretrained={cfg.model_path},peft={model_name},add_bos_token=True,trust_remote_code=True"
    else:
        model_args = f"pretrained={model_name},add_bos_token=True,trust_remote_code=True"

    args = [
        "launch",
        "-m",
        "lm_eval",
        "--model",
        "hf",
        "--model_args",
        model_args,
        "--tasks",
        tasks,
        "--batch_size",
        "auto:4",
        "--output_path",
        f"{output_dir}/downstream_tasks",
    ]

    full_command = [command] + args
    print(f"Command: {' '.join(full_command)}")

    # Environment Setup
    env = os.environ.copy()
    env["HF_DATASETS_OFFLINE"] = "1"
    env["HF_HUB_OFFLINE"] = "1"
    env["HF_DATASETS_TRUST_REMOTE_CODE"] = "1"

    try:
        subprocess.run(full_command, check=True, env=env)
    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e}")

    results_dict = summary_results(output_dir)
    return results_dict


def model_eval(cfg, model, tokenizer, save_dir, eval_unlearn_step=None):
    eval_unlearn_step = "last" if eval_unlearn_step == None else eval_unlearn_step
    aggregated_eval_logs = {}
    for (
        i,
        (
            folder,
            split,
            question_key,
            answer_key,
            eval_task,
            base_answer_key,
            perturbed_answer_key,
        ),
    ) in enumerate(
        zip(
            cfg.eval.data_path,
            cfg.eval.split_list,
            cfg.eval.question_key,
            cfg.eval.answer_key,
            cfg.eval.eval_task,
            cfg.eval.base_answer_key,
            cfg.eval.perturbed_answer_key,
        )
    ):
        os.makedirs(save_dir, exist_ok=True)
        save_filename = os.path.join(save_dir, f"{eval_task}.json")

        if os.path.exists(save_filename):
            print(f"Skipping {eval_task} because {save_filename} already exists")
            eval_logs = json.load(open(save_filename, "r"))
        else:
            eval_dataloader, base_eval_dataloader, perturb_dataloader, idk_dataloader = get_dataloader(
                cfg.eval,
                eval_task,
                tokenizer,
                folder,
                split,
                question_key,
                answer_key,
                base_answer_key,
                perturbed_answer_key,
            )

            eval_logs = get_all_evals(
                cfg.eval,
                model,
                tokenizer,
                folder,
                split,
                eval_task,
                eval_dataloader,
                base_eval_dataloader,
                perturb_dataloader,
                idk_dataloader,
                False,
            )

            with open(save_filename, "w") as f:
                json.dump(eval_logs, f, indent=4)

        aggregated_eval_logs[f"{eval_task}.json"] = eval_logs

    aggregated_eval_log_filename = os.path.join(save_dir, "eval_log_aggregated.json")
    with open(aggregated_eval_log_filename, "w") as f:
        # pretty write json to f
        json.dump(aggregated_eval_logs, f, indent=4)

    eval_results = get_eval_results(aggregated_eval_logs)
    aaggregate_stat = {**eval_results}

    print(aaggregate_stat)

    aaggregate_stat["split"] = cfg.split
    aaggregate_stat["forget_loss"] = cfg.forget_loss
    aaggregate_stat["forget_coeff"] = cfg.forget_coeff
    aaggregate_stat["regularization_coeff"] = cfg.regularization_coeff
    aaggregate_stat["learning_rate"] = cfg.lr
    aaggregate_stat["epochs"] = cfg.num_epochs
    aaggregate_stat["fix_ref_model"] = cfg.fix_ref_model
    aaggregate_stat["mask"] = cfg.mask
    aaggregate_stat["unlearn_step"] = eval_unlearn_step

    with open(os.path.join(save_dir, "unlearning_results.txt"), "w") as txtfile:
        for key, value in aaggregate_stat.items():
            txtfile.write(f"{key}: {value}\n")

    save_file = os.path.join(save_dir, "unlearning_results.csv")
    with open(save_file, "a") as f:
        w = csv.DictWriter(f, aaggregate_stat.keys())
        w.writeheader()
        w.writerow(aaggregate_stat)

    all_task_save_file = os.path.join(cfg.save_dir, "all_unlearning_results.csv")
    if (
        not os.path.exists(all_task_save_file)
        or os.path.getsize(all_task_save_file) == 0
    ):
        with open(all_task_save_file, "a") as f:
            w = csv.DictWriter(f, aaggregate_stat.keys())
            w.writeheader()
            w.writerow(aaggregate_stat)
    else:
        with open(all_task_save_file, "a") as f:
            w = csv.DictWriter(f, aaggregate_stat.keys())
            w.writerow(aaggregate_stat)

    return aaggregate_stat


@hydra.main(version_base=None, config_path="config", config_name="forget")
def main(cfg):
    if os.environ.get("LOCAL_RANK") is not None:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        device_map = {"": local_rank}

    seed = cfg.seed
    set_random_seed(seed)

    model_cfg = get_model_identifiers_from_yaml(cfg.model_family)
    model_id = model_cfg["hf_key"]

    curr_save_dir = cfg.save_dir
    curr_checkpoint_dir = os.path.join(
        curr_save_dir, f"checkpoint-{cfg.eval_unlearn_step}"
    )
    if cfg.eval_unlearn_step == 0:
        curr_checkpoint_dir = cfg.model_path
    else:
        if not os.path.exists(curr_checkpoint_dir):
            print(f"{curr_checkpoint_dir} does not exist.")
            exit()

    curr_eval_dir = os.path.join(curr_save_dir, f"eval_results-{cfg.eval_unlearn_step}")

    # if os.path.exists(os.path.join(curr_eval_dir, 'unlearning_results.csv')):
    #     print(f'{curr_eval_dir} already evaluated.')
    #     exit()

    # NEIGHBOR SET EVAL
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    config = AutoConfig.from_pretrained(model_id)
    #
    if cfg.use_LoRA:
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model_path,
            config=config,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        model = PeftModel.from_pretrained(model, curr_checkpoint_dir)
        model = model.merge_and_unload()
    else:
        model = AutoModelForCausalLM.from_pretrained(
            curr_checkpoint_dir,
            config=config,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
    model = model.eval()

    eval_results = model_eval(
        cfg, model, tokenizer, curr_eval_dir, cfg.eval_unlearn_step
    )

    print(
        "After Unlearn Step %s, Untargeted Forget Efficacy %.6f, Targeted Forget Efficacy %.6f, Model Utility %.6f"
        % (
            cfg.eval_unlearn_step,
            eval_results["Untargeted Forget Efficacy"],
            eval_results["Targeted Forget Efficacy"],
            eval_results["Model Utility"],
        )
    )
    del model

    # DOWNSTREAM TASKS EVAL
    task_lists = [
        "arc_challenge",  # ARC-c
        "truthfulqa_mc1",
        "mmlu",
        "triviaqa",
        "gsm8k"
    ]

    geneal_results = general_eval(cfg, curr_checkpoint_dir, task_lists, curr_eval_dir)
    print(geneal_results)
    all_results = {**geneal_results, **eval_results}

    with open(os.path.join(curr_eval_dir, "aggr_results.csv"), "a") as f:
        w = csv.DictWriter(f, all_results.keys())
        w.writeheader()
        w.writerow(all_results)

    with open(os.path.join(curr_eval_dir, "aggregate_stat.txt"), "w") as txtfile:
        for key, value in all_results.items():
            txtfile.write(f"{key}: {value}\n")

    if not cfg.save_checkpoint:
        # last unlearning tasks and do not save checkpoints
        if (os.path.exists(curr_checkpoint_dir)) and (cfg.eval_unlearn_step != 0):
            shutil.rmtree(curr_checkpoint_dir)


if __name__ == "__main__":
    main()
