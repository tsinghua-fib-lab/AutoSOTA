import csv
import json
import os
import shutil
import warnings

import hydra
import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

from metrics import get_all_evals, get_dataloader, get_eval_results
from utils import get_model_identifiers_from_yaml

warnings.filterwarnings("ignore")


def model_eval(
    cfg,
    task_id,
    unlearn_times,
    model,
    tokenizer,
    save_dir,
    curr_forget_path,
    eval_unlearn_step=None,
):
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
        if eval_task == "eval_log_forget":
            # load forge data from processed task data
            folder = curr_forget_path
            split = "forget_perturbed"

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
                True,
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
    aaggregate_stat["task_id"] = task_id
    aaggregate_stat["unlearn_times"] = unlearn_times

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

    return eval_results


@hydra.main(version_base=None, config_path="config", config_name="forget")
def main(cfg):
    # TODO: evaluate whether targeted unlearning outputs are "I don't know" etc.
    if os.environ.get("LOCAL_RANK") is not None:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        device_map = {"": local_rank}

    model_cfg = get_model_identifiers_from_yaml(cfg.model_family)
    model_id = model_cfg["hf_key"]

    task_list = os.getenv("TASK_LIST").split(",")
    task_list = [int(i) for i in task_list]
    cfg.save_dir = os.path.join(cfg.save_dir, os.getenv("TASK_LIST").replace(",", "-"))

    unlearn_times = task_list.index(cfg.task_id) + 1
    curr_save_dir = os.path.join(cfg.save_dir, f"unlearn_times_{unlearn_times}")
    curr_data_path = os.path.join(curr_save_dir, "task_data")

    curr_checkpoint_dir = os.path.join(
        curr_save_dir, f"checkpoint-{cfg.eval_unlearn_step}"
    )
    if cfg.eval_unlearn_step == 0:
        curr_checkpoint_dir = cfg.model_path
        print(cfg.model_path)

    else:
        if not os.path.exists(curr_checkpoint_dir):
            print(f"{curr_checkpoint_dir} does not exist.")
            exit()

    curr_eval_dir = os.path.join(curr_save_dir, f"eval_results-{cfg.eval_unlearn_step}")
    if os.path.exists(os.path.join(curr_eval_dir, "aggregate_stat.csv")):
        print(f"{curr_eval_dir} already evaluated.")
        exit()

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    config = AutoConfig.from_pretrained(model_id)

    if cfg.use_LoRA:
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model_path,
            config=config,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
            device_map=device_map,
        )
        model = PeftModel.from_pretrained(model, curr_checkpoint_dir)
        model = model.merge_and_unload()
    else:
        model = AutoModelForCausalLM.from_pretrained(
            curr_checkpoint_dir,
            config=config,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
            device_map=device_map,
        )
    model = model.eval()

    eval_results = model_eval(
        cfg,
        cfg.task_id,
        unlearn_times,
        model,
        tokenizer,
        curr_eval_dir,
        curr_data_path,
        cfg.eval_unlearn_step,
    )
    print(
        "After Unlearn Task %d, Unlearn Step %s, Untargeted Forget Efficacy %.6f, Targeted Forget Efficacy %.6f, Model Utility %.6f"
        % (
            cfg.task_id,
            cfg.eval_unlearn_step,
            eval_results["Untargeted Forget Efficacy"],
            eval_results["Targeted Forget Efficacy"],
            eval_results["Model Utility"],
        )
    )

    if unlearn_times == len(task_list) and not cfg.save_checkpoint:
        # last unlearning tasks and do not save checkpoints
        if (os.path.exists(curr_checkpoint_dir)) and (cfg.eval_unlearn_step != 0):
            shutil.rmtree(curr_checkpoint_dir)


if __name__ == "__main__":
    main()
