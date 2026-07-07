import os
import shutil
import warnings
from pathlib import Path

import datasets
import hydra
import torch
import numpy as np
import transformers
from omegaconf import OmegaConf
from torch.utils.data import ConcatDataset
from peft import LoraConfig, get_peft_model, AutoPeftModelForCausalLM, PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

from dataset import TextForgetDatasetQA, TextForgetDatasetQASingle, dataset_to_json, custom_data_collator_forget, \
    custom_data_collator_forget_single
from trainer import CustomTrainerForgetting, CustomTrainerForgettingAlternate
from utils import get_model_identifiers_from_yaml, set_random_seed

warnings.filterwarnings("ignore")


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    if "lm_head" in lora_module_names:  # needed for 16-bit
        lora_module_names.remove("lm_head")
    return list(lora_module_names)


def get_task_data(data_path, split, task_id, unlearned_tasks, curr_save_dir):
    local_rank = int(os.environ["LOCAL_RANK"])
    forget_data = datasets.load_dataset(
        "json", data_files=os.path.join(data_path, split + ".json"), split="train"
    )
    forget_pertrubed_data = datasets.load_dataset(
        "json",
        data_files=os.path.join(data_path, split + "_perturbed.json"),
        split="train",
    )
    # 100
    # include 10 continual unlearning tasks
    retain_split = "retain" + str(
        100 - min(10 * int(split.replace("forget", "")), 90)
    ).zfill(2)
    retain_data = datasets.load_dataset(
        "json",
        data_files=os.path.join(data_path, retain_split + ".json"),
        split="train",
    )

    forget_retain_data = forget_data.filter(
        lambda x: int(x["task_id"]) not in unlearned_tasks
    )
    curr_forget_data = forget_data.filter(lambda x: int(x["task_id"]) == task_id)

    curr_retain_data = datasets.concatenate_datasets([retain_data, forget_retain_data])

    curr_forget_perturbed_data = forget_pertrubed_data.filter(
        lambda x: int(x["task_id"]) == task_id
    )

    if local_rank == 0:
        curr_data_path = os.path.join(curr_save_dir, "task_data")
        os.makedirs(curr_data_path, exist_ok=True)
        dataset_to_json(curr_forget_data, os.path.join(curr_data_path, "forget.json"))
        dataset_to_json(
            curr_forget_perturbed_data,
            os.path.join(curr_data_path, "forget_perturbed.json"),
        )
        dataset_to_json(curr_retain_data, os.path.join(curr_data_path, "retain.json"))

    return curr_forget_data, curr_retain_data


@hydra.main(version_base=None, config_path="config", config_name="tofu")
def main(cfg):
    num_devices = int(os.environ.get("WORLD_SIZE", 1))

    if os.environ.get("LOCAL_RANK") is not None:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        device_map = {"": local_rank}

    seed = cfg.seed
    set_random_seed(seed)

    model_cfg = get_model_identifiers_from_yaml(cfg.model_family)
    model_id = model_cfg["hf_key"]

    config = AutoConfig.from_pretrained(model_id)

    # get the sequence of continual unlearning tasks
    task_list = os.getenv("TASK_LIST").split(",")
    task_list = [int(i) for i in task_list]
    # the order of unlearning tasks
    cfg.save_dir = os.path.join(cfg.save_dir, os.getenv("TASK_LIST").replace(",", "-"))
    # number of times to unlearn
    unlearn_times = task_list.index(cfg.task_id) + 1
    curr_save_dir = os.path.join(cfg.save_dir, f"unlearn_times_{unlearn_times}")

    if os.path.exists(
        os.path.join(curr_save_dir, "eval_results-last", "aggregate_stat.txt")
    ):
        print(f"Task {cfg.task_id} already unlearned.")
        exit()

    if local_rank == 0:
        Path(cfg.save_dir).mkdir(parents=True, exist_ok=True)
        with open(f"{cfg.save_dir}/config.yaml", "w") as file:
            OmegaConf.save(cfg, file)

    # get the unlearned model of the last unlearning task
    last_checkpoint_dir = os.path.join(
        cfg.save_dir, f"unlearn_times_{unlearn_times - 1}", "checkpoint-last"
    )
    if (unlearn_times > 1) and (not os.path.exists(last_checkpoint_dir)):
        print("last checkpoint does not exist.")
        exit()

    # process current forget set and retain set for unlearning
    curr_forget_data, curr_retain_data = get_task_data(
        cfg.data_path, cfg.split, cfg.task_id, task_list[:unlearn_times], curr_save_dir
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    if cfg.alternate:
        forget_set = TextForgetDatasetQASingle(
            tokenizer=tokenizer,
            model_family=cfg.model_family,
            data=curr_forget_data,
            set_name="forget",
            max_length=500,
            mask=cfg.mask,
        )
        retain_set = TextForgetDatasetQASingle(
            tokenizer=tokenizer,
            model_family=cfg.model_family,
            data=curr_retain_data,
            set_name="retain",
            max_length=500,
            mask=cfg.mask,
        )
        torch_format_dataset = ConcatDataset([forget_set, retain_set])
    else:
        torch_format_dataset = TextForgetDatasetQA(
            tokenizer=tokenizer,
            model_family=cfg.model_family,
            forget_data=curr_forget_data,
            retain_data=curr_retain_data,
            max_length=500,
            mask=cfg.mask,
        )

    batch_size = cfg.batch_size
    gradient_accumulation_steps = cfg.gradient_accumulation_steps

    if cfg.alternate:
        if np.ceil(len(forget_set) / cfg.forget_freq) <= np.ceil(len(retain_set) / cfg.retain_freq):
            steps_per_epoch = int((1 + cfg.retain_freq/cfg.forget_freq) * np.ceil(
                len(forget_set) / (batch_size * gradient_accumulation_steps * num_devices)
            ))
        else:
            steps_per_epoch = int((1 + cfg.forget_freq/cfg.retain_freq) * np.ceil(
                len(retain_set) / (batch_size * gradient_accumulation_steps * num_devices)
            ))
    else:
        steps_per_epoch = int(np.ceil(
            len(torch_format_dataset) / (batch_size * gradient_accumulation_steps * num_devices)
        ))

    if cfg.max_steps is not None:
        max_steps = cfg.max_steps
    else:
        max_steps = cfg.num_epochs * steps_per_epoch

    warmup_steps = steps_per_epoch if steps_per_epoch > 1 else 0

    if len(task_list) > 1:
        # only evaluate the last checkpoint of each task for continual unlearning by default
        save_steps = max_steps
    else:
        if cfg.save_steps == "steps_per_epoch":
            save_steps = steps_per_epoch
        elif cfg.save_steps == "last":
            save_steps = max_steps
        else:
            save_steps = cfg.save_steps

    if local_rank == 0:
        print("\n######### Unlearn Task %d #########" % (unlearn_times))
        print("Saving to: ", curr_save_dir)
        print("Max steps:", max_steps)


    training_args = transformers.TrainingArguments(
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=warmup_steps,
        max_steps=max_steps,
        learning_rate=cfg.lr,
        bf16=True,
        bf16_full_eval=True,
        output_dir=curr_save_dir,
        # deepspeed=ds_config,
        fsdp="full_shard",
        fsdp_config={
            # "min_num_params": 1e12,
            # "activation_checkpointing": True,
            "use_orig_params": True,
            "backward_prefetch": "backward_pre",
        },
        save_steps=save_steps,
        save_only_model=True,
        ddp_find_unused_parameters=False,
        weight_decay=cfg.weight_decay,
        eval_strategy="no",
        adam_beta1=0.9,
        adam_beta2=0.95,
    )

    # for continual unlearning, load the target model from last task
    model_path = cfg.model_path if unlearn_times == 1 else last_checkpoint_dir
    # fix the reference model
    reference_model_path = cfg.model_path if cfg.fix_ref_model else model_path

    # load target LLM
    if cfg.use_LoRA and unlearn_times > 1:
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model_path,
            config=config,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
        )
        model.generation_config.do_sample = True
        if model_cfg["gradient_checkpointing"] == "true":
            # model.gradient_checkpointing_enable()
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        peft_config = LoraConfig(
            task_type="CAUSAL_LM",
            inference_mode=False,
            target_modules=find_all_linear_names(model),
            r=cfg.LoRA.r,
            lora_alpha=cfg.LoRA.alpha,
            lora_dropout=cfg.LoRA.dropout,
        )
        model = PeftModel.from_pretrained(
            model, last_checkpoint_dir, config=peft_config, is_trainable=True
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            config=config,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
        )
        model.generation_config.do_sample = True
        if model_cfg["gradient_checkpointing"] == "true":
            # model.gradient_checkpointing_enable()
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

        # Configure LoRA parameters
        if cfg.use_LoRA:
            peft_config = LoraConfig(
                task_type="CAUSAL_LM",
                inference_mode=False,
                target_modules=find_all_linear_names(model),
                r=cfg.LoRA.r,
                lora_alpha=cfg.LoRA.alpha,
                lora_dropout=cfg.LoRA.dropout,
            )
            model = get_peft_model(model, peft_config)

    # load reference model
    if "DPO" in cfg.forget_loss or ("NPO" in cfg.forget_loss and "SimNPO" not in cfg.forget_loss) or "KL" in cfg.forget_loss:
        reference_model = AutoModelForCausalLM.from_pretrained(
            reference_model_path,
            config=config,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
        )
        reference_model = reference_model.eval()
    else:
        reference_model = None

    if cfg.alternate:
        trainer = CustomTrainerForgettingAlternate(
            alternate=cfg.alternate,
            optim_cfg=cfg.optim_cfg,
            forget_lr=float(cfg.forget_lr),
            retain_lr=float(cfg.lr),
            model=model,
            tokenizer=tokenizer,
            train_dataset=torch_format_dataset,
            eval_dataset=torch_format_dataset,
            # the callback for computing metrics, None in this case since you're doing it in your callback
            compute_metrics=None,
            # callbacks=[GlobalStepDeletionCallback],
            args=training_args,
            data_collator=custom_data_collator_forget_single,
            loss_type=cfg.forget_loss,
            ref_model=reference_model,
            beta=cfg.beta,
            forget_coeff=cfg.forget_coeff,
            regularization_coeff=cfg.regularization_coeff,
            forget_freq=cfg.forget_freq,
            retain_freq=cfg.retain_freq,
            alpha=cfg.alpha,
            beta1=cfg.beta1,
            beta2=cfg.beta2,
            base_beta1=cfg.base_beta1,
            base_beta2=cfg.base_beta2,
            max_steps=max_steps,
        )

    else:
        trainer = CustomTrainerForgetting(
            model=model,
            tokenizer=tokenizer,
            train_dataset=torch_format_dataset,
            eval_dataset=torch_format_dataset,
            # the callback for computing metrics, None in this case since you're doing it in your callback
            compute_metrics=None,
            # callbacks=[GlobalStepDeletionCallback],
            args=training_args,
            data_collator=custom_data_collator_forget,
            loss_type=cfg.forget_loss,
            ref_model=reference_model,
            beta=cfg.beta,
            forget_coeff=cfg.forget_coeff,
            regularization_coeff=cfg.regularization_coeff,
        )
    model.config.use_cache = (
        False
    )  # silence the warnings. Please re-enable for inference!

    print("Start Training ...")
    # Start training
    trainer.train()

    if local_rank == 0:
        if os.path.exists(os.path.join(curr_save_dir, f"checkpoint-{max_steps}")):
            if len(task_list) > 1 or cfg.save_steps == "last":
                # continual
                if os.path.exists(os.path.join(curr_save_dir, f"checkpoint-last")):
                    shutil.rmtree(os.path.join(curr_save_dir, f"checkpoint-last"))
                shutil.move(
                    os.path.join(curr_save_dir, f"checkpoint-{max_steps}"),
                    os.path.join(curr_save_dir, f"checkpoint-last"),
                )
            else:
                # single
                if cfg.save_checkpoint:
                    shutil.copytree(
                        os.path.join(curr_save_dir, f"checkpoint-{max_steps}"),
                        os.path.join(curr_save_dir, f"checkpoint-last"),
                    )

        if os.path.exists(last_checkpoint_dir) and not cfg.save_checkpoint:
            # for evaluate last task
            # For continual unlearning, remove the last model checkpoint
            if os.path.exists(
                os.path.join(
                    cfg.save_dir,
                    f"unlearn_times_{unlearn_times - 1}",
                    "eval_results-last",
                )
            ):
                shutil.rmtree(last_checkpoint_dir)
                print("Removed %s" % last_checkpoint_dir)


if __name__ == "__main__":
    torch.cuda.empty_cache()
    main()
