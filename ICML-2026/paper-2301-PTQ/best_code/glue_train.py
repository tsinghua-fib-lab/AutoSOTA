# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0

"""
Finetuning a 🤗 Transformers model for sequence classification on GLUE.
"""

import argparse
import json
import logging
import math
import os
import random
from pathlib import Path
from collections import defaultdict

import datasets
import evaluate
import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset
from huggingface_hub import HfApi
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    PretrainedConfig,
    SchedulerType,
    default_data_collator,
    get_scheduler,
)
from transformers.utils.versions import require_version

from peft import (
    PeftModel,
    prepare_model_for_kbit_training,
)

logger = get_logger(__name__)

require_version(
    "datasets>=1.8.0",
    "To fix: pip install -r examples/pytorch/text-classification/requirements.txt",
)

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Finetune a transformers model on a text classification task"
    )
    parser.add_argument(
        "--task_name",
        type=str,
        default=None,
        help="The name of the glue task to train on.",
        choices=list(task_to_keys.keys()),
    )
    parser.add_argument(
        "--train_file",
        type=str,
        default=None,
        help="A csv or a json file containing the training data.",
    )
    parser.add_argument(
        "--validation_file",
        type=str,
        default=None,
        help="A csv or a json file containing the validation data.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_length` is passed."
        ),
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.0, help="Weight decay to use."
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=3,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=[
            "linear",
            "cosine",
            "cosine_with_restarts",
            "polynomial",
            "constant",
            "constant_with_warmup",
        ],
    )
    parser.add_argument(
        "--num_warmup_steps",
        type=int,
        default=0,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None, help="Where to store the final model."
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="A seed for reproducible training."
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether or not to push the model to the Hub.",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--hub_token", type=str, help="The token to use to push to the Model Hub."
    )
    parser.add_argument(
        "--trust_remote_code",
        type=bool,
        default=False,
        help=(
            "Whether or not to allow for custom models defined on the Hub in their own modeling files. This option "
            "should only be set to `True` for repositories you trust and in which you have read the code, as it will "
            "execute code present on the Hub on your local machine."
        ),
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="all",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"`, `"comet_ml"` and `"clearml"`. Use `"all"` (default) to report to all integrations. '
            "Only applicable when `--with_tracking` is passed."
        ),
    )

    parser.add_argument(
        "--lora_adapter_dir", type=str, default=None, help="'NA' for full-finetune"
    )
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument(
        "--use_gradient_checkpointing", action="store_true", default=False
    )
    parser.add_argument("--config_name", type=str, default=None)
    parser.add_argument("--tokenizer_name", type=str, default=None)
    parser.add_argument("--wandb-tags", type=str, nargs="+", default=None)
    parser.add_argument("--init_method", type=str, default=None)

    parser.add_argument(
        "--freeze_alpha_default",
        type=float,
        default=0.1,
        help="Default grad scaling alpha for frozen LoRA ranks when no per-rank alpha is provided.",
    )
    parser.add_argument(
        "--freeze_alpha_dict_filename",
        type=str,
        default="freeze_alpha_dict.json",
        help="Optional JSON filename under lora_adapter_dir to provide per-rank alpha via s_vals.",
    )

    parser.add_argument(
        "--debug_freeze_hooks",
        action="store_true",
        default=False,
        help="Enable verbose debug prints for freeze hooks (rank-wise alpha scaling).",
    )
    parser.add_argument(
        "--debug_freeze_hooks_first_n_steps",
        type=int,
        default=3,
        help="Print hook debug for the first N optimizer steps only (to avoid flooding).",
    )
    parser.add_argument(
        "--debug_freeze_hooks_max_prints_per_param",
        type=int,
        default=1,
        help="Max number of hook debug prints per parameter.",
    )
    parser.add_argument(
        "--debug_alpha_strict_no_default",
        action="store_true",
        default=False,
        help="If set, raise error if any hook falls back to default alpha.",
    )

    parser.add_argument(
        "--debug_freeze_verify",
        action="store_true",
        default=False,
        help="Verify scaling is applied AND inspect actual parameter updates around optimizer.step().",
    )
    parser.add_argument(
        "--debug_freeze_verify_steps",
        type=int,
        default=2,
        help="Run verify logs for first N optimizer steps only.",
    )
    parser.add_argument(
        "--debug_freeze_verify_max_params",
        type=int,
        default=6,
        help="Max number of LoRA params to track for update-delta verification.",
    )

    args = parser.parse_args()

    # Sanity checks
    if (
        args.task_name is None
        and args.train_file is None
        and args.validation_file is None
    ):
        raise ValueError("Need either a task name or a training/validation file.")
    else:
        if args.train_file is not None:
            ext = args.train_file.split(".")[-1]
            assert ext in ["csv", "json"], "`train_file` should be a csv or a json file."
        if args.validation_file is not None:
            ext = args.validation_file.split(".")[-1]
            assert ext in ["csv", "json"], "`validation_file` should be a csv or a json file."

    if args.push_to_hub:
        assert (
            args.output_dir is not None
        ), "Need an `output_dir` to create a repo when `--push_to_hub` is passed."

    if args.lora_adapter_dir is not None and args.lora_adapter_dir == "NA":
        args.lora_adapter_dir = None

    return args


def _compute_alpha_vec(
    matched_key: str,
    freeze_r: int,
    freeze_alpha_dict: dict | None,
    default_freeze_alpha: float,
):
    if freeze_r <= 0:
        return [], "none", None

    if freeze_alpha_dict is not None and matched_key in freeze_alpha_dict:
        entry = freeze_alpha_dict[matched_key]
        s_vals = None
        if isinstance(entry, dict) and "s_vals" in entry:
            s_vals = entry["s_vals"]
        elif isinstance(entry, list):
            s_vals = entry

        if isinstance(s_vals, list) and len(s_vals) > freeze_r:
            s_ref = float(s_vals[freeze_r])

            alpha_vec = []
            for i in range(freeze_r):
                a_i = 0.1
                alpha_vec.append(a_i)

            return alpha_vec, "file", s_ref

    return [default_freeze_alpha] * freeze_r, "default", None


def _compute_alpha_vec_sgd(
    matched_key: str,
    freeze_r: int,
    freeze_alpha_dict: dict | None,
    default_freeze_alpha: float,
):
    if freeze_r <= 0:
        return [], "none", None

    alpha = 5.0

    if freeze_alpha_dict is not None and matched_key in freeze_alpha_dict:
        entry = freeze_alpha_dict[matched_key]
        s_vals = None
        if isinstance(entry, dict) and "s_vals" in entry:
            s_vals = entry["s_vals"]
        elif isinstance(entry, list):
            s_vals = entry

        if isinstance(s_vals, list) and len(s_vals) > freeze_r:
            s_ref = float(s_vals[freeze_r])

            max_sigma = max(float(s) for s in s_vals)

            if max_sigma <= 1e-30:
                return [default_freeze_alpha] * freeze_r, "file", s_ref

            alpha_vec = []
            for i in range(freeze_r):
                sigma_i = float(s_vals[i])
                if sigma_i < 0.0:
                    print(f"[WARN] negative sigma_i detected")

                denom = alpha * sigma_i + max_sigma
                if denom <= 1e-30:
                    a_i = default_freeze_alpha
                else:
                    lambda_i = ((alpha + 1.0) * sigma_i) / denom

                    a_i = 1.0 - lambda_i

                    if a_i < 0.0:
                        a_i = 0.0
                    elif a_i > 1.0:
                        a_i = 1.0

                alpha_vec.append(a_i)

            return alpha_vec, "file", s_ref

    return [default_freeze_alpha] * freeze_r, "default", None



def _fmt_alpha_list(a, head=10, tail=4):
    if len(a) <= head + tail:
        return "[" + ", ".join([f"{x:.4g}" for x in a]) + "]"
    return (
        "[" + ", ".join([f"{x:.4g}" for x in a[:head]])
        + ", ... , "
        + ", ".join([f"{x:.4g}" for x in a[-tail:]])
        + "]"
    )


def main():
    args = parse_args()
    mixed_precision = "bf16"
    accelerator = (
        Accelerator(
            log_with=args.report_to,
            project_dir=args.output_dir,
            mixed_precision=mixed_precision,
        )
        if args.with_tracking
        else Accelerator(mixed_precision="fp16")
    )

    IS_MAIN = accelerator.is_main_process

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    api = None
    repo_id = None
    if accelerator.is_main_process:
        if args.push_to_hub:
            repo_name = args.hub_model_id
            if repo_name is None:
                repo_name = Path(args.output_dir).absolute().name
            api = HfApi()
            repo_id = api.create_repo(
                repo_name, exist_ok=True, token=args.hub_token
            ).repo_id

            with open(os.path.join(args.output_dir, ".gitignore"), "w+") as gitignore:
                if "step_*" not in gitignore:
                    gitignore.write("step_*\n")
                if "epoch_*" not in gitignore:
                    gitignore.write("epoch_*\n")
        elif args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        accelerator.wait_for_everyone()

    # Get the datasets
    if args.task_name is not None:
        raw_datasets = load_dataset("nyu-mll/glue", args.task_name)
    else:
        data_files = {}
        if args.train_file is not None:
            data_files["train"] = args.train_file
        if args.validation_file is not None:
            data_files["validation"] = args.validation_file
        extension = (
            args.train_file if args.train_file is not None else args.validation_file
        ).split(".")[-1]
        raw_datasets = load_dataset(extension, data_files=data_files)

    # Labels
    if args.task_name is not None:
        is_regression = args.task_name == "stsb"
        if not is_regression:
            label_list = raw_datasets["train"].features["label"].names
            num_labels = len(label_list)
        else:
            num_labels = 1
    else:
        is_regression = raw_datasets["train"].features["label"].dtype in [
            "float32",
            "float64",
        ]
        if is_regression:
            num_labels = 1
        else:
            label_list = raw_datasets["train"].unique("label")
            label_list.sort()
            num_labels = len(label_list)

    # Load pretrained model and tokenizer
    if args.config_name is not None:
        config = AutoConfig.from_pretrained(
            args.config_name,
            num_labels=num_labels,
            finetuning_task=args.task_name,
            trust_remote_code=args.trust_remote_code,
        )
    else:
        config = AutoConfig.from_pretrained(
            args.model_name_or_path,
            num_labels=num_labels,
            finetuning_task=args.task_name,
            trust_remote_code=args.trust_remote_code,
        )

    if args.tokenizer_name is not None:
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_name,
            use_fast=not args.use_slow_tokenizer,
            trust_remote_code=args.trust_remote_code,
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path,
            use_fast=not args.use_slow_tokenizer,
            trust_remote_code=args.trust_remote_code,
        )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    config.pad_token_id = tokenizer.pad_token_id

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=args.trust_remote_code,
        ignore_mismatched_sizes=False,
        num_labels=num_labels,
    )

    _model_loaded_from_local = Path(args.model_name_or_path).exists()

    _model_is_bnb_quantized = False
    if hasattr(model, "is_quantized") and model.is_quantized:
        _model_is_bnb_quantized = True
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=args.use_gradient_checkpointing,
            gradient_checkpointing_kwargs={"use_reentrant": True},
        )

    _adapter_is_applied = False
    if args.lora_adapter_dir is not None:
        assert Path(
            args.lora_adapter_dir
        ).exists(), f"{args.lora_adapter_dir} does not exist."
        model = PeftModel.from_pretrained(
            model,
            args.lora_adapter_dir,
            is_trainable=True,
            ignore_mismatched_sizes=True,
        )
        _adapter_is_applied = True

    logger.info(f"🔍 model loaded from local: {_model_loaded_from_local}")
    logger.info(
        f"🔍 model is bnb quantized: {_model_is_bnb_quantized} (emulated quantization is possible if False)"
    )
    logger.info(f"🔍 adapter is applied: {_adapter_is_applied}")

    if _adapter_is_applied:
        trainable_params, all_param = model.get_nb_trainable_parameters()
        logger.info(
            f"🔍 trainable params: {trainable_params:,d} || all params: {all_param:,d} || trainable%: {100 * trainable_params / all_param:.4f}"
        )
    else:
        logger.info(
            f"🔍 Full-finetune: all params: {model.num_parameters()}, trainable: {model.num_parameters(only_trainable=True)}"
        )

    # Preprocessing keys
    if args.task_name is not None:
        sentence1_key, sentence2_key = task_to_keys[args.task_name]
    else:
        non_label_column_names = [
            name for name in raw_datasets["train"].column_names if name != "label"
        ]
        if "sentence1" in non_label_column_names and "sentence2" in non_label_column_names:
            sentence1_key, sentence2_key = "sentence1", "sentence2"
        else:
            if len(non_label_column_names) >= 2:
                sentence1_key, sentence2_key = non_label_column_names[:2]
            else:
                sentence1_key, sentence2_key = non_label_column_names[0], None

    # label mapping
    label_to_id = None
    if (
        model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id
        and args.task_name is not None
        and not is_regression
    ):
        label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
        if sorted(label_name_to_id.keys()) == sorted(label_list):
            logger.info(
                f"The configuration of the model provided the following label correspondence: {label_name_to_id}. Using it!"
            )
            label_to_id = {
                i: label_name_to_id[label_list[i]] for i in range(num_labels)
            }
        else:
            logger.warning(
                "Your model seems to have been trained with labels, but they don't match the dataset: "
                f"model labels: {sorted(label_name_to_id.keys())}, dataset labels: {sorted(label_list)}.\nIgnoring the model labels."
            )
    elif args.task_name is None and not is_regression:
        label_to_id = {v: i for i, v in enumerate(label_list)}

    if label_to_id is not None:
        model.config.label2id = label_to_id
        model.config.id2label = {i: l for l, i in model.config.label2id.items()}
    elif args.task_name is not None and not is_regression:
        model.config.label2id = {l: i for i, l in enumerate(label_list)}
        model.config.id2label = {i: l for l, i in model.config.label2id.items()}

    padding = "max_length" if args.pad_to_max_length else False

    def preprocess_function(examples):
        texts = (
            (examples[sentence1_key],)
            if sentence2_key is None
            else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*texts, padding=padding, max_length=args.max_length, truncation=True)
        if "label" in examples:
            if label_to_id is not None:
                result["labels"] = [label_to_id[l] for l in examples["label"]]
            else:
                result["labels"] = examples["label"]
        return result

    with accelerator.main_process_first():
        processed_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            remove_columns=raw_datasets["train"].column_names,
            desc="Running tokenizer on dataset",
        )

    train_dataset = processed_datasets["train"]
    eval_dataset = processed_datasets[
        "validation_matched" if args.task_name == "mnli" else "validation"
    ]

    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # DataLoaders
    if args.pad_to_max_length:
        data_collator = default_data_collator
    else:
        data_collator = DataCollatorWithPadding(
            tokenizer,
            pad_to_multiple_of=(8 if accelerator.mixed_precision == "fp16" else None),
        )

    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=data_collator,
        batch_size=args.per_device_train_batch_size,
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        collate_fn=data_collator,
        batch_size=args.per_device_eval_batch_size,
    )

    freeze_rank_dict = {}
    freeze_alpha_dict = None

    if args.lora_adapter_dir is not None:
        fr_path = Path(args.lora_adapter_dir) / "freeze_rank_dict.json"
        if fr_path.exists():
            with open(fr_path, "r") as f:
                freeze_rank_dict = json.load(f)

        fa_path = Path(args.lora_adapter_dir) / args.freeze_alpha_dict_filename
        if fa_path.exists():
            with open(fa_path, "r") as f:
                freeze_alpha_dict = json.load(f)

    if len(freeze_rank_dict) > 0:
        for name, module in model.named_modules():
            adapter_name = "default"

            matched_key = None
            for key in freeze_rank_dict:
                if key in name:
                    matched_key = key
                    break
            if matched_key is None:
                continue

            freeze_r = int(freeze_rank_dict[matched_key])
            if freeze_r <= 0:
                continue

            if not hasattr(module, "lora_A") or not hasattr(module, "lora_B"):
                continue
            if adapter_name not in module.lora_A or adapter_name not in module.lora_B:
                continue

            lora_A = module.lora_A[adapter_name]
            lora_B = module.lora_B[adapter_name]
            total_r = lora_A.out_features
            assert freeze_r < total_r, f"freeze_r ({freeze_r}) must be < LoRA rank ({total_r})"

            old_A = lora_A.weight.data.clone()
            old_B = lora_B.weight.data.clone()

            del module.lora_A
            del module.lora_B

            module.lora_A_frozen = torch.nn.Linear(module.in_features, freeze_r, bias=False)
            module.lora_B_frozen = torch.nn.Linear(freeze_r, module.out_features, bias=False)
            module.lora_A_frozen.weight = torch.nn.Parameter(old_A[:freeze_r], requires_grad=True)
            module.lora_B_frozen.weight = torch.nn.Parameter(old_B[:, :freeze_r], requires_grad=True)

            train_r = total_r - freeze_r
            module.lora_A_train = torch.nn.Linear(module.in_features, train_r, bias=False)
            module.lora_B_train = torch.nn.Linear(train_r, module.out_features, bias=False)
            module.lora_A_train.weight = torch.nn.Parameter(old_A[freeze_r:], requires_grad=True)
            module.lora_B_train.weight = torch.nn.Parameter(old_B[:, freeze_r:], requires_grad=True)

            def lora_forward_split(self, x):
                result = torch.nn.functional.linear(x, self.weight, self.bias)

                active = getattr(self, "active_adapter", "default")
                if isinstance(active, (list, tuple)) and len(active) > 0:
                    an = active[0]
                elif isinstance(active, str):
                    an = active
                else:
                    an = "default"

                if isinstance(getattr(self, "lora_dropout", None), dict) and an in self.lora_dropout:
                    x_dropped = self.lora_dropout[an](x)
                else:
                    x_dropped = x

                if isinstance(getattr(self, "scaling", None), dict) and an in self.scaling:
                    sc = self.scaling[an]
                else:
                    sc = getattr(self, "scaling", 1.0)

                # LoRA deltas
                delta_frozen = torch.nn.functional.linear(
                    torch.nn.functional.linear(x, self.lora_A_frozen.weight),
                    self.lora_B_frozen.weight,
                )
                delta_train = torch.nn.functional.linear(
                    torch.nn.functional.linear(x_dropped, self.lora_A_train.weight),
                    self.lora_B_train.weight,
                )
                return result + (delta_frozen + delta_train) * sc

            module.forward = lora_forward_split.__get__(module, module.__class__)

    # Optimizer groups
    no_decay = ["bias", "layer_norm.weight"]
    params_with_weight_decay = []
    param_names_with_weight_decay = []
    params_without_weight_decay = []
    params_without_weight_decay_names = []


    alpha_cache = {}  
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue

        matched_key = None
        for key in freeze_rank_dict:
            if key in n:
                matched_key = key
                break

        if matched_key is not None and ("lora_A_frozen.weight" in n or "lora_B_frozen.weight" in n):
            freeze_r = int(freeze_rank_dict[matched_key])

            if matched_key not in alpha_cache:
                alpha_vec, alpha_source, s_ref = _compute_alpha_vec(
                    matched_key=matched_key,
                    freeze_r=freeze_r,
                    freeze_alpha_dict=freeze_alpha_dict,
                    default_freeze_alpha=args.freeze_alpha_default,
                )
                alpha_cache[matched_key] = (freeze_r, alpha_vec, alpha_source, s_ref)

                if args.debug_alpha_strict_no_default and alpha_source == "default":
                    raise RuntimeError(
                        f"[debug_alpha_strict_no_default] alpha fell back to default for key={matched_key}"
                    )

            freeze_r, alpha_vec, alpha_source, _ = alpha_cache[matched_key]

            def make_hook(
                param_name: str,
                is_A: bool,
                key: str,
                freeze_r_local: int,
                alpha_vec_local: list[float],
            ):
                def _hook(grad: torch.Tensor):
                    if grad is None:
                        return grad

                    a = torch.tensor(alpha_vec_local, device=grad.device, dtype=grad.dtype)

                    with torch.no_grad():
                        before_norm = float(grad.norm().item())

                    if is_A:
                        grad2 = grad * a.view(-1, 1)
                    else:
                        grad2 = grad * a.view(1, -1)

                    with torch.no_grad():
                        after_norm = float(grad2.norm().item())
                    
                    return grad2

                return _hook

            if "lora_A_frozen.weight" in n:
                p.register_hook(make_hook(n, True, matched_key, freeze_r, alpha_vec))
            elif "lora_B_frozen.weight" in n:
                p.register_hook(make_hook(n, False, matched_key, freeze_r, alpha_vec))

    # build optimizer groups
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if any(nd in n for nd in no_decay):
            params_without_weight_decay.append(p)
            params_without_weight_decay_names.append(n)
        else:
            params_with_weight_decay.append(p)
            param_names_with_weight_decay.append(n)

    optimizer_grouped_parameters = [
        {"params": params_with_weight_decay, "weight_decay": args.weight_decay},
        {"params": params_without_weight_decay, "weight_decay": 0.0},
    ]
    if IS_MAIN:
        logger.info(f"Parameters with weight decay: {param_names_with_weight_decay}")
        logger.info(f"Parameters without weight decay: {params_without_weight_decay_names}")

    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Scheduler setup
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    # Prepare with accelerator
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )

    # Recalculate steps after prepare
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    # Trackers
    if args.with_tracking:
        experiment_config = vars(args)
        experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
        tracker_init_kwargs = {"wandb": {}}
        if args.run_name is not None:
            tracker_init_kwargs["wandb"]["name"] = args.run_name
        if args.wandb_tags is not None:
            tracker_init_kwargs["wandb"]["tags"] = args.wandb_tags
        accelerator.init_trackers("glue_train", experiment_config, tracker_init_kwargs)

    # Metric
    metric = evaluate.load("glue", args.task_name) if args.task_name is not None else evaluate.load("accuracy")

    # Train info
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0
    resume_step = None

    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
            checkpoint_path = args.resume_from_checkpoint
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[-1]
            checkpoint_path = path
            path = os.path.basename(checkpoint_path)

        accelerator.print(f"Resumed from checkpoint: {checkpoint_path}")
        accelerator.load_state(checkpoint_path)
        training_difference = os.path.splitext(path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
            completed_steps = starting_epoch * num_update_steps_per_epoch
        else:
            resume_step = int(training_difference.replace("step_", "")) * args.gradient_accumulation_steps
            starting_epoch = resume_step // len(train_dataloader)
            completed_steps = resume_step // args.gradient_accumulation_steps
            resume_step -= starting_epoch * len(train_dataloader)

    progress_bar.update(completed_steps)


    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()

        epoch_loss_sum = 0.0
        epoch_loss_count = 0

        if args.with_tracking:
            total_loss_for_tracker = 0.0

        if args.resume_from_checkpoint and epoch == starting_epoch and resume_step is not None:
            active_dataloader = accelerator.skip_first_batches(train_dataloader, resume_step)
        else:
            active_dataloader = train_dataloader

        for step, batch in enumerate(active_dataloader):
            outputs = model(**batch)
            loss = outputs.loss

            # accumulate epoch loss
            epoch_loss_sum += float(loss.detach().float().item())
            epoch_loss_count += 1

            if args.with_tracking:
                total_loss_for_tracker += float(loss.detach().float().item())


            loss = loss / args.gradient_accumulation_steps
            accelerator.backward(loss)

            do_step = (step % args.gradient_accumulation_steps == 0) or (step == len(train_dataloader) - 1)


            if do_step:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                progress_bar.update(1)
                completed_steps += 1

                if args.with_tracking:
                    accelerator.log({"train_step_loss": loss.detach().float()}, step=completed_steps)


            if isinstance(checkpointing_steps, int):
                if completed_steps % checkpointing_steps == 0:
                    output_dir = f"step_{completed_steps}"
                    if args.output_dir is not None:
                        output_dir = os.path.join(args.output_dir, output_dir)
                    accelerator.save_state(output_dir)

            if completed_steps >= args.max_train_steps:
                break

        # Eval
        model.eval()
        samples_seen = 0
        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                outputs = model(**batch)

            predictions = outputs.logits.argmax(dim=-1) if not is_regression else outputs.logits.squeeze()
            predictions, references = accelerator.gather((predictions, batch["labels"]))

            if accelerator.num_processes > 1:
                if step == len(eval_dataloader) - 1:
                    predictions = predictions[: len(eval_dataloader.dataset) - samples_seen]
                    references = references[: len(eval_dataloader.dataset) - samples_seen]
                else:
                    samples_seen += references.shape[0]

            metric.add_batch(predictions=predictions, references=references)

        eval_metric = metric.compute()
        logger.info(f"epoch {epoch}: {eval_metric}")

        # Push intermediate checkpoints
        if args.push_to_hub and epoch < args.num_train_epochs - 1:
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(
                args.output_dir,
                is_main_process=accelerator.is_main_process,
                save_function=accelerator.save,
            )
            if accelerator.is_main_process:
                tokenizer.save_pretrained(args.output_dir)
                api.upload_folder(
                    commit_message=f"Training in progress epoch {epoch}",
                    folder_path=args.output_dir,
                    repo_id=repo_id,
                    repo_type="model",
                    token=args.hub_token,
                )

        if args.checkpointing_steps == "epoch":
            output_dir = f"epoch_{epoch}"
            if args.output_dir is not None:
                output_dir = os.path.join(args.output_dir, output_dir)
            accelerator.save_state(output_dir)

    if args.with_tracking:
        accelerator.end_training()

    # Save final
    if args.output_dir is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            args.output_dir,
            is_main_process=accelerator.is_main_process,
            save_function=accelerator.save,
        )
        if accelerator.is_main_process:
            tokenizer.save_pretrained(args.output_dir)
            if args.push_to_hub:
                api.upload_folder(
                    commit_message="End of training",
                    folder_path=args.output_dir,
                    repo_id=repo_id,
                    repo_type="model",
                    token=args.hub_token,
                )

    # MNLI mismatch eval
    if args.task_name == "mnli":
        eval_dataset_mm = processed_datasets["validation_mismatched"]
        eval_dataloader_mm = DataLoader(
            eval_dataset_mm,
            collate_fn=data_collator,
            batch_size=args.per_device_eval_batch_size,
        )
        eval_dataloader_mm = accelerator.prepare(eval_dataloader_mm)

        metric_mm = evaluate.load("glue", "mnli")
        model.eval()
        for step, batch in enumerate(eval_dataloader_mm):
            with torch.no_grad():
                outputs = model(**batch)
            predictions = outputs.logits.argmax(dim=-1)
            metric_mm.add_batch(
                predictions=accelerator.gather(predictions),
                references=accelerator.gather(batch["labels"]),
            )

        eval_metric_mm = metric_mm.compute()
        logger.info(f"mnli-mm: {eval_metric_mm}")

    # Save results json
    if args.output_dir is not None and accelerator.is_main_process:
        all_results = {f"eval_{k}": v for k, v in eval_metric.items()}
        with open(os.path.join(args.output_dir, "all_results.json"), "w") as f:
            json.dump(all_results, f, indent=2)


if __name__ == "__main__":
    main()
