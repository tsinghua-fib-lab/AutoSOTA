# flake8: noqa
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
# regular:
python examples/scripts/sft.py \
    --model_name_or_path="facebook/opt-350m" \
    --report_to="wandb" \
    --learning_rate=1.41e-5 \
    --per_device_train_batch_size=64 \
    --gradient_accumulation_steps=16 \
    --output_dir="sft_openassistant-guanaco" \
    --logging_steps=1 \
    --num_train_epochs=3 \
    --max_steps=-1 \
    --push_to_hub \
    --gradient_checkpointing

# peft:
python examples/scripts/sft.py \
    --model_name_or_path="facebook/opt-350m" \
    --report_to="wandb" \
    --learning_rate=1.41e-5 \
    --per_device_train_batch_size=64 \
    --gradient_accumulation_steps=16 \
    --output_dir="sft_openassistant-guanaco" \
    --logging_steps=1 \
    --num_train_epochs=3 \
    --max_steps=-1 \
    --push_to_hub \
    --gradient_checkpointing \
    --use_peft \
    --lora_r=64 \
    --lora_alpha=16
"""


# export CUDA_HOME=/usr/local/cuda/

import logging
import os
from contextlib import nullcontext
os.environ['HF_HOME'] = './models'
TRL_USE_RICH = os.environ.get("TRL_USE_RICH", False)

from trl.commands.cli_utils import init_zero_verbose, SFTScriptArguments, TrlParser
from utils import set_pad_token
from tokenizers.processors import TemplateProcessing

if TRL_USE_RICH:
    init_zero_verbose()
    FORMAT = "%(message)s"

    from rich.console import Console
    from rich.logging import RichHandler

import torch
from datasets import load_dataset, load_from_disk
from accelerate.utils import release_memory
from tqdm.rich import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling

from trl import (
    ModelConfig,
    RichProgressCallback,
    SFTConfig,
    SFTTrainer,
    get_peft_config,
    get_quantization_config,
    get_kbit_device_map,
)

tqdm.pandas()

if TRL_USE_RICH:
    logging.basicConfig(format=FORMAT, datefmt="[%X]", handlers=[RichHandler()], level=logging.INFO)


if __name__ == "__main__":

    parser = TrlParser((SFTScriptArguments, SFTConfig, ModelConfig))
    args, training_args, model_config = parser.parse_args_and_config()

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        training_args.gradient_accumulation_steps = training_args.gradient_accumulation_steps // world_size

    # Force use our print callback
    if TRL_USE_RICH:
        training_args.disable_tqdm = True
        console = Console()

    if training_args.gradient_checkpointing:
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": False}

    ################
    # Model & Tokenizer
    ################
    torch_dtype = (
        model_config.torch_dtype
        if model_config.torch_dtype in ["auto", None]
        else getattr(torch, model_config.torch_dtype)
    )
    quantization_config = get_quantization_config(model_config)
    model_kwargs = dict(
        revision=model_config.model_revision,
        trust_remote_code=model_config.trust_remote_code,
        attn_implementation=model_config.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        # device_map=PartialState().process_index,
        # device_map={"": Accelerator().local_process_index}, 
        # device_map=get_kbit_device_map() if quantization_config is not None else None,
        # device_map="auto",
        # device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)},
        device_map=device_map,
        quantization_config=quantization_config,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_config.model_name_or_path, use_fast=True, TOKENIZERS_PARALLELISM=True)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    bos = tokenizer.bos_token
    eos = tokenizer.eos_token
    tokenizer._tokenizer.post_processor =TemplateProcessing(
        single=f"{bos}:0 $A:0 {eos}:0",
        pair=f"{bos}:0 $A:0 {eos}:0 {bos}:1 $B:1 {eos}:1",
        special_tokens=[
            (f"{bos}", tokenizer.bos_token_id), 
            (f"{eos}", tokenizer.eos_token_id)
        ],
    )


    # tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(model_config.model_name_or_path, **model_kwargs)

    # tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    # data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    # tokenizer.add_eos_token = True
    # tokenizer.add_bos_token = True
    # set_pad_token(tokenizer, model)
    # tokenizer.add_eos_token = True
    # if tokenizer.pad_token is None:
    #     tokenizer.pad_token = tokenizer.eos_token
    


    ################
    # Dataset
    ################
    raw_datasets = load_from_disk(args.dataset_name)
    # raw_datasets = load_dataset("kashif/sft_openassistant-guanaco")
    print(raw_datasets)
    
    train_dataset = raw_datasets[args.dataset_train_split]
    eval_dataset = raw_datasets[args.dataset_test_split]

    ################
    # Optional rich context managers
    ###############
    init_context = nullcontext() if not TRL_USE_RICH else console.status("[bold green]Initializing the SFTTrainer...")
    save_context = (
        nullcontext()
        if not TRL_USE_RICH
        else console.status(f"[bold green]Training completed! Saving the model to {training_args.output_dir}")
    )

    ################
    # Training
    ################
    # peft_config=get_peft_config(model_config)
    # peft_config.device_map={"": PartialState().process_index}

    if not ddp and torch.cuda.device_count() > 1:
        # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        model_config.is_parallelizable = True
        model_config.model_parallel = True
    else:
        model_config.is_parallelizable = False
        model_config.model_parallel = True

    with init_context:
        trainer = SFTTrainer(
            # model=model_config.model_name_or_path,
            # model_init_kwargs=model_kwargs,
            # dataset_kwargs={"add_special_tokens": False},
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            # eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            peft_config=get_peft_config(model_config),
            callbacks=[RichProgressCallback] if TRL_USE_RICH else None,
            # packing=True,
        )

    trainer.train()

    with save_context:
        trainer.save_model(training_args.output_dir)