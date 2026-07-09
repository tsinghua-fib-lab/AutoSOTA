import hydra
import logging
import os
import numpy as np
import pandas as pd
import pprint

import sys
import torch
from contextlib import nullcontext
from datasets import Dataset
from ..test_functions.finetune_utils import (
    EvaluatorEditPairs,
    EvaluatorPlainPairs,
    get_response_template_edit_pairs,
    get_response_template_plain_pairs,
    formatting_texts_func_edit_pairs,
    formatting_texts_func_plain_pairs,
    wandb_setup,
)
from ..core.model_client import ModelClient
from omegaconf import DictConfig, OmegaConf
from ..train.seq2seq_sft_trainer import S3Callback, Seq2SeqSFTConfig, Seq2SeqSFTTrainer
from tqdm.rich import tqdm
from trl import init_zero_verbose
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    set_seed,
)
from transformers.utils import logging as transformers_logging

from trl import (
    ModelConfig,
    RichProgressCallback,
    get_peft_config,
    get_quantization_config,
    get_kbit_device_map,
)
from ..train.data_collators import DataCollatorForCompletionOnlyLM
from typing import Any, List, Optional

TRL_USE_RICH = os.environ.get("TRL_USE_RICH", False)

if TRL_USE_RICH:
    init_zero_verbose()
    FORMAT = "%(message)s"

    from rich.console import Console
    from rich.logging import RichHandler

logger = logging.getLogger(__name__)

tqdm.pandas()

if TRL_USE_RICH:
    logging.basicConfig(
        format=FORMAT, datefmt="[%X]", handlers=[RichHandler()], level=logging.INFO
    )

from ..infrastructure.file_handler import LocalOrS3Client


@hydra.main(config_path="../../../config/finetune", config_name="pythia-2.8b")
def main(cfg: DictConfig):
    wandb_setup(cfg)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        level=cfg.log_level.upper(),
        force=True,
    )

    cfg_dict = OmegaConf.to_container(cfg)
    generation_config = GenerationConfig(**cfg_dict["generation_config"])
    training_args = Seq2SeqSFTConfig(
        **cfg_dict["training_args"],
        generation_config=generation_config,
    )
    model_config = ModelConfig(**cfg_dict["model_config"])
    logger.info(f"training_args: {training_args}")
    logger.info(f"model_config: {model_config}")
    logger.info(f"generation config: {training_args.generation_config.to_dict()}")

    # Force use our print callback
    if TRL_USE_RICH:
        training_args.disable_tqdm = True
        console = Console()

    ################
    # Model & Tokenizer
    ################
    torch_dtype = (
        model_config.dtype
        if model_config.dtype in ["auto", None]
        else getattr(torch, model_config.dtype)
    )
    quantization_config = get_quantization_config(model_config)
    device_map = get_kbit_device_map() if quantization_config is not None else None
    model_kwargs = dict(
        # revision=model_config.model_revision,
        trust_remote_code=model_config.trust_remote_code,
        attn_implementation=model_config.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False,
        device_map=device_map,
        quantization_config=quantization_config,
    )
    logger.info(f"Model initialization args: {pprint.pformat(model_kwargs)}")
    set_seed(training_args.seed)
    if model_config.model_name_or_path.startswith("s3://"):
        # use the ModelClient class since it has utilities for loading models from S3
        model_client = ModelClient(
            model_config.model_name_or_path, logger=logger, **model_kwargs
        )
        model = model_client.model
        tokenizer = model_client.tokenizer
        tokenizer.padding_side = "right"
    elif cfg.train_from_scratch:
        model_cfg_dict = OmegaConf.to_container(cfg.init_model_config)
        config = AutoConfig.from_pretrained(
            model_config.model_name_or_path, **model_cfg_dict
        )
        model = AutoModelForCausalLM.from_config(config)
        tokenizer = AutoTokenizer.from_pretrained(
            model_config.model_name_or_path, use_fast=True
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_config.model_name_or_path, **model_kwargs
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_config.model_name_or_path, use_fast=True
        )
    tokenizer.pad_token = tokenizer.eos_token

    ################
    # Dataset
    ################
    raw_df = pd.read_json(cfg.data_fp, orient="records", lines=True)
    raw_datasets = Dataset.from_pandas(raw_df).train_test_split(
        train_size=cfg.train_size, shuffle=True, seed=training_args.seed
    )
    train_dataset = raw_datasets["train"]
    eval_dataset = raw_datasets["test"]
    print(f"cfg.data_fp : {cfg.data_fp}")
    print(f"train_dataset len : {len(train_dataset)}")
    print(f"eval_dataset len : {len(eval_dataset)}")

    if cfg.sanity_check:
        logger.info("In sanity check mode, will reduce the number of training steps.")
        train_dataset = train_dataset.select(range(min(len(train_dataset), 1000)))
        eval_dataset = eval_dataset.select(range(min(len(eval_dataset), 50)))
        training_args.num_train_epochs = 1
        training_args.save_strategy = "epoch"
        training_args.eval_strategy = "steps"
        training_args.eval_steps = 5
    else:
        if cfg.max_eval_size is not None:
            eval_dataset = eval_dataset.select(
                range(min(cfg.max_eval_size, len(eval_dataset)))
            )
    logger.info(
        f"Training dataset: {len(train_dataset)} examples\nEval dataset: {len(eval_dataset)} examples\n"
    )

    ################
    # Optional rich context managers
    ###############
    init_context = (
        nullcontext()
        if not TRL_USE_RICH
        else console.status("[bold green]Initializing the SFTTrainer...")
    )
    save_context = (
        nullcontext()
        if not TRL_USE_RICH
        else console.status(
            f"[bold green]Training completed! Saving the model to {training_args.output_dir}"
        )
    )

    ################
    # Training
    ################
    with init_context:
        # use transformers logger during the training loop
        transformers_logging.enable_default_handler()
        transformers_logging.enable_propagation()
        transformers_logger = transformers_logging.get_logger("transformers")
        try:
            transformers_logger.setLevel(cfg.log_level.upper())
        except (ValueError, AttributeError):
            logger.warning(
                f"Could not set transformers logger to level {cfg.log_level}. Keeping defaults..."
            )

        if cfg.format_type == "plain_pairs":
            formatting_fn_batched = formatting_texts_func_plain_pairs
        elif cfg.format_type == "edit_pairs":
            formatting_fn_batched = formatting_texts_func_edit_pairs
        else:
            raise ValueError(f"Unsupported format type: {cfg.format_type}")

        # trl 0.29 calls formatting_func per-example (not batched).
        # Wrap the batched function to handle single-example dicts.
        def formatting_fn(example):
            batch = {k: [v] for k, v in example.items()}
            return formatting_fn_batched(batch)[0]

        if cfg.train_from_scratch:
            # train new tokenizer
            logging.info(
                f"Training new tokenizer with vocab size {cfg.init_model_config.vocab_size}"
            )
            if cfg.format_type == "plain_pairs":
                formatted_inputs = formatting_fn_batched(train_dataset)
            else:
                formatted_inputs = formatting_fn_batched(
                    train_dataset, include_target=True
                )
            logging.info(
                f"Example of first 5 examples used for training custom tokenizer: {formatted_inputs[:5]}"
            )
            batch_size = 1000
            train_iterator = [
                formatted_inputs[i : i + batch_size]
                for i in np.arange(0, len(formatted_inputs), batch_size)
            ]
            tokenizer = tokenizer.train_new_from_iterator(
                train_iterator, vocab_size=cfg.init_model_config.vocab_size
            )
            tokenizer.save_pretrained(f"{training_args.output_dir}/custom_tokenizer")
            logging.info(
                f"Trained and saved custom tokenizer with {len(tokenizer)} tokens."
            )

        if cfg.format_type == "plain_pairs":
            response_template_ids = get_response_template_plain_pairs(tokenizer)
            compute_metrics = EvaluatorPlainPairs(
                cfg, tokenizer, logger=transformers_logger
            )
        elif cfg.format_type == "edit_pairs":
            response_template_ids = get_response_template_edit_pairs(tokenizer)
            compute_metrics = EvaluatorEditPairs(
                cfg, tokenizer, logger=transformers_logger
            )
        else:
            raise ValueError(f"Unsupported format type: {cfg.format_type}")

        collator = DataCollatorForCompletionOnlyLM(
            response_template_ids, tokenizer=tokenizer
        )
        callbacks = [RichProgressCallback] if TRL_USE_RICH else []
        if cfg.s3_output_dir is not None:
            s3_callback = S3Callback(cfg.s3_output_dir, logger=transformers_logger)
            callbacks.append(s3_callback)
        trainer = Seq2SeqSFTTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
            peft_config=get_peft_config(model_config),
            callbacks=callbacks,
            formatting_func=formatting_fn,
            data_collator=collator,
            compute_metrics=compute_metrics,
        )

    trainer.evaluate()

    trainer.train()

    ################################
    # Prediction with final model
    ################################
    predict_output = trainer.predict(trainer.eval_dataset)

    def find_list_in_other_list(
        haystack: List[Any], needle: List[Any]
    ) -> Optional[int]:
        """If needle exists in haystack, return the first index at which it can be found."""
        for i, elem in enumerate(haystack):
            if elem == needle[0] and haystack[i : i + len(needle)] == needle:
                return i
        return None

    output_token_ids = predict_output.predictions
    output_token_ids[output_token_ids == -100] = tokenizer.pad_token_id
    output_pred_rows = []
    for ex, ex_token_ids in zip(eval_dataset, output_token_ids):
        ex_token_ids_list = ex_token_ids.tolist()
        response_template_idx = find_list_in_other_list(
            ex_token_ids_list, response_template_ids
        )
        if response_template_idx is None:
            transformers_logger.warning(
                f"Could not find response template IDs '{response_template_ids}'"
                + f" in generated tokens. Generated tokens:\n{ex_token_ids_list}"
            )
            continue
        gen_input = ex_token_ids_list[
            : response_template_idx + len(response_template_ids)
        ]
        gen_output = ex_token_ids_list[
            response_template_idx + len(response_template_ids) :
        ]
        output_pred_rows.append(
            {
                "generation_input": tokenizer.decode(
                    gen_input,
                    skip_special_tokens=True,
                    clean_up_tokenization_steps=True,
                ),
                "generation_output": tokenizer.decode(
                    gen_output,
                    skip_special_tokens=True,
                    clean_up_tokenization_steps=True,
                ),
                **ex,
            }
        )
    output_pred_rows = pd.DataFrame(output_pred_rows)
    output_pred_rows.to_json(
        os.path.join(training_args.output_dir, "final_predictions.jsonl"),
        orient="records",
        lines=True,
    )

    with save_context:
        trainer.save_model(training_args.output_dir)
        if cfg.s3_output_dir is not None:
            transformers_logger.info("Copy output files to s3_output_dir")

            # Determine if we're using S3 or local storage
            is_s3 = cfg.s3_output_dir.startswith("s3://")
            fs = LocalOrS3Client(init_s3=is_s3)

            for fn in os.listdir(training_args.output_dir):
                if fn not in ["runs"]:  # skip the runs dir
                    fp = os.path.join(training_args.output_dir, fn)
                    recursive = os.path.isdir(fp)
                    transformers_logger.info(f"Copying {fp} to {cfg.s3_output_dir}...")
                    fs.put(fp, cfg.s3_output_dir, recursive=recursive)


if __name__ == "__main__":
    main()
