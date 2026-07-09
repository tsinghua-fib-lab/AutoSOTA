import dataclasses
import hydra
import json
import logging
import os
import pandas as pd
import sys
import torch
from contextlib import nullcontext
from datasets import Dataset
from ..infrastructure.file_handler import LocalOrS3Client
from ..test_functions.finetune_utils import (
    formatting_texts_func_edit_pairs,
    get_ehrlich_metrics_for_outputs,
    get_ehrlich_rewards,
    load_test_fn_from_file,
    strtobool,
    wandb_setup,
)
from ..core.model_client import ModelClient
from ..data_contracts import (
    HIGHER_SCORE,
    HIGHER_SCORE_PARTICLE,
    LOWER_SCORE,
    LOWER_SCORE_PARTICLE,
)
from omegaconf import DictConfig, OmegaConf
from .marge_trainer import MargeTrainer, MargeConfig
from .seq2seq_sft_trainer import S3Callback
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, set_seed
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from transformers.utils import logging as transformers_logging
from trl import (
    ModelConfig,
    RichProgressCallback,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
from trl import init_zero_verbose

TRL_USE_RICH = strtobool(os.getenv("TRL_USE_RICH", "0"))

if TRL_USE_RICH:
    init_zero_verbose()
    FORMAT = "%(message)s"

    from rich.console import Console
    from rich.logging import RichHandler

if TRL_USE_RICH:
    logging.basicConfig(
        format=FORMAT, datefmt="[%X]", handlers=[RichHandler()], level=logging.INFO
    )


@hydra.main(config_path="../../../config/pref_tuning", config_name="pythia-2.8b-marge")
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
    script_args = cfg_dict.get("dpo_script_args", {})

    # Filter out config keys not in MargeConfig (e.g. old DPO-specific fields)
    marge_cfg = cfg_dict["marge_config"]
    valid_fields = {f.name for f in dataclasses.fields(MargeConfig)}
    removed = {k: marge_cfg.pop(k) for k in list(marge_cfg) if k not in valid_fields}
    if removed:
        logging.info(f"Dropped unsupported MargeConfig fields: {list(removed)}")
    training_args = MargeConfig(**marge_cfg)
    model_config = ModelConfig(**cfg_dict["model_config"])

    if TRL_USE_RICH:
        training_args.disable_tqdm = True
        console = Console()

    ################
    # Model & Tokenizer
    ################
    set_seed(training_args.seed)
    torch_dtype = (
        model_config.dtype
        if model_config.dtype in ["auto", None]
        else getattr(torch, model_config.dtype)
    )
    quantization_config = get_quantization_config(model_config)
    model_kwargs = dict(
        revision=model_config.model_revision,
        attn_implementation=model_config.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    # use transformers logger during the training loop
    transformers_logging.enable_default_handler()
    transformers_logging.enable_propagation()
    transformers_logger = transformers_logging.get_logger("transformers")
    try:
        transformers_logger.setLevel(cfg.log_level.upper())
    except (ValueError, AttributeError):
        transformers_logger.warning(
            f"Could not set transformers logger to level {cfg.log_level}. Keeping defaults..."
        )

    peft_config = get_peft_config(model_config)
    # TODO: uncomment this later after fixing find_and_log_checkpoints to not rely on S3
    # if training_args.resume_from_checkpoint:
    #     # look for latest checkpoint!
    #     fs = s3fs.S3FileSystem()
    #     latest_local_ckpt_dir = find_and_log_checkpoints(
    #         fs,
    #         cfg.s3_output_dir,
    #         training_args.output_dir,
    #         num_gpus=torch.cuda.device_count(),
    #         num_shards=3,
    #         logger=transformers_logger,
    #     )
    # else:
    latest_local_ckpt_dir = None
    if latest_local_ckpt_dir is not None:
        model = AutoModelForCausalLM.from_pretrained(
            latest_local_ckpt_dir, trust_remote_code=True, **model_kwargs
        )
        if peft_config is None:
            ref_model = AutoModelForCausalLM.from_pretrained(
                latest_local_ckpt_dir, trust_remote_code=True, **model_kwargs
            )
        else:
            ref_model = None
        tokenizer = AutoTokenizer.from_pretrained(latest_local_ckpt_dir)
    else:
        # use the ModelClient class since it has utilities for loading models from S3
        model_client = ModelClient(
            model_config.model_name_or_path, logger=transformers_logger, **model_kwargs
        )
        model = model_client.model
        if peft_config is None:
            ref_model_client = ModelClient(
                model_config.model_name_or_path,
                logger=transformers_logger,
                **model_kwargs,
            )
            ref_model = ref_model_client.model
            ref_model.generation_config = GenerationConfig(
                **OmegaConf.to_container(cfg.generation_config)
            )
        else:
            ref_model = None
        tokenizer = model_client.tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.generation_config = GenerationConfig(
        **OmegaConf.to_container(cfg.generation_config)
    )
    if script_args.get("ignore_bias_buffers", False):
        # torch distributed hack
        model._ddp_params_and_buffers_to_ignore = [
            name for name, buffer in model.named_buffers() if buffer.dtype == torch.bool
        ]

    ################
    # Loggers and rich context managers
    ###############
    init_context = (
        nullcontext()
        if not TRL_USE_RICH
        else console.status("[bold green]Initializing the DPOTrainer...")
    )
    save_context = (
        nullcontext()
        if not TRL_USE_RICH
        else console.status(
            f"[bold green]Training completed! Saving the model to {training_args.output_dir}"
        )
    )

    ################
    # Dataset
    ################
    if not cfg.pretokenized:
        raw_df = pd.read_json(cfg.data_fp, orient="records", lines=True)
        ds = Dataset.from_pandas(raw_df).train_test_split(
            train_size=cfg.train_size, shuffle=True, seed=training_args.seed
        )

        # TODO change processing function once data format is determined
        def process(row):
            row[cfg.marge_config.input_field_name] = formatting_texts_func_edit_pairs(
                {HIGHER_SCORE_PARTICLE: [row[HIGHER_SCORE_PARTICLE]]},
                include_target=False,
                higher_score_particle_field=HIGHER_SCORE_PARTICLE,
            )[0]
            row[cfg.marge_config.target_field_name] = json.dumps(
                [int(x) for x in row[LOWER_SCORE_PARTICLE]]
            )
            row[cfg.marge_config.input_score_field_name] = row[HIGHER_SCORE]
            row[cfg.marge_config.target_score_field_name] = row[LOWER_SCORE]
            return row

        ds = ds.map(
            process,
            load_from_cache_file=False,
        ).remove_columns(
            [
                HIGHER_SCORE_PARTICLE,
                LOWER_SCORE_PARTICLE,
                HIGHER_SCORE,
                LOWER_SCORE,
            ]
        )
        train_dataset = ds["train"]
        eval_dataset = ds["test"]
        transformers_logger.info("Printing first 2 examples of formatted dataset:")
        for ex in train_dataset.select(range(2)):
            transformers_logger.info(ex)
    else:
        # Load pre-tokenized data instead
        transformers_logger.info(
            f"Loading pre-tokenized datasets from {cfg.pretokenized_train_fp} and {cfg.pretokenized_eval_fp}."
        )
        train_dataset = Dataset.load_from_disk(cfg.pretokenized_train_fp)
        transformers_logger.info(
            f"Finished loading training dataset from {cfg.pretokenized_train_fp}."
        )
        eval_dataset = Dataset.load_from_disk(cfg.pretokenized_eval_fp)
        transformers_logger.info(
            f"Finished loading eval dataset from {cfg.pretokenized_eval_fp}."
        )

    if script_args.get("sanity_check", False):
        # in sanity check, train and eval on only a very small subset of the data
        train_data_size = min(1000, len(train_dataset))
        eval_data_size = min(50, len(eval_dataset))
        train_dataset = train_dataset.select(range(train_data_size))
        eval_dataset = eval_dataset.select(range(eval_data_size))
        training_args.eval_strategy = "epoch"
        training_args.save_strategy = "no"
        training_args.load_best_model_at_end = False
        training_args.num_train_epochs = 1
        training_args.logging_steps = 1
    elif cfg.max_eval_size is not None:
        eval_dataset = eval_dataset.select(
            range(min(len(eval_dataset), cfg.max_eval_size))
        )
    if cfg.max_train_size is not None:
        train_dataset = train_dataset.select(
            range(min(len(train_dataset), cfg.max_train_size))
        )

    ################
    # Training
    ################
    with init_context:
        test_fn = load_test_fn_from_file(cfg.test_fn_fp, cfg.test_fn_type)
        callbacks = [RichProgressCallback] if TRL_USE_RICH else []
        if cfg.s3_output_dir is not None:
            s3_callback = S3Callback(cfg.s3_output_dir, logger=transformers_logger)
            callbacks.append(s3_callback)

        def metrics_fn(ds, outputs):
            return get_ehrlich_metrics_for_outputs(
                ds,
                test_fn,
                outputs,
                training_args.input_field_name,
                training_args.input_score_field_name,
            )

        def rewards_fn(batch):
            return get_ehrlich_rewards(
                batch[training_args.input_score_field_name],
                batch[training_args.target_score_field_name],
            )

        trainer = MargeTrainer(
            metrics_fn=metrics_fn,
            rewards_fn=rewards_fn,
            num_generate_batches=cfg.num_generate_batches,
            model=model,
            ref_model=ref_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            pretokenized=cfg.pretokenized,
            processing_class=tokenizer,
            callbacks=callbacks,
        )
    trainer.evaluate()
    trainer.train(resume_from_checkpoint=latest_local_ckpt_dir)
    trainer.evaluate()

    with save_context:
        trainer.save_model(training_args.output_dir)
        # Now loop through files in the directory and move to S3 or parent directory (excluding the checkpoint directories)
        if cfg.s3_output_dir is not None:
            if not cfg.s3_output_dir.endswith("/"):
                cfg.s3_output_dir += "/"
            # Use LocalOrS3Client to handle both S3 and local paths
            is_s3 = cfg.s3_output_dir.startswith("s3://")
            fs = LocalOrS3Client(init_s3=is_s3)
            for fn in os.listdir(training_args.output_dir):
                if fn.startswith(PREFIX_CHECKPOINT_DIR):
                    continue
                fp = os.path.join(training_args.output_dir, fn)
                recursive = os.path.isdir(fp)
                transformers_logger.info(f"Copying {fp} to {cfg.s3_output_dir}...")
                fs.put(fp, cfg.s3_output_dir, recursive=recursive)


if __name__ == "__main__":
    main()
