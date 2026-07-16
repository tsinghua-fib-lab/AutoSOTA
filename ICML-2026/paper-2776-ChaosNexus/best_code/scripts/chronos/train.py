"""
Training/fine-tuning script, adapted from chronos-forecasting
"""

import logging
from functools import partial
from pathlib import Path

import hydra
import torch
import transformers
from gluonts.dataset.common import FileDataset
from gluonts.itertools import Filter
from gluonts.transform import LastValueImputation
from omegaconf import OmegaConf
from scaleformer.augmentations import (
    RandomAffineTransform,
    RandomConvexCombinationTransform,
    RandomFourierSeries,
    RandomPhaseSurrogate,
    RandomTakensEmbedding,
    StandardizeTransform,
)
from scaleformer.chronos.dataset import ChronosDataset
from scaleformer.chronos.model import ChronosConfig
from scaleformer.utils import (
    ensure_contiguous,
    get_next_path,
    has_enough_observations,
    is_main_process,
    load_chronos_model,
    log_on_main,
    save_training_info,
)
from transformers import Trainer, TrainingArguments

import wandb


@hydra.main(config_path="../../config", config_name="config", version_base=None)
def main(cfg):
    # set up wandb project and logging if enabled
    if cfg.wandb.log and is_main_process():
        run = wandb.init(
            project=cfg.wandb.project_name,
            entity=cfg.wandb.entity,
            name=cfg.run_name,
            config=dict(cfg),
            sync_tensorboard=False,  # auto-upload tensorboard metrics
            group=cfg.wandb.group_name,
            resume=cfg.wandb.resume,
            tags=cfg.wandb.tags,
        )
        log_on_main(f"Wandb initialized: {run.id}", logger)

    # check model type is valid
    assert cfg.chronos.model_type in ["seq2seq", "causal"]

    # set floating point precision
    use_tf32 = cfg.train.tf32
    if use_tf32 and not (
        torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
    ):
        # TF32 floating point format is available only on NVIDIA GPUs
        # with compute capability 8 and above. See link for details.
        # https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capability-8-x
        log_on_main(
            "TF32 format is only available on devices with compute capability >= 8. "
            "Setting tf32 to False.",
            logger,
        )
        use_tf32 = False

    # set random seed
    log_on_main(f"Using SEED: {cfg.train.seed}", logger)
    transformers.set_seed(seed=cfg.train.seed)

    # get train data paths
    train_data_paths = []
    if cfg.train_data_dirs is not None:
        for train_data_dirs in cfg.train_data_dirs:
            train_data_paths.extend(
                list(
                    filter(
                        lambda file: file.is_file(), Path(train_data_dirs).rglob("*")
                    )
                )
            )

    # create a new output directory to save results
    output_dir = get_next_path(
        cfg.run_name if cfg.run_name else "run",
        base_dir=Path(cfg.train.output_dir),
        file_type="",
        overwrite=cfg.train.resume_from_checkpoint is not None,
    )

    log_on_main(f"Logging dir: {output_dir}", logger)
    log_on_main(f"Loading and filtering {len(train_data_paths)} datasets ", logger)

    # load datasets and apply loading filters on the fly
    train_datasets = [
        Filter(
            partial(
                has_enough_observations,
                min_length=cfg.min_past + cfg.chronos.prediction_length,
                max_missing_prop=cfg.max_missing_prop,
            ),
            FileDataset(path=Path(data_path), one_dim_target=False, freq="h"),  # type: ignore
        )
        for data_path in train_data_paths
    ]

    # set probabilities (how we weight draws from each data file)
    if isinstance(cfg.probability, float):
        probability = cfg.probability
    elif cfg.probability is None:
        probability = [1.0 / len(train_datasets)] * len(train_datasets)
    assert isinstance(probability, list)
    assert len(train_datasets) == len(probability)

    # adapt number of workers to the number of datasets if there are more workers than datasets
    dataloader_num_workers = cfg.train.dataloader_num_workers
    if dataloader_num_workers > len(train_datasets):
        log_on_main(
            f"Setting the number of data loader workers to {len(train_datasets)}, "
            f"instead of {dataloader_num_workers}.",
            logger,
        )
        dataloader_num_workers = len(train_datasets)

    chronos_config = ChronosConfig(
        tokenizer_class=cfg.chronos.tokenizer_class,
        tokenizer_kwargs=dict(cfg.chronos.tokenizer_kwargs),
        n_tokens=cfg.chronos.n_tokens,
        n_special_tokens=cfg.chronos.n_special_tokens,
        pad_token_id=cfg.chronos.pad_token_id,
        eos_token_id=cfg.chronos.eos_token_id,
        use_eos_token=cfg.chronos.use_eos_token,
        model_type=cfg.chronos.model_type,
        context_length=cfg.chronos.context_length,
        prediction_length=cfg.chronos.prediction_length,
        num_samples=cfg.chronos.num_samples,
        temperature=cfg.chronos.temperature,
        top_k=cfg.chronos.top_k,
        top_p=cfg.chronos.top_p,
    )
    tokenizer = chronos_config.create_tokenizer()

    log_on_main(f"Initializing model: {cfg.chronos.model_id}", logger)
    model = load_chronos_model(
        model_id=cfg.chronos.model_id,
        model_type=cfg.chronos.model_type,
        vocab_size=cfg.chronos.n_tokens,
        random_init=cfg.chronos.random_init,
        tie_embeddings=cfg.chronos.tie_embeddings,
        pad_token_id=cfg.chronos.pad_token_id,
        eos_token_id=cfg.chronos.eos_token_id,
        chronos_config=chronos_config,
    )

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log_on_main(f"Total trainable parameters: {trainable_params:,}", logger)

    # Note: these augmentations are applied to the multivariate target traj
    augmentations = [
        RandomTakensEmbedding(
            lag_range=cfg.augmentations.lag_range,
            random_seed=cfg.train.seed,
        ),
        RandomConvexCombinationTransform(
            alpha=1.0,
            random_seed=cfg.train.seed,
            dim_range=cfg.augmentations.dim_range,
        ),
        RandomAffineTransform(
            dim_range=cfg.augmentations.dim_range,
            scale=1.0,
            random_seed=cfg.train.seed,
        ),
        RandomPhaseSurrogate(
            cutoff=cfg.augmentations.phase_surrogate_cutoff,
            random_seed=cfg.train.seed,
        ),
        RandomFourierSeries(
            max_wavenumber=cfg.augmentations.max_wavenumber,
            max_amp=cfg.augmentations.max_amp,
            mode_range=cfg.augmentations.mode_range,
            random_seed=cfg.train.seed,
        ),
    ]
    if cfg.augmentations.probabilities is None:
        cfg.augmentations.probabilities = [1.0 / len(augmentations)] * len(
            augmentations
        )
    else:  # ensure probabilities sum to 1
        cfg.augmentations.probabilities = [
            prob / sum(cfg.augmentations.probabilities)
            for prob in cfg.augmentations.probabilities
        ]

    log_on_main(
        f"Using augmentations: {[aug for aug, prob in zip(augmentations, cfg.augmentations.probabilities) if prob > 0.0]}",
        logger,
    )

    transforms: list = [StandardizeTransform()]

    shuffled_train_dataset = ChronosDataset(
        datasets=train_datasets,
        probabilities=probability,
        tokenizer=tokenizer,
        context_length=cfg.chronos.context_length,
        prediction_length=cfg.chronos.prediction_length,
        min_past=cfg.min_past,
        model_type=cfg.chronos.model_type,
        imputation_method=LastValueImputation()
        if cfg.chronos.model_type == "causal"
        else None,
        mode="train",
        augmentations=augmentations,
        augmentation_rate=cfg.augmentations.augmentation_rate,
        augmentation_probabilities=cfg.augmentations.probabilities,
        transforms=transforms,
    ).shuffle(shuffle_buffer_length=cfg.shuffle_buffer_length)

    training_args = TrainingArguments(
        run_name=cfg.run_name,
        output_dir=str(output_dir),
        per_device_train_batch_size=cfg.train.per_device_train_batch_size,
        learning_rate=cfg.train.learning_rate,
        lr_scheduler_type=cfg.train.lr_scheduler_type,
        warmup_ratio=cfg.train.warmup_ratio,
        max_grad_norm=cfg.train.max_grad_norm,
        weight_decay=cfg.train.weight_decay,
        optim=cfg.train.optim,
        log_on_each_node=False,
        logging_dir=str(output_dir / "logs")
        if not (cfg.wandb.log and is_main_process())
        else f"wandb/{run.name}_{run.id}/logs",
        logging_strategy="steps",
        logging_steps=cfg.train.log_steps,
        save_strategy="steps",
        save_steps=cfg.train.save_steps,
        report_to=["wandb"] if cfg.wandb.log else ["tensorboard"],
        max_steps=cfg.train.max_steps,
        gradient_accumulation_steps=cfg.train.gradient_accumulation_steps,
        dataloader_num_workers=dataloader_num_workers,
        dataloader_prefetch_factor=cfg.train.dataloader_prefetch_factor,
        tf32=use_tf32,  # remove this if not using Ampere GPUs (e.g., A100)
        torch_compile=cfg.train.torch_compile,
        ddp_find_unused_parameters=cfg.train.ddp_find_unused_parameters,
        remove_unused_columns=cfg.train.remove_unused_columns,
        ddp_backend=cfg.train.ddp_backend,
        seed=cfg.train.seed,
        resume_from_checkpoint=cfg.train.resume_from_checkpoint,
    )

    # check if model weights are contiguous in memory; if not, make them contiguous tensors.
    # This speeds up training and allows checkpoint saving by transformers Trainer
    ensure_contiguous(model)

    trainer = Trainer(
        model=model, args=training_args, train_dataset=shuffled_train_dataset
    )

    log_on_main("Training", logger)
    trainer.train(
        resume_from_checkpoint=cfg.train.resume_from_checkpoint
    )  # Transformers trainer will save model checkpoints automatically

    # save final model checkpoint and training info locally
    if is_main_process():
        model.save_pretrained(output_dir / "checkpoint-final")
        save_training_info(
            output_dir / "checkpoint-final",
            model_config=vars(chronos_config),  # TODO: add model_id to this
            train_config=OmegaConf.to_container(cfg.train, resolve=True),  # type: ignore
            all_config=OmegaConf.to_container(cfg, resolve=True),  # type: ignore
        )

    # terminate wandb run after training
    if cfg.wandb.log:
        wandb.finish(exit_code=0)


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    main()
