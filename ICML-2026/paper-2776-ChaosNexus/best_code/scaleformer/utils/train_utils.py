"""
Utils for training/fine-tuning
"""

import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any

import accelerate
import gluonts
import numpy as np
import torch
import torch.distributed as dist
import transformers
from scaleformer.chronos.model import (
    ChronosConfig,
    ChronosModel,
)
from scaleformer.scaleformer.scaleformer import (
    PatchTSTConfig,
    PatchTSTForPrediction,
    PatchTSTForPretraining,
)
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
)


### Utils for training
def is_main_process() -> bool:
    """
    Check if we're on the main process.
    """
    if not dist.is_torchelastic_launched():
        return True
    return int(os.environ["RANK"]) == 0


def log_on_main(msg: str, logger: logging.Logger, log_level: int = logging.INFO):
    """
    Log the given message using the given logger, if we're on the main process.
    """
    if is_main_process():
        logger.log(log_level, msg)


def get_training_job_info() -> dict:  # not currently used
    """
    Returns info about this training job.
    """
    job_info = {}

    # CUDA info
    job_info["cuda_available"] = torch.cuda.is_available()
    if torch.cuda.is_available():
        job_info["device_count"] = torch.cuda.device_count()

        job_info["device_names"] = {
            idx: torch.cuda.get_device_name(idx)
            for idx in range(torch.cuda.device_count())
        }
        job_info["mem_info"] = {
            idx: torch.cuda.mem_get_info(device=idx)
            for idx in range(torch.cuda.device_count())
        }

    # DDP info
    job_info["torchelastic_launched"] = dist.is_torchelastic_launched()

    if dist.is_torchelastic_launched():
        job_info["world_size"] = dist.get_world_size()

    # Versions
    job_info["python_version"] = sys.version.replace("\n", " ")
    job_info["torch_version"] = torch.__version__
    job_info["numpy_version"] = np.__version__
    job_info["gluonts_version"] = gluonts.__version__
    job_info["transformers_version"] = transformers.__version__
    job_info["accelerate_version"] = accelerate.__version__

    return job_info


def save_training_info(
    ckpt_path: Path, model_config: dict, train_config: dict, all_config: dict
):
    """
    Save info about this training job in a json file for documentation.
    """
    assert ckpt_path.is_dir()
    with open(ckpt_path / "training_info.json", "w") as fp:
        json.dump(
            {
                "model_config": model_config,
                "train_config": train_config,
                "all_config": all_config,
                "job_info": get_training_job_info(),
            },
            fp,
            indent=4,
        )


def get_next_path(
    base_fname: str,
    base_dir: Path,
    file_type: str = "yaml",
    separator: str = "-",
    overwrite: bool = False,
):
    """
    Gets the next available path in a directory. For example, if `base_fname="results"`
    and `base_dir` has files ["results-0.yaml", "results-1.yaml"], this function returns
    "results-2.yaml".
    """
    if file_type == "":
        # Directory
        items = filter(
            lambda x: x.is_dir() and re.match(f"^{base_fname}{separator}\\d+$", x.stem),
            base_dir.glob("*"),
        )
    else:
        # File
        items = filter(
            lambda x: re.match(f"^{base_fname}{separator}\\d+$", x.stem),
            base_dir.glob(f"*.{file_type}"),
        )
    run_nums = list(
        map(lambda x: int(x.stem.replace(base_fname + separator, "")), items)
    ) + [-1]

    next_num = max(run_nums) + (0 if overwrite else 1)
    fname = f"{base_fname}{separator}{next_num}" + (
        f".{file_type}" if file_type != "" else ""
    )

    return base_dir / fname


def load_chronos_model(
    model_id="google/t5-efficient-tiny",
    model_type="seq2seq",
    vocab_size=4096,
    random_init=False,
    tie_embeddings=False,
    pad_token_id=0,
    eos_token_id=1,
    chronos_config: ChronosConfig | None = None,
    logger: logging.Logger | None = None,
) -> ChronosModel:
    """
    Load the specified HuggingFace model, adjusting the vocabulary
    size, special token IDs, and initialization options.

    This allows to set a model up for training on a new vocabulary
    of tokens.
    """
    assert model_type in ["seq2seq", "causal"]
    AutoModelClass = (
        AutoModelForSeq2SeqLM if model_type == "seq2seq" else AutoModelForCausalLM
    )
    config = AutoConfig.from_pretrained(model_id)
    if chronos_config is not None:
        config.chronos_config = chronos_config.__dict__  # type: ignore

    if random_init:
        if logger is not None:
            log_on_main("Using random initialization", logger)
        # The default initializer_factor (1.0) in transformers is too large
        config.initializer_factor = 0.05
        config.tie_word_embeddings = tie_embeddings
        model = AutoModelClass.from_config(config)
    else:
        if logger is not None:
            log_on_main(f"Using pretrained initialization from {model_id}", logger)
        model = AutoModelClass.from_pretrained(model_id, config=config)

    model.resize_token_embeddings(vocab_size)  # type: ignore
    model.config.pad_token_id = model.generation_config.pad_token_id = pad_token_id  # type: ignore
    model.config.eos_token_id = model.generation_config.eos_token_id = eos_token_id  # type: ignore

    return model  # type: ignore


def load_patchtst_model(
    mode: str,
    model_config: dict[str, Any],
    pretrained_encoder_path: str | None = None,
    pretained_checkpoint: str | None = None,
) -> PatchTSTForPretraining | PatchTSTForPrediction:
    """
    Load a PatchTST model in either pretraining or prediction mode.

    Args:
        mode: Either "pretrain" or "predict" to specify model type
        model_config: Dictionary containing model configuration parameters
        pretrained_encoder_path: Optional path to pretrained encoder weights for prediction mode

    Returns:
        PatchTSTForPretraining or PatchTSTForPrediction model instance
    """
    config = PatchTSTConfig(**model_config)
    if mode == "pretrain":
        model = PatchTSTForPretraining(config)
    elif mode == "predict":
        model = PatchTSTForPrediction(config)
    else:
        raise ValueError(f"Invalid mode: {mode}")

    if pretrained_encoder_path is not None and mode == "predict":
        pretrained_model = PatchTSTForPretraining.from_pretrained(
            pretrained_encoder_path
        )
        # replace the current encoder with the pretrained encoder
        if hasattr(pretrained_model, "model"):
            pretained_trunk = getattr(pretrained_model, "model")
            assert hasattr(pretained_trunk, "encoder"), "PatchTST must have an encoder"
            model.model.encoder = pretained_trunk.encoder
        else:
            raise Exception("No model found in pretrained model")
    elif pretained_checkpoint is not None and mode == "predict":
        # load a pretrained prediction model for SFT
        pretrained_model = PatchTSTForPrediction.from_pretrained(
            pretained_checkpoint,
            config=config,
        )
        return pretrained_model  # type: ignore

    return model


def has_enough_observations(
    entry: dict, min_length: int = 0, max_missing_prop: float = 1.0
) -> bool:
    """
    Check if the given entry has enough observations in the ``"target"`` attribute.

    Parameters
    ----------
    entry
        The data entry (dictionary) to be tested.
    min_length
        The minimum length the ``"target"`` attribute must have.
    max_missing_prop
        The maximum proportion of missing data allowed in the ``"target"``
        attribute.
    """
    if (
        entry["target"].shape[-1] >= min_length
        and np.isnan(entry["target"]).mean() <= max_missing_prop
    ):
        return True
    return False


def ensure_contiguous(model):
    """
    Ensure that all parameters in the model are contiguous.
    If any parameter is not contiguous, make it contiguous.

    :param model: The model whose parameters need to be checked.
    """
    for name, param in model.named_parameters():
        if not param.is_contiguous():
            print(f"Parameter {name} is not contiguous. Making it contiguous.")
            param.data = param.data.contiguous()
