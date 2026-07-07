"""
Convert pytorch_model.bin checkpoints to safetensors format.

Downloads each model, re-saves with safe_serialization=True,
and writes the safetensors files back into the HF cache snapshot
directory so that the streaming pipeline can find them.

Run: python scripts/convert_to_safetensors.py
"""

from run import setup_env

setup_env()

import logging
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODELS = [
    "jeffwan/llama-7b-hf",
    "jeffwan/llama-13b-hf",
    "jeffwan/llama-30b-hf",
]


def convert_model(model_id: str) -> None:
    """Download a model and save safetensors into its HF cache snapshot."""
    logger.info(f"Downloading {model_id} (bin files)...")
    snapshot_dir = Path(snapshot_download(model_id, allow_patterns=["*.bin", "*.json"]))
    logger.info(f"Snapshot dir: {snapshot_dir}")

    # Check if already converted
    if list(snapshot_dir.glob("*.safetensors")):
        logger.info(f"{model_id} already has safetensors, skipping")
        return

    logger.info(f"Loading {model_id} into memory...")
    model = AutoModelForCausalLM.from_pretrained(snapshot_dir, torch_dtype=torch.float16)

    logger.info(f"Saving as safetensors to {snapshot_dir}...")
    model.save_pretrained(snapshot_dir, safe_serialization=True)

    # Clean up bin files to save disk space
    for bin_file in snapshot_dir.glob("pytorch_model*.bin"):
        logger.info(f"Removing {bin_file.name}")
        bin_file.unlink()

    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    logger.info(f"Done: {model_id}")


def main() -> None:
    for model_id in MODELS:
        convert_model(model_id)
    logger.info("All models converted.")


if __name__ == "__main__":
    main()
