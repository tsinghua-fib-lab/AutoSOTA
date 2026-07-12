"""Entry point for the three cramming regimes (full / progressive / low-dim).

The trainer is selected from the parsed CLI flags:

* default                 -> :class:`FullCrammingTrainer`
* ``--progressive_train`` -> :class:`ProgressiveCrammingTrainer`
* ``--low_dim_train``     -> :class:`LowDimTrainer`

Use it from the shell via ``python scripts/run_cramming.py ...`` (or the installed
``progressive-cramming`` console script), or programmatically via
:func:`run_training` (see ``examples/quickstart.py``).
"""

from __future__ import annotations

import os

import transformers
from accelerate import PartialState
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling

from progressive_cramming.data import load_or_create_tokenized_dataset
from progressive_cramming.train import (
    FullCrammingTrainer,
    LowDimTrainer,
    ProgressiveCrammingTrainer,
)
from progressive_cramming.train.arguments import MyTrainingArguments
from progressive_cramming.utils.launch import resolve_torch_dtype, set_launch_seed

# Tokenized-dataset cache. Override with PC_DATA_CACHE; gitignored by default.
DATA_CACHE_DIR = os.path.join(os.environ.get("PC_DATA_CACHE", ".cache"), "tokenized_datasets")


def select_trainer_cls(args: MyTrainingArguments):
    """Pick the trainer class from the CLI flags (progressive > low-dim > full)."""
    if args.progressive_train:
        return ProgressiveCrammingTrainer
    if args.low_dim_train:
        return LowDimTrainer
    return FullCrammingTrainer


def load_model_and_tokenizer(args: MyTrainingArguments):
    """Load a frozen causal-LM + its tokenizer with a portable attention backend."""
    torch_dtype = resolve_torch_dtype(args.dtype)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_checkpoint,
        torch_dtype=torch_dtype,
        attn_implementation=args.attn_implementation,
    )
    # Cramming optimizes only the compression embedding; the base model stays frozen.
    for parameter in model.parameters():
        parameter.requires_grad = False

    tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return model, tokenizer


def run_training(args: MyTrainingArguments, train_dataset=None) -> str | None:
    """Load model + data, run the selected trainer, return the saved-artifact path.

    ``args.output_dir`` must be set (the trainer writes the per-sample dataset and
    embeddings there). Returns the artifact subdirectory path (``compressed_prefixes``
    for full/low-dim, ``progressive_prefixes`` for progressive) or ``None``.

    If ``train_dataset`` is provided, it is used as-is and ``args.dataset_name`` /
    ``limit_dataset_items`` / ``offset_dataset_items`` are ignored. This is how callers
    cram arbitrary in-memory text (e.g. the demo gallery builder); the default path
    tokenises from a HuggingFace dataset name on the Hub.
    """
    if not args.output_dir:
        raise ValueError("args.output_dir must be set before calling run_training().")
    os.makedirs(args.output_dir, exist_ok=True)
    if not args.logging_dir:
        args.logging_dir = args.output_dir

    set_launch_seed(args.random_seed)
    model, tokenizer = load_model_and_tokenizer(args)

    if train_dataset is None:
        os.makedirs(DATA_CACHE_DIR, exist_ok=True)
        # Under multi-GPU (`accelerate launch`) only the main process tokenizes + writes the
        # cache; the others wait and load it. No-op for a single process.
        with PartialState().main_process_first():
            train_dataset = load_or_create_tokenized_dataset(
                cache_dir=DATA_CACHE_DIR,
                dataset_name=args.dataset_name,
                split="test",
                tokenizer=tokenizer,
                max_sequence_length=args.max_sequence_length,
                model_checkpoint=args.model_checkpoint,
                no_bos_token=args.no_bos_token,
                limit_dataset_items=getattr(args, "limit_dataset_items", None),
                offset_dataset_items=getattr(args, "offset_dataset_items", None),
                cache_prefix="dataset",
            )
    print(f"train_dataset: {len(train_dataset)} samples")

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer_cls = select_trainer_cls(args)
    print(f"Trainer: {trainer_cls.__name__}")
    trainer = trainer_cls(
        model,
        processing_class=tokenizer,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=None,
        data_collator=data_collator,
    )
    artifact_path = trainer.train()
    print(f"Saved compression artifacts to: {artifact_path}")
    return artifact_path


def _default_output_dir(args: MyTrainingArguments) -> str:
    """Build a readable default ``runs/<mode>_<model>_seq<L>`` output directory."""
    if args.progressive_train:
        mode = "progressive"
    elif args.low_dim_train:
        mode = f"lowdim{args.low_dim_size}"
    else:
        mode = "full"
    model_short = args.model_checkpoint.rstrip("/").split("/")[-1]
    return os.path.join("runs", f"{mode}_{model_short}_seq{args.max_sequence_length}")


def main() -> None:
    """CLI: parse :class:`MyTrainingArguments`, resolve an output dir, train."""
    transformers.logging.set_verbosity_info()
    parser = transformers.HfArgumentParser(MyTrainingArguments)
    (args,) = parser.parse_args_into_dataclasses()

    if not args.output_dir or args.output_dir == "trainer_output":
        # HF's TrainingArguments forces a non-empty output_dir; treat its placeholder
        # default as "unset" and build a readable one from the run configuration.
        args.output_dir = _default_output_dir(args)
    args.logging_dir = args.output_dir
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"output_dir: {args.output_dir}")

    run_training(args)
    print(f"Done. Results in: {args.output_dir}")


if __name__ == "__main__":
    main()
