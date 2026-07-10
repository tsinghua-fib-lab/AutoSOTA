#!/usr/bin/env python3
"""
Supervised fine-tuning of a causal LM (optionally with LoRA / 4-bit quantization)
on multi-turn chat data produced by ``discoverllm.data.build_dataset``.

Run as a module so distributed launchers can find it:

    python -m discoverllm.training.trainers.sft \\
        --dataset_repo path/to/processed_dataset \\
        --output_dir outputs/sft/my_run \\
        --model_name meta-llama/Llama-3.1-8B-Instruct \\
        --use_lora --system_prompt_type ours

See ``scripts/train/sft.sh`` for a complete example with multi-GPU launch flags.
"""

from __future__ import annotations

import argparse
import os

from trl import SFTConfig, SFTTrainer

from discoverllm.training._common import (
    apply_config_file_overrides,
    deepspeed_zero2_config,
    finish_wandb_if_configured,
    init_distributed,
    init_wandb_if_configured,
    load_model_and_tokenizer,
    make_base_argparser,
    make_bnb_config,
    make_lora_config,
    precision_kwargs,
)
from discoverllm.training.datasets.multiturn import MultiturnDataset


def parse_args() -> argparse.Namespace:
    p = make_base_argparser("multiturn SFT trainer")
    # SFT-specific: filter rows by a metric threshold before training.
    p.add_argument("--lower_bound_metric", type=str, default=None,
                   help="Drop rows whose metric (dot-path) is below --lower_bound.")
    p.add_argument("--lower_bound", type=float, default=0.0)
    return apply_config_file_overrides(p.parse_args())


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    init_distributed()

    ds = MultiturnDataset(
        args.dataset_repo,
        system_prompt_type=args.system_prompt_type,
    ).to_sft_dataset(
        eval_ratio=args.eval_ratio,
        lower_bound_metric=args.lower_bound_metric,
        lower_bound=args.lower_bound,
    )

    bnb_cfg = make_bnb_config(args)
    lora_cfg = make_lora_config(args)
    model, tok = load_model_and_tokenizer(
        args.model_name, bnb_cfg=bnb_cfg, device=args.device, is_eval=False,
    )

    train_cfg = SFTConfig(
        output_dir=args.output_dir,
        logging_steps=args.logging_steps,
        warmup_ratio=args.warmup_ratio,
        optim="adamw_torch",
        report_to="wandb" if args.wandb_project else "none",
        do_eval=True,
        eval_steps=args.eval_steps,
        save_strategy="epoch",
        eval_strategy="steps",
        max_length=args.max_seq_length,
        group_by_length=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        lr_scheduler_type="cosine",
        metric_for_best_model="eval_loss",
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        save_total_limit=args.save_total_limit,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        run_name=args.output_dir,
        deepspeed=deepspeed_zero2_config(args),
        **precision_kwargs(),
    )
    init_wandb_if_configured(args, train_cfg.to_dict())

    trainer = SFTTrainer(
        model=model,
        train_dataset=ds["train"],
        eval_dataset=ds["eval"],
        processing_class=tok,
        peft_config=lora_cfg,
        args=train_cfg,
    )
    trainer.train(resume_from_checkpoint=args.resume_ckpt_dir)
    trainer.save_model(args.output_dir)
    tok.save_pretrained(args.output_dir)

    if args.push_to_hub and args.hf_org:
        repo = f"sft-{args.dataset_repo.replace('/', '_')}"
        trainer.model.push_to_hub(f"{args.hf_org}/{repo}", private=True)
        tok.push_to_hub(f"{args.hf_org}/{repo}", private=True)

    finish_wandb_if_configured(args)


if __name__ == "__main__":
    main()
