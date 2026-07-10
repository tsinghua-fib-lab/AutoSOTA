#!/usr/bin/env python3
"""
Offline DPO on multi-turn preference pairs synthesised by the user simulator
(``discoverllm.data.build_dataset``).

Two common training schedules:

1. **DPO from base** — point ``--model_name`` at the original HF checkpoint
   (e.g. ``meta-llama/Llama-3.1-8B-Instruct``). This is the
   ``offline_dpo_from_base`` variant in our paper.
2. **DPO from SFT** — point ``--model_name`` at an earlier SFT LoRA output
   (e.g. ``outputs/sft/llama-3.1-8b_my_run``). This is ``offline_dpo_from_sft``.

Run as a module:

    python -m discoverllm.training.trainers.offline_dpo \\
        --dataset_repo path/to/processed_dataset \\
        --model_name <base or SFT-output path> \\
        --output_dir outputs/offline_dpo/my_run \\
        --use_lora --system_prompt_type ours

See ``scripts/train/offline_dpo.sh`` for a full multi-GPU launch example.
"""

from __future__ import annotations

import argparse
import json
import os

from peft import PeftModel
from trl import DPOConfig, DPOTrainer

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
    p = make_base_argparser("multiturn DPO trainer")
    # DPO-specific extras.
    p.add_argument("--max_length", type=int, default=2048,
                   help="Max combined prompt+completion length passed to DPOTrainer.")
    p.add_argument("--minimum_gap", type=float, default=0.05,
                   help="Drop pairs whose chosen-vs-rejected score gap is below this.")
    return apply_config_file_overrides(p.parse_args())


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    init_distributed()

    ds = MultiturnDataset(
        args.dataset_repo, system_prompt_type=args.system_prompt_type,
    ).to_dpo_dataset(eval_ratio=args.eval_ratio, minimum_gap=args.minimum_gap)

    bnb_cfg = make_bnb_config(args)
    lora_cfg = make_lora_config(args)
    model, tok = load_model_and_tokenizer(
        args.model_name, bnb_cfg=bnb_cfg, device=args.device, is_eval=False,
    )

    # If model already came back as a PeftModel (resuming from an SFT
    # adapter), don't pass peft_config to DPOTrainer — TRL would refuse.
    peft_config_for_trainer = None if isinstance(model, PeftModel) else lora_cfg

    train_cfg = DPOConfig(
        output_dir=args.output_dir,
        beta=0.1,
        loss_type="sigmoid",
        max_grad_norm=1.0,
        optim="adamw_torch",
        report_to="wandb" if args.wandb_project else "none",
        do_eval=True,
        eval_steps=args.eval_steps,
        save_strategy="epoch",
        eval_strategy="steps",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        lr_scheduler_type="cosine",
        metric_for_best_model="eval_loss",
        warmup_ratio=args.warmup_ratio,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        num_train_epochs=args.num_train_epochs,
        save_total_limit=args.save_total_limit,
        max_length=args.max_length,
        max_prompt_length=None,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        run_name=args.output_dir,
        deepspeed=deepspeed_zero2_config(args),
        **precision_kwargs(),
    )
    init_wandb_if_configured(args, train_cfg.to_dict())

    # Apply chat template to prompt/chosen/rejected, then verify reconstruction.
    def process(row):
        reference = tok.apply_chat_template(
            row["prompt"] + [{"role": "assistant", "content": row["chosen"].strip()}],
            tokenize=False,
        ).strip()
        row["prompt"] = tok.apply_chat_template(
            row["prompt"], tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        row["chosen"] = row["chosen"].strip() + tok.eos_token
        row["rejected"] = row["rejected"].strip() + tok.eos_token
        if row["prompt"] + row["chosen"] != reference:
            raise ValueError(
                "DPO prompt + chosen does not match the chat-templated reference.\n"
                f"[PROMPT] {json.dumps(row['prompt'])}\n"
                f"[CHOSEN] {json.dumps(row['chosen'])}\n"
                f"[REFERENCE] {json.dumps(reference)}"
            )
        return row

    ds["train"] = ds["train"].map(process, load_from_cache_file=False)
    ds["eval"] = ds["eval"].map(process, load_from_cache_file=False)

    trainer = DPOTrainer(
        model=model,
        train_dataset=ds["train"],
        eval_dataset=ds["eval"],
        processing_class=tok,
        peft_config=peft_config_for_trainer,
        args=train_cfg,
    )
    trainer.train(resume_from_checkpoint=args.resume_ckpt_dir)
    trainer.save_model(args.output_dir)
    tok.save_pretrained(args.output_dir)

    if args.push_to_hub and args.hf_org:
        repo = f"offline_dpo-{args.dataset_repo.replace('/', '_')}"
        trainer.model.push_to_hub(f"{args.hf_org}/{repo}", private=True)
        tok.push_to_hub(f"{args.hf_org}/{repo}", private=True)

    finish_wandb_if_configured(args)


if __name__ == "__main__":
    main()
