#!/usr/bin/env python3
"""
GRPO trainer driven by the user-simulator reward.

On every step the trainer rolls out :math:`G` completions per prompt via
TRL's native vLLM colocation, scores each one with
:class:`discoverllm.training.reward.DesignLLMRewardComputer`, and applies the
group-relative advantage update from the GRPO paper (Shao et al., 2024).

Compared to a vanilla TRL GRPO setup, this script replaces the scalar reward
model with our intent-aware multi-turn judge.

Run as a module:

    python -m discoverllm.training.trainers.grpo \\
        --dataset_repo path/to/processed_dataset \\
        --model_name path/to/sft_or_dpo_checkpoint \\
        --output_dir outputs/grpo/my_run \\
        --num_generations 4 \\
        --use_lora --system_prompt_type ours

See ``scripts/train/grpo.sh`` for a full launch example.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from datetime import datetime

from dotenv import load_dotenv
from peft import PeftModel
from trl import GRPOConfig, GRPOTrainer

from discoverllm.training._common import (
    apply_config_file_overrides,
    assert_has_trainable_params,
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
from discoverllm.training.reward import DesignLLMRewardComputer

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = make_base_argparser("multiturn GRPO trainer")
    # GRPO-specific extras.
    p.add_argument("--assistant_generation_kwargs", type=json.loads, default="{}",
                   help='JSON dict; e.g. {"model": "outputs/sft/foo", "temperature": 0.6}')
    p.add_argument("--max_new_turns", type=int, default=0,
                   help="Future-turn lookahead used by DesignLLMRewardComputer.")
    p.add_argument("--max_completion_length", type=int, default=1024)
    p.add_argument("--max_metric_workers", type=int, default=4)
    p.add_argument("--user_simulator_config", type=json.loads, default="{}",
                   help='JSON dict for the user simulator the reward computer drives; '
                        'e.g. {"model_name": "claude-haiku-4-5", "temperature": 0.3}. '
                        'Defaults to claude-haiku-4-5.')
    p.add_argument("--gpu_memory_utilization", type=float, default=0.3)
    p.add_argument("--num_generations", type=int, default=8,
                   help="Number of generations per prompt (G in GRPO). Default 8.")
    # GRPO loss knobs
    p.add_argument("--beta", type=float, default=0.0,
                   help="KL coefficient. 0.0 (default) skips the reference model.")
    p.add_argument("--epsilon", type=float, default=0.28, help="Clipping lower bound.")
    p.add_argument("--epsilon_high", type=float, default=None,
                   help="Clipping upper bound; defaults to --epsilon.")
    p.add_argument("--num_iterations", type=int, default=1,
                   help="Number of optimizer iterations per batch (mu).")
    p.add_argument("--scale_rewards", type=str, default="group",
                   choices=["group", "batch", "none"])
    p.add_argument("--loss_type", type=str, default="dapo",
                   choices=["grpo", "dapo", "dr_grpo", "bnpo", "cispo", "sapo"])
    p.add_argument("--mask_truncated_completions", action="store_true", default=False)
    # Logging
    p.add_argument("--log_completions", action="store_true", default=False)
    p.add_argument("--log_completions_file", type=str, default=None)
    return apply_config_file_overrides(p.parse_args())


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    init_distributed()

    ds = MultiturnDataset(
        args.dataset_repo, system_prompt_type=args.system_prompt_type,
    ).to_inputs_dataset(eval_ratio=0.0)

    bnb_cfg = make_bnb_config(args)
    lora_cfg = make_lora_config(args)
    model, tok = load_model_and_tokenizer(
        args.model_name, bnb_cfg=bnb_cfg, device=args.device, is_eval=False,
    )
    # GRPO needs left-padding for generation.
    tok.padding_side = "left"
    assert_has_trainable_params(
        model, hint="Right after loading. Make sure your policy adapter is trainable.",
    )
    peft_config_for_trainer = None if isinstance(model, PeftModel) else lora_cfg

    gen_kwargs = args.assistant_generation_kwargs
    scale_rewards_value = False if args.scale_rewards == "none" else args.scale_rewards

    train_cfg = GRPOConfig(
        output_dir=args.output_dir,
        # GRPO-specific
        beta=args.beta,
        epsilon=args.epsilon,
        epsilon_high=args.epsilon_high,
        num_generations=args.num_generations,
        num_iterations=args.num_iterations,
        scale_rewards=scale_rewards_value,
        loss_type=args.loss_type,
        mask_truncated_completions=args.mask_truncated_completions,
        max_completion_length=args.max_completion_length,
        # General training
        max_grad_norm=1.0,
        optim="adamw_torch",
        report_to="wandb" if args.wandb_project else "none",
        do_eval=False,
        eval_strategy="no",
        save_strategy="steps",
        save_steps=1,
        save_total_limit=args.save_total_limit,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        run_name=args.output_dir,
        deepspeed=deepspeed_zero2_config(args),
        # vLLM colocation for on-policy generation.
        use_vllm=True,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=args.gpu_memory_utilization,
        temperature=gen_kwargs.get("temperature", 0.7),
        log_completions=args.log_completions,
        chat_template_kwargs={"enable_thinking": False},
        remove_unused_columns=False,
        **precision_kwargs(),
    )
    init_wandb_if_configured(args, train_cfg.to_dict())

    # ------------------------------------------------------------------ #
    # Map each templated string prompt back to its (criteria_history,    #
    # original message list) tuple so the reward fn can score it.        #
    # ------------------------------------------------------------------ #
    def compute_hash(text: str) -> str:
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    str_prompt_to_data: dict = {}

    def index_prompt(row):
        templated = tok.apply_chat_template(
            row["prompt"], tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        str_prompt_to_data.setdefault(
            compute_hash(templated),
            {k: row[k] for k in ("criteria_history", "prompt")},
        )
        # Keep prompt as a message list — TRL/vLLM template it themselves.
        return row

    ds["train"] = ds["train"].map(index_prompt, load_from_cache_file=False)

    reward_computer = DesignLLMRewardComputer(
        reward_assistant_config=args.assistant_generation_kwargs,
        window_size=args.max_new_turns,
        max_workers=args.max_metric_workers,
        verbose=False,
        user_simulator_config=args.user_simulator_config,
    )

    reward_fn = _make_reward_fn(reward_computer, str_prompt_to_data, compute_hash, tok, args)

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_fn,
        train_dataset=ds["train"],
        processing_class=tok,
        peft_config=peft_config_for_trainer,
        args=train_cfg,
    )
    trainer.train(resume_from_checkpoint=args.resume_ckpt_dir)
    trainer.save_model(args.output_dir)
    tok.save_pretrained(args.output_dir)

    if args.push_to_hub and args.hf_org:
        repo = f"grpo-{args.dataset_repo.replace('/', '_')}"
        trainer.model.push_to_hub(f"{args.hf_org}/{repo}", private=True)
        tok.push_to_hub(f"{args.hf_org}/{repo}", private=True)

    finish_wandb_if_configured(args)


# --------------------------------------------------------------------------- #
# Reward function builder (closure captures the per-run state)                #
# --------------------------------------------------------------------------- #
def _extract_completion_text(completion) -> str:
    """TRL hands completions either as a str or as a list of message dicts."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion and isinstance(completion[0], dict):
        return completion[0].get("content", "")
    if isinstance(completion, dict):
        return completion.get("content", str(completion))
    return str(completion)


def _make_reward_fn(reward_computer, prompt_to_data, hash_fn, tok, args):
    """Closure-builder for the GRPO reward function."""

    def _prompt_to_string(prompt) -> str:
        if isinstance(prompt, str):
            return prompt
        return tok.apply_chat_template(
            prompt, tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )

    step_counter = {"n": 0}

    def reward_fn(prompts, completions, **_):
        if args.log_completions and step_counter["n"] % args.logging_steps == 0:
            _log_completions(prompts, completions, step_counter["n"], tok, args)
        step_counter["n"] += 1

        tasks = []
        idx_to_original: dict = {}
        for idx, (prompt, completion) in enumerate(zip(prompts, completions)):
            data = prompt_to_data.get(hash_fn(_prompt_to_string(prompt)))
            if data is None:
                logger.warning(f"Could not find prompt data for completion {idx}")
                continue
            idx_to_original[len(tasks)] = idx
            criteria_history = data.get("criteria_history")
            if isinstance(criteria_history, str):
                criteria_history = json.loads(criteria_history)
            tasks.append({
                "idx": len(tasks),
                "chat_history": data["prompt"],
                "criteria_history": criteria_history,
                "assistant_response": _extract_completion_text(completion),
            })

        results = reward_computer.compute_rewards(tasks) if tasks else []
        per_orig = {idx_to_original[r["idx"]]: r["reward"] for r in results}
        return [float(per_orig.get(i, 0.0)) for i in range(len(prompts))]

    return reward_fn


def _log_completions(prompts, completions, step, tok, args) -> None:
    entries = []
    for i, (prompt, completion) in enumerate(zip(prompts, completions)):
        prompt_text = prompt if isinstance(prompt, str) else tok.apply_chat_template(
            prompt, tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        completion_text = _extract_completion_text(completion)
        entries.append({
            "step": step,
            "prompt_idx": i,
            "prompt": (prompt_text[:500] + "...") if len(prompt_text) > 500 else prompt_text,
            "completion": (completion_text[:1000] + "...") if len(completion_text) > 1000 else completion_text,
            "timestamp": datetime.now().isoformat(),
        })
    if args.wandb_project and os.environ.get("LOCAL_RANK", "0") == "0":
        try:
            import wandb

            table = wandb.Table(
                columns=["step", "prompt_idx", "prompt", "completion"],
                data=[[e["step"], e["prompt_idx"], e["prompt"], e["completion"]] for e in entries],
            )
            wandb.log({f"completions/step_{step}": table}, step=step)
        except Exception as e:
            logger.warning(f"Failed to log completions to wandb: {e}")
    if args.log_completions_file:
        try:
            os.makedirs(os.path.dirname(args.log_completions_file) or ".", exist_ok=True)
            with open(args.log_completions_file, "a") as f:
                for entry in entries:
                    f.write(json.dumps(entry) + "\n")
        except Exception as e:
            logger.warning(f"Failed to write completions to file: {e}")


if __name__ == "__main__":
    load_dotenv(".env")
    main()
