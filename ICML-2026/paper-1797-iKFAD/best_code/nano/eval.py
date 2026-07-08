"""Single-run evaluation for iKFAD GPT2-Nano Shakespeare.
Usage: python3 eval.py --seed 1337 [--grad-clip 2.0]
Outputs JSON with test_loss to stdout.
"""
import sys, os, argparse, json
import torch

# Suppress training step output
import logging
logging.disable(logging.CRITICAL)

from config.default_config import get_default_config
from train import GPTTrainer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--max-iters", type=int, default=5001)
    parser.add_argument("--grad-clip", type=float, default=None)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    config = get_default_config()
    config.update({
        "optimizer_name": "iKFAD",
        "optimizer_params": {
            "h": 0.49412914916046896,
            "alpha": 2.1955477450604213,
            "mu": 4.552149543615035e-06,
            "gamma": 0.0,
        },
        "weight_decay": 0.0,
        "out_dir": f"out-eval/seed_{args.seed}",
        "dataset": "shakespeare-char",
        "device": "cuda",
        "dtype": "float16",
        "compile": False,
        "grad_clip": args.grad_clip if args.grad_clip is not None else 1.0,
        "seed": args.seed,
        "n_layer": 4, "n_head": 4, "n_embd": 128,
        "block_size": 64, "batch_size": 16,
        "dropout": 0., "bias": False, "flash_attn": False,
        "gradient_accumulation_steps": 1,
        "max_iters": args.max_iters,
        "eval_interval": 100,
        "eval_iters": 100,
        "log_interval": 100,
        "always_save_checkpoint": False,
        "ckpt_interval": 20000,
        # h scheduling (warmup + cosine decay)
        "h_max": 0.49412914916046896,
        "h_min": 0.08,
        "warmup_iters": 500,
        "lr_decay_iters": 5001,
        # alpha damping scheduling
        "alpha_max": 2.1955477450604213,
        "alpha_min": 0.8,
        "alpha_decay_start": 1000,
    })

    trainer = GPTTrainer(config)
    best_val_loss = trainer.train()
    del trainer
    torch.cuda.empty_cache()

    result = {"test_loss": float(best_val_loss), "seed": args.seed}
    print(json.dumps(result))
    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f)
    return result

if __name__ == "__main__":
    main()
