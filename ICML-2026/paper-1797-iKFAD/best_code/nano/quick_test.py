import sys, os
sys.path.insert(0, "best_hparams")
sys.path.insert(0, ".")

import torch
from config.default_config import get_default_config
from train import GPTTrainer

config = get_default_config()
config.update({
    "optimizer_name": "iKFAD",
    "optimizer_params": {"h": 0.49412914916046896, "alpha": 2.1955477450604213, "mu": 4.552149543615035e-06, "gamma": 0.0},
    "weight_decay": 0.0,
    "out_dir": "out-test-quick",
    "dataset": "shakespeare-char",
    "device": "cuda",
    "dtype": "float16",
    "compile": False,
    "grad_clip": 1.0,
    "seed": 1337,
    "n_layer": 4, "n_head": 4, "n_embd": 128,
    "block_size": 64, "batch_size": 16,
    "dropout": 0., "bias": False, "flash_attn": False,
    "gradient_accumulation_steps": 1,
    "max_iters": 50,
    "eval_interval": 10,
    "eval_iters": 10,
    "log_interval": 10,
    "always_save_checkpoint": False,
    "ckpt_interval": 1000,
})
trainer = GPTTrainer(config)
result = trainer.train()
print(f"Final best val loss: {result}")
del trainer
torch.cuda.empty_cache()
print("Quick test PASSED")
