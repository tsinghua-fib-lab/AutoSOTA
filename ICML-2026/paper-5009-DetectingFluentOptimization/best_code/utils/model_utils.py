import os
import yaml
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def load_model_configs(model_name):
    config_path = os.path.join("config", "models.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Central model config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    if model_name not in config.get("model_configs", {}):
        raise ValueError(f"Model '{model_name}' not found in config file")
    
    # Merge global settings with model-specific config
    model_config = config["model_configs"][model_name]
    model_config["alpha"] = config.get("alpha", 1.5)
    model_config["window_size"] = config.get("window_size", 20)
    # Device selection:
    # - Default comes from config/models.yaml (cuda:0).
    # - Override with DEVICE env var (e.g., DEVICE=cuda:1 or DEVICE=cpu).
    # - Use DEVICE=auto to enable automatic device_map placement.
    default_device = config.get("device", "cuda:0")
    model_config["device"] = os.environ.get("DEVICE", default_device)
    return model_config

def initialize_model(model_config):
    name = model_config["name"]
    tok_id = model_config["tokenizer"]
    device = str(model_config.get("device", "cuda:0")).strip()

    # 1) Load tokenizer (fallback to slow if fast tokenizer fails, e.g., due to tokenizers version issues)
    try:
        tokenizer = AutoTokenizer.from_pretrained(tok_id, trust_remote_code=True)
    except Exception as exc:
        logging.warning("Fast tokenizer load failed for %s (%s); falling back to use_fast=False", tok_id, exc)
        tokenizer = AutoTokenizer.from_pretrained(tok_id, use_fast=False, trust_remote_code=True)

    if device.lower() == "auto":
        device_map = "auto"
    else:
        device_map = {"": device}

    # fp16 is appropriate on CUDA; default to fp32 on CPU for broad compatibility.
    dtype = torch.float16 if "cuda" in device.lower() else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        tok_id,
        device_map=device_map,
        trust_remote_code=True,
        torch_dtype=dtype,

    )
    model.eval()

    logging.info(f"Loaded {name} on {model.device}")
    return model, tokenizer
