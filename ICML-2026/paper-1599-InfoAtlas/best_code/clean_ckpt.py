"""
Clean a Lightning checkpoint for inference-only use.

Removes optimizer states, lr_scheduler states, and other training-only data,
keeping only the model ``state_dict`` and a minimal, inference-relevant config.

The embedded config is reduced to the fields ``infer.load_ckpt`` actually needs to
rebuild the model and reproduce preprocessing (architecture dims + soft-rank /
whitening settings). Environment-specific and training-only entries (paths, data
pipeline, optimizer schedule, run names, ...) are dropped so the released file is
small and self-contained. Pass ``--keep_full_cfg`` to keep the config verbatim.

If the checkpoint does not already contain a 'cfg' key, you can optionally
supply --cfg_path to embed one.

Usage:
    python clean_ckpt.py --ckpt_path /path/to/last.ckpt
    python clean_ckpt.py --ckpt_path /path/to/last.ckpt --output_path /path/to/clean.ckpt
    python clean_ckpt.py --ckpt_path /path/to/last.ckpt --cfg_path /path/to/config.yaml
"""
import argparse
import os

import torch
from omegaconf import OmegaConf

# Fields kept in the embedded config. These are exactly what load_ckpt reads to
# rebuild the model (encoder / decoder / query generator / hypernetwork) and to
# reproduce the input preprocessing. Anything not listed here is dropped.
INFERENCE_CFG_KEYS = (
    # input dimensionality
    "input_dim_x", "input_dim_y",
    # encoder (Perceiver)
    "latent_num", "latent_dim",
    "cross_attn_heads", "self_attn_heads",
    "num_self_attn_per_block", "num_self_attn_blocks",
    # legacy encoder fields (older checkpoints)
    "encoder2_expand_dim", "encoder2_hiddim", "encoder2_block",
    # decoder / query generator
    "decoder_query_dim", "targetnet_hiddim",
    # hypernetwork critic
    "weight_dim", "enc_dec_dim", "opt_block_dim", "opt_mid_dim",
    "num_opt_mlp_layer", "num_enc_dec_layer", "num_layers",
    "weight_split_dim", "nhead", "lr_scheme_method", "use_compile",
    "replicate_blocks", "target_layers_num", "ablation",
    # preprocessing (must match training)
    "softrank_reg", "gauss_copula", "whiten", "whiten_eps",
)


def scrub_cfg(cfg_container: dict) -> dict:
    """Keep only inference-relevant keys from an embedded config dict."""
    return {k: cfg_container[k] for k in INFERENCE_CFG_KEYS if k in cfg_container}


def clean_ckpt(ckpt_path: str, output_path: str = None, cfg_path: str = None,
               keep_full_cfg: bool = False):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    if "state_dict" not in ckpt:
        raise KeyError(f"'state_dict' not found in checkpoint: {ckpt_path}")

    clean = {"state_dict": ckpt["state_dict"]}

    # Resolve the source config (from the ckpt, or a supplied yaml).
    cfg_container = None
    if "cfg" in ckpt:
        cfg = ckpt["cfg"]
        cfg_container = cfg if isinstance(cfg, dict) else OmegaConf.to_container(
            OmegaConf.create(cfg), resolve=True)
        print("[clean_ckpt] cfg found in checkpoint.")
    elif cfg_path is not None:
        if not os.path.exists(cfg_path):
            raise FileNotFoundError(f"Config file not found: {cfg_path}")
        cfg_container = OmegaConf.to_container(OmegaConf.load(cfg_path), resolve=True)
        print(f"[clean_ckpt] cfg embedded from: {cfg_path}")
    else:
        print("[clean_ckpt] WARNING: no cfg found in checkpoint and no --cfg_path provided. "
              "The cleaned checkpoint will not contain config.")

    if cfg_container is not None:
        if keep_full_cfg:
            clean["cfg"] = cfg_container
            print("[clean_ckpt] keeping full cfg (--keep_full_cfg).")
        else:
            clean["cfg"] = scrub_cfg(cfg_container)
            dropped = sorted(set(cfg_container) - set(clean["cfg"]))
            print(f"[clean_ckpt] cfg reduced to {len(clean['cfg'])} inference keys; "
                  f"dropped {len(dropped)} training/environment keys.")

    if output_path is None:
        base, ext = os.path.splitext(ckpt_path)
        output_path = f"{base}_clean{ext}"

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    torch.save(clean, output_path)

    orig_size = os.path.getsize(ckpt_path) / (1024 * 1024)
    new_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"[clean_ckpt] original:  {ckpt_path} ({orig_size:.1f} MB)")
    print(f"[clean_ckpt] cleaned:   {output_path} ({new_size:.1f} MB)")
    print(f"[clean_ckpt] size reduction: {orig_size - new_size:.1f} MB ({(1 - new_size / orig_size) * 100:.1f}%)")

    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean checkpoint for inference")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to the Lightning .ckpt file")
    parser.add_argument("--output_path", type=str, default=None, help="Output path (default: <name>_clean.ckpt)")
    parser.add_argument("--cfg_path", type=str, default=None,
                        help="Path to config .yaml to embed (only needed if ckpt has no cfg)")
    parser.add_argument("--keep_full_cfg", action="store_true",
                        help="Keep the embedded config verbatim instead of reducing it to inference keys")
    args = parser.parse_args()

    clean_ckpt(args.ckpt_path, args.output_path, args.cfg_path, args.keep_full_cfg)
