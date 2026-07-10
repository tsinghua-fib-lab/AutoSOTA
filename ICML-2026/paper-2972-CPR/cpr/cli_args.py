"""Shared CLI arguments for CPR entry points."""

import argparse
import os


def add_common_args(parser: argparse.ArgumentParser):
    parser.add_argument("--use_ts", action="store_true", help="Thompson sampling during retrieval")
    parser.add_argument(
        "--encoder_dir",
        type=str,
        default=None,
        help="SentenceTransformer path (env CPR_ENCODER_DIR)",
    )
    parser.add_argument("--train", type=str, default="./data/webqsp/train.json")
    parser.add_argument("--test", type=str, default="./data/webqsp/test.json")
    parser.add_argument("--max_train", type=int, default=None)
    parser.add_argument("--max_test", type=int, default=None)
    parser.add_argument("--max_hop", type=int, default=2)
    parser.add_argument("--calib_frac", type=float, default=0.1)
    parser.add_argument("--conformal_mode", type=str, default="path", choices=["path", "legacy"])
    parser.add_argument("--no_llm", action="store_true", help="Disable LLM relation hints")
    parser.add_argument("--vllm_url", type=str, default=None, help="vLLM base URL (env CPR_VLLM_URL)")
    parser.add_argument("--llm_model", type=str, default=None, help="LLM model name (env CPR_LLM_MODEL)")

    parser.add_argument("--beam_size", type=int, default=32)
    parser.add_argument("--treeg_branch_size", type=int, default=32)
    parser.add_argument("--treeg_active_size", type=int, default=32)
    # IDEA-12 best config (asymmetric TreeG weights): local=2.0, path=0.5, prior=0.5
    parser.add_argument("--w_local", type=float, default=2.0)
    parser.add_argument("--w_path", type=float, default=0.5)
    parser.add_argument("--w_prior", type=float, default=0.5)

    parser.add_argument("--learn_residual_value", type=int, default=1)
    # IDEA-12 best config: residual_lambda=2.0
    parser.add_argument("--residual_lambda", type=float, default=2.0)
    parser.add_argument("--delta_clip", type=float, default=0.2)
    parser.add_argument("--delta_l2", type=float, default=0.5)
    parser.add_argument("--value_hidden", type=int, default=256)
    # IDEA-12 best config: RCVNet trained for value_epochs=12 at value_lr=1e-3
    parser.add_argument("--value_lr", type=float, default=1e-3)
    parser.add_argument("--value_epochs", type=int, default=12)
    parser.add_argument("--value_batch_size", type=int, default=256)
    parser.add_argument("--value_max_negs", type=int, default=8)
    parser.add_argument("--value_l2", type=float, default=1e-4)
    parser.add_argument("--hard_neg_frac", type=float, default=0.5)
    parser.add_argument("--value_norm", type=int, default=1)
    parser.add_argument("--value_use_embeddings", type=int, default=2)

    parser.add_argument("--puct_calib", action="store_true")
    parser.add_argument("--puct_calib_num_sims", type=int, default=32)
    parser.add_argument("--puct_calib_cpuct", type=float, default=2.0)
    parser.add_argument("--puct_calib_temp", type=float, default=2.0)
    parser.add_argument("--puct_calib_prior_w", type=float, default=0.5)
    parser.add_argument("--puct_calib_update_scale", type=float, default=1.0)
    parser.add_argument("--puct_calib_fail_beta", type=float, default=0.05)

    parser.add_argument(
        "--alpha",
        type=float,
        default=0.1,
        dest="alpha",
        help="Conformal risk level alpha (target miscoverage; coverage >= 1-alpha)",
    )


def apply_config(args, config: dict):
    """Overlay YAML config onto argparse namespace."""
    for k, v in config.items():
        if hasattr(args, k):
            setattr(args, k, v)


def load_yaml_config(path: str) -> dict:
    try:
        import yaml
    except ImportError:
        raise ImportError("PyYAML required for --config. Install: pip install pyyaml")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolve_env_paths(args):
    if args.encoder_dir is None:
        args.encoder_dir = os.environ.get(
            "CPR_ENCODER_DIR", "sentence-transformers/all-MiniLM-L6-v2"
        )
    if args.vllm_url is None:
        args.vllm_url = os.environ.get("CPR_VLLM_URL", "http://127.0.0.1:8000/v1")
    if args.llm_model is None:
        args.llm_model = os.environ.get("CPR_LLM_MODEL", "Qwen/Qwen3-8B")
    args.use_llm = not args.no_llm
    return args
