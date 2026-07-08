"""
Evaluate CLIP models with CSR defense on clean and adversarial samples.

Usage:
    python scripts/evaluate.py --model CLIP-L-14 --defense csr --device cuda:0
    python scripts/evaluate.py --model CLIP-L-14 --defense none --device cuda:0
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from csr.config import PathConfig, EvalConfig, CSRConfig, TTCConfig, HF_MODEL_CONFIGS
from csr.defense import CSRDefense, FastCSRDefense, TTCDefense
from csr.evaluation import CLIPBenchmark


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate CLIP robustness")
    parser.add_argument("--model", type=str, default="CLIP-L-14")
    parser.add_argument("--defense", type=str, default="csr", choices=["none", "csr", "fast_csr", "ttc"])
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--sample_n", type=int, default=1000)
    parser.add_argument("--datasets", type=str, nargs="+", default=None)
    parser.add_argument("--adv_root", type=str, default=None, help="Root for adversarial samples")
    parser.add_argument("--adv_attack", type=str, default="PGD", help="Attack name for adv subfolder")
    parser.add_argument("--output", type=str, default=None)
    # CSR hyperparameters
    parser.add_argument("--lpf_radius", type=int, default=40)
    parser.add_argument("--detect_thresh", type=float, default=0.85)
    parser.add_argument("--purify_steps", type=int, default=3)
    parser.add_argument("--filter_type", type=str, default="gaussian")
    parser.add_argument("--purify_eps", type=float, default=4/255, help="Purification epsilon budget")
    parser.add_argument("--purify_alpha", type=float, default=2/255, help="Purification step size")
    parser.add_argument("--butterworth_order", type=int, default=2, help="Butterworth filter order")
    # TTC hyperparameters
    parser.add_argument("--ttc_eps", type=float, default=4 / 255)
    parser.add_argument("--ttc_alpha", type=float, default=2 / 255)
    parser.add_argument("--ttc_steps", type=int, default=3)
    parser.add_argument("--ttc_tau", type=float, default=0.3)
    parser.add_argument("--ttc_beta", type=float, default=2.0)
    return parser.parse_args()


def main():
    args = parse_args()
    paths = PathConfig()
    eval_cfg = EvalConfig(batch_size=args.batch_size, sample_n=args.sample_n)

    model_path = HF_MODEL_CONFIGS[args.model]

    # Build model (with or without defense)
    if args.defense == "none":
        from transformers import CLIPModel, CLIPProcessor
        model = CLIPModel.from_pretrained(model_path, local_files_only=True).to(args.device).eval()
        model.processor = CLIPProcessor.from_pretrained(model_path, local_files_only=True)
        model_label = args.model
    elif args.defense in {"csr", "fast_csr"}:
        csr_cfg = CSRConfig(
            filter_type=args.filter_type,
            lpf_radius=args.lpf_radius,
            detect_thresh=args.detect_thresh,
            purify_steps=args.purify_steps,
            purify_eps=args.purify_eps,
            purify_alpha=args.purify_alpha,
            butterworth_order=args.butterworth_order,
        )
        DefClass = FastCSRDefense if args.defense == "fast_csr" else CSRDefense
        model = DefClass(model_path, config=csr_cfg, device=args.device)
        model_label = f"{args.model}+{args.defense.upper()}"
        model.display_name = model_label
        model.adv_subdir_name = args.model
    else:
        ttc_cfg = TTCConfig(
            eps=args.ttc_eps,
            alpha=args.ttc_alpha,
            steps=args.ttc_steps,
            tau=args.ttc_tau,
            beta=args.ttc_beta,
        )
        model = TTCDefense(model_path, config=ttc_cfg, device=args.device)
        model_label = f"{args.model}+TTC"
        model.display_name = model_label
        model.adv_subdir_name = args.model

    # Configure adversarial sources
    adv_roots = {}
    if args.adv_root:
        adv_roots[args.adv_attack] = args.adv_root

    # Run benchmark
    benchmark = CLIPBenchmark(
        data_root=paths.data_root,
        adv_roots_dict=adv_roots,
        device=args.device,
        eval_config=eval_cfg,
    )

    datasets = args.datasets or eval_cfg.datasets
    benchmark.run(
        model_configs={model_label: model},
        datasets_list=datasets,
    )

    benchmark.print_results()

    if args.output:
        benchmark.save_results(args.output)


if __name__ == "__main__":
    main()
