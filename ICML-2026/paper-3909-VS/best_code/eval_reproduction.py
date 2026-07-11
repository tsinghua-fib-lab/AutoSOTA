#!/usr/bin/env python3
"""Reproduction evaluation script for paper 3909: Verbalized Sampling.
Runs the poem continuation experiment and computes diversity, ngram, and length metrics.

Usage: cd /repo && python3 eval_reproduction.py

Paper settings: 100 prompts, 30 responses, k=5 candidates, temperature=0.7, top_p=1.0
Reduced to 5 prompts for time-constrained reproduction. Increase --num-prompts for full scale.
"""

import os, sys, json, argparse
from pathlib import Path

# --- Proxy/API setup ---
# Remove SOCKS proxy that interferes with DeepSeek API connections
for k in ["ALL_PROXY", "all_proxy"]:
    os.environ.pop(k, None)
for k in ["no_proxy", "NO_PROXY"]:
    val = os.environ.get(k, "")
    if "api.deepseek.com" not in val:
        os.environ[k] = val + ",api.deepseek.com" if val else "api.deepseek.com"

# DeepSeek API as OpenAI-compatible backend
os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY", "[REDACTED]")
os.environ["OPENAI_BASE_URL"] = os.environ.get("OPENAI_BASE_URL", "https://api.deepseek.com/v1")
os.environ["HF_TOKEN"] = os.environ.get("HF_TOKEN", "[REDACTED]")
os.environ["HUGGINGFACE_HUB_TOKEN"] = os.environ.get("HUGGINGFACE_HUB_TOKEN", "[REDACTED]")

from verbalized_sampling.methods import Method
from verbalized_sampling.pipeline import (
    EvaluationConfig,
    ExperimentConfig,
    Pipeline,
    PipelineConfig,
)
from verbalized_sampling.tasks import Task


def main():
    parser = argparse.ArgumentParser(description="Reproduce Poem Continuation experiment")
    parser.add_argument("--model", type=str, default="openai/deepseek-v4-flash",
                        help="Model name (OpenAI-compatible)")
    parser.add_argument("--num-prompts", type=int, default=5,
                        help="Number of prompts (paper uses 100)")
    parser.add_argument("--num-responses", type=int, default=30,
                        help="Responses per prompt")
    parser.add_argument("--num-samples", type=int, default=5,
                        help="Candidates per LLM call (k)")
    parser.add_argument("--output-dir", type=str, default="reproduction_results",
                        help="Output directory")
    parser.add_argument("--workers", type=int, default=8,
                        help="Number of concurrent API workers")
    parser.add_argument("--methods", type=str, default="direct,vs_standard",
                        help="Comma-separated methods to run")
    parser.add_argument("--metrics", type=str, default="diversity,ngram,length",
                        help="Comma-separated metrics")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="LLM temperature (paper: 0.7)")
    parser.add_argument("--top-p", type=float, default=1.0,
                        help="LLM top_p (paper: 1.0)")
    parser.add_argument("--probability-tuning", type=float, default=-1,
                        help="Probability threshold for VS tail sampling")
    parser.add_argument("--probability-definition", type=str, default="implicit",
                        help="Probability definition type for VS prompt")
    parser.add_argument("--local-embed-model", type=str, default=None,
                        help="Local sentence-transformers model for diversity embeddings")
    args = parser.parse_args()

    method_map = {
        "direct": (Method.DIRECT, False, 1),
        "vs_standard": (Method.VS_STANDARD, True, 5),
        "vs_cot": (Method.VS_COT, True, args.num_samples),
        "vs_multi": (Method.VS_MULTI, True, 5),
        "sequence": (Method.SEQUENCE, True, 5),
    }

    methods = []
    for mname in args.methods.split(","):
        mname = mname.strip()
        if mname in method_map:
            meth, strict, samples = method_map[mname]
            methods.append({"method": meth, "strict_json": strict, "num_samples": samples})

    # Set local embed model if specified
    if args.local_embed_model:
        os.environ["LOCAL_EMBED_MODEL"] = args.local_embed_model

    experiments = []
    for mcfg in methods:
        base = {
            "task": Task.POEM,
            "model_name": args.model,
            "num_responses": args.num_responses,
            "num_prompts": args.num_prompts,
            "target_words": 200,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "random_seed": 42,
            "use_vllm": True,
            "probability_tuning": args.probability_tuning,
            "probability_definition": args.probability_definition,
        }
        experiments.append(ExperimentConfig(name=mcfg["method"].value, **base, **mcfg))

    metrics = [m.strip() for m in args.metrics.split(",")]
    model_basename = args.model.replace("/", "_")
    output_dir = Path("{}/{}p_{}".format(args.output_dir, args.num_prompts, model_basename))

    config = PipelineConfig(
        experiments=experiments,
        evaluation=EvaluationConfig(metrics=metrics),
        output_base_dir=output_dir,
        skip_existing=False,
        num_workers=args.workers,
    )

    pipeline = Pipeline(config)
    pipeline.run_complete_pipeline()

    # Print key results
    eval_dir = output_dir / "evaluation"
    print("\n" + "=" * 60)
    print("REPRODUCTION RESULTS")
    print("=" * 60)
    for method_name in [m["method"].value for m in methods]:
        print("\n--- {} ---".format(method_name))
        for metric in metrics:
            fpath = eval_dir / method_name / "{}_results.json".format(metric)
            if fpath.exists():
                with open(fpath) as f:
                    data = json.load(f)
                    overall = data.get("overall_metrics", {})
                    if metric == "diversity":
                        avg = overall.get("avg_diversity", 0) * 100
                        std = overall.get("std_diversity", 0) * 100
                        print("  Diversity: {:.2f}% ± {:.2f}%".format(avg, std))
                    elif metric == "ngram":
                        avg = overall.get("avg_rouge_l", 0) * 100
                        std = overall.get("std_rouge_l", 0) * 100
                        print("  Rouge-L:   {:.2f}% ± {:.2f}%".format(avg, std))
                        print("  Distinct-N:{:.4f}".format(overall.get("avg_distinct_n", 0)))
                    elif metric == "length":
                        for k, v in overall.items():
                            if "length" in k:
                                print("  {}: {:.2f}".format(k, v))
    return 0


if __name__ == "__main__":
    sys.exit(main())
