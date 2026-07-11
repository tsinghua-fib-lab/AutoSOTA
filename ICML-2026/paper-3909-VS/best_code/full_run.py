import os, sys

# Remove SOCKS proxy that breaks connections to DeepSeek API
for k in ["ALL_PROXY", "all_proxy"]:
    os.environ.pop(k, None)

# Add api.deepseek.com to no_proxy
for k in ["no_proxy", "NO_PROXY"]:
    val = os.environ.get(k, "")
    if "api.deepseek.com" not in val:
        os.environ[k] = val + ",api.deepseek.com" if val else "api.deepseek.com"

os.environ["OPENAI_API_KEY"] = "[REDACTED]"
os.environ["OPENAI_BASE_URL"] = "https://api.deepseek.com/v1"
os.environ["HF_TOKEN"] = "[REDACTED]"
os.environ["HUGGINGFACE_HUB_TOKEN"] = "[REDACTED]"

from pathlib import Path
from verbalized_sampling.methods import Method
from verbalized_sampling.pipeline import (
    EvaluationConfig,
    ExperimentConfig,
    Pipeline,
    PipelineConfig,
)
from verbalized_sampling.tasks import Task

MODEL = "openai/deepseek-v4-pro"
OUTPUT_DIR = "reproduction_results"

# Full paper settings: 100 prompts, 30 responses, k=5
methods = [
    {
        "method": Method.DIRECT,
        "strict_json": False,
        "num_samples": 1,
    },
    {
        "method": Method.VS_STANDARD,
        "strict_json": True,
        "num_samples": 5,
    },
]

experiments = []
for mcfg in methods:
    method_name = mcfg["method"].value
    base = {
        "task": Task.POEM,
        "model_name": MODEL,
        "num_responses": 30,
        "num_prompts": 100,
        "target_words": 200,
        "temperature": 0.7,
        "top_p": 1.0,
        "random_seed": 42,
        "use_vllm": True,
    }
    experiments.append(ExperimentConfig(name=method_name, **base, **mcfg))

model_basename = MODEL.replace("/", "_")
config = PipelineConfig(
    experiments=experiments,
    evaluation=EvaluationConfig(metrics=["diversity", "ngram", "length"]),
    output_base_dir=Path("{}/{}_{}_poem".format(OUTPUT_DIR, "full", model_basename)),
    skip_existing=False,
    num_workers=16,
)

pipeline = Pipeline(config)
results = pipeline.run_complete_pipeline()

# Print key metrics
import json
eval_dir = Path("{}/{}_{}_poem/evaluation".format(OUTPUT_DIR, "full", model_basename))
for method_name in ["direct", "vs_standard"]:
    for metric in ["diversity", "ngram", "length"]:
        fpath = eval_dir / method_name / "{}_results.json".format(metric)
        if fpath.exists():
            with open(fpath) as f:
                data = json.load(f)
                overall = data.get("overall_metrics", {})
                if metric == "diversity":
                    print("{} {}: avg_diversity={:.4f}".format(method_name, metric, overall.get("avg_diversity", 0)))
                elif metric == "ngram":
                    print("{} {}: avg_rouge_l={:.4f}".format(method_name, metric, overall.get("avg_rouge_l", 0)))
                elif metric == "length":
                    print("{} {}: avg_response_length={:.4f}".format(method_name, metric, overall.get("avg_response_length", 0)))
