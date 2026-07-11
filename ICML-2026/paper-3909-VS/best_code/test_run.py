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
        "num_prompts": 5,
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
    output_base_dir=Path("{}/{}_{}_poem".format(OUTPUT_DIR, "test", model_basename)),
    skip_existing=False,
    num_workers=16,
)

pipeline = Pipeline(config)
pipeline.run_complete_pipeline()
