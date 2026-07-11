# Copyright 2025 CHATS-Lab. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Simple script for running math tasks with standard generation.
Similar to run_jokes_local.py but focused on math reasoning.
"""

from pathlib import Path

from verbalized_sampling.methods import Method
from verbalized_sampling.pipeline import (
    EvaluationConfig,
    ExperimentConfig,
    Pipeline,
    PipelineConfig,
)
from verbalized_sampling.tasks import Task


def run_math_generation(
    model_name: str, task: Task, output_dir: str = "generated_data/math_simple"
):
    """Run simple math generation experiment."""

    print(f"🧮 Running Math Generation: {model_name} on {task.value}")

    # Simple configuration for standard generation
    experiment = ExperimentConfig(
        name="direct_generation",
        task=task,
        model_name=model_name,
        method=Method.DIRECT,
        num_responses=1,
        num_prompts=20,  # Small test set
        target_words=0,
        temperature=0.1,  # Low temperature for math
        random_seed=42,
        use_vllm=True,
        strict_json=False,
    )

    model_basename = model_name.replace("/", "_")
    config = PipelineConfig(
        experiments=[experiment],
        evaluation=EvaluationConfig(metrics=["accuracy"]),
        output_base_dir=Path(f"{output_dir}/{model_basename}_{task.value}"),
        skip_existing=False,
        num_workers=32,
    )

    pipeline = Pipeline(config)
    pipeline.run_complete_pipeline()
    print(f"✅ Results: {output_dir}/{model_basename}_{task.value}/pipeline_report.html")


if __name__ == "__main__":
    # Simple test with one model and one math dataset

    # Model to test (change as needed)
    model = "Qwen/Qwen3-4B-Base"

    # Test on MATH dataset
    run_math_generation(model_name=model, task=Task.MATH, output_dir="generated_data/math_simple")

    print("🎉 Math generation test complete!")
    print("To test other models/datasets, edit the script and change:")
    print("  - model = 'Qwen/Qwen3-4B-Thinking-2507'")
    print("  - task = Task.AIME  # or Task.AMC")
