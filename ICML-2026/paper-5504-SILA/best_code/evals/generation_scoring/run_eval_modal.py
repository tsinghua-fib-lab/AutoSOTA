#!/usr/bin/env python3
"""
Modal app for evaluating reflective coherence of SAE latent labels.
Thin wrapper that runs evaluation functions on Modal infrastructure.

Multi-GPU Configuration for 70B Models:
  To run with 70B models (e.g., Llama-3.3-70B-Instruct):
  1. Update gpu parameter in @app.function decorator: gpu="A100-80GB:3" (for 3 GPUs)
  2. Set device="auto" in your config JSON's model_config section
  3. Consider reducing batch sizes (e.g., label_generation_batch_size=16, reward_evaluation_batch_size=8)
  
  Example usage:
    modal run run_eval_modal.py --config-file configs/config_70b_label_dataset.json ...
"""

import modal
from pathlib import Path

# Create the Modal app
app = modal.App("sae-reflective-coherence-eval")

# Get the repo root (two levels up from this file)
REPO_ROOT = Path(__file__).parent.parent.parent

# Create the base image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "torch>=2.0.0",
        "transformers>=4.44.0",
        "accelerate>=0.20.0",
        "huggingface_hub>=0.20.0",
        "datasets>=2.0.0",
        "wandb>=0.16.0",
        "numpy>=1.24.0",
        "tqdm>=4.65.0",
        "nnsight>=0.4.0,<0.5.0",
        "safetensors>=0.3.0",
        "sae-lens>=4.0.0",
    )
    # Copy the entire clean repo into the image
    .add_local_dir(
        local_path=str(REPO_ROOT),
        remote_path="/root/selfie_adapters_repo",
        ignore=[".venv", "venv", "__pycache__", ".git", "*.pyc", "*.pyo", 
                "wandb", "results", ".cache", "cache", "checkpoints", "datasets",
                "*.pt", "*.pth", "*.safetensors", "*.bin",
                ".DS_Store", ".vscode", ".idea", ".modal", ".mypy_cache"]
    )
)

# Volumes for caching models/data, storing results, and SAE data
cache_volume = modal.Volume.from_name("sae-model-cache", create_if_missing=True)
results_volume = modal.Volume.from_name("sae-eval-results", create_if_missing=True)
sae_data_volume = modal.Volume.from_name("sae-data", create_if_missing=True)
training_volume = modal.Volume.from_name("selfie-adapter-training", create_if_missing=True)


@app.function(
    image=image,
    gpu="A100-80GB",  # Single GPU for 8B models. For 70B models, use gpu="A100-80GB:3" and set device="auto" in config
    memory=64 * 1024,  # 64GB memory
    timeout=4 * 3600,  # 4 hour timeout
    volumes={
        "/cache": cache_volume,
        "/results": results_volume,
        "/sae-data": sae_data_volume,
        "/training": training_volume,  # selfie-adapter-training volume
    },
    secrets=[
        modal.Secret.from_name("huggingface-token"),
        modal.Secret.from_name("wandb-secret"),
    ],
)
def evaluate_reflective_coherence_shard(
    config_json: str, shard_id: int, total_shards: int
):
    """Evaluate reflective coherence for a shard of latents/labels."""
    
    import json
    import os
    import sys
    import traceback
    from datetime import datetime

    # Add repo to path for imports
    sys.path.insert(0, "/root/selfie_adapters_repo")
    sys.path.insert(0, "/root/selfie_adapters_repo/evals/generation_scoring")

    import wandb
    from config import EvaluationConfig, LabelGeneratorConfig, ModelConfig, RewardConfig
    from evaluation_functions import (
        evaluate_label_dataset,
        evaluate_label_generator,
        generate_labels_only,
    )
    from master_json_loader import (
        get_label_dataset_from_master_json,
        get_vectors_from_master_json,
    )

    # Configure HuggingFace cache to use Modal cache volume
    cache_base = "/cache/huggingface"
    cache_hub = f"{cache_base}/hub"
    cache_datasets = f"{cache_base}/datasets"

    os.makedirs(cache_hub, exist_ok=True)
    os.makedirs(cache_datasets, exist_ok=True)

    os.environ["HF_HOME"] = cache_base
    os.environ["TRANSFORMERS_CACHE"] = cache_hub
    os.environ["HF_DATASETS_CACHE"] = cache_datasets
    os.environ["HF_HUB_CACHE"] = cache_hub

    print(f"🗂️  Using HuggingFace cache: {cache_base}")

    try:
        # Parse configuration
        config_dict = json.loads(config_json)

        # Initialize WandB early for error tracking
        wandb.init(
            project="sae-reflective-coherence-eval",
            name=f"eval_shard_{shard_id}_{datetime.now().isoformat()}",
            config=config_dict,
        )

        # Construct nested config objects
        model_config = ModelConfig(**config_dict["model_config"])
        reward_config = RewardConfig(**config_dict["reward_config"])

        label_generator_config = None
        if config_dict.get("label_generator_config"):
            label_generator_config = LabelGeneratorConfig(
                **config_dict["label_generator_config"]
            )

        # Create the evaluation config with properly constructed nested objects
        config_dict_fixed = config_dict.copy()
        config_dict_fixed["model_config"] = model_config
        config_dict_fixed["reward_config"] = reward_config
        config_dict_fixed["label_generator_config"] = label_generator_config

        config = EvaluationConfig(**config_dict_fixed)

        # Validate master JSON configuration
        if not config.master_json_path:
            raise ValueError("master_json_path must be set in config")
        if not config.master_json_dataset_name:
            raise ValueError("master_json_dataset_name must be set in config")
        if config.master_json_layer is None:
            raise ValueError("master_json_layer must be set in config")

        # Load input data from master JSON
        print(f"📂 Loading data from master JSON: {config.master_json_path}")
        print(f"   Dataset: {config.master_json_dataset_name}")
        print(f"   Layer: {config.master_json_layer}")
        print(f"   Split: {config.master_json_split}")
        
        if config.evaluation_mode == "label_dataset":
            input_data = get_label_dataset_from_master_json(
                master_json_path=config.master_json_path,
                dataset_name=config.master_json_dataset_name,
                layer=config.master_json_layer,
                split=config.master_json_split,
                data_volume_path=config.data_volume_path,
                max_latents=config.max_latents,
                specific_latent_indices=config.specific_latent_indices,
            )
            print(f"✓ Loaded {len(input_data)} label entries from master JSON")
        
        elif config.evaluation_mode in ["label_generator", "label_generation_only"]:
            input_data = get_vectors_from_master_json(
                master_json_path=config.master_json_path,
                dataset_name=config.master_json_dataset_name,
                layer=config.master_json_layer,
                split=config.master_json_split,
                data_volume_path=config.data_volume_path,
                device=config.model_config.device,
                max_latents=config.max_latents,
                specific_latent_indices=config.specific_latent_indices,
            )
            print(f"✓ Loaded {len(input_data)} vector entries from master JSON")
        else:
            raise ValueError(f"Unknown evaluation mode: {config.evaluation_mode}")

        # Compute shard bounds
        total_items = len(input_data)
        items_per_shard = (total_items + total_shards - 1) // total_shards
        start_idx = shard_id * items_per_shard
        end_idx = min(start_idx + items_per_shard, total_items)

        shard_data = input_data[start_idx:end_idx]

        print(f"🚀 Starting evaluation shard {shard_id + 1}/{total_shards}")
        print(f"   Processing items {start_idx} to {end_idx - 1} ({len(shard_data)} items)")
        print(f"   Evaluation mode: {config.evaluation_mode}")

        if config.evaluation_mode == "label_dataset":
            results = evaluate_label_dataset(config, shard_data, shard_id)
        elif config.evaluation_mode == "label_generator":
            results = evaluate_label_generator(config, shard_data, shard_id)
        elif config.evaluation_mode == "label_generation_only":
            results = generate_labels_only(config, shard_data, shard_id)
        else:
            raise ValueError(f"Unknown evaluation mode: {config.evaluation_mode}")

        # Save results to Modal volume
        run_id = getattr(config, "run_id", datetime.now().strftime("%Y%m%d_%H%M%S"))
        result_filename = f"eval_results_shard_{shard_id:03d}_of_{total_shards:03d}_{run_id}.json"
        result_path = os.path.join(config.output_volume_path, result_filename)

        os.makedirs(os.path.dirname(result_path), exist_ok=True)

        # Convert config objects to dictionaries for JSON serialization
        results_serializable = results.copy()
        results_serializable["config"] = {
            "model_config": config.model_config.__dict__,
            "reward_config": config.reward_config.__dict__,
            "label_generator_config": config.label_generator_config.__dict__
            if config.label_generator_config
            else None,
            "evaluation_mode": config.evaluation_mode,
            "scale_values": config.scale_values,
            "num_labels_per_scale": config.num_labels_per_scale,
            "num_reward_samples": config.num_reward_samples,
            "label_generation_batch_size": config.label_generation_batch_size,
            "reward_evaluation_batch_size": config.reward_evaluation_batch_size,
            "num_parallel_instances": config.num_parallel_instances,
            "checkpoint_every_n_latents": config.checkpoint_every_n_latents,
            "output_volume_path": config.output_volume_path,
            "run_id": run_id,
        }

        with open(result_path, "w") as f:
            json.dump(results_serializable, f, indent=2)

        # Commit volume changes
        results_volume.commit()

        print(f"✅ Shard {shard_id + 1}/{total_shards} completed - saved to {result_filename}")
        wandb.log({"shard_completed": shard_id, "results_saved": result_path})
        wandb.finish()

        return {
            "shard_id": shard_id,
            "items_processed": len(shard_data),
            "result_file": result_filename,
            "success": True,
        }

    except Exception as e:
        print(f"❌ Error in shard {shard_id + 1}/{total_shards}: {e}")
        print(traceback.format_exc())
        wandb.log({"error": str(e), "shard_failed": shard_id})
        wandb.finish()

        return {"shard_id": shard_id, "error": str(e), "success": False}


@app.local_entrypoint()
def main(config_file: str, num_parallel_instances: int = 1):
    """
    Local entrypoint for evaluating reflective coherence.

    Args:
        config_file: Path to JSON file with evaluation configuration
        num_parallel_instances: Number of parallel Modal instances to run
    """
    
    import json
    from datetime import datetime

    # Generate a unique run ID for this evaluation
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Load configuration
    with open(config_file, "r") as f:
        config_dict = json.load(f)

    # Update config with run settings
    config_dict["num_parallel_instances"] = num_parallel_instances
    config_dict["run_id"] = run_id
    config_json = json.dumps(config_dict)

    print(f"🚀 Starting evaluation with {num_parallel_instances} parallel instances")
    print(f"   Config: {config_file}")
    print(f"   Run ID: {run_id}")

    # Launch parallel evaluation shards
    print(f"🚀 Launching {num_parallel_instances} parallel shards...")

    shard_args = [
        (config_json, shard_id, num_parallel_instances)
        for shard_id in range(num_parallel_instances)
    ]

    shard_results = list(evaluate_reflective_coherence_shard.starmap(shard_args))

    # Process results
    for shard_id, result in enumerate(shard_results):
        print(f"✅ Shard {shard_id + 1} completed")
        if result["success"]:
            print(f"   📁 Result file: {result['result_file']}")
        else:
            print(f"   ❌ Error: {result['error']}")

    # Print summary
    successful_shards = [r for r in shard_results if r["success"]]
    failed_shards = [r for r in shard_results if not r["success"]]

    print("\n📊 Evaluation Summary:")
    print(f"   Total shards: {num_parallel_instances}")
    print(f"   Successful: {len(successful_shards)}")
    print(f"   Failed: {len(failed_shards)}")

    if successful_shards:
        print("\n📁 Results saved to Modal volume 'sae-eval-results'")
        for result in successful_shards:
            print(f"   - {result['result_file']}")
        print("\nTo merge results, run:")
        print(f"   python evals/generation_scoring/merge_shards.py --run-id {run_id} --output merged_results_{run_id}.json")

    if failed_shards:
        print("\n❌ Failed shards:")
        for result in failed_shards:
            print(f"   - Shard {result['shard_id']}: {result['error']}")

    return shard_results


if __name__ == "__main__":
    print("Use Modal to run this script:")
    print("  modal run run_eval_modal.py --config-file configs/example_label_generator.json --num-parallel-instances 4")
