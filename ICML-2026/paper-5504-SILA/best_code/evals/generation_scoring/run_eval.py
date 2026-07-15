#!/usr/bin/env python3
"""
Local version of reflective coherence evaluation (no Modal dependencies).
Runs directly on a GPU machine.
"""

import argparse
import json
import os
import traceback
from datetime import datetime
from pathlib import Path

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


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate reflective coherence locally (no Modal)"
    )
    parser.add_argument(
        "--config-file",
        type=str,
        required=True,
        help="Path to JSON file with evaluation configuration",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./results",
        help="Directory to save results (default: ./results)",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="HuggingFace cache directory (default: use HF_HOME or default cache)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Base directory where .pt files are located (default: workspace root)",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable Weights & Biases logging",
    )
    parser.add_argument(
        "--num-parallel-instances",
        type=int,
        default=None,
        help="NOT SUPPORTED for local runs (Modal only)",
    )

    args = parser.parse_args()

    # Check for unsupported parallel instances argument
    if args.num_parallel_instances is not None:
        print("❌ ERROR: --num-parallel-instances is not supported for local runs.")
        print("   This feature is only available when running on Modal.")
        print("   For local runs, the evaluation runs on a single GPU.")
        print("")
        print("   To run with parallel instances, use:")
        print("   modal run run_eval_modal.py --input-file ... --config-file ... --num-parallel-instances 4")
        return 1

    # Set up cache directories
    if args.cache_dir:
        cache_base = args.cache_dir
    else:
        cache_base = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    
    cache_hub = os.path.join(cache_base, "hub")
    cache_datasets = os.path.join(cache_base, "datasets")

    # Create cache directories if they don't exist
    os.makedirs(cache_hub, exist_ok=True)
    os.makedirs(cache_datasets, exist_ok=True)

    # Set environment variables for HuggingFace caching
    os.environ["HF_HOME"] = cache_base
    os.environ["TRANSFORMERS_CACHE"] = cache_hub
    os.environ["HF_DATASETS_CACHE"] = cache_datasets
    os.environ["HF_HUB_CACHE"] = cache_hub

    print(f"🗂️  Using HuggingFace cache: {cache_base}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Load configuration
        print(f"📋 Loading config from: {args.config_file}")
        with open(args.config_file, "r") as f:
            config_dict = json.load(f)

        # Generate run_id if not present
        if "run_id" not in config_dict or config_dict["run_id"] is None:
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            config_dict["run_id"] = run_id
            print(f"🆔 Generated new run_id: {run_id}")
        else:
            run_id = config_dict["run_id"]
            print(f"🆔 Using existing run_id: {run_id}")

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
        config_dict_fixed["run_id"] = run_id

        # Override output_volume_path to local directory
        config_dict_fixed["output_volume_path"] = str(output_dir)

        config = EvaluationConfig(**config_dict_fixed)

        print(f"✓ Config loaded: {config.evaluation_mode} mode")
        print(f"✓ Layer: {config.model_config.sae_layer_number}")

        # Set data_volume_path if not already set in config
        if args.data_dir:
            config_dict_fixed["data_volume_path"] = args.data_dir
        elif "data_volume_path" not in config_dict_fixed or config_dict_fixed["data_volume_path"] == "/sae-data":
            # Default to workspace root for local runs
            config_dict_fixed["data_volume_path"] = os.getcwd()
        
        # Reload config with data_volume_path set
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
            # Load existing labels from master JSON
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
            # Load vectors from master JSON
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

        # Initialize WandB
        if not args.no_wandb:
            wandb.init(
                project="sae-reflective-coherence-eval",
                name=f"eval_local_{run_id}",
                config=config_dict,
            )
            print("✓ WandB initialized")
        else:
            # Set up dummy wandb for offline mode
            os.environ["WANDB_MODE"] = "disabled"
            wandb.init(mode="disabled")
            print("✓ WandB disabled (running offline)")

        # Run evaluation
        print(f"\n🚀 Starting evaluation...")
        print(f"   Mode: {config.evaluation_mode}")
        print(f"   Items: {len(input_data)}")
        
        if config.evaluation_mode == "label_dataset":
            results = evaluate_label_dataset(config, input_data, shard_id=0)
        elif config.evaluation_mode == "label_generator":
            results = evaluate_label_generator(config, input_data, shard_id=0)
        elif config.evaluation_mode == "label_generation_only":
            results = generate_labels_only(config, input_data, shard_id=0)
        else:
            raise ValueError(f"Unknown evaluation mode: {config.evaluation_mode}")

        # Save results to local file
        layer_num = config.model_config.sae_layer_number
        result_filename = f"eval_results_layer_{layer_num}_{run_id}.json"
        result_path = output_dir / result_filename

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
            "output_volume_path": str(output_dir),
            "run_id": run_id,
        }

        with open(result_path, "w") as f:
            json.dump(results_serializable, f, indent=2)

        print(f"\n✅ Evaluation completed!")
        print(f"📁 Results saved to: {result_path}")

        if not args.no_wandb:
            wandb.log({"evaluation_completed": True, "results_saved": str(result_path)})
            wandb.finish()

        return 0

    except Exception as e:
        print(f"\n❌ Error during evaluation: {e}")
        print(traceback.format_exc())
        
        # Only log to wandb if it was successfully initialized
        if not args.no_wandb and wandb.run is not None:
            try:
                wandb.log({"error": str(e), "evaluation_failed": True})
                wandb.finish()
            except Exception:
                pass  # Silently fail if wandb logging fails
        
        return 1


if __name__ == "__main__":
    exit(main())

