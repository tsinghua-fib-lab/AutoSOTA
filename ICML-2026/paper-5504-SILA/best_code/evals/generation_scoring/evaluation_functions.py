#!/usr/bin/env python3
"""
Evaluation functions for label dataset and label generator modes.
Separated into a different file to avoid large edits.
"""

import json
import os
from pathlib import Path
from typing import Dict, List

import torch
import wandb
from config import EvaluationConfig
from label_generator import LabelGenerator
from master_json_loader import (
    get_dataset_metadata,
    get_label_dataset_from_master_json,
    get_vectors_from_master_json,
)
from reward_system import SAERewardSystem
from sae_core import ObservableLanguageModel, load_sae
from tqdm.auto import tqdm


def _get_latent_index(item: Dict) -> int:
    """Helper function to get latent index from item, handling both key formats."""
    if "latent_index" in item:
        return item["latent_index"]
    elif "index_in_sae" in item:
        return item["index_in_sae"]
    else:
        raise KeyError("Item must have either 'latent_index' or 'index_in_sae' key")


def save_generated_labels(
    generated_labels_data: List[Dict],
    config: EvaluationConfig,
    output_path: str,
) -> None:
    """
    Save generated labels to a JSON file.
    
    Args:
        generated_labels_data: List of dicts with keys: latent_index, label, scale, label_index
        config: Evaluation configuration
        output_path: Path where the JSON file should be saved
    """
    # Extract SAE type from dataset name for cleaner identification
    dataset_name = config.master_json_dataset_name or "unknown"
    layer = config.master_json_layer if config.master_json_layer is not None else config.model_config.sae_layer_number
    
    # Determine SAE type from dataset name
    sae_type = "unknown"
    if "goodfire" in dataset_name.lower() or "Goodfire" in dataset_name:
        sae_type = "goodfire"
    elif "llamascope" in dataset_name.lower() or "LlamaScope" in dataset_name:
        sae_type = "llamascope"
    
    # Get additional metadata from label generator config
    label_gen_config = config.label_generator_config
    
    # Build metadata dict
    metadata = {
        "dataset_name": dataset_name,
        "layer": layer,
        "sae_type": sae_type,
        "run_id": config.run_id,
        "scale_values": config.scale_values,
        "num_labels_per_scale": config.num_labels_per_scale,
        "evaluation_mode": config.evaluation_mode,
        "split": config.master_json_split,
    }
    
    # Add all label_generator_config properties if present
    if label_gen_config:
        metadata["label_generator_config"] = {
            "num_soft_tokens": label_gen_config.num_soft_tokens,
            "template": label_gen_config.template,
            "max_generation_length": label_gen_config.max_generation_length,
            "which_vector": label_gen_config.which_vector,
            "normalize_vectors": label_gen_config.normalize_vectors,
            "adapter_checkpoint_path": label_gen_config.adapter_checkpoint_path,
            "temperature": label_gen_config.temperature,
            "do_sample": label_gen_config.do_sample,
            "top_p": label_gen_config.top_p,
            "repetition_penalty": label_gen_config.repetition_penalty,
        }
    
    output_data = {
        "metadata": metadata,
        "generated_labels": generated_labels_data
    }
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"📝 Saved {len(generated_labels_data)} generated labels to: {output_path}")


def _check_evaluation_failures(results: Dict) -> None:
    """
    Check evaluation results for failures and raise an error if any samples failed.

    Args:
        results: The results dict from evaluate_label_dataset or evaluate_label_generator

    Raises:
        RuntimeError: If any samples failed
    """
    total_samples = 0
    failed_samples = 0

    for evaluation in results["evaluations"]:
        if "scale_evaluations" in evaluation:
            # Label generator format
            for scale_eval in evaluation["scale_evaluations"]:
                for label_eval in scale_eval["generated_labels"]:
                    for sample in label_eval["reward_samples"]:
                        total_samples += 1
                        if sample["error"] is not None:
                            failed_samples += 1
        elif "reward_samples" in evaluation:
            # Label dataset format
            for sample in evaluation["reward_samples"]:
                total_samples += 1
                if sample["error"] is not None:
                    failed_samples += 1

    if total_samples > 0:
        failure_rate = failed_samples / total_samples
        print("\n📊 Final Evaluation Summary:")
        print(f"   Total reward samples: {total_samples}")
        print(f"   Failed samples: {failed_samples}")
        print(f"   Success rate: {(1 - failure_rate):.1%}")

        # Fail the shard if ANY samples failed
        if failed_samples > 0:
            raise RuntimeError(
                f"Shard failed: {failed_samples} out of {total_samples} samples failed ({failure_rate:.1%}). "
                f"All samples must succeed. Check logs for CUDA OOM or other errors."
            )


def evaluate_label_dataset(
    config: EvaluationConfig, shard_data: List[Dict], shard_id: int
) -> Dict:
    """Evaluate a dataset of pre-existing labels.
    
    Args:
        config: Evaluation configuration
        shard_data: Input data - can be either:
            - Old format: list of dicts with 'latent_index' and 'label' keys
            - New format: loaded from master JSON by main script
        shard_id: Shard ID for parallel execution
    """

    print("🧠 Setting up base model...")

    # Load base model
    cache_dir = os.environ.get("TRANSFORMERS_CACHE", "/cache/huggingface/hub")
    print(f"🗂️  ObservableLanguageModel will use cache: {cache_dir}")

    base_model = ObservableLanguageModel(
        model=config.model_config.name,
        device=config.model_config.device,
        dtype=getattr(torch, config.model_config.dtype),
        cache_dir=cache_dir,
    )

    # Workaround for tokenization bug
    base_model.tokenizer.clean_up_tokenization_spaces = False
    print("Set tokenizer.clean_up_tokenization_spaces to False")

    print("🔬 Setting up reward system...")

    # Initialize reward system with loaded model
    reward_system = SAERewardSystem(
        config.model_config,
        config.reward_config,
        existing_model=base_model._original_model,
        existing_tokenizer=base_model.tokenizer,
    )
    reward_system.setup()

    print("✅ Reward system ready")

    results = {
        "evaluation_mode": "label_dataset",
        "shard_id": shard_id,
        "config": {},  # Will be filled in by main function with proper serialization
        "evaluations": [],  # type: ignore
    }

    # Process in batches for checkpointing
    checkpoint_batch_size = config.checkpoint_every_n_latents
    num_batches = (len(shard_data) + checkpoint_batch_size - 1) // checkpoint_batch_size

    for batch_idx in range(num_batches):
        start_idx = batch_idx * checkpoint_batch_size
        end_idx = min(start_idx + checkpoint_batch_size, len(shard_data))
        batch_data = shard_data[start_idx:end_idx]

        print(
            f"📦 Processing batch {batch_idx + 1}/{num_batches} ({len(batch_data)} items)"
        )

        # Extract labels and latent indices for this batch
        batch_labels = []
        batch_latent_indices = []
        batch_original_items = []

        for item in batch_data:
            latent_index = _get_latent_index(item)
            label = item["label"]

            # Generate multiple reward samples for this label
            for sample_idx in range(config.num_reward_samples):
                batch_labels.append(label)
                batch_latent_indices.append(latent_index)
                batch_original_items.append(
                    {"original_item": item, "sample_index": sample_idx}
                )

        # Compute rewards in batches
        print(f"   Computing {len(batch_labels)} reward samples...")
        reward_results = reward_system.compute_batch_rewards(
            labels=batch_labels,
            target_latent_indices=batch_latent_indices,
            batch_size=config.reward_evaluation_batch_size,
            return_detailed_activations=True,
        )

        # Process results and group by original items
        batch_evaluations = {}

        for i, (reward_result, original_info) in enumerate(
            zip(reward_results, batch_original_items)
        ):
            original_item = original_info["original_item"]
            sample_idx = original_info["sample_index"]

            # Create unique key for this latent/label combination
            latent_idx = _get_latent_index(original_item)
            item_key = f"{latent_idx}_{hash(original_item['label'])}"

            if item_key not in batch_evaluations:
                batch_evaluations[item_key] = {
                    "latent_index": latent_idx,
                    "label": original_item["label"],
                    "reward_samples": [],
                }

            # Add this reward sample
            if isinstance(reward_result, dict) and "error" in reward_result:
                # Handle error case
                sample_result = {
                    "sample_index": sample_idx,
                    "error": reward_result["error"],
                    "per_token_activations": None,
                    "num_tokens": 0,
                    "generated_text": reward_result.get("generated_text"),
                }
            else:
                # Handle success case
                sample_result = {
                    "sample_index": sample_idx,
                    "error": None,
                    "per_token_activations": reward_result["per_token_activations"],
                    "num_tokens": reward_result["num_tokens"],
                    "generated_text": reward_result.get("generated_text"),
                }

            batch_evaluations[item_key]["reward_samples"].append(sample_result)

        # Add batch evaluations to results
        results["evaluations"].extend(list(batch_evaluations.values()))  # type: ignore

        # Log progress
        completed_items = end_idx
        total_items = len(shard_data)
        wandb.log(
            {
                "progress": completed_items / total_items,
                "completed_items": completed_items,
                "current_batch": batch_idx + 1,
                "total_batches": num_batches,
            }
        )

        print(
            f"   ✅ Batch {batch_idx + 1}/{num_batches} completed ({completed_items}/{total_items} items done)"
        )

        # Clean up GPU memory
        if hasattr(reward_system, "clear_gpu_memory"):
            reward_system.clear_gpu_memory()

    # Check for failures and raise error if any found
    _check_evaluation_failures(results)

    return results


def evaluate_label_generator(
    config: EvaluationConfig, shard_data: List[Dict], shard_id: int
) -> Dict:
    """Evaluate using a label generator model.
    
    Args:
        config: Evaluation configuration
        shard_data: Input data - can be either:
            - Old format: list of dicts with 'latent_index' key (vectors loaded from SAE)
            - New format: list of dicts with 'latent_index' and 'sae_vector' keys (from master JSON)
        shard_id: Shard ID for parallel execution
    """

    print("🧠 Setting up base model...")

    # Load base model (shared between label generator and reward system)
    cache_dir = os.environ.get("TRANSFORMERS_CACHE", "/cache/huggingface/hub")
    print(f"🗂️  ObservableLanguageModel will use cache: {cache_dir}")

    base_model = ObservableLanguageModel(
        model=config.model_config.name,
        device=config.model_config.device,
        dtype=getattr(torch, config.model_config.dtype),
        cache_dir=cache_dir,
    )

    print("🔬 Setting up reward system...")

    # Initialize reward system with shared model
    reward_system = SAERewardSystem(
        config.model_config,
        config.reward_config,
        existing_model=base_model._original_model,
        existing_tokenizer=base_model.tokenizer,
    )
    reward_system.setup()

    print("🏷️  Setting up label generator...")

    # Initialize label generator (adapter handles all configuration)
    label_generator = LabelGenerator(
        model_dim=base_model.hidden_size,
        base_model=base_model,
        config=config.label_generator_config,
    )

    # Set to eval mode
    label_generator.eval()

    print("✅ Label generator ready")

    # Check if shard_data already has vectors (new format) or if we need to load SAE (old format)
    using_master_json = len(shard_data) > 0 and "sae_vector" in shard_data[0]
    
    if not using_master_json:
        # Old format: Load SAE to get decoder vectors
        print("🔍 Loading SAE for decoder vectors (old format)...")

        # Determine SAE device: if multi-GPU (device="auto"), use first GPU
        sae_device = "cuda:0" if config.model_config.device == "auto" else config.model_config.device
        
        # Load SAE using SAELens
        sae = load_sae(
            release=config.model_config.sae_release,
            sae_id=config.model_config.sae_id,
            device=sae_device,
        )
    else:
        print("✓ Using vectors from master JSON (new format)")
        sae = None  # Not needed in new format

    results = {
        "evaluation_mode": "label_generator",
        "shard_id": shard_id,
        "config": {},  # Will be filled in by main function with proper serialization
        "evaluations": [],  # type: ignore
    }
    
    # Track all generated labels for saving to separate file
    all_generated_labels = []

    # Process in batches for checkpointing
    checkpoint_batch_size = config.checkpoint_every_n_latents
    num_batches = (len(shard_data) + checkpoint_batch_size - 1) // checkpoint_batch_size

    for batch_idx in range(num_batches):
        start_idx = batch_idx * checkpoint_batch_size
        end_idx = min(start_idx + checkpoint_batch_size, len(shard_data))
        batch_data = shard_data[start_idx:end_idx]

        print(
            f"📦 Processing batch {batch_idx + 1}/{num_batches} ({len(batch_data)} items)"
        )

        # Step 1: Generate ALL labels for this checkpoint batch
        print("   Step 1: Collecting vectors and preparing for batched generation...")
        
        # Check if we should normalize vectors
        normalize_vectors = config.label_generator_config.normalize_vectors if hasattr(config.label_generator_config, 'normalize_vectors') else True
        if normalize_vectors:
            print("   Vector normalization: ENABLED (normalizing before scaling)")
        else:
            print("   Vector normalization: DISABLED (using raw vector magnitudes)")

        # Collect all (scaled_vector, metadata) pairs first
        generation_tasks = []

        for item in batch_data:
            latent_index = _get_latent_index(item)

            # Get SAE vector - either from master JSON or from SAE
            if using_master_json:
                # New format: vector already loaded
                sae_vector = item["sae_vector"]
            else:
                # Old format: extract from SAE
                with torch.no_grad():
                    if config.label_generator_config is None:
                        raise ValueError(
                            "label_generator_config is required for label generator mode"
                        )

                    if config.label_generator_config.which_vector == "decoder":
                        # SAELens: W_dec is [d_sae, d_in], so each decoder vector is a ROW
                        sae_vector = sae.W_dec[latent_index, :]
                    else:
                        # SAELens: W_enc is [d_in, d_sae], so each encoder vector is a COLUMN
                        sae_vector = sae.W_enc[:, latent_index]

            # Normalize vector if requested (before scaling!)
            if normalize_vectors:
                vector_norm = torch.norm(sae_vector)
                if vector_norm > 0:
                    sae_vector = sae_vector / vector_norm

            # Create tasks for each scale and label
            for scale_value in config.scale_values:
                scaled_vector = sae_vector * scale_value
                for label_idx in range(config.num_labels_per_scale):
                    generation_tasks.append(
                        {
                            "scaled_vector": scaled_vector,
                            "latent_index": latent_index,
                            "scale_value": scale_value,
                            "label_index": label_idx,
                        }
                    )

        print(f"   Prepared {len(generation_tasks)} generation tasks")

        # Generate labels in batches
        print(
            f"   Generating labels in batches of {config.label_generation_batch_size}..."
        )
        latent_label_data = []

        for batch_start in tqdm(range(
            0, len(generation_tasks), config.label_generation_batch_size
        )):
            batch_end = min(
                batch_start + config.label_generation_batch_size, len(generation_tasks)
            )
            batch_tasks = generation_tasks[batch_start:batch_end]

            # Stack vectors into a batch tensor
            batch_vectors = torch.stack([task["scaled_vector"] for task in batch_tasks])

            # Generate labels for this batch
            generated_labels = label_generator(batch_vectors)

            # Pair each generated label with its metadata
            for task, label in zip(batch_tasks, generated_labels):
                latent_label_data.append(
                    (
                        task["latent_index"],
                        task["scale_value"],
                        task["label_index"],
                        label,
                    )
                )
                
                # Also track for separate generated labels file
                all_generated_labels.append({
                    "latent_index": task["latent_index"],
                    "label": label,
                    "scale": task["scale_value"],
                    "label_index": task["label_index"],
                })

        print(f"   Generated {len(latent_label_data)} labels total")

        # Step 2: Create batched reward evaluation tasks
        print("   Step 2: Preparing reward evaluation batch...")
        batch_labels = []
        batch_latent_indices = []
        batch_metadata = []  # To map results back

        for latent_index, scale_value, label_idx, label in latent_label_data:
            # Create num_reward_samples copies of each (label, latent) pair
            for sample_idx in range(config.num_reward_samples):
                batch_labels.append(label)
                batch_latent_indices.append(latent_index)
                batch_metadata.append(
                    {
                        "latent_index": latent_index,
                        "scale_value": scale_value,
                        "label_index": label_idx,
                        "label": label,
                        "sample_index": sample_idx,
                    }
                )

        # Step 3: Compute all rewards in one batched call
        print(f"   Step 3: Computing {len(batch_labels)} reward evaluations...")
        reward_results = reward_system.compute_batch_rewards(
            labels=batch_labels,
            target_latent_indices=batch_latent_indices,
            batch_size=config.reward_evaluation_batch_size,
            return_detailed_activations=True,
        )

        # Step 4: Unpack results back into hierarchical structure
        print("   Step 4: Organizing results...")
        # Create nested dict structure: latent_index -> scale_value -> label_index -> [samples]
        organized_results = {}

        for reward_result, metadata in zip(reward_results, batch_metadata):
            latent_idx = metadata["latent_index"]
            scale_val = metadata["scale_value"]
            label_idx = metadata["label_index"]
            label = metadata["label"]
            sample_idx = metadata["sample_index"]

            # Initialize nested structure
            if latent_idx not in organized_results:
                organized_results[latent_idx] = {}
            if scale_val not in organized_results[latent_idx]:
                organized_results[latent_idx][scale_val] = {}
            if label_idx not in organized_results[latent_idx][scale_val]:
                organized_results[latent_idx][scale_val][label_idx] = {
                    "label": label,
                    "reward_samples": [],
                }

            # Add reward sample
            if isinstance(reward_result, dict) and "error" in reward_result:
                sample_result = {
                    "sample_index": sample_idx,
                    "error": reward_result["error"],
                    "per_token_activations": None,
                    "num_tokens": 0,
                    "generated_text": reward_result.get("generated_text"),
                }
            else:
                sample_result = {
                    "sample_index": sample_idx,
                    "error": None,
                    "per_token_activations": reward_result["per_token_activations"],
                    "num_tokens": reward_result["num_tokens"],
                    "generated_text": reward_result.get("generated_text"),
                }

            organized_results[latent_idx][scale_val][label_idx][
                "reward_samples"
            ].append(sample_result)

        # Step 5: Convert to final output format
        batch_evaluations = []
        for latent_idx in sorted(organized_results.keys()):
            latent_eval = {
                "latent_index": latent_idx,
                "scale_evaluations": [],
            }

            for scale_val in sorted(organized_results[latent_idx].keys()):
                scale_eval = {
                    "scale_value": scale_val,
                    "generated_labels": [],
                }

                for label_idx in sorted(
                    organized_results[latent_idx][scale_val].keys()
                ):
                    label_data = organized_results[latent_idx][scale_val][label_idx]
                    scale_eval["generated_labels"].append(
                        {
                            "label": label_data["label"],
                            "label_index": label_idx,
                            "reward_samples": label_data["reward_samples"],
                        }
                    )

                latent_eval["scale_evaluations"].append(scale_eval)

            batch_evaluations.append(latent_eval)

        # Add batch evaluations to results
        results["evaluations"].extend(batch_evaluations)

        # Log progress
        completed_items = end_idx
        total_items = len(shard_data)
        wandb.log(
            {
                "progress": completed_items / total_items,
                "completed_items": completed_items,
                "current_batch": batch_idx + 1,
                "total_batches": num_batches,
            }
        )

        print(
            f"   ✅ Batch {batch_idx + 1}/{num_batches} completed ({completed_items}/{total_items} items done)"
        )

        # Clean up GPU memory
        if hasattr(reward_system, "clear_gpu_memory"):
            reward_system.clear_gpu_memory()

    # Check for failures and raise error if any found
    _check_evaluation_failures(results)
    
    # Save generated labels to separate file
    if all_generated_labels:
        run_id = config.run_id or "unknown"
        dataset_name = config.master_json_dataset_name or config.model_config.sae_release
        layer = config.master_json_layer if config.master_json_layer is not None else config.model_config.sae_layer_number
        
        # Create filename for generated labels
        dataset_short = dataset_name.split("/")[-1] if "/" in dataset_name else dataset_name
        generated_labels_filename = f"generated_labels_{dataset_short}_layer{layer}_shard{shard_id:03d}_{run_id}.json"
        generated_labels_path = os.path.join(config.output_volume_path, generated_labels_filename)
        
        save_generated_labels(all_generated_labels, config, generated_labels_path)
        results["generated_labels_file"] = generated_labels_filename

    return results


def generate_labels_only(
    config: EvaluationConfig, shard_data: List[Dict], shard_id: int
) -> Dict:
    """Generate labels without evaluation (label_generation_only mode).
    
    Args:
        config: Evaluation configuration
        shard_data: Input data with 'latent_index' and 'sae_vector' keys (from master JSON)
        shard_id: Shard ID for parallel execution
    """

    print("🧠 Setting up base model...")

    # Load base model for label generation
    cache_dir = os.environ.get("TRANSFORMERS_CACHE", "/cache/huggingface/hub")
    print(f"🗂️  ObservableLanguageModel will use cache: {cache_dir}")

    base_model = ObservableLanguageModel(
        model=config.model_config.name,
        device=config.model_config.device,
        dtype=getattr(torch, config.model_config.dtype),
        cache_dir=cache_dir,
    )

    print("🏷️  Setting up label generator...")

    # Initialize label generator (adapter handles all configuration)
    label_generator = LabelGenerator(
        model_dim=base_model.hidden_size,
        base_model=base_model,
        config=config.label_generator_config,
    )

    # Set to eval mode
    label_generator.eval()

    print("✅ Label generator ready (generation-only mode)")

    # Track all generated labels
    all_generated_labels = []

    # Process in batches for checkpointing
    checkpoint_batch_size = config.checkpoint_every_n_latents
    num_batches = (len(shard_data) + checkpoint_batch_size - 1) // checkpoint_batch_size

    for batch_idx in range(num_batches):
        start_idx = batch_idx * checkpoint_batch_size
        end_idx = min(start_idx + checkpoint_batch_size, len(shard_data))
        batch_data = shard_data[start_idx:end_idx]

        print(
            f"📦 Processing batch {batch_idx + 1}/{num_batches} ({len(batch_data)} items)"
        )

        # Collect all generation tasks for this batch
        generation_tasks = []
        
        # Check if we should normalize vectors
        normalize_vectors = config.label_generator_config.normalize_vectors if hasattr(config.label_generator_config, 'normalize_vectors') else True
        if normalize_vectors:
            print("   Vector normalization: ENABLED (normalizing before scaling)")
        else:
            print("   Vector normalization: DISABLED (using raw vector magnitudes)")

        for item in batch_data:
            latent_index = _get_latent_index(item)
            sae_vector = item["sae_vector"]

            # Normalize vector if requested (before scaling!)
            if normalize_vectors:
                vector_norm = torch.norm(sae_vector)
                if vector_norm > 0:
                    sae_vector = sae_vector / vector_norm

            # Create tasks for each scale and label
            for scale_value in config.scale_values:
                scaled_vector = sae_vector * scale_value
                for label_idx in range(config.num_labels_per_scale):
                    generation_tasks.append(
                        {
                            "scaled_vector": scaled_vector,
                            "latent_index": latent_index,
                            "scale_value": scale_value,
                            "label_index": label_idx,
                        }
                    )

        print(f"   Prepared {len(generation_tasks)} generation tasks")

        # Generate labels in batches
        print(
            f"   Generating labels in batches of {config.label_generation_batch_size}..."
        )

        for batch_start in tqdm(range(
            0, len(generation_tasks), config.label_generation_batch_size
        )):
            batch_end = min(
                batch_start + config.label_generation_batch_size, len(generation_tasks)
            )
            batch_tasks = generation_tasks[batch_start:batch_end]

            # Stack vectors into a batch tensor
            batch_vectors = torch.stack([task["scaled_vector"] for task in batch_tasks])

            # Generate labels for this batch
            generated_labels = label_generator(batch_vectors)

            # Store each generated label with its metadata
            for task, label in zip(batch_tasks, generated_labels):
                all_generated_labels.append({
                    "latent_index": task["latent_index"],
                    "label": label,
                    "scale": task["scale_value"],
                    "label_index": task["label_index"],
                })

        print(f"   Generated {len(all_generated_labels)} labels so far")

        # Log progress
        completed_items = end_idx
        total_items = len(shard_data)
        wandb.log(
            {
                "progress": completed_items / total_items,
                "completed_items": completed_items,
                "current_batch": batch_idx + 1,
                "total_batches": num_batches,
            }
        )

        print(
            f"   ✅ Batch {batch_idx + 1}/{num_batches} completed ({completed_items}/{total_items} items done)"
        )

    # Save generated labels to file
    run_id = config.run_id or "unknown"
    dataset_name = config.master_json_dataset_name or config.model_config.sae_release
    layer = config.master_json_layer if config.master_json_layer is not None else config.model_config.sae_layer_number
    
    # Create filename for generated labels
    dataset_short = dataset_name.split("/")[-1] if "/" in dataset_name else dataset_name
    generated_labels_filename = f"generated_labels_{dataset_short}_layer{layer}_shard{shard_id:03d}_{run_id}.json"
    generated_labels_path = os.path.join(config.output_volume_path, generated_labels_filename)
    
    save_generated_labels(all_generated_labels, config, generated_labels_path)

    results = {
        "evaluation_mode": "label_generation_only",
        "shard_id": shard_id,
        "config": {},  # Will be filled in by main function
        "num_labels_generated": len(all_generated_labels),
        "generated_labels_file": generated_labels_filename,
    }

    return results
