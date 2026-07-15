#!/usr/bin/env python3
"""
Modal deployment script for SelfIE Bridge Entity Extraction.

This wrapper allows running the bridge extraction experiment on Modal's cloud infrastructure.

Usage:
    modal run run_modal.py --config-path config_toasty_mountain_70.json
    modal run run_modal.py --config-path config.json --question-id 124088
"""

from typing import Optional
import modal

# Create the Modal app
app = modal.App("selfie-bridge-extraction")

# Create the base image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "accelerate",
        "numpy",
        "matplotlib",
        "seaborn",
        "tqdm",
    )
    .add_local_dir(
        local_path=".",
        remote_path="/workspace",
    )
)

# Volume for storing results and bias vectors
results_volume = modal.Volume.from_name("bridge-extraction-results", create_if_missing=True)
cache_volume = modal.Volume.from_name("hf-cache", create_if_missing=True)


@app.function(
    image=image,
    gpu="A100",  # Llama 3.1 8B needs a good GPU
    memory=32 * 1024,  # 32GB memory
    timeout=3 * 3600,  # 3 hour timeout
    volumes={
        "/results": results_volume,
        "/cache": cache_volume,
    },
    secrets=[
        modal.Secret.from_name("huggingface-token"),  # Configure this secret in your Modal account
    ],
)
def run_bridge_extraction(
    config_path: str,
    question_id: Optional[str] = None,
    output_dir_override: Optional[str] = None,
):
    """
    Run the SelfIE bridge extraction experiment on Modal.

    Args:
        config_path: Path to the config JSON file (relative to workspace)
        question_id: Optional question ID to override the one in config
        output_dir_override: Optional output directory override
    """
    import sys
    import os
    import json
    from pathlib import Path
    
    # Set up paths
    sys.path.insert(0, "/workspace")
    os.chdir("/workspace")
    
    # Set cache directories
    cache_dir = "/cache/huggingface/hub"
    os.environ["HF_HOME"] = "/cache/huggingface"
    os.environ["HF_HUB_CACHE"] = cache_dir
    os.environ["TRANSFORMERS_CACHE"] = cache_dir
    
    for cache_directory in ["/cache/huggingface", cache_dir]:
        os.makedirs(cache_directory, exist_ok=True)
    
    print("=" * 60)
    print("SelfIE Bridge Entity Extraction on Modal")
    print("=" * 60)
    print(f"Config: {config_path}")
    
    # Import the main script components
    from run_selfie_bridge_extraction import SelfIEBridgeExtractor
    
    # Load and potentially modify config
    config_full_path = Path(config_path)
    if not config_full_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_full_path, 'r') as f:
        config_data = json.load(f)
    
    # Apply overrides
    if question_id is not None:
        print(f"Overriding question_id: {config_data.get('question_id')} -> {question_id}")
        config_data['question_id'] = question_id
    
    if output_dir_override is not None:
        print(f"Overriding output_dir: {config_data.get('output_dir')} -> {output_dir_override}")
        config_data['output_dir'] = output_dir_override
    
    # Ensure output goes to results volume
    original_output_dir = config_data.get('output_dir', 'results')
    results_output_dir = f"/results/{original_output_dir}"
    config_data['output_dir'] = results_output_dir
    
    # Handle bias vector path if it exists
    if config_data.get('bias_vector_path'):
        original_bias_path = config_data['bias_vector_path']
        # Check if it exists locally first
        if not Path(original_bias_path).exists():
            # Try in results volume
            results_bias_path = f"/results/{original_bias_path}"
            if Path(results_bias_path).exists():
                print(f"Using bias vector from results volume: {results_bias_path}")
                config_data['bias_vector_path'] = results_bias_path
            else:
                print(f"WARNING: Bias vector not found at {original_bias_path} or {results_bias_path}")
                print(f"If this is a trained_bias method, the run will fail.")
    
    # Save modified config temporarily
    temp_config_path = "/tmp/config_modal.json"
    with open(temp_config_path, 'w') as f:
        json.dump(config_data, f, indent=2)
    
    print(f"Question ID: {config_data['question_id']}")
    print(f"Model: {config_data['model_name']}")
    print(f"Method: {config_data['selfie_method']}")
    print(f"Output dir: {results_output_dir}")
    print()
    
    # Reload volume to see any uploaded files
    results_volume.reload()
    
    # Initialize extractor with modified config
    extractor = SelfIEBridgeExtractor(temp_config_path)
    
    # Load question
    dataset_path = config_data.get('dataset_path', 'twohopfact_filtered.json')
    print(f"Loading question from dataset: {dataset_path}")
    question_data = extractor.load_question(dataset_path)
    print(f"Question: {question_data['question']}")
    print(f"Bridge entity (e2): {question_data['e2_value']}")
    print(f"Aliases: {question_data['e2_aliases']}")
    print(f"Correct answer (e3): {question_data['correct_answer']}")
    print()
    
    # Run analysis
    results = extractor.run_analysis(question_data)
    
    # Save results
    output_dir = Path(results_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save results summary (excluding large description data for JSON)
    results_summary = {
        'question_data': results['question_data'],
        'config': results['config'],
        'start_pos': results['start_pos'],
        'layers': results['layers'],
        'token_positions': results['token_positions'],
        'token_texts': results['token_texts'],
        'match_fractions': results['match_fractions']
    }
    
    results_path = output_dir / f"results_q{config_data['question_id']}.json"
    with open(results_path, 'w') as f:
        json.dump(results_summary, f, indent=2)
    print(f"Results summary saved to: {results_path}")
    
    # Save full results with descriptions
    full_results_path = output_dir / f"results_full_q{config_data['question_id']}.json"
    with open(full_results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Full results saved to: {full_results_path}")
    
    # Create heatmap
    heatmap_path = output_dir / f"heatmap_q{config_data['question_id']}.png"
    extractor.create_heatmap(results, str(heatmap_path))
    
    # Commit volume changes
    results_volume.commit()
    
    print()
    print("=" * 60)
    print("✓ Analysis complete!")
    print("=" * 60)
    
    return {
        "question_id": config_data['question_id'],
        "bridge_entity": question_data['e2_value'],
        "method": config_data['selfie_method'],
        "results_path": str(results_path),
        "full_results_path": str(full_results_path),
        "heatmap_path": str(heatmap_path),
    }


@app.local_entrypoint()
def main(
    config_path: str = "config.json",
    question_id: Optional[str] = None,
    output_dir: Optional[str] = None,
):
    """
    Local entrypoint for running bridge extraction on Modal.
    
    Args:
        config_path: Path to config JSON file (default: config.json)
        question_id: Optional question ID to override the one in config
        output_dir: Optional output directory name (will be created in /results/)
    """
    print(f"Starting bridge extraction on Modal with config: {config_path}")
    
    if question_id:
        print(f"Using question ID: {question_id}")
    
    if output_dir:
        print(f"Using output directory: {output_dir}")
    
    result = run_bridge_extraction.remote(
        config_path=config_path,
        question_id=question_id,
        output_dir_override=output_dir,
    )
    
    print("\n" + "=" * 60)
    print("Completed successfully!")
    print("=" * 60)
    print(f"Results: {result}")
    print("\nTo download results from Modal volume:")
    print("  modal volume get bridge-extraction-results <remote_path> <local_path>")
    
    return result


if __name__ == "__main__":
    print("Use Modal to run this script:")
    print("  modal run run_modal.py --config-path config.json")
    print("  modal run run_modal.py --config-path config_toasty_mountain_70.json")
    print("  modal run run_modal.py --config-path config.json --question-id 124088")

