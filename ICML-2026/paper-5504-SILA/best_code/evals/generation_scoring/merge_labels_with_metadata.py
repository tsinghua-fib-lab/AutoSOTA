#!/tmp/venv/bin/python
"""
Merge generated labels with original dataset metadata.

This script takes generated labels (which only have combined dataset indices 0-999)
and enriches them with the original metadata (layer, global_index, etc.) from the
fifty-thousand-things dataset.

Usage:
    python merge_labels_with_metadata.py \
        --labels results/qwen_scaling/72b_trained/generated_labels_*.json \
        --metadata outputs/qwen_scaling/qwen25_72b_instruct_fifty_thousand_things_combined_val_1000_metadata.json \
        --output results/qwen_scaling/72b_trained/labels_with_metadata.json
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List


def load_generated_labels(labels_path: str) -> Dict:
    """Load generated labels JSON file."""
    with open(labels_path, "r") as f:
        data = json.load(f)
    return data


def load_metadata(metadata_path: str) -> List[Dict]:
    """Load metadata JSON file (list of metadata entries, one per combined dataset index)."""
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    return metadata


def merge_labels_with_metadata(
    generated_labels_data: Dict, metadata: List[Dict]
) -> Dict:
    """
    Merge generated labels with original dataset metadata.

    Args:
        generated_labels_data: Generated labels data with 'metadata' and 'generated_labels'
        metadata: List of metadata dicts, indexed by combined dataset index

    Returns:
        Enriched data with original indices added to each label
    """
    # Extract the generated labels list
    generated_labels = generated_labels_data.get("generated_labels", [])

    # Enrich each label with original metadata
    enriched_labels = []
    for label_entry in generated_labels:
        latent_index = label_entry.get("latent_index")

        # Get the corresponding metadata
        if latent_index >= len(metadata):
            print(
                f"Warning: latent_index {latent_index} exceeds metadata length {len(metadata)}, skipping"
            )
            continue

        original_metadata = metadata[latent_index]

        # Create enriched entry with both combined and original indices
        enriched_entry = {
            "combined_index": latent_index,  # Index in the combined dataset (0-999)
            "original_layer": original_metadata.get("original_layer"),
            "original_global_index": original_metadata.get("original_global_index"),
            "original_local_val_index": original_metadata.get(
                "original_local_val_index"
            ),
            "split": original_metadata.get("split"),
            "label": label_entry.get("label"),
            "scale": label_entry.get("scale"),
            "label_index": label_entry.get("label_index"),
        }

        enriched_labels.append(enriched_entry)

    # Create output data with enriched labels
    output_data = {
        "metadata": generated_labels_data.get("metadata", {}),
        "enriched_labels": enriched_labels,
        "num_labels": len(enriched_labels),
    }

    return output_data


def main():
    parser = argparse.ArgumentParser(
        description="Merge generated labels with original dataset metadata"
    )
    parser.add_argument(
        "--labels",
        type=str,
        required=True,
        help="Path to generated labels JSON file",
    )
    parser.add_argument(
        "--metadata",
        type=str,
        required=True,
        help="Path to metadata JSON file (from data prep)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to save enriched labels JSON",
    )

    args = parser.parse_args()

    # Load inputs
    print(f"📥 Loading generated labels from: {args.labels}")
    generated_labels_data = load_generated_labels(args.labels)
    print(
        f"   Found {len(generated_labels_data.get('generated_labels', []))} generated labels"
    )

    print(f"📥 Loading metadata from: {args.metadata}")
    metadata = load_metadata(args.metadata)
    print(f"   Found {len(metadata)} metadata entries")

    # Merge
    print("🔗 Merging labels with metadata...")
    enriched_data = merge_labels_with_metadata(generated_labels_data, metadata)

    # Save output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"💾 Saving enriched labels to: {args.output}")
    with open(output_path, "w") as f:
        json.dump(enriched_data, f, indent=2)

    print(f"✅ Done! Enriched {enriched_data['num_labels']} labels")

    # Print example entries
    print("\n📋 Example enriched entries:")
    for i, entry in enumerate(enriched_data["enriched_labels"][:3]):
        print(f"\nEntry {i}:")
        print(f"  Combined index: {entry['combined_index']}")
        print(f"  Original layer: {entry['original_layer']}")
        print(f"  Original global index: {entry['original_global_index']}")
        print(f"  Label: {entry['label']}")
        print(f"  Scale: {entry['scale']}")


if __name__ == "__main__":
    main()
