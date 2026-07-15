#!/usr/bin/env python3
"""
Merge evaluation results from multiple shards into a single file.

This script:
1. Downloads specific result files from Modal volume
2. Validates that configs are identical across shards
3. Merges evaluations sections and sorts by latent_index
4. Saves merged result file

Works for both label_dataset and label_generator evaluation modes.
"""

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Dict, List


def run_modal_command(cmd: List[str]) -> str:
    """Run a modal CLI command and return output."""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Modal command failed: {e.stderr}")


def list_modal_volume_files(volume_name: str = "sae-eval-results") -> List[str]:
    """List all files in a Modal volume."""
    output = run_modal_command(["modal", "volume", "ls", volume_name])
    # Parse output to get filenames
    files = [line.strip() for line in output.split("\n") if line.strip()]
    return files


def find_shards_by_run_id(run_id: str) -> List[str]:
    """Find all shard files matching a specific run_id."""
    all_files = list_modal_volume_files()

    # Filter for eval_results files with this run_id (exact match at end)
    matching_files = [
        f
        for f in all_files
        if f.startswith("eval_results_shard_") and f.endswith(f"{run_id}.json")
    ]

    # Sort by shard number
    matching_files.sort()

    return matching_files


def download_specific_files(
    filenames: List[str], temp_dir: str = "temp_results"
) -> List[Path]:
    """Download specific files from Modal volume."""

    print(f"📁 Downloading {len(filenames)} files from Modal volume...")

    for f in filenames:
        print(f"  - {f}")

    # Create temp directory
    temp_path = Path(temp_dir)
    temp_path.mkdir(exist_ok=True)

    # Download each file
    downloaded_files = []
    for filename in filenames:
        print(f"Downloading {filename}...")
        local_path = temp_path / filename

        try:
            run_modal_command(
                [
                    "modal",
                    "volume",
                    "get",
                    "sae-eval-results",
                    filename,
                    str(local_path),
                ]
            )
            downloaded_files.append(local_path)
            print(f"  ✅ Downloaded to {local_path}")
        except Exception as e:
            print(f"  ❌ Failed to download {filename}: {e}")
            # Don't fail completely, but warn about missing files

    if not downloaded_files:
        raise ValueError("No files successfully downloaded")

    return downloaded_files


def validate_configs(results_files: List[Dict]) -> Dict:
    """Validate that all config sections are identical. Returns the common config."""

    if not results_files:
        raise ValueError("No results files provided")

    # Get reference config from first file
    reference_config = results_files[0]["config"]

    # Compare all other configs
    for i, result_file in enumerate(results_files[1:], 1):
        current_config = result_file["config"]

        if current_config != reference_config:
            print("❌ Config mismatch detected!")
            print(
                f"Reference config (file 0): {json.dumps(reference_config, indent=2)}"
            )
            print(
                f"Mismatched config (file {i}): {json.dumps(current_config, indent=2)}"
            )
            raise ValueError(
                f"Config mismatch in file {i} - all shard configs must be identical"
            )

    print(f"✅ All {len(results_files)} configs are identical")
    return reference_config


def merge_evaluations(results_files: List[Dict]) -> List[Dict]:
    """Merge and sort evaluations from all result files."""

    all_evaluations = []

    for i, result_file in enumerate(results_files):
        evaluations = result_file.get("evaluations", [])
        print(f"Shard {i}: {len(evaluations)} evaluations")
        all_evaluations.extend(evaluations)

    print(f"Total evaluations: {len(all_evaluations)}")

    # Sort by latent_index
    all_evaluations.sort(key=lambda x: x["latent_index"])

    print(f"✅ Merged and sorted {len(all_evaluations)} evaluations")
    return all_evaluations


def merge_eval_results(filenames: List[str], output_filename: str) -> bool:
    """Main function to merge evaluation results."""

    print("🔄 Merging evaluation results...")
    print(f"Files: {filenames}")
    print(f"Output: {output_filename}")

    try:
        # Download specific files
        downloaded_files = download_specific_files(filenames)

        if not downloaded_files:
            print("❌ No files downloaded")
            return False

        # Load all JSON files
        print("\n📖 Loading result files...")
        results_files = []

        for file_path in downloaded_files:
            print(f"Loading {file_path}...")
            with open(file_path, "r") as f:
                data = json.load(f)
                results_files.append(data)

        # Validate configs
        print("\n🔍 Validating configs...")
        common_config = validate_configs(results_files)

        # Get evaluation mode
        evaluation_mode = results_files[0].get("evaluation_mode", "unknown")

        # Merge evaluations
        print("\n🔗 Merging evaluations...")
        merged_evaluations = merge_evaluations(results_files)

        # Create merged result
        merged_result = {
            "evaluation_mode": f"{evaluation_mode}_merged",
            "config": common_config,
            "evaluations": merged_evaluations,
            "merge_info": {
                "source_files": [f.name for f in downloaded_files],
                "total_shards_merged": len(results_files),
                "merge_timestamp": os.popen("date").read().strip(),
            },
        }

        # Save merged result
        print(f"\n💾 Saving merged result to {output_filename}...")
        with open(output_filename, "w") as f:
            json.dump(merged_result, f, indent=2)

        print(
            f"✅ Successfully merged {len(results_files)} files into {output_filename}"
        )
        print(f"   Total evaluations: {len(merged_evaluations):,}")
        print(f"   Evaluation mode: {evaluation_mode}")

        # Clean up temp files
        for file_path in downloaded_files:
            file_path.unlink()

        temp_dir = downloaded_files[0].parent
        if temp_dir.exists() and temp_dir.name.startswith("temp_"):
            temp_dir.rmdir()
            print("🧹 Cleaned up temporary files")

        return True

    except Exception as e:
        print(f"❌ Merge failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Merge evaluation results from multiple shards",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Merge by explicit filenames:
  python merge_eval_results.py \
    eval_results_shard_000_of_004_20241006_192834.json \
    eval_results_shard_001_of_004_20241006_192834.json \
    eval_results_shard_002_of_004_20241006_192834.json \
    eval_results_shard_003_of_004_20241006_192834.json \
    --output merged_results.json

  # Merge by run_id (finds all shards from that run automatically):
  python merge_eval_results.py --run-id 20241006_192834 --output merged_results.json
        """,
    )

    parser.add_argument(
        "filenames",
        nargs="*",
        help="Specific filenames to merge (space-separated list)",
    )
    parser.add_argument(
        "--run-id",
        help="Run ID to find all shards from the same run (e.g., 20241006_192834)",
    )
    parser.add_argument(
        "--output",
        default="merged_eval_results.json",
        help="Output filename (default: merged_eval_results.json)",
    )

    args = parser.parse_args()

    # Determine which files to merge
    if args.run_id:
        print(f"🔍 Finding shards with run_id: {args.run_id}")
        filenames = find_shards_by_run_id(args.run_id)

        if not filenames:
            print(f"❌ No shard files found with run_id {args.run_id}")
            print("\n💡 Available files in Modal volume:")
            try:
                all_files = list_modal_volume_files()
                eval_files = [
                    f for f in all_files if f.startswith("eval_results_shard_")
                ]
                for f in eval_files[:20]:  # Show first 20
                    print(f"   - {f}")
                if len(eval_files) > 20:
                    print(f"   ... and {len(eval_files) - 20} more")
            except:
                pass
            exit(1)

        print(f"✅ Found {len(filenames)} shards")
    elif args.filenames:
        filenames = args.filenames
    else:
        print("❌ Must provide either filenames or --run-id")
        parser.print_help()
        exit(1)

    success = merge_eval_results(filenames, args.output)

    if not success:
        exit(1)

    print(f"\n🎉 Merge complete! Results saved to: {args.output}")


if __name__ == "__main__":
    main()
