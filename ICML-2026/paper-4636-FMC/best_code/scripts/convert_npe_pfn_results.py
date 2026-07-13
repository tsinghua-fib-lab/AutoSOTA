#!/usr/bin/env python3
"""Convert NPE-PFN results to the comparison benchmark format."""

import json
from pathlib import Path
from datetime import datetime

def convert_npe_pfn_results():
    """Convert NPE-PFN batched results to comparison benchmark format."""

    npe_pfn_results_dir = Path("/home/pruhlman/Projects/npe-pfn/results/cluster_results")
    output_base = Path("/home/pruhlman/Projects/ropefm/results/comparison_benchmark/shared/baselines")

    # Seed mapping: npe-pfn uses 0,1,2,3,4, we map first 3 to 33,43,53
    seed_mapping = {0: 33, 1: 43, 2: 53}

    # Load and convert each task's results
    files_to_process = [
        ("batched_results_final.json", "pendulum"),
        ("results_wind_tunnel_batched_partial.json", "wind_tunnel"),
    ]

    for filename, expected_task in files_to_process:
        filepath = npe_pfn_results_dir / filename
        if not filepath.exists():
            print(f"Warning: {filepath} not found")
            continue

        with open(filepath) as f:
            data = json.load(f)

        for task, ncal_data in data.items():
            if task != expected_task:
                print(f"Note: Task mismatch in {filename}: expected {expected_task}, got {task}")

            for ncal_str, seed_data in ncal_data.items():
                ncal = int(ncal_str)

                for seed_str, metrics in seed_data.items():
                    seed_idx = int(seed_str)

                    # Only process seeds we have a mapping for
                    if seed_idx not in seed_mapping:
                        continue

                    seed = seed_mapping[seed_idx]

                    # Create output directory
                    output_dir = output_base / task / f"seed_{seed}" / "npe_pfn" / f"ncal_{ncal}"
                    output_dir.mkdir(parents=True, exist_ok=True)

                    # Convert to expected format
                    converted_metrics = {
                        "status": "evaluated",
                        "metrics": {
                            "joint_c2st": metrics["joint_c2st"],
                            "joint_mmd": metrics.get("mmd", metrics.get("joint_mmd")),
                            "joint_wasserstein": metrics.get("wasserstein", metrics.get("joint_wasserstein")),
                        },
                        "timestamp": metrics.get("timestamp", datetime.now().isoformat()),
                        "num_test_observations": 5000,
                        "source": "npe_pfn"
                    }

                    # Write metrics file
                    output_file = output_dir / "metrics.json"
                    with open(output_file, 'w') as f:
                        json.dump(converted_metrics, f, indent=2)

                    print(f"Created: {output_file}")


if __name__ == "__main__":
    convert_npe_pfn_results()
    print("\nDone! NPE-PFN results converted to comparison benchmark format.")
