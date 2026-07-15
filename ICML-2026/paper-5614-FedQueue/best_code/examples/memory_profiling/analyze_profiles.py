#!/usr/bin/env python3
"""
Script to analyze memory profiles and generate optimization recommendations
"""

import os
import argparse
import subprocess
from pathlib import Path


def analyze_profile(profile_path: str, output_dir: str):
    """Analyze a single memory profile"""
    profile_name = Path(profile_path).stem

    # Generate flamegraph
    flamegraph_path = f"{output_dir}/{profile_name}_flamegraph.html"
    subprocess.run(["memray", "flamegraph", "--output", flamegraph_path, profile_path])

    # Generate summary statistics
    summary_path = f"{output_dir}/{profile_name}_summary.txt"
    with open(summary_path, "w") as f:
        subprocess.run(["memray", "summary", profile_path], stdout=f)

    # Generate table with top allocations
    table_path = f"{output_dir}/{profile_name}_table.txt"
    with open(table_path, "w") as f:
        subprocess.run(["memray", "table", profile_path], stdout=f)

    print(f"Generated analysis for {profile_name}:")
    print(f"  - Flamegraph: {flamegraph_path}")
    print(f"  - Summary: {summary_path}")
    print(f"  - Table: {table_path}")


def compare_profiles(profile_dir: str):
    """Compare memory usage between server and clients"""
    profiles = list(Path(profile_dir).glob("*.bin"))

    if not profiles:
        print(f"No .bin files found in {profile_dir}")
        return

    output_dir = f"{profile_dir}/analysis"
    os.makedirs(output_dir, exist_ok=True)

    print(f"Found {len(profiles)} profiles to analyze:")
    for profile in profiles:
        print(f"  - {profile.name}")

    print("\nAnalyzing profiles...")
    for profile in profiles:
        analyze_profile(str(profile), output_dir)

    print(f"\nAnalysis complete! Results saved to: {output_dir}")
    print("\nRecommendations will be based on the generated reports.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze memory profiles")
    parser.add_argument("profile_dir", help="Directory containing .bin profile files")

    args = parser.parse_args()
    compare_profiles(args.profile_dir)
