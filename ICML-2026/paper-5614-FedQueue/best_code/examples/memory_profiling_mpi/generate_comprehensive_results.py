#!/usr/bin/env python3
"""
Comprehensive Results Generation Script for APPFL MPI Memory Optimization Experiments

This script generates detailed analysis and comparison results similar to the gRPC memory profiling,
but adapted for MPI-specific federated learning scenarios.
"""

import os
import sys
import json
import subprocess
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Optional
from dataclasses import dataclass
from datetime import datetime


@dataclass
class MPIProfileMetrics:
    """Container for MPI memory profile metrics."""

    total_allocated: float  # in MB
    peak_memory: float  # in MB
    total_allocations: int
    avg_allocation_size: float  # in bytes
    allocation_rate: float  # allocations per second
    profile_duration: float  # in seconds
    profile_file: str
    rank: str
    profile_type: str  # 'original' or 'optimized'
    role: str  # 'server' or 'client'


class MPIResultsGenerator:
    """Comprehensive results generator for MPI memory profiling experiments."""

    def __init__(self, profiles_dir: str, output_dir: str = None):
        self.profiles_dir = Path(profiles_dir)
        self.output_dir = (
            Path(output_dir) if output_dir else self.profiles_dir / "analysis"
        )
        self.output_dir.mkdir(exist_ok=True)

        self.profiles = self._discover_profiles()
        self.metrics = []

        # Set up plotting style
        plt.style.use("default")
        sns.set_palette("husl")

    def _discover_profiles(self) -> Dict[str, Dict[str, str]]:
        """Discover all profile files organized by rank and type."""
        profiles = {}

        for profile_file in self.profiles_dir.glob("*.bin"):
            filename = profile_file.name

            # Parse filename: mpi_rank_X_TYPE_memory_profile.bin
            if filename.startswith("mpi_rank_") and filename.endswith(
                "_memory_profile.bin"
            ):
                parts = filename.replace("_memory_profile.bin", "").split("_")
                if len(parts) >= 4:  # mpi_rank_X_type
                    rank = parts[2]
                    profile_type = parts[3]  # original or optimized

                    if rank not in profiles:
                        profiles[rank] = {}
                    profiles[rank][profile_type] = str(profile_file)

        return profiles

    def _extract_metrics(
        self, profile_path: str, rank: str, profile_type: str
    ) -> Optional[MPIProfileMetrics]:
        """Extract detailed metrics from a memory profile using python -m memray."""
        # Memray --json writes to a file, not stdout. Try to handle this properly.
        try:
            import os

            # First try: Check if memray writes JSON to automatic filename
            auto_json_file = f"{profile_path}.json"

            # Try running memray stats with --json (it may write to auto filename)
            result = subprocess.run(
                ["python", "-m", "memray", "stats", profile_path, "--json"],
                capture_output=True,
                text=True,
                timeout=60,
            )

            # Check if JSON file was created automatically
            if os.path.exists(auto_json_file):
                try:
                    with open(auto_json_file) as f:
                        stats_data = json.load(f)

                    # Clean up the JSON file
                    os.unlink(auto_json_file)

                    # Determine role (server = rank 0, clients = rank > 0)
                    role = "server" if rank == "0" else "client"

                    # Extract key metrics
                    return MPIProfileMetrics(
                        total_allocated=self._bytes_to_mb(
                            stats_data.get("total_memory_allocated", 0)
                        ),
                        peak_memory=self._bytes_to_mb(stats_data.get("peak_memory", 0)),
                        total_allocations=stats_data.get("total_allocations", 0),
                        avg_allocation_size=stats_data.get(
                            "average_allocation_size", 0
                        ),
                        allocation_rate=stats_data.get("allocation_rate", 0),
                        profile_duration=stats_data.get("duration", 0),
                        profile_file=profile_path,
                        rank=rank,
                        profile_type=profile_type,
                        role=role,
                    )
                except (json.JSONDecodeError, KeyError) as e:
                    print(f"JSON file parsing failed for {profile_path}: {e}")
                    # Clean up and fall back to text parsing
                    if os.path.exists(auto_json_file):
                        os.unlink(auto_json_file)
                    return self._parse_text_stats(profile_path, rank, profile_type)
            else:
                # No JSON file created, check if it wrote to a different location
                # Look for pattern in output
                if result.stdout and "Wrote" in result.stdout:
                    lines = result.stdout.split("\n")
                    for line in lines:
                        if line.startswith("Wrote ") and line.endswith(".json"):
                            json_path = line.replace("Wrote ", "").strip()
                            if os.path.exists(json_path):
                                try:
                                    with open(json_path) as f:
                                        stats_data = json.load(f)

                                    # Clean up the JSON file
                                    os.unlink(json_path)

                                    role = "server" if rank == "0" else "client"

                                    return MPIProfileMetrics(
                                        total_allocated=self._bytes_to_mb(
                                            stats_data.get("total_memory_allocated", 0)
                                        ),
                                        peak_memory=self._bytes_to_mb(
                                            stats_data.get("peak_memory", 0)
                                        ),
                                        total_allocations=stats_data.get(
                                            "total_allocations", 0
                                        ),
                                        avg_allocation_size=stats_data.get(
                                            "average_allocation_size", 0
                                        ),
                                        allocation_rate=stats_data.get(
                                            "allocation_rate", 0
                                        ),
                                        profile_duration=stats_data.get("duration", 0),
                                        profile_file=profile_path,
                                        rank=rank,
                                        profile_type=profile_type,
                                        role=role,
                                    )
                                except Exception as e:
                                    print(f"Failed to parse JSON from {json_path}: {e}")
                                    if os.path.exists(json_path):
                                        os.unlink(json_path)

                # JSON approach failed, fall back to text parsing
                return self._parse_text_stats(profile_path, rank, profile_type)

        except Exception as e:
            print(f"JSON extraction failed for {profile_path}: {e}")
            return self._parse_text_stats(profile_path, rank, profile_type)

    def _parse_text_stats(
        self, profile_path: str, rank: str, profile_type: str
    ) -> Optional[MPIProfileMetrics]:
        """Fallback text parsing for python -m memray stats output."""
        try:
            # Use a more robust approach to avoid broken pipe issues
            result = subprocess.run(
                ["python", "-m", "memray", "stats", profile_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=60,
            )

            if result.returncode == 0 and result.stdout:
                lines = result.stdout.split("\n")

                total_allocated = 0
                peak_memory = 0
                total_allocations = 0
                duration = 10  # Default duration to avoid division by zero

                for line in lines:
                    line = line.strip()
                    if not line:
                        continue

                    # Look for emoji-prefixed stats (current memray format)
                    if "📦 Total memory allocated:" in line:
                        total_allocated = self._parse_memory_value(
                            line.replace("📦", "").strip()
                        )
                    elif "📈 Peak memory usage:" in line:
                        peak_memory = self._parse_memory_value(
                            line.replace("📈", "").strip()
                        )
                    elif "📏 Total allocations:" in line:
                        total_allocations = self._parse_int_value(
                            line.replace("📏", "").strip()
                        )
                    elif "Duration:" in line:
                        duration = self._parse_duration_value(line)
                    # Fallback to older format without emojis
                    elif "Total memory allocated:" in line:
                        total_allocated = self._parse_memory_value(line)
                    elif (
                        "Peak memory usage:" in line or "Peak memory allocated:" in line
                    ):
                        peak_memory = self._parse_memory_value(line)
                    elif "Total allocations:" in line:
                        total_allocations = self._parse_int_value(line)

                # If we got valid data, calculate derived metrics
                if total_allocated > 0 or total_allocations > 0:
                    avg_allocation_size = (
                        (total_allocated * 1024 * 1024 / total_allocations)
                        if total_allocations > 0
                        else 0
                    )
                    allocation_rate = (
                        total_allocations / duration if duration > 0 else 0
                    )
                    role = "server" if rank == "0" else "client"

                    return MPIProfileMetrics(
                        total_allocated=total_allocated,
                        peak_memory=peak_memory,
                        total_allocations=total_allocations,
                        avg_allocation_size=avg_allocation_size,
                        allocation_rate=allocation_rate,
                        profile_duration=duration,
                        profile_file=profile_path,
                        rank=rank,
                        profile_type=profile_type,
                        role=role,
                    )
                else:
                    print(f"No valid metrics found in text output for {profile_path}")
            else:
                print(
                    f"memray stats text command failed for {profile_path}: {result.stderr}"
                )

        except subprocess.TimeoutExpired:
            print(f"memray stats text command timed out for {profile_path}")
        except Exception as e:
            print(f"Warning: Text parsing failed for {profile_path}: {e}")

        return None

    def _bytes_to_mb(self, bytes_value: int) -> float:
        """Convert bytes to megabytes."""
        return bytes_value / (1024 * 1024)

    def _parse_memory_value(self, line: str) -> float:
        """Parse memory value from memray output (e.g., '150.2MB' or '150.2 MB' -> 150.2)."""
        try:
            import re

            # Look for patterns like "150.2MB", "150.2 MB", "1.5GB", "1,234.5MB", etc.
            # Remove emoji and extra whitespace
            cleaned_line = re.sub(r"[^\d.,\s\w]", " ", line)

            # Pattern to match number + unit (with or without space)
            pattern = r"(\d+(?:,\d+)*(?:\.\d+)?)\s*(GB|MB|KB|B|BYTES)\b"
            match = re.search(pattern, cleaned_line.upper())

            if match:
                value_str, unit = match.groups()
                value = float(value_str.replace(",", ""))

                if unit == "GB":
                    return value * 1024
                elif unit == "MB":
                    return value
                elif unit == "KB":
                    return value / 1024
                elif unit in ["B", "BYTES"]:
                    return value / (1024 * 1024)

            # Fallback: try to find any float/int value and assume MB
            number_match = re.search(r"(\d+(?:,\d+)*(?:\.\d+)?)", cleaned_line)
            if number_match:
                return float(number_match.group(1).replace(",", ""))

            return 0.0
        except Exception as e:
            print(f"Memory parsing error for line '{line}': {e}")
            return 0.0

    def _parse_int_value(self, line: str) -> int:
        """Parse integer value from memray output."""
        try:
            import re

            # Remove emojis and clean the line
            cleaned_line = re.sub(r"[^\d,\s\w:]", " ", line)

            # Look for comma-separated numbers
            number_match = re.search(r"(\d+(?:,\d+)*)", cleaned_line)
            if number_match:
                return int(number_match.group(1).replace(",", ""))
            return 0
        except Exception as e:
            print(f"Integer parsing error for line '{line}': {e}")
            return 0

    def _parse_duration_value(self, line: str) -> float:
        """Parse duration value from memray output."""
        try:
            # Look for patterns like "12.5s" or "1.2 seconds"
            parts = line.split()
            for part in parts:
                if "s" in part or "sec" in part.lower():
                    value_str = (
                        part.replace("s", "")
                        .replace("sec", "")
                        .replace("onds", "")
                        .replace(",", "")
                    )
                    return float(value_str)
            return 0.0
        except:  # noqa E722
            return 0.0

    def extract_all_metrics(self):
        """Extract metrics from all discovered profiles."""
        print("Extracting metrics from all profiles...")

        for rank, profile_types in self.profiles.items():
            for profile_type, profile_path in profile_types.items():
                metrics = self._extract_metrics(profile_path, rank, profile_type)
                if metrics:
                    self.metrics.append(metrics)
                    print(f"  ✓ Extracted metrics for rank {rank} ({profile_type})")
                else:
                    print(
                        f"  ✗ Failed to extract metrics for rank {rank} ({profile_type})"
                    )

    def generate_flamegraphs(self):
        """Generate flamegraph HTML files for all profiles."""
        print("\nGenerating flamegraphs for all profiles...")

        flamegraphs_dir = self.output_dir / "flamegraphs"
        flamegraphs_dir.mkdir(exist_ok=True)

        for rank, profile_types in self.profiles.items():
            for profile_type, profile_path in profile_types.items():
                profile_name = f"mpi_rank_{rank}_{profile_type}"
                html_file = flamegraphs_dir / f"{profile_name}_flamegraph.html"

                try:
                    result = subprocess.run(
                        [
                            "python",
                            "-m",
                            "memray",
                            "flamegraph",
                            profile_path,
                            "-o",
                            str(html_file),
                        ],
                        capture_output=True,
                        text=True,
                        timeout=120,
                    )

                    if result.returncode == 0:
                        print(f"  ✓ Generated flamegraph: {html_file.name}")
                    else:
                        print(
                            f"  ✗ Failed to generate flamegraph for {profile_name}: {result.stderr}"
                        )

                except subprocess.TimeoutExpired:
                    print(f"  ✗ Flamegraph generation timed out for {profile_name}")
                except Exception as e:
                    print(f"  ✗ Error generating flamegraph for {profile_name}: {e}")

    def generate_summary_reports(self):
        """Generate detailed summary reports for all profiles."""
        print("\nGenerating summary reports...")

        summaries_dir = self.output_dir / "summaries"
        summaries_dir.mkdir(exist_ok=True)

        for rank, profile_types in self.profiles.items():
            for profile_type, profile_path in profile_types.items():
                profile_name = f"mpi_rank_{rank}_{profile_type}"
                summary_file = summaries_dir / f"{profile_name}_summary.txt"

                try:
                    with open(summary_file, "w") as f:
                        result = subprocess.run(
                            ["python", "-m", "memray", "stats", profile_path],
                            stdout=f,
                            text=True,
                            timeout=60,
                        )

                    if result.returncode == 0:
                        print(f"  ✓ Generated summary: {summary_file.name}")
                    else:
                        print(f"  ✗ Failed to generate summary for {profile_name}")

                except Exception as e:
                    print(f"  ✗ Error generating summary for {profile_name}: {e}")

    def generate_optimization_comparison_plots(self):
        """Generate comprehensive comparison plots between original and optimized versions."""
        if not self.metrics:
            print("No metrics available for plotting.")
            return

        print("\nGenerating optimization comparison plots...")

        # Create DataFrame for easier plotting
        df_data = []
        for metric in self.metrics:
            df_data.append(
                {
                    "rank": metric.rank,
                    "role": metric.role,
                    "type": metric.profile_type,
                    "total_allocated_mb": metric.total_allocated,
                    "peak_memory_mb": metric.peak_memory,
                    "total_allocations": metric.total_allocations,
                    "avg_allocation_size_bytes": metric.avg_allocation_size,
                    "allocation_rate": metric.allocation_rate,
                    "duration_seconds": metric.profile_duration,
                }
            )

        df = pd.DataFrame(df_data)

        if df.empty:
            print("No data available for plotting.")
            return

        # Set up the plotting environment
        plt.style.use("default")
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle(
            "APPFL MPI Memory Optimization Comparison", fontsize=16, fontweight="bold"
        )

        # 1. Total Memory Allocated Comparison
        ax = axes[0, 0]
        if "total_allocated_mb" in df.columns:
            sns.barplot(data=df, x="rank", y="total_allocated_mb", hue="type", ax=ax)
            ax.set_title("Total Memory Allocated (MB)")
            ax.set_xlabel("MPI Rank")
            ax.set_ylabel("Memory (MB)")

            # Add value labels on bars
            for container in ax.containers:
                ax.bar_label(container, fmt="%.1f")

        # 2. Peak Memory Usage Comparison
        ax = axes[0, 1]
        if "peak_memory_mb" in df.columns:
            sns.barplot(data=df, x="rank", y="peak_memory_mb", hue="type", ax=ax)
            ax.set_title("Peak Memory Usage (MB)")
            ax.set_xlabel("MPI Rank")
            ax.set_ylabel("Memory (MB)")

            # Add value labels on bars
            for container in ax.containers:
                ax.bar_label(container, fmt="%.1f")

        # 3. Total Allocations Comparison
        ax = axes[0, 2]
        if "total_allocations" in df.columns:
            sns.barplot(data=df, x="rank", y="total_allocations", hue="type", ax=ax)
            ax.set_title("Total Memory Allocations")
            ax.set_xlabel("MPI Rank")
            ax.set_ylabel("Number of Allocations")

            # Format y-axis to show values in K/M
            ax.ticklabel_format(style="scientific", axis="y", scilimits=(0, 0))

        # 4. Average Allocation Size Comparison
        ax = axes[1, 0]
        if "avg_allocation_size_bytes" in df.columns:
            sns.barplot(
                data=df, x="rank", y="avg_allocation_size_bytes", hue="type", ax=ax
            )
            ax.set_title("Average Allocation Size (Bytes)")
            ax.set_xlabel("MPI Rank")
            ax.set_ylabel("Bytes")

            # Format y-axis for better readability
            ax.ticklabel_format(style="scientific", axis="y", scilimits=(0, 0))

        # 5. Allocation Rate Comparison
        ax = axes[1, 1]
        if "allocation_rate" in df.columns:
            sns.barplot(data=df, x="rank", y="allocation_rate", hue="type", ax=ax)
            ax.set_title("Allocation Rate (allocs/second)")
            ax.set_xlabel("MPI Rank")
            ax.set_ylabel("Allocations per Second")

        # 6. Duration Comparison
        ax = axes[1, 2]
        if "duration_seconds" in df.columns:
            sns.barplot(data=df, x="rank", y="duration_seconds", hue="type", ax=ax)
            ax.set_title("Profile Duration (seconds)")
            ax.set_xlabel("MPI Rank")
            ax.set_ylabel("Duration (s)")

            # Add value labels on bars
            for container in ax.containers:
                ax.bar_label(container, fmt="%.1f")

        plt.tight_layout()

        # Save the plot
        plot_file = self.output_dir / "mpi_memory_optimization_comparison.png"
        plt.savefig(plot_file, dpi=300, bbox_inches="tight")
        print(f"  ✓ Generated comparison plot: {plot_file.name}")
        plt.close()

        # Generate role-based comparison (Server vs Clients)
        self._generate_role_based_comparison(df)

        # Generate optimization impact analysis
        self._generate_optimization_impact_analysis(df)

    def _generate_role_based_comparison(self, df: pd.DataFrame):
        """Generate comparison plots between server and client roles."""
        print("  Generating role-based comparison plots...")

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(
            "APPFL MPI Server vs Client Memory Usage", fontsize=14, fontweight="bold"
        )

        # 1. Total Memory by Role
        ax = axes[0, 0]
        if "total_allocated_mb" in df.columns:
            sns.boxplot(data=df, x="role", y="total_allocated_mb", hue="type", ax=ax)
            ax.set_title("Total Memory Allocated by Role")
            ax.set_ylabel("Memory (MB)")

        # 2. Peak Memory by Role
        ax = axes[0, 1]
        if "peak_memory_mb" in df.columns:
            sns.boxplot(data=df, x="role", y="peak_memory_mb", hue="type", ax=ax)
            ax.set_title("Peak Memory Usage by Role")
            ax.set_ylabel("Memory (MB)")

        # 3. Allocations by Role
        ax = axes[1, 0]
        if "total_allocations" in df.columns:
            sns.boxplot(data=df, x="role", y="total_allocations", hue="type", ax=ax)
            ax.set_title("Total Allocations by Role")
            ax.set_ylabel("Number of Allocations")
            ax.ticklabel_format(style="scientific", axis="y", scilimits=(0, 0))

        # 4. Duration by Role
        ax = axes[1, 1]
        if "duration_seconds" in df.columns:
            sns.boxplot(data=df, x="role", y="duration_seconds", hue="type", ax=ax)
            ax.set_title("Profile Duration by Role")
            ax.set_ylabel("Duration (s)")

        plt.tight_layout()

        # Save the plot
        plot_file = self.output_dir / "mpi_role_based_comparison.png"
        plt.savefig(plot_file, dpi=300, bbox_inches="tight")
        print(f"    ✓ Generated role-based plot: {plot_file.name}")
        plt.close()

    def _generate_optimization_impact_analysis(self, df: pd.DataFrame):
        """Generate detailed analysis of optimization impact."""
        print("  Generating optimization impact analysis...")

        # Create pivot tables for easier comparison
        metrics_to_analyze = [
            "total_allocated_mb",
            "peak_memory_mb",
            "total_allocations",
            "duration_seconds",
        ]

        impact_data = []

        for rank in df["rank"].unique():
            rank_data = df[df["rank"] == rank]
            original = rank_data[rank_data["type"] == "original"]
            optimized = rank_data[rank_data["type"] == "optimized"]

            if not original.empty and not optimized.empty:
                original_row = original.iloc[0]
                optimized_row = optimized.iloc[0]

                for metric in metrics_to_analyze:
                    if (
                        metric in df.columns
                        and not pd.isna(original_row[metric])
                        and not pd.isna(optimized_row[metric])
                    ):
                        orig_val = original_row[metric]
                        opt_val = optimized_row[metric]

                        # Calculate percentage improvement (reduction is positive improvement)
                        if orig_val > 0:
                            improvement_pct = ((orig_val - opt_val) / orig_val) * 100
                        else:
                            improvement_pct = 0

                        impact_data.append(
                            {
                                "rank": rank,
                                "role": original_row["role"],
                                "metric": metric,
                                "original_value": orig_val,
                                "optimized_value": opt_val,
                                "improvement_pct": improvement_pct,
                                "improvement_absolute": orig_val - opt_val,
                            }
                        )

        impact_df = pd.DataFrame(impact_data)

        if not impact_df.empty:
            # Create improvement visualization
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle(
                "APPFL MPI Memory Optimization Impact Analysis",
                fontsize=14,
                fontweight="bold",
            )

            metrics_display_names = {
                "total_allocated_mb": "Total Memory (MB)",
                "peak_memory_mb": "Peak Memory (MB)",
                "total_allocations": "Total Allocations",
                "duration_seconds": "Duration (s)",
            }

            for idx, metric in enumerate(metrics_to_analyze):
                ax = axes[idx // 2, idx % 2]
                metric_data = impact_df[impact_df["metric"] == metric]

                if not metric_data.empty:
                    # Create bar plot showing improvement percentages
                    sns.barplot(data=metric_data, x="rank", y="improvement_pct", ax=ax)
                    ax.set_title(
                        f"{metrics_display_names.get(metric, metric)} Improvement (%)"
                    )
                    ax.set_xlabel("MPI Rank")
                    ax.set_ylabel("Improvement (%)")
                    ax.axhline(y=0, color="red", linestyle="--", alpha=0.5)

                    # Add value labels on bars
                    for container in ax.containers:
                        ax.bar_label(container, fmt="%.1f%%")

            plt.tight_layout()

            # Save the plot
            plot_file = self.output_dir / "mpi_optimization_impact_analysis.png"
            plt.savefig(plot_file, dpi=300, bbox_inches="tight")
            print(f"    ✓ Generated impact analysis plot: {plot_file.name}")
            plt.close()

            # Save impact data to CSV
            csv_file = self.output_dir / "mpi_optimization_impact_data.csv"
            impact_df.to_csv(csv_file, index=False)
            print(f"    ✓ Saved impact data: {csv_file.name}")

    def generate_comprehensive_report(self):
        """Generate a comprehensive text report summarizing all findings."""
        print("\nGenerating comprehensive report...")

        report_file = self.output_dir / "mpi_memory_optimization_report.txt"

        with open(report_file, "w") as f:
            f.write("=" * 80 + "\n")
            f.write("APPFL MPI Memory Optimization Experiment Results\n")
            f.write("=" * 80 + "\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Profiles directory: {self.profiles_dir}\n")
            f.write(f"Analysis output directory: {self.output_dir}\n")
            f.write("\n")

            # Experiment Overview
            f.write("EXPERIMENT OVERVIEW\n")
            f.write("-" * 40 + "\n")
            f.write(f"Total profiles analyzed: {len(self.metrics)}\n")
            f.write(f"MPI ranks tested: {len(self.profiles)}\n")

            original_profiles = [
                m for m in self.metrics if m.profile_type == "original"
            ]
            optimized_profiles = [
                m for m in self.metrics if m.profile_type == "optimized"
            ]

            f.write(f"Original version profiles: {len(original_profiles)}\n")
            f.write(f"Optimized version profiles: {len(optimized_profiles)}\n")

            server_profiles = [m for m in self.metrics if m.role == "server"]
            client_profiles = [m for m in self.metrics if m.role == "client"]

            f.write(f"Server profiles (rank 0): {len(server_profiles)}\n")
            f.write(f"Client profiles (rank > 0): {len(client_profiles)}\n")
            f.write("\n")

            # Detailed Metrics by Rank
            f.write("DETAILED METRICS BY RANK\n")
            f.write("-" * 40 + "\n")

            for rank in sorted(self.profiles.keys(), key=int):
                rank_metrics = [m for m in self.metrics if m.rank == rank]
                if not rank_metrics:
                    continue

                f.write(
                    f"\nRank {rank} ({'Server' if rank == '0' else f'Client {rank}'}):\n"
                )
                f.write("  " + "-" * 35 + "\n")

                original = next(
                    (m for m in rank_metrics if m.profile_type == "original"), None
                )
                optimized = next(
                    (m for m in rank_metrics if m.profile_type == "optimized"), None
                )

                if original:
                    f.write("  Original Version:\n")
                    f.write(
                        f"    Total Memory Allocated: {original.total_allocated:.2f} MB\n"
                    )
                    f.write(f"    Peak Memory Usage: {original.peak_memory:.2f} MB\n")
                    f.write(f"    Total Allocations: {original.total_allocations:,}\n")
                    f.write(
                        f"    Average Allocation Size: {original.avg_allocation_size:.1f} bytes\n"
                    )
                    f.write(
                        f"    Allocation Rate: {original.allocation_rate:.1f} allocs/sec\n"
                    )
                    f.write(
                        f"    Profile Duration: {original.profile_duration:.2f} seconds\n"
                    )
                    f.write("\n")

                if optimized:
                    f.write("  Optimized Version:\n")
                    f.write(
                        f"    Total Memory Allocated: {optimized.total_allocated:.2f} MB\n"
                    )
                    f.write(f"    Peak Memory Usage: {optimized.peak_memory:.2f} MB\n")
                    f.write(f"    Total Allocations: {optimized.total_allocations:,}\n")
                    f.write(
                        f"    Average Allocation Size: {optimized.avg_allocation_size:.1f} bytes\n"
                    )
                    f.write(
                        f"    Allocation Rate: {optimized.allocation_rate:.1f} allocs/sec\n"
                    )
                    f.write(
                        f"    Profile Duration: {optimized.profile_duration:.2f} seconds\n"
                    )
                    f.write("\n")

                if original and optimized:
                    f.write("  Optimization Impact:\n")

                    # Memory improvements
                    mem_improvement = (
                        (
                            (original.total_allocated - optimized.total_allocated)
                            / original.total_allocated
                        )
                        * 100
                        if original.total_allocated > 0
                        else 0
                    )
                    peak_improvement = (
                        (
                            (original.peak_memory - optimized.peak_memory)
                            / original.peak_memory
                        )
                        * 100
                        if original.peak_memory > 0
                        else 0
                    )
                    alloc_improvement = (
                        (
                            (original.total_allocations - optimized.total_allocations)
                            / original.total_allocations
                        )
                        * 100
                        if original.total_allocations > 0
                        else 0
                    )
                    duration_improvement = (
                        (
                            (original.profile_duration - optimized.profile_duration)
                            / original.profile_duration
                        )
                        * 100
                        if original.profile_duration > 0
                        else 0
                    )

                    f.write(f"    Total Memory Reduction: {mem_improvement:+.1f}%\n")
                    f.write(f"    Peak Memory Reduction: {peak_improvement:+.1f}%\n")
                    f.write(
                        f"    Allocation Count Reduction: {alloc_improvement:+.1f}%\n"
                    )
                    f.write(f"    Duration Change: {duration_improvement:+.1f}%\n")
                    f.write("\n")

            # Overall Summary
            f.write("OPTIMIZATION SUMMARY\n")
            f.write("-" * 40 + "\n")

            if original_profiles and optimized_profiles:
                # Calculate overall improvements
                orig_total_mem = sum(m.total_allocated for m in original_profiles)
                opt_total_mem = sum(m.total_allocated for m in optimized_profiles)

                orig_peak_mem = sum(m.peak_memory for m in original_profiles)
                opt_peak_mem = sum(m.peak_memory for m in optimized_profiles)

                orig_allocations = sum(m.total_allocations for m in original_profiles)
                opt_allocations = sum(m.total_allocations for m in optimized_profiles)

                f.write("Overall Results (across all ranks):\n")
                f.write(
                    f"  Total Memory Reduction: {((orig_total_mem - opt_total_mem) / orig_total_mem * 100):+.1f}%\n"
                )
                f.write(
                    f"  Peak Memory Reduction: {((orig_peak_mem - opt_peak_mem) / orig_peak_mem * 100):+.1f}%\n"
                )
                f.write(
                    f"  Allocation Count Reduction: {((orig_allocations - opt_allocations) / orig_allocations * 100):+.1f}%\n"
                )
                f.write("\n")

            # MPI-Specific Optimizations Applied
            f.write("MPI-SPECIFIC OPTIMIZATIONS APPLIED\n")
            f.write("-" * 40 + "\n")
            f.write("The following memory optimizations were implemented:\n\n")
            f.write("1. MPIServerCommunicator optimizations:\n")
            f.write(
                "   • Memory-efficient model serialization using context managers\n"
            )
            f.write("   • Strategic cleanup of request buffers and response data\n")
            f.write("   • Periodic garbage collection during message processing\n")
            f.write("   • CPU-first loading to reduce memory pressure\n\n")
            f.write("2. MPIClientCommunicator optimizations:\n")
            f.write("   • Optimized model transfer before sending to server\n")
            f.write("   • Resource cleanup of communication buffers\n")
            f.write("   • Memory-aware deserialization with immediate cleanup\n\n")
            f.write("3. MPI Serializer optimizations:\n")
            f.write("   • Memory-efficient BytesIO handling with context managers\n")
            f.write("   • CPU loading for models to reduce memory pressure\n")
            f.write("   • Automatic garbage collection after operations\n\n")

            # Recommendations
            f.write("RECOMMENDATIONS\n")
            f.write("-" * 40 + "\n")
            f.write("Based on the analysis:\n\n")

            # Generate recommendations based on results
            if optimized_profiles:
                avg_memory_reduction = (
                    sum(
                        (
                            (orig.total_allocated - opt.total_allocated)
                            / orig.total_allocated
                            * 100
                        )
                        for orig, opt in zip(original_profiles, optimized_profiles)
                        if orig.total_allocated > 0
                    )
                    / len(optimized_profiles)
                    if optimized_profiles
                    else 0
                )

                if avg_memory_reduction > 10:
                    f.write(
                        "✓ STRONG RECOMMENDATION: Enable MPI memory optimizations in production\n"
                    )
                    f.write(
                        f"  Average memory reduction: {avg_memory_reduction:.1f}%\n\n"
                    )
                elif avg_memory_reduction > 5:
                    f.write(
                        "✓ RECOMMENDATION: Consider enabling MPI memory optimizations\n"
                    )
                    f.write(
                        f"  Moderate memory reduction: {avg_memory_reduction:.1f}%\n\n"
                    )
                else:
                    f.write("• NEUTRAL: Memory optimizations show minimal impact\n")
                    f.write(
                        f"  Limited memory reduction: {avg_memory_reduction:.1f}%\n\n"
                    )

            f.write("Generated Files:\n")
            f.write(f"  • Flamegraphs: {self.output_dir}/flamegraphs/\n")
            f.write(f"  • Summaries: {self.output_dir}/summaries/\n")
            f.write(f"  • Comparison plots: {self.output_dir}/*.png\n")
            f.write(
                f"  • Impact data: {self.output_dir}/mpi_optimization_impact_data.csv\n"
            )
            f.write("\n")
            f.write("=" * 80 + "\n")

        print(f"  ✓ Generated comprehensive report: {report_file.name}")

    def run_complete_analysis(self):
        """Run the complete analysis pipeline."""
        print("Starting comprehensive MPI memory optimization analysis...")
        print(f"Profiles directory: {self.profiles_dir}")
        print(f"Output directory: {self.output_dir}")
        print(f"Found {len(self.profiles)} ranks with profiles")

        # 1. Extract metrics from all profiles
        self.extract_all_metrics()

        # 2. Generate flamegraphs
        self.generate_flamegraphs()

        # 3. Generate summary reports
        self.generate_summary_reports()

        # 4. Generate comparison plots
        self.generate_optimization_comparison_plots()

        # 5. Generate comprehensive text report
        self.generate_comprehensive_report()

        print("\n" + "=" * 80)
        print("COMPREHENSIVE MPI ANALYSIS COMPLETE!")
        print("=" * 80)
        print(f"Results saved to: {self.output_dir}")
        print("\nGenerated outputs:")
        print(f"  📊 Comparison plots: {self.output_dir}/*.png")
        print(f"  🔥 Flamegraphs: {self.output_dir}/flamegraphs/")
        print(f"  📋 Summary reports: {self.output_dir}/summaries/")
        print(f"  📈 Impact data: {self.output_dir}/mpi_optimization_impact_data.csv")
        print(
            f"  📝 Comprehensive report: {self.output_dir}/mpi_memory_optimization_report.txt"
        )
        print("\nUse these files to:")
        print("  • Compare memory usage between original and optimized versions")
        print("  • Analyze memory allocation patterns using flamegraphs")
        print("  • Quantify the impact of MPI memory optimizations")
        print("  • Make data-driven decisions about deploying optimizations")


def main():
    parser = argparse.ArgumentParser(
        description="Generate comprehensive results for APPFL MPI memory optimization experiments"
    )
    parser.add_argument(
        "profiles_dir", help="Directory containing MPI memory profile .bin files"
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        help="Output directory for analysis results (default: profiles_dir/analysis)",
    )

    args = parser.parse_args()

    # Check if profiles directory exists
    if not os.path.exists(args.profiles_dir):
        print(f"Error: Profiles directory not found: {args.profiles_dir}")
        sys.exit(1)

    # Check if memray is available
    try:
        subprocess.run(
            ["python", "-m", "memray", "--help"], capture_output=True, check=True
        )
    except:  # noqa E722
        print("Error: memray not found. Please install: pip install memray")
        sys.exit(1)

    # Run the analysis
    analyzer = MPIResultsGenerator(args.profiles_dir, args.output_dir)
    analyzer.run_complete_analysis()


if __name__ == "__main__":
    main()
