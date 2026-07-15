#!/usr/bin/env python3
"""
Result Profile Analysis Tool for APPFL MPI Memory Optimization Experiments

This script provides comprehensive analysis and visualization of memory profiling results
from MPI federated learning experiments, comparing original vs optimized implementations.
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
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime


@dataclass
class ProfileMetrics:
    """Container for memory profile metrics."""

    total_allocated: float  # in MB
    peak_memory: float  # in MB
    total_allocations: int
    avg_allocation_size: float  # in bytes
    allocation_rate: float  # allocations per second
    profile_duration: float  # in seconds
    profile_file: str
    rank: str
    profile_type: str  # 'original' or 'optimized'


class MPIResultAnalyzer:
    """Comprehensive analyzer for MPI memory profiling results."""

    def __init__(self, profiles_dir: str, output_dir: str = None):
        self.profiles_dir = Path(profiles_dir)
        self.output_dir = (
            Path(output_dir) if output_dir else self.profiles_dir / "analysis"
        )
        self.output_dir.mkdir(exist_ok=True)

        self.profiles = self._discover_profiles()
        self.metrics = []

        # Set up plotting style
        plt.style.use(
            "seaborn-v0_8" if "seaborn-v0_8" in plt.style.available else "default"
        )
        sns.set_palette("husl")

    def _discover_profiles(self) -> Dict[str, Dict[str, str]]:
        """Discover and categorize profile files by rank and type."""
        profile_files = list(self.profiles_dir.glob("*.bin"))

        profiles = {}
        for profile_file in profile_files:
            filename = profile_file.name

            # Parse filename: mpi_rank_X_type_memory_profile.bin
            if filename.startswith("mpi_rank_"):
                parts = filename.replace(".bin", "").split("_")
                if len(parts) >= 4:
                    rank = parts[2]
                    profile_type = parts[3]  # original or optimized

                    if rank not in profiles:
                        profiles[rank] = {}
                    profiles[rank][profile_type] = str(profile_file)

        return profiles

    def extract_metrics_from_profile(
        self, profile_path: str, rank: str, profile_type: str
    ) -> Optional[ProfileMetrics]:
        """Extract detailed metrics from a memory profile using python -m memray."""
        try:
            result = subprocess.run(
                ["python", "-m", "memray", "stats", profile_path, "--json"],
                capture_output=True,
                text=True,
                timeout=60,
            )

            if result.returncode == 0:
                stats_data = json.loads(result.stdout)

                # Extract key metrics
                return ProfileMetrics(
                    total_allocated=self._bytes_to_mb(
                        stats_data.get("total_memory_allocated", 0)
                    ),
                    peak_memory=self._bytes_to_mb(stats_data.get("peak_memory", 0)),
                    total_allocations=stats_data.get("total_allocations", 0),
                    avg_allocation_size=stats_data.get("average_allocation_size", 0),
                    allocation_rate=stats_data.get("allocation_rate", 0),
                    profile_duration=stats_data.get("duration", 0),
                    profile_file=profile_path,
                    rank=rank,
                    profile_type=profile_type,
                )
            else:
                # Fallback to text parsing if JSON not available
                return self._parse_text_stats(profile_path, rank, profile_type)

        except Exception as e:
            print(f"Warning: Could not extract metrics from {profile_path}: {e}")
            return None

    def _parse_text_stats(
        self, profile_path: str, rank: str, profile_type: str
    ) -> Optional[ProfileMetrics]:
        """Fallback text parsing for python -m memray stats output."""
        try:
            result = subprocess.run(
                ["python", "-m", "memray", "stats", profile_path],
                capture_output=True,
                text=True,
                timeout=60,
            )

            if result.returncode == 0:
                lines = result.stdout.split("\n")

                total_allocated = 0
                peak_memory = 0
                total_allocations = 0
                duration = 1  # default to avoid division by zero

                for line in lines:
                    line = line.strip()
                    if "Total memory allocated:" in line:
                        total_allocated = self._parse_memory_value(line)
                    elif "Peak memory usage:" in line:
                        peak_memory = self._parse_memory_value(line)
                    elif "Total allocations:" in line:
                        total_allocations = self._parse_number_value(line)
                    elif "Duration:" in line:
                        duration = self._parse_duration_value(line)

                return ProfileMetrics(
                    total_allocated=total_allocated,
                    peak_memory=peak_memory,
                    total_allocations=total_allocations,
                    avg_allocation_size=total_allocated
                    * 1024
                    * 1024
                    / max(total_allocations, 1),
                    allocation_rate=total_allocations / max(duration, 1),
                    profile_duration=duration,
                    profile_file=profile_path,
                    rank=rank,
                    profile_type=profile_type,
                )

        except Exception as e:
            print(f"Warning: Text parsing failed for {profile_path}: {e}")

        return None

    def _bytes_to_mb(self, bytes_value: int) -> float:
        """Convert bytes to megabytes."""
        return bytes_value / (1024 * 1024)

    def _parse_memory_value(self, line: str) -> float:
        """Parse memory value from memray output (e.g., '150.2 MB' -> 150.2)."""
        try:
            parts = line.split(":")[-1].strip().split()
            if parts:
                value = float(parts[0].replace(",", ""))
                unit = parts[1].upper() if len(parts) > 1 else "MB"

                if unit == "GB":
                    return value * 1024
                elif unit == "KB":
                    return value / 1024
                else:  # MB
                    return value
        except (ValueError, IndexError):
            pass
        return 0.0

    def _parse_number_value(self, line: str) -> int:
        """Parse integer value from memray output."""
        try:
            parts = line.split(":")[-1].strip().split()
            if parts:
                return int(parts[0].replace(",", ""))
        except (ValueError, IndexError):
            pass
        return 0

    def _parse_duration_value(self, line: str) -> float:
        """Parse duration value from memray output."""
        try:
            parts = line.split(":")[-1].strip().split()
            if parts:
                value = float(parts[0])
                unit = parts[1].lower() if len(parts) > 1 else "s"

                if unit == "ms":
                    return value / 1000
                elif unit == "min":
                    return value * 60
                else:  # seconds
                    return value
        except (ValueError, IndexError):
            pass
        return 1.0

    def collect_all_metrics(self) -> List[ProfileMetrics]:
        """Collect metrics from all discovered profiles."""
        print("Extracting metrics from profiles...")

        for rank, rank_profiles in self.profiles.items():
            for profile_type, profile_path in rank_profiles.items():
                print(f"  Processing rank {rank} ({profile_type})...")

                metrics = self.extract_metrics_from_profile(
                    profile_path, rank, profile_type
                )
                if metrics:
                    self.metrics.append(metrics)

        print(f"Collected metrics from {len(self.metrics)} profiles")
        return self.metrics

    def generate_comparison_plots(self):
        """Generate comprehensive comparison plots."""
        if not self.metrics:
            print("No metrics available for plotting")
            return

        # Convert to DataFrame for easier manipulation
        df_data = []
        for metric in self.metrics:
            df_data.append(
                {
                    "rank": metric.rank,
                    "rank_type": "Server"
                    if metric.rank == "0"
                    else f"Client {metric.rank}",
                    "profile_type": metric.profile_type,
                    "total_allocated_mb": metric.total_allocated,
                    "peak_memory_mb": metric.peak_memory,
                    "total_allocations": metric.total_allocations,
                    "allocation_rate": metric.allocation_rate,
                    "avg_allocation_size": metric.avg_allocation_size,
                    "duration": metric.profile_duration,
                }
            )

        df = pd.DataFrame(df_data)

        # Create comprehensive plots
        self._plot_memory_usage_comparison(df)
        self._plot_allocation_patterns(df)
        self._plot_optimization_impact(df)
        self._plot_rank_comparison(df)

    def _plot_memory_usage_comparison(self, df: pd.DataFrame):
        """Plot memory usage comparison between original and optimized."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle("MPI Memory Usage Comparison: Original vs Optimized", fontsize=16)

        # Peak Memory Usage
        sns.barplot(
            data=df,
            x="rank_type",
            y="peak_memory_mb",
            hue="profile_type",
            ax=axes[0, 0],
        )
        axes[0, 0].set_title("Peak Memory Usage by Rank")
        axes[0, 0].set_ylabel("Peak Memory (MB)")
        axes[0, 0].tick_params(axis="x", rotation=45)

        # Total Allocated Memory
        sns.barplot(
            data=df,
            x="rank_type",
            y="total_allocated_mb",
            hue="profile_type",
            ax=axes[0, 1],
        )
        axes[0, 1].set_title("Total Allocated Memory by Rank")
        axes[0, 1].set_ylabel("Total Allocated (MB)")
        axes[0, 1].tick_params(axis="x", rotation=45)

        # Total Allocations
        sns.barplot(
            data=df,
            x="rank_type",
            y="total_allocations",
            hue="profile_type",
            ax=axes[1, 0],
        )
        axes[1, 0].set_title("Total Number of Allocations")
        axes[1, 0].set_ylabel("Total Allocations")
        axes[1, 0].tick_params(axis="x", rotation=45)

        # Allocation Rate
        sns.barplot(
            data=df,
            x="rank_type",
            y="allocation_rate",
            hue="profile_type",
            ax=axes[1, 1],
        )
        axes[1, 1].set_title("Allocation Rate")
        axes[1, 1].set_ylabel("Allocations per Second")
        axes[1, 1].tick_params(axis="x", rotation=45)

        plt.tight_layout()
        plt.savefig(
            self.output_dir / "memory_usage_comparison.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.show()

    def _plot_allocation_patterns(self, df: pd.DataFrame):
        """Plot allocation pattern analysis."""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle("Memory Allocation Pattern Analysis", fontsize=16)

        # Average Allocation Size
        sns.boxplot(data=df, x="profile_type", y="avg_allocation_size", ax=axes[0])
        axes[0].set_title("Average Allocation Size Distribution")
        axes[0].set_ylabel("Average Allocation Size (bytes)")

        # Duration vs Peak Memory
        sns.scatterplot(
            data=df,
            x="duration",
            y="peak_memory_mb",
            hue="profile_type",
            style="rank_type",
            s=100,
            ax=axes[1],
        )
        axes[1].set_title("Profile Duration vs Peak Memory")
        axes[1].set_xlabel("Duration (seconds)")
        axes[1].set_ylabel("Peak Memory (MB)")

        plt.tight_layout()
        plt.savefig(
            self.output_dir / "allocation_patterns.png", dpi=300, bbox_inches="tight"
        )
        plt.show()

    def _plot_optimization_impact(self, df: pd.DataFrame):
        """Plot optimization impact metrics."""
        # Calculate improvement percentages
        improvement_data = []

        for rank_type in df["rank_type"].unique():
            rank_df = df[df["rank_type"] == rank_type]

            original = rank_df[rank_df["profile_type"] == "original"]
            optimized = rank_df[rank_df["profile_type"] == "optimized"]

            if len(original) > 0 and len(optimized) > 0:
                orig_row = original.iloc[0]
                opt_row = optimized.iloc[0]

                for metric in [
                    "peak_memory_mb",
                    "total_allocated_mb",
                    "total_allocations",
                ]:
                    if orig_row[metric] > 0:
                        improvement = (
                            (orig_row[metric] - opt_row[metric]) / orig_row[metric]
                        ) * 100
                        improvement_data.append(
                            {
                                "rank_type": rank_type,
                                "metric": metric.replace("_", " ").title(),
                                "improvement_percent": improvement,
                            }
                        )

        if improvement_data:
            improvement_df = pd.DataFrame(improvement_data)

            plt.figure(figsize=(12, 8))
            sns.barplot(
                data=improvement_df,
                x="rank_type",
                y="improvement_percent",
                hue="metric",
            )
            plt.title("Memory Optimization Impact by Rank and Metric", fontsize=16)
            plt.ylabel("Improvement (%)")
            plt.xlabel("Rank Type")
            plt.axhline(y=0, color="black", linestyle="-", alpha=0.3)
            plt.xticks(rotation=45)
            plt.legend(title="Metric", bbox_to_anchor=(1.05, 1), loc="upper left")

            plt.tight_layout()
            plt.savefig(
                self.output_dir / "optimization_impact.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.show()

    def _plot_rank_comparison(self, df: pd.DataFrame):
        """Plot comparison between different ranks (server vs clients)."""
        plt.figure(figsize=(15, 10))

        # Create a 2x2 subplot for different metrics
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle("MPI Rank Performance Comparison", fontsize=16)

        metrics = [
            "peak_memory_mb",
            "total_allocated_mb",
            "total_allocations",
            "allocation_rate",
        ]
        titles = [
            "Peak Memory Usage",
            "Total Allocated Memory",
            "Total Allocations",
            "Allocation Rate",
        ]
        y_labels = [
            "Memory (MB)",
            "Memory (MB)",
            "Number of Allocations",
            "Allocations/sec",
        ]

        for i, (metric, title, y_label) in enumerate(zip(metrics, titles, y_labels)):
            ax = axes[i // 2, i % 2]

            # Create grouped bar plot
            sns.barplot(data=df, x="profile_type", y=metric, hue="rank_type", ax=ax)
            ax.set_title(title)
            ax.set_ylabel(y_label)
            ax.set_xlabel("Profile Type")

            if i == 1:  # Only show legend for one subplot
                ax.legend(title="Rank", bbox_to_anchor=(1.05, 1), loc="upper left")
            else:
                ax.legend().set_visible(False)

        plt.tight_layout()
        plt.savefig(
            self.output_dir / "rank_comparison.png", dpi=300, bbox_inches="tight"
        )
        plt.show()

    def generate_summary_report(self):
        """Generate a comprehensive text summary report."""
        if not self.metrics:
            print("No metrics available for report generation")
            return

        report_file = self.output_dir / "memory_optimization_report.txt"

        with open(report_file, "w") as f:
            f.write("=" * 80 + "\n")
            f.write("APPFL MPI MEMORY OPTIMIZATION ANALYSIS REPORT\n")
            f.write("=" * 80 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Profiles analyzed: {len(self.metrics)}\n")
            f.write(f"Output directory: {self.output_dir}\n\n")

            # Group metrics by rank
            rank_metrics = {}
            for metric in self.metrics:
                if metric.rank not in rank_metrics:
                    rank_metrics[metric.rank] = {}
                rank_metrics[metric.rank][metric.profile_type] = metric

            # Generate per-rank analysis
            for rank in sorted(rank_metrics.keys()):
                rank_type = "Server" if rank == "0" else f"Client {rank}"
                f.write(f"\n{rank_type} (Rank {rank}) Analysis:\n")
                f.write("-" * 50 + "\n")

                rank_data = rank_metrics[rank]

                for profile_type in ["original", "optimized"]:
                    if profile_type in rank_data:
                        metric = rank_data[profile_type]
                        f.write(f"\n{profile_type.capitalize()} Implementation:\n")
                        f.write(f"  Peak Memory: {metric.peak_memory:.2f} MB\n")
                        f.write(f"  Total Allocated: {metric.total_allocated:.2f} MB\n")
                        f.write(f"  Total Allocations: {metric.total_allocations:,}\n")
                        f.write(
                            f"  Average Allocation Size: {metric.avg_allocation_size:.2f} bytes\n"
                        )
                        f.write(
                            f"  Allocation Rate: {metric.allocation_rate:.2f} allocs/sec\n"
                        )
                        f.write(
                            f"  Profile Duration: {metric.profile_duration:.2f} seconds\n"
                        )

                # Calculate improvements if both versions exist
                if "original" in rank_data and "optimized" in rank_data:
                    orig = rank_data["original"]
                    opt = rank_data["optimized"]

                    f.write(f"\nOptimization Impact for {rank_type}:\n")

                    metrics_to_compare = [
                        ("peak_memory", "Peak Memory", "MB"),
                        ("total_allocated", "Total Allocated", "MB"),
                        ("total_allocations", "Total Allocations", "count"),
                    ]

                    for attr, name, unit in metrics_to_compare:
                        orig_val = getattr(orig, attr)
                        opt_val = getattr(opt, attr)

                        if orig_val > 0:
                            improvement = ((orig_val - opt_val) / orig_val) * 100
                            if improvement > 0:
                                f.write(f"  {name}: {improvement:.1f}% reduction\n")
                            elif improvement < 0:
                                f.write(f"  {name}: {abs(improvement):.1f}% increase\n")
                            else:
                                f.write(f"  {name}: No significant change\n")

            # Generate overall summary
            f.write(f"\n\n{'=' * 50}\n")
            f.write("OVERALL OPTIMIZATION SUMMARY\n")
            f.write(f"{'=' * 50}\n")

            # Calculate total improvements across all ranks
            total_orig_peak = sum(
                m.peak_memory for m in self.metrics if m.profile_type == "original"
            )
            total_opt_peak = sum(
                m.peak_memory for m in self.metrics if m.profile_type == "optimized"
            )

            total_orig_allocated = sum(
                m.total_allocated for m in self.metrics if m.profile_type == "original"
            )
            total_opt_allocated = sum(
                m.total_allocated for m in self.metrics if m.profile_type == "optimized"
            )

            if total_orig_peak > 0 and total_opt_peak > 0:
                peak_improvement = (
                    (total_orig_peak - total_opt_peak) / total_orig_peak
                ) * 100
                f.write(f"Overall Peak Memory Improvement: {peak_improvement:.1f}%\n")

            if total_orig_allocated > 0 and total_opt_allocated > 0:
                allocated_improvement = (
                    (total_orig_allocated - total_opt_allocated) / total_orig_allocated
                ) * 100
                f.write(
                    f"Overall Allocated Memory Improvement: {allocated_improvement:.1f}%\n"
                )

            f.write("\nKey Optimization Features:\n")
            f.write("  • Memory-efficient model serialization using context managers\n")
            f.write("  • Strategic garbage collection during MPI operations\n")
            f.write("  • CPU-first model loading to reduce memory pressure\n")
            f.write("  • Immediate cleanup of communication buffers\n")
            f.write("  • Resource cleanup after MPI message processing\n")

            f.write("\nGenerated Files:\n")
            f.write(
                "  • memory_usage_comparison.png - Memory usage comparison charts\n"
            )
            f.write(
                "  • allocation_patterns.png - Memory allocation pattern analysis\n"
            )
            f.write("  • optimization_impact.png - Optimization impact visualization\n")
            f.write("  • rank_comparison.png - Performance comparison between ranks\n")
            f.write("  • memory_optimization_report.txt - This comprehensive report\n")

        print(f"Summary report saved to: {report_file}")
        return report_file

    def run_full_analysis(self):
        """Run complete analysis pipeline."""
        print("Starting MPI memory profile analysis...")

        if not self.profiles:
            print("No profile files found!")
            return

        print(f"Found profiles for ranks: {list(self.profiles.keys())}")

        # Collect metrics
        self.collect_all_metrics()

        if not self.metrics:
            print("No metrics could be extracted from profiles!")
            return

        # Generate visualizations
        print("Generating comparison plots...")
        self.generate_comparison_plots()

        # Generate summary report
        print("Generating summary report...")
        self.generate_summary_report()

        print(f"\nAnalysis complete! Results saved to: {self.output_dir}")

        # List generated files
        generated_files = list(self.output_dir.glob("*.png")) + list(
            self.output_dir.glob("*.txt")
        )
        if generated_files:
            print("\nGenerated files:")
            for file in sorted(generated_files):
                print(f"  • {file.name}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze MPI memory profiling results from APPFL experiments"
    )
    parser.add_argument(
        "--profiles-dir",
        type=str,
        default="./memory_profiles",
        help="Directory containing memory profile files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory for analysis results (default: profiles-dir/analysis)",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip plot generation (useful for headless environments)",
    )

    args = parser.parse_args()

    # Check if profiles directory exists
    if not os.path.exists(args.profiles_dir):
        print(f"Error: Profiles directory not found: {args.profiles_dir}")
        return 1

    # Check if memray is available
    try:
        subprocess.run(
            ["python", "-m", "memray", "--help"], capture_output=True, check=True
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Error: memray not found. Please install: pip install memray")
        return 1

    # Create analyzer and run analysis
    analyzer = MPIResultAnalyzer(args.profiles_dir, args.output_dir)

    if args.no_plots:
        # Skip plot generation, just collect metrics and generate report
        analyzer.collect_all_metrics()
        analyzer.generate_summary_report()
    else:
        analyzer.run_full_analysis()

    return 0


if __name__ == "__main__":
    sys.exit(main())
