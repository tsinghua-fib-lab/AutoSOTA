"""
Configuration example for SSD offloading system.

This file shows recommended settings for different scene scales.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class StorageConfig:
    """Configuration for storage system."""

    # Storage paths
    ssd_cache_dir: str = "./ssd_cache"
    checkpoint_dir: str = "./checkpoints"

    # Block configuration
    block_size: int = 4096  # Points per block
    point_dim: int = 59     # Dimension per Gaussian point

    # RAM cache configuration
    max_ram_gb: float = 16.0
    eviction_threshold: float = 0.8  # Trigger eviction at 80% full
    prefetch_distance: int = 5       # Number of future blocks to prefetch

    # Pipeline configuration
    prefetch_ahead: int = 3          # Look ahead N iterations
    use_pinned_memory: bool = True   # Use pinned memory for faster H2D/D2H

    # TSP scheduling
    num_camera_clusters: int = 10
    cameras_per_cluster: int = 10

    # Compaction settings
    enable_compaction: bool = False  # Enable background compaction
    compaction_min_patches: int = 20 # Min patches before compaction

    # Monitoring
    log_interval: int = 100          # Print stats every N iterations
    save_checkpoint_interval: int = 1000


# Preset configurations for different scales

SMALL_SCENE = StorageConfig(
    block_size=2048,
    max_ram_gb=4.0,
    prefetch_distance=3,
    num_camera_clusters=5,
    cameras_per_cluster=5
)

MEDIUM_SCENE = StorageConfig(
    block_size=4096,
    max_ram_gb=16.0,
    prefetch_distance=5,
    num_camera_clusters=10,
    cameras_per_cluster=10
)

LARGE_SCENE = StorageConfig(
    block_size=8192,
    max_ram_gb=32.0,
    prefetch_distance=10,
    num_camera_clusters=20,
    cameras_per_cluster=15,
    enable_compaction=True
)

BILLION_SCALE = StorageConfig(
    block_size=16384,
    max_ram_gb=64.0,
    prefetch_distance=15,
    num_camera_clusters=50,
    cameras_per_cluster=20,
    enable_compaction=True,
    compaction_min_patches=50
)


def get_config_for_scene_size(num_points: int) -> StorageConfig:
    """
    Get recommended configuration based on scene size.

    Args:
        num_points: Total number of Gaussian points

    Returns:
        Recommended StorageConfig
    """
    if num_points < 1_000_000:  # < 1M
        return SMALL_SCENE
    elif num_points < 10_000_000:  # < 10M
        return MEDIUM_SCENE
    elif num_points < 100_000_000:  # < 100M
        return LARGE_SCENE
    else:  # >= 100M (billion scale)
        return BILLION_SCALE


def estimate_storage_requirements(
    num_points: int,
    point_dim: int = 59,
    dtype_bytes: int = 4
) -> dict:
    """
    Estimate storage requirements for a given scene.

    Args:
        num_points: Total number of Gaussian points
        point_dim: Dimension per point
        dtype_bytes: Bytes per value (4 for float32)

    Returns:
        Dictionary with size estimates
    """
    total_bytes = num_points * point_dim * dtype_bytes

    return {
        'total_size_gb': total_bytes / (1024**3),
        'base_file_gb': total_bytes / (1024**3),
        'recommended_ram_gb': min(64, max(4, total_bytes / (1024**3) * 0.1)),  # 10% of data
        'recommended_ssd_gb': total_bytes / (1024**3) * 1.5,  # 150% for patches
        'num_points': num_points,
        'point_dim': point_dim
    }


# Example usage
if __name__ == "__main__":
    # Example 1: Get config for 50M point scene
    scene_points = 50_000_000
    config = get_config_for_scene_size(scene_points)
    print(f"Config for {scene_points:,} points:")
    print(f"  Block size: {config.block_size}")
    print(f"  RAM cache: {config.max_ram_gb} GB")
    print(f"  Camera clusters: {config.num_camera_clusters}")

    # Example 2: Estimate storage
    print(f"\nStorage requirements for {scene_points:,} points:")
    reqs = estimate_storage_requirements(scene_points)
    for key, value in reqs.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")

    # Example 3: Different scene sizes
    print("\nRecommended configurations by scene size:")
    for size, name in [
        (500_000, "Small"),
        (5_000_000, "Medium"),
        (50_000_000, "Large"),
        (1_000_000_000, "Billion-scale")
    ]:
        cfg = get_config_for_scene_size(size)
        est = estimate_storage_requirements(size)
        print(f"\n{name} ({size:,} points):")
        print(f"  Block size: {cfg.block_size:,}")
        print(f"  RAM: {cfg.max_ram_gb} GB")
        print(f"  SSD needed: {est['recommended_ssd_gb']:.1f} GB")
        print(f"  Camera clusters: {cfg.num_camera_clusters}")
