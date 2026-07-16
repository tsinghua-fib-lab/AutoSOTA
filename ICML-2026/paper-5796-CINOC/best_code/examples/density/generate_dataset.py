"""
NS2D Shape Formation Dataset Generator

Generates structured, learnable pairs of (initial smoke, target smoke) for 
training multi-agent DPC controllers.

Key Design Principles:
1. Initial = random blob at random position
2. Target = same blob moved to a different target position
3. This creates a TRANSLATION task that is clearly learnable

The goal: Learn to move smoke from point A to point B using mobile injectors.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, Optional
import argparse


# =============================================================================
# Shape Generation Functions
# =============================================================================

def generate_gaussian_blob(
    Nx: int, Ny: int,
    center: Tuple[float, float],
    sigma: float = 0.12,
    amplitude: float = 1.0
) -> np.ndarray:
    """Generate a single Gaussian blob at specified center."""
    x = np.linspace(0, 1, Nx)
    y = np.linspace(0, 1.25, Ny)  # Match domain aspect ratio
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    dist_sq = (X - center[0])**2 + (Y - center[1])**2
    blob = amplitude * np.exp(-dist_sq / (2 * sigma**2))
    
    return blob


def generate_ring(
    Nx: int, Ny: int,
    center: Tuple[float, float],
    radius: float = 0.2,
    width: float = 0.04,
    amplitude: float = 0.8
) -> np.ndarray:
    """Generate a ring at specified center."""
    x = np.linspace(0, 1, Nx)
    y = np.linspace(0, 1.25, Ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    dist = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
    ring = amplitude * np.exp(-((dist - radius)**2) / (2 * width**2))
    
    return ring


def generate_double_blob(
    Nx: int, Ny: int,
    center: Tuple[float, float],
    separation: float = 0.15,
    sigma: float = 0.08,
    amplitude: float = 0.8
) -> np.ndarray:
    """Generate two blobs separated horizontally."""
    c1 = (center[0] - separation/2, center[1])
    c2 = (center[0] + separation/2, center[1])
    
    blob1 = generate_gaussian_blob(Nx, Ny, c1, sigma, amplitude)
    blob2 = generate_gaussian_blob(Nx, Ny, c2, sigma, amplitude)
    
    return np.clip(blob1 + blob2, 0, 1)


# =============================================================================
# Structured Dataset: Translation Tasks
# =============================================================================

def generate_translation_pair(
    Nx: int, Ny: int,
    shape_type: str = 'blob',
    rng: np.random.Generator = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a pair where target is initial shape translated to new position.
    
    This creates a clearly LEARNABLE task: move smoke from A to B.
    """
    if rng is None:
        rng = np.random.default_rng()
    
    # Random initial position (in bottom-left region)
    init_x = rng.uniform(0.2, 0.5)
    init_y = rng.uniform(0.2, 0.5)
    init_center = (init_x, init_y)
    
    # Random target position (different from initial, tend to be higher/right)
    target_x = rng.uniform(0.4, 0.8)
    target_y = rng.uniform(0.5, 1.0)
    target_center = (target_x, target_y)
    
    # Ensure meaningful displacement
    while np.sqrt((target_x - init_x)**2 + (target_y - init_y)**2) < 0.25:
        target_x = rng.uniform(0.4, 0.8)
        target_y = rng.uniform(0.5, 1.0)
        target_center = (target_x, target_y)
    
    # Generate shape at both positions
    sigma = rng.uniform(0.08, 0.14)
    amplitude = rng.uniform(0.7, 1.0)
    
    if shape_type == 'blob':
        initial = generate_gaussian_blob(Nx, Ny, init_center, sigma, amplitude)
        target = generate_gaussian_blob(Nx, Ny, target_center, sigma, amplitude)
    elif shape_type == 'ring':
        radius = rng.uniform(0.12, 0.2)
        initial = generate_ring(Nx, Ny, init_center, radius, amplitude=amplitude)
        target = generate_ring(Nx, Ny, target_center, radius, amplitude=amplitude)
    else:  # double_blob
        initial = generate_double_blob(Nx, Ny, init_center, sigma=sigma*0.7, amplitude=amplitude)
        target = generate_double_blob(Nx, Ny, target_center, sigma=sigma*0.7, amplitude=amplitude)
    
    return initial, target


def generate_expansion_pair(
    Nx: int, Ny: int,
    rng: np.random.Generator = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a pair where target is expanded version of initial.
    
    Task: Spread smoke from concentrated blob to diffuse region.
    """
    if rng is None:
        rng = np.random.default_rng()
    
    # Center position
    cx = rng.uniform(0.3, 0.7)
    cy = rng.uniform(0.4, 0.8)
    center = (cx, cy)
    
    # Initial: concentrated blob
    sigma_init = rng.uniform(0.06, 0.10)
    initial = generate_gaussian_blob(Nx, Ny, center, sigma_init, amplitude=1.0)
    
    # Target: expanded blob (same center)
    sigma_target = sigma_init * rng.uniform(2.0, 3.0)
    target = generate_gaussian_blob(Nx, Ny, center, sigma_target, amplitude=0.5)
    
    return initial, target


def generate_concentration_pair(
    Nx: int, Ny: int,
    rng: np.random.Generator = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a pair where target is concentrated version.
    
    Task: Gather diffuse smoke into tight blob.
    """
    if rng is None:
        rng = np.random.default_rng()
    
    # Center position
    cx = rng.uniform(0.3, 0.7)
    cy = rng.uniform(0.4, 0.8)
    center = (cx, cy)
    
    # Initial: diffuse blob
    sigma_init = rng.uniform(0.15, 0.22)
    initial = generate_gaussian_blob(Nx, Ny, center, sigma_init, amplitude=0.5)
    
    # Target: concentrated blob (same center)  
    sigma_target = sigma_init * rng.uniform(0.3, 0.5)
    target = generate_gaussian_blob(Nx, Ny, center, sigma_target, amplitude=1.0)
    
    return initial, target


# =============================================================================
# Dataset Generation
# =============================================================================

def generate_dataset(
    n_samples: int,
    Nx: int = 64,
    Ny: int = 80,
    seed: int = 42,
    shape_type: str = 'blob'  # Focus on simpler shapes for learning
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate training/test dataset of structured (initial, target) pairs.
    
    For advection learning, we focus on blob translation which is the
    most learnable task for fan-only control.
    """
    rng = np.random.default_rng(seed)
    
    rho_init = np.zeros((n_samples, Nx, Ny))
    rho_target = np.zeros((n_samples, Nx, Ny))
    
    for i in range(n_samples):
        # Use specified shape type for consistency
        init, target = generate_translation_pair(Nx, Ny, shape_type, rng)
        
        rho_init[i] = init
        rho_target[i] = target
    
    return rho_init, rho_target


def visualize_samples(rho_init, rho_target, n_show: int = 5, filename: str = 'sample_pairs.png'):
    """Visualize sample initial/target pairs with shared colorbar."""
    fig = plt.figure(figsize=(4*n_show + 1, 8))
    
    # GridSpec: 2 rows, n_show+1 cols (last col for colorbar)
    gs = fig.add_gridspec(2, n_show + 1, width_ratios=[1]*n_show + [0.08], wspace=0.15, hspace=0.25)
    
    vmin, vmax = 0, 1
    
    im = None
    for i in range(n_show):
        idx = i
        
        # Initial
        ax_init = fig.add_subplot(gs[0, i])
        im = ax_init.imshow(rho_init[idx].T, origin='lower', cmap='hot', vmin=vmin, vmax=vmax, aspect='auto')
        ax_init.set_title(f'Initial {idx}', fontsize=12)
        ax_init.axis('off')
        
        # Target
        ax_target = fig.add_subplot(gs[1, i])
        ax_target.imshow(rho_target[idx].T, origin='lower', cmap='hot', vmin=vmin, vmax=vmax, aspect='auto')
        ax_target.set_title(f'Target {idx}', fontsize=12)
        ax_target.axis('off')
    
    # Shared colorbar
    cax = fig.add_subplot(gs[:, -1])
    cbar = fig.colorbar(im, cax=cax, orientation='vertical')
    cbar.set_label(r'Smoke Density $\rho$', fontsize=14)
    
    plt.suptitle('NS2D Shape Formation: Structured Pairs', fontsize=16, fontweight='bold')
    plt.savefig(filename, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"Saved: {filename}")
    plt.close()


def visualize_task_types(Nx, Ny, seed=42, filename='task_types.png'):
    """Visualize different task types."""
    rng = np.random.default_rng(seed)
    
    fig, axes = plt.subplots(3, 2, figsize=(10, 12))
    
    # Translation
    init, target = generate_translation_pair(Nx, Ny, 'blob', rng)
    axes[0, 0].imshow(init.T, origin='lower', cmap='hot', vmin=0, vmax=1)
    axes[0, 0].set_title('Translation: Initial', fontsize=12)
    axes[0, 0].axis('off')
    axes[0, 1].imshow(target.T, origin='lower', cmap='hot', vmin=0, vmax=1)
    axes[0, 1].set_title('Translation: Target', fontsize=12)
    axes[0, 1].axis('off')
    
    # Expansion
    init, target = generate_expansion_pair(Nx, Ny, rng)
    axes[1, 0].imshow(init.T, origin='lower', cmap='hot', vmin=0, vmax=1)
    axes[1, 0].set_title('Expansion: Initial', fontsize=12)
    axes[1, 0].axis('off')
    axes[1, 1].imshow(target.T, origin='lower', cmap='hot', vmin=0, vmax=1)
    axes[1, 1].set_title('Expansion: Target', fontsize=12)
    axes[1, 1].axis('off')
    
    # Concentration
    init, target = generate_concentration_pair(Nx, Ny, rng)
    axes[2, 0].imshow(init.T, origin='lower', cmap='hot', vmin=0, vmax=1)
    axes[2, 0].set_title('Concentration: Initial', fontsize=12)
    axes[2, 0].axis('off')
    im = axes[2, 1].imshow(target.T, origin='lower', cmap='hot', vmin=0, vmax=1)
    axes[2, 1].set_title('Concentration: Target', fontsize=12)
    axes[2, 1].axis('off')
    
    plt.colorbar(im, ax=axes, shrink=0.8, label='Smoke Density')
    plt.suptitle('NS2D Control Task Types', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(filename, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"Saved: {filename}")
    plt.close()


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Generate NS2D shape formation dataset')
    parser.add_argument('--n_train', type=int, default=1000, help='Number of training samples')
    parser.add_argument('--n_test', type=int, default=100, help='Number of test samples')
    parser.add_argument('--Nx', type=int, default=64, help='Grid resolution X')
    parser.add_argument('--Ny', type=int, default=80, help='Grid resolution Y')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--shape', type=str, default='blob', choices=['blob', 'ring', 'double_blob'],
                        help='Shape type for translation task')
    args = parser.parse_args()
    
    print("="*60)
    print("NS2D Shape Formation Dataset Generation")
    print("="*60)
    
    data_dir = Path(__file__).parent / 'data'
    data_dir.mkdir(exist_ok=True)
    
    # Physics parameters
    config = {
        'Nx': args.Nx,
        'Ny': args.Ny,
        'Lx': 1.0,
        'Ly': 1.25,
        'dt': 1.0,
        'buoyancy': 0.5,
        'n_agents': 25,  # 5x5 grid
        'sigma': 0.05,
    }
    
    print(f"\nGrid: {args.Nx}x{args.Ny}")
    print(f"Agents: {config['n_agents']}")
    print(f"Training samples: {args.n_train}")
    print(f"Test samples: {args.n_test}")
    print(f"Shape type: {args.shape}")
    
    # Generate training data
    print("\nGenerating training data...")
    print(f"  Shape type: {args.shape} (blob-only for easier learning)")
    rho_init_train, rho_target_train = generate_dataset(
        args.n_train, args.Nx, args.Ny, seed=args.seed, shape_type=args.shape
    )
    
    # Generate test data
    print("Generating test data...")
    rho_init_test, rho_target_test = generate_dataset(
        args.n_test, args.Nx, args.Ny, seed=args.seed + 1000, shape_type=args.shape
    )
    
    # Save
    print("\nSaving datasets...")
    np.savez(data_dir / 'config.npz', **config)
    np.savez(data_dir / 'train_data.npz', 
             rho_init=rho_init_train, rho_target=rho_target_train)
    np.savez(data_dir / 'test_data.npz',
             rho_init=rho_init_test, rho_target=rho_target_test)
    
    print(f"  config.npz: Grid and physics parameters")
    print(f"  train_data.npz: {args.n_train} samples")
    print(f"  test_data.npz: {args.n_test} samples")
    
    # Visualize samples
    print("\nVisualizing samples...")
    visualize_samples(rho_init_train, rho_target_train, n_show=5, 
                     filename=str(data_dir / 'sample_pairs.png'))
    
    visualize_task_types(args.Nx, args.Ny, seed=args.seed,
                        filename=str(data_dir / 'task_types.png'))
    
    print("\nDone!")


if __name__ == "__main__":
    main()
