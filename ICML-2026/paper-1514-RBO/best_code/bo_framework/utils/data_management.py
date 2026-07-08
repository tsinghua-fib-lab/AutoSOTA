"""Data management utilities for BO experiments.

Pure functions for generating, loading, and saving experimental data.
"""

import torch
from typing import Optional, Dict, Any
from pathlib import Path


def generate_sobol_points(search_space, n: int, seed: Optional[int] = None) -> torch.Tensor:
    """Generate Sobol sequence points in the search space.
    
    Sobol sequences are quasi-random low-discrepancy sequences that provide
    better space-filling properties than random sampling.
    
    Args:
        search_space: SearchSpace defining the domain
        n: Number of points to generate
        seed: Random seed for scrambling (optional)
        
    Returns:
        Tensor of Sobol points [n, n_dims] in the search space coordinates
        (normalized to [0,1] if dimensions have normalize=True)
    """
    from torch.quasirandom import SobolEngine
    
    # Create Sobol engine with optional scrambling
    sobol = SobolEngine(dimension=search_space.n_dims, scramble=True, seed=seed)
    
    # Generate points in [0, 1]^d
    sobol_01 = sobol.draw(n).double()
    
    # Scale to search space bounds
    bounds = search_space.bounds
    lower, upper = bounds[0], bounds[1]
    
    # Scale from [0,1] to the bounds
    # Note: bounds are already normalized if dimension.normalize=True
    sobol_scaled = sobol_01 * (upper - lower) + lower
    
    return sobol_scaled


def generate_latin_hypercube_points(search_space, n: int, seed: Optional[int] = None) -> torch.Tensor:
    """Generate Latin Hypercube samples in the search space.
    
    Latin Hypercube Sampling (LHS) is a statistical method for generating
    a near-random sample that ensures each dimension is evenly sampled.
    
    Args:
        search_space: SearchSpace defining the domain
        n: Number of points to generate
        seed: Random seed for reproducibility (optional)
        
    Returns:
        Tensor of LHS points [n, n_dims] in the search space coordinates
    """
    import numpy as np
    from scipy.stats import qmc
    
    # Set random seed if provided
    if seed is not None:
        np.random.seed(seed)
    
    # Generate LHS samples in [0, 1]^d
    sampler = qmc.LatinHypercube(d=search_space.n_dims, seed=seed)
    lhs_01 = sampler.random(n=n)
    
    # Convert to torch tensor
    lhs_01 = torch.from_numpy(lhs_01).double()
    
    # Scale to search space bounds
    bounds = search_space.bounds
    lower, upper = bounds[0], bounds[1]
    
    # Scale from [0,1] to the bounds
    lhs_scaled = lhs_01 * (upper - lower) + lower
    
    return lhs_scaled


def generate_random_points(search_space, n: int, seed: Optional[int] = None) -> torch.Tensor:
    """Generate uniform random points in the search space.
    
    Args:
        search_space: SearchSpace defining the domain
        n: Number of points to generate
        seed: Random seed for reproducibility (optional)
        
    Returns:
        Tensor of random points [n, n_dims] in the search space coordinates
    """
    if seed is not None:
        torch.manual_seed(seed)
    
    # Generate random points in [0, 1]^d
    random_01 = torch.rand(n, search_space.n_dims, dtype=torch.double)
    
    # Scale to search space bounds
    bounds = search_space.bounds
    lower, upper = bounds[0], bounds[1]
    
    # Scale from [0,1] to the bounds
    random_scaled = random_01 * (upper - lower) + lower
    
    return random_scaled


def load_initial_points(path: str, n: Optional[int] = None) -> torch.Tensor:
    """Load initial points from file.
    
    Args:
        path: Path to the saved points file
        n: Number of points to load (if None, loads all)
        
    Returns:
        Tensor of loaded points [n, n_dims]
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        KeyError: If the file doesn't contain 'initial_points'
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    
    data = torch.load(path)
    
    if 'initial_points' not in data:
        raise KeyError(f"File {path} does not contain 'initial_points'")
    
    points = data['initial_points']
    
    if n is not None:
        if n > len(points):
            raise ValueError(f"Requested {n} points but file only contains {len(points)}")
        points = points[:n]
    
    return points


def save_initial_points(points: torch.Tensor, path: str, 
                       metadata: Optional[Dict[str, Any]] = None) -> None:
    """Save initial points to file with optional metadata.
    
    The points are saved in a dictionary format that can be loaded later.
    
    Args:
        points: Tensor of points to save [n, n_dims]
        path: Path where to save the points
        metadata: Optional metadata to save alongside points (e.g., seed, search_space info)
    """
    path = Path(path)
    
    # Create parent directory if it doesn't exist
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Prepare save data
    save_data = {'initial_points': points}
    
    if metadata:
        save_data.update(metadata)
    
    # Save to file
    torch.save(save_data, path)


def load_experiment_results(path: str) -> Dict[str, Any]:
    """Load complete experiment results from file.
    
    Args:
        path: Path to the saved results file
        
    Returns:
        Dictionary containing experiment results
        
    Raises:
        FileNotFoundError: If the file doesn't exist
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    
    return torch.load(path)


def save_experiment_results(results: Dict[str, Any], path: str) -> None:
    """Save experiment results to file.
    
    Args:
        results: Dictionary containing experiment results
        path: Path where to save the results
    """
    path = Path(path)
    
    # Create parent directory if it doesn't exist
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save to file
    torch.save(results, path)


def combine_results(*result_dicts: Dict[str, Any]) -> Dict[str, Any]:
    """Combine multiple experiment results into a single dictionary.
    
    Useful for comparing multiple runs or methods.
    
    Args:
        *result_dicts: Variable number of result dictionaries
        
    Returns:
        Combined dictionary with results indexed by experiment number
    """
    combined = {}
    for i, results in enumerate(result_dicts):
        combined[f'experiment_{i}'] = results
    return combined