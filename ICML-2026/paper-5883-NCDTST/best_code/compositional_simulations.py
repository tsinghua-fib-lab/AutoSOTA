"""
Compositional Two Sample Test Simulations

This module provides reproducible simulations comparing the T_p_mn spherical test
statistic against CLR based methods (Energy Distance, MMD) for compositional data.

Key features:
- Master seed for full reproducibility
- Save/load simulation results to/from files
- Multiple experiment types:
  1. Dispersion alternatives (same mean, different concentration)
  2. Small mean shifts
- KDE plots and empirical size/power tables

Usage:
    python compositional_simulations.py
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import norm, gaussian_kde
import pandas as pd
import json
import os
from datetime import datetime
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Tuple, Optional, Callable
from tqdm import tqdm

# Project code
from optimized_estimators import SphericalTestConfig, OptimizedTestStatistic
from dgp_dm_two_sample import (
    DMScenario,
    generate_dm_two_sample,
    validate_simplex,
)
from empirical_size import (
    add_pseudocount_and_renorm,
    clr_transform,
    pairwise_l2_dist,
    median_heuristic_sigma_from_dist,
    gaussian_kernel_from_dist,
    energy_from_dist,
    mmd2_unbiased_from_kernel,
)

# =============================================================================
# Configuration and Data Classes
# =============================================================================

@dataclass
class CompositionalSimConfig:
    """Configuration for a single simulation scenario."""
    name: str
    d: int                      # Number of compositional components
    m: int                      # Sample size for group X
    n: int                      # Sample size for group Y
    N_lib: int                  # Library size controls sparsity
    num_replications: int       # Number of Monte Carlo replications
    B_perm: int                 # Number of permutations for p-value computation

    # Distribution parameters
    mu0: np.ndarray             # Mean for group X (on simplex)
    mu1: np.ndarray             # Mean for group Y (on simplex)
    kappa_base: float = 25.0    # Base concentration parameter
    eta: float = 1.0            # Dispersion multiplier for group Y

    # Method parameters
    p_values: Tuple[int, ...] = (2, 4, 8)  # Truncation parameters for T_p_mn
    eps_clr: float = 1e-8       # Pseudocount perturbation sfor CLR transform

    # Alternative metadata
    alternative_type: str = "null"  # "null", "dispersion", "mean_shift"
    effect_size: float = 0.0    # Effect size parameter

    def __post_init__(self):
        """Convert arrays and validate"""
        self.mu0 = validate_simplex(np.asarray(self.mu0, dtype=float), "mu0")
        self.mu1 = validate_simplex(np.asarray(self.mu1, dtype=float), "mu1")
        self.p_values = tuple(self.p_values)


@dataclass
class SimulationResult:
    """Results from a simulation run"""
    config: CompositionalSimConfig

    # P-values for each method across all replications
    pvalues: Dict[str, np.ndarray]  # {method_name: array of p-values}

    # Rejection rates at different alpha levels
    rejection_rates: Dict[str, Dict[float, float]]  # {method: {alpha: rate}}

    # Metadata
    master_seed: int
    timestamp: str
    pz_empirical_mean: float    # Mean empirical zero fraction
    pz_empirical_std: float     # Std of empirical zero fraction


# =============================================================================
# Transform Functions
# =============================================================================

def sqrt_transform(X: np.ndarray) -> np.ndarray:
    """
    Transform compositions to sphere via the square root map
    """
    return np.sqrt(np.maximum(X, 0.0))


def l2_normalize(X: np.ndarray) -> np.ndarray:
    """
    Transform compositions to sphere via L2 normalization.
    Preserves ratios: x_i/x_j unchanged.
    """
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(norms, 1e-12)


# =============================================================================
# Alternative Constructors
# =============================================================================

def make_base_mean(d: int) -> np.ndarray:
    """
    Create a standard base mean on the simplex
    Uses linearly increasing weights
    """
    mu = np.linspace(1, d, d)
    return mu / mu.sum()


def make_log_ratio_shift(mu0: np.ndarray, delta: float, pattern: str = "sparse2") -> np.ndarray:
    """
    Create mean shift in log-ratio space (multiplicative)

    mu1 is proportional to mu0 * exp(delta * v), where v is a contrast vector

    This is the standard Aitchison perturbation

    Parameters
    ----------
    mu0 : array
        Base mean on simplex
    delta : float
        Effect size (>= 0)
    pattern : str
        "sparse2": perturb first two components (+1, -1)
        "sparse2_reversed": perturb last two components (+1, -1)
        "uniform": same contrast across all components
    """
    mu0 = np.asarray(mu0, dtype=float)
    mu0 = mu0 / mu0.sum()
    d = len(mu0)

    if delta == 0.0:
        return mu0.copy()

    if pattern == "sparse2":
        # Perturb 2 smallest/rarest taxa (indices 0 and 1)
        v = np.zeros(d)
        v[0] = 1.0
        v[1] = -1.0
    elif pattern == "sparse2_reversed":
        # Perturb 2 largest/dominant taxa (indices d-2 and d-1)
        v = np.zeros(d)
        v[-2] = 1.0
        v[-1] = -1.0
    elif pattern == "uniform":
        # Equal perturbation to all taxa with alternating signs
        v = np.ones(d)
        v[::2] = 1.0
        v[1::2] = -1.0
    else:
        raise ValueError(f"Unknown pattern: {pattern}")

    v = v - v.mean()  # Center to sum to 0

    mu1 = mu0 * np.exp(delta * v)
    mu1 = np.maximum(mu1, 1e-16)
    mu1 = mu1 / mu1.sum()

    return mu1


# =============================================================================
# T_p_mn Test Statistic
# =============================================================================

# Cache for SphericalTestConfig objects
_config_cache = {}

def get_spherical_config(p: int, d: int) -> SphericalTestConfig:
    """Get or create a cached SphericalTestConfig"""
    key = (p, d)
    if key not in _config_cache:
        _config_cache[key] = SphericalTestConfig(p=p, d=d)
    return _config_cache[key]


def compute_spherical_test(
    X_sphere: np.ndarray,
    Y_sphere: np.ndarray,
    p: int
) -> float:
    """
    Compute T_p_mn test statistic for data already on the sphere

    Returns the studentized test statistic T (asymptotically N(0,1) under H0)
    """
    d_ambient = X_sphere.shape[1]
    d = d_ambient - 1  # S^{d-1} in R^d

    config = get_spherical_config(p, d)
    calculator = OptimizedTestStatistic(config)

    return calculator.compute(X_sphere, Y_sphere, use_unbiased=True)


# =============================================================================
# Permutation Testing
# =============================================================================

def compute_all_pvalues(
    X_comp: np.ndarray,
    Y_comp: np.ndarray,
    # Note this is the reproducing kernel truncation parameter "p" for the T_p_mn statistic
    p_values: Tuple[int, ...],
    eps_clr: float,
    B_perm: int,
    rng: np.random.Generator
) -> Dict[str, float]:
    """
    Compute p-values for all methods on one replicate

    Methods (permutation p-values):
    - CLR-ED: Energy distance on CLR transformed data
    - CLR-MMD: MMD with Gaussian kernel on CLR data
    - HEL-MMD: MMD with Gaussian kernel on square root transformed data

    Methods (asymptotic N(0,1) p-values):
    - SPH-sqrt-p{k}: T_p_mn with sqrt transform
    - SPH-L2-p{k}: T_p_mn with L2 normalization
    """
    m, n = X_comp.shape[0], Y_comp.shape[0]
    N = m + n
    idxX = np.arange(m)
    idxY = np.arange(m, N)

    # Pool the data
    Z_comp = np.vstack([X_comp, Y_comp])

    # === CLR based methods ===
    Z_clr = clr_transform(add_pseudocount_and_renorm(Z_comp, eps_clr))
    D_clr = pairwise_l2_dist(Z_clr)
    sigma_clr = median_heuristic_sigma_from_dist(D_clr)
    K_clr = gaussian_kernel_from_dist(D_clr, sigma_clr)

    # Observed CLR statistics
    ED_obs = energy_from_dist(D_clr, idxX, idxY)
    MMD_clr_obs = mmd2_unbiased_from_kernel(K_clr, idxX, idxY)

    # === Spherical methods ===
    # Transform to sphere
    Z_sqrt = sqrt_transform(Z_comp)
    Z_l2 = l2_normalize(Z_comp)

    # === HEL-MMD (sqrt transform + MMD) ===
    D_hel = pairwise_l2_dist(Z_sqrt)
    sigma_hel = median_heuristic_sigma_from_dist(D_hel)
    K_hel = gaussian_kernel_from_dist(D_hel, sigma_hel)
    MMD_hel_obs = mmd2_unbiased_from_kernel(K_hel, idxX, idxY)

    # Observed T_p_mn statistics
    sph_sqrt_obs = {}
    sph_l2_obs = {}
    for p in p_values:
        sph_sqrt_obs[p] = compute_spherical_test(Z_sqrt[:m], Z_sqrt[m:], p)
        sph_l2_obs[p] = compute_spherical_test(Z_l2[:m], Z_l2[m:], p)

    # === Permutation test for CLR and HEL-MMD methods ===
    count_ED = 0
    count_MMD_clr = 0
    count_MMD_hel = 0

    for _ in range(B_perm):
        perm = rng.permutation(N)
        idx_X_perm = perm[:m]
        idx_Y_perm = perm[m:]

        # CLR-ED permutation statistic
        ED_perm = energy_from_dist(D_clr, idx_X_perm, idx_Y_perm)
        count_ED += (ED_perm >= ED_obs)

        # CLR-MMD permutation statistic
        MMD_clr_perm = mmd2_unbiased_from_kernel(K_clr, idx_X_perm, idx_Y_perm)
        count_MMD_clr += (MMD_clr_perm >= MMD_clr_obs)

        # HEL-MMD permutation statistic
        MMD_hel_perm = mmd2_unbiased_from_kernel(K_hel, idx_X_perm, idx_Y_perm)
        count_MMD_hel += (MMD_hel_perm >= MMD_hel_obs)

    # Compute p-values
    # CLR and HEL-MMD use permutation p-values
    pvals = {
        'CLR-ED': (1 + count_ED) / (B_perm + 1),
        'CLR-MMD': (1 + count_MMD_clr) / (B_perm + 1),
        'HEL-MMD': (1 + count_MMD_hel) / (B_perm + 1),
    }
    # T_p_mn uses asymptotic N(0,1) p-values (one-sided, reject for large T)
    for p in p_values:
        pvals[f'SPH-sqrt-p{p}'] = norm.sf(sph_sqrt_obs[p])  # P(Z > T_obs)
        pvals[f'SPH-L2-p{p}'] = norm.sf(sph_l2_obs[p])      # P(Z > T_obs)

    return pvals


# ===========================================================================
# Simulation Runner
# ===========================================================================

def run_single_simulation(
    config: CompositionalSimConfig,
    master_seed: int,
    show_progress: bool = True
) -> SimulationResult:
    """
    Run a single simulation scenario

    Parameters
    ----------
    config : CompositionalSimConfig
        Simulation configuration
    master_seed : int
        Gloabl seed for reproducibility
    show_progress : bool
        Whether to show progress bar

    Returns
    -------
    result : SimulationResult
        Simulation results
    """
    # Initialize storage
    methods = ['CLR-ED', 'CLR-MMD', 'HEL-MMD']
    for p in config.p_values:
        methods.append(f'SPH-sqrt-p{p}')
        methods.append(f'SPH-L2-p{p}')

    all_pvalues = {m: [] for m in methods}
    pz_values = []

    # Create scenario
    sc = DMScenario(
        d=config.d,
        m=config.m,
        n=config.n,
        N=config.N_lib,
        mu0=config.mu0,
        mu1=config.mu1,
        kappa_base=config.kappa_base,
        eta=config.eta
    )

    iterator = range(config.num_replications)
    if show_progress:
        iterator = tqdm(iterator, desc=f"  {config.name}", leave=False)

    for i in iterator:
        # Derive seed for this replication
        rep_seed = master_seed + i

        # Generate data
        out = generate_dm_two_sample(sc, seed=rep_seed)
        X_comp = out["comps0"]
        Y_comp = out["comps1"]
        pz_values.append(out["pz_empirical"])

        # Compute p-values
        rng = np.random.default_rng(rep_seed + 1_000_000)
        pvals = compute_all_pvalues(
            X_comp, Y_comp,
            config.p_values,
            config.eps_clr,
            config.B_perm,
            rng
        )

        for m in methods:
            all_pvalues[m].append(pvals[m])

    # Convert to arrays
    pvalues_arrays = {m: np.array(v) for m, v in all_pvalues.items()}

    # Compute rejection rates
    alphas = [0.01, 0.05, 0.10]
    rejection_rates = {}
    for m in methods:
        rejection_rates[m] = {}
        for alpha in alphas:
            rejection_rates[m][alpha] = float(np.mean(pvalues_arrays[m] <= alpha))

    return SimulationResult(
        config=config,
        pvalues=pvalues_arrays,
        rejection_rates=rejection_rates,
        master_seed=master_seed,
        timestamp=datetime.now().isoformat(),
        pz_empirical_mean=float(np.mean(pz_values)),
        pz_empirical_std=float(np.std(pz_values))
    )


def run_simulation_grid(
    configs: List[CompositionalSimConfig],
    master_seed: int,
    show_progress: bool = True
) -> List[SimulationResult]:
    """
    Run multiple simulation scenarios
    """
    results = []

    for i, config in enumerate(configs):
        print(f"\n[{i+1}/{len(configs)}] Running: {config.name}")
        print(f"    d={config.d}, m={config.m}, n={config.n}, "
              f"type={config.alternative_type}, effect={config.effect_size:.3f}")

        # Each config gets a different derived seed
        config_seed = master_seed + i * 100_000

        result = run_single_simulation(config, config_seed, show_progress)
        results.append(result)

        # Print summary using largest p value from config
        max_p = max(config.p_values)
        sph_key = f'SPH-sqrt-p{max_p}'
        sph_rate = result.rejection_rates.get(sph_key, {}).get(0.05, 0)
        print(f"    pz_emp={result.pz_empirical_mean:.3f}, "
              f"CLR-ED@5%={result.rejection_rates['CLR-ED'][0.05]:.3f}, "
              f"{sph_key}@5%={sph_rate:.3f}")

    return results


# =============================================================================
# Save/Load Functions
# ============================================================================

def save_results(results: List[SimulationResult], filepath: str) -> None:
    """Save simulation results to a .npz file"""
    data = {'num_results': len(results)}

    for i, result in enumerate(results):
        prefix = f'result_{i}_'

        # Save config
        config_dict = {
            'name': result.config.name,
            'd': result.config.d,
            'm': result.config.m,
            'n': result.config.n,
            'N_lib': result.config.N_lib,
            'num_replications': result.config.num_replications,
            'B_perm': result.config.B_perm,
            'mu0': result.config.mu0.tolist(),
            'mu1': result.config.mu1.tolist(),
            'kappa_base': result.config.kappa_base,
            'eta': result.config.eta,
            'p_values': list(result.config.p_values),
            'eps_clr': result.config.eps_clr,
            'alternative_type': result.config.alternative_type,
            'effect_size': result.config.effect_size,
        }
        data[prefix + 'config'] = json.dumps(config_dict)

        # Save p-values
        for method, pvals in result.pvalues.items():
            data[prefix + f'pvalues_{method}'] = pvals

        # Save metadata
        data[prefix + 'master_seed'] = result.master_seed
        data[prefix + 'timestamp'] = result.timestamp
        data[prefix + 'pz_empirical_mean'] = result.pz_empirical_mean
        data[prefix + 'pz_empirical_std'] = result.pz_empirical_std
        data[prefix + 'rejection_rates'] = json.dumps(result.rejection_rates)

    np.savez_compressed(filepath, **data)
    print(f"Saved {len(results)} results to {filepath}")


def load_results(filepath: str) -> List[SimulationResult]:
    """Load simulation results from a .npz file."""
    data = np.load(filepath, allow_pickle=True)

    num_results = int(data['num_results'])
    results = []

    for i in range(num_results):
        prefix = f'result_{i}_'

        # Load config
        config_dict = json.loads(str(data[prefix + 'config']))
        config = CompositionalSimConfig(
            name=config_dict['name'],
            d=config_dict['d'],
            m=config_dict['m'],
            n=config_dict['n'],
            N_lib=config_dict['N_lib'],
            num_replications=config_dict['num_replications'],
            B_perm=config_dict['B_perm'],
            mu0=np.array(config_dict['mu0']),
            mu1=np.array(config_dict['mu1']),
            kappa_base=config_dict['kappa_base'],
            eta=config_dict['eta'],
            p_values=tuple(config_dict['p_values']),
            eps_clr=config_dict['eps_clr'],
            alternative_type=config_dict['alternative_type'],
            effect_size=config_dict['effect_size'],
        )

        # Load p-values
        pvalues = {}
        methods = ['CLR-ED', 'CLR-MMD', 'HEL-MMD']
        for p in config.p_values:
            methods.append(f'SPH-sqrt-p{p}')
            methods.append(f'SPH-L2-p{p}')
        for method in methods:
            key = prefix + f'pvalues_{method}'
            if key in data:
                pvalues[method] = data[key]

        # Load rejection rates
        rejection_rates = json.loads(str(data[prefix + 'rejection_rates']))
        # Convert string keys back to floats
        rejection_rates = {
            m: {float(a): r for a, r in rates.items()}
            for m, rates in rejection_rates.items()
        }

        result = SimulationResult(
            config=config,
            pvalues=pvalues,
            rejection_rates=rejection_rates,
            master_seed=int(data[prefix + 'master_seed']),
            timestamp=str(data[prefix + 'timestamp']),
            pz_empirical_mean=float(data[prefix + 'pz_empirical_mean']),
            pz_empirical_std=float(data[prefix + 'pz_empirical_std']),
        )
        results.append(result)

    print(f"Loaded {len(results)} results from {filepath}")
    return results


# =============================================================================
# Plotting Functions
# =============================================================================

def plot_pvalue_histograms(
    results: List[SimulationResult],
    methods: List[str] = None,
    save_path: str = None,
    title: str = None
) -> plt.Figure:
    """
    Plot histograms of p-values for each method
    """
    if methods is None:
        methods = list(results[0].pvalues.keys())

    n_results = len(results)
    n_methods = len(methods)

    fig, axes = plt.subplots(n_results, n_methods, figsize=(3 * n_methods, 2.5 * n_results))
    if n_results == 1:
        axes = axes.reshape(1, -1)
    if n_methods == 1:
        axes = axes.reshape(-1, 1)

    for i, result in enumerate(results):
        for j, method in enumerate(methods):
            ax = axes[i, j]
            pvals = result.pvalues.get(method, np.array([]))

            if len(pvals) > 0:
                ax.hist(pvals, bins=20, range=(0, 1), density=True, alpha=0.7, edgecolor='black')
                ax.axhline(1.0, color='red', linestyle='--', label='Uniform')
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 3)

            if i == 0:
                ax.set_title(method, fontsize=9)
            if j == 0:
                ax.set_ylabel(f"{result.config.name}\n(d={result.config.d})", fontsize=8)
            if i == n_results - 1:
                ax.set_xlabel('p-value', fontsize=8)

    if title:
        fig.suptitle(title, fontsize=11, fontweight='bold')

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved figure to {save_path}")

    return fig


def create_results_table(
    results: List[SimulationResult],
    alpha: float = 0.05
) -> pd.DataFrame:
    """
    Create a summary table of rejection rates
    """
    rows = []

    for result in results:
        config = result.config

        for method, rates in result.rejection_rates.items():
            rows.append({
                'name': config.name,
                'd': config.d,
                'm': config.m,
                'n': config.n,
                'N_lib': config.N_lib,
                'alternative': config.alternative_type,
                'effect': config.effect_size,
                'eta': config.eta,
                'pz_emp': result.pz_empirical_mean,
                'method': method,
                'rej@1%': rates.get(0.01, np.nan),
                'rej@5%': rates.get(0.05, np.nan),
                'rej@10%': rates.get(0.10, np.nan),
            })

    return pd.DataFrame(rows)


def print_summary_table(results: List[SimulationResult], alpha: float = 0.05) -> None:
    """Print a formatted summary table"""
    df = create_results_table(results, alpha)

    print("\n" + "=" * 120)
    print("SIMULATION RESULTS SUMMARY")
    print("=" * 120)

    # Pivot for better display
    for alt_type in df['alternative'].unique():
        print(f"\n--- Alternative: {alt_type} ---")
        df_sub = df[df['alternative'] == alt_type]

        pivot = df_sub.pivot_table(
            index=['d', 'm', 'effect', 'eta'],
            columns='method',
            values='rej@5%'
        )

        print(pivot.round(3).to_string())

    print("\n" + "=" * 120)


# =============================================================================
# Comprehensive Simulation Study
# =============================================================================

def run_comprehensive_simulation(
    master_seed: int,
    output_dir: str,
) -> Tuple[List[SimulationResult], str]:
    """
    Run the comprehensive simulation study

    This is the main experiment runner with configuration:
    - d = 10 (10-dimensional compositions)
    - Sample sizes: (m, n) = (50, 50) and (100, 100)
    - Sparsity levels: ~5% zeros (N_lib=200) and ~50% zeros (N_lib=20)
    - Mean shift patterns: sparse2, sparse2_reversed, uniform
    - Mean shift deltas: {0.05, 0.25, 0.6}
    - Dispersion eta: {1.2, 1.5, 2.0}
    - T_p_mn truncation p_values: {2, 4}
    - Replications: 200
    - Permutations: 200

    Methods compared:
    - CLR-ED: Energy distance on CLR transformed data
    - CLR-MMD: MMD with Gaussian kernel on CLR data
    - HEL-MMD: MMD on square root transformed data
    - SPH-sqrt-p2, SPH-sqrt-p4: T_p_mn with sqrt transform
    - SPH-L2-p2, SPH-L2-p4: T_p_mn with L2 normalization

    Parameters
    ----------
    master_seed : int
        Master seed for full reproducibility. All derived seeds are deterministic functions
        of this seed.
    output_dir : str
        Directory to save results.

    Returns
    -------
    results : List[SimulationResult]
        All simulation results
    exp_dir : str
        Path to output directory
    """
    # SIMULATION CONFIGURATION
    D = 10                                      # Number of compositional components
    SAMPLE_SIZES = ((50, 50), (100, 100))       # (m, n) pairs
    SPARSITY_CONFIG = {                         # N_lib values calibrated for target sparsity
        'low': {'N_lib': 100, 'label': '~5%'},   # Calibrated: 5.6% zeros
        'high': {'N_lib': 8, 'label': '~50%'},   # Calibrated: 49.2% zeros
    }
    MEAN_SHIFT_PATTERNS = ('sparse2', 'sparse2_reversed', 'uniform')
    DELTA_VALUES = (0.0, 0.05, 0.25, 0.6)       # Mean shift effect sizes (0.0 = H0 for size control)
    ETA_VALUES = (1.0, 1.2, 1.5, 2.0)           # Dispersion multipliers (1.0 = H0 for size control)
    P_VALUES = (2, 4)                           # T_p_mn truncation parameters
    NUM_REPLICATIONS = 200
    B_PERM = 200
    KAPPA_BASE = 50.0                           # Base concentration parameter
    EPS_CLR = 1e-8                              # Pseudocount for CLR

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(output_dir, f"comprehensive_simulation_{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)

    # Save configuration for reproducibility
    config_info = {
        'master_seed': master_seed,
        'd': D,
        'sample_sizes': SAMPLE_SIZES,
        'sparsity_config': SPARSITY_CONFIG,
        'mean_shift_patterns': MEAN_SHIFT_PATTERNS,
        'delta_values': DELTA_VALUES,
        'eta_values': ETA_VALUES,
        'p_values': P_VALUES,
        'num_replications': NUM_REPLICATIONS,
        'B_perm': B_PERM,
        'kappa_base': KAPPA_BASE,
        'eps_clr': EPS_CLR,
        'timestamp': timestamp,
    }
    with open(os.path.join(exp_dir, 'config.json'), 'w') as f:
        json.dump(config_info, f, indent=2, default=str)

    print("\n" + "=" * 80)
    print("COMPREHENSIVE SIMULATION STUDY")
    print("=" * 80)
    print(f"\nMaster Seed: {master_seed}")
    print(f"Output Directory: {exp_dir}")
    print(f"\nConfiguration:")
    print(f"  - d = {D}")
    print(f"  - Sample sizes: {SAMPLE_SIZES}")
    print(f"  - Sparsity levels: {[(k, v['label']) for k, v in SPARSITY_CONFIG.items()]}")
    print(f"  - Mean shift patterns: {MEAN_SHIFT_PATTERNS}")
    print(f"  - Delta values: {DELTA_VALUES}")
    print(f"  - Eta values: {ETA_VALUES}")
    print(f"  - T_p_mn p values: {P_VALUES}")
    print(f"  - Replications: {NUM_REPLICATIONS}")
    print(f"  - Permutations per test: {B_PERM}")

    mu0 = make_base_mean(D)
    configs = []
    config_idx = 0

    # --- MEAN SHIFT EXPERIMENTS ---
    for sparsity_key, sparsity_info in SPARSITY_CONFIG.items():
        N_lib = sparsity_info['N_lib']
        pz_label = sparsity_info['label']

        for m, n in SAMPLE_SIZES:
            for pattern in MEAN_SHIFT_PATTERNS:
                for delta in DELTA_VALUES:
                    mu1 = make_log_ratio_shift(mu0, delta, pattern=pattern)
                    name = f"mean_{pattern}_d{D}_m{m}_pz{pz_label}_delta{delta}"

                    # Label as null if delta=0, otherwise as mean_shift
                    if delta == 0.0:
                        alt_type = "null"
                    else:
                        alt_type = f"mean_shift_{pattern}"

                    configs.append(CompositionalSimConfig(
                        name=name,
                        d=D, m=m, n=n,
                        N_lib=N_lib,
                        num_replications=NUM_REPLICATIONS,
                        B_perm=B_PERM,
                        mu0=mu0,
                        mu1=mu1,
                        kappa_base=KAPPA_BASE,
                        eta=1.0,  # No dispersion change
                        p_values=P_VALUES,
                        eps_clr=EPS_CLR,
                        alternative_type=alt_type,
                        effect_size=delta,
                    ))
                    config_idx += 1

    # --- DISPERSION EXPERIMENTS ---
    for sparsity_key, sparsity_info in SPARSITY_CONFIG.items():
        N_lib = sparsity_info['N_lib']
        pz_label = sparsity_info['label']

        for m, n in SAMPLE_SIZES:
            for eta in ETA_VALUES:
                name = f"disp_d{D}_m{m}_pz{pz_label}_eta{eta}"

                # Label as null if eta=1.0, otherwise as dispersion
                if eta == 1.0:
                    alt_type = "null"
                else:
                    alt_type = "dispersion"

                configs.append(CompositionalSimConfig(
                    name=name,
                    d=D, m=m, n=n,
                    N_lib=N_lib,
                    num_replications=NUM_REPLICATIONS,
                    B_perm=B_PERM,
                    mu0=mu0,
                    mu1=mu0,  # Same mean, different dispersion
                    kappa_base=KAPPA_BASE,
                    eta=eta,
                    p_values=P_VALUES,
                    eps_clr=EPS_CLR,
                    alternative_type=alt_type,
                    effect_size=eta,
                ))
                config_idx += 1

    # Estimate the runtime
    total_configs = len(configs)
    total_replicates = total_configs * NUM_REPLICATIONS
    num_methods = 3 + 2 * len(P_VALUES)  # CLR-ED, CLR-MMD, HEL-MMD + 2 per p value
    total_permutations = total_replicates * B_PERM * num_methods

    print(f"\n  Total configurations: {total_configs}")
    print(f"  Total replicates: {total_replicates:,}")
    print(f"  Total permutation tests: {total_permutations:,}")

    # Execute the experiment
    print("\n" + "=" * 80)
    print("RUNNING SIMULATIONS")
    print("=" * 80)

    all_results = []
    start_time = datetime.now()

    for i, config in enumerate(configs):
        print(f"\n[{i+1}/{total_configs}] {config.name}")
        print(f"    type={config.alternative_type}, effect={config.effect_size:.3f}")

        # Derive deterministic seed from master seed and config index
        config_seed = master_seed + i * 100_000

        result = run_single_simulation(config, config_seed, show_progress=True)
        all_results.append(result)

        # Print key results using largest p value
        max_p = max(P_VALUES)
        sph_key = f'SPH-sqrt-p{max_p}'
        clr_ed_rate = result.rejection_rates['CLR-ED'][0.05]
        sph_rate = result.rejection_rates.get(sph_key, {}).get(0.05, 0)
        print(f"    pz_emp={result.pz_empirical_mean:.3f}, "
              f"CLR-ED@5%={clr_ed_rate:.1%}, {sph_key}@5%={sph_rate:.1%}")

        # Periodic checkpoint saves
        if (i + 1) % 10 == 0:
            checkpoint_path = os.path.join(exp_dir, f"checkpoint_{i+1}.npz")
            save_results(all_results, checkpoint_path)
            # Also save CSV for easy progress monitoring
            df = create_results_table(all_results)
            df.to_csv(os.path.join(exp_dir, "results_in_progress.csv"), index=False)
            print(f"    [Checkpoint saved: {checkpoint_path}]")

    # Persist the results
    print("\n" + "=" * 80)
    print("SAVING RESULTS")
    print("=" * 80)

    # Save all results
    final_path = os.path.join(exp_dir, "results.npz")
    save_results(all_results, final_path)

    # Create and save summary table
    df = create_results_table(all_results)
    csv_path = os.path.join(exp_dir, "results_table.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved results table to {csv_path}")

    elapsed = datetime.now() - start_time
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print(f"\nTotal runtime: {elapsed}")

    # Summary by experiment type
    for alt_type in df['alternative'].unique():
        print(f"\n--- {alt_type.upper()} ---")
        df_sub = df[df['alternative'] == alt_type]

        # Pivot table: rows = (effect, pz_emp), columns = method
        pivot = df_sub.pivot_table(
            index=['m', 'effect', 'pz_emp'],
            columns='method',
            values='rej@5%'
        ).round(3)

        print(pivot.to_string())

    # Remove checkpoints after successful completion
    for f in os.listdir(exp_dir):
        if f.startswith('checkpoint_') and f.endswith('.npz'):
            os.remove(os.path.join(exp_dir, f))

    print("\n" + "=" * 80)
    print("SIMULATION COMPLETE")
    print("=" * 80)
    print(f"Results saved to: {exp_dir}")

    return all_results, exp_dir


if __name__ == "__main__":
    MASTER_SEED = 10397
    OUTPUT_DIR = "compositional_simulations"

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Run the comprehensive simulation
    results, exp_dir = run_comprehensive_simulation(
        master_seed=MASTER_SEED,
        output_dir=OUTPUT_DIR,
    )
