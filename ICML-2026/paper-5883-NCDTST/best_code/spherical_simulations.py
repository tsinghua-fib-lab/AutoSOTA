"""
Spherical Two-Sample Test Simulations (T_p_mn)

This module provides reproducible simulations for the spherical harmonics
energy distance based two-sample test statistic T_p_mn.

Key features:
- Master seed for full reproducibility
- Save/load simulation results to/from files
- Grid plots comparing KDE vs N(0,1) across parameter configurations
- Support for various distributions on the sphere (uniform, von Mises Fisher)

Usage:
    python spherical_simulations.py
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import norm, kstest, gaussian_kde, vonmises_fisher
import json
import os
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional, Callable
from tqdm import tqdm

from optimized_estimators import SphericalTestConfig, OptimizedTestStatistic

# =============================================================================
# Configuration and Data Classes
# =============================================================================

@dataclass
class SimulationConfig:
    """Configuration for a single iteration."""
    name: str
    p: int                    # Truncation parameter for reproducing kernel
    d: int                    # Sphere dimension (S^d in R^{d+1})
    m: int                    # Sample size for X
    n: int                    # Sample size for Y
    num_simulations: int      # Number of Monte Carlo replications
    use_unbiased: bool = True # Use bias-corrected variance estimator

    # Distribution parameters
    distribution_X: str = "uniform"
    distribution_Y: str = "uniform"
    dist_params_X: Optional[Dict] = None
    dist_params_Y: Optional[Dict] = None


@dataclass
class SimulationResult:
    """Results from a simulation run."""
    config: SimulationConfig
    T_values: np.ndarray      # Array of test statistic values
    master_seed: int          # Master seed used
    timestamp: str            # When simulation was run

    # Summary statistics
    mean: float
    std: float
    skewness: float
    kurtosis: float
    ks_statistic: float
    ks_pvalue: float


# =============================================================================
# Sampling Functions
# =============================================================================

def random_points_on_sphere(n: int, d: int, seed: int | None = None) -> np.ndarray:
    """
    Generate n random points uniformly on the unit sphere S^d ⊂ R^{d+1}.

    Uses the Muller/Marsaglia method: sample from a standard normal distribution
    and normalize to unit length (Haar / rotation invariant measure).
    """
    if n < 0:
        raise ValueError("n must be nonnegative")
    if d < 0:
        raise ValueError("d must be nonnegative")

    rng = np.random.default_rng(seed)
    points = rng.standard_normal((n, d + 1), dtype=np.float64)
    norms = np.linalg.norm(points, axis=1)

    points /= norms[:, None]
    return points


def sample_von_mises_fisher(
    n: int,
    d: int,
    mu: np.ndarray,
    kappa: float,
    seed: int | None = None
) -> np.ndarray:
    """
    Sample n points from the von Mises Fisher distribution on the unit sphere S^d.

    The vMF distribution is parameterized by a mean direction mu and a concentration
    parameter kappa ≥ 0. Larger kappa yield stronger concentration around mu; kappa = 0
    reduces to the uniform (Haar) distribution on S^d.

    Sampling is delegated to SciPy's `scipy.stats.vonmises_fisher`.

    Parameters
    ----------
    n : int
        Number of points to sample.
    d : int
        Dimension of the sphere
    mu : np.ndarray
        Mean direction vector of shape (d+1,). This vector will be normalized
        internally and must be nonzero.
    kappa : float
        Concentration parameter (kappa >= 0). Larger values concentrate samples
        more tightly around mu.
    seed : int, optional
        Seed for NumPy's random number generator, for reproducibility.

    Returns
    -------
    np.ndarray
        Array of shape (n, d+1) containing samples on the unit sphere S^d.

    Notes
    -----
    - Requires SciPy >= 1.11.
    - All returned samples have unit Euclidean norm
    - If kappa == 0, the distribution is uniform on S^d.
    """
    if n < 0:
        raise ValueError("n must be nonnegative")
    if d < 0:
        raise ValueError("d must be nonnegative")
    if kappa < 0 or not np.isfinite(kappa):
        raise ValueError("kappa must be a nonnegative finite scalar")

    mu = np.asarray(mu, dtype=np.float64)
    if mu.ndim != 1 or mu.shape[0] != d + 1:
        raise ValueError("mu must have shape (d+1,)")

    norm_mu = np.linalg.norm(mu)
    if norm_mu == 0:
        raise ValueError("mu must be a nonzero vector")
    mu = mu / norm_mu

    rng = np.random.default_rng(seed)
    vmf = vonmises_fisher(mu=mu, kappa=kappa)
    samples = vmf.rvs(size=n, random_state=rng)

    if samples.shape != (n, d + 1):
        raise RuntimeError("Unexpected sample shape from vonmises_fisher")

    return samples


# =============================================================================
# Simulation Runner
# =============================================================================

def create_sampler(distribution: str, d: int, sample_size: int,
                   params: Optional[Dict] = None) -> Callable:
    """
    Create a sampler function for the specified distribution.

    Parameters
    ----------
    distribution : str
        One of: "uniform", "vmf" (von Mises Fisher)
    d : int
        Sphere dimension
    sample_size : int
        Number of points per sample
    params : dict, optional
        Distribution-specific parameters
        - vmf: {"mu": array, "kappa": float}

    Returns
    -------
    sampler : callable
        Function that takes seed and returns sample array
    """
    params = params or {}

    if distribution == "uniform":
        return lambda seed: random_points_on_sphere(sample_size, d, seed=seed)

    elif distribution == "vmf":
        mu = np.array(params.get("mu", [1.0] + [0.0] * d))
        kappa = params.get("kappa", 1.0)
        return lambda seed: sample_von_mises_fisher(sample_size, d, mu, kappa, seed=seed)

    else:
        raise ValueError(f"Unknown distribution: {distribution}")


def run_single_simulation(config: SimulationConfig, master_seed: int,
                          show_progress: bool = True) -> SimulationResult:
    """
    Run a single simulation scenario.

    Parameters
    ----------
    config : SimulationConfig
        Simulation configuration
    master_seed : int
        Master seed for reproducibility
    show_progress : bool
        Whether to show progress bar

    Returns
    -------
    result : SimulationResult
        Simulation results including test statistic values
    """
    # Create samplers
    sampler_X = create_sampler(
        config.distribution_X, config.d, config.m, config.dist_params_X
    )
    sampler_Y = create_sampler(
        config.distribution_Y, config.d, config.n, config.dist_params_Y
    )

    # Create optimized calculator
    sph_config = SphericalTestConfig(config.p, config.d)
    calculator = OptimizedTestStatistic(sph_config)

    T_values = np.zeros(config.num_simulations)

    iterator = range(config.num_simulations)
    if show_progress:
        iterator = tqdm(iterator, desc=f"  {config.name}")

    for i in iterator:
        # Derive seeds from master seed and iteration
        seed_X = master_seed + i * 2
        seed_Y = master_seed + i * 2 + 1

        X = sampler_X(seed=seed_X)
        Y = sampler_Y(seed=seed_Y)

        T = calculator.compute(X, Y, use_unbiased=config.use_unbiased)
        T_values[i] = T

    # Compute summary statistics
    mean = np.mean(T_values)
    std = np.std(T_values, ddof=1)
    skewness = stats.skew(T_values)
    kurtosis = stats.kurtosis(T_values)
    ks_stat, ks_pval = kstest(T_values, 'norm', args=(0, 1))

    return SimulationResult(
        config=config,
        T_values=T_values,
        master_seed=master_seed,
        timestamp=datetime.now().isoformat(),
        mean=mean,
        std=std,
        skewness=skewness,
        kurtosis=kurtosis,
        ks_statistic=ks_stat,
        ks_pvalue=ks_pval
    )


def run_simulation_grid(configs: List[SimulationConfig], master_seed: int,
                        show_progress: bool = True) -> List[SimulationResult]:
    """
    Run multiple simulation scenarios.

    Parameters
    ----------
    configs : list of SimulationConfig
        List of simulation configurations
    master_seed : int
        Master seed for reproducibility
    show_progress : bool
        Whether to show progress bars

    Returns
    -------
    results : list of SimulationResult
        List of simulation results
    """
    results = []

    for i, config in enumerate(configs):
        print(f"\n[{i+1}/{len(configs)}] Running: {config.name}")

        # Each config gets a different derived seed
        config_seed = master_seed + i * 1_000_000

        result = run_single_simulation(config, config_seed, show_progress)
        results.append(result)

        print(f"    Mean: {result.mean:.4f}, Std: {result.std:.4f}, "
              f"KS p-value: {result.ks_pvalue:.4f}")

    return results


# =============================================================================
# Save/Load Functions
# =============================================================================

def save_results(results: List[SimulationResult], filepath: str) -> None:
    """
    Save simulation results to a .npz file.

    Parameters
    ----------
    results : list of SimulationResult
        Simulation results to save
    filepath : str
        Output file path (should end in .npz)
    """
    # Prepare data for saving
    data = {
        'num_results': len(results),
    }

    for i, result in enumerate(results):
        prefix = f'result_{i}_'

        # Save config as JSON string
        config_dict = asdict(result.config)
        data[prefix + 'config'] = json.dumps(config_dict)

        # Save T_values
        data[prefix + 'T_values'] = result.T_values

        # Save metadata
        data[prefix + 'master_seed'] = result.master_seed
        data[prefix + 'timestamp'] = result.timestamp
        data[prefix + 'mean'] = result.mean
        data[prefix + 'std'] = result.std
        data[prefix + 'skewness'] = result.skewness
        data[prefix + 'kurtosis'] = result.kurtosis
        data[prefix + 'ks_statistic'] = result.ks_statistic
        data[prefix + 'ks_pvalue'] = result.ks_pvalue

    np.savez_compressed(filepath, **data)
    print(f"Saved {len(results)} results to {filepath}")


def load_results(filepath: str) -> List[SimulationResult]:
    """
    Load simulation results from a .npz file.

    Parameters
    ----------
    filepath : str
        Input file path

    Returns
    -------
    results : list of SimulationResult
        Loaded simulation results
    """
    data = np.load(filepath, allow_pickle=True)

    num_results = int(data['num_results'])
    results = []

    for i in range(num_results):
        prefix = f'result_{i}_'

        # Load config
        config_dict = json.loads(str(data[prefix + 'config']))
        config = SimulationConfig(**config_dict)

        # Load T_values and metadata
        result = SimulationResult(
            config=config,
            T_values=data[prefix + 'T_values'],
            master_seed=int(data[prefix + 'master_seed']),
            timestamp=str(data[prefix + 'timestamp']),
            mean=float(data[prefix + 'mean']),
            std=float(data[prefix + 'std']),
            skewness=float(data[prefix + 'skewness']),
            kurtosis=float(data[prefix + 'kurtosis']),
            ks_statistic=float(data[prefix + 'ks_statistic']),
            ks_pvalue=float(data[prefix + 'ks_pvalue'])
        )
        results.append(result)

    print(f"Loaded {len(results)} results from {filepath}")
    return results


# =============================================================================
# Plotting Functions
# =============================================================================

def plot_kde_grid(results: List[SimulationResult],
                  nrows: int = None, ncols: int = None,
                  figsize: Tuple[float, float] = None,
                  x_range: Tuple[float, float] = (-4, 4),
                  auto_scale_columns: bool = True,
                  save_path: str = None,
                  title: str = None,
                  col_headers: List[str] = None,
                  row_labels: List[str] = None) -> plt.Figure:
    """
    Create a grid of KDE plots comparing test statistic distribution to N(0,1).

    Style matches the reference image: only KDE and N(0,1) PDF, no histogram.

    Parameters
    ----------
    results : list of SimulationResult
        Simulation results to plot. Should be ordered row-by-row
        (i.e., results[0:ncols] is first row, results[ncols:2*ncols] is second row, etc.)
    nrows, ncols : int, optional
        Grid dimensions. If not specified, will be computed automatically.
    figsize : tuple, optional
        Figure size (width, height) in inches
    x_range : tuple
        Default range for x-axis. Used when auto_scale_columns=False, or as
        minimum range when auto_scale_columns=True.
    auto_scale_columns : bool
        If True, each column gets its own x-range based on the data in that
        column, while ensuring N(0,1) is always visible (at least -3 to 3).
        Default is True.
    save_path : str, optional
        If provided, save figure to this path
    title : str, optional
        Overall figure title
    col_headers : list of str, optional
        Headers for each column (displayed above first row)
    row_labels : list of str, optional
        Labels for each row (displayed on left side)

    Returns
    -------
    fig : matplotlib.Figure
        The created figure
    """
    n = len(results)

    # Determine grid dimensions
    if nrows is None and ncols is None:
        ncols = min(4, n)
        nrows = (n + ncols - 1) // ncols
    elif nrows is None:
        nrows = (n + ncols - 1) // ncols
    elif ncols is None:
        ncols = (n + nrows - 1) // nrows

    # Determine figure size
    if figsize is None:
        figsize = (3.5 * ncols, 2.8 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)

    # Handle single row/column cases
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes.reshape(1, -1)
    elif ncols == 1:
        axes = axes.reshape(-1, 1)

    # Compute x-range per column if auto-scaling
    if auto_scale_columns:
        col_x_ranges = []
        for col in range(ncols):
            # Gather all T_values for this column
            col_min = -3.5  # Ensure N(0,1) is visible (covers ~99.9%)
            col_max = 3.5
            for row in range(nrows):
                idx = row * ncols + col
                if idx < n:
                    T_vals = results[idx].T_values
                    col_min = min(col_min, np.min(T_vals) - 0.5)
                    col_max = max(col_max, np.max(T_vals) + 0.5)
            # Round to nice values
            col_min = np.floor(col_min)
            col_max = np.ceil(col_max)
            col_x_ranges.append((col_min, col_max))
    else:
        col_x_ranges = [x_range] * ncols

    for idx, result in enumerate(results):
        row = idx // ncols
        col = idx % ncols
        ax = axes[row, col]

        # Get x-range for this column
        col_x_range = col_x_ranges[col]

        # X values for plotting (use column-specific range)
        x = np.linspace(col_x_range[0], col_x_range[1], 500)

        # Standard normal PDF
        normal_pdf = norm.pdf(x, 0, 1)

        # Compute KDE
        T_values = result.T_values
        kde = gaussian_kde(T_values)
        kde_values = kde(x)

        # Plot N(0,1) PDF (red, solid)
        ax.plot(x, normal_pdf, 'r-', linewidth=1.5, label='N(0,1)')

        # Plot KDE (blue, solid)
        ax.plot(x, kde_values, 'b-', linewidth=1.5, label='KDE')

        # Format subplot
        config = result.config
        subplot_title = f"m={config.m}, n={config.n}, p={config.p}"
        ax.set_title(subplot_title, fontsize=9)
        ax.set_xlim(col_x_range)
        ax.set_ylim(0, 0.5)

        # Minimal tick formatting
        ax.tick_params(axis='both', labelsize=8)
        ax.set_yticks([0, 0.2, 0.4])

        # Only show x-axis label on bottom row
        if row == nrows - 1:
            ax.set_xlabel('T', fontsize=9)

    # Hide empty subplots
    for idx in range(n, nrows * ncols):
        row = idx // ncols
        col = idx % ncols
        axes[row, col].set_visible(False)

    # Add column headers (above first row)
    if col_headers:
        for col, header in enumerate(col_headers[:ncols]):
            axes[0, col].annotate(
                header, xy=(0.5, 1.15), xycoords='axes fraction',
                ha='center', va='bottom', fontsize=10, fontweight='bold'
            )

    # Add row labels (on the left side)
    if row_labels:
        for row, label in enumerate(row_labels[:nrows]):
            axes[row, 0].set_ylabel(label, fontsize=9)

    # Add legend to bottom-right subplot
    if n > 0:
        axes[nrows-1, ncols-1].legend(loc='upper right', fontsize=7)

    # Add overall title
    if title:
        fig.suptitle(title, fontsize=12, fontweight='bold', y=1.02)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved figure to {save_path}")

    return fig


def print_summary_table(results: List[SimulationResult]) -> None:
    """
    Print a summary table of simulation results.

    Parameters
    ----------
    results : list of SimulationResult
        Simulation results to summarize
    """
    print("\n" + "=" * 100)
    print("SIMULATION RESULTS SUMMARY")
    print("=" * 100)
    print(f"{'Name':<30} {'d':>3} {'p':>4} {'m':>6} {'n':>6} "
          f"{'Mean':>8} {'Std':>8} {'Skew':>8} {'Kurt':>8} {'KS p-val':>10}")
    print("-" * 100)

    for result in results:
        c = result.config
        print(f"{c.name:<30} {c.d:>3} {c.p:>4} {c.m:>6} {c.n:>6} "
              f"{result.mean:>8.4f} {result.std:>8.4f} "
              f"{result.skewness:>8.4f} {result.kurtosis:>8.4f} "
              f"{result.ks_pvalue:>10.4f}")

    print("-" * 100)
    print(f"{'Expected for N(0,1):':<30} {'':>3} {'':>4} {'':>6} {'':>6} "
          f"{'0.0000':>8} {'1.0000':>8} {'0.0000':>8} {'0.0000':>8} {'> 0.05':>10}")
    print("=" * 100)


# =============================================================================
# Main Simulation Functions
# =============================================================================

def run_four_scenario_grid(
    master_seed: int,
    d: int = 2,
    num_simulations: int = 500,
    output_dir: str = "simulation_results"
) -> Tuple[List[SimulationResult], plt.Figure]:
    """
    Run the main 4x4 simulation grid for the spherical two-sample test.

    Grid structure:
    - Columns (4 scenarios):
        1. Uniform-Uniform (H₀ true)
        2. vMF-vMF same parameters (H₀ true)
        3. vMF-vMF different means (H₁ - location alternative)
        4. vMF-vMF different κ (H₁ - dispersion alternative)

    - Rows (4 sample size combinations with m = 2n):
        (m, n) = (100, 50), (500, 250), (1000, 500), (2000, 1000)
        p = round(log(N)^1.5) where N = m + n

    Parameters
    ----------
    master_seed : int
        Master seed for full reproducibility
    d : int
        Sphere dimension (default: 2)
    num_simulations : int
        Number of Monte Carlo replications per cell (default: 500)
    output_dir : str
        Directory to save results and plots

    Returns
    -------
    results : list of SimulationResult
        All 16 simulation results (row-major order)
    fig : matplotlib.Figure
        The grid plot figure
    """
    os.makedirs(output_dir, exist_ok=True)

    # Sample size configurations: (m, n, p) with m = 2n
    # p values chosen to scale reasonably with sample size
    sample_configs = [
        (100, 50, 3),     # N=150, p=3
        (250, 125, 5),    # N=375, p=5
        (500, 250, 7),    # N=750, p=7
        (1000, 500, 9)    # N=1500, p=9
    ]

    # Extract sample_sizes for compatibility
    sample_sizes = [(m, n) for m, n, p in sample_configs]

    # Lookup p from config
    def compute_p(m, n):
        for m_, n_, p in sample_configs:
            if m == m_ and n == n_:
                return p
        raise ValueError(f"No p value configured for (m={m}, n={n})")

    # vMF parameters
    # Base mean direction
    mu_base = np.zeros(d + 1)
    mu_base[0] = 1.0

    # Rotated mean (7 degrees rotation in first two coordinates)
    angle_rad = np.radians(7.0)
    mu_rotated = np.zeros(d + 1)
    mu_rotated[0] = np.cos(angle_rad)
    mu_rotated[1] = np.sin(angle_rad)

    # Concentration parameters
    kappa_base = 10.0  # For vMF same and vMF different means
    kappa_X_disp = 5.0   # For dispersion alternative (less concentrated)
    kappa_Y_disp = 8.75  # For dispersion alternative (more concentrated, ratio=1.75)

    # Scale kappa by dimension for d > 2
    kappa_scale = d / 2 if d > 2 else 1.0
    kappa_base *= kappa_scale
    kappa_X_disp *= kappa_scale
    kappa_Y_disp *= kappa_scale

    print("=" * 80)
    print("FOUR-SCENARIO SIMULATION GRID")
    print("=" * 80)
    print(f"Master seed: {master_seed}")
    print(f"Sphere dimension: d = {d}")
    print(f"Simulations per cell: {num_simulations}")
    print(f"\nSample sizes (m, n) with m = 2n:")
    for m, n in sample_sizes:
        p = compute_p(m, n)
        print(f"  (m={m}, n={n}), N={m+n}, p={p}")
    print(f"\nvMF parameters:")
    print(f"  Base κ: {kappa_base:.1f}")
    print(f"  Dispersion alternative: κ_X={kappa_X_disp:.1f}, κ_Y={kappa_Y_disp:.1f}")
    print(f"  Mean rotation: 7 degrees")
    print("=" * 80)

    # Define the four scenarios
    scenario_names = [
        "Uniform (H₀)",
        "vMF Same (H₀)",
        "vMF Δμ (H₁)",
        "vMF Δκ (H₁)"
    ]

    def make_scenario_config(scenario_idx, m, n, p, row_idx):
        """Create a SimulationConfig for a given scenario and sample size."""
        base_name = f"S{scenario_idx+1}_m{m}_n{n}_p{p}"

        if scenario_idx == 0:
            # Scenario 1: Uniform-Uniform (H_0)
            return SimulationConfig(
                name=base_name,
                p=p, d=d, m=m, n=n,
                num_simulations=num_simulations,
                use_unbiased=True,
                distribution_X="uniform",
                distribution_Y="uniform"
            )
        elif scenario_idx == 1:
            # Scenario 2: vMF-vMF same parameters (H_0)
            return SimulationConfig(
                name=base_name,
                p=p, d=d, m=m, n=n,
                num_simulations=num_simulations,
                use_unbiased=True,
                distribution_X="vmf",
                distribution_Y="vmf",
                dist_params_X={"mu": mu_base.tolist(), "kappa": kappa_base},
                dist_params_Y={"mu": mu_base.tolist(), "kappa": kappa_base}
            )
        elif scenario_idx == 2:
            # Scenario 3: vMF-vMF different means (H_1 - location)
            return SimulationConfig(
                name=base_name,
                p=p, d=d, m=m, n=n,
                num_simulations=num_simulations,
                use_unbiased=True,
                distribution_X="vmf",
                distribution_Y="vmf",
                dist_params_X={"mu": mu_base.tolist(), "kappa": kappa_base},
                dist_params_Y={"mu": mu_rotated.tolist(), "kappa": kappa_base}
            )
        else:
            # Scenario 4: vMF-vMF different kappa (H_1 - dispersion)
            return SimulationConfig(
                name=base_name,
                p=p, d=d, m=m, n=n,
                num_simulations=num_simulations,
                use_unbiased=True,
                distribution_X="vmf",
                distribution_Y="vmf",
                dist_params_X={"mu": mu_base.tolist(), "kappa": kappa_X_disp},
                dist_params_Y={"mu": mu_base.tolist(), "kappa": kappa_Y_disp}
            )

    # Build all configurations (row-major order: iterate rows, then columns)
    configs = []
    for row_idx, (m, n) in enumerate(sample_sizes):
        p = compute_p(m, n)
        for scenario_idx in range(4):
            config = make_scenario_config(scenario_idx, m, n, p, row_idx)
            configs.append(config)

    print(f"\nTotal cells: {len(configs)}")

    # Run all simulations
    results = run_simulation_grid(configs, master_seed, show_progress=True)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = os.path.join(output_dir, f"four_scenario_d{d}_{timestamp}.npz")
    save_results(results, results_path)

    # Print summary table
    print_summary_table(results)

    # Create row labels
    row_labels = [f"m={m}, n={n}" for m, n in sample_sizes]

    # Create the grid plot
    fig = plot_kde_grid(
        results,
        nrows=4, ncols=4,
        figsize=(14, 11),
        title=None,  # No main figure title
        col_headers=scenario_names,
        row_labels=row_labels,
        save_path=os.path.join(output_dir, f"four_scenario_d{d}_{timestamp}.png")
    )

    # Also save a PDF version
    fig.savefig(
        os.path.join(output_dir, f"four_scenario_d{d}_{timestamp}.pdf"),
        dpi=300, bbox_inches='tight'
    )
    print(f"Saved PDF to {output_dir}/four_scenario_d{d}_{timestamp}.pdf")

    return results, fig


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    # =========================================================================
    # CONFIGURATION
    # =========================================================================
    # Master seed for all simulations - change this to get different results
    MASTER_SEED = 5086

    # Output directory
    OUTPUT_DIR = "simulation_results"

    # Sphere dimension
    D = 2

    # Number of replications per scenario (subplot)
    NUM_SIMS = 500

    # =========================================================================
    # RUN SIMULATIONS
    # =========================================================================
    print("\n" + "#" * 80)
    print("# T_p_mn TWO-SAMPLE TEST SIMULATIONS")
    print("#" * 80)
    print(f"\nMaster Seed: {MASTER_SEED}")
    print(f"Output Directory: {OUTPUT_DIR}")
    print(f"Sphere Dimension: d = {D}")
    print(f"Simulations per cell: {NUM_SIMS}")

    # Run the main four-scenario grid
    results, fig = run_four_scenario_grid(
        master_seed=MASTER_SEED,
        d=D,
        num_simulations=NUM_SIMS,
        output_dir=OUTPUT_DIR
    )

    plt.show()

    print("\n" + "#" * 80)
    print("# SIMULATIONS COMPLETE")
    print("#" * 80)
