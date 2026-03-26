# OLLA_NIPS/experiments/run_analysis.py

import os
import sys
import argparse
import importlib
import time
import numpy as np
import pandas as pd
import torch
import arviz as az
import matplotlib.pyplot as plt

from typing import List, Dict, Any
from torch.func import vmap

# --- Project Root Setup ---
# Add project root to path to allow importing 'src' modules.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- Import Sampler Classes ---
from src.samplers.olla import OLLA
from src.samplers.olla_h import OLLA_H
from src.samplers.clangevin import CLangevin
from src.samplers.chmc import CHMC
from src.samplers.cghmc import CGHMC

# --- Main Configuration ---
# Maps sampler names to their respective classes for dynamic loading.
SAMPLER_CLASSES = {
    'OLLA': OLLA,
    'OLLA-H': OLLA_H,
    'CLangevin': CLangevin,
    'CHMC': CHMC,
    'CGHMC': CGHMC,
}

# --- Task-Specific Test Functions for High-Dimensional Problems ---
def get_highdim_test_functions(constraint_name: str, constraint_module: Any) -> Dict[str, Any]:
    """Returns a dictionary of test functions based on the experiment name."""
    if constraint_name == 'highdim_stress':
        return {
            'P(x1>0)': lambda x: (x[..., 0] > 0).float().mean(),
            'E[sin(x1)exp(x2)+log|x3|tanh(x4)+prod(cos(x5-9))]': lambda x: torch.sin(x[..., 0]) * torch.exp(x[..., 1]) + torch.log(torch.abs(x[..., 2]) + 1) * torch.tanh(x[..., 3]) + torch.prod(torch.cos(x[..., 4:9]), dim=-1)
        }
    elif constraint_name == 'highdim_polymer':
        def radius_of_gyration(x: torch.Tensor) -> torch.Tensor:
            """Calculates the Radius of Gyration squared (Rg^2)."""
            n_atoms = constraint_module.N_ATOMS
            p = x.view(*x.shape[:-1], n_atoms, 3)
            p_cm = torch.mean(p, dim=-2, keepdim=True)
            return torch.mean(torch.sum((p - p_cm)**2, dim=-1), dim=-1)

        return {
            'Radius of Gyration (Rg2)': radius_of_gyration,
            'Square norm of positions': lambda x: torch.sum(x**2, dim=-1)
        }
    else:
        return {}

# --- Metric Calculation ---
def calculate_metrics(results: Dict[str, Any], is_single_chain: bool, constraint_module: Any) -> Dict[str, float]:
    """Calculates all specified metrics from the raw sampler output."""
    metrics = {}
    trajectory = results['trajectory']
    runtime = results['runtime']

    if is_single_chain:
        n_steps = trajectory.shape[0]
        start_step = int(n_steps * 0.2)
        final_samples = trajectory[start_step::5, 0, :]
    else:
        final_samples = trajectory[-1, :, :]

    if final_samples.shape[0] < 4: return {'error': 1.0}

    # 1. Effective Sample Size (ESS)
    if is_single_chain:
        ess_data = np.expand_dims(final_samples, axis=0)
    else:
        ess_data = trajectory.transpose(1, 0, 2)

    try:
        az_dataset = az.convert_to_dataset({'x': ess_data})
        ess_values = az.ess(az_dataset)
        metrics['ess'] = ess_values['x'].min().item()
    except Exception as e:
        print(f"  [WARN] ESS calculation failed for {results['constraint_name']}: {e}")
        metrics['ess'] = np.nan
    metrics['cpu_time'] = runtime

    # 2. Constraint Violations
    if 'h_vals' in results and results['h_vals'].size > 0:
        if is_single_chain:
            metrics['h_violation'] = np.mean(np.abs(results['h_vals'][start_step::5]))
        else:
            metrics['h_violation'] = np.mean(np.abs(results['h_vals']))
    if 'g_vals' in results and results['g_vals'].size > 0:
        if is_single_chain:
            metrics['g_violation'] = np.mean(np.max(np.maximum(0, results['g_vals'][start_step::5]), axis=-1))
        else:
            metrics['g_violation'] = np.mean(np.max(np.maximum(0, results['g_vals']), axis=-1))

    # 3. Problem-Specific Metrics
    samples_torch = torch.tensor(final_samples, dtype=torch.float64)
    if results['constraint_name'] == 'german_credit':
        logits = constraint_module._logits_torch(samples_torch, constraint_module._X_test, constraint_module._A_test)
        
        # Calculate mean log-likelihood per data point, not the sum.
        loglik = (torch.nn.functional.logsigmoid(logits) * constraint_module._y_test +
                  torch.nn.functional.logsigmoid(-logits) * (1.0 - constraint_module._y_test)).mean(dim=-1)
        metrics['test_nll'] = -loglik.mean().item()

    elif 'highdim' in results['constraint_name']:
        test_funcs = get_highdim_test_functions(results['constraint_name'], constraint_module)
        for name, func in test_funcs.items():
            metrics[f'test_func_{name}'] = func(samples_torch).mean().item()

    return metrics

# --- Plotting Functions ---
def plot_2d_results(df: pd.DataFrame, output_dir: str):
    """Generates and saves a grid plot for 2D experiment results."""
    constraints = sorted([c for c in df['constraint'].unique() if df[df['constraint'] == c]['dim'].iloc[0] == 2])
    if not constraints: return

    samplers = sorted(df['sampler'].unique())
    n_constraints = len(constraints)
    n_samplers = len(samplers)

    fig, axes = plt.subplots(n_constraints, n_samplers, figsize=(5 * n_samplers, 5 * n_constraints), sharex=True, sharey=True, squeeze=False)

    all_pts = np.vstack([np.vstack(df[df['constraint'] == c]['final_samples'].values) for c in constraints])
    xlim = (all_pts[:, 0].min() - 1, all_pts[:, 0].max() + 1)
    ylim = (all_pts[:, 1].min() - 1, all_pts[:, 1].max() + 1)
    
    colors = plt.get_cmap('tab10').colors
    color_map = {sampler: colors[i % len(colors)] for i, sampler in enumerate(samplers)}

    for i, cname in enumerate(constraints):
        cm = importlib.import_module(f'src.constraints.{cname}')
        xs, ys = np.linspace(xlim[0], xlim[1], 200), np.linspace(ylim[0], ylim[1], 200)
        XX, YY = np.meshgrid(xs, ys)
        grid_pts = np.stack([XX.ravel(), YY.ravel()], axis=1)

        for j, sampler in enumerate(samplers):
            ax = axes[i, j]

            if hasattr(cm, 'g_fns') and cm.g_fns:
                g_vals = np.stack([np.array([g(torch.tensor(p, dtype=torch.float64)).item() for p in grid_pts]) for g in cm.g_fns])
                mask = np.all(g_vals <= 0, axis=0).reshape(XX.shape)
                ax.contourf(XX, YY, mask, levels=[0.5, 1.5], colors=['#98FB98'], alpha=0.3)
                for g_val in g_vals: ax.contour(XX, YY, g_val.reshape(XX.shape), levels=[0], colors='k', linestyles='--')
            if hasattr(cm, 'h_fns') and cm.h_fns:
                for h in cm.h_fns:
                    h_val = np.array([h(torch.tensor(p, dtype=torch.float64)).item() for p in grid_pts])
                    ax.contour(XX, YY, h_val.reshape(XX.shape), levels=[0], colors='k', linestyles='-')

            sampler_data = df[(df['sampler'] == sampler) & (df['constraint'] == cname)]
            if not sampler_data.empty:
                samples = np.vstack(sampler_data['final_samples'].values)
                ax.scatter(samples[:, 0], samples[:, 1], s=20, alpha=0.7, color=color_map[sampler])
            
            ax.set_xlim(xlim); ax.set_ylim(ylim); ax.grid(True, ls='--', alpha=0.5)
            
            name_map = {
                'two_lobe': 'Two Lobes', 'star_shape': 'Star',
                'quadratic_poly': 'Quadratic Poly', 'mix_gaussian': 'Mixture Gaussian'
            }
            if j == 0:
                display_name = name_map.get(cname, cname)
                ax.set_ylabel(display_name, fontsize=16)
            if i == 0:
                ax.set_title(sampler, fontsize=16)

    fig.tight_layout()
    plt.savefig(os.path.join(output_dir, 'plot_2d_summary.pdf'), bbox_inches='tight')
    plt.close(fig)
    print(f"Saved 2D summary plot to {os.path.join(output_dir, 'plot_2d_summary.pdf')}")


# --- Main Runner ---
def main():
    parser = argparse.ArgumentParser(description="Run and analyze constrained sampling experiments.")
    parser.add_argument('--experiments', nargs='+', default=['circle', 'two_lobe', 'quadratic_poly', 'mix_gaussian','highdim_stress', 'highdim_polymer','german_credit'], help="List of experiments to run.")
    parser.add_argument('--samplers', nargs='+', default=['OLLA-H','CLangevin', 'CHMC', 'CGHMC',], help="List of samplers to test.")
    parser.add_argument('--output_dir', type=str, default='analysis_results', help="Directory to save plots and reports.")
    parser.add_argument('--device', type=str, default='cpu', help="Torch device ('cpu' or 'cuda').")
    parser.add_argument('--seed', type=int, default=1, help="Global random seed.")
    #---- highdim_stress params ----#
    parser.add_argument('--n_dim', type=int, default=1000, help="Number of dimensions for highdim_stress.")
    parser.add_argument('--n_eq', type=int, default=10, help="Number of equality constraints for highdim_stress.")
    parser.add_argument('--n_ineq', type=int, default=10, help="Number of inequality constraints for highdim_stress.")
    #---- highdim_polymer params ----#
    parser.add_argument('--natoms', type=int, default=5, help="Number of atoms in the polymer.")
    #---- german_credit params ----#
    parser.add_argument('--NN_dim', type=int, default=4994, help="Number of dimensions for the German Credit NN.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    all_metrics = []

    for exp_name in args.experiments:
        print(f"\n{'='*20} Running Experiment: {exp_name.upper()} {'='*20}")
        try:
            constraint_module = importlib.import_module(f'src.constraints.{exp_name}')
            exp_settings = getattr(constraint_module, 'EXPERIMENT_SETTINGS', {})
            sampler_hparams = getattr(constraint_module, 'SAMPLER_SETTINGS', {})
        
            if exp_name == "highdim_stress":
                constraint_module.build_constraints(n_dim = args.n_dim, n_eq = args.n_eq, n_ineq = args.n_ineq)
            elif exp_name == "highdim_polymer":
                constraint_module.build_constraints(n_atoms = args.natoms)
            elif exp_name == "german_credit":
                constraint_module.build_constraints(n_dim = args.NN_dim)
                
        except ImportError:
            print(f"  [ERROR] Could not import constraint module for '{exp_name}'. Skipping.")
            continue

        print(f"(d, m, l) = ({constraint_module.dim}, {len(constraint_module.h_fns)}, {len(constraint_module.g_fns)})")
        x0 = constraint_module.generate_samples(num_samples=exp_settings.get('n_particles', 1), seed=args.seed)
        
        for sampler_name in args.samplers:
            if sampler_name not in SAMPLER_CLASSES:
                print(f"  [WARN] Sampler '{sampler_name}' not found. Skipping.")
                continue

            print(f"  --- Sampler: {sampler_name} ---")
            
            sampler_class = SAMPLER_CLASSES[sampler_name]
            hparams = sampler_hparams.get(sampler_name, {})
            
            constraint_fns = {'h': getattr(constraint_module, 'h_fns', []), 'g': getattr(constraint_module, 'g_fns', [])}
            sampler = sampler_class(
                constraint_funcs=constraint_fns,
                num_steps=exp_settings.get('n_steps', 1000),
                seed=args.seed,
                device=torch.device(args.device),
                **hparams
            )
            
            param_string = ", ".join([f"{key}={value}" for key, value in hparams.items()])
            print(f"    > Hyperparameters: {param_string}")

            start_time = time.time()
            
            raw_results = sampler.sample(x0=x0, potential_fn=constraint_module.potential_fn)
            raw_results['runtime'] = time.time() - start_time
            raw_results['constraint_name'] = exp_name
            print(f"    ... completed in {raw_results['runtime']:.2f} seconds.")

            is_single = exp_settings.get('is_single_chain', False)
            metrics = calculate_metrics(raw_results, is_single, constraint_module)
            metrics.update({
                'constraint': exp_name,
                'sampler': sampler_name,
                'dim': constraint_module.dim,
                'final_samples': raw_results['trajectory'][-1] if not is_single else \
                                 raw_results['trajectory'][int(raw_results['trajectory'].shape[0]*0.2)::10, 0, :]
            })
            all_metrics.append(metrics)

    if not all_metrics:
        print("\nNo experiments were successfully run. Exiting.")
        return

    metrics_df = pd.DataFrame(all_metrics)
    
    summary = metrics_df.drop(columns=['final_samples', 'error'], errors='ignore').groupby(['constraint', 'sampler']).mean()
    print("\n--- Experiment Summary (Mean Metrics) ---")
    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 120):
        print(summary.to_string(float_format="%.3g"))
    
    summary.to_csv(os.path.join(args.output_dir, 'report_summary.csv'))
    print(f"\nSaved summary report to {os.path.join(args.output_dir, 'report_summary.csv')}")
    
    plot_2d_results(metrics_df, args.output_dir)

    print("\nAnalysis complete.")

if __name__ == '__main__':
    main()

