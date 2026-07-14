#!/usr/bin/env python3
"""Evaluation script for MetaDNS Potts model.

Loads a trained checkpoint, generates samples, generates SW ground truth,
and computes all 5 rubric metrics:
  - Mag. (magnetization): fraction of spins in majority state, SNIS-weighted
  - Corr. (2-point correlation): same-state probability at distance 1
  - NESS: normalized effective sample size
  - E. JS Div.: Jensen-Shannon divergence of energy distributions vs SW
  - CV JS Div.: Jensen-Shannon divergence of 2D CV distributions vs SW
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

# Add repo root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from model import ExponentialMovingAverage, get_rope_vit_model
from utils import ess
from utils_potts import potts2d_ham, potts2d_magnetization_all
from utils_train import _compute_log_stats, rnd
from bias import BiasPotentialMultiDim
from baselines.swendsen_wang_sampling import swendsen_wang_step_potts


def potts2d_2pt_corr_same_state(S, rx, ry):
    """Compute same-state 2-point correlation for Potts model.

    C(r) = P(spins at distance r are in same state)

    Args:
        S: tensor of shape (B, L*L) or (B, L, L) with values in {0,..,q-1}
        rx, ry: int, horizontal and vertical distance

    Returns:
        float: average same-state probability over all valid pairs
    """
    if S.ndim == 2:
        B, L2 = S.shape
        L = int(np.sqrt(L2))
        S = S.view(B, L, L)
    else:
        B, L, L = S.shape

    # Periodic boundary conditions
    S_shifted_x = torch.roll(S, shifts=rx, dims=2)  # shift horizontally
    S_shifted_y = torch.roll(S, shifts=ry, dims=1)  # shift vertically

    # Same-state indicator for both directions
    same_x = (S == S_shifted_x).float()
    same_y = (S == S_shifted_y).float()

    # Average over all sites and batches
    corr_x = same_x.mean().item()
    corr_y = same_y.mean().item()

    return (corr_x + corr_y) / 2.0


def compute_potts_cv(x, q=3):
    """Compute 2D concentration projection CV for Potts model.

    Args:
        x: numpy array of shape (B, D) with values in {0,..,q-1}
        q: number of Potts states

    Returns:
        cv: numpy array of shape (B, 2)
    """
    if x.ndim == 1:
        x = x.reshape(1, -1)

    B, D = x.shape
    counts = np.zeros((B, q))
    for i in range(q):
        counts[:, i] = np.sum(x == i, axis=1)

    concentrations = counts / D

    if q == 3:
        c1 = concentrations[:, 0]
        c2 = concentrations[:, 1]
        c3 = concentrations[:, 2]
        proj_x = c1 - 0.5 * (c2 + c3)
        proj_y = (np.sqrt(3)/2) * (c2 - c3)
        cv = np.stack([proj_x, proj_y], axis=1)
        return cv
    else:
        return concentrations[:, :-1]


def compute_js_divergence(samples_a, samples_b, bins=100, weights_a=None, weights_b=None):
    """Compute Jensen-Shannon divergence between two 1D sample sets.

    Args:
        samples_a, samples_b: 1D numpy arrays
        bins: number of bins for histogram
        weights_a, weights_b: optional importance weights

    Returns:
        float: JS divergence
    """
    # Determine common range
    all_samples = np.concatenate([samples_a, samples_b])
    range_min = all_samples.min()
    range_max = all_samples.max()

    # Build histograms
    hist_a, _ = np.histogram(samples_a, bins=bins, range=(range_min, range_max),
                              weights=weights_a, density=True)
    hist_b, _ = np.histogram(samples_b, bins=bins, range=(range_min, range_max),
                              weights=weights_b, density=True)

    # Normalize
    bin_width = (range_max - range_min) / bins
    hist_a = hist_a * bin_width
    hist_b = hist_b * bin_width

    # Ensure non-negative and valid
    hist_a = np.maximum(hist_a, 1e-30)
    hist_b = np.maximum(hist_b, 1e-30)
    hist_a = hist_a / hist_a.sum()
    hist_b = hist_b / hist_b.sum()

    # JS divergence
    m = (hist_a + hist_b) / 2.0
    kl_am = np.sum(hist_a * np.log(hist_a / m))
    kl_bm = np.sum(hist_b * np.log(hist_b / m))
    js_div = (kl_am + kl_bm) / 2.0

    return js_div


def compute_cv_js_divergence(samples_a, samples_b, grid_size=17,
                              cv_min=(-0.6, -1.0), cv_max=(1.1, 1.0),
                              q=3, weights_a=None, weights_b=None):
    """Compute JS divergence of 2D CV distributions.

    Args:
        samples_a, samples_b: numpy arrays of shape (B, D) with configs
        grid_size: int, number of bins per CV dimension
        cv_min, cv_max: tuples of CV bounds
        q: number of Potts states
        weights_a, weights_b: optional importance weights

    Returns:
        float: JS divergence
    """
    cv_a = compute_potts_cv(samples_a, q)
    cv_b = compute_potts_cv(samples_b, q)

    # Build 2D histograms
    hist_a, _, _ = np.histogram2d(
        cv_a[:, 0], cv_a[:, 1], bins=grid_size,
        range=[[cv_min[0], cv_max[0]], [cv_min[1], cv_max[1]]],
        weights=weights_a, density=True
    )
    hist_b, _, _ = np.histogram2d(
        cv_b[:, 0], cv_b[:, 1], bins=grid_size,
        range=[[cv_min[0], cv_max[0]], [cv_min[1], cv_max[1]]],
        weights=weights_b, density=True
    )

    # Compute bin areas
    dx = (cv_max[0] - cv_min[0]) / grid_size
    dy = (cv_max[1] - cv_min[1]) / grid_size
    bin_area = dx * dy

    hist_a = hist_a * bin_area
    hist_b = hist_b * bin_area

    # Flatten and normalize
    hist_a_flat = hist_a.flatten()
    hist_b_flat = hist_b.flatten()
    hist_a_flat = np.maximum(hist_a_flat, 1e-30)
    hist_b_flat = np.maximum(hist_b_flat, 1e-30)
    hist_a_flat = hist_a_flat / hist_a_flat.sum()
    hist_b_flat = hist_b_flat / hist_b_flat.sum()

    # JS divergence
    m = (hist_a_flat + hist_b_flat) / 2.0
    kl_am = np.sum(hist_a_flat * np.log(hist_a_flat / m))
    kl_bm = np.sum(hist_b_flat * np.log(hist_b_flat / m))
    js_div = (kl_am + kl_bm) / 2.0

    return js_div


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Potts MetaDNS model")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--L", type=int, default=4, help="Lattice size")
    parser.add_argument("--q", type=int, default=3, help="Number of Potts states")
    parser.add_argument("--beta", type=float, default=1.2, help="Inverse temperature")
    parser.add_argument("--J", type=float, default=1.0, help="Coupling constant")
    parser.add_argument("--num-samples", type=int, default=10000, help="Number of samples")
    parser.add_argument("--batch-size", type=int, default=1024, help="Sampling batch size")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output-dir", type=str, default="runs/eval_potts")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sw-blocks", type=int, default=40, help="SW blocks")
    parser.add_argument("--sw-steps-per-block", type=int, default=500, help="SW steps per block")
    parser.add_argument("--sw-burn-in", type=int, default=2048, help="SW burn-in steps")
    parser.add_argument("--sw-n-configs", type=int, default=10240, help="SW total configs")
    return parser.parse_args()


def main():
    args = parse_args()

    device = args.device
    L = args.L
    D = L * L
    q = args.q
    beta = args.beta
    J = args.J
    num_samples = args.num_samples
    batch_size = args.batch_size

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ============================================================
    # Step 1: Load trained model
    # ============================================================
    print(f"Loading checkpoint: {args.ckpt}")
    checkpoint = torch.load(args.ckpt, map_location=device)
    cfg = checkpoint.get('cfg', {})

    model = get_rope_vit_model(
        L, embed_dim=cfg.get('model', {}).get('hidden_size', 128),
        depth=cfg.get('model', {}).get('n_blocks', 4),
        num_heads=cfg.get('model', {}).get('n_heads', 4),
        vocab_size=q + 1,
        dtype=cfg.get('model', {}).get('dtype', 'bfloat16'),
        device=device
    )
    ema = ExponentialMovingAverage(model.parameters(), decay=0.9999)
    model.load_state_dict(checkpoint['model_state_dict'])
    if 'ema_state_dict' in checkpoint:
        ema.load_state_dict(checkpoint['ema_state_dict'])
    model.eval()
    print(f"Model loaded. Params: {sum(p.numel() for p in model.parameters())}")

    # Load bias potential
    bias_pot = None
    use_bias = cfg.get('use_bias', False)
    if use_bias:
        cv_min = [float(x) for x in cfg.get('cv_min', '-0.6,-1.0').split(',')]
        cv_max = [float(x) for x in cfg.get('cv_max', '1.1,1.0').split(',')]
        grid_size = [int(x) for x in str(cfg.get('bias_grid_size', '17')).split(',')]
        sigma = [float(x) for x in str(cfg.get('bias_sigma', '0.05')).split(',')]

        bias_height = cfg.get('bias_height', 0.0833)
        train_batch_size = cfg.get('batch_size', 128)
        effective_height = bias_height / train_batch_size

        bias_pot = BiasPotentialMultiDim(
            cv_min=cv_min, cv_max=cv_max,
            grid_size=grid_size, sigma=sigma,
            initial_height=effective_height,
            bias_factor=cfg.get('bias_factor', 10.0),
            T=1.0/beta,
            kernel_type=cfg.get('kernel_type', 'gaussian'),
            device=device,
            energy_scaling=1.0
        )
        if 'bias_potential' in checkpoint:
            bias_pot.load_state_dict(checkpoint['bias_potential'])
        print(f"Bias potential loaded.")

    # ============================================================
    # Step 2: Define reward and CV functions
    # ============================================================
    def reward_fn_potts(S, beta_val=beta, J_val=J, q_val=q):
        return -beta_val * potts2d_ham(S, J_val, q_val)

    def compute_cv(x):
        if isinstance(x, torch.Tensor):
            x_np = x.detach().cpu().numpy()
        else:
            x_np = x
        if x_np.ndim == 1:
            x_np = x_np.reshape(1, -1)
        B, D_ = x_np.shape
        counts = np.zeros((B, q))
        for i in range(q):
            counts[:, i] = np.sum(x_np == i, axis=1)
        concentrations = counts / D_
        c1, c2, c3 = concentrations[:, 0], concentrations[:, 1], concentrations[:, 2]
        proj_x = c1 - 0.5 * (c2 + c3)
        proj_y = (np.sqrt(3)/2) * (c2 - c3)
        cv = np.stack([proj_x, proj_y], axis=1)
        return torch.tensor(cv, device=device, dtype=torch.float32)

    def reward_fn_biased(x, beta_val=beta, use_bias_val=True):
        r = reward_fn_potts(x, beta_val=beta_val)
        if use_bias_val and bias_pot is not None:
            s = compute_cv(x)
            v = bias_pot.evaluate(s)
            r = r - beta_val * v
        return r

    # ============================================================
    # Step 3: Generate samples from trained model
    # ============================================================
    print(f"\nGenerating {num_samples} samples from MetaDNS model...")

    all_configs = []
    all_energies = []
    all_log_rnd = []
    all_log_rw = []
    all_weights = []

    # Use EMA for sampling
    with torch.no_grad():
        ema.store(model.parameters())
        ema.copy_to(model.parameters())

    n_batches = (num_samples + batch_size - 1) // batch_size
    for _ in tqdm(range(n_batches)):
        actual_batch = min(batch_size, num_samples - len(all_configs))
        if actual_batch <= 0:
            break

        x, log_rnd_vals = rnd(model, lambda x, **kw: reward_fn_biased(x, beta_val=beta, use_bias_val=use_bias),
                              actual_batch, device=device, J=J)

        x_long = x.long()
        energies = potts2d_ham(x_long, J, q)
        log_rw_vals = -beta * energies + log_rnd_vals

        logf_t_vals, logp_x_vals = _compute_log_stats(
            x_long, log_rnd_vals,
            lambda x, **kw: reward_fn_potts(x),
            model, J=J, bias_potential=bias_pot, cv_compute_fn=compute_cv
        )

        all_configs.append(x_long.cpu().numpy())
        all_energies.append(energies.detach().cpu().numpy())
        all_log_rnd.append(log_rnd_vals.detach().cpu().numpy())
        all_log_rw.append(log_rw_vals.detach().cpu().numpy())

        if bias_pot is not None:
            s = compute_cv(x_long)
            bias_vals = bias_pot.evaluate(s)
            w = torch.exp(beta * bias_vals)
            all_weights.append(w.cpu().numpy())

    with torch.no_grad():
        ema.restore(model.parameters())

    configs = np.concatenate(all_configs, axis=0)[:num_samples]
    energies = np.concatenate(all_energies, axis=0)[:num_samples]
    log_rnd = np.concatenate(all_log_rnd, axis=0)[:num_samples]
    log_rw = np.concatenate(all_log_rw, axis=0)[:num_samples]

    if all_weights:
        weights = np.concatenate(all_weights, axis=0)[:num_samples]
        weights = weights / weights.sum()
    else:
        weights = np.ones(num_samples) / num_samples

    print(f"Generated {len(configs)} samples")
    print(f"Energy range: [{energies.min():.2f}, {energies.max():.2f}]")

    # ============================================================
    # Step 4: Generate SW ground truth samples using CLI
    # ============================================================
    print(f"\nGenerating SW ground truth samples via CLI...")
    import subprocess
    import tempfile
    import pickle as pkl

    T_val = 1.0 / beta
    sw_output_dir = output_dir / "sw_samples"
    sw_output_dir.mkdir(parents=True, exist_ok=True)

    # Use the SW sampler CLI
    sw_cmd = [
        sys.executable, str(Path(__file__).resolve().parent / "baselines" / "swendsen_wang_sampling.py"),
        "--model-type", "potts",
        "--dim", str(L),
        "--q", str(q),
        "--J", str(J),
        "--temps", str(T_val),
        "--chem-pots", "0.0",
        "--batch-size", str(min(1024, args.sw_n_configs)),
        "--samples-per-block", str(args.sw_n_configs),
        "--steps-per-block", str(args.sw_steps_per_block),
        "--num-blocks", str(1),  # single block with many samples
        "--burn-in", str(args.sw_burn_in),
        "--output-dir", str(sw_output_dir),
        "--seed", str(args.seed + 100),
        "--overwrite",
    ]
    repo_root = str(Path(__file__).resolve().parent)
    env = os.environ.copy()
    env['PYTHONPATH'] = repo_root + (':' + env.get('PYTHONPATH', '') if env.get('PYTHONPATH') else '')
    print(f"Running SW sampler: {' '.join(sw_cmd)}")
    result = subprocess.run(sw_cmd, capture_output=True, text=True, cwd=repo_root, env=env)
    if result.returncode != 0:
        print(f"SW sampler stderr: {result.stderr[-2000:]}")
        raise RuntimeError(f"SW sampler failed with code {result.returncode}")

    # Load SW output
    sw_block_files = sorted(sw_output_dir.glob("block_*.pkl"))
    if not sw_block_files:
        raise RuntimeError(f"No SW block files found in {sw_output_dir}")

    with open(sw_block_files[-1], 'rb') as f:
        sw_data = pkl.load(f)

    # SW sampler uses key format: f"{temp:.1f}K_mu{chem_pot:.2f}"
    key = f"{T_val:.1f}K_mu{0.0:.2f}"
    if key not in sw_data.get('configs', {}):
        available = list(sw_data.get('configs', {}).keys())
        if available:
            key = available[0]
            print(f"Using SW key: {key}")
        else:
            raise RuntimeError(f"No configs found in SW output. Keys: {list(sw_data.keys())}")

    sw_configs_flat = sw_data['configs'][key]  # shape: (n_samples, L*L)
    sw_configs_flat = sw_configs_flat[:args.sw_n_configs]
    sw_energies = sw_data.get('energies', {}).get(key, None)

    if sw_energies is None or len(sw_energies) == 0:
        # Compute energies ourselves
        print("Computing SW energies...")
        sw_energies = []
        for i in range(0, len(sw_configs_flat), 1024):
            batch = sw_configs_flat[i:i+1024]
            sw_energies.append(potts2d_ham(torch.tensor(batch).long(), J, q).cpu().numpy())
        sw_energies = np.concatenate(sw_energies)
    else:
        sw_energies = np.array(sw_energies[:args.sw_n_configs])

    print(f"SW samples: {len(sw_configs_flat)}, energy range: [{sw_energies.min():.2f}, {sw_energies.max():.2f}]")

    # ============================================================
    # Step 5: Compute metrics
    # ============================================================
    print(f"\nComputing metrics...")

    # ---- NESS ----
    log_rnd_tensor = torch.tensor(log_rnd)
    ness_val = ess(log_rnd_tensor, normalize=True)
    ness_val = float(ness_val) if isinstance(ness_val, (float, int)) else ness_val.item()
    print(f"NESS: {ness_val:.4f}")

    # ---- Mag. (magnetization, SNIS-weighted) ----
    mag_per_sample = potts2d_magnetization_all(configs, q)
    mag_weighted = np.average(mag_per_sample, weights=weights)
    print(f"Mag. (weighted): {mag_weighted:.4f}")

    # ---- SW Mag. for reference ----
    sw_mag_per_sample = potts2d_magnetization_all(sw_configs_flat, q)
    sw_mag = sw_mag_per_sample.mean()
    print(f"SW Mag.: {sw_mag:.4f}")

    # ---- Corr. (2-point correlation, SNIS-weighted) ----
    configs_t = torch.tensor(configs).long()
    # Compute per-sample correlation
    B = configs.shape[0]
    corr_per_sample = np.zeros(B)
    S_t = configs_t.view(B, L, L)
    for b in range(B):
        corr_per_sample[b] = potts2d_2pt_corr_same_state(S_t[b:b+1], rx=1, ry=0)

    corr_weighted = np.average(corr_per_sample, weights=weights)
    print(f"Corr. (weighted): {corr_weighted:.4f}")

    # ---- SW Corr. for reference ----
    sw_corr_per_sample = np.zeros(len(sw_configs_flat))
    sw_S_t = torch.tensor(sw_configs_flat).long().view(len(sw_configs_flat), L, L)
    for b in range(len(sw_configs_flat)):
        sw_corr_per_sample[b] = potts2d_2pt_corr_same_state(sw_S_t[b:b+1], rx=1, ry=0)
    sw_corr = sw_corr_per_sample.mean()
    print(f"SW Corr.: {sw_corr:.4f}")

    # ---- E. JS Div. (energy distribution JS divergence) ----
    e_js_div = compute_js_divergence(
        energies, sw_energies, bins=50, weights_a=weights
    )
    print(f"E. JS Div.: {e_js_div:.6f}")

    # ---- CV JS Div. (2D CV distribution JS divergence) ----
    cv_js_div = compute_cv_js_divergence(
        configs, sw_configs_flat, grid_size=17,
        cv_min=(-0.6, -1.0), cv_max=(1.1, 1.0), q=q,
        weights_a=weights
    )
    print(f"CV JS Div.: {cv_js_div:.6f}")

    # ============================================================
    # Step 6: Save results
    # ============================================================
    results = {
        "Mag.": float(mag_weighted),
        "Corr.": float(corr_weighted),
        "NESS": float(ness_val),
        "E. JS Div.": float(e_js_div),
        "CV JS Div.": float(cv_js_div),
        "SW_Mag.": float(sw_mag),
        "SW_Corr.": float(sw_corr),
        "num_samples": num_samples,
        "sw_num_configs": len(sw_configs_flat),
        "beta": beta,
        "L": L,
        "q": q,
    }

    results_path = output_dir / "metrics.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Results saved to {results_path}")
    print(f"{'='*60}")
    for key, val in results.items():
        if isinstance(val, float):
            print(f"  {key}: {val:.6f}")
        else:
            print(f"  {key}: {val}")

    return results


if __name__ == "__main__":
    main()
