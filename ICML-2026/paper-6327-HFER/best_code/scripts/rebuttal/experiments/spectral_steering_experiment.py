"""
Spectral Steering Experiment
==============================
Addresses: Reviewer 6GDW ("does spectral steering work?")

Modifies SVD spectrum of mlp.down_proj at target layer, measures HFER change
and validity improvement. Requires GPU.

Usage:
    python scripts/rebuttal/spectral_steering_experiment.py --model meta-llama/Llama-3.2-3B-Instruct --load-in-4bit
"""

import os, sys, json, glob, argparse
import numpy as np
import torch
from tqdm import tqdm
from copy import deepcopy

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams.update({
    'font.family': 'serif', 'font.size': 10, 'figure.dpi': 300,
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.05,
})

def apply_spectral_steering(model, layer_idx, alpha=-0.3):
    """Apply spectral steering to mlp.down_proj at given layer.
    Moves to CPU for SVD stability.
    """
    device = model.model.layers[layer_idx].mlp.down_proj.weight.device
    dtype = model.model.layers[layer_idx].mlp.down_proj.weight.dtype
    W = model.model.layers[layer_idx].mlp.down_proj.weight.data.float().cpu()
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    S_new = S * (1 + alpha * (S - S.mean()) / S.std())
    W_new = U @ torch.diag(S_new) @ Vh
    model.model.layers[layer_idx].mlp.down_proj.weight.data = W_new.to(device=device, dtype=dtype)
    return model

def run_steering(args):
    print("=" * 70)
    print("  SPECTRAL STEERING EXPERIMENT")
    print("=" * 70)

    os.makedirs('output/rebuttal', exist_ok=True)
    os.makedirs('data/results/rebuttal', exist_ok=True)

    from spectral_trust import GSPDiagnosticsFramework, GSPConfig

    model_kwargs = {"output_attentions": True, "output_hidden_states": True}
    if args.load_in_4bit: model_kwargs["load_in_4bit"] = True

    config = GSPConfig(model_name=args.model, device=args.device,
                       model_kwargs=model_kwargs)

    # Select proofs: 20 invalid + 20 valid
    valid_files = sorted(glob.glob(os.path.join(args.data_dir, 'valid', '*.lean')))[:20]
    invalid_files = sorted(glob.glob(os.path.join(args.data_dir, 'invalid', '*.lean')))[:20]
    test_files = [(fp, 'valid') for fp in valid_files] + [(fp, 'invalid') for fp in invalid_files]

    # Phase 1: Base model
    print(f"\n--- PHASE 1: BASE MODEL ---")
    base_results = []
    with GSPDiagnosticsFramework(config) as framework:
        framework.instrumenter.load_model(args.model)
        num_layers = framework.instrumenter.model.config.num_hidden_layers
        target_layer = int(0.75 * num_layers)
        steer_layer = target_layer  # Steer at same layer we measure

        for fp, label in tqdm(test_files, desc='Base'):
            with open(fp, 'r', encoding='utf-8') as f: text = f.read()
            try:
                analysis = framework.analyze_text(text, save_results=False)
                layers = analysis.get('layer_diagnostics', [])
                hfer = float(getattr(layers[target_layer], 'hfer')) if target_layer < len(layers) else None
                base_results.append({'file': os.path.basename(fp), 'label': label, 'hfer_base': hfer})
            except Exception as e:
                print(f"  ERROR: {e}")
                base_results.append({'file': os.path.basename(fp), 'label': label, 'hfer_base': None})

        # Phase 2: Apply steering and re-evaluate
        print(f"\n--- PHASE 2: STEERED MODEL (alpha={args.alpha}, layer={steer_layer}) ---")
        model = framework.instrumenter.model
        apply_spectral_steering(model, steer_layer, alpha=args.alpha)

        for i, (fp, label) in enumerate(tqdm(test_files, desc='Steered')):
            with open(fp, 'r', encoding='utf-8') as f: text = f.read()
            try:
                analysis = framework.analyze_text(text, save_results=False)
                layers = analysis.get('layer_diagnostics', [])
                hfer = float(getattr(layers[target_layer], 'hfer')) if target_layer < len(layers) else None
                base_results[i]['hfer_steered'] = hfer
            except Exception as e:
                base_results[i]['hfer_steered'] = None

    # Analysis
    print(f"\n--- RESULTS ---")
    for label in ['valid', 'invalid']:
        subset = [r for r in base_results if r['label'] == label and r['hfer_base'] is not None and r['hfer_steered'] is not None]
        if subset:
            base_hfers = [r['hfer_base'] for r in subset]
            steered_hfers = [r['hfer_steered'] for r in subset]
            delta = np.array(steered_hfers) - np.array(base_hfers)
            print(f"\n{label.upper()} proofs (n={len(subset)}):")
            print(f"  Base HFER:    {np.mean(base_hfers):.4f} ± {np.std(base_hfers):.4f}")
            print(f"  Steered HFER: {np.mean(steered_hfers):.4f} ± {np.std(steered_hfers):.4f}")
            print(f"  Delta:        {np.mean(delta):+.4f} ± {np.std(delta):.4f}")

            from scipy.stats import ttest_rel
            if len(subset) > 2:
                t, p = ttest_rel(base_hfers, steered_hfers)
                print(f"  Paired t-test: t={t:.3f}, p={p:.4e}")

    # Save
    with open('data/results/rebuttal/steering_results.json', 'w') as f:
        json.dump(base_results, f, indent=2)
    print(f"\nSaved: data/results/rebuttal/steering_results.json")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='meta-llama/Llama-3.2-3B-Instruct')
    parser.add_argument('--data-dir', default='data/experiment_ready')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--load-in-4bit', action='store_true')
    parser.add_argument('--alpha', type=float, default=-0.3)
    run_steering(parser.parse_args())
