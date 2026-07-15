"""
Prefix / Partial Proof Evolution
==================================
Addresses: Reviewer mxpA Q5 ("Do signatures emerge on intermediate prefixes?")

Runs model on progressively longer prefixes (25%, 50%, 75%, 100%) and tracks
HFER to see if invalid proofs can be detected early. Requires GPU.

Usage:
    python scripts/rebuttal/prefix_evolution.py --model meta-llama/Meta-Llama-3.1-8B-Instruct --load-in-4bit
"""

import os, sys, json, glob, argparse
import numpy as np
import torch
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams.update({
    'font.family': 'serif', 'font.size': 10, 'figure.dpi': 300,
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.05,
})

def run_prefix(args):
    print("=" * 70)
    print("  PREFIX / PARTIAL PROOF EVOLUTION")
    print("=" * 70)

    os.makedirs('output/rebuttal', exist_ok=True)
    os.makedirs('data/results/rebuttal', exist_ok=True)

    from spectral_trust import GSPDiagnosticsFramework, GSPConfig

    model_kwargs = {"output_attentions": True, "output_hidden_states": True}
    if args.load_in_4bit: model_kwargs["load_in_4bit"] = True

    config = GSPConfig(model_name=args.model, device=args.device, model_kwargs=model_kwargs)

    # Select 10 valid + 10 invalid proofs
    valid_files = sorted(glob.glob(os.path.join(args.data_dir, 'valid', '*.lean')))[:10]
    invalid_files = sorted(glob.glob(os.path.join(args.data_dir, 'invalid', '*.lean')))[:10]
    test_files = [(fp, 'valid') for fp in valid_files] + [(fp, 'invalid') for fp in invalid_files]

    fractions = [0.25, 0.50, 0.75, 1.00]
    results = []

    with GSPDiagnosticsFramework(config) as framework:
        framework.instrumenter.load_model(args.model)
        num_layers = framework.instrumenter.model.config.num_hidden_layers
        target_layer = int(0.75 * num_layers)

        for fp, label in tqdm(test_files, desc='Proofs'):
            with open(fp, 'r', encoding='utf-8') as f: text = f.read()
            tokens = text.split()
            n_tokens = len(tokens)

            proof_result = {'file': os.path.basename(fp), 'label': label, 'n_tokens': n_tokens, 'hfers': {}}

            for frac in fractions:
                n_prefix = max(10, int(n_tokens * frac))
                prefix_text = ' '.join(tokens[:n_prefix])

                try:
                    analysis = framework.analyze_text(prefix_text, save_results=False)
                    layers = analysis.get('layer_diagnostics', [])
                    hfer = float(getattr(layers[target_layer], 'hfer')) if target_layer < len(layers) else None
                    proof_result['hfers'][str(frac)] = hfer
                except Exception as e:
                    proof_result['hfers'][str(frac)] = None

            results.append(proof_result)

    # Analysis
    print(f"\n--- EVOLUTION SUMMARY ---")
    print(f"{'Fraction':>8} | {'Valid HFER (mean±std)':>22} | {'Invalid HFER (mean±std)':>24} | {'d':>6} | {'Acc':>6}")
    print("-" * 80)

    summary = []
    for frac in fractions:
        valid_h = [r['hfers'].get(str(frac)) for r in results if r['label'] == 'valid' and r['hfers'].get(str(frac)) is not None]
        invalid_h = [r['hfers'].get(str(frac)) for r in results if r['label'] == 'invalid' and r['hfers'].get(str(frac)) is not None]

        if len(valid_h) >= 2 and len(invalid_h) >= 2:
            d = (np.mean(valid_h) - np.mean(invalid_h)) / np.sqrt(
                ((len(valid_h)-1)*np.std(valid_h, ddof=1)**2 + (len(invalid_h)-1)*np.std(invalid_h, ddof=1)**2) /
                (len(valid_h) + len(invalid_h) - 2))
            all_v = valid_h + invalid_h
            all_l = [1]*len(valid_h) + [0]*len(invalid_h)
            # Simple threshold accuracy
            thresholds = np.percentile(all_v, np.linspace(0, 100, 50))
            acc = max(max(np.mean((np.array(all_v) < t) == np.array(all_l)),
                         np.mean((np.array(all_v) > t) == np.array(all_l))) for t in thresholds)
        else:
            d, acc = 0, 0.5

        print(f"  {frac*100:>5.0f}% | {np.mean(valid_h):.4f} ± {np.std(valid_h):.4f}       | "
              f"{np.mean(invalid_h):.4f} ± {np.std(invalid_h):.4f}         | {d:>5.2f} | {acc*100:>5.1f}%")
        summary.append({'fraction': frac, 'd': d, 'accuracy': acc,
                        'valid_mean': float(np.mean(valid_h)), 'invalid_mean': float(np.mean(invalid_h))})

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    fracs_pct = [s['fraction'] * 100 for s in summary]
    ax.plot(fracs_pct, [s['valid_mean'] for s in summary], 'o-', color='#1565C0', label='Valid', linewidth=2)
    ax.plot(fracs_pct, [s['invalid_mean'] for s in summary], 's-', color='#C62828', label='Invalid', linewidth=2)
    ax.set_xlabel('Prefix Length (%)')
    ax.set_ylabel('Mean HFER')
    ax.set_title('HFER Evolution During Proof Generation')
    ax.legend()
    ax.grid(True, alpha=0.3)
    # Annotate d values
    for s in summary:
        ax.annotate(f'd={s["d"]:.1f}', xy=(s['fraction']*100, (s['valid_mean']+s['invalid_mean'])/2),
                   fontsize=8, ha='center', va='bottom', color='gray')

    plt.savefig('output/rebuttal/prefix_evolution.pdf')
    plt.savefig('output/rebuttal/prefix_evolution.png')
    print(f"\nSaved: output/rebuttal/prefix_evolution.pdf")

    with open('data/results/rebuttal/prefix_evolution.json', 'w') as f:
        json.dump({'summary': summary, 'per_proof': results}, f, indent=2, default=str)
    print(f"Saved: data/results/rebuttal/prefix_evolution.json")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='meta-llama/Meta-Llama-3.1-8B-Instruct')
    parser.add_argument('--data-dir', default='data/experiment_ready')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--load-in-4bit', action='store_true')
    run_prefix(parser.parse_args())
