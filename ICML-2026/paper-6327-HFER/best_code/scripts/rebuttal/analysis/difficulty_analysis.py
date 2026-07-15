"""
Difficulty / Complexity Stratification Analysis
================================================
Addresses: Reviewer MEs9 ("how well does the approach deal with computationally
difficult problems?")

Parses raw proof files to compute complexity metrics per proof (tactic count,
have-step count, nesting depth, etc.). When experiment JSONs are available
(via --results-json), cross-references with spectral accuracy per stratum.

Usage:
    # Complexity metrics only (no experiment data needed):
    python scripts/rebuttal/difficulty_analysis.py

    # Full analysis with spectral results:
    python scripts/rebuttal/difficulty_analysis.py --results-json data/results/experiment_results_Meta-Llama-3.1-8B-Instruct.json --reclaimed data/reclaimed/8B_list_b_confident_invalid.json
"""

import os
import sys
import json
import re
import glob
import argparse
from collections import Counter, defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})

# Lean tactics to detect
TACTICS = [
    'norm_num', 'simp', 'ring', 'omega', 'linarith', 'nlinarith',
    'exact', 'apply', 'have', 'calc', 'intro', 'rw', 'cases',
    'induction', 'refine', 'use', 'constructor', 'ext', 'funext',
    'congr', 'trivial', 'tauto', 'decide', 'contradiction',
    'exfalso', 'push_neg', 'by_contra', 'rcases', 'obtain',
    'specialize', 'field_simp', 'ring_nf', 'norm_cast',
]


def parse_proof_complexity(proof_text):
    """Extract complexity metrics from a Lean proof."""
    if not proof_text:
        return None

    lines = proof_text.split('\n')

    # Find the proof body (between begin/end or by)
    in_proof = False
    proof_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped == 'begin':
            in_proof = True
            continue
        if stripped == 'end' and in_proof:
            in_proof = False
            continue
        if in_proof:
            proof_lines.append(line)

    proof_body = '\n'.join(proof_lines)
    proof_lower = proof_body.lower()

    # 1. Count tactic applications
    tactic_counts = {}
    total_tactics = 0
    for tactic in TACTICS:
        # Match tactic as word boundary
        count = len(re.findall(r'\b' + tactic + r'\b', proof_lower))
        if count > 0:
            tactic_counts[tactic] = count
            total_tactics += count

    # 2. Count 'have' steps (intermediate lemmas)
    have_count = len(re.findall(r'\bhave\b', proof_lower))

    # 3. Count unique identifiers referenced
    # Match things like `foo.bar` or `foo_bar` that look like lemma names
    identifiers = set(re.findall(r'[a-z][a-z0-9_]+\.[a-z][a-z0-9_]+', proof_lower))

    # 4. Nesting depth (count indentation levels or nested begin/end)
    max_indent = 0
    for line in proof_lines:
        if line.strip():
            indent = len(line) - len(line.lstrip())
            max_indent = max(max_indent, indent)
    nesting_depth = max_indent // 2  # Approximate nesting levels

    # 5. Total lines in proof body
    proof_line_count = len([l for l in proof_lines if l.strip()])

    # 6. Token count (rough approximation)
    token_count = len(proof_body.split())

    # 7. Primary tactic type
    if tactic_counts:
        primary_tactic = max(tactic_counts, key=tactic_counts.get)
    else:
        primary_tactic = 'unknown'

    # 8. Classify complexity
    if total_tactics <= 1:
        complexity_bracket = 'Trivial'
    elif total_tactics <= 3:
        complexity_bracket = 'Simple'
    elif total_tactics <= 6:
        complexity_bracket = 'Moderate'
    else:
        complexity_bracket = 'Complex'

    # 9. Classify have-step complexity
    if have_count == 0:
        have_bracket = '0 (one-shot)'
    elif have_count <= 2:
        have_bracket = '1-2 steps'
    else:
        have_bracket = '3+ steps'

    return {
        'total_tactics': total_tactics,
        'tactic_counts': tactic_counts,
        'have_count': have_count,
        'unique_identifiers': len(identifiers),
        'nesting_depth': nesting_depth,
        'proof_lines': proof_line_count,
        'token_count': token_count,
        'primary_tactic': primary_tactic,
        'complexity_bracket': complexity_bracket,
        'have_bracket': have_bracket,
    }


def parse_theorem_source(filename):
    """Extract problem source from theorem name."""
    base = filename.replace('.lean', '').split('_')
    # Remove trailing number (variant index)
    if base and base[-1].isdigit() and len(base) > 1:
        base = base[:-1]
    name = '_'.join(base)

    if name.startswith('aime') or name.startswith('aimeII'):
        return 'AIME'
    elif name.startswith('amc12') or name.startswith('amc10'):
        return 'AMC'
    elif name.startswith('imo'):
        return 'IMO'
    elif name.startswith('mathd'):
        return 'MathD'
    elif name.startswith('algebra'):
        return 'Algebra'
    elif name.startswith('numbertheory'):
        return 'NumberTheory'
    elif name.startswith('induction'):
        return 'Induction'
    else:
        return 'Other'


def run_difficulty_analysis(args):
    print("=" * 70)
    print("  DIFFICULTY / COMPLEXITY STRATIFICATION")
    print("=" * 70)

    os.makedirs('output/rebuttal', exist_ok=True)
    os.makedirs('data/results/rebuttal', exist_ok=True)

    # Load all proof files
    proof_dirs = {
        'valid': 'data/experiment_ready/valid',
        'invalid': 'data/experiment_ready/invalid',
    }

    all_proofs = []
    for label, proof_dir in proof_dirs.items():
        if not os.path.exists(proof_dir):
            print(f"WARNING: {proof_dir} not found, skipping.")
            continue
        for filepath in sorted(glob.glob(os.path.join(proof_dir, '*.lean'))):
            filename = os.path.basename(filepath)
            with open(filepath, 'r', encoding='utf-8') as f:
                text = f.read()
            complexity = parse_proof_complexity(text)
            source = parse_theorem_source(filename)
            if complexity:
                all_proofs.append({
                    'file': filename,
                    'label': label,
                    'source': source,
                    **complexity,
                })

    print(f"\nParsed {len(all_proofs)} proofs ({sum(1 for p in all_proofs if p['label']=='valid')} valid, "
          f"{sum(1 for p in all_proofs if p['label']=='invalid')} invalid)")

    # === COMPLEXITY METRICS ANALYSIS (no experiment data needed) ===

    # 1. Tactic count distribution
    print("\n--- COMPLEXITY DISTRIBUTION ---")
    bracket_counts = Counter(p['complexity_bracket'] for p in all_proofs)
    print(f"\n{'Bracket':<12} | {'Valid':>5} | {'Invalid':>7} | {'Total':>5}")
    print("-" * 40)
    for bracket in ['Trivial', 'Simple', 'Moderate', 'Complex']:
        n_valid = sum(1 for p in all_proofs if p['complexity_bracket'] == bracket and p['label'] == 'valid')
        n_invalid = sum(1 for p in all_proofs if p['complexity_bracket'] == bracket and p['label'] == 'invalid')
        print(f"{bracket:<12} | {n_valid:>5} | {n_invalid:>7} | {n_valid + n_invalid:>5}")

    # 2. Have-step distribution
    print("\n--- HAVE-STEP DISTRIBUTION ---")
    have_counts = Counter(p['have_bracket'] for p in all_proofs)
    print(f"\n{'Have Steps':<15} | {'Valid':>5} | {'Invalid':>7} | {'Total':>5}")
    print("-" * 45)
    for bracket in ['0 (one-shot)', '1-2 steps', '3+ steps']:
        n_valid = sum(1 for p in all_proofs if p['have_bracket'] == bracket and p['label'] == 'valid')
        n_invalid = sum(1 for p in all_proofs if p['have_bracket'] == bracket and p['label'] == 'invalid')
        print(f"{bracket:<15} | {n_valid:>5} | {n_invalid:>7} | {n_valid + n_invalid:>5}")

    # 3. Source distribution
    print("\n--- SOURCE DISTRIBUTION ---")
    source_counts = Counter(p['source'] for p in all_proofs)
    print(f"\n{'Source':<15} | {'Valid':>5} | {'Invalid':>7} | {'Total':>5}")
    print("-" * 45)
    for source in sorted(source_counts.keys()):
        n_valid = sum(1 for p in all_proofs if p['source'] == source and p['label'] == 'valid')
        n_invalid = sum(1 for p in all_proofs if p['source'] == source and p['label'] == 'invalid')
        print(f"{source:<15} | {n_valid:>5} | {n_invalid:>7} | {n_valid + n_invalid:>5}")

    # 4. Primary tactic distribution
    print("\n--- PRIMARY TACTIC DISTRIBUTION ---")
    tactic_dist = Counter(p['primary_tactic'] for p in all_proofs)
    for tactic, count in tactic_dist.most_common(10):
        n_valid = sum(1 for p in all_proofs if p['primary_tactic'] == tactic and p['label'] == 'valid')
        n_invalid = sum(1 for p in all_proofs if p['primary_tactic'] == tactic and p['label'] == 'invalid')
        print(f"  {tactic:<15}: {count:>4} total ({n_valid} valid, {n_invalid} invalid)")

    # 5. Summary statistics
    print("\n--- SUMMARY STATISTICS ---")
    for label in ['valid', 'invalid']:
        proofs = [p for p in all_proofs if p['label'] == label]
        tactics = [p['total_tactics'] for p in proofs]
        haves = [p['have_count'] for p in proofs]
        lines = [p['proof_lines'] for p in proofs]
        tokens = [p['token_count'] for p in proofs]
        print(f"\n{label.upper()} proofs (n={len(proofs)}):")
        print(f"  Tactics: mean={np.mean(tactics):.1f}, median={np.median(tactics):.0f}, max={max(tactics)}")
        print(f"  Have steps: mean={np.mean(haves):.1f}, median={np.median(haves):.0f}, max={max(haves)}")
        print(f"  Lines: mean={np.mean(lines):.1f}, median={np.median(lines):.0f}, max={max(lines)}")
        print(f"  Tokens: mean={np.mean(tokens):.1f}, median={np.median(tokens):.0f}, max={max(tokens)}")

    # === SPECTRAL ACCURACY PER STRATUM (if experiment data available) ===
    if args.results_json and os.path.exists(args.results_json):
        print(f"\n--- SPECTRAL STRATIFICATION (from {args.results_json}) ---")
        with open(args.results_json, 'r') as f:
            results_data = json.load(f)

        # Load reclaimed list for label correction
        reclaimed_files = set()
        if args.reclaimed and os.path.exists(args.reclaimed):
            with open(args.reclaimed, 'r') as f:
                reclaimed_list = json.load(f)
                reclaimed_files = set(item['file'] for item in reclaimed_list)

        # Build file → spectral data map
        spectral_map = {}
        for label_type in ['valid', 'invalid']:
            for item in results_data.get(label_type, []):
                corrected_label = label_type
                if label_type == 'invalid' and item['file'] in reclaimed_files:
                    corrected_label = 'valid'
                spectral_map[item['file']] = {
                    'label': corrected_label,
                    'trajectory': item.get('trajectory', [])
                }

        # Detect num_layers and pick 75th percentile layer
        sample_traj = list(spectral_map.values())[0]['trajectory'] if spectral_map else []
        num_layers = len(sample_traj)
        target_layer = int(0.75 * num_layers) if num_layers > 0 else 0
        print(f"  Model depth: {num_layers} layers, using L{target_layer} (75th percentile)")

        # Cross-reference complexity with spectral
        for bracket in ['Trivial', 'Simple', 'Moderate', 'Complex']:
            bracket_proofs = [p for p in all_proofs if p['complexity_bracket'] == bracket]
            valid_hfers = []
            invalid_hfers = []
            for p in bracket_proofs:
                spec = spectral_map.get(p['file'])
                if spec and target_layer < len(spec['trajectory']):
                    hfer = spec['trajectory'][target_layer].get('hfer')
                    if hfer is not None:
                        if spec['label'] == 'valid':
                            valid_hfers.append(hfer)
                        else:
                            invalid_hfers.append(hfer)

            if len(valid_hfers) >= 2 and len(invalid_hfers) >= 2:
                from scipy import stats as scipy_stats
                d = (np.mean(valid_hfers) - np.mean(invalid_hfers)) / np.sqrt(
                    ((len(valid_hfers)-1)*np.std(valid_hfers, ddof=1)**2 +
                     (len(invalid_hfers)-1)*np.std(invalid_hfers, ddof=1)**2) /
                    (len(valid_hfers) + len(invalid_hfers) - 2)
                )
                print(f"  {bracket}: n_valid={len(valid_hfers)}, n_invalid={len(invalid_hfers)}, "
                      f"Cohen's d={d:.2f}")
            else:
                print(f"  {bracket}: insufficient data for effect size")
    else:
        print("\n[INFO] No --results-json provided. Skipping spectral stratification.")
        print("       Run with: --results-json data/results/experiment_results_<model>.json")

    # === GENERATE FIGURES ===
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Left: Tactic count distribution by validity
    valid_tactics = [p['total_tactics'] for p in all_proofs if p['label'] == 'valid']
    invalid_tactics = [p['total_tactics'] for p in all_proofs if p['label'] == 'invalid']
    bins = range(0, max(max(valid_tactics, default=0), max(invalid_tactics, default=0)) + 2)
    axes[0].hist(valid_tactics, bins=bins, alpha=0.6, color='#1565C0', label='Valid', density=True)
    axes[0].hist(invalid_tactics, bins=bins, alpha=0.6, color='#C62828', label='Invalid', density=True)
    axes[0].set_xlabel('Total Tactic Applications')
    axes[0].set_ylabel('Density')
    axes[0].set_title('Proof Complexity Distribution')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Middle: Have-step distribution
    valid_haves = [p['have_count'] for p in all_proofs if p['label'] == 'valid']
    invalid_haves = [p['have_count'] for p in all_proofs if p['label'] == 'invalid']
    bins_h = range(0, max(max(valid_haves, default=0), max(invalid_haves, default=0)) + 2)
    axes[1].hist(valid_haves, bins=bins_h, alpha=0.6, color='#1565C0', label='Valid', density=True)
    axes[1].hist(invalid_haves, bins=bins_h, alpha=0.6, color='#C62828', label='Invalid', density=True)
    axes[1].set_xlabel('Number of "have" Steps')
    axes[1].set_ylabel('Density')
    axes[1].set_title('Reasoning Chain Length')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Right: Source distribution stacked bar
    sources = sorted(set(p['source'] for p in all_proofs))
    valid_by_source = [sum(1 for p in all_proofs if p['source'] == s and p['label'] == 'valid') for s in sources]
    invalid_by_source = [sum(1 for p in all_proofs if p['source'] == s and p['label'] == 'invalid') for s in sources]
    x = np.arange(len(sources))
    axes[2].bar(x - 0.2, valid_by_source, 0.4, color='#1565C0', label='Valid')
    axes[2].bar(x + 0.2, invalid_by_source, 0.4, color='#C62828', label='Invalid')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(sources, rotation=45, ha='right')
    axes[2].set_ylabel('Count')
    axes[2].set_title('Problem Source Distribution')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    fig_path = 'output/rebuttal/difficulty_stratification.pdf'
    plt.savefig(fig_path)
    plt.savefig(fig_path.replace('.pdf', '.png'))
    print(f"\nSaved: {fig_path}")

    # === SAVE DATA ===
    save_data = {
        'total_proofs': len(all_proofs),
        'valid_count': sum(1 for p in all_proofs if p['label'] == 'valid'),
        'invalid_count': sum(1 for p in all_proofs if p['label'] == 'invalid'),
        'complexity_distribution': {
            bracket: {
                'valid': sum(1 for p in all_proofs if p['complexity_bracket'] == bracket and p['label'] == 'valid'),
                'invalid': sum(1 for p in all_proofs if p['complexity_bracket'] == bracket and p['label'] == 'invalid'),
            }
            for bracket in ['Trivial', 'Simple', 'Moderate', 'Complex']
        },
        'have_step_distribution': {
            bracket: {
                'valid': sum(1 for p in all_proofs if p['have_bracket'] == bracket and p['label'] == 'valid'),
                'invalid': sum(1 for p in all_proofs if p['have_bracket'] == bracket and p['label'] == 'invalid'),
            }
            for bracket in ['0 (one-shot)', '1-2 steps', '3+ steps']
        },
        'source_distribution': {
            source: {
                'valid': sum(1 for p in all_proofs if p['source'] == source and p['label'] == 'valid'),
                'invalid': sum(1 for p in all_proofs if p['source'] == source and p['label'] == 'invalid'),
            }
            for source in sorted(set(p['source'] for p in all_proofs))
        },
        'per_proof_complexity': [
            {k: v for k, v in p.items() if k != 'tactic_counts'}
            for p in all_proofs
        ],
    }

    json_path = 'data/results/rebuttal/difficulty_analysis.json'
    with open(json_path, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"Saved: {json_path}")

    # === SAVE LATEX TABLE ===
    latex_lines = [
        r'\begin{table}[t]',
        r'\caption{Proof complexity stratification by tactic count and reasoning chain length.}',
        r'\label{tab:difficulty}',
        r'\centering',
        r'\begin{tabular}{lrrrr}',
        r'\toprule',
        r'Complexity & Tactic Steps & Valid & Invalid & Total \\',
        r'\midrule',
    ]
    for bracket in ['Trivial', 'Simple', 'Moderate', 'Complex']:
        ranges = {'Trivial': '1', 'Simple': '2--3', 'Moderate': '4--6', 'Complex': '7+'}
        n_v = save_data['complexity_distribution'][bracket]['valid']
        n_i = save_data['complexity_distribution'][bracket]['invalid']
        latex_lines.append(f"  {bracket} & {ranges[bracket]} & {n_v} & {n_i} & {n_v+n_i} \\\\")
    latex_lines.extend([
        r'\bottomrule',
        r'\end{tabular}',
        r'\end{table}',
    ])

    with open('output/rebuttal/difficulty_stratification.tex', 'w') as f:
        f.write('\n'.join(latex_lines))
    print(f"Saved: output/rebuttal/difficulty_stratification.tex")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Difficulty/Complexity Stratification Analysis')
    parser.add_argument('--results-json', type=str, default=None,
                        help='Path to experiment results JSON for spectral stratification')
    parser.add_argument('--reclaimed', type=str, default=None,
                        help='Path to reclaimed proofs JSON for label correction')
    args = parser.parse_args()
    run_difficulty_analysis(args)
