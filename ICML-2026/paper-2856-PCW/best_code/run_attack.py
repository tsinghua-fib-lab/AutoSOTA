#!/usr/bin/env python3
"""
Watermark Copy-Paste Attack Pipeline
=====================================

Apply copy-paste attacks, run detection (z-score + WinMax + WinMax-C), aggregate to CSV.

Usage:
    python run_attack.py --cp-ratios 0.1,0.15,0.2 --token-lengths 65
    python run_attack.py --pareto-csv results_attack/pareto_configs.csv --token-lengths 65
    python run_attack.py --seed 42 --models gpt2 --datasets c4
    python run_attack.py aggregate --input-dir results_attack_detail

Output:
    results_attack_detail/<experiment>.json
    results_attack/aggregated_attack_results.csv
"""

import argparse
import csv
import gc as gc_mod
import gzip
import json
import re
import time
import os
import sys
import random
import numpy as np
import torch
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parent / 'watermark_reliability_release'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils.copy_paste_attack import k_insertion_t_len
from run_experiment_combined import WatermarkDetectorNgram
from theorytools import compute_dp_params, load_klfront_params


# ============================================================================
# DETECTOR - WatermarkDetectorNgram + WinMax fix + WinMax-C
# ============================================================================

class WatermarkDetectorNgramFixed(WatermarkDetectorNgram):
    """WatermarkDetectorNgram with WinMax fix + WinMax-C detector."""

    def _score_windows_impl_batched(self, input_ids, window_size, window_stride=1):
        _orig = WatermarkDetectorNgram._get_green_at_T_booleans
        def _compat(self_inner, ids, lookup):
            return _orig(self_inner, ids, lookup)[:3]
        WatermarkDetectorNgram._get_green_at_T_booleans = _compat
        try:
            return super(WatermarkDetectorNgram, self)._score_windows_impl_batched(
                input_ids, window_size, window_stride)
        finally:
            WatermarkDetectorNgram._get_green_at_T_booleans = _orig

    def score_complement(self, input_ids):
        """WinMax-C: remove each possible segment, compute z on complement, return max."""
        from math import sqrt as msqrt
        if len(input_ids) < 3:
            return 0.0, 0
        ngram_to_watermark_lookup, _ = self._score_ngrams_in_passage(input_ids)
        result = WatermarkDetectorNgram._get_green_at_T_booleans(
            self, input_ids, ngram_to_watermark_lookup)
        green_ids = result[1].float()
        n = len(green_ids)
        if n < 3:
            gc = green_ids.sum().item()
            denom = msqrt(n * self.gamma * (1 - self.gamma)) if n > 0 else 1
            return (gc - self.gamma * n) / denom if denom > 0 else 0.0, 0
        prefix_sum = torch.cumsum(green_ids, dim=0)
        total_green = prefix_sum[-1].item()
        best_z = -float('inf')
        best_removed = 0
        for w in range(1, n):
            comp_n = n - w
            if comp_n < 1:
                continue
            n_windows = n - w + 1
            window_greens = torch.empty(n_windows)
            window_greens[0] = prefix_sum[w - 1]
            if n_windows > 1:
                window_greens[1:] = prefix_sum[w:] - prefix_sum[:n_windows - 1]
            comp_greens = total_green - window_greens
            denom = msqrt(comp_n * self.gamma * (1 - self.gamma))
            if denom == 0:
                continue
            z_scores = (comp_greens - self.gamma * comp_n) / denom
            max_z = z_scores.max().item()
            if max_z > best_z:
                best_z = max_z
                best_removed = w
        return (best_z if best_z > -float('inf') else 0.0), best_removed


def create_detector(tokenizer, config, device):
    """Create detector from experiment config."""
    return WatermarkDetectorNgramFixed(
        vocab=list(tokenizer.get_vocab().values()),
        gamma=config['gamma'],
        delta=config['delta'],
        seeding_scheme=config.get('seeding_scheme', 'simple_1'),
        device=device,
        tokenizer=tokenizer,
        z_threshold=config.get('z_threshold', 4.0),
        normalizers=[],
        ignore_repeated_ngrams=True,
        select_green_tokens=True,
    )


def detect_tokens(detector, token_ids, window_size=None):
    """Run detection: z-score + WinMax + WinMax-C."""
    if len(token_ids) < 2:
        return {'green_token_mask': [], 'z_score': 0.0,
                'green_fraction': 0.0, 'num_tokens_scored': 0,
                'winmax_z': 0.0, 'winmax_window_size': 0,
                'winmaxc_z': 0.0, 'winmaxc_removed_size': 0}
    tok_tensor = torch.tensor(token_ids, device=detector.device)

    # Standard detection
    result = detector.detect(
        tokenized_text=tok_tensor,
        return_prediction=True,
        return_scores=True,
        return_green_token_mask=True,
    )
    out = {
        'green_token_mask': result.get('green_token_mask', []),
        'z_score': float(result.get('z_score', 0.0)),
        'green_fraction': float(result.get('green_fraction', 0.0)),
        'num_tokens_scored': int(result.get('num_tokens_scored', 0)),
        'prediction': bool(result.get('prediction', False)),
        'p_value': float(result.get('p_value', 1.0)),
    }

    # WinMax + WinMax-C
    if window_size is not None:
        try:
            wm_result = detector.detect(
                tokenized_text=tok_tensor,
                window_size=window_size,
                window_stride=1,
                return_prediction=True,
                return_scores=True,
                return_green_token_mask=False,
            )
            out['winmax_z'] = float(wm_result.get('z_score', 0.0))
            out['winmax_window_size'] = int(wm_result.get('num_tokens_scored', 0))
        except (ValueError, RuntimeError):
            if len(token_ids) < 3:
                out['winmax_z'] = out['z_score']
                out['winmax_window_size'] = out['num_tokens_scored']
            else:
                raise

        try:
            wmc_z, wmc_removed = detector.score_complement(tok_tensor)
            out['winmaxc_z'] = float(wmc_z)
            out['winmaxc_removed_size'] = int(wmc_removed)
        except Exception:
            out['winmaxc_z'] = out['z_score']
            out['winmaxc_removed_size'] = 0
    return out


# ============================================================================
# EXPERIMENT DEFINITIONS
# ============================================================================

MODELS = ['opt-125m', 'pythia-160m', 'gpt2']
DATASETS = ['c4', 'lfqa', 'wikipedia']
GAMMA_LIST = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
DELTA_LIST = [0.5, 1, 2, 5, 10]
SEED_MAX = 5
TEMPERATURE = 0.7

MODEL_NAMES = {
    'opt-125m': 'facebook/opt-125m',
    'gpt2': 'gpt2',
    'pythia-160m': 'EleutherAI/pythia-160m',
}


def generate_experiment_list():
    dp_params = compute_dp_params()
    kl_params = load_klfront_params()
    t = TEMPERATURE
    experiments = []
    for model in MODELS:
        for dataset in DATASETS:
            for seed in range(SEED_MAX):
                for gamma in GAMMA_LIST:
                    for delta in DELTA_LIST:
                        experiments.append({
                            'experiment_name': f'GRID_m_{model}_d_{dataset}_g{gamma:.2f}_d{delta:.5f}_t{t:.2f}_s{seed}',
                            'method': 'GRID', 'model': model, 'dataset': dataset,
                            'gamma': gamma, 'delta': delta, 'seed': seed,
                        })
                for gamma, delta in kl_params:
                    experiments.append({
                        'experiment_name': f'KLFRONT_m_{model}_d_{dataset}_g{gamma:.2f}_d{delta:.5f}_t{t:.2f}_s{seed}',
                        'method': 'KLFRONT', 'model': model, 'dataset': dataset,
                        'gamma': gamma, 'delta': delta, 'seed': seed,
                    })
                for gamma, delta in dp_params:
                    experiments.append({
                        'experiment_name': f'DP_m_{model}_d_{dataset}_g{gamma:.2f}_d{delta:.5f}_t{t:.2f}_s{seed}',
                        'method': 'DP', 'model': model, 'dataset': dataset,
                        'gamma': gamma, 'delta': delta, 'seed': seed,
                    })
                for gamma in GAMMA_LIST:
                    for delta in DELTA_LIST:
                        experiments.append({
                            'experiment_name': f'OPT_m_{model}_d_{dataset}_g{gamma:.2f}_d{delta:.5f}_t{t:.2f}_s{seed}',
                            'method': 'OPT', 'model': model, 'dataset': dataset,
                            'gamma': gamma, 'delta': delta, 'seed': seed,
                        })
    return experiments


def find_file(experiment_name, results_dirs):
    for d in results_dirs:
        p = Path(d) / f'{experiment_name}.json'
        if p.exists():
            return str(p.resolve())
    return None


def build_baseline_map(results_dirs):
    baseline_map = {}
    pattern = re.compile(
        r'^[A-Z]+_m_(?P<model>[^_]+(?:-[^_]+)?)_d_(?P<dataset>[^_]+)'
        r'_g(?P<gamma>[0-9.]+)_d0\.00000_t[0-9.]+_s(?P<seed>\d+)\.json$'
    )
    for d in results_dirs:
        dp = Path(d)
        if not dp.exists():
            continue
        for f in dp.glob('*_d0.00000_*.json'):
            m = pattern.match(f.name)
            if not m:
                continue
            key = (m.group('model'), m.group('dataset'), m.group('gamma'), int(m.group('seed')))
            if key not in baseline_map:
                baseline_map[key] = str(f.resolve())
    return baseline_map


_baseline_cache = {}

def _load_baseline_tokens(path):
    if path in _baseline_cache:
        return _baseline_cache[path]
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    tokens = [s['watermarked']['generated_tokens'] for s in data.get('results', [])]
    _baseline_cache[path] = tokens
    return tokens


# ============================================================================
# PROCESS ONE EXPERIMENT
# ============================================================================

def process_experiment(wm_info, tokenizer, detector, max_token_lengths,
                       save_tokens, output_dir, baseline_path=None, cp_ratios=None,
                       global_seed=42):
    """Apply copy-paste attacks, detect, save."""
    # Seed per experiment: global_seed + experiment seed
    exp_seed = global_seed + wm_info.get('seed', 0)
    random.seed(exp_seed)
    np.random.seed(exp_seed)
    torch.manual_seed(exp_seed)

    with open(wm_info['path'], 'r', encoding='utf-8') as f:
        wm_data = json.load(f)

    base_tokens_list = None
    if baseline_path and cp_ratios:
        base_tokens_list = _load_baseline_tokens(baseline_path)

    samples = []
    for si, sample in enumerate(wm_data.get('results', [])):
        wm_tok_full = sample['watermarked']['generated_tokens']
        # Per-token KL from generation (may be None for some experiments)
        kl_full = sample['watermarked'].get('kl_divergence')

        for ml in max_token_lengths:
            if len(wm_tok_full) < ml:
                continue
            wm_tok = wm_tok_full[:ml]
            kl_arr = kl_full[:ml] if kl_full and len(kl_full) >= ml else None

            # Original detection
            od = detect_tokens(detector, wm_tok, window_size="max")

            # Original KL: mean of per-token KL for the truncated sequence
            orig_kl_mean = float(np.mean(kl_arr)) if kl_arr else None

            entry = {
                'sample_idx': si,
                'n_tokens': ml,
                'original': {
                    'z_score': od['z_score'],
                    'green_fraction': od['green_fraction'],
                    'num_tokens_scored': od['num_tokens_scored'],
                    'green_token_mask': od['green_token_mask'],
                    'winmax_z': od.get('winmax_z', 0.0),
                    'winmax_window_size': od.get('winmax_window_size', 0),
                    'winmaxc_z': od.get('winmaxc_z', 0.0),
                    'winmaxc_removed_size': od.get('winmaxc_removed_size', 0),
                    'kl_mean': orig_kl_mean,
                },
                'attacks': {},
            }
            if save_tokens:
                entry['original']['tokens'] = wm_tok

            # Copy-paste attacks
            if base_tokens_list and cp_ratios and si < len(base_tokens_list):
                base_tok_full = base_tokens_list[si]
                base_tok = base_tok_full[:ml] if len(base_tok_full) >= ml else None
                if base_tok:
                    for ratio in cp_ratios:
                        try:
                            t = int(ml * ratio)
                            if t < 1:
                                continue
                            min_count = min(len(wm_tok), len(base_tok))
                            if t >= min_count:
                                continue
                            attacked = k_insertion_t_len(1, t, min_count, wm_tok, base_tok)
                            attacked_list = attacked.tolist()

                            # Find replaced positions by comparing tokens
                            wm_tensor = torch.tensor(wm_tok)
                            atk_tensor_cmp = torch.tensor(attacked_list[:len(wm_tok)])
                            replaced_mask = (wm_tensor != atk_tensor_cmp).numpy()

                            # KL of watermarked (non-replaced) portion only
                            if kl_arr is not None:
                                kl_np = np.array(kl_arr[:len(replaced_mask)])
                                wm_kl = kl_np[~replaced_mask]
                                kl_wm_mean = float(wm_kl.mean()) if len(wm_kl) > 0 else 0.0
                                kl_full_mean = float(kl_np[~replaced_mask].sum() / len(kl_np))
                                n_replaced = int(replaced_mask.sum())
                            else:
                                kl_wm_mean = None
                                kl_full_mean = None
                                n_replaced = int(replaced_mask.sum())

                            ad = detect_tokens(detector, attacked_list, window_size="max")
                            a = {
                                'z_score': ad['z_score'],
                                'green_fraction': ad['green_fraction'],
                                'num_tokens_scored': ad['num_tokens_scored'],
                                'green_token_mask': ad['green_token_mask'],
                                'winmax_z': ad.get('winmax_z', 0.0),
                                'winmax_window_size': ad.get('winmax_window_size', 0),
                                'winmaxc_z': ad.get('winmaxc_z', 0.0),
                                'winmaxc_removed_size': ad.get('winmaxc_removed_size', 0),
                                'n_tokens_after': len(attacked_list),
                                'num_insertions': 1,
                                'insertion_len': t,
                                'n_replaced': n_replaced,
                                'kl_wm_only': kl_wm_mean,
                                'kl_full_avg': kl_full_mean,
                            }
                            if save_tokens:
                                a['tokens'] = attacked_list
                            entry['attacks'][f'copy_paste_{ratio}'] = a
                        except Exception:
                            pass

            samples.append(entry)

    result = {
        'metadata': {
            'experiment_name': wm_info['experiment_name'],
            'method': wm_info['method'],
            'model': wm_info['model'],
            'dataset': wm_info['dataset'],
            'gamma': wm_info['gamma'],
            'delta': wm_info['delta'],
            'seed': wm_info['seed'],
            'global_seed': global_seed,
        },
        'samples': samples,
    }
    out = output_dir / f"{wm_info['experiment_name']}.json"
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(result, f)
    return len(samples)


# ============================================================================
# AGGREGATION
# ============================================================================

CSV_COLUMNS = [
    'experiment', 'method', 'model', 'dataset', 'gamma', 'delta', 'seed',
    'n_tokens', 'sample_idx',
    'original_z_score', 'original_green_fraction', 'original_num_tokens_scored',
    'original_kl_mean',
    'winmax_z', 'winmax_window_size',
    'winmaxc_z', 'winmaxc_removed_size',
    'attack_type',
    'attacked_z_score', 'attacked_green_fraction', 'attacked_num_tokens_scored',
    'attacked_n_tokens',
    'attacked_winmax_z', 'attacked_winmax_window_size',
    'attacked_winmaxc_z', 'attacked_winmaxc_removed_size',
    'param_num_insertions', 'param_insertion_len',
    'n_replaced', 'kl_wm_only', 'kl_full_avg',
]


def aggregate(output_dir, summary_csv):
    """Aggregate detail JSONs to CSV (streaming)."""
    from tqdm.auto import tqdm
    import pandas as pd

    detail_files = sorted(list(output_dir.glob('*.json')) + list(output_dir.glob('*.json.gz')))
    print(f"Aggregating {len(detail_files)} files...")

    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    total_rows = 0
    batch = []

    with open(summary_csv, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for fp in tqdm(detail_files, desc="Aggregating"):
            try:
                if fp.suffix == '.gz':
                    with gzip.open(str(fp), 'rt', encoding='utf-8') as f:
                        data = json.load(f)
                else:
                    with open(fp, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                meta = data['metadata']
                for sample in data['samples']:
                    orig = sample['original']
                    base = {
                        'experiment': meta['experiment_name'],
                        'method': meta['method'],
                        'model': meta['model'],
                        'dataset': meta['dataset'],
                        'gamma': meta['gamma'],
                        'delta': meta['delta'],
                        'seed': meta['seed'],
                        'n_tokens': sample['n_tokens'],
                        'sample_idx': sample['sample_idx'],
                        'original_z_score': orig['z_score'],
                        'original_green_fraction': orig['green_fraction'],
                        'original_num_tokens_scored': orig['num_tokens_scored'],
                        'original_kl_mean': orig.get('kl_mean'),
                        'winmax_z': orig.get('winmax_z'),
                        'winmax_window_size': orig.get('winmax_window_size'),
                        'winmaxc_z': orig.get('winmaxc_z'),
                        'winmaxc_removed_size': orig.get('winmaxc_removed_size'),
                    }
                    for atk_name, atk_data in sample['attacks'].items():
                        row = {
                            **base,
                            'attack_type': atk_name,
                            'attacked_z_score': atk_data['z_score'],
                            'attacked_green_fraction': atk_data['green_fraction'],
                            'attacked_num_tokens_scored': atk_data['num_tokens_scored'],
                            'attacked_n_tokens': atk_data.get('n_tokens_after'),
                            'attacked_winmax_z': atk_data.get('winmax_z'),
                            'attacked_winmax_window_size': atk_data.get('winmax_window_size'),
                            'attacked_winmaxc_z': atk_data.get('winmaxc_z'),
                            'attacked_winmaxc_removed_size': atk_data.get('winmaxc_removed_size'),
                            'param_num_insertions': atk_data.get('num_insertions'),
                            'param_insertion_len': atk_data.get('insertion_len'),
                            'n_replaced': atk_data.get('n_replaced'),
                            'kl_wm_only': atk_data.get('kl_wm_only'),
                            'kl_full_avg': atk_data.get('kl_full_avg'),
                        }
                        batch.append(row)
                        if len(batch) >= 500:
                            writer.writerows(batch)
                            total_rows += len(batch)
                            batch.clear()
                del data
            except Exception as e:
                print(f"  Error: {fp.name}: {e}")
        if batch:
            writer.writerows(batch)
            total_rows += len(batch)
            batch.clear()

    gc_mod.collect()
    df = pd.read_csv(summary_csv, usecols=[
        'method', 'model', 'dataset', 'attack_type', 'attacked_z_score'])
    print(f"\nSaved: {summary_csv}  ({len(df):,} rows)")
    print(f"Methods: {sorted(df['method'].unique())}")
    print(f"Attacks: {sorted(df['attack_type'].unique())}")
    return df


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Watermark copy-paste attack pipeline")
    sub = parser.add_subparsers(dest='command', required=True)

    # run (attack + detect + aggregate)
    p_run = sub.add_parser('run', help='Run copy-paste attacks + detection + aggregate')
    p_run.add_argument('--token-lengths', type=str, default='65')
    p_run.add_argument('--cp-ratios', type=str, default='0.1,0.15,0.2')
    p_run.add_argument('--seed', type=str, default='42',
                        help='Global random seed(s), comma-separated for multiple runs (e.g. 42,43,44)')
    p_run.add_argument('--models', type=str, default=None)
    p_run.add_argument('--datasets', type=str, default=None)
    p_run.add_argument('--hf-cache-dir', type=str, default='./hf_cache')
    p_run.add_argument('--output-dir', type=str, default='results_attack_detail')
    p_run.add_argument('--summary-csv', type=str, default='results_attack/aggregated_attack_results.csv')
    p_run.add_argument('--results-dirs', type=str, default='results_final')
    p_run.add_argument('--pareto-csv', type=str, default=None,
                        help='Only run configs in this CSV (e.g. results_attack/pareto_configs.csv)')
    p_run.add_argument('--no-save-tokens', action='store_true')
    p_run.add_argument('--force', action='store_true')

    # aggregate only
    p_agg = sub.add_parser('aggregate', help='Re-aggregate existing detail JSONs to CSV')
    p_agg.add_argument('--input-dir', type=str, default='results_attack_detail')
    p_agg.add_argument('--summary-csv', type=str, default='results_attack/aggregated_attack_results.csv')

    args = parser.parse_args()

    if args.command == 'aggregate':
        aggregate(Path(args.input_dir), Path(args.summary_csv))
        return

    # Run
    max_token_lengths = [int(x) for x in args.token_lengths.split(',')]
    cp_ratios = [float(x) for x in args.cp_ratios.split(',')]
    seeds = [int(x) for x in args.seed.split(',')]
    selected_models = set(args.models.split(',')) if args.models else None
    selected_datasets = set(args.datasets.split(',')) if args.datasets else None
    results_dirs = args.results_dirs.split(',')
    save_tokens = not args.no_save_tokens

    # Common tag for cp ratios
    cp_tag = '_'.join(f'{r:.2f}'.replace('0.', '') for r in cp_ratios)

    # Build experiment list (shared across seeds)
    all_experiments = generate_experiment_list()
    if selected_models or selected_datasets:
        all_experiments = [e for e in all_experiments
                          if (not selected_models or e['model'] in selected_models)
                          and (not selected_datasets or e['dataset'] in selected_datasets)]

    wm_files = []
    for exp in all_experiments:
        path = find_file(exp['experiment_name'], results_dirs)
        if path:
            exp['path'] = path
            wm_files.append(exp)

    if args.pareto_csv:
        import pandas as pd
        pcfg = pd.read_csv(args.pareto_csv)
        pareto_keys = set(zip(pcfg['method'], pcfg['model'], pcfg['dataset'],
                              pcfg['gamma'].round(6), pcfg['delta'].round(6)))
        wm_files = [e for e in wm_files
                    if (e['method'], e['model'], e['dataset'],
                        round(e['gamma'], 6), round(e['delta'], 6)) in pareto_keys]

    baseline_map = build_baseline_map(results_dirs)
    for exp in wm_files:
        key = (exp['model'], exp['dataset'], f"{exp['gamma']:.2f}", exp['seed'])
        exp['baseline_path'] = baseline_map.get(key)

    print(f"Experiments: {len(wm_files)}, Methods: {Counter(f['method'] for f in wm_files).most_common()}")

    # Load tokenizers + detectors once (shared across seeds)
    from transformers import AutoTokenizer
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizers = {}
    detectors = {}
    for mk in set(wm['model'] for wm in wm_files):
        print(f"Loading tokenizer: {mk}")
        tokenizers[mk] = AutoTokenizer.from_pretrained(
            MODEL_NAMES.get(mk, mk), cache_dir=args.hf_cache_dir)

    # Loop over (seed, token_length, cp_ratio_set). Each gets its own directory.
    from tqdm.auto import tqdm
    configs = [(s, n) for s in seeds for n in max_token_lengths]
    print(f"\nTotal configs: {len(configs)} (seeds={seeds} x lengths={max_token_lengths})")

    for global_seed, n_tok in configs:
        run_tag = f"s{global_seed}_n{n_tok}_cp{cp_tag}"
        output_dir = Path(f'results_attack/{run_tag}/detail')
        summary_csv = Path(f'results_attack/{run_tag}/aggregated.csv')
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'=' * 60}")
        print(f"seed={global_seed}  n={n_tok}  cp={cp_ratios}  ->  {output_dir}/")
        print(f"{'=' * 60}")

        # Skip done
        if args.force:
            todo = wm_files
        else:
            already_done = {f.stem for f in output_dir.glob('*.json')}
            todo = [wm for wm in wm_files if wm['experiment_name'] not in already_done]
            if already_done:
                print(f"Already done: {len(already_done)}, to process: {len(todo)}")

        if not todo:
            print("Nothing to process.")
        else:
            t_start = time.time()
            total_samples = 0
            errors = []

            for wm_info in tqdm(todo, desc=f"s{global_seed}/n{n_tok}"):
                try:
                    with open(wm_info['path'], 'r', encoding='utf-8') as f:
                        exp_config = json.load(f).get('config', {})
                    cfg_gamma = exp_config.get('gamma', wm_info['gamma'])
                    cfg_scheme = exp_config.get('seeding_scheme', 'simple_1')
                    det_key = (wm_info['model'], cfg_gamma, cfg_scheme)
                    if det_key not in detectors:
                        detectors[det_key] = create_detector(
                            tokenizers[wm_info['model']], exp_config, device)
                    wm_info['gamma'] = cfg_gamma

                    n = process_experiment(
                        wm_info=wm_info,
                        tokenizer=tokenizers[wm_info['model']],
                        detector=detectors[det_key],
                        max_token_lengths=[n_tok],
                        save_tokens=save_tokens,
                        output_dir=output_dir,
                        baseline_path=wm_info.get('baseline_path'),
                        cp_ratios=cp_ratios,
                        global_seed=global_seed,
                    )
                    total_samples += n
                except Exception as e:
                    errors.append((wm_info['experiment_name'], str(e)))

            elapsed = time.time() - t_start
            print(f"Completed {len(todo) - len(errors)} experiments, "
                  f"{total_samples} samples in {elapsed / 60:.1f} min")
            if errors:
                print(f"Errors: {len(errors)}")
                for name, err in errors[:10]:
                    print(f"  {name}: {err[:80]}")

        # Aggregate this config
        print(f"Aggregating -> {summary_csv}")
        aggregate(output_dir, summary_csv)


if __name__ == '__main__':
    main()
