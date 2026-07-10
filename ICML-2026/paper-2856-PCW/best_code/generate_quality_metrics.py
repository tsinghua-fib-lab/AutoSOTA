"""
Generate quality metrics data (BLEU, ROUGE, BERTScore) for watermark experiments.
This script computes text quality scores comparing watermarked vs non-watermarked outputs.
Results are saved to CSV files only; no figures are generated.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import norm
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

# ============== Configuration ==============
RESULTS_DIR = Path('results_final')
EXPERIMENT_PARAMS_FILE = './optimal_power_results_combined.csv'
OUTPUT_DIR = Path('./quality_metrics')

# ============== Scoring Functions ==============
_rouge_scorer = None
def get_rouge_scorer():
    global _rouge_scorer
    if _rouge_scorer is None:
        _rouge_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    return _rouge_scorer


def compute_bleu_batch(pairs):
    """Batch compute BLEU scores"""
    if len(pairs) == 0:
        return []
    smoothie = SmoothingFunction().method1
    results = []
    for ref, hyp in pairs:
        if not ref or not hyp:
            results.append(0.0)
        else:
            try:
                results.append(sentence_bleu([ref.split()], hyp.split(), smoothing_function=smoothie))
            except:
                results.append(0.0)
    return results


def compute_rouge_batch(pairs):
    """Batch compute ROUGE-L scores"""
    if len(pairs) == 0:
        return []
    scorer = get_rouge_scorer()
    results = []
    for ref, hyp in pairs:
        if not ref or not hyp:
            results.append(0.0)
        else:
            try:
                scores = scorer.score(ref, hyp)
                results.append(scores['rougeL'].fmeasure)
            except:
                results.append(0.0)
    return results


# --- Unused scoring variants (kept for reference) ---
# compute_bleu_batch_truncated: BLEU with first N tokens only
# compute_rouge_batch_truncated: ROUGE-L with first N tokens only
# compute_rouge1_batch: ROUGE-1 (unigram)
# compute_rouge2_batch: ROUGE-2 (bigram)
# compute_bleu_batch_method4: BLEU with method4 smoothing


# ============== Baseline Loading ==============
GAMMA_TO_BASEGAMMA = None


def build_gamma_to_basegamma_map(experiment_params):
    """Build gamma -> base_gamma mapping from CSV"""
    if 'base_gamma' not in experiment_params.columns:
        print("No base_gamma column found; using exact/approx baseline matching only")
        return {}

    gamma_map = {}
    for _, row in experiment_params.iterrows():
        gamma = row['gamma']
        base_gamma = row['base_gamma']
        gamma_map[gamma] = base_gamma
    print(f"Built gamma -> base_gamma map with {len(gamma_map)} entries")
    return gamma_map


def get_gamma_map(experiment_params):
    global GAMMA_TO_BASEGAMMA
    if GAMMA_TO_BASEGAMMA is None:
        GAMMA_TO_BASEGAMMA = build_gamma_to_basegamma_map(experiment_params)
    return GAMMA_TO_BASEGAMMA


def load_nonwatermark_baseline(model, dataset):
    """Load non-watermarked (delta=0) texts as baseline reference."""
    result_files = sorted([f for f in RESULTS_DIR.glob('*.json') if not f.name.endswith('_config.json')])

    baseline_data = {}
    gamma_values_loaded = set()

    for file_path in result_files:
        if model not in file_path.name or dataset not in file_path.name:
            continue

        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        delta = data['config']['delta']
        if delta != 0:
            continue

        seed = data['config']['seed']
        temp = data['config']['temperature']
        gamma = data['config']['gamma']
        gamma_values_loaded.add(gamma)

        for idx, result in enumerate(data['results']):
            nonwm_text = result['watermarked'].get('generated_text', '')
            prefix = result.get('truncated_text', '')
            key = (temp, gamma, seed, prefix)
            baseline_data[key] = {
                'text': nonwm_text,
                'prefix': prefix
            }

    print(f"Loaded {len(baseline_data)} non-watermarked baseline texts")
    print(f"  Unique gamma values in baseline: {sorted(gamma_values_loaded)[:5]}... (total {len(gamma_values_loaded)})")
    return baseline_data


def find_baseline_gamma(experiment_gamma, baseline_data, temp, seed, prefix, experiment_params):
    """Try multiple ways to find matching baseline using prefix as key."""
    # Method 1: Exact match
    key = (temp, experiment_gamma, seed, prefix)
    if key in baseline_data:
        return baseline_data[key], 'exact'

    # Method 2: Use base_gamma mapping
    gamma_map = get_gamma_map(experiment_params)
    if experiment_gamma in gamma_map:
        base_gamma = gamma_map[experiment_gamma]
        key = (temp, base_gamma, seed, prefix)
        if key in baseline_data:
            return baseline_data[key], 'base_gamma'

    # Method 3: Find closest gamma in baseline (tolerance 0.01)
    for (t, g, s, p), data in baseline_data.items():
        if t == temp and s == seed and p == prefix and abs(g - experiment_gamma) < 0.01:
            return data, 'approx'

    return None, 'not_found'


def verify_prefix_match(prefix1, prefix2, tolerance=0.95):
    """Verify that two prefixes match (allowing minor differences)."""
    if not prefix1 or not prefix2:
        return False
    if prefix1 == prefix2:
        return True
    if prefix1 in prefix2 or prefix2 in prefix1:
        return True

    min_len = min(len(prefix1), len(prefix2))
    max_len = max(len(prefix1), len(prefix2))
    if min_len == 0:
        return False

    matching = 0
    for c1, c2 in zip(prefix1, prefix2):
        if c1 == c2:
            matching += 1
        else:
            break

    return matching / max_len >= tolerance


# ============== Data Generation ==============
def generate_quality_data(model, dataset, metric='bleu', experiment_params=None):
    """Generate quality data for a specific model/dataset/metric combination.

    Args:
        model: Model name (e.g., 'opt-125m')
        dataset: Dataset name (e.g., 'c4', 'lfqa', 'wikipedia')
        metric: 'bleu' or 'rouge'
        experiment_params: DataFrame with experiment parameters

    Returns:
        results_df: DataFrame with per-sample results
        agg_df: DataFrame with aggregated results
    """
    result_files = sorted([f for f in RESULTS_DIR.glob('*.json') if not f.name.endswith('_config.json')])
    print(f"Found {len(result_files)} result files")

    include = [f for f in result_files if dataset in f.name and model in f.name]
    result_files = include
    print(f"Using {len(result_files)} result files")

    baseline_data = load_nonwatermark_baseline(model, dataset)

    # Collect all data
    data_dicts = {'grid': {}, 'klt': {}, 'dp': {}, 'opt': {}}
    stats = {m: {'match': 0, 'no_base': 0, 'match_type': {}} for m in data_dicts}

    all_pairs = []
    pair_index_map = []

    for file_path in result_files:
        fname = file_path.name
        if 'GRID' in fname:
            method = 'grid'
        elif 'KLFRONT' in fname:
            method = 'klt'
        elif 'DP' in fname:
            method = 'dp'
        elif fname.startswith('OPT_'):
            method = 'opt'
        else:
            continue

        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        seed, temp = data['config']['seed'], data['config']['temperature']
        gamma, delta = data['config']['gamma'], data['config']['delta']

        if delta == 0:
            continue

        if method == 'grid' and (gamma not in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] or delta > 10):
            continue
        if method == 'opt' and gamma not in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            continue

        for length in [30, 50, 100]:
            key = (temp, gamma, delta, length)
            if key not in data_dicts[method]:
                data_dicts[method][key] = {'zscore': [], 'meta': []}

            for idx, result in enumerate(data['results']):
                ngram = result['watermarked']['ngram_info']
                gt = np.array(result['watermarked']['green_token_mask'])
                if ngram:
                    gt = gt[np.array(ngram) == True]
                if len(gt) < length:
                    continue
                gt = gt[:length]

                zscore = (np.sum(gt) - gamma * length) / np.sqrt(length * gamma * (1 - gamma))

                prefix = result.get('truncated_text', '')
                baseline_info, match_type = find_baseline_gamma(gamma, baseline_data, temp, seed, prefix, experiment_params)
                if baseline_info is None:
                    stats[method]['no_base'] += 1
                    continue

                stats[method]['match'] += 1
                stats[method]['match_type'][match_type] = stats[method]['match_type'].get(match_type, 0) + 1

                global_idx = len(all_pairs)
                all_pairs.append((baseline_info['text'], result['watermarked'].get('generated_text', '')))

                data_dicts[method][key]['zscore'].append(zscore)
                data_dicts[method][key]['meta'].append({
                    'seed': seed, 'temp': temp, 'gamma': gamma, 'delta': delta,
                    'length': length, 'idx': idx, 'global_idx': global_idx
                })

    # Compute quality scores
    print(f"Computing {metric} scores for {len(all_pairs)} pairs...")
    compute_batch = compute_bleu_batch if metric == 'bleu' else compute_rouge_batch
    all_quality = compute_batch(all_pairs)

    # Assign quality scores back
    for method, method_dict in data_dicts.items():
        for key, val in method_dict.items():
            val['quality'] = [all_quality[m['global_idx']] for m in val['meta']]

    # Print statistics
    print(f"\nPrefix-based matching statistics:")
    total_all, nobase_all = 0, 0
    for m in ['grid', 'klt', 'dp', 'opt']:
        s = stats[m]
        total = s['match'] + s['no_base']
        if total > 0:
            nobase_pct = s['no_base'] / total * 100
            print(f"  {m.upper()}: matched={s['match']}, no_base={s['no_base']} ({nobase_pct:.1f}%), match_types={s['match_type']}")
            total_all += total
            nobase_all += s['no_base']
    if total_all > 0:
        print(f"  TOTAL: matched={total_all - nobase_all}, no_base={nobase_all} ({nobase_all / total_all * 100:.1f}%)")

    # Build results DataFrame
    all_results = []
    for method, method_dict in data_dicts.items():
        for key, val in method_dict.items():
            temp, gamma, delta, length = key
            for zscore, quality, meta in zip(val['zscore'], val['quality'], val['meta']):
                all_results.append({
                    'model': model, 'dataset': dataset, 'method': method, 'metric': metric,
                    'temp': temp, 'gamma': gamma, 'delta': delta, 'length': length,
                    'seed': meta['seed'], 'idx': meta['idx'], 'zscore': zscore, 'quality': quality
                })

    results_df = pd.DataFrame(all_results)

    # Aggregation
    agg_results = []
    for row in experiment_params.to_numpy():
        alpha, n, gamma, delta = row[:4]
        z_threshold = norm.ppf(1 - alpha)

        for method, method_dict in data_dicts.items():
            for key, val in method_dict.items():
                t, g, d, length = key
                if length != n:
                    continue
                if method == 'klt':
                    if not (abs(g - gamma) < 1e-5 and abs(d - delta) < 1e-5):
                        continue

                total = len(val['zscore'])
                if total == 0:
                    continue

                tpr = sum(1 for z in val['zscore'] if z > z_threshold) / total
                mean_q = np.mean(val['quality']) if val['quality'] else 0
                std_q = np.std(val['quality']) if val['quality'] else 0

                agg_results.append({
                    'model': model, 'dataset': dataset, 'method': method, 'metric': metric,
                    'alpha': alpha, 'n': n, 'gamma': g, 'delta': d,
                    'TPR': tpr, 'mean_quality': mean_q, 'std_quality': std_q, 'n_samples': total
                })

    agg_df = pd.DataFrame(agg_results)

    return results_df, agg_df, data_dicts


def generate_bertscore_data(model, dataset, experiment_params=None):
    """Generate BERTScore quality data."""
    from bert_score import score as bert_score_func

    result_files = sorted([f for f in RESULTS_DIR.glob('*.json') if not f.name.endswith('_config.json')])
    print(f"Found {len(result_files)} result files")

    include = [f for f in result_files if dataset in f.name and model in f.name]
    result_files = include
    print(f"Using {len(result_files)} result files")

    baseline_data = load_nonwatermark_baseline(model, dataset)

    data_dicts = {'grid': {}, 'klt': {}, 'dp': {}, 'opt': {}}
    stats = {m: {'match': 0, 'no_base': 0, 'match_type': {}} for m in data_dicts}

    all_refs, all_hyps = [], []

    for file_path in result_files:
        fname = file_path.name
        if 'GRID' in fname:
            method = 'grid'
        elif 'KLFRONT' in fname:
            method = 'klt'
        elif 'DP' in fname:
            method = 'dp'
        elif fname.startswith('OPT_'):
            method = 'opt'
        else:
            continue

        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        seed, temp = data['config']['seed'], data['config']['temperature']
        gamma, delta = data['config']['gamma'], data['config']['delta']

        if delta == 0:
            continue

        if method == 'grid' and delta > 10:
            continue

        for length in [30, 50, 100]:
            key = (temp, gamma, delta, length)
            if key not in data_dicts[method]:
                data_dicts[method][key] = {'zscore': [], 'meta': []}

            for idx, result in enumerate(data['results']):
                ngram = result['watermarked']['ngram_info']
                gt = np.array(result['watermarked']['green_token_mask'])
                if ngram:
                    gt = gt[np.array(ngram) == True]
                if len(gt) < length:
                    continue
                gt = gt[:length]

                zscore = (np.sum(gt) - gamma * length) / np.sqrt(length * gamma * (1 - gamma))

                prefix = result.get('truncated_text', '')
                baseline_info, match_type = find_baseline_gamma(gamma, baseline_data, temp, seed, prefix, experiment_params)
                if baseline_info is None:
                    stats[method]['no_base'] += 1
                    continue

                stats[method]['match'] += 1
                stats[method]['match_type'][match_type] = stats[method]['match_type'].get(match_type, 0) + 1

                global_idx = len(all_refs)
                ref_text = baseline_info['text'] if baseline_info['text'] else " "
                hyp_text = result['watermarked'].get('generated_text', '') or " "
                all_refs.append(ref_text)
                all_hyps.append(hyp_text)

                data_dicts[method][key]['zscore'].append(zscore)
                data_dicts[method][key]['meta'].append({
                    'seed': seed, 'temp': temp, 'gamma': gamma, 'delta': delta,
                    'length': length, 'idx': idx, 'global_idx': global_idx
                })

    print(f"\nPrefix-based matching statistics:")
    total_all, nobase_all = 0, 0
    for m in ['grid', 'klt', 'dp', 'opt']:
        s = stats[m]
        total = s['match'] + s['no_base']
        if total > 0:
            nobase_pct = s['no_base'] / total * 100
            print(f"  {m.upper()}: matched={s['match']}, no_base={s['no_base']} ({nobase_pct:.1f}%), match_types={s['match_type']}")
            total_all += total
            nobase_all += s['no_base']
    if total_all > 0:
        print(f"  TOTAL: matched={total_all - nobase_all}, no_base={nobase_all} ({nobase_all / total_all * 100:.1f}%)")

    # Compute BERTScore
    print(f"\nComputing BERTScore for {len(all_refs)} pairs...")
    if len(all_refs) > 0:
        P, R, F1 = bert_score_func(all_hyps, all_refs, model_type='roberta-large', verbose=True)
        all_quality = F1.tolist()
    else:
        all_quality = []

    # Assign scores back
    for method, method_dict in data_dicts.items():
        for key, val in method_dict.items():
            val['quality'] = [all_quality[m['global_idx']] for m in val['meta']]

    # Build results DataFrame
    all_results = []
    for method, method_dict in data_dicts.items():
        for key, val in method_dict.items():
            temp, gamma, delta, length = key
            for zscore, quality, meta in zip(val['zscore'], val['quality'], val['meta']):
                all_results.append({
                    'model': model, 'dataset': dataset, 'method': method, 'metric': 'bertscore',
                    'temp': temp, 'gamma': gamma, 'delta': delta, 'length': length,
                    'seed': meta['seed'], 'idx': meta['idx'], 'zscore': zscore, 'quality': quality
                })

    results_df = pd.DataFrame(all_results)

    # Aggregation
    agg_results = []
    for row in experiment_params.to_numpy():
        alpha, n, gamma, delta = row[:4]
        z_threshold = norm.ppf(1 - alpha)

        for method, method_dict in data_dicts.items():
            for key, val in method_dict.items():
                t, g, d, length = key
                if length != n:
                    continue
                if method == 'klt':
                    if not (abs(g - gamma) < 1e-5 and abs(d - delta) < 1e-5):
                        continue

                total = len(val['zscore'])
                if total == 0:
                    continue

                tpr = sum(1 for z in val['zscore'] if z > z_threshold) / total
                mean_q = np.mean(val['quality']) if val['quality'] else 0
                std_q = np.std(val['quality']) if val['quality'] else 0

                agg_results.append({
                    'model': model, 'dataset': dataset, 'method': method, 'metric': 'bertscore',
                    'alpha': alpha, 'n': n, 'gamma': g, 'delta': d,
                    'TPR': tpr, 'mean_quality': mean_q, 'std_quality': std_q, 'n_samples': total
                })

    agg_df = pd.DataFrame(agg_results)

    return results_df, agg_df, data_dicts


# ============== Main ==============
def _run_metric(metric_name, compute_fn, models, datasets, experiment_params, force=False):
    """Run a single metric for all model/dataset combos."""
    all_results = []
    all_agg = []

    for model in models:
        for dataset in datasets:
            agg_file = OUTPUT_DIR / f"quality_agg_vsNonWM_{metric_name}_{model}_{dataset}.csv"
            data_file = OUTPUT_DIR / f"quality_data_vsNonWM_{metric_name}_{model}_{dataset}.csv"

            if not force and agg_file.exists() and data_file.exists():
                print(f"\n[SKIP] {metric_name.upper()} for {model} on {dataset} - files exist")
                all_agg.append(pd.read_csv(agg_file))
                all_results.append(pd.read_csv(data_file))
                continue

            print(f"\n{'='*50}")
            print(f"{metric_name.upper()} for {model} on {dataset}")
            print('='*50)
            results_df, agg_df, _ = compute_fn(model, dataset, experiment_params=experiment_params)

            if len(results_df) > 0:
                results_df.to_csv(data_file, index=False)
                all_results.append(results_df)
            if len(agg_df) > 0:
                agg_df.to_csv(agg_file, index=False)
                all_agg.append(agg_df)

    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        combined.to_csv(OUTPUT_DIR / f'quality_data_vsNonWM_{metric_name}_all.csv', index=False)
        print(f"\nCombined {metric_name.upper()} data saved: {len(combined)} records")

    if all_agg:
        combined_agg = pd.concat(all_agg, ignore_index=True)
        combined_agg.to_csv(OUTPUT_DIR / f'quality_agg_vsNonWM_{metric_name}_all.csv', index=False)
        print(f"Combined {metric_name.upper()} aggregated data saved: {len(combined_agg)} records")


def main(force=False, only_metrics=None, models=None, datasets=None):
    """Generate all quality data and save to CSV files."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    experiment_params = pd.read_csv(EXPERIMENT_PARAMS_FILE)

    models = models or ['opt-125m', 'gpt2', 'pythia-160m']
    datasets = datasets or ['c4', 'lfqa', 'wikipedia']

    run_metrics = set(m.strip().lower() for m in only_metrics.split(',')) if only_metrics else {'bleu', 'rouge', 'bertscore'}

    if 'bleu' in run_metrics:
        print("\n" + "=" * 60)
        print("Generating BLEU data")
        print("=" * 60)
        _run_metric('bleu', lambda m, d, **kw: generate_quality_data(m, d, metric='bleu', **kw),
                    models, datasets, experiment_params, force)

    if 'rouge' in run_metrics:
        print("\n" + "=" * 60)
        print("Generating ROUGE data")
        print("=" * 60)
        _run_metric('rouge', lambda m, d, **kw: generate_quality_data(m, d, metric='rouge', **kw),
                    models, datasets, experiment_params, force)

    if 'bertscore' in run_metrics:
        print("\n" + "=" * 60)
        print("Generating BERTScore data")
        print("=" * 60)
        _run_metric('bertscore', generate_bertscore_data,
                    models, datasets, experiment_params, force)

    print("\n" + "=" * 60)
    print("Data generation complete!")
    print("=" * 60)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Generate quality metrics data')
    parser.add_argument('--force', action='store_true', help='Force recompute, ignore existing files')
    parser.add_argument('--metrics', type=str, default=None, help='Comma-separated metrics to compute: bleu,rouge,bertscore')
    parser.add_argument('--results-dir', type=str, default=str(RESULTS_DIR))
    parser.add_argument('--output-dir', type=str, default=str(OUTPUT_DIR))
    parser.add_argument('--params-file', type=str, default=EXPERIMENT_PARAMS_FILE)
    parser.add_argument('--models', type=str, default=None, help='Comma-separated model keys')
    parser.add_argument('--datasets', type=str, default=None, help='Comma-separated dataset keys')
    args = parser.parse_args()

    RESULTS_DIR = Path(args.results_dir)
    OUTPUT_DIR = Path(args.output_dir)
    EXPERIMENT_PARAMS_FILE = args.params_file
    OUTPUT_DIR.mkdir(exist_ok=True)

    selected_models = [x.strip() for x in args.models.split(',')] if args.models else None
    selected_datasets = [x.strip() for x in args.datasets.split(',')] if args.datasets else None
    main(force=args.force, only_metrics=args.metrics,
         models=selected_models, datasets=selected_datasets)
