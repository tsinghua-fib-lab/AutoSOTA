import argparse, os, json
import numpy as np
import pandas as pd
from sklearn.metrics import auc, roc_curve

def calc_performance(scores_original, scores_watermarked):
    y_true = [0] * len(scores_original) + [1] * len(scores_watermarked)
    y_true = pd.Series(y_true).fillna(0)
    y_scores = pd.Series(scores_original + scores_watermarked).fillna(0)
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    def safe_interp(x):
        if x < fpr[0]: return tpr[0]
        elif x > fpr[-1]: return tpr[-1]
        return float(np.interp(x, fpr, tpr))
    return {
        'roc_auc': roc_auc,
        'tpr@0.1%': safe_interp(0.001),
        'tpr@0.5%': safe_interp(0.005),
        'tpr@1%': safe_interp(0.01),
        'tpr@5%': safe_interp(0.05),
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--watermark_block_size', type=int, default=8)
    parser.add_argument('--dataset_name', type=str, default='c4')
    args = parser.parse_args()
    
    B = args.watermark_block_size
    D = args.dataset_name
    det_file = f'_result/detection/block_size_{B}/{D}_AliMark_facebook_opt-1.3b.json'
    
    if not os.path.exists(det_file):
        print(f'ERROR: {det_file} not found')
        exit(1)
    
    df = pd.read_json(det_file, orient='index')
    print(f'Loaded {len(df)} detection results')
    print(f'Columns: {list(df.columns)}')
    
    # Collect scores
    scores_map = {}
    for idx, row in df.iterrows():
        for col in df.columns:
            if col in ['question', 'reference', 'unwatermarked_result']:
                continue
            if not isinstance(row[col], dict):
                continue
            result = row[col]
            if 'detect_result' in result:
                score = result['detect_result']['score']
                scores_map.setdefault(col, []).append(score)
    
    original_scores = scores_map.get('original_result', [])
    print(f'Original scores: {len(original_scores)}')
    
    # Print results for each attack type
    attack_cols = [c for c in scores_map if 'result' in c and c != 'original_result' and c != 'unwatermarked_result']
    print(f'\n{"Attack":<45} | {"AUROC":<8} | {"TPR@0.1%":<10} | {"TPR@0.5%":<10} | {"TPR@1%":<9} | {"TPR@5%":<9} | {"N":<6}')
    print('-' * 120)
    
    for col in sorted(attack_cols):
        if col not in scores_map:
            continue
        sw = scores_map[col]
        if len(sw) == 0:
            continue
        perf = calc_performance(original_scores, sw)
        label = col.replace('_result', '')
        print(f'{label:<45} | {perf["roc_auc"]:.4f}  | {perf["tpr@0.1%"]:.4f}    | {perf["tpr@0.5%"]:.4f}    | {perf["tpr@1%"]:.4f}   | {perf["tpr@5%"]:.4f}   | {len(sw):<4}')
