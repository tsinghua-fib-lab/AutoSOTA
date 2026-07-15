import json
import os
import glob
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

def load_data(results_file):
    with open(results_file, 'r') as f:
        data = json.load(f)
    return data

def get_feature(traj, layer_idx, metric):
    if not traj: return None
    # Handle negative index for 'last'
    if layer_idx == -1:
        return traj[-1].get(metric)
    
    if layer_idx < len(traj):
        return traj[layer_idx].get(metric)
    return None

def optimize_threshold(X, y):
    # Returns best_acc, best_thresh, direction
    if len(X) == 0: return 0, 0, 'lower'
    
    min_v, max_v = X.min(), X.max()
    thresholds = np.linspace(min_v, max_v, 50) # 50 steps
    best_acc = 0
    best_t = 0
    best_d = 'lower'
    
    for t in thresholds:
        # Lower
        pred1 = (X < t).astype(int)
        acc1 = np.mean(pred1 == y)
        if acc1 > best_acc:
            best_acc = acc1
            best_t = t
            best_d = 'lower'
            
        # Higher
        pred2 = (X > t).astype(int)
        acc2 = np.mean(pred2 == y)
        if acc2 > best_acc:
            best_acc = acc2
            best_t = t
            best_d = 'higher'
            
    return best_acc, best_t, best_d

def run_experiment_4(results_dir):
    files = glob.glob(os.path.join(results_dir, "experiment_results_*.json"))
    
    metrics = ["fiedler_value", "hfer", "smoothness", "entropy"]
    layers = [5, 10, 15, 20, 25, 30, -1] # Search space
    
    print(f"| Model | Nested CV Accuracy | Best Config (Freq) |")
    print(f"|---|---|---|")
    
    for res_file in files:
        base = os.path.basename(res_file)
        model_name = base.replace("experiment_results_", "").replace(".json", "")
        data = load_data(res_file)
        
        # Prepare Data Objects
        # We need a way to access (metric, layer) quickly.
        # Let's pre-extract all features?
        # N samples.
        # Features: Dict maps (metric, layer) -> val
        
        samples = []
        for item in data['valid']:
            traj = item['trajectory']
            feat = {}
            for m in metrics:
                for l in layers:
                    v = get_feature(traj, l, m)
                    if v is not None: feat[(m, l)] = v
            samples.append({'features': feat, 'label': 1})
            
        for item in data['invalid']:
            traj = item['trajectory']
            feat = {}
            for m in metrics:
                for l in layers:
                    v = get_feature(traj, l, m)
                    if v is not None: feat[(m, l)] = v
            samples.append({'features': feat, 'label': 0})
            
        if len(samples) < 20: continue
        
        y = np.array([s['label'] for s in samples])
        
        # Outer Loop
        outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        outer_scores = []
        best_configs = []
        
        for train_ix, test_ix in outer_cv.split(samples, y):
            # Inner Loop (Grid Search)
            train_samples = [samples[i] for i in train_ix]
            train_y = y[train_ix]
            
            # We want to find best (metric, layer).
            # For each (m, l), we do 4-Fold CV to find avg validation score.
            
            inner_cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
            
            grid_scores = {} # (m, l) -> avg_score
            
            # Pre-compute X arrays for each config to speed up
            # Dict: (m, l) -> np.array of values for train_samples
            X_dict = {}
            for m in metrics:
                for l in layers:
                    vals = []
                    valid_mask = [] # To handle missing values?
                    # If value missing for a sample, we can't use it or impute.
                    # Current strategy: Drop samples with missing values for that feature?
                    # Or impute mean.
                    # Let's simple impute mean of observed.
                    raw_vals = [s['features'].get((m,l)) for s in train_samples]
                    valid_chk = [v for v in raw_vals if v is not None]
                    if not valid_chk: 
                        X_dict[(m,l)] = None
                        continue
                    mean_v = np.mean(valid_chk)
                    final_vals = [v if v is not None else mean_v for v in raw_vals]
                    X_dict[(m,l)] = np.array(final_vals)

            best_inner_score = -1
            best_inner_config = None # (m, l, t, d)
            
            # Iterate Grid
            for m in metrics:
                for l in layers:
                    X_feat = X_dict.get((m,l))
                    if X_feat is None: continue
                    
                    # 4-Fold inner CV
                    fold_scores = []
                    thresholds_votes = [] # To Average? Or just re-fit on full train?
                    # Proper Nested CV:
                    # For grid search, we compare hyperparameters. Hyperparams here are (Metric, Layer).
                    # Threshold is a parameter learned on training.
                    # So: For split k: Fit Threshold on Inner_Train, Score on Inner_Val.
                    
                    for in_tr_ix, in_val_ix in inner_cv.split(X_feat, train_y):
                        X_in_tr, X_in_val = X_feat[in_tr_ix], X_feat[in_val_ix]
                        y_in_tr, y_in_val = train_y[in_tr_ix], train_y[in_val_ix]
                        
                        acc, t, d = optimize_threshold(X_in_tr, y_in_tr)
                        
                        # Apply to Val
                        if d == 'lower':
                            pred = (X_in_val < t).astype(int)
                        else:
                            pred = (X_in_val > t).astype(int)
                        fold_scores.append(np.mean(pred == y_in_val))
                        
                    avg_score = np.mean(fold_scores)
                    if avg_score > best_inner_score:
                        best_inner_score = avg_score
                        best_inner_config = (m, l)
            
            # Refit best config on FULL Outer Train
            bm, bl = best_inner_config
            # Get X for full outer train
            X_train_best = X_dict[(bm, bl)]
            # Optimize Threshold on full outer train
            acc, bt, bd = optimize_threshold(X_train_best, train_y)
            
            # Evaluate on Outer Test
            test_samples = [samples[i] for i in test_ix]
            # Impute using TRAIN mean
            train_mean = np.mean(X_train_best)
            raw_test = [s['features'].get((bm,bl)) for s in test_samples]
            X_test_best = np.array([v if v is not None else train_mean for v in raw_test])
            
            if bd == 'lower':
                test_pred = (X_test_best < bt).astype(int)
            else:
                test_pred = (X_test_best > bt).astype(int)
                
            test_acc = np.mean(test_pred == y[test_ix])
            outer_scores.append(test_acc)
            best_configs.append(f"{bm}@L{bl}")
            
        mean_acc = np.mean(outer_scores)
        std_acc = np.std(outer_scores)
        
        # Find most frequent config
        from collections import Counter
        most_common = Counter(best_configs).most_common(1)[0][0]
        
        print(f"| {model_name} | {mean_acc*100:.1f}% ± {std_acc*100:.1f}% | {most_common} |")

if __name__ == "__main__":
    run_experiment_4("data/results")
