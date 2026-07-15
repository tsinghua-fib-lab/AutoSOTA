import json
import os
import glob
import numpy as np
import random
from sklearn.model_selection import train_test_split

def load_data(results_file):
    with open(results_file, 'r') as f:
        data = json.load(f)
    return data

def safe_get(traj, layer_idx, metric):
    if not traj:
        return None
    # If trajectory is shorter than layer_idx, use the last one? 
    # Or return None (missing data)?
    # User asked for L11. If model has 32 layers, L11 exists.
    # If model is shallow or proof is short?
    # Let's use safe indexing. If index out of bounds, use LAST layer (common practice for fixed-depth probes if early exit)
    # OR return None.
    # Given "spectral features" are usually fixed size padded or interpolated, 
    # but here `trajectory` is a list.
    # I will try to get index 11. If not, last index.
    idx = layer_idx
    if idx >= len(traj):
        idx = -1
    return traj[idx].get(metric)

def run_experiment_3(results_dir, output_file, target_layer=11, target_metric="hfer"):
    results = []
    
    files = glob.glob(os.path.join(results_dir, "experiment_results_*.json"))
    
    for res_file in files:
        base = os.path.basename(res_file)
        model_name = base.replace("experiment_results_", "").replace(".json", "")
        
        data = load_data(res_file)
        
        # Collect samples
        # Valid (Human/Reference) -> Label 1
        # Invalid (Model) -> Label 0
        samples = []
        for item in data['valid']:
            val = safe_get(item['trajectory'], target_layer, target_metric)
            if val is not None:
                samples.append({'val': val, 'label': 1})
                
        for item in data['invalid']:
            val = safe_get(item['trajectory'], target_layer, target_metric)
            if val is not None:
                samples.append({'val': val, 'label': 0})
                
        # Total should be around 454
        if len(samples) < 10:
            print(f"Skipping {model_name}: Not enough samples ({len(samples)})")
            continue
            
        # Shuffle (Reproducible?)
        random.seed(42)
        random.shuffle(samples)
        
        X = np.array([s['val'] for s in samples])
        y = np.array([s['label'] for s in samples])
        
        # Split: 272 train / 91 val / 91 test
        # 454 total. 
        # 272/454 = 0.6
        # 91/454 = 0.2
        # Train (60%), Rest (40%)
        X_train, X_rest, y_train, y_rest = train_test_split(X, y, train_size=0.6, random_state=42)
        # Split Rest into Val (50%) and Test (50%) -> 20% total each
        X_val, X_test, y_val, y_test = train_test_split(X_rest, y_rest, test_size=0.5, random_state=42)
        
        # Val: Sweep thresholds
        best_acc = 0
        best_thresh = 0
        
        # Determine sweep range
        min_v, max_v = X_train.min(), X_train.max()
        thresholds = np.linspace(min_v, max_v, 100)
        
        for t in thresholds:
            # Predict
            # HFER usually: Lower is "smoother"? Or "valid"?
            # Typically HFER is High Freq Energy Ratio.
            # Valid proofs (Human) might be smoother -> Lower HFER?
            # Invalid (Hallucination) -> High HFER (Noise)?
            # Let's check direction.
            # We try both directions: < t is Valid, and > t is Valid.
            
            # Direction 1: Val < t => 1
            pred1 = (X_val < t).astype(int)
            acc1 = np.mean(pred1 == y_val)
            
            # Direction 2: Val > t => 1
            pred2 = (X_val > t).astype(int)
            acc2 = np.mean(pred2 == y_val)
            
            if acc1 > best_acc:
                best_acc = acc1
                best_thresh = (t, 'lower') # Valid is Lower
            
            if acc2 > best_acc:
                best_acc = acc2
                best_thresh = (t, 'higher') # Valid is Higher
                
        # Test: Apply best threshold
        t, direction = best_thresh
        if direction == 'lower':
            test_preds = (X_test < t).astype(int)
        else:
            test_preds = (X_test > t).astype(int)
            
        test_acc = np.mean(test_preds == y_test)
        train_acc = 0 # Placeholder if needed
        
        results.append({
            "Model": model_name,
            "Train_N": len(X_train),
            "Val_Acc": best_acc,
            "Test_Acc": test_acc,
            "Threshold": t,
            "Direction": direction
        })
        
    # Output to markdown
    print(f"| Model | Test Accuracy | Val Accuracy | Direction | Threshold |")
    print(f"|---|---|---|---|---|")
    for r in results:
        print(f"| {r['Model']} | **{r['Test_Acc']*100:.1f}%** | {r['Val_Acc']*100:.1f}% | {r['Direction']} | {r['Threshold']:.4f} |")

if __name__ == "__main__":
    run_experiment_3("data/results", "exp3.md")
