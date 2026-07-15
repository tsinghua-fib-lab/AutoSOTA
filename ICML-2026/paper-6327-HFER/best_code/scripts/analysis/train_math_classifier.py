import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneOut, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

RESULTS_FILE = "data/results/experiment_results_MATH_Llama-1B.json"

def load_data():
    with open(RESULTS_FILE, 'r') as f:
        data = json.load(f)
    
    # Extract features
    # We have 'valid' and 'invalid' lists
    # Each item has 'spectral_stats': { 'layer_0': { 'hfer': ..., 'entropy': ... } }
    
    X = []
    y = []
    
    # Metrics to use
    metrics = ['hfer', 'fiedler_value', 'smoothness', 'entropy', 'energy']
    
    def extract_features(item):
        feats = []
        traj = item['trajectory']
        # Traj is a list of dicts, one per layer
        for layer_stats in traj:
            for m in metrics:
                feats.append(layer_stats.get(m, 0))
        return feats

    for item in data['valid']:
        try:
            X.append(extract_features(item))
            y.append(1) # Valid
        except KeyError as e:
            print(f"KeyError in VALID item: {e}")
            print(f"Keys: {item.keys()}")
            # return np.array([]), np.array([]) # simple crash
            raise e
        
    for item in data['invalid']:
        try:
            X.append(extract_features(item))
            y.append(0) # Invalid
        except KeyError as e:
            print(f"KeyError in INVALID item: {e}")
            print(f"Keys: {item.keys()}")
            raise e
        
    return np.array(X), np.array(y)

def main():
    print(f"Loading data from {RESULTS_FILE}...")
    try:
        X, y = load_data()
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    print(f"Data shape: {X.shape}")
    print(f"Class distribution: Valid={sum(y)}, Invalid={len(y)-sum(y)}")
    
    # Scaling is important for LR
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 1. Logistic Regression (Linear Combo)
    print("\n--- Logistic Regression (Leave-One-Out CV) ---")
    lr = LogisticRegression(class_weight='balanced', solver='liblinear', penalty='l1', C=1.0)
    # Using L1 for sparsity (feature selection)
    
    cv = LeaveOneOut()
    scores_lr = cross_val_score(lr, X_scaled, y, cv=cv, scoring='accuracy')
    print(f"Mean CV Accuracy: {scores_lr.mean():.4f}")
    
    # 2. Random Forest (Non-linear Combo)
    print("\n--- Random Forest (Leave-One-Out CV) ---")
    rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', max_depth=5, random_state=42)
    scores_rf = cross_val_score(rf, X_scaled, y, cv=cv, scoring='accuracy')
    print(f"Mean CV Accuracy: {scores_rf.mean():.4f}")
    
    # 3. Best Single Feature (Baseline)
    print("\n--- Best Single Feature Baseline ---")
    best_acc = 0
    best_feat_idx = -1
    
    # Iterate all features
    for i in range(X.shape[1]):
        # Simple threshold classifier for this feature
        feat = X[:, i]
        # Check both directions
        # Direction 1: > threshold is valid
        # Direction 2: < threshold is valid
        # Efficient way: sort and try split points
        
        # Quick approximation: usage of decision stump
        # Or just use raw correlation logic to pick direction
        
        # Let's just use LR on single feature to be fair
        lr_single = LogisticRegression(class_weight='balanced', solver='liblinear')
        s = cross_val_score(lr_single, feat.reshape(-1, 1), y, cv=cv, scoring='accuracy')
        acc = s.mean()
        
        if acc > best_acc:
            best_acc = acc
            best_feat_idx = i
            
    print(f"Best Single Feature CV Accuracy: {best_acc:.4f}")
    
    # Interpretation
    print("\n" + "="*40)
    print(f"MAX ACHIEVABLE ACCURACY (Combined): {max(scores_lr.mean(), scores_rf.mean()):.4f}")
    print("="*40)

if __name__ == "__main__":
    main()
