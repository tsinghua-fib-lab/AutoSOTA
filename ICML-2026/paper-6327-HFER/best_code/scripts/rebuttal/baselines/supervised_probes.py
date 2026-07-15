"""
Supervised Probes Comparison (Day 2 CPU)
==========================================
Addresses: Reviewer QygV ("doesn't compare against supervised approaches")

Input: data/results/rebuttal/hidden_states_llama8b.npz
Output: comparison table

Usage:
    python scripts/rebuttal/supervised_probes.py
"""

import os, sys, json, time
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

NPZ_FILE = "data/results/rebuttal/hidden_states_llama8b.npz"


def run_probes():
    print("=" * 70)
    print("  SUPERVISED PROBES COMPARISON")
    print("=" * 70)

    if not os.path.exists(NPZ_FILE):
        print(f"ERROR: {NPZ_FILE} not found. Run day1_llama8b_batch.py first.")
        sys.exit(1)

    os.makedirs("output/rebuttal", exist_ok=True)
    os.makedirs("data/results/rebuttal", exist_ok=True)

    data = np.load(NPZ_FILE, allow_pickle=True)
    X_L24 = data["hidden_L24"].astype(np.float32)
    X_L30 = data["hidden_L30"].astype(np.float32)
    y = data["labels"]
    filenames = data["filenames"]

    print(f"Loaded: {len(y)} proofs, dim={X_L30.shape[1]}")
    print(f"  Valid: {sum(y)}, Invalid: {len(y) - sum(y)}")

    # 60/20/20 split
    X_train, X_test, y_train, y_test = train_test_split(
        X_L30, y, test_size=0.2, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.25, random_state=42, stratify=y_train)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    results = []

    # 1. Linear Probe
    t0 = time.time()
    lr = LogisticRegression(max_iter=1000, C=1.0)
    lr.fit(X_train_s, y_train)
    t_lr = time.time() - t0
    test_acc = lr.score(X_test_s, y_test)
    results.append({
        "method": "Linear Probe", "test_acc": round(test_acc, 4),
        "params": int(X_L30.shape[1] + 1), "train_data": len(X_train),
        "train_time": f"{t_lr:.1f}s"
    })

    # 2. Random Forest (PCA to 100 dims)
    pca = PCA(n_components=min(100, X_L30.shape[1]))
    X_train_pca = pca.fit_transform(X_train_s)
    X_test_pca = pca.transform(X_test_s)
    t0 = time.time()
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train_pca, y_train)
    t_rf = time.time() - t0
    test_acc = rf.score(X_test_pca, y_test)
    results.append({
        "method": "Random Forest (PCA-100)", "test_acc": round(test_acc, 4),
        "params": "~100k trees", "train_data": len(X_train),
        "train_time": f"{t_rf:.1f}s"
    })

    # 3. MLP Probe
    t0 = time.time()
    mlp = MLPClassifier(hidden_layer_sizes=(256,), max_iter=500, random_state=42,
                         early_stopping=True, validation_fraction=0.15)
    mlp.fit(X_train_s, y_train)
    t_mlp = time.time() - t0
    test_acc = mlp.score(X_test_s, y_test)
    n_params = X_L30.shape[1] * 256 + 256 + 256 + 1
    results.append({
        "method": "MLP (256-dim)", "test_acc": round(test_acc, 4),
        "params": f"~{n_params//1000}k", "train_data": len(X_train),
        "train_time": f"{t_mlp:.1f}s"
    })

    # 4. Spectral baselines (from paper)
    results.append({
        "method": "Spectral (calibrated)", "test_acc": 0.941,
        "params": 1, "train_data": "50 calibration", "train_time": "0s"
    })
    results.append({
        "method": "Spectral (nested CV)", "test_acc": 0.859,
        "params": 1, "train_data": "0 (unsupervised)", "train_time": "0s"
    })

    # Print table
    print(f"\n{'Method':<25} | {'Test Acc':>8} | {'Params':>10} | {'Train':>15} | {'Time':>6}")
    print("-" * 75)
    for r in results:
        acc_str = f"{r['test_acc']*100:.1f}%"
        print(f"{r['method']:<25} | {acc_str:>8} | {str(r['params']):>10} | "
              f"{str(r['train_data']):>15} | {r['train_time']:>6}")

    # Save
    with open("data/results/rebuttal/supervised_probes.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved: data/results/rebuttal/supervised_probes.json")

    # LaTeX
    latex = [
        r"\begin{table}[t]",
        r"\caption{Supervised probes vs.\ training-free spectral method (Llama-3.1-8B, L30).}",
        r"\label{tab:probes}",
        r"\centering\small",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Method & Acc & \# Params & Training Data \\",
        r"\midrule",
    ]
    for r in results:
        latex.append(f"  {r['method']} & {r['test_acc']*100:.1f}\\% & "
                    f"{r['params']} & {r['train_data']} \\\\")
    latex.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    with open("output/rebuttal/supervised_probes_table.tex", "w") as f:
        f.write("\n".join(latex))
    print(f"Saved: output/rebuttal/supervised_probes_table.tex")


if __name__ == "__main__":
    run_probes()
