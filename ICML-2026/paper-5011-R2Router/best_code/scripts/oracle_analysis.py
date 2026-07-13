#!/usr/bin/env python3
"""Oracle imitation: train on 10% data, evaluate routing accuracy on full set."""
import pickle, numpy as np, math
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from collections import Counter

TRAINING_DATA = "/orange/qi855292.ucf/ah872032.ucf/category_router/training_data.pkl"
MODELS_4 = ["235b", "80b", "gemini-flash", "haiku"]
CATEGORY_NAMES = ["Code", "Math", "Knowledge", "NLU", "Translation", "Trivia", "Domain"]

with open(TRAINING_DATA, "rb") as f:
    data = pickle.load(f)

embeddings = data["embeddings"]      # (8400, 1024)
categories = data["categories"]      # (8400,)
models_data = data["models"]
N = len(embeddings)

# --- Oracle labels: best model on concise per query ---
oracle_labels = np.zeros(N, dtype=int)
for i in range(N):
    best_acc = -1
    best_model = 0
    for j, mn in enumerate(MODELS_4):
        acc = models_data[mn]["concise"]["accuracy"][i]
        if acc > best_acc:
            best_acc = acc
            best_model = j
    oracle_labels[i] = best_model

print("Oracle label distribution:")
for j, mn in enumerate(MODELS_4):
    cnt = (oracle_labels == j).sum()
    print(f"  {mn:<20} {cnt:>5} ({cnt/N*100:.1f}%)")

# --- Arena score ---
beta = 0.1
def arena_score(mean_acc, cost_per_1kq):
    if cost_per_1kq <= 0: return 0
    A = mean_acc
    C = (math.log2(200) - math.log2(cost_per_1kq)) / (math.log2(200) - math.log2(0.0044))
    C = max(0.001, min(1, C))
    return ((1 + beta**2) * A * C) / (beta**2 * A + C) * 100

# --- Per-query cost on concise ---
query_costs = {}
for mn in MODELS_4:
    query_costs[mn] = models_data[mn]["concise"].get("output_tokens", np.zeros(N))

# Output prices per token
PRICES = {"235b": 0.463e-6, "80b": 1.1e-6, "gemini-flash": 2.5e-6, "haiku": 1.25e-6}

def routing_accuracy(predictions, accuracies_by_model):
    """Given model predictions per query, compute mean accuracy using concise budget."""
    accs = np.zeros(N)
    costs = np.zeros(N)
    for i in range(N):
        mn = MODELS_4[predictions[i]]
        accs[i] = accuracies_by_model[mn][i]
        tokens = query_costs[mn][i] if isinstance(query_costs[mn], np.ndarray) else 50
        costs[i] = tokens * PRICES[mn]
    return accs.mean(), costs.mean() * 1000

# Pre-compute per-model concise accuracy
acc_by_model = {mn: models_data[mn]["concise"]["accuracy"] for mn in MODELS_4}

# --- 10/90 split ---
sss = StratifiedShuffleSplit(n_splits=5, train_size=0.1, random_state=42)

print(f"\n{'='*75}")
print(f"{'Method':<35} {'Clf Acc':>8} {'Route Acc':>10} {'Cost/1kq':>10} {'Arena':>8}")
print(f"{'='*75}")

# Baseline: always 235b
always_235b_acc = acc_by_model["235b"].mean()
always_235b_cost = (query_costs["235b"] * PRICES["235b"]).mean() * 1000
always_235b_arena = arena_score(always_235b_acc, always_235b_cost)
print(f"{'Always 235b (no routing)':<35} {'--':>8} {always_235b_acc*100:>9.2f}% {f'${always_235b_cost:.4f}':>10} {always_235b_arena:>7.2f}")

# Oracle upper bound
oracle_route_acc, oracle_route_cost = routing_accuracy(oracle_labels, acc_by_model)
oracle_arena = arena_score(oracle_route_acc, oracle_route_cost)
print(f"{'Oracle (perfect, concise-only)':<35} {'100%':>8} {oracle_route_acc*100:>9.2f}% {f'${oracle_route_cost:.4f}':>10} {oracle_arena:>7.2f}")
print(f"{'-'*75}")

# Classifiers to test
classifiers = [
    ("LogReg C=0.01", lambda: Pipeline([("s", StandardScaler()), ("c", LogisticRegression(C=0.01, max_iter=3000))])),
    ("LogReg C=0.1", lambda: Pipeline([("s", StandardScaler()), ("c", LogisticRegression(C=0.1, max_iter=3000))])),
    ("LogReg C=1", lambda: Pipeline([("s", StandardScaler()), ("c", LogisticRegression(C=1.0, max_iter=3000))])),
    ("LogReg C=10", lambda: Pipeline([("s", StandardScaler()), ("c", LogisticRegression(C=10.0, max_iter=3000))])),
    ("SVM-linear C=1", lambda: Pipeline([("s", StandardScaler()), ("c", SVC(kernel="linear", C=1.0))])),
    ("SVM-RBF C=1", lambda: Pipeline([("s", StandardScaler()), ("c", SVC(kernel="rbf", C=1.0, gamma="scale"))])),
    ("SVM-RBF C=10", lambda: Pipeline([("s", StandardScaler()), ("c", SVC(kernel="rbf", C=10.0, gamma="scale"))])),
]

for clf_name, make_clf in classifiers:
    clf_accs = []
    route_accs = []
    route_costs = []

    for train_idx, test_idx in sss.split(embeddings, oracle_labels):
        X_train, X_test = embeddings[train_idx], embeddings[test_idx]
        y_train, y_test = oracle_labels[train_idx], oracle_labels[test_idx]

        clf = make_clf()
        clf.fit(X_train, y_train)

        # Predict on FULL dataset (not just test)
        y_pred_full = clf.predict(embeddings)

        # Classifier accuracy on test set
        y_pred_test = y_pred_full[test_idx]
        clf_acc = (y_pred_test == y_test).mean()
        clf_accs.append(clf_acc)

        # Routing accuracy on full set
        ra, rc = routing_accuracy(y_pred_full, acc_by_model)
        route_accs.append(ra)
        route_costs.append(rc)

    mean_clf = np.mean(clf_accs)
    mean_ra = np.mean(route_accs)
    mean_rc = np.mean(route_costs)
    mean_arena = arena_score(mean_ra, mean_rc)
    print(f"{clf_name:<35} {mean_clf*100:>7.2f}% {mean_ra*100:>9.2f}% {f'${mean_rc:.4f}':>10} {mean_arena:>7.2f}")

print(f"{'='*75}")

# --- Per-category analysis with best global classifier ---
print(f"\n=== Per-category breakdown (LogReg C=0.1, 10% train, full eval) ===")
# Train one instance
train_idx, test_idx = next(sss.split(embeddings, oracle_labels))
clf = Pipeline([("s", StandardScaler()), ("c", LogisticRegression(C=0.1, max_iter=3000))])
clf.fit(embeddings[train_idx], oracle_labels[train_idx])
y_pred = clf.predict(embeddings)

print(f"\n{'Category':<15} {'N':>5} {'Clf Acc':>8} {'Route Acc':>10} {'235b Acc':>9} {'Oracle Acc':>10}")
print(f"{'-'*60}")
for cat_idx, cat_name in enumerate(CATEGORY_NAMES):
    mask = categories == cat_idx
    n_cat = mask.sum()
    # Classifier accuracy
    clf_acc = (y_pred[mask] == oracle_labels[mask]).mean()
    # Route accuracy
    route_acc = np.array([acc_by_model[MODELS_4[y_pred[i]]][i] for i in range(N) if mask[i]]).mean()
    # Always-235b
    a235 = acc_by_model["235b"][mask].mean()
    # Oracle
    oracle_cat = np.array([acc_by_model[MODELS_4[oracle_labels[i]]][i] for i in range(N) if mask[i]]).mean()
    print(f"{cat_name:<15} {n_cat:>5} {clf_acc*100:>7.2f}% {route_acc*100:>9.2f}% {a235*100:>8.2f}% {oracle_cat*100:>9.2f}%")

# Model selection distribution
print(f"\nModel selection distribution (predicted):")
pred_dist = Counter(y_pred)
for j, mn in enumerate(MODELS_4):
    cnt = pred_dist.get(j, 0)
    oracle_cnt = (oracle_labels == j).sum()
    print(f"  {mn:<20} pred={cnt:>5} ({cnt/N*100:.1f}%)  oracle={oracle_cnt:>5} ({oracle_cnt/N*100:.1f}%)")
