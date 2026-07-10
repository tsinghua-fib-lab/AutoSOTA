from pathlib import Path

import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

from clari_tree import CLARITree, Greedy


def load_split(path: Path):
    data = pd.read_csv(path).to_numpy(dtype=float)
    return data[:, :-1], data[:, -1]


split_dir = Path("data/airfoil/splits/outer_0")
X_train, y_train = load_split(split_dir / "train.csv")
X_test, y_test = load_split(split_dir / "test.csv")


def evaluate_model(name, model):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(f"\n===== {name} =====")
    print(f"Test MSE: {mean_squared_error(y_test, y_pred):.4f}")
    print(f"Test R^2: {r2_score(y_test, y_pred):.4f}")
    model.print_tree()


# CLARITree
claritree = CLARITree(
    depth=4,
    lambda_=0.001,
    kappa=0.001,
    n_thresholds=20,
    thresholds_strategy="quantile",
    min_leaf_node_size=0,
    verbose=False,
)

# Greedy
greedy = Greedy(
    depth=4,
    lambda_=0.001,
    kappa=0.001,
    n_thresholds=20,
    thresholds_strategy="quantile",
    min_leaf_node_size=0,
    verbose=False,
)

evaluate_model("CLARITree", claritree)
evaluate_model("Greedy", greedy)