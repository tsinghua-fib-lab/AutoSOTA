"""Quickstart: train a TT-Sparse model and extract rules."""

import numpy as np
import torch
from sklearn.datasets import fetch_openml
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from tt_sparse import TabularEncoder, TTSparseModel, train, prune, explain, predict_rules

torch.manual_seed(0)
np.random.seed(0)

# Load Pima Indians Diabetes (OpenML ID 37)
dataset = fetch_openml(data_id=37, as_frame=True)
df = dataset.frame
df = df.rename(columns={"class": "target"})

# Train/test split
df_train, df_test = train_test_split(df, test_size=0.2, random_state=0)

# Encode continuous features into binary via thermometer encoding
encoder = TabularEncoder(target="target", task_type="binary", num_bits=9)
train_data = encoder.fit_transform(df_train)

# Build model
model = TTSparseModel(
    input_size=encoder.n_ltt_features,
    num_nodes=30,
    num_classes=1,
    n_bits=5,
    tau=0.01,
    task_type="binary",
    skip_size=encoder.n_skip_features,
)

# Train
train(model, train_data, epochs=100, device="cpu", seed=0, verbose=True)

# Prune
prune(model, train_data, max_drop_pct=2.0, finetune_epochs=30, device="cpu", seed=0, verbose=True)

# Extract human-readable rules
rules = explain(model, encoder)
print(f"\nExtracted {len(rules.rules)} rules (complexity: {rules.complexity}):\n")
for rule in rules.rules:
    print(f"  {rule['r']}")
    print(f"    w={rule['w']}\n")

# Evaluate on held-out test set
test_data = encoder.transform(df_test)
X_test = torch.tensor(test_data["X_ltt"])
X_skip_test = torch.tensor(test_data["X_skip"])
y_test = test_data["y"]

# Model inference
model.eval()
with torch.no_grad():
    model_preds = model(X_test, X_skip_test).numpy().ravel()
model_auc = roc_auc_score(y_test, model_preds)

# Rule inference (no neural network needed)
rule_preds = predict_rules(rules, df_test.drop(columns=["target"]), encoder=encoder)
rule_auc = roc_auc_score(y_test, rule_preds)

print(f"Model AUC: {model_auc:.4f}")
print(f"Rule AUC:  {rule_auc:.4f}")
assert np.isclose(model_auc, rule_auc, atol=1e-6), "Model and rule predictions should match"
