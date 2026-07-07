#!/usr/bin/env python3
"""Reproduction script for paper 1506: Ambiguous Strategic Classification.
Target: k=3 discrete ambiguity, separable synthetic data, strategic accuracy.
"""
import os, sys, json, random
import numpy as np
import torch
from datetime import datetime

sys.path.insert(0, "/repo")
from trainer import StrategicTrainer
from dataloader import SCPIDataset, create_dataloaders
from model import StrategicClassifierForWarmup, StrategicClassifierFiniteSet
from model_utils import HingeLoss, BasicStrategicHingeLoss, AmbiguousStrategicHingeLoss

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def generate_and_label_points(
    n_points=1000,
    x_low=-6.0,
    x_high=4.0,
    y_low=-10.0,
    y_high=10.0,
    shift=1.0,
    seed=101,
    cost_scaling=0.5,
    device="cpu",
    dtype=torch.float32
):
    if seed is not None:
        np.random.seed(seed)

    W = [[1, 0], [1, 1], [1, -1]]
    b = [-1, 2, 2]
    W = np.asarray(W)
    b = np.asarray(b)
    w_chosen = W[0]

    X = np.column_stack([
        np.random.uniform(x_low, x_high, size=n_points),
        np.random.uniform(y_low, y_high, size=n_points)
    ])
    margins = X @ W.T + b[None, :]
    cond_chosen = margins[:, 0] >= 0
    two_norm = (2.0 / cost_scaling) * np.linalg.norm(w_chosen)
    cond_intersection = np.all(margins >= -two_norm, axis=1)
    positive = cond_chosen | cond_intersection
    y = np.where(positive, 1, -1)

    X_moved = X.copy()
    X_moved[positive, 0] += shift / 2
    X_moved[~positive, 0] -= shift / 2

    return X_moved, y

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    cost_scaling = 0.5
    set_seed(1)

    # Generate data per paper Appendix B.1.1
    X, y = generate_and_label_points(
        n_points=1000, seed=101, shift=1.0, cost_scaling=cost_scaling,
        x_high=4.0, x_low=-6.0, y_high=10.0, y_low=-10.0
    )
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)

    # Create dataloaders: 50:10:40 split
    dataset = SCPIDataset(X_t, y_t)
    dl_train, dl_val, dl_test = create_dataloaders(
        dataset, batch_size=1000, test_ratio=0.4, val_ratio=0.1, data_seed=100
    )

    results = {}

    # k=3: Ambiguous strategic (paper method)
    print("\n=== Training k=3 Ambiguous Strategic Classifier ===")
    model_k3 = StrategicClassifierFiniteSet(
        d=2, num_classifiers=3, dev=0.4, cost_scaling=cost_scaling
    )
    loss = AmbiguousStrategicHingeLoss()
    opt = torch.optim.Adam(model_k3.parameters(), lr=0.006)
    trainer_k3 = StrategicTrainer(
        model=model_k3, loss_fn=loss, optimizer=opt,
        reg_classifier=0.001, reg_auxiliary=0.001, device=device,
    )
    trainer_k3.fit(dl_train=dl_train, dl_val=dl_val, num_epochs=500)
    test_result_k3 = trainer_k3.predict(dl_test)
    results["k=3"] = {k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                      for k, v in test_result_k3.items()}
    print(f"k=3 test results: {results['k=3']}")

    # k=2: Ambiguous strategic
    print("\n=== Training k=2 Ambiguous Strategic Classifier ===")
    set_seed(1)
    model_k2 = StrategicClassifierFiniteSet(
        d=2, num_classifiers=2, dev=0.4, cost_scaling=cost_scaling
    )
    loss_k2 = AmbiguousStrategicHingeLoss()
    opt_k2 = torch.optim.Adam(model_k2.parameters(), lr=0.006)
    trainer_k2 = StrategicTrainer(
        model=model_k2, loss_fn=loss_k2, optimizer=opt_k2,
        reg_classifier=0.001, reg_auxiliary=0.001, device=device,
    )
    trainer_k2.fit(dl_train=dl_train, dl_val=dl_val, num_epochs=500)
    test_result_k2 = trainer_k2.predict(dl_test)
    results["k=2"] = {k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                      for k, v in test_result_k2.items()}
    print(f"k=2 test results: {results['k=2']}")

    # k=1: Standard strategic hinge baseline
    print("\n=== Training k=1 Standard Strategic Classifier (baseline) ===")
    set_seed(1)
    model_k1 = StrategicClassifierForWarmup(d=2, cost_scaling=cost_scaling)
    loss_k1 = BasicStrategicHingeLoss(scale_loss=cost_scaling)
    opt_k1 = torch.optim.Adam(model_k1.parameters(), lr=0.006)
    trainer_k1 = StrategicTrainer(
        model=model_k1, loss_fn=loss_k1, optimizer=opt_k1,
        reg_classifier=0.001, write_metrics=False, device=device,
    )
    trainer_k1.fit(dl_train=dl_train, dl_val=dl_val, num_epochs=500, no_val=True)
    test_result_k1 = trainer_k1.predict(dl_test)
    results["k=1"] = {k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                      for k, v in test_result_k1.items()}
    print(f"k=1 test results: {results['k=1']}")

    # Naive hinge non-strategic
    print("\n=== Training Naive (non-strategic) Classifier ===")
    set_seed(1)
    model_naive = StrategicClassifierForWarmup(d=2, cost_scaling=cost_scaling)
    loss_naive = HingeLoss()
    opt_naive = torch.optim.Adam(model_naive.parameters(), lr=0.006)
    trainer_naive = StrategicTrainer(
        model=model_naive, loss_fn=loss_naive, optimizer=opt_naive,
        reg_classifier=0.001, write_metrics=False, device=device,
    )
    trainer_naive.fit(dl_train=dl_train, dl_val=dl_val, num_epochs=500, no_val=True)
    test_result_naive = trainer_naive.predict(dl_test)
    results["naive"] = {k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                        for k, v in test_result_naive.items()}
    print(f"naive test results: {results['naive']}")

    # Summary
    print("\n" + "=" * 60)
    print("REPRODUCTION RESULTS SUMMARY")
    print("=" * 60)
    for name, res in results.items():
        acc = res.get("acc", -1) * 100
        print(f"  {name:8s}: Strategic Accuracy = {acc:.2f}%")

    print("\nPaper reports: k=3 -> 99.2%, k=2 -> 94.5%, k=1 (strategic baseline) -> 88.4%, naive -> 62.9%")

    # Write results to file for parsing
    output = {
        "paper_id": 1506,
        "timestamp": datetime.now().isoformat(),
        "results": results,
        "rubric_target": {"k": 3, "paper_value": 99.2, "ci_lower": 88.4, "ci_upper": 100.28},
    }
    with open("/repo/reproduction_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print("\nResults saved to /repo/reproduction_results.json")

    return results

if __name__ == "__main__":
    main()
