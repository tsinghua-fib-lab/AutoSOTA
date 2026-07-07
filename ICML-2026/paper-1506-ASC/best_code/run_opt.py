#!/usr/bin/env python3
"""Optimized experiment runner with independent reg and adam weight decay params."""
import torch, numpy as np, random, sys, os, json
sys.path.insert(0, '/repo')
from trainer import StrategicTrainer
from dataloader import SCPIDataset, create_dataloaders
from model import StrategicClassifierForWarmup, StrategicClassifierFiniteSet
from model_utils import HingeLoss, BasicStrategicHingeLoss, AmbiguousStrategicHingeLoss

def set_seed(seed):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def generate_and_label_points(n_points=1000, x_low=-6.0, x_high=4.0, y_low=-10.0, y_high=10.0,
                              shift=1.0, seed=101, cost_scaling=0.5):
    if seed is not None: np.random.seed(seed)
    W = np.asarray([[1, 0], [1, 1], [1, -1]]); b = np.asarray([-1, 2, 2])
    X = np.column_stack([np.random.uniform(x_low, x_high, size=n_points),
                         np.random.uniform(y_low, y_high, size=n_points)])
    margins = X @ W.T + b[None, :]
    cond_chosen = margins[:, 0] >= 0
    two_norm = (2.0 / cost_scaling) * np.linalg.norm(W[0])
    cond_intersection = np.all(margins >= -two_norm, axis=1)
    positive = cond_chosen | cond_intersection
    y = np.where(positive, 1, -1)
    X_moved = X.copy(); X_moved[positive, 0] += shift / 2; X_moved[~positive, 0] -= shift / 2
    return X_moved, y

def run_experiment(k=3, cost_scaling=0.5, n_points=1000, lr=0.006, epochs=500, seed=1,
                   adam_weight_decay=0.0, reg_classifier=0.001, reg_auxiliary=0.001,
                   dev=0.4, tau=0.15, data_seed=101):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}, k={k}, lr={lr}, epochs={epochs}, adam_wd={adam_weight_decay}, "
          f"reg_cls={reg_classifier}, reg_aux={reg_auxiliary}, dev={dev}, tau={tau}")

    X_np, y_np = generate_and_label_points(n_points=n_points, seed=data_seed, shift=1.0,
                                           cost_scaling=cost_scaling, x_high=4.0, x_low=-6.0,
                                           y_high=10.0, y_low=-10.0)
    X = torch.tensor(X_np, dtype=torch.float32); y = torch.tensor(y_np, dtype=torch.float32)
    dataset = SCPIDataset(X, y)
    dl_train, dl_val, dl_test = create_dataloaders(dataset, batch_size=n_points, test_ratio=0.4,
                                                    val_ratio=0.1, data_seed=100)

    if k <= 1:
        model = StrategicClassifierForWarmup(d=2, cost_scaling=cost_scaling)
        loss_fn = BasicStrategicHingeLoss(scale_loss=cost_scaling) if k == 1 else HingeLoss()
        write_metrics, no_val = (False, True)
    else:
        model = StrategicClassifierFiniteSet(d=2, num_classifiers=k, dev=dev, tau=tau,
                                             cost_scaling=cost_scaling)
        loss_fn = AmbiguousStrategicHingeLoss()
        write_metrics, no_val = (True, False)

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=adam_weight_decay)

    if k <= 1:
        trainer = StrategicTrainer(model=model, loss_fn=loss_fn, optimizer=opt,
                                   reg_classifier=reg_classifier, write_metrics=write_metrics,
                                   device=device)
        trainer.fit(dl_train=dl_train, dl_val=dl_val, num_epochs=epochs, no_val=no_val)
        results = trainer.predict(dl_test)
    else:
        trainer = StrategicTrainer(model=model, loss_fn=loss_fn, optimizer=opt,
                                   reg_classifier=reg_classifier, reg_auxiliary=reg_auxiliary,
                                   device=device)
        trainer.fit(dl_train=dl_train, dl_val=dl_val, num_epochs=epochs)
        results = trainer.predict(dl_test)

    acc = results.get('acc', results.get('accuracy', None))
    print(f"k={k} Test Accuracy: {acc}")
    
    # JSON summary for parsing
    summary = {k: float(v) if isinstance(v, (np.floating, np.integer, np.ndarray)) else v
               for k, v in results.items() if k != 'values_of_proj'}
    summary['accuracy_pct'] = float(acc * 100) if acc else None
    print(f"\nJSON_RESULT: {json.dumps(summary)}")
    
    final_acc_pct = float(acc * 100) if acc else 0.0
    print(f"\nFINAL_RESULT: k={k} accuracy={acc:.4f} ({final_acc_pct:.2f}%)")
    return results

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--k', type=int, default=3)
    p.add_argument('--cost-scaling', type=float, default=0.5)
    p.add_argument('--lr', type=float, default=0.006)
    p.add_argument('--epochs', type=int, default=500)
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--adam-weight-decay', type=float, default=0.0)
    p.add_argument('--reg-classifier', type=float, default=0.001)
    p.add_argument('--reg-auxiliary', type=float, default=0.001)
    p.add_argument('--dev', type=float, default=0.4)
    p.add_argument('--tau', type=float, default=0.15)
    args = p.parse_args()
    run_experiment(k=args.k, cost_scaling=args.cost_scaling, lr=args.lr, epochs=args.epochs,
                   seed=args.seed, adam_weight_decay=args.adam_weight_decay,
                   reg_classifier=args.reg_classifier, reg_auxiliary=args.reg_auxiliary,
                   dev=args.dev, tau=args.tau)
