import torch
import numpy as np
import random
import sys
import os

# Add repo to path
sys.path.insert(0, '/repo')

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
    x_low=-3.0,
    x_high=3.0,
    y_low=-3.0,
    y_high=3.0,
    shift=0.1,
    seed=101,
    cost_scaling=1.0,
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
    b_chosen = b[0]

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


def run_experiment(k=3, cost_scaling=0.5, n_points=1000, lr=0.006, weight_decay=0.001,
                   epochs=500, seed=1, data_seed=101, device_str='cuda'):
    set_seed(seed)
    device = torch.device(device_str if torch.cuda.is_available() else 'cpu')
    print("Running experiment: k={}, cost_scaling={}, epochs={}, lr={}, device={}".format(
        k, cost_scaling, epochs, lr, device))

    # Generate data
    X_np, y_np = generate_and_label_points(
        n_points=n_points, seed=data_seed, shift=1.0, cost_scaling=cost_scaling,
        x_high=4.0, x_low=-6.0, y_high=10.0, y_low=-10.0
    )

    X = torch.tensor(X_np, dtype=torch.float32)
    y = torch.tensor(y_np, dtype=torch.float32)

    # Create dataloaders with 50:10:40 split
    dataset = SCPIDataset(X, y)
    dl_train, dl_val, dl_test = create_dataloaders(
        dataset, batch_size=n_points, test_ratio=0.4, val_ratio=0.1, data_seed=100
    )

    if k == 0:
        # Naive (non-strategic) model
        model = StrategicClassifierForWarmup(d=2, cost_scaling=cost_scaling)
        loss_fn = HingeLoss()
        write_metrics = False
        no_val = True
    elif k == 1:
        # Standard strategic (k=1)
        model = StrategicClassifierForWarmup(d=2, cost_scaling=cost_scaling)
        loss_fn = BasicStrategicHingeLoss(scale_loss=cost_scaling)
        write_metrics = False
        no_val = True
    else:
        # Ambiguous strategic with k classifiers
        model = StrategicClassifierFiniteSet(d=2, num_classifiers=k, dev=0.4, cost_scaling=cost_scaling)
        loss_fn = AmbiguousStrategicHingeLoss()
        write_metrics = True
        no_val = False

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    if k <= 1:
        trainer = StrategicTrainer(
            model=model,
            loss_fn=loss_fn,
            optimizer=opt,
            reg_classifier=weight_decay,
            write_metrics=write_metrics,
            device=device
        )
        trainer.fit(
            dl_train=dl_train,
            dl_val=dl_val,
            num_epochs=epochs,
            early_stopping=None,
            no_val=no_val
        )
        results = trainer.predict(dl_test)
    else:
        trainer = StrategicTrainer(
            model=model,
            loss_fn=loss_fn,
            optimizer=opt,
            reg_classifier=weight_decay,
            reg_auxiliary=weight_decay,
            device=device
        )
        metrics = trainer.fit(
            dl_train=dl_train,
            dl_val=dl_val,
            num_epochs=epochs,
            early_stopping=None
        )
        results = trainer.predict(dl_test)

    acc = results.get('acc', None)
    if acc is None:
        acc = results.get('accuracy', None)
    print("k={} Test Accuracy: {}".format(k, acc))
    print("All results: {}".format(results))

    # Also get the train/val metrics if available
    if hasattr(trainer, 'metrics') and trainer.metrics:
        train_accs = trainer.metrics.get('train_accuracy', trainer.metrics.get('train_acc', []))
        val_accs = trainer.metrics.get('val_accuracy', trainer.metrics.get('val_acc', []))
        if train_accs:
            print("Final train accuracy: {}".format(train_accs[-1]))
        if val_accs:
            print("Final val accuracy: {}".format(val_accs[-1]))

    return results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--k', type=int, default=3)
    parser.add_argument('--cost-scaling', type=float, default=0.5)
    parser.add_argument('--n-points', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=0.006)
    parser.add_argument('--weight-decay', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    results = run_experiment(
        k=args.k,
        cost_scaling=args.cost_scaling,
        n_points=args.n_points,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        seed=args.seed,
        device_str=args.device
    )

    acc = results.get('acc', results.get('accuracy', None))
    if acc is not None:
        acc_pct = acc * 100.0
        print("\nFINAL_RESULT: k={} accuracy={:.4f} ({:.2f}%)".format(args.k, acc, acc_pct))
