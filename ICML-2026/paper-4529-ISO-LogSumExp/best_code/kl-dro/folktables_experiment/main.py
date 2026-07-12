import torch
import torch.nn.functional as F
import torch.optim as optim
import math
import multiprocessing as mp
from torch.utils.data import DataLoader, TensorDataset
import os
import time
from tqdm import tqdm
from collections import defaultdict
from pathlib import Path
import json

from utils import load_folk, LinearRegressionModel, train_linreg


def batch_logsumexp(preds: torch.Tensor, targets: torch.Tensor, lam=1.):
    errors = (preds - targets) ** 2
    L = errors.squeeze(1)
    with torch.no_grad():
        p = torch.softmax(L/lam, dim=0)
    loss = torch.sum(p * L)
    return loss


def softplus_approx(preds: torch.Tensor, targets: torch.Tensor, model, rho: float, lam=1.):
    errors = (preds - targets) ** 2
    L = errors.squeeze(1)
    exponent = (L - model.alpha) / lam + math.log(rho)
    loss = (lam / rho) * F.softplus(exponent).mean() + model.alpha
    return loss


def evaluate_test_metrics2(model, test_loader, transform_target=True):
    model.eval()

    group_mse = defaultdict(float)
    group_mae = defaultdict(float)
    group_smape = defaultdict(float)
    group_counts = defaultdict(int)

    total_mse = 0.
    total_mae = 0.
    total_smape = 0.
    total_samples = 0

    with torch.no_grad():
        for batch_X, y_true, groups in test_loader:
            y_pred = model(batch_X)

            if transform_target:
                y_true, y_pred = torch.sinh(y_true), torch.sinh(y_pred)

            sq_error = (y_pred - y_true) ** 2
            abs_error = torch.abs(y_pred - y_true)
            smape_error = abs_error / ((torch.abs(y_true) + torch.abs(y_pred)) / 2. + 1e-9)

            total_mse += torch.sum(sq_error).item()
            total_mae += torch.sum(abs_error).item()
            total_smape += torch.sum(smape_error).item()
            total_samples += y_true.size(0)

            unique_groups = torch.unique(groups)
            for g in unique_groups:
                mask = (groups == g)
                g_val = g.item()
                group_mse[g_val] += torch.sum(sq_error[mask]).item()
                group_mae[g_val] += torch.sum(abs_error[mask]).item()
                group_smape[g_val] += torch.sum(smape_error[mask]).item()
                group_counts[g_val] += torch.sum(mask).item()

    rmse = (total_mse / total_samples) ** 0.5
    mae = total_mae / total_samples
    smape = (total_smape / total_samples) * 100

    max_rmse = 0.
    max_mae = 0.
    max_smape = 0.

    for g in group_counts:
        g_rmse = (group_mse[g] / group_counts[g]) ** 0.5
        g_mae = group_mae[g] / group_counts[g]
        g_smape = (group_smape[g] / group_counts[g]) * 100

        max_rmse = max(max_rmse, g_rmse)
        max_mae = max(max_mae, g_mae)
        max_smape = max(max_smape, g_smape)

    return rmse, mae, smape, max_rmse, max_mae, max_smape


def train(train_dataset, batch_sz, lr, rho, seed, lam):
    start = time.time()
    torch.manual_seed(seed)
    train_loader = DataLoader(train_dataset, batch_size=batch_sz, shuffle=True)
    X, y = train_dataset.tensors

    model = LinearRegressionModel(input_dim=X.shape[1], with_alpha=rho is not None)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True)

    n_epochs = 30

    for epoch in range(1, n_epochs + 1):
        # Training loop
        model.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = batch_logsumexp(preds, y_batch, lam=lam) if rho is None \
                else softplus_approx(preds, y_batch, model, rho, lam=lam)
            if not torch.isfinite(loss):
                print(f'inf or nan, batch{batch_sz}_lr{lr:.0e}_rho{rho}_seed{seed}')
                return None
            loss.backward()
            optimizer.step()

    fname = f'lam{lam}_batch{batch_sz}_rho{rho}_lr{lr}_seed{seed}.pth'
    torch.save(model.state_dict(), "model_weights/" + fname)
    end = time.time()
    print(f'took {(end - start) / 60:.1f} minutes')


def main():
    train_data = torch.load('folk_train.pt')
    test_data = torch.load('folk_test.pt')

    X_train, y_train = train_data['X'], train_data['y']
    X_test, y_test = test_data['X'], test_data['y']

    if not os.path.exists('linreg_weights2.pt'):
        print('Training least squares LinReg for initialization')
        train_linreg(X_train, y_train)

    train_dataset = TensorDataset(X_train, y_train)

    params = [  # (rho, lam, batch_sz, lr)
        (None, .2, 10, 1e-6),
        (1e-1, .2, 10, 1e-6),
        (1e-3, .2, 10, 1e-6),
        (1e-5, .2, 10, 1e-7),

        (None, 1., 10, 1e-6),
        (1e-1, 1., 10, 1e-5),
        (1e-3, 1., 10, 1e-6),
        (1e-5, 1., 10, 1e-7),

        (None, 5., 10, 1e-6),
        (1e-1, 5., 10, 1e-5),
        (1e-3, 5., 10, 1e-6),
        (1e-5, 5., 10, 1e-7),

        (None, .2, 100, 1e-5),
        (1e-1, .2, 100, 1e-5),
        (1e-3, .2, 100, 1e-5),
        (1e-5, .2, 100, 1e-7),

        (None, 1., 100, 1e-5),
        (1e-1, 1., 100, 1e-5),
        (1e-3, 1., 100, 1e-5),
        (1e-5, 1., 100, 1e-7),

        (None, 5., 100, 1e-5),
        (1e-1, 5., 100, 1e-4),
        (1e-3, 5., 100, 1e-4),
        (1e-5, 5., 100, 1e-6),

        (None, .2, 1000, 1e-5),
        (1e-1, .2, 1000, 1e-4),
        (1e-3, .2, 1000, 1e-4),
        (1e-5, .2, 1000, 1e-6),

        (None, 1., 1000, 1e-4),
        (1e-1, 1., 1000, 1e-4),
        (1e-3, 1., 1000, 1e-4),
        (1e-5, 1., 1000, 1e-6),

        (None, 5., 1000, 1e-4),
        (1e-1, 5., 1000, 1e-4),
        (1e-3, 5., 1000, 1e-4),
        (1e-5, 5., 1000, 1e-6),
    ]

    seeds = list(range(5))
    tasks = [(train_dataset, batch_sz, lr, rho, seed, lam)
             for rho, lam, batch_sz, lr in params
             for seed in seeds]

    with mp.Pool(processes=3) as pool:
        pool.starmap(train, tasks)


def eval_models(model_weights_dir='model_weights'):
    test_data = torch.load('folk_test.pt')
    X_test, y_test = test_data['X'].cuda(), test_data['y'].cuda()
    g_test = test_data['g'].cuda()
    test_dataset = TensorDataset(X_test, y_test, g_test)
    test_loader = DataLoader(test_dataset, batch_size=20000, shuffle=False)

    model_without_alpha = LinearRegressionModel(input_dim=X_test.shape[1], with_alpha=False).cuda()
    model_with_alpha = LinearRegressionModel(input_dim=X_test.shape[1], with_alpha=True).cuda()

    jsonl_file = 'evaluation_results2.jsonl'

    completed_entries = set()
    if os.path.exists(jsonl_file):
        with open(jsonl_file, 'r') as f:
            for line in f:
                if line.strip():
                    entry = json.loads(line)
                    key = (entry['lam'], entry['batch_sz'], entry['rho'], entry['lr'], entry['seed'])
                    completed_entries.add(key)
        print(f"Found {len(completed_entries)} completed evaluations in {jsonl_file}")

    model_files = list(Path(model_weights_dir).glob('*.pth'))
    results = []

    for file in tqdm(model_files):
        filename = file.stem
        parts = filename.split('_')

        lam = float([p for p in parts if p.startswith('lam')][0][3:])
        batch_sz = int([p for p in parts if p.startswith('batch')][0][5:])

        rho_part = [p for p in parts if p.startswith('rho')][0]
        rho_str = rho_part[3:]  # Remove 'rho' prefix
        rho = None if rho_str == 'None' else float(rho_str)

        lr = float([p for p in parts if p.startswith('lr')][0][2:])
        seed = int([p for p in parts if p.startswith('seed')][0][4:])

        entry_key = (lam, batch_sz, rho, lr, seed)
        if entry_key in completed_entries:
            continue

        try:
            if rho is None:
                model = model_without_alpha
            else:
                model = model_with_alpha

            model.load_state_dict(torch.load(file, map_location='cuda'))

            rmse, mae, smape, w_rmse, w_mae, w_smape = evaluate_test_metrics2(model, test_loader)

            result = {
                'lam': lam,
                'batch_sz': batch_sz,
                'lr': lr,
                'seed': seed,
                'rho': rho,
                'rmse': rmse,
                'mae': mae,
                'smape': smape,
                'w_rmse': w_rmse,
                'w_mae': w_mae,
                'w_smape': w_smape,
            }
            results.append(result)

            with open(jsonl_file, 'a') as f:
                f.write(json.dumps(result) + '\n')

            completed_entries.add(entry_key)

        except Exception as e:
            print(f"  ERROR evaluating {file}: {e}")
            continue


if __name__ == "__main__":
    if not os.path.exists('folk_train.pt') or not os.path.exists('folk_test.pt'):
        load_folk()
    else:
        print("Preprocessed dataset found.")

    if not os.path.exists('model_weights'):
        os.mkdir('model_weights')

    main()
    eval_models()
