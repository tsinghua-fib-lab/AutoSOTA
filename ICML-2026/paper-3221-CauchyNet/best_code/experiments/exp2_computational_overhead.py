"""
Experiment 2: Computational Overhead Table
==========================================
Wall-clock time, FLOPs (estimated), memory, parameter counts.
"""

import os
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error, mean_absolute_error

from shared import (
    CauchyNet1D, FNN, SIREN, Transformer, count_real_params,
    prepare_1d_data, cauchy_real_params, matched_fnn_hidden,
    HIDDEN_SIZE, BATCH_SIZE, LR, EPOCHS, WEIGHT_DECAY, IMAG_PENALTY
)


def benchmark_model(model_factory, X_train, Y_train, X_test, Y_test,
                    epochs=EPOCHS, n_runs=5, device='cpu'):
    results = {
        'times': [], 'mse': [], 'mae': [],
        'num_params': None, 'forward_time_ms': [], 'backward_time_ms': []
    }

    for run in range(n_runs):
        torch.manual_seed(42 + run)
        np.random.seed(42 + run)
        model = model_factory().to(device)

        if results['num_params'] is None:
            results['num_params'] = count_real_params(model)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

        train_ds = torch.utils.data.TensorDataset(X_train, Y_train)
        train_loader = torch.utils.data.DataLoader(
            train_ds, batch_size=BATCH_SIZE, shuffle=True
        )

        patience = 30
        wait = 0
        start = time.time()
        best_loss = float('inf')
        best_state = None
        best_epoch = 0
        actual_epochs = 0
        fwd_times, bwd_times = [], []

        for epoch in range(epochs):
            model.train()
            epoch_loss = 0.0
            n_batches = 0
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()

                t0 = time.perf_counter()
                out = model(xb)
                t1 = time.perf_counter()

                y_exp = yb.unsqueeze(-1)
                if isinstance(out, tuple):
                    y_real, y_imag = out
                    loss = criterion(y_real, y_exp) + IMAG_PENALTY * criterion(
                        y_imag, torch.zeros_like(y_imag))
                else:
                    loss = criterion(out, y_exp)

                t2 = time.perf_counter()
                loss.backward()
                t3 = time.perf_counter()

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1

                if epoch == epochs - 1 or (patience and wait >= patience - 1):
                    fwd_times.append((t1 - t0) * 1000)
                    bwd_times.append((t3 - t2) * 1000)

            scheduler.step()
            avg_loss = epoch_loss / n_batches
            actual_epochs = epoch + 1
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                best_epoch = epoch + 1
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    break

        elapsed = time.time() - start
        results['times'].append(elapsed)
        results['forward_time_ms'].append(np.mean(fwd_times) if fwd_times else 0)
        results['backward_time_ms'].append(np.mean(bwd_times) if bwd_times else 0)

        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            out = model(X_test.to(device))
            preds = out[0].cpu().numpy().flatten() if isinstance(out, tuple) else out.cpu().numpy().flatten()
        truths = Y_test.numpy().flatten()
        results['mse'].append(mean_squared_error(truths, preds))
        results['mae'].append(mean_absolute_error(truths, preds))

    return results


def near_singular(x):
    """Near-singular target: peak + smooth background."""
    return 1.0 / ((x - 0.3) ** 2 + 0.01) + np.sin(4 * x)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    n_runs = 5

    # Parameter-matched: CauchyNet h=64 -> 256 real params
    cauchy_h = HIDDEN_SIZE
    target_params = cauchy_real_params(cauchy_h)
    fnn_h = matched_fnn_hidden(target_params)
    print(f"CauchyNet h={cauchy_h} ({target_params} params) -> FNN/SIREN h={fnn_h}")

    models = {
        'CauchyNet': lambda: CauchyNet1D(cauchy_h),
        'FNN': lambda: FNN(1, fnn_h, 1),
        'SIREN': lambda: SIREN(1, fnn_h, 1),
        'Transformer': lambda: Transformer(1, HIDDEN_SIZE, 1),
    }

    # --- Target 1: Paper's 1D function (smooth) ---
    X_train, Y_train, _, _, X_test, Y_test, _, _ = prepare_1d_data()

    # --- Target 2: Near-singular ---
    from sklearn.preprocessing import MinMaxScaler
    x_tr_ns = np.linspace(-1, 1, 50).astype(np.float32)
    x_te_ns = np.linspace(-1, 1, 200).astype(np.float32)
    sc = MinMaxScaler()
    y_tr_ns = sc.fit_transform(near_singular(x_tr_ns).reshape(-1,1).astype(np.float32)).flatten().astype(np.float32)
    y_te_ns = sc.transform(near_singular(x_te_ns).reshape(-1,1).astype(np.float32)).flatten().astype(np.float32)
    X_tr_ns = torch.tensor(x_tr_ns).unsqueeze(-1)
    Y_tr_ns = torch.tensor(y_tr_ns)
    X_te_ns = torch.tensor(x_te_ns).unsqueeze(-1)
    Y_te_ns = torch.tensor(y_te_ns)

    targets = {
        "Paper's 1D (smooth)": (X_train, Y_train, X_test, Y_test),
        "Near-singular": (X_tr_ns, Y_tr_ns, X_te_ns, Y_te_ns),
    }

    out_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(out_dir, exist_ok=True)

    all_results = {}

    for tgt_name, (Xtr, Ytr, Xte, Yte) in targets.items():
        print(f"\n{'='*50}")
        print(f"Target: {tgt_name}")
        print('=' * 50)
        all_results[tgt_name] = {}

        for name, factory in models.items():
            print(f"  Benchmarking {name}...")
            r = benchmark_model(factory, Xtr, Ytr, Xte, Yte,
                               n_runs=n_runs, device=device)
            all_results[tgt_name][name] = r
            print(f"    Params: {r['num_params']}")
            print(f"    Time: {np.mean(r['times']):.3f}s  MSE: {np.mean(r['mse']):.6f}")
            print(f"    Fwd: {np.mean(r['forward_time_ms']):.3f}ms  Bwd: {np.mean(r['backward_time_ms']):.3f}ms")

    save_data = {}
    for tgt_name in all_results:
        save_data[tgt_name] = {}
        for name, r in all_results[tgt_name].items():
            save_data[tgt_name][name] = {
                'num_params': r['num_params'], 'times': r['times'],
                'mse': r['mse'], 'mae': r['mae'],
                'forward_time_ms': r['forward_time_ms'],
                'backward_time_ms': r['backward_time_ms'],
            }
    with open(os.path.join(out_dir, 'exp2_overhead_results.json'), 'w') as f:
        json.dump(save_data, f, indent=2)

    # LaTeX table with both targets
    tex = r"""\begin{table}[ht]\centering
\caption{Computational overhead. 5 runs, 50 data points, 200 epochs. Timing on smooth target.}
\label{tab:overhead}
\begin{tabular}{lcccccc}
\toprule
Model & \# Params & Fwd (ms) & Bwd (ms) & Total (s) & MSE (smooth) & MSE (near-sing.) \\
\midrule
"""
    smooth = all_results["Paper's 1D (smooth)"]
    ns = all_results["Near-singular"]
    for name in models:
        rs = smooth[name]
        rn = ns[name]
        tex += (f"{name} & {rs['num_params']} & "
                f"${np.mean(rs['forward_time_ms']):.3f}$ & "
                f"${np.mean(rs['backward_time_ms']):.3f}$ & "
                f"${np.mean(rs['times']):.2f}$ & "
                f"${np.mean(rs['mse']):.4f}$ & "
                f"${np.mean(rn['mse']):.4f}$ \\\\\n")
    tex += r"""\bottomrule
\end{tabular}
\end{table}"""
    print("\n" + tex)

    with open(os.path.join(out_dir, 'exp2_overhead_table.tex'), 'w') as f:
        f.write(tex)
    print(f"\nSaved to {out_dir}/")


if __name__ == '__main__':
    main()
