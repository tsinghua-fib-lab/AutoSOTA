"""
Experiment 11: Untrained pole locations (fixed on an ellipse).

Tests whether the structural inductive bias of CauchyNet — pole atoms on a
contour — is enough on its own. We freeze xi_k = r_re*cos(2*pi*k/h) + i*r_im*sin(2*pi*k/h)
for k=0,...,h-1 (uniformly on an ellipse symmetric about the real axis) and train
ONLY the complex output coefficients lambda. The result is a Cauchy-kernel
least-squares fit; pole locations come from the proof of the universal-approximation
theorem rather than from optimization.

Compared at:
  - matched hidden size h=64 against vanilla CauchyNet (trainable xi) and FNN
  - matched trainable-parameter count

Two targets:
  - Paper's 1D target (smooth-ish mixed function)
  - Near-singular: 1/((x-0.3)^2 + 0.01) + sin(4x)
"""
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from shared import (
    HIDDEN_SIZE,
    BATCH_SIZE,
    LR,
    EPOCHS,
    IMAG_PENALTY,
    WEIGHT_DECAY,
    ReciprocalActivation,
    CauchyNet1D,
    FNN,
    train_and_eval,
    target_function,
)


class FixedEllipseCauchyNet1D(nn.Module):
    """1D CauchyNet with fixed pole locations on an ellipse.

    Pole k is xi_k = r_re*cos(2*pi*k/h) + i*r_im*sin(2*pi*k/h).
    For inputs x in [-1, 1], we need r_re > 1 so the contour strictly
    encloses the real input domain. Only lambda (theta) trains.
    """

    def __init__(self, hidden_size, output_size=1, r_re=1.5, r_im=0.5, eps=1e-8):
        super().__init__()
        assert r_re > 1.0, (
            f"r_re={r_re} must exceed 1 so the ellipse encloses input domain [-1,1]"
        )
        self.lambda_ = nn.Parameter(
            torch.normal(
                mean=0.0,
                std=0.1,
                size=(hidden_size, output_size),
                dtype=torch.cfloat,
            )
        )
        k = torch.arange(hidden_size, dtype=torch.float32)
        angles = 2 * np.pi * k / hidden_size
        xi_real = r_re * torch.cos(angles)
        xi_imag = r_im * torch.sin(angles)
        xi = torch.complex(xi_real, xi_imag)
        self.register_buffer("xi", xi)
        self.activation = ReciprocalActivation(eps=eps)
        self.input_size = 1
        self.r_re = r_re
        self.r_im = r_im

    def forward(self, x):
        xc = torch.complex(x, torch.zeros_like(x)).unsqueeze(-1)
        xc = xc.view(x.size(0), 1, -1)
        activated = self.activation(self.xi.view(-1, 1) - xc)
        activated_flat = activated.view(x.size(0), -1)
        y = torch.matmul(activated_flat, self.lambda_)
        return (
            y.real.squeeze(-1).unsqueeze(-1),
            y.imag.squeeze(-1).unsqueeze(-1),
        )


def near_singular_target(x):
    return 1.0 / ((x - 0.3) ** 2 + 0.01) + np.sin(4 * x)


def prepare_target_data(target_fn, n_points=50, seed=0):
    from sklearn.preprocessing import MinMaxScaler

    x_np = np.linspace(-1, 1, n_points).astype(np.float32)
    y_np = target_fn(x_np).astype(np.float32)
    scaler_y = MinMaxScaler()
    y_scaled = scaler_y.fit_transform(y_np.reshape(-1, 1)).flatten().astype(np.float32)

    indices = np.arange(n_points)
    rng = np.random.RandomState(seed)
    rng.shuffle(indices)
    n_train = int(0.5 * n_points)
    n_val = int(0.25 * n_points)

    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val :]

    X_train = torch.tensor(x_np[train_idx]).unsqueeze(-1)
    Y_train = torch.tensor(y_scaled[train_idx])
    X_val = torch.tensor(x_np[val_idx]).unsqueeze(-1)
    Y_val = torch.tensor(y_scaled[val_idx])
    X_test = torch.tensor(x_np[test_idx]).unsqueeze(-1)
    Y_test = torch.tensor(y_scaled[test_idx])
    return X_train, Y_train, X_val, Y_val, X_test, Y_test


def run_one(model_factory, target_fn, seeds, n_points=50):
    mses, maes, params = [], [], []
    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        X_train, Y_train, X_val, Y_val, X_test, Y_test = prepare_target_data(
            target_fn, n_points=n_points, seed=seed
        )
        model = model_factory()
        result = train_and_eval(
            model,
            X_train,
            Y_train,
            X_test,
            Y_test,
            epochs=EPOCHS,
            X_val=X_val,
            Y_val=Y_val,
            patience=30,
        )
        mses.append(result["mse"])
        maes.append(result["mae"])
        params.append(result["num_params"])
    return {
        "mse": mses,
        "mae": maes,
        "mse_mean": float(np.mean(mses)),
        "mse_std": float(np.std(mses)),
        "num_params": int(np.median(params)),
    }


def main():
    seeds = list(range(10))
    targets = {
        "Paper 1D": target_function,
        "Near-singular": near_singular_target,
    }

    h_default = HIDDEN_SIZE
    h_fixed_matched = 128

    results = {}
    for tname, tfn in targets.items():
        print(f"\n== Target: {tname} ==")
        cell = {}

        print(f"  Vanilla CauchyNet h={h_default}...")
        cell["CauchyNet (trainable xi)"] = run_one(
            lambda h=h_default: CauchyNet1D(hidden_size=h, output_size=1),
            tfn,
            seeds,
        )

        print(f"  Fixed-ellipse CauchyNet h={h_default}...")
        cell["FixedEllipse CN (h=64)"] = run_one(
            lambda h=h_default: FixedEllipseCauchyNet1D(hidden_size=h, output_size=1),
            tfn,
            seeds,
        )

        print(f"  Fixed-ellipse CauchyNet h={h_fixed_matched}...")
        cell[f"FixedEllipse CN (h={h_fixed_matched})"] = run_one(
            lambda h=h_fixed_matched: FixedEllipseCauchyNet1D(
                hidden_size=h, output_size=1
            ),
            tfn,
            seeds,
        )

        print(f"  FNN matched...")
        cell["FNN (param-matched)"] = run_one(
            lambda: FNN(input_size=1, hidden_size=85, output_size=1),
            tfn,
            seeds,
        )

        for name, r in cell.items():
            print(
                f"    {name}: MSE={r['mse_mean']:.5f} +- {r['mse_std']:.5f}  "
                f"params={r['num_params']}"
            )
        results[tname] = cell

    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / "exp11_fixed_circle_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved results/exp11_fixed_circle_results.json")


if __name__ == "__main__":
    main()
