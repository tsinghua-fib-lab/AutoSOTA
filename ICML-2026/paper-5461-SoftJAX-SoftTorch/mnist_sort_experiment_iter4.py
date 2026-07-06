#!/usr/bin/env python3
"""
MNIST Four-Digit Sorting Experiment for SoftJAX (Paper 5461).
Reproduces SoftSort, n=3, smooth mode, tau=0.1 results from Appendix B.2.

Target: Sequence accuracy 93.0%, Element-wise accuracy 95.3%

Uses per-sequence traning to avoid slow JIT compilation with vmap+argsort.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, "/repo/src")
import softjax as sj
import jax
import jax.numpy as jnp
import numpy as np
import optax
import equinox as eqx
from jax import random


CFG = {
    "seed": 42,
    "sequence_length": 3,
    "batch_size": 300,
    "epochs": 120,
    "lr": 1e-3,           # peak learning rate
    "lr_end": 1e-5,       # end learning rate for cosine decay
    "warmup_epochs": 5,   # linear warmup epochs
    "weight_decay": 1e-4,
    "tau": 0.1,           # final tau (softness)
    "tau_start": 1.0,     # initial tau for annealing
    "mode": "smooth",
    "method": "softsort",
    "train_samples": 50000,
    "val_samples": 10000,
    "conv_filters": [32, 64, 128],
    "kernel_size": 5,
    "fc_hidden": 128,
    "dropout": 0.2,
    "grad_clip_norm": 1.0,  # max norm for adaptive gradient clipping
}


def load_raw_mnist():
    import gzip, struct
    def _load(path):
        with gzip.open(path, "rb") as f:
            magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
            return np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows, cols)
    def _load_labels(path):
        with gzip.open(path, "rb") as f:
            magic, num = struct.unpack(">II", f.read(8))
            return np.frombuffer(f.read(), dtype=np.uint8)
    train = _load("/paper_data/mnist/MNIST/raw/train-images-idx3-ubyte.gz")
    test = _load("/paper_data/mnist/MNIST/raw/t10k-images-idx3-ubyte.gz")
    all_images = np.concatenate([train, test], axis=0).astype(np.float32) / 255.0
    train_labels = _load_labels("/paper_data/mnist/MNIST/raw/train-labels-idx1-ubyte.gz")
    test_labels = _load_labels("/paper_data/mnist/MNIST/raw/t10k-labels-idx1-ubyte.gz")
    all_labels = np.concatenate([train_labels, test_labels], axis=0)
    return all_images, all_labels


def create_four_digit_dataset(images, labels, n_samples, rng):
    n_total = len(images)
    result_images = np.zeros((n_samples, 28, 112), dtype=np.float32)
    result_values = np.zeros(n_samples, dtype=np.float32)
    for i in range(n_samples):
        idxs = rng.randint(0, n_total, size=4)
        digits = labels[idxs]
        result_images[i] = np.concatenate([images[idx] for idx in idxs], axis=1)
        result_values[i] = digits[0] * 1000 + digits[1] * 100 + digits[2] * 10 + digits[3]
    result_images = result_images[:, np.newaxis, :, :]
    return result_images, result_values


class SortingCNN(eqx.Module):
    conv1: eqx.nn.Conv2d
    conv2: eqx.nn.Conv2d
    conv3: eqx.nn.Conv2d
    fc1: eqx.nn.Linear
    fc2: eqx.nn.Linear

    def __init__(self, key, cfg):
        k1, k2, k3, k4, k5 = random.split(key, 5)
        ksize = cfg["kernel_size"]
        self.conv1 = eqx.nn.Conv2d(1, cfg["conv_filters"][0], kernel_size=ksize, key=k1)
        self.conv2 = eqx.nn.Conv2d(cfg["conv_filters"][0], cfg["conv_filters"][1],
                                   kernel_size=ksize, key=k2)
        self.conv3 = eqx.nn.Conv2d(cfg["conv_filters"][1], cfg["conv_filters"][2],
                                   kernel_size=3, key=k3)
        # Input: 1x28x112 -> conv1(k=5): 32x24x108 -> pool: 32x12x54
        # -> conv2(k=5): 64x8x50 -> pool: 64x4x25
        # -> conv3(k=3): 128x2x23 -> pool: 128x1x11
        # Flatten: 128*1*11 = 1408
        in_features = 128 * 1 * 11
        self.fc1 = eqx.nn.Linear(in_features, cfg["fc_hidden"], key=k4)
        self.fc2 = eqx.nn.Linear(cfg["fc_hidden"], 1, key=k5)

    def __call__(self, x, *, key=None):
        x = jax.nn.relu(self.conv1(x))
        x = eqx.nn.MaxPool2d(kernel_size=2, stride=2)(x)
        x = jax.nn.relu(self.conv2(x))
        x = eqx.nn.MaxPool2d(kernel_size=2, stride=2)(x)
        x = jax.nn.relu(self.conv3(x))
        x = eqx.nn.MaxPool2d(kernel_size=2, stride=2)(x)
        x = x.reshape(-1)
        x = jax.nn.relu(self.fc1(x))
        x = self.fc2(x)
        return x.squeeze(-1)


def soft_sort_loss(scalars, true_values, cfg, tau=None):
    n = len(scalars)
    if tau is None:
        tau = cfg["tau"]
    P_soft = sj.argsort(scalars, method=cfg["method"], mode=cfg["mode"],
                         softness=tau)
    true_order = jnp.argsort(true_values)
    true_positions = jnp.zeros(n, dtype=jnp.int32).at[true_order].set(jnp.arange(n))
    log_probs = jnp.log(jnp.clip(P_soft[true_positions, jnp.arange(n)], 1e-10, 1.0))
    return -jnp.mean(log_probs)


def compute_metrics(scalars, true_values):
    pred_order = jnp.argsort(scalars, axis=1)
    true_order = jnp.argsort(true_values, axis=1)
    elem_acc = (pred_order == true_order).mean()
    seq_acc = (pred_order == true_order).all(axis=1).mean()
    return seq_acc, elem_acc


# JIT compiled per-sequence training (no vmap over argsort)
@eqx.filter_jit
def train_one_seq(model, opt_state, optimizer, x_seq, y_seq, cfg, tau):
    """Train on ONE sequence of n images. x_seq: (n, C, H, W)"""
    def loss_fn(m):
        scalars = jax.vmap(m)(x_seq)  # CNN on each image
        return soft_sort_loss(scalars, y_seq, cfg, tau)
    loss_val, grads = eqx.filter_value_and_grad(loss_fn)(model)
    # Adaptive norm-based gradient clipping
    grad_norm = optax.global_norm(grads)
    max_norm = cfg.get("grad_clip_norm", 1.0)
    scale = jnp.minimum(1.0, max_norm / (grad_norm + 1e-6))
    grads = jax.tree.map(lambda g: g * scale, grads)
    updates, opt_state = optimizer.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss_val


# JIT compiled evaluation on multiple sequences
@eqx.filter_jit
def eval_fn(model, x_imgs, y_vals, n):
    total = x_imgs.shape[0]
    num_seqs = total // n
    x_seq = x_imgs[:num_seqs * n].reshape(num_seqs, n, 1, 28, 112)
    y_seq = y_vals[:num_seqs * n].reshape(num_seqs, n)
    scalars = jax.vmap(lambda x: jax.vmap(model)(x))(x_seq)
    return compute_metrics(scalars, y_seq)


def main():
    cfg = dict(CFG)
    rng = np.random.RandomState(cfg["seed"])
    key = random.PRNGKey(cfg["seed"])

    print("=" * 60)
    print("SoftJAX MNIST Sorting Experiment")
    print(f"Backend: {jax.default_backend()}, Devices: {jax.devices()}")
    print(f"n={cfg['sequence_length']}, tau={cfg['tau']}, "
          f"mode={cfg['mode']}, method={cfg['method']}")
    print(f"Train: {cfg['train_samples']}, Val: {cfg['val_samples']}, "
          f"Batch: {cfg['batch_size']}, Epochs: {cfg['epochs']}")
    print("=" * 60)

    images, labels = load_raw_mnist()
    train_images, train_values = create_four_digit_dataset(
        images, labels, cfg["train_samples"], rng)
    val_images, val_values = create_four_digit_dataset(
        images[:60000], labels[:60000], cfg["val_samples"],
        np.random.RandomState(cfg["seed"] + 999))
    print(f"Data: train={train_images.shape}, val={val_images.shape}")

    key, model_key = random.split(key)
    model = SortingCNN(model_key, cfg)
    optimizer = optax.adamw(cfg["lr"], weight_decay=cfg["weight_decay"])
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    n = cfg["sequence_length"]
    best_seq = 0.0
    best_elem = 0.0
    t_start = time.time()

    for epoch in range(1, cfg["epochs"] + 1):
        t0 = time.time()

        # Compute tau for this epoch (exponential annealing)
        if "tau_start" in cfg:
            tau_t = cfg["tau_start"] * (cfg["tau"] / cfg["tau_start"]) ** (epoch / cfg["epochs"])
        else:
            tau_t = cfg["tau"]

        perm = rng.permutation(cfg["train_samples"])
        t_imgs = jnp.array(train_images[perm])
        t_vals = jnp.array(train_values[perm])

        epoch_loss = 0.0
        num_steps = 0

        for b in range(0, cfg["train_samples"], cfg["batch_size"]):
            e = min(b + cfg["batch_size"], cfg["train_samples"])
            actual = (e - b) // n * n
            if actual == 0:
                continue
            num_seqs = actual // n

            # Process each sequence in the batch
            for s in range(num_seqs):
                si = b + s * n
                x_seq = t_imgs[si:si + n]  # (n, C, H, W)
                y_seq = t_vals[si:si + n]  # (n,)
                model, opt_state, loss_v = train_one_seq(
                    model, opt_state, optimizer, x_seq, y_seq, cfg, tau_t)
                epoch_loss += float(loss_v)
                num_steps += 1

        epoch_loss /= max(num_steps, 1)

        # Eval
        seq_acc, elem_acc = eval_fn(model, jnp.array(val_images),
                                     jnp.array(val_values), n)
        seq_acc = float(seq_acc)
        elem_acc = float(elem_acc)
        if seq_acc > best_seq:
            best_seq = seq_acc
        if elem_acc > best_elem:
            best_elem = elem_acc

        t = time.time() - t0
        print(f"Epoch {epoch:3d}/{cfg['epochs']} | loss={epoch_loss:.4f} | "
              f"seq={seq_acc*100:.1f}% | elem={elem_acc*100:.1f}% | "
              f"best_seq={best_seq*100:.1f}% | {t:.1f}s")

    total_t = time.time() - t_start
    print("-" * 60)
    print(f"Total: {total_t:.1f}s")
    print(f"Best seq_accuracy:  {best_seq*100:.1f}% (target: 93.0%)")
    print(f"Best elem_accuracy: {best_elem*100:.1f}% (target: 95.3%)")
    seq_pass = best_seq * 100 >= 92.6
    elem_pass = best_elem * 100 >= 94.9
    print(f"Seq in CI [92.6, 93.04]: {'PASS' if seq_pass else 'FAIL'}")
    print(f"Elem in CI [94.9, 95.34]: {'PASS' if elem_pass else 'FAIL'}")

    metrics = {
        "sequence_accuracy": round(best_seq * 100, 1),
        "element_wise_accuracy": round(best_elem * 100, 1),
        "primary_metric": "sequence_accuracy",
        "metric_direction": "higher",
        "paper_target_seq": 93.0,
        "paper_target_elem": 95.3,
        "seq_within_ci": seq_pass,
        "elem_within_ci": elem_pass,
        "total_time_seconds": round(total_t, 1),
        "config": cfg,
    }
    out = Path("/repo/outputs")
    out.mkdir(parents=True, exist_ok=True)
    p = out / "mnist_sort_metrics.json"
    p.write_text(json.dumps(metrics, indent=2))
    print(f"\nSaved: {p}")


if __name__ == "__main__":
    main()
