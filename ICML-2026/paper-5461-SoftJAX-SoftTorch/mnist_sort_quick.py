#!/usr/bin/env python3
"""
MNIST Four-Digit Sorting Experiment for SoftJAX (Paper 5461).
Quick validation version: small dataset, few epochs.
"""
from __future__ import annotations

import json
import os
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
    "batch_size": 96,  # Must be divisible by n=3
    "epochs": 5,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "tau": 0.1,
    "mode": "smooth",
    "method": "softsort",
    "train_samples": 2400,
    "val_samples": 600,
    "conv_filters": [32, 64],
    "kernel_size": 5,
    "fc_hidden": 64,
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
    fc1: eqx.nn.Linear
    fc2: eqx.nn.Linear

    def __init__(self, key, cfg):
        k1, k2, k3, k4 = random.split(key, 4)
        ksize = cfg["kernel_size"]
        self.conv1 = eqx.nn.Conv2d(1, cfg["conv_filters"][0], kernel_size=ksize, key=k1)
        self.conv2 = eqx.nn.Conv2d(cfg["conv_filters"][0], cfg["conv_filters"][1],
                                   kernel_size=ksize, key=k2)
        in_features = cfg["conv_filters"][1] * 4 * 25
        self.fc1 = eqx.nn.Linear(in_features, cfg["fc_hidden"], key=k3)
        self.fc2 = eqx.nn.Linear(cfg["fc_hidden"], 1, key=k4)

    def __call__(self, x, *, key=None):
        x = jax.nn.relu(self.conv1(x))
        x = eqx.nn.MaxPool2d(kernel_size=2, stride=2)(x)
        x = jax.nn.relu(self.conv2(x))
        x = eqx.nn.MaxPool2d(kernel_size=2, stride=2)(x)
        x = x.reshape(-1)
        x = jax.nn.relu(self.fc1(x))
        x = self.fc2(x)
        return x.squeeze(-1)


def soft_sort_loss(scalars, true_values, cfg):
    n = len(scalars)
    P_soft = sj.argsort(scalars, method=cfg["method"], mode=cfg["mode"],
                         softness=cfg["tau"])
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


# Batched training: process multiple sequences at once
@eqx.filter_jit
def make_train_step(model, opt_state, x_seqs, y_seqs, cfg):
    """x_seqs: (num_seqs, n, C, H, W), y_seqs: (num_seqs, n)"""
    def loss_fn(m):
        # vmap over sequences: each sequence maps n images → n scalars
        scalars_per_seq = jax.vmap(lambda seq: jax.vmap(m)(seq))(x_seqs)
        # vmap over sequences for loss
        losses = jax.vmap(lambda s, v: soft_sort_loss(s, v, cfg))(scalars_per_seq, y_seqs)
        return jnp.mean(losses)

    loss_val, grads = eqx.filter_value_and_grad(loss_fn)(model)
    grads = jax.tree.map(lambda g: jnp.clip(g, -1.0, 1.0), grads)
    updates, opt_state = optimizer.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss_val


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
    print("SoftJAX MNIST Sorting - Quick Validation")
    print(f"Backend: {jax.default_backend()}, Devices: {jax.devices()}")
    print("=" * 60)

    images, labels = load_raw_mnist()
    train_images, train_values = create_four_digit_dataset(images, labels, cfg["train_samples"], rng)
    val_images, val_values = create_four_digit_dataset(
        images[:60000], labels[:60000], cfg["val_samples"],
        np.random.RandomState(cfg["seed"] + 999))
    print(f"Train: {train_images.shape}, Val: {val_images.shape}")

    key, model_key = random.split(key)
    model = SortingCNN(model_key, cfg)

    global optimizer
    optimizer = optax.adamw(cfg["lr"], weight_decay=cfg["weight_decay"])
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    n = cfg["sequence_length"]
    batch_size = cfg["batch_size"]

    print(f"\nEpochs: {cfg['epochs']}, n={n}, batch_size={batch_size}")
    print("-" * 60)

    for epoch in range(1, cfg["epochs"] + 1):
        t0 = time.time()
        perm = rng.permutation(cfg["train_samples"])
        t_imgs = jnp.array(train_images[perm])
        t_vals = jnp.array(train_values[perm])

        epoch_loss = 0.0
        num_steps = 0

        for b in range(0, cfg["train_samples"], batch_size):
            e = min(b + batch_size, cfg["train_samples"])
            actual = (e - b) // n * n
            if actual == 0:
                continue
            num_seqs = actual // n
            x_batch = t_imgs[b:b + actual].reshape(num_seqs, n, 1, 28, 112)
            y_batch = t_vals[b:b + actual].reshape(num_seqs, n)

            key, tkey = random.split(key)
            model, opt_state, loss_v = make_train_step(model, opt_state, x_batch, y_batch, cfg)
            epoch_loss += float(loss_v)
            num_steps += 1

        epoch_loss /= max(num_steps, 1)

        key, ekey = random.split(key)
        seq_acc, elem_acc = eval_fn(model, jnp.array(val_images), jnp.array(val_values), n)

        t = time.time() - t0
        print(f"Epoch {epoch} | loss={epoch_loss:.4f} | seq={float(seq_acc)*100:.1f}% | "
              f"elem={float(elem_acc)*100:.1f}% | {t:.1f}s")

    print(f"\nTarget: seq=93.0%, elem=95.3% (this is a quick validation, not full reproduction)")
    print("Done!")


if __name__ == "__main__":
    main()
