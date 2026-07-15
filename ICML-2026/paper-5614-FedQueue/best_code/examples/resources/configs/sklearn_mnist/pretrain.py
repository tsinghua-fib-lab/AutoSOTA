"""
Pre-train an SGDClassifier on full MNIST and save a joblib checkpoint.

Usage (run from examples/):
    python resources/configs/sklearn_mnist/pretrain.py \
        --output checkpoints/sgd_mnist.joblib \
        --epochs 20
"""

import argparse
import os
import joblib
import numpy as np
import torchvision
import torchvision.transforms as transforms

from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score

parser = argparse.ArgumentParser()
parser.add_argument(
    "--output",
    type=str,
    default="./checkpoints/sgd_mnist.joblib",
    help="Where to save the checkpoint.",
)
parser.add_argument(
    "--epochs",
    type=int,
    default=20,
    help="Number of full passes over the training set.",
)
parser.add_argument("--loss", type=str, default="log_loss")
parser.add_argument("--alpha", type=float, default=1e-4)
parser.add_argument("--random_state", type=int, default=42)
args = parser.parse_args()

# ── load MNIST ────────────────────────────────────────────────────────────────
data_dir = "./datasets/RawData"
transform = transforms.ToTensor()

print("Loading MNIST...")
train_raw = torchvision.datasets.MNIST(
    data_dir, train=True, download=True, transform=transform
)
test_raw = torchvision.datasets.MNIST(
    data_dir, train=False, download=False, transform=transform
)


def to_numpy(dataset):
    X = np.stack([dataset[i][0].numpy().ravel() for i in range(len(dataset))])
    y = np.array([dataset[i][1] for i in range(len(dataset))])
    return X, y


X_train, y_train = to_numpy(train_raw)
X_test, y_test = to_numpy(test_raw)
print(f"  Train: {X_train.shape}  Test: {X_test.shape}")

# ── train ─────────────────────────────────────────────────────────────────────
classes = np.arange(10)
model = SGDClassifier(loss=args.loss, alpha=args.alpha, random_state=args.random_state)

print(f"\nTraining for {args.epochs} epoch(s)...")
for epoch in range(args.epochs):
    model.partial_fit(X_train, y_train, classes=classes)
    train_acc = accuracy_score(y_train, model.predict(X_train)) * 100
    test_acc = accuracy_score(y_test, model.predict(X_test)) * 100
    print(
        f"  Epoch {epoch + 1:>2}/{args.epochs}  train={train_acc:.1f}%  test={test_acc:.1f}%"
    )

# ── save ──────────────────────────────────────────────────────────────────────
os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
joblib.dump(model, args.output)
print(f"\nCheckpoint saved to {args.output}")
print(f"  coef_ shape:      {model.coef_.shape}")
print(f"  intercept_ shape: {model.intercept_.shape}")
