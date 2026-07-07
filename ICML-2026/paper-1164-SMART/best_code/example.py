"""SMART example."""

from __future__ import annotations

import torch

from metrics import accuracy, expected_calibration_error, negative_log_likelihood
from smart import SMART


def make_overconfident_logits(n_samples: int, n_classes: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    logits = 2.0 * torch.randn(n_samples, n_classes, generator=generator)
    label_probs = torch.softmax(logits / 2.5, dim=1)
    labels = torch.multinomial(label_probs, num_samples=1, replacement=True, generator=generator).squeeze(1)
    return logits, labels


def main() -> None:
    val_logits, val_labels = make_overconfident_logits(1024, 10, seed=0)
    test_logits, test_labels = make_overconfident_logits(2048, 10, seed=1)

    before_ece = expected_calibration_error(logits=test_logits, labels=test_labels, n_bins=15)
    before_nll = negative_log_likelihood(test_logits, test_labels)
    before_acc = accuracy(test_logits, test_labels)

    calibrator = SMART(epochs=300, lr=5e-3, hidden_dim=16, loss="smooth_soft_ece", n_bins=15, seed=1)
    calibrator.fit(val_logits, val_labels)
    calibrated_logits = calibrator.calibrate(test_logits, return_logits=True)

    after_ece = expected_calibration_error(logits=calibrated_logits, labels=test_labels, n_bins=15)
    after_nll = negative_log_likelihood(calibrated_logits, test_labels)
    after_acc = accuracy(calibrated_logits, test_labels)

    print(f"ECE before SMART: {before_ece:.4f}")
    print(f"ECE after  SMART: {after_ece:.4f}")
    print(f"NLL before SMART: {before_nll:.4f}")
    print(f"NLL after  SMART: {after_nll:.4f}")
    print(f"Acc before SMART: {before_acc:.4f}")
    print(f"Acc after  SMART: {after_acc:.4f}")


if __name__ == "__main__":
    main()
