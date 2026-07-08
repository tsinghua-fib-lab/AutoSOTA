"""Evaluator for CIFAR-10 with fixed architecture and LR/WD tuning."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Union

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from bo_framework.base.evaluator import BaseEvaluator
from bo_framework.base.evaluation_result import EvaluationResult


def _build_resnet18_for_cifar() -> nn.Module:
    """Create a ResNet-18 adjusted for 32×32 CIFAR inputs."""
    model = torchvision.models.resnet18(weights=None)

    # Adapt first conv layer to CIFAR resolution (kernel 3, stride 1) and remove max pool
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, 10)

    return model


class HPTCIFARLearningRateWeightDecayEvaluator(BaseEvaluator):
    """Train ResNet-18 on CIFAR-10, tuning only learning rate and weight decay."""

    def __init__(
        self,
        max_epochs: int = 5,
        batch_size: int = 128,
        device: str = "cuda",
        data_dir: str = "./data",
        penalty_value: float = 0.0,
    ) -> None:
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.penalty_value = penalty_value
        self.data_dir = Path(data_dir)

        self._train_set, self._val_set = self._load_datasets()

    def _load_datasets(self):
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        )

        cifar_root = self.data_dir
        cifar_path = cifar_root / "cifar-10-batches-py"
        download = not cifar_path.exists()

        train_set = torchvision.datasets.CIFAR10(
            root=cifar_root, train=True, download=download, transform=transform
        )
        val_set = torchvision.datasets.CIFAR10(
            root=cifar_root, train=False, download=download, transform=transform
        )

        return train_set, val_set

    def evaluate(self, params: Union[Dict[str, Any], torch.Tensor]) -> EvaluationResult:
        if isinstance(params, torch.Tensor):
            raise ValueError("Evaluator expects a parameter dictionary, not a tensor")

        learning_rate = float(params["learning_rate"])
        weight_decay = float(params["weight_decay"])

        train_loader = DataLoader(
            self._train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
        )
        val_loader = DataLoader(
            self._val_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        )

        model = _build_resnet18_for_cifar().to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

        try:
            start = time.time()
            model.train()
            for epoch in range(self.max_epochs):
                for images, targets in train_loader:
                    images = images.to(self.device)
                    targets = targets.to(self.device)

                    optimizer.zero_grad()
                    outputs = model(images)
                    loss = criterion(outputs, targets)

                    if torch.isnan(loss) or torch.isinf(loss) or loss.item() > 1000:
                        raise RuntimeError("Training diverged")

                    loss.backward()
                    optimizer.step()

                scheduler.step()

            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for images, targets in val_loader:
                    images = images.to(self.device)
                    targets = targets.to(self.device)
                    outputs = model(images)
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()

            accuracy = correct / total
            print(
                f"LR={learning_rate:.2e}, WD={weight_decay:.2e}: "
                f"accuracy={accuracy:.4f}, time={time.time() - start:.1f}s"
            )
            return EvaluationResult.from_true_value(params, accuracy)

        except Exception as exc:
            print(f"Evaluation failed for params {params}: {exc}")
            return EvaluationResult.from_true_value(params, self.penalty_value)
