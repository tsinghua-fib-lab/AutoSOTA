from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


PRIMARY_METRIC = "test_accuracy"


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def make_images(n: int, image_size: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    images = torch.zeros(n, 1, image_size, image_size)
    labels = torch.arange(n) % 3
    for i in range(n):
        label = int(labels[i].item())
        img = torch.zeros(image_size, image_size)
        offset = int(torch.randint(-2, 3, (1,), generator=generator).item())
        if label == 0:
            col = max(2, min(image_size - 3, image_size // 2 + offset))
            img[:, col - 1 : col + 2] = 1.0
        elif label == 1:
            row = max(2, min(image_size - 3, image_size // 2 + offset))
            img[row - 1 : row + 2, :] = 1.0
        else:
            for r in range(2, image_size - 2):
                c = max(1, min(image_size - 2, r + offset))
                img[r - 1 : r + 2, c - 1 : c + 2] = 1.0
        img += 0.32 * torch.randn(image_size, image_size, generator=generator)
        images[i, 0] = img.clamp(0.0, 1.0)
    perm = torch.randperm(n, generator=generator)
    return images[perm].float(), labels[perm].long()


class TinyCNN(nn.Module):
    def __init__(self, channels: int, dropout: float, image_size: int):
        super().__init__()
        pooled = image_size // 2
        self.net = nn.Sequential(
            nn.Conv2d(1, channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels * 2, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(dropout),
            nn.Conv2d(channels * 2, channels * 2, 3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(channels * 2 * pooled * pooled, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            pred = model(x).argmax(dim=1)
            correct += int((pred == y).sum().item())
            total += int(y.numel())
    return correct / max(1, total)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/current.json")
    parser.add_argument("--output", default="outputs/metrics.json")
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    seed = int(cfg.get("seed", 9002))
    random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(int(cfg.get("torch_num_threads", 2)))
    started = time.time()

    cuda_available = torch.cuda.is_available()
    gpu_count = torch.cuda.device_count() if cuda_available else 0
    device = torch.device("cuda:0" if cuda_available else "cpu")
    gpu_names = [torch.cuda.get_device_name(i) for i in range(gpu_count)] if cuda_available else []
    print(f"cuda_available={cuda_available} gpu_count={gpu_count} gpu_names={gpu_names}")

    train_x, train_y = make_images(int(cfg["train_size"]), int(cfg["image_size"]), seed)
    test_x, test_y = make_images(int(cfg["test_size"]), int(cfg["image_size"]), seed + 101)
    train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=int(cfg["batch_size"]), shuffle=True)
    test_loader = DataLoader(TensorDataset(test_x, test_y), batch_size=int(cfg["batch_size"]), shuffle=False)

    model: nn.Module = TinyCNN(
        channels=int(cfg["channels"]),
        dropout=float(cfg["dropout"]),
        image_size=int(cfg["image_size"]),
    )
    data_parallel_used = bool(cuda_available and gpu_count >= 2 and cfg.get("use_data_parallel", True))
    if data_parallel_used:
        model = nn.DataParallel(model)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["lr"]), weight_decay=float(cfg["weight_decay"]))
    criterion = nn.CrossEntropyLoss()
    last_loss = 0.0
    noise = float(cfg.get("augmentation_noise", 0.0))
    for _ in range(int(cfg["epochs"])):
        model.train()
        for x, y in train_loader:
            if noise > 0:
                x = (x + noise * torch.randn_like(x)).clamp(0.0, 1.0)
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            last_loss = float(loss.detach().cpu().item())

    accuracy = evaluate(model, test_loader, device)
    metrics = {
        PRIMARY_METRIC: round(float(accuracy), 6),
        "primary_metric": PRIMARY_METRIC,
        "metric_direction": "higher",
        "train_loss": round(last_loss, 6),
        "cuda_used": int(cuda_available),
        "gpu_visible_count": int(gpu_count),
        "data_parallel_used": int(data_parallel_used),
        "elapsed_seconds": round(time.time() - started, 3),
        "config": cfg,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
