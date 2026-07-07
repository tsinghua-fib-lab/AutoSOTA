from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


PRIMARY_METRIC = "sequence_accuracy"


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def make_sequences(n: int, length: int, vocab: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    x = torch.randint(0, vocab, (n, length), generator=generator)
    count_3 = (x == 3).sum(dim=1)
    count_7 = (x == 7).sum(dim=1)
    motif = ((x[:, :-1] == 4) & (x[:, 1:] == 9)).sum(dim=1)
    early = (x[:, : length // 2] == 11).sum(dim=1)
    late = (x[:, length // 2 :] == 2).sum(dim=1)
    score = 1.3 * count_3 + 1.1 * count_7 + 2.4 * motif + 0.7 * early - 0.8 * late
    threshold = score.float().median()
    y = (score.float() > threshold).long()
    perm = torch.randperm(n, generator=generator)
    return x[perm].long(), y[perm].long()


class TinySequenceClassifier(nn.Module):
    def __init__(self, vocab: int, embedding_dim: int, hidden_dim: int, bidirectional: bool, dropout: float):
        super().__init__()
        self.embedding = nn.Embedding(vocab, embedding_dim)
        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=bidirectional,
        )
        width = hidden_dim * (2 if bidirectional else 1)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(width, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(x)
        _, h = self.gru(emb)
        if self.gru.bidirectional:
            pooled = torch.cat([h[-2], h[-1]], dim=1)
        else:
            pooled = h[-1]
        return self.head(pooled)


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
    seed = int(cfg.get("seed", 9003))
    random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(int(cfg.get("torch_num_threads", 2)))
    started = time.time()

    cuda_available = torch.cuda.is_available()
    gpu_count = torch.cuda.device_count() if cuda_available else 0
    device = torch.device("cuda:0" if cuda_available else "cpu")
    gpu_names = [torch.cuda.get_device_name(i) for i in range(gpu_count)] if cuda_available else []
    print(f"cuda_available={cuda_available} gpu_count={gpu_count} gpu_names={gpu_names}")

    train_x, train_y = make_sequences(int(cfg["train_size"]), int(cfg["sequence_length"]), int(cfg["vocab_size"]), seed)
    val_x, val_y = make_sequences(int(cfg["val_size"]), int(cfg["sequence_length"]), int(cfg["vocab_size"]), seed + 101)
    train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=int(cfg["batch_size"]), shuffle=True)
    val_loader = DataLoader(TensorDataset(val_x, val_y), batch_size=int(cfg["batch_size"]), shuffle=False)

    model: nn.Module = TinySequenceClassifier(
        vocab=int(cfg["vocab_size"]),
        embedding_dim=int(cfg["embedding_dim"]),
        hidden_dim=int(cfg["hidden_dim"]),
        bidirectional=bool(cfg.get("bidirectional", False)),
        dropout=float(cfg["dropout"]),
    )
    data_parallel_used = bool(cuda_available and gpu_count >= 2 and cfg.get("use_data_parallel", True))
    if data_parallel_used:
        model = nn.DataParallel(model)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["lr"]), weight_decay=float(cfg["weight_decay"]))
    criterion = nn.CrossEntropyLoss()
    last_loss = 0.0
    for _ in range(int(cfg["epochs"])):
        model.train()
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            last_loss = float(loss.detach().cpu().item())

    accuracy = evaluate(model, val_loader, device)
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
