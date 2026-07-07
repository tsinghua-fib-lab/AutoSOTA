from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from sdi import CheckpointSpec, ProjectedTracInSDI


class ToyDataset(Dataset):
    def __init__(self, *, n: int, seq_len: int, vocab_size: int, seed: int) -> None:
        g = torch.Generator().manual_seed(seed)
        self.tokens = torch.randint(0, vocab_size, (n, seq_len), generator=g)

    def __len__(self) -> int:
        return self.tokens.shape[0]

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.tokens[idx]


def collate(tokens: list[torch.Tensor]) -> torch.Tensor:
    return torch.stack(tokens, dim=0)


class ToyBlock(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_model)
        self.fc2 = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.fc2(F.relu(self.fc1(x)))


class ToyLoopedModel(nn.Module):
    def __init__(
        self, *, vocab_size: int, seq_len: int, d_model: int, tau: int
    ) -> None:
        super().__init__()
        self.tau = tau
        self.embed = nn.Embedding(vocab_size, d_model)
        self.block = ToyBlock(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.embed(tokens)
        state = torch.zeros_like(x)
        for _ in range(self.tau):
            state = self.block(state + x)
        return self.head(state)


def per_example_loss(model: nn.Module, tokens: torch.Tensor) -> torch.Tensor:
    logits = model(tokens)
    targets = tokens[:, 1:]
    logits = logits[:, :-1, :]
    per_pos = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        reduction="none",
    ).view(tokens.size(0), -1)
    return per_pos.mean(dim=1)


def test_sdi_conservation(tmp_path: Path) -> None:
    torch.manual_seed(0)
    vocab_size = 32
    seq_len = 12
    tau = 4
    d_model = 32

    model = ToyLoopedModel(
        vocab_size=vocab_size, seq_len=seq_len, d_model=d_model, tau=tau
    )
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    ckpt_path = tmp_path / "ckpt.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": opt.state_dict(),
        },
        ckpt_path,
    )

    train_ds = ToyDataset(n=12, seq_len=seq_len, vocab_size=vocab_size, seed=1)
    query_ds = ToyDataset(n=6, seq_len=seq_len, vocab_size=vocab_size, seed=2)
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=False, collate_fn=collate)
    query_loader = DataLoader(query_ds, batch_size=3, shuffle=False, collate_fn=collate)

    est = ProjectedTracInSDI(
        model=model,
        target_modules=model.block,
        projection_size=64,
        loss_reduction="sum",
        expected_steps=tau,
        seed=0,
    )
    out = est.influence_across_checkpoints(
        checkpoints=[CheckpointSpec(path=str(ckpt_path))],
        train_loader=train_loader,
        query_loader=query_loader,
        loss_fn=per_example_loss,
        mode="sdi",
    )
    assert out.sdi is not None
    assert torch.allclose(out.tracin, out.sdi.sum(dim=2), atol=1e-4, rtol=1e-4)
