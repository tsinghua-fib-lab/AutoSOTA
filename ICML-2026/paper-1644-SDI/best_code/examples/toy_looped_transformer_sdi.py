"""
Toy example: train a tiny looped transformer on random data, save checkpoints,
then compute SDI (projected) and fast TracIn.

Run:
  uv run python examples/toy_looped_transformer_sdi.py --device cuda
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from sdi import CheckpointSpec, ProjectedTracInSDI


@dataclass
class Batch:
    tokens: torch.Tensor  # (B, T)


class RandomTokenDataset(Dataset):
    def __init__(self, *, n: int, seq_len: int, vocab_size: int, seed: int) -> None:
        g = torch.Generator().manual_seed(seed)
        self.tokens = torch.randint(0, vocab_size, (n, seq_len), generator=g)

    def __len__(self) -> int:
        return self.tokens.shape[0]

    def __getitem__(self, idx: int) -> Batch:
        return Batch(tokens=self.tokens[idx])


def batchify(rows: list[Batch]) -> Batch:
    return Batch(tokens=torch.stack([r.tokens for r in rows], dim=0))


class TinyBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        B, T, D = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, T, self.n_head, D // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, D // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, D // self.n_head).transpose(1, 2)
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        attn = attn.transpose(1, 2).contiguous().view(B, T, D)
        x = x + self.proj(attn)
        x = x + self.mlp(F.layer_norm(x, (D,)))
        return x


class TinyLoopedTransformer(nn.Module):
    def __init__(
        self, *, vocab_size: int, seq_len: int, d_model: int, n_head: int, tau: int
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.tau = tau
        self.read_in = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, seq_len, d_model))
        self.recurrent_block = TinyBlock(d_model, n_head)
        self.read_out = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.read_in(tokens) + self.pos_emb
        state = torch.zeros_like(x)
        for _ in range(self.tau):
            state = self.recurrent_block(state + x)
        logits = self.read_out(state)
        return logits


def iter_batches(loader: DataLoader) -> Iterator[Batch]:
    while True:
        for batch in loader:
            yield batch


def per_example_loss(model: nn.Module, batch: Batch) -> torch.Tensor:
    tokens = batch.tokens.to(next(model.parameters()).device)
    logits = model(tokens)
    # Next-token prediction: predict token[t+1] from logits[t].
    targets = tokens[:, 1:]
    logits = logits[:, :-1, :]
    per_pos = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)), targets.reshape(-1), reduction="none"
    )
    per_pos = per_pos.view(tokens.size(0), -1)
    return per_pos.mean(dim=1)


def train_and_save_checkpoints(
    *,
    model: nn.Module,
    loader: DataLoader,
    steps: int,
    checkpoint_every: int,
    out_dir: Path,
    device: torch.device,
) -> list[CheckpointSpec]:
    out_dir.mkdir(parents=True, exist_ok=True)
    model.to(device)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    ckpts: list[CheckpointSpec] = []

    batch_iter = iter_batches(loader)
    for step in range(1, steps + 1):
        batch = next(batch_iter)
        loss = per_example_loss(model, batch).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % checkpoint_every == 0 or step == steps:
            path = out_dir / f"ckpt_step_{step:04d}.pt"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": opt.state_dict(),
                    "step": step,
                },
                path,
            )
            ckpts.append(CheckpointSpec(path=path, weight=None))
    return ckpts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device", type=str, default=None, help="cpu/cuda (default: auto)"
    )
    args = parser.parse_args()

    device = (
        torch.device(args.device)
        if args.device
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    torch.manual_seed(0)

    vocab_size = 64
    seq_len = 32
    tau = 6
    d_model = 64
    n_head = 4

    train_size = 64
    query_size = 16
    batch_size = 8
    steps = 40
    checkpoint_every = 10
    projection_size = 256

    dataset = RandomTokenDataset(
        n=train_size + query_size, seq_len=seq_len, vocab_size=vocab_size, seed=1
    )
    train_ds = torch.utils.data.Subset(dataset, list(range(train_size)))
    query_ds = torch.utils.data.Subset(
        dataset, list(range(train_size, train_size + query_size))
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, collate_fn=batchify
    )
    query_loader = DataLoader(
        query_ds, batch_size=batch_size, shuffle=False, collate_fn=batchify
    )

    model = TinyLoopedTransformer(
        vocab_size=vocab_size,
        seq_len=seq_len,
        d_model=d_model,
        n_head=n_head,
        tau=tau,
    )

    repo_root = Path(__file__).resolve().parent.parent
    ckpt_dir = repo_root / "checkpoints"
    checkpoints = train_and_save_checkpoints(
        model=model,
        loader=train_loader,
        steps=steps,
        checkpoint_every=checkpoint_every,
        out_dir=ckpt_dir,
        device=device,
    )

    # SDI with on-the-fly compression.
    estimator = ProjectedTracInSDI(
        model=model,
        target_modules=model.recurrent_block,
        projection_size=projection_size,
        loss_reduction="sum",
        expected_steps=tau,
    )

    out = estimator.influence_across_checkpoints(
        checkpoints=checkpoints,
        train_loader=train_loader,
        query_loader=query_loader,
        loss_fn=per_example_loss,
        mode="sdi",
        train_chunk_size=16,
        query_chunk_size=8,
    )
    assert out.sdi is not None
    sdi = out.sdi
    print("SDI shape:", tuple(sdi.shape), "TracIn shape:", tuple(out.tracin.shape))
    print(
        "Conservation check |tracin - sum(sdi)|:",
        float((out.tracin - sdi.sum(dim=2)).abs().max()),
    )

    # Fast TracIn: scalar-only, avoids SDI tensor allocation.
    tracin_only = estimator.influence_across_checkpoints(
        checkpoints=checkpoints,
        train_loader=train_loader,
        query_loader=query_loader,
        loss_fn=per_example_loss,
        mode="tracin",
        train_chunk_size=16,
        query_chunk_size=8,
    )
    print("Fast TracIn shape:", tuple(tracin_only.tracin.shape))

    # Example: most influential training samples for query 0.
    topk = torch.topk(tracin_only.tracin[:, 0], k=5).indices.tolist()
    print("Top-5 influencing train indices for query 0:", topk)


if __name__ == "__main__":
    main()
