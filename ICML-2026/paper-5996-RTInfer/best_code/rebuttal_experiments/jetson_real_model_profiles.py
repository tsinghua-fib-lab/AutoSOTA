from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn


@dataclass
class ProfileResult:
    name: str
    family: str
    input_shape: str
    dtype: str
    params_m: float
    latency_ms: float
    peak_allocated_mib: float
    peak_reserved_mib: float
    output_shape: str
    notes: str


class ConvAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, stride: int = 1) -> None:
        super().__init__()
        pad = k // 2
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, k, stride=stride, padding=pad, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Bottleneck(nn.Module):
    def __init__(self, ch: int) -> None:
        super().__init__()
        hidden = max(ch // 2, 8)
        self.cv1 = ConvAct(ch, hidden, 1)
        self.cv2 = ConvAct(hidden, ch, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.cv2(self.cv1(x))


class YoloLikeDetector(nn.Module):
    """YOLOv8-style high-resolution detector backbone/head without external deps."""

    def __init__(self, width: int = 32, repeats: tuple[int, ...] = (2, 3, 3, 2)) -> None:
        super().__init__()
        channels = [width, width * 2, width * 4, width * 8]
        blocks: list[nn.Module] = [ConvAct(3, channels[0], 3, 2)]
        in_ch = channels[0]
        for out_ch, repeat in zip(channels, repeats):
            blocks.append(ConvAct(in_ch, out_ch, 3, 2))
            blocks.extend(Bottleneck(out_ch) for _ in range(repeat))
            in_ch = out_ch
        self.backbone = nn.Sequential(*blocks)
        self.head = nn.Sequential(
            ConvAct(in_ch, in_ch, 3),
            nn.Conv2d(in_ch, 84, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))


class PatchTransformer(nn.Module):
    def __init__(
        self,
        image_size: int,
        patch: int,
        dim: int,
        depth: int,
        heads: int,
        num_classes: int,
        stem: bool = False,
    ) -> None:
        super().__init__()
        if stem:
            self.stem = nn.Sequential(ConvAct(3, 32, 3, 2), ConvAct(32, 64, 3, 2))
            patch_in = 64
            patch_stride = patch // 4
            grid = image_size // patch
        else:
            self.stem = nn.Identity()
            patch_in = 3
            patch_stride = patch
            grid = image_size // patch
        self.patch = nn.Conv2d(patch_in, dim, patch_stride, stride=patch_stride)
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos = nn.Parameter(torch.zeros(1, grid * grid + 1, dim))
        enc = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=dim * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc, num_layers=depth)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.patch(x).flatten(2).transpose(1, 2)
        cls = self.cls.expand(x.size(0), -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos[:, : x.size(1)]
        x = self.encoder(x)
        return self.head(self.norm(x[:, 0]))


class GPTBlock(nn.Module):
    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))

    def forward(
        self, x: torch.Tensor, past: tuple[torch.Tensor, torch.Tensor] | None
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        bsz, steps, dim = x.shape
        qkv = self.qkv(self.norm1(x)).view(bsz, steps, 3, self.heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        if past is not None:
            k = torch.cat([past[0], k], dim=2)
            v = torch.cat([past[1], v], dim=2)
        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = torch.softmax(attn, dim=-1)
        y = torch.matmul(attn, v).transpose(1, 2).contiguous().view(bsz, steps, dim)
        x = x + self.proj(y)
        x = x + self.mlp(self.norm2(x))
        return x, (k, v)


class GPT2KVSmall(nn.Module):
    def __init__(self, vocab: int = 8192, dim: int = 768, heads: int = 12, depth: int = 6) -> None:
        super().__init__()
        self.token = nn.Embedding(vocab, dim)
        self.pos = nn.Parameter(torch.zeros(1, 512, dim))
        self.blocks = nn.ModuleList(GPTBlock(dim, heads) for _ in range(depth))
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab, bias=False)

    def forward(
        self, tokens: torch.Tensor, past: list[tuple[torch.Tensor, torch.Tensor] | None] | None = None
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        x = self.token(tokens) + self.pos[:, : tokens.size(1)]
        next_past = []
        if past is None:
            past = [None] * len(self.blocks)
        for block, block_past in zip(self.blocks, past):
            x, kv = block(x, block_past)
            next_past.append(kv)
        return self.head(self.norm(x[:, -1])), next_past


def count_params(model: nn.Module) -> float:
    return sum(p.numel() for p in model.parameters()) / 1_000_000.0


def to_dtype(model: nn.Module, dtype: torch.dtype) -> nn.Module:
    if dtype == torch.float16:
        return model.half()
    return model.float()


def cuda_mib(value: int) -> float:
    return value / (1024.0 * 1024.0)


def benchmark_module(
    name: str,
    family: str,
    model: nn.Module,
    input_tensor: torch.Tensor,
    dtype_name: str,
    warmup: int,
    repeat: int,
    notes: str,
) -> ProfileResult:
    device = torch.device("cuda")
    model = model.to(device).eval()
    input_tensor = input_tensor.to(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        for _ in range(warmup):
            out = model(input_tensor)
            if isinstance(out, tuple):
                out = out[0]
            torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        for _ in range(repeat):
            out = model(input_tensor)
            if isinstance(out, tuple):
                out = out[0]
            torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) * 1000.0 / repeat
    return ProfileResult(
        name=name,
        family=family,
        input_shape=str(tuple(input_tensor.shape)),
        dtype=dtype_name,
        params_m=count_params(model),
        latency_ms=elapsed,
        peak_allocated_mib=cuda_mib(torch.cuda.max_memory_allocated()),
        peak_reserved_mib=cuda_mib(torch.cuda.max_memory_reserved()),
        output_shape=str(tuple(out.shape)),
        notes=notes,
    )


def benchmark_gpt_kv(dtype_name: str, warmup: int, repeat: int, quick: bool) -> list[ProfileResult]:
    dtype = torch.float16 if dtype_name == "fp16" else torch.float32
    model = to_dtype(GPT2KVSmall(depth=4 if quick else 6), dtype).cuda().eval()
    results: list[ProfileResult] = []
    chunks = [32, 32, 64] if quick else [64, 64, 128]
    past = None
    with torch.inference_mode():
        for chunk_idx, length in enumerate(chunks, start=1):
            tokens = torch.randint(0, 8192, (1, length), device="cuda")
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            for _ in range(max(1, warmup)):
                _, tmp_past = model(tokens, past)
                torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            for _ in range(repeat):
                logits, next_past = model(tokens, past)
                torch.cuda.synchronize()
            elapsed = (time.perf_counter() - start) * 1000.0 / repeat
            past = next_past
            kv_tokens = sum(chunks[:chunk_idx])
            results.append(
                ProfileResult(
                    name=f"gpt2_small_kv_step{chunk_idx}",
                    family="edge_gpt2_kv",
                    input_shape=f"tokens=(1,{length}), kv_tokens={kv_tokens}",
                    dtype=dtype_name,
                    params_m=count_params(model),
                    latency_ms=elapsed,
                    peak_allocated_mib=cuda_mib(torch.cuda.max_memory_allocated()),
                    peak_reserved_mib=cuda_mib(torch.cuda.max_memory_reserved()),
                    output_shape=str(tuple(logits.shape)),
                    notes="autoregressive KV-cache step; past K/V kept live between chunks",
                )
            )
    return results


def build_profiles(dtype_name: str, quick: bool) -> list[tuple[str, str, nn.Module, torch.Tensor, str]]:
    dtype = torch.float16 if dtype_name == "fp16" else torch.float32
    yolo_l_shape = (1, 3, 720, 1280) if quick else (1, 3, 1080, 1920)
    vit_size = 768 if quick else 1024
    return [
        (
            "yolov8l_like_highres",
            "modern_cnn_detection",
            to_dtype(YoloLikeDetector(width=24 if quick else 32, repeats=(1, 2, 2, 1)), dtype),
            torch.randn(yolo_l_shape, dtype=dtype),
            "YOLOv8-style CSP detector; high-resolution activation-dominated profile",
        ),
        (
            "yolov8n_like_640",
            "modern_cnn_detection",
            to_dtype(YoloLikeDetector(width=12, repeats=(1, 1, 1, 1)), dtype),
            torch.randn((1, 3, 640, 640), dtype=dtype),
            "YOLOv8n-style compact detector for the rebuttal mixed traffic setup",
        ),
        (
            "mobilevit_s_like_512",
            "mobile_vit_scene",
            to_dtype(PatchTransformer(512, patch=16, dim=192, depth=2, heads=4, num_classes=15, stem=True), dtype),
            torch.randn((1, 3, 512, 512), dtype=dtype),
            "MobileViT-S-style scene classifier with convolutional stem plus transformer blocks",
        ),
        (
            "vit_l_like_1024",
            "large_vit_scene",
            to_dtype(PatchTransformer(vit_size, patch=32, dim=512, depth=4 if quick else 6, heads=8, num_classes=15), dtype),
            torch.randn((1, 3, vit_size, vit_size), dtype=dtype),
            "ViT-L-style high-resolution transformer profile; patch=32 keeps the board run bounded",
        ),
    ]


def write_outputs(results: list[ProfileResult], out_dir: Path, quick: bool) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "jetson_real_model_profiles.csv"
    json_path = out_dir / "jetson_real_model_profiles.json"
    md_path = out_dir / "jetson_real_model_profiles.md"
    rows = [r.__dict__ for r in results]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with json_path.open("w") as f:
        json.dump(rows, f, indent=2)
    with md_path.open("w") as f:
        f.write("# Jetson Real-Model Profiles\n\n")
        f.write("These are measured CUDA profiles on Jetson Xavier NX using pure PyTorch model graphs.\n")
        f.write("They replace the earlier purely synthetic rebuttal profiles where official packages or pretrained weights are unavailable.\n\n")
        f.write(f"- quick mode: `{quick}`\n")
        f.write("- weights: randomly initialized, architecture-level profiling only\n")
        f.write("- metric: single-stream latency and CUDA peak allocated/reserved memory\n\n")
        f.write("| model | family | input | dtype | params M | latency ms | peak alloc MiB | peak reserved MiB |\n")
        f.write("| --- | --- | --- | --- | ---: | ---: | ---: | ---: |\n")
        for r in results:
            f.write(
                f"| {r.name} | {r.family} | `{r.input_shape}` | {r.dtype} | "
                f"{r.params_m:.2f} | {r.latency_ms:.2f} | {r.peak_allocated_mib:.1f} | {r.peak_reserved_mib:.1f} |\n"
            )
        f.write("\n## Notes\n\n")
        for r in results:
            f.write(f"- `{r.name}`: {r.notes}\n")
    print(f"wrote {csv_path}")
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="outputs/jetson_real_profiles")
    parser.add_argument("--dtype", choices=("fp16", "fp32"), default="fp16")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--quick", action="store_true", help="Use bounded shapes for faster bring-up runs.")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Jetson profiling")

    results: list[ProfileResult] = []
    for name, family, model, tensor, notes in build_profiles(args.dtype, args.quick):
        print(f"profiling {name} ...", flush=True)
        result = benchmark_module(name, family, model, tensor, args.dtype, args.warmup, args.repeat, notes)
        results.append(result)
        print(
            f"{name}: latency={result.latency_ms:.2f}ms "
            f"alloc={result.peak_allocated_mib:.1f}MiB reserved={result.peak_reserved_mib:.1f}MiB",
            flush=True,
        )
        del model, tensor
        torch.cuda.empty_cache()

    print("profiling gpt2_small_kv ...", flush=True)
    results.extend(benchmark_gpt_kv(args.dtype, args.warmup, args.repeat, args.quick))
    write_outputs(results, Path(args.out_dir), args.quick)


if __name__ == "__main__":
    main()
