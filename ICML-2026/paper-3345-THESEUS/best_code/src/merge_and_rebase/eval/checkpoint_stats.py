from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from merge_and_rebase.io.ckpt import normalize_common_prefixes, unwrap_state_dict
from merge_and_rebase.io.peft_helpers import load_peft_adapter_dir_components, normalize_peft_adapter_dir_checkpoint


@dataclass(frozen=True)
class CheckpointRow:
    checkpoint_path: str
    task: str | None
    strategy: str | None
    checkpoint_kind: str | None
    checkpoint_format: str
    layer_name: str
    param_kind: str
    shape: list[int]
    requires_grad: bool | None
    norm_1: float
    norm_3: float
    avg: float
    dtype: str

    def to_json(self) -> dict[str, Any]:
        return {
            "checkpoint_path": self.checkpoint_path,
            "task": self.task,
            "strategy": self.strategy,
            "checkpoint_kind": self.checkpoint_kind,
            "checkpoint_format": self.checkpoint_format,
            "layer_name": self.layer_name,
            "param_kind": self.param_kind,
            "shape": list(self.shape),
            "requires_grad": self.requires_grad,
            "norm_1": self.norm_1,
            "norm_3": self.norm_3,
            "avg": self.avg,
            "dtype": self.dtype,
        }

    def to_csv_row(self) -> dict[str, Any]:
        data = self.to_json()
        data["shape"] = "x".join(str(v) for v in self.shape)
        return data


def _checkpoint_kind_from_name(path: Path) -> str | None:
    name = path.name
    if "_best_ep" in name:
        return "best_ep"
    if "_last_ep" in name:
        return "last_ep"
    return None


def _tensor_stats(tensor: torch.Tensor) -> tuple[list[int], float, float, float, str]:
    cpu = tensor.detach().cpu()
    work = cpu.to(dtype=torch.float64)
    shape = [int(v) for v in cpu.shape]
    norm_1 = float(work.abs().sum().item())
    norm_3 = float(work.abs().pow(3).sum().pow(1.0 / 3.0).item())
    avg = float(work.mean().item()) if work.numel() > 0 else float("nan")
    return shape, norm_1, norm_3, avg, str(cpu.dtype)


def _make_row(
    *,
    checkpoint_path: Path,
    task: str | None,
    strategy: str | None,
    checkpoint_format: str,
    layer_name: str,
    param_kind: str,
    tensor: torch.Tensor,
    requires_grad: bool | None,
) -> CheckpointRow:
    shape, norm_1, norm_3, avg, dtype = _tensor_stats(tensor)
    return CheckpointRow(
        checkpoint_path=str(checkpoint_path),
        task=task,
        strategy=strategy,
        checkpoint_kind=_checkpoint_kind_from_name(checkpoint_path),
        checkpoint_format=checkpoint_format,
        layer_name=layer_name,
        param_kind=param_kind,
        shape=shape,
        requires_grad=requires_grad,
        norm_1=norm_1,
        norm_3=norm_3,
        avg=avg,
        dtype=dtype,
    )


def _collect_peft_rows(checkpoint_path: Path, payload: dict[str, Any]) -> tuple[list[CheckpointRow], dict[str, Any]]:
    normalized = normalize_peft_adapter_dir_checkpoint(payload, checkpoint_path=str(checkpoint_path))
    adapter_dir = str(normalized["peft_adapter_dir"])
    lora_state, _ = load_peft_adapter_dir_components(adapter_dir, checkpoint_path=str(checkpoint_path))
    dense_state_raw = normalized.get("peft_dense_state", {})
    dense_state = dense_state_raw if isinstance(dense_state_raw, dict) else {}
    dense_trainable_keys = normalized.get("dense_trainable_keys", ())
    dense_trainable = {str(key) for key in dense_trainable_keys} if isinstance(dense_trainable_keys, (list, tuple)) else set()

    task = str(normalized.get("task")) if normalized.get("task") is not None else checkpoint_path.parent.name
    strategy = str(normalized.get("strategy")) if normalized.get("strategy") is not None else None

    rows: list[CheckpointRow] = []
    for name, tensor in sorted(lora_state.items()):
        if torch.is_tensor(tensor):
            rows.append(
                _make_row(
                    checkpoint_path=checkpoint_path,
                    task=task,
                    strategy=strategy,
                    checkpoint_format="peft",
                    layer_name=str(name),
                    param_kind="lora",
                    tensor=tensor,
                    requires_grad=True,
                )
            )
    for name, tensor in sorted(dense_state.items()):
        if torch.is_tensor(tensor):
            requires_grad = True if not dense_trainable else (str(name) in dense_trainable)
            rows.append(
                _make_row(
                    checkpoint_path=checkpoint_path,
                    task=task,
                    strategy=strategy,
                    checkpoint_format="peft",
                    layer_name=str(name),
                    param_kind="dense",
                    tensor=tensor,
                    requires_grad=requires_grad,
                )
            )

    meta = {
        "checkpoint_path": str(checkpoint_path),
        "task": task,
        "strategy": strategy,
        "checkpoint_kind": _checkpoint_kind_from_name(checkpoint_path),
        "checkpoint_format": "peft",
        "row_count": len(rows),
        "lora_rows": sum(1 for row in rows if row.param_kind == "lora"),
        "dense_rows": sum(1 for row in rows if row.param_kind == "dense"),
        "peft_trainable_plan": normalized.get("peft_trainable_plan", {}),
        "peft_cfg": normalized.get("peft_cfg", {}),
        "adapter_dir": adapter_dir,
    }
    return rows, meta


def _collect_full_rows(checkpoint_path: Path, payload: dict[str, Any]) -> tuple[list[CheckpointRow], dict[str, Any]]:
    task = str(payload.get("task")) if payload.get("task") is not None else checkpoint_path.parent.name
    strategy = str(payload.get("strategy")) if payload.get("strategy") is not None else None

    raw_state = payload.get("state_dict", payload)
    state = normalize_common_prefixes(unwrap_state_dict(raw_state))

    rows: list[CheckpointRow] = []
    for name, tensor in sorted(state.items()):
        rows.append(
            _make_row(
                checkpoint_path=checkpoint_path,
                task=task,
                strategy=strategy,
                checkpoint_format="full",
                layer_name=str(name),
                param_kind="dense",
                tensor=tensor,
                requires_grad=None,
            )
        )

    meta = {
        "checkpoint_path": str(checkpoint_path),
        "task": task,
        "strategy": strategy,
        "checkpoint_kind": _checkpoint_kind_from_name(checkpoint_path),
        "checkpoint_format": "full",
        "row_count": len(rows),
    }
    return rows, meta


def _collect_adapter_dir_rows(adapter_dir: Path) -> tuple[list[CheckpointRow], dict[str, Any]]:
    lora_state, cfg_map = load_peft_adapter_dir_components(str(adapter_dir))
    task = adapter_dir.parent.name
    rows: list[CheckpointRow] = []
    for name, tensor in sorted(lora_state.items()):
        rows.append(
            _make_row(
                checkpoint_path=adapter_dir,
                task=task,
                strategy=adapter_dir.name,
                checkpoint_format="peft_adapter_dir",
                layer_name=str(name),
                param_kind="lora",
                tensor=tensor,
                requires_grad=True,
            )
        )

    meta = {
        "checkpoint_path": str(adapter_dir),
        "task": task,
        "strategy": adapter_dir.name,
        "checkpoint_kind": None,
        "checkpoint_format": "peft_adapter_dir",
        "row_count": len(rows),
        "peft_cfg": cfg_map.get("default", {}) if isinstance(cfg_map, dict) else {},
    }
    return rows, meta


def _inspect_checkpoint(path: Path) -> tuple[list[CheckpointRow], dict[str, Any]]:
    if path.is_dir():
        return _collect_adapter_dir_rows(path)

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError(f"Unsupported checkpoint payload type for {path}: {type(payload).__name__}")

    checkpoint_format = str(payload.get("format", "")).strip().lower()
    if checkpoint_format == "peft" and isinstance(payload.get("peft_adapter_dir"), str):
        return _collect_peft_rows(path, payload)
    if checkpoint_format == "full" or "state_dict" in payload:
        return _collect_full_rows(path, payload)
    raise ValueError(
        f"Unsupported checkpoint format for {path}. "
        "Expected PEFT payload with 'peft_adapter_dir' or full payload with 'state_dict'."
    )


def _iter_candidate_paths(input_path: Path, checkpoint_kind: str) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir() and (input_path / "adapter_config.json").exists():
        return [input_path]
    if input_path.is_dir() and (input_path / "adapter_model.safetensors").exists():
        return [input_path]

    patterns: list[str]
    if checkpoint_kind == "best":
        patterns = ["**/*_best_ep.pt"]
    elif checkpoint_kind == "last":
        patterns = ["**/*_last_ep.pt"]
    elif checkpoint_kind == "all":
        patterns = ["**/*_best_ep.pt", "**/*_last_ep.pt"]
    else:
        raise ValueError("checkpoint_kind must be one of: best, last, all")

    found: list[Path] = []
    seen: set[str] = set()
    for pattern in patterns:
        for path in sorted(input_path.glob(pattern)):
            key = str(path.resolve())
            if key not in seen:
                seen.add(key)
                found.append(path)
    return found


def _write_csv(path: Path, rows: list[CheckpointRow]) -> None:
    fieldnames = [
        "checkpoint_path",
        "task",
        "strategy",
        "checkpoint_kind",
        "checkpoint_format",
        "layer_name",
        "param_kind",
        "shape",
        "requires_grad",
        "norm_1",
        "norm_3",
        "avg",
        "dtype",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.to_csv_row())


def _write_json(path: Path, items: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump({"checkpoints": items}, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def _print_checkpoint(meta: dict[str, Any], rows: list[CheckpointRow]) -> None:
    print(
        f"\n[{meta['checkpoint_format']}] {meta['checkpoint_path']} "
        f"task={meta.get('task')} kind={meta.get('checkpoint_kind')} rows={meta['row_count']}"
    )
    for row in rows:
        stats = {
            "kind": row.param_kind,
            "shape": row.shape,
            "requires_grad": row.requires_grad,
            "norm_1": row.norm_1,
            "norm_3": row.norm_3,
            "avg": row.avg,
        }
        print(f"{row.layer_name}: {json.dumps(stats, ensure_ascii=False, sort_keys=True)}")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Inspect LoRA/full checkpoints and print per-parameter statistics.")
    parser.add_argument(
        "--input",
        dest="inputs",
        action="append",
        required=True,
        help="Checkpoint file, adapter dir, or experiment root to scan recursively.",
    )
    parser.add_argument(
        "--checkpoint-kind",
        choices=["best", "last", "all"],
        default="all",
        help="When an input is a directory, which checkpoints to scan.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Optional directory where checkpoint_stats.json and checkpoint_stats.csv are saved.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Do not print every parameter row to stdout.",
    )
    return parser


def main() -> None:
    args = build_argparser().parse_args()

    input_paths = [Path(path) for path in args.inputs]
    candidates: list[Path] = []
    seen: set[str] = set()
    for input_path in input_paths:
        for candidate in _iter_candidate_paths(input_path, args.checkpoint_kind):
            key = str(candidate.resolve())
            if key not in seen:
                seen.add(key)
                candidates.append(candidate)

    if not candidates:
        raise FileNotFoundError("No checkpoints matched the provided inputs.")

    all_rows: list[CheckpointRow] = []
    json_items: list[dict[str, Any]] = []
    for candidate in candidates:
        rows, meta = _inspect_checkpoint(candidate)
        all_rows.extend(rows)
        json_items.append(
            {
                **meta,
                "rows": [row.to_json() for row in rows],
            }
        )
        if not args.quiet:
            _print_checkpoint(meta, rows)

    print(
        f"\nScanned {len(candidates)} checkpoint references, collected {len(all_rows)} parameter rows."
    )

    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        json_path = output_dir / "checkpoint_stats.json"
        csv_path = output_dir / "checkpoint_stats.csv"
        _write_json(json_path, json_items)
        _write_csv(csv_path, all_rows)
        print(f"Saved JSON stats to {json_path}")
        print(f"Saved CSV stats to {csv_path}")


if __name__ == "__main__":
    main()
