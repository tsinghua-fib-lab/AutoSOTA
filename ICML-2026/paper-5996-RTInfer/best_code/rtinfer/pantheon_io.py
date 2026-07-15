from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

from .model import BlockProfile, ExitProfile, ModelProfile, TaskSpec


_FIELD_RE = re.compile(r"^\s*([A-Za-z_]+)\s*:\s*(.+?)\s*$")


def _parse_scalar(value: str) -> str:
    value = value.strip()
    if value.startswith('"') and value.endswith('"'):
        return value[1:-1]
    return value


def parse_config_pbtxt(path: Path) -> Tuple[str, Tuple[int, ...], List[Tuple[int, int]], List[Tuple[int, int, float]]]:
    name = ""
    dims: List[int] = []
    blocks: List[Tuple[int, int]] = []
    exits: List[Tuple[int, int, float]] = []
    section: str | None = None
    current: Dict[str, str] = {}

    def flush() -> None:
        nonlocal current, section
        if section == "block_profile":
            block_id = int(current.get("id", len(blocks)))
            blocks.append((block_id, int(float(current["latency"]))))
        elif section == "exit_profile":
            exit_id = int(current.get("id", len(exits)))
            latency = int(float(current["latency"]))
            accuracy = float(current.get("accuracy", "0"))
            exits.append((exit_id, latency, accuracy))
        current = {}
        section = None

    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.endswith("{"):
            section = line[:-1].strip()
            current = {}
            continue
        if line == "}":
            flush()
            continue
        match = _FIELD_RE.match(line)
        if not match:
            continue
        key, value = match.group(1), _parse_scalar(match.group(2))
        if section is None:
            if key == "name":
                name = value
            elif key == "dims":
                dims.append(int(value))
        else:
            current[key] = value
    return name, tuple(dims), blocks, exits


def _read_profile_csv(path: Path) -> Tuple[List[float], List[int], List[int], List[float]]:
    block_memory_mib: List[float] = []
    block_latency_us: List[int] = []
    branch_latency_us: List[int] = []
    accuracy: List[float] = []
    if not path.exists():
        return block_memory_mib, block_latency_us, branch_latency_us, accuracy
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            block_memory_mib.append(float(row.get("block mem", 0.0)))
            block_latency_us.append(int(float(row.get("block latency", 0.0)) * 1000))
            branch_latency_us.append(int(float(row.get("branch latency", 0.0)) * 1000))
            accuracy.append(float(row.get("accuracy", 0.0)))
    return block_memory_mib, block_latency_us, branch_latency_us, accuracy


def load_model_profile(model_dir: Path, profile_root: Path | None = None) -> ModelProfile:
    name, dims, blocks_raw, exits_raw = parse_config_pbtxt(model_dir / "config.pbtxt")
    csv_memory, csv_block_latency, csv_branch_latency, csv_accuracy = [], [], [], []
    if profile_root is not None:
        csv_memory, csv_block_latency, csv_branch_latency, csv_accuracy = _read_profile_csv(profile_root / name / "profile.csv")

    blocks: List[BlockProfile] = []
    for block_id, latency_us in blocks_raw:
        memory_mib = csv_memory[block_id] if block_id < len(csv_memory) else 1.0
        if latency_us <= 0 and block_id < len(csv_block_latency):
            latency_us = csv_block_latency[block_id]
        blocks.append(BlockProfile(block_id=block_id, latency_us=latency_us, memory_mib=memory_mib))

    exits: List[ExitProfile] = []
    for idx, (previous_block_id, latency_us, accuracy) in enumerate(exits_raw):
        if latency_us <= 0 and previous_block_id < len(csv_branch_latency):
            latency_us = csv_branch_latency[previous_block_id]
        if accuracy <= 0 and previous_block_id < len(csv_accuracy):
            accuracy = csv_accuracy[previous_block_id]
        exits.append(
            ExitProfile(
                exit_id=idx,
                previous_block_id=previous_block_id,
                latency_us=latency_us,
                accuracy=accuracy,
            )
        )
    exits.sort(key=lambda exit_profile: exit_profile.previous_block_id)
    return ModelProfile(name=name, dims=dims, blocks=tuple(blocks), exits=tuple(exits))


def load_profile_csv_model(profile_dir: Path) -> ModelProfile:
    csv_memory, csv_block_latency, csv_branch_latency, csv_accuracy = _read_profile_csv(profile_dir / "profile.csv")
    blocks = tuple(
        BlockProfile(block_id=idx, latency_us=latency_us, memory_mib=csv_memory[idx] if idx < len(csv_memory) else 1.0)
        for idx, latency_us in enumerate(csv_block_latency)
    )
    exits = tuple(
        ExitProfile(
            exit_id=idx,
            previous_block_id=idx,
            latency_us=csv_branch_latency[idx] if idx < len(csv_branch_latency) else 1,
            accuracy=csv_accuracy[idx] if idx < len(csv_accuracy) else 0.0,
        )
        for idx in range(len(blocks))
    )
    return ModelProfile(name=profile_dir.name, dims=tuple(), blocks=blocks, exits=exits)


def load_repository(repo_root: Path, profile_root: Path | None = None) -> Dict[str, ModelProfile]:
    deploy_dir = repo_root / "variants_for_deployment"
    models: Dict[str, ModelProfile] = {}
    for model_dir in sorted(deploy_dir.iterdir()):
        if (model_dir / "config.pbtxt").exists():
            model = load_model_profile(model_dir, profile_root)
            models[model.name] = model
    if profile_root is not None and profile_root.exists():
        for profile_dir in sorted(profile_root.iterdir()):
            if profile_dir.is_dir() and (profile_dir / "profile.csv").exists() and profile_dir.name not in models:
                models[profile_dir.name] = load_profile_csv_model(profile_dir)
    return models


def load_tasks(workload_json: Path) -> List[TaskSpec]:
    data = json.loads(workload_json.read_text())
    tasks: List[TaskSpec] = []
    for item in data["workloads"]:
        start = int(item.get("start", 0))
        if start < 1000:
            start *= 1000
        end = int(item.get("end", 0))
        tasks.append(
            TaskSpec(
                model_name=item["model_name"],
                deadline_us=int(item["deadline"]),
                period_us=int(item["period"]),
                start_us=start,
                end_us=end,
                shape=tuple(int(v) for v in item.get("shape", [])),
            )
        )
    return tasks


def load_memory_budget_mib(deploy_json: Path, default_mib: float = 512.0) -> float:
    data = json.loads(deploy_json.read_text())
    return float(data.get("max_memory", default_mib))
