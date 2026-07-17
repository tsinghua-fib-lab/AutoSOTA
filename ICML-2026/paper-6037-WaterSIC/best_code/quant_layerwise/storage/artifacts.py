from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict

import torch


def safe_stem(name: str) -> str:
    return (
        str(name)
        .replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
        .replace(" ", "_")
    )


@dataclass
class LayerArtifact:
    """A saved quantization result for exactly one weight matrix."""

    method: str  # "gptq" | "zsic"
    module_name: str  # e.g. "layers.0.attention.wq"
    weight_name: str  # e.g. "layers.0.attention.wq.weight"
    shape: tuple[int, int]

    # Method-specific payload.
    payload: Dict[str, Any] = None

    # Bookkeeping
    created_at: float = 0.0

    def __post_init__(self):
        if self.payload is None:
            self.payload = {}
        if not self.created_at:
            self.created_at = time.time()

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["shape"] = list(self.shape)
        return d

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "LayerArtifact":
        dd = dict(d)
        dd["shape"] = tuple(int(x) for x in dd["shape"])
        return LayerArtifact(**dd)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        torch.save(self.to_dict(), tmp)
        tmp.replace(path)
        return path

    @staticmethod
    def load(path: str | Path, map_location: str | torch.device = "cpu") -> "LayerArtifact":
        path = Path(path)
        d = torch.load(path, map_location=map_location)
        return LayerArtifact.from_dict(d)


@dataclass
class RunManifest:
    """Tracks a partially quantized model."""

    model_name: str
    method: str
    run_id: str
    config: Dict[str, Any]

    # module_name -> relative artifact path
    # When world_size > 1, this stores the rank-0 path (convention).
    # Other ranks derive their path by replacing "shard0ofN" with "shard{rank}ofN".
    artifacts: Dict[str, str]

    created_at: float = 0.0
    updated_at: float = 0.0

    # Number of tensor-parallel shards used during quantization.
    # 1 = single-GPU (legacy default). >1 = multi-GPU tensor parallel.
    world_size: int = 1

    def __post_init__(self):
        if not self.created_at:
            self.created_at = time.time()
        if not self.updated_at:
            self.updated_at = self.created_at

    def add(self, module_name: str, relpath: str):
        self.artifacts[module_name] = relpath
        self.updated_at = time.time()

    def has(self, module_name: str) -> bool:
        return module_name in self.artifacts

    def remove(self, module_name: str):
        """Remove a module from the manifest (e.g. incomplete shards on resume)."""
        self.artifacts.pop(module_name, None)
        self.updated_at = time.time()

    def artifact_relpath_for_rank(self, module_name: str, rank: int) -> str:
        """Return artifact relpath for a specific rank.

        For world_size == 1 (or old manifests), returns the path as-is.
        For world_size > 1, replaces shard0ofN with shard{rank}ofN.
        """
        relpath = self.artifacts[module_name]
        if self.world_size <= 1:
            return relpath
        return relpath.replace(f"shard0of{self.world_size}", f"shard{rank}of{self.world_size}")

    def to_json(self) -> str:
        d = asdict(self)
        # `default=str` makes Path / dtype / etc. JSON-safe.
        return json.dumps(d, indent=2, sort_keys=True, default=str)

    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json())

    @staticmethod
    def load(path: str | Path) -> "RunManifest":
        path = Path(path)
        d = json.loads(path.read_text())
        return RunManifest(**d)
