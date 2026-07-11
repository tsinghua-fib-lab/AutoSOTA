"""
Configuration loading.

A run is parametrised entirely by a TOML file that declares the dataset
schema, the model metadata, the experiment hyperparameters, and the cache
location.  This keeps notebooks dataset-agnostic.
"""

from __future__ import annotations

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ============================================================
# ======================== DATACLASSES =======================
# ============================================================

@dataclass(frozen=True)
class ModelSpec:
    """Per-model metadata."""
    id: int
    name: str
    huggingface_id: str | None = None
    latex_tag: str | None = None
    size_b: float | None = None      # parameter count in billions


@dataclass(frozen=True)
class DatasetConfig:
    """Dataset schema and on-disk location."""
    name: str
    path: str
    embedding_column: str = "response_embeddings"
    label_column: str = "hallucination"
    prompt_column: str = "prompt_id"
    model_column: str = "model_id"
    # SOCRATES-specific: collapse 2020/2022 prompts onto a single prompt_id
    # axis (prompt_id += 100 for the second year).  False for CoQA bridge.
    unify_years: bool = False
    max_responses_per_prompt: int | None = None


@dataclass(frozen=True)
class ExperimentConfig:
    """Hyperparameters shared across analyses."""
    best_lambda: float = 1.2
    n_permutations: int = 100
    random_state: int = 42
    min_per_class: int = 5


@dataclass(frozen=True)
class CacheConfig:
    """Where intermediate and final artifacts are written."""
    root: str = "cache"
    fig_dir: str = "fig_dir"


@dataclass(frozen=True)
class Config:
    dataset: DatasetConfig
    experiment: ExperimentConfig
    cache: CacheConfig
    models: tuple[ModelSpec, ...]

    # ----- convenience accessors -----

    @property
    def model_ids(self) -> list[int]:
        return [m.id for m in self.models]

    @property
    def model_names(self) -> dict[int, str]:
        return {m.id: m.name for m in self.models}

    @property
    def model_latextags(self) -> dict[int, str]:
        return {m.id: (m.latex_tag or m.name) for m in self.models}

    def model_by_id(self, model_id: int) -> ModelSpec:
        for m in self.models:
            if m.id == model_id:
                return m
        raise KeyError(f"No model with id={model_id}")

    def model_order_by_size(self) -> list[int]:
        """Model ids sorted by ascending parameter count.

        Models with no declared size are pushed to the end in id order.
        """
        sized   = sorted([m for m in self.models if m.size_b is not None],
                         key=lambda m: m.size_b)
        unsized = sorted([m for m in self.models if m.size_b is None],
                         key=lambda m: m.id)
        return [m.id for m in sized + unsized]


# ============================================================
# ========================== LOADER ==========================
# ============================================================

def load_config(path: str | Path) -> Config:
    """Load a TOML config file into a :class:`Config` instance.

    The TOML file is expected to declare at least ``[dataset]`` and a list
    of ``[[models]]`` tables.  ``[experiment]`` and ``[cache]`` are
    optional and fall back to defaults.
    """
    path = Path(path)
    with path.open("rb") as f:
        raw: dict[str, Any] = tomllib.load(f)

    dataset    = DatasetConfig(**raw["dataset"])
    experiment = ExperimentConfig(**raw.get("experiment", {}))
    cache      = CacheConfig(**raw.get("cache", {}))

    models = tuple(ModelSpec(**m) for m in raw.get("models", []))

    return Config(
        dataset=dataset,
        experiment=experiment,
        cache=cache,
        models=models,
    )
