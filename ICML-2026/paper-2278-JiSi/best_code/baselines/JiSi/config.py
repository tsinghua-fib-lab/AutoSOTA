"""
Configuration management for JiSi.

The open-source release keeps deployment-specific values in JSON/YAML files or
environment variables. No private endpoint, token, or filesystem path should be
hard-coded in source.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, List, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _is_path_like(value: Any) -> bool:
    return isinstance(value, str) and value.strip() not in {"", "none", "null"}


def _resolve_path(value: Any, config_dir: Path) -> Any:
    """Resolve user paths against the repo root, with config-dir fallback."""
    if not _is_path_like(value):
        return value

    path = Path(value).expanduser()
    if path.is_absolute():
        return str(path)

    config_relative = (config_dir / path).resolve()
    if config_relative.exists():
        return str(config_relative)

    return str((PROJECT_ROOT / path).resolve())


def _resolve_optional_path(value: Any, config_dir: Path) -> Any:
    if value in {None, "", "none", "null"}:
        return None
    return _resolve_path(value, config_dir)


@dataclass
class JiSiConfig:
    """Runtime configuration for JiSi routing and aggregation."""

    train_data_path: str
    test_data_path: str
    baseline_scores_path: str

    seed: int = 42
    max_router: int = 1
    top_k: int = 1

    max_workers: int = 4
    routing_batch_size: int = 1000
    max_tokens: int = 7500

    embedding_model: str = "text-embedding-3-large"
    embedding_base_url: str = "http://localhost:8000/v1"
    embedding_api_key: str = "EMPTY"
    embedding_config_path: Optional[str] = None

    api_config_path: Optional[str] = None
    cache_config: Optional[str] = None
    deepseek_tokenizer_path: str = "deepseek-ai/DeepSeek-V3"

    excluded_models: List[str] = field(default_factory=list)
    excluded_datasets: List[str] = field(default_factory=list)
    dataset_exclusion_mode: str = "hard"
    ood_datasets: List[str] = field(default_factory=list)

    mode: str = "router"
    rag_num: int = 400
    rag_thres: float = 0.95
    agg_N: int = 8
    agg_max_tokens: int = 32768
    sample_n: int = 3
    select_n: int = 5
    ppl_coef: float = 0.5
    agg_model: str = "Meta-Llama-3.3-70B-Instruct"
    result_dir: str = "results/jisi"
    agg_temperature: float = 0.7
    process_batch_size: int = 8
    weighted_score: bool = True

    dev_re_route_mode: str = "expert_query_agg_query"
    dev_length_score_order: int = 2
    dev_subset_p: float = 0.5
    dev_embed_sim_score_mode: str = "s2s"
    dev_length_score_coef: float = 0.4
    dev_query_score_coef: float = 0.0
    dev_agg_prompt: str = "normal"
    dev_force_agg_num: int = 3
    divide_t: float = 0.8
    cut_length: int = 13000

    config_file_path: Optional[str] = None

    def __post_init__(self) -> None:
        self.excluded_models = [m for m in self.excluded_models if m]
        self._validate_config()

    def _validate_config(self) -> None:
        required_paths = {
            "train_data_path": self.train_data_path,
            "test_data_path": self.test_data_path,
            "baseline_scores_path": self.baseline_scores_path,
        }
        for name, path in required_paths.items():
            if not path:
                raise ValueError(f"{name} is required")
            if not Path(path).exists():
                raise FileNotFoundError(f"{name} not found: {path}")

        optional_files = {
            "embedding_config_path": self.embedding_config_path,
            "api_config_path": self.api_config_path,
            "cache_config": self.cache_config,
        }
        for name, path in optional_files.items():
            if path and not Path(path).exists():
                raise FileNotFoundError(f"{name} not found: {path}")
        if self.max_router <= 0:
            raise ValueError(f"max_router must be positive, got {self.max_router}")
        if self.max_workers <= 0:
            raise ValueError(f"max_workers must be positive, got {self.max_workers}")
        if self.top_k <= 0:
            raise ValueError(f"top_k must be positive, got {self.top_k}")
        if self.max_tokens <= 0:
            raise ValueError(f"max_tokens must be positive, got {self.max_tokens}")
        if self.dataset_exclusion_mode not in {"soft", "hard"}:
            raise ValueError("dataset_exclusion_mode must be 'soft' or 'hard'")
        if self.routing_batch_size <= 0:
            raise ValueError(f"routing_batch_size must be positive, got {self.routing_batch_size}")
        if self.mode not in {"router", "aggregator"}:
            raise ValueError("mode must be one of: router, aggregator")
        if self.rag_num <= 0:
            raise ValueError(f"rag_num must be positive, got {self.rag_num}")
        if self.process_batch_size <= 0:
            raise ValueError(f"process_batch_size must be positive, got {self.process_batch_size}")
        if self.dev_force_agg_num <= 0:
            raise ValueError(f"dev_force_agg_num must be positive, got {self.dev_force_agg_num}")
        if not 0.0 < self.divide_t <= 1.0:
            raise ValueError("divide_t must be in the range (0.0, 1.0]")
        valid_re_route_modes = {
            "expert_response_agg_query",
            "expert_response_agg_response",
            "expert_query_agg_query",
            "expert_query_agg_response",
        }
        if self.dev_re_route_mode not in valid_re_route_modes:
            raise ValueError(
                "dev_re_route_mode must be one of: "
                + ", ".join(sorted(valid_re_route_modes))
            )

        if self.max_tokens > 8000:
            logging.warning(
                "max_tokens=%s may exceed some embedding endpoint limits; adjust the "
                "embedding_config_path context limit if needed.",
                self.max_tokens,
            )

    @classmethod
    def from_env(
        cls,
        train_data_path: str,
        test_data_path: str,
        baseline_scores_path: Optional[str] = None,
        **kwargs: Any,
    ) -> "JiSiConfig":
        """Create configuration from environment variables and CLI overrides."""
        config_dict: dict[str, Any] = {
            "train_data_path": train_data_path,
            "test_data_path": test_data_path,
            "baseline_scores_path": baseline_scores_path or os.getenv("BASELINE_SCORES_PATH"),
        }

        env_mappings = {
            "SEED": ("seed", int),
            "MAX_ROUTER": ("max_router", int),
            "TOP_K": ("top_k", int),
            "MAX_WORKERS": ("max_workers", int),
            "ROUTING_BATCH_SIZE": ("routing_batch_size", int),
            "MAX_TOKENS": ("max_tokens", int),
            "EMBEDDING_MODEL": ("embedding_model", str),
            "EMBEDDING_BASE_URL": ("embedding_base_url", str),
            "EMBEDDING_API_KEY": ("embedding_api_key", str),
            "EMBEDDING_CONFIG_PATH": ("embedding_config_path", str),
            "API_CONFIG_PATH": ("api_config_path", str),
            "CACHE_CONFIG": ("cache_config", str),
            "DEEPSEEK_TOKENIZER_PATH": ("deepseek_tokenizer_path", str),
            "EXCLUDED_MODELS": ("excluded_models", str),
            "EXCLUDED_DATASETS": ("excluded_datasets", str),
            "OOD_DATASETS": ("ood_datasets", str),
            "DATASET_EXCLUSION_MODE": ("dataset_exclusion_mode", str),
            "JISI_RESULT_DIR": ("result_dir", str),
        }

        for env_var, (attr_name, attr_type) in env_mappings.items():
            env_value = os.getenv(env_var)
            if env_value is None:
                continue
            try:
                if attr_name in {"excluded_models", "excluded_datasets", "ood_datasets"}:
                    config_dict[attr_name] = [item.strip() for item in env_value.split(",") if item.strip()]
                else:
                    config_dict[attr_name] = attr_type(env_value)
            except ValueError as exc:
                logging.warning("Invalid value for %s=%s: %s", env_var, env_value, exc)

        config_dict.update({k: v for k, v in kwargs.items() if v is not None})
        return cls(**config_dict)

    @classmethod
    def from_file(cls, config_file: str) -> "JiSiConfig":
        """Load configuration from a JSON file."""
        config_path = Path(config_file).expanduser().resolve()
        with open(config_path, "r", encoding="utf-8-sig") as f:
            config_dict = json.load(f)

        config_dict["config_file_path"] = str(config_path)
        config_dict = cls._resolve_config_paths(config_dict, config_path.parent)
        config_dict = cls._filter_known_fields(config_dict)
        return cls(**config_dict)

    @staticmethod
    def _filter_known_fields(config_dict: dict[str, Any]) -> dict[str, Any]:
        known = {field.name for field in fields(JiSiConfig)}
        unknown = sorted(set(config_dict) - known)
        for key in unknown:
            logging.warning("Ignoring unknown JiSi config field: %s", key)
        return {key: value for key, value in config_dict.items() if key in known}

    @staticmethod
    def _resolve_config_paths(config_dict: dict[str, Any], config_dir: Path) -> dict[str, Any]:
        resolved = dict(config_dict)
        repo_relative_files = [
            "train_data_path",
            "test_data_path",
            "baseline_scores_path",
            "embedding_config_path",
            "api_config_path",
            "cache_config",
        ]
        for key in repo_relative_files:
            if key in resolved:
                resolved[key] = _resolve_optional_path(resolved[key], config_dir)

        for key in ["result_dir"]:
            if key in resolved and _is_path_like(resolved[key]):
                resolved[key] = _resolve_path(resolved[key], config_dir)
        return resolved

    def save(self, config_file: str) -> None:
        """Save a redacted configuration snapshot."""
        safe_config = {
            key: value
            for key, value in self.to_dict().items()
            if "key" not in key.lower() and "secret" not in key.lower()
        }
        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(safe_config, f, indent=2, ensure_ascii=False)

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to a dictionary, redacting API keys."""
        output = {
            key: value
            for key, value in self.__dict__.items()
            if key != "embedding_api_key"
        }
        output["embedding_api_key"] = "<redacted>" if self.embedding_api_key else ""
        return output


def setup_logging(level: str = "INFO") -> None:
    """Set up standard logging and keep noisy optional loggers at the same level."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("jisi.log", encoding="utf-8"),
        ],
    )

    try:
        from loguru import logger
        import sys

        logger.remove()
        logger.add(sys.stderr, level=level.upper())
    except ImportError:
        pass
