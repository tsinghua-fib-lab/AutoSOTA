
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

DEFAULT_PROMPT_TEMPLATE = (
    "You are a helpful assistant. Please answer the following question clearly.\n\n"
    "Question:\n{question}\n\nAnswer:"
)


def _merge_dict(base: Dict[str, Any], overrides: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if overrides is None:
        return base
    merged = dict(base)
    for key, value in overrides.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = _merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


@dataclass
class ModelConfig:
    name: str
    provider: str = "hf"
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    prompt_template: Optional[str] = None
    system_prompt: Optional[str] = None
    use_chat_template: bool = True
    stop_sequences: List[str] = field(default_factory=list)
    generation_kwargs: Dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "ModelConfig":
        if "name" not in data:
            raise ValueError("Each model configuration must contain a 'name' field.")
        return ModelConfig(**data)

    @property
    def resolved_prompt_template(self) -> str:
        return self.prompt_template or DEFAULT_PROMPT_TEMPLATE


@dataclass
class JudgeConfig:
    type: str = "openai"
    model: str = "gpt-4o-mini"
    temperature: float = 0.0
    max_tokens: int = 5
    system_prompt: str = (
        "You are a strict evaluator. You must and can only output 'A' or 'B', and no other content is allowed."
    )
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    fallback_behavior: str = "error"  # "error" | "heuristic"

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "JudgeConfig":
        return JudgeConfig(**data)


@dataclass
class HuggingFaceConfig:
    device_map: str = "auto"
    dtype: str = "float16"
    quantization: Optional[str] = None  
    trust_remote_code: bool = False
    revision: Optional[str] = None
    max_memory: Optional[Dict[str, str]] = None
    cache_dir: Optional[str] = None
    low_cpu_mem_usage: bool = True

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "HuggingFaceConfig":
        return HuggingFaceConfig(**data)


@dataclass
class OutputConfig:
    dir: str = "real_results"
    save_interactions: bool = True
    save_serpant_trace: bool = True
    save_config_snapshot: bool = True
    save_csv: bool = True
    log_level: str = "INFO"

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "OutputConfig":
        return OutputConfig(**data)


@dataclass
class PerformanceConfig:
    batch_size: int = 1
    num_workers: int = 0
    prefetch_factor: Optional[int] = None
    pin_memory: bool = False

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "PerformanceConfig":
        return PerformanceConfig(**data)


@dataclass
class RealModeConfig:
    alpha: float = 0.1
    max_t: int = 200
    sampling_method: str = "tournament"
    max_tournament_samples: int = 400
    top_k: Optional[int] = None
    verbose: bool = True
    questions: List[str] = field(default_factory=list)
    question_file: Optional[str] = None
    question_shuffle: bool = True
    max_questions: Optional[int] = None
    models: List[ModelConfig] = field(default_factory=list)
    judge: JudgeConfig = field(default_factory=JudgeConfig)
    huggingface: HuggingFaceConfig = field(default_factory=HuggingFaceConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    use_covariates: bool = False
    theta_update_interval: int = 1

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "RealModeConfig":
        models_data = data.get("models", [])
        if not models_data:
            raise ValueError("Real mode requires at least one model configuration.")
        models = [ModelConfig.from_dict(item) for item in models_data]

        judge = JudgeConfig.from_dict(data.get("judge", {}))
        hf_cfg = HuggingFaceConfig.from_dict(data.get("huggingface", {}))
        output_cfg = OutputConfig.from_dict(data.get("output", {}))
        perf_cfg = PerformanceConfig.from_dict(data.get("performance", {}))

        questions_raw = data.get("questions", [])
        if isinstance(questions_raw, str):
            question_file = questions_raw
            questions = []
        else:
            questions = questions_raw
            question_file = data.get("question_file")
        
        if not questions and not question_file:
            raise ValueError("Please provide questions list or question_file.")

        return RealModeConfig(
            alpha=data.get("alpha", 0.1),
            max_t=data.get("max_t", 200),
            sampling_method=data.get("sampling_method", "tournament"),
            max_tournament_samples=data.get("max_tournament_samples", 400),
            top_k=data.get("top_k"),
            verbose=data.get("verbose", True),
            questions=questions,
            question_file=question_file,
            question_shuffle=data.get("question_shuffle", True),
            max_questions=data.get("max_questions"),
            models=models,
            judge=judge,
            huggingface=hf_cfg,
            output=output_cfg,
            performance=perf_cfg,
            use_covariates=data.get("use_covariates", False),
            theta_update_interval=data.get("theta_update_interval", 1),
        )

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["models"] = [asdict(model) for model in self.models]
        payload["judge"] = asdict(self.judge)
        payload["huggingface"] = asdict(self.huggingface)
        payload["output"] = asdict(self.output)
        payload["performance"] = asdict(self.performance)
        return payload


def _load_config_payload(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        if path.suffix.lower() in {".yml", ".yaml"}:
            return yaml.safe_load(f) or {}
        if path.suffix.lower() == ".json":
            return json.load(f)
        raise ValueError("Only YAML or JSON config files are supported.")


def load_real_mode_config(
    path: str,
    overrides: Optional[Dict[str, Any]] = None,
) -> RealModeConfig:
    config_path = Path(path).expanduser().resolve()
    base_payload = _load_config_payload(config_path)
    merged_payload = _merge_dict(base_payload, overrides)
    return RealModeConfig.from_dict(merged_payload)


