"""Helper utilities for config handling, output management, and metrics serialization."""

import json
from pathlib import Path

import numpy as np


def get_default_parameters() -> dict:
    """Return default experiment parameters."""
    return {
        "dataset": {
            "name": "xl-sum",
            "subset": "english",
            "num_samples": 8192,
            "prompt_tokens": 20,
            "completion_tokens": 200,
            "completion_tokens_diff": 10,
            "filters": {
                "max_doc_length": 2048,
                "id_prefix": None,
            },
            "test_size": 0.2,
        },
        "generation": {
            "model_name": "meta-llama/Llama-3.2-1B-Instruct",
            "temperature": 0.7,
            "top_p": 0.9,
            "use_chat_template": True,
            "batch_size": 8,
            "max_input_tokens": 1024,
        },
        "extraction": {
            "layers": None,
            "tokens": [-1],
            "mode": "mean",
            "use_prompt_for_activations": True,
        },
        "training": {
            "classifier": "lda",
            "standardize_activations": True,
            "compute_perplexity": True,
            "compute_mass_mean": True,
            "classifier_params": {
                "lda": {
                    "solver": "lsqr",
                    "shrinkage": "auto",
                },
            },
            "random_seed": 42,
            "bootstrap_B": 1000,
            "bootstrap_seed": None,
        },
        "paths": {
            "save_name": None,
        },
    }


def flatten_yaml_config(d: dict) -> dict:
    """Flatten nested YAML config into a single-level dictionary."""
    ds = d.get("dataset", {})
    gn = d.get("generation", {})
    ex = d.get("extraction", {})
    tr = d.get("training", {})
    pa = d.get("paths", {})
    fi = ds.get("filters", {}) or {}

    max_doc_len = fi.get("max_doc_length", None)
    id_prefix = fi.get("id_prefix", None)
    if id_prefix is not None:
        id_prefix = str(id_prefix).strip()
        if id_prefix in ("", "None"):
            id_prefix = None

    raw_layers = ex.get("layers", [-3, -2, -1])
    if raw_layers is None or (isinstance(raw_layers, str) and str(raw_layers).strip().lower() == "none"):
        layers_to_analyze = None
    else:
        layers_to_analyze = list(raw_layers)

    flat = {
        "NUM_SAMPLES": int(ds.get("num_samples", 4096)),
        "DATASET_NAME": str(ds.get("name", "xsum")),
        "DATASET_SUBSET": None if ds.get("subset", None) in (None, "", "None") else str(ds.get("subset")),
        "MODEL_NAME": str(gn.get("model_name", "meta-llama/Llama-3.2-1B-Instruct")),
        "TEST_SIZE": float(ds.get("test_size", 0.2)),
        "USE_CHAT_TEMPLATE": bool(gn.get("use_chat_template", True)),
        "LAYERS_TO_ANALYZE": layers_to_analyze,
        "TOKENS_TO_ANALYZE": list(ex.get("tokens", [-1])),
        "EXTRACTION_MODE": str(ex.get("mode", "mean")),
        "USE_PROMPT_FOR_ACTIVATIONS": bool(ex.get("use_prompt_for_activations", True)),
        "TEMPERATURE": float(gn.get("temperature", 0.8)),
        "TOP_P": float(gn.get("top_p", 1.0)),
        "TARGET_PROMPT_TOKENS": int(ds.get("prompt_tokens", 20)),
        "TARGET_COMPLETION_TOKENS": int(ds.get("completion_tokens", 200)),
        "TARGET_COMPLETION_TOKENS_DIFF": int(ds.get("completion_tokens_diff", 10)),
        "STANDARDIZE_ACTIVATIONS": bool(tr.get("standardize_activations", True)),
        "COMPUTE_PERPLEXITY": bool(tr.get("compute_perplexity", True)),
        "COMPUTE_MASS_MEAN": bool(tr.get("compute_mass_mean", True)),
        "BATCH_SIZE": int(gn.get("batch_size", 16)),
        "MAX_INPUT_TOKENS": int(gn.get("max_input_tokens")) if gn.get("max_input_tokens") is not None else None,
        "CLASSIFIER_TYPE": str(tr.get("classifier", "lda")),
        "CLASSIFIER_PARAMS": tr.get("classifier_params", {}),
        "MAX_DOC_LENGTH": int(max_doc_len) if max_doc_len is not None else None,
        "ID_PREFIX_FILTER": id_prefix,
        "RANDOM_SEED": int(tr.get("random_seed", 42)),
        "BOOTSTRAP_B": int(tr.get("bootstrap_B", 1000)),
        "BOOTSTRAP_SEED": int(tr.get("bootstrap_seed")) if tr.get("bootstrap_seed") is not None else None,
        "SAVE_NAME": pa.get("save_name", None),
    }
    return flat


def build_base_name(config: dict) -> str:
    """Build a descriptive run directory name."""
    dataset_name = config["DATASET_NAME"]
    subset = config.get("DATASET_SUBSET")
    model_short = str(config["MODEL_NAME"]).rsplit("/", 1)[-1]
    num_samples = config["NUM_SAMPLES"]
    extraction_mode = config["EXTRACTION_MODE"]
    temperature = config["TEMPERATURE"]
    top_p = config.get("TOP_P", 1.0)

    if dataset_name == "c4":
        ds_part = "c4"
    elif dataset_name == "xsum":
        ds_part = "xsum"
    elif dataset_name == "xl-sum":
        ds_part = "xlsum"
        id_prefix = config.get("ID_PREFIX_FILTER")
        if id_prefix:
            ds_part += f"-{id_prefix}"
    else:
        raise ValueError(f"Invalid dataset name: {dataset_name}")

    if subset:
        ds_part += f"-{subset}"

    parts = [
        ds_part,
        model_short,
        f"ns{num_samples}",
        f"em-{extraction_mode}",
        f"t{temperature}",
    ]

    if top_p != 1.0:
        parts.append(f"tp{top_p}")

    if dataset_name == "c4":
        parts.insert(4, f"pt{config['TARGET_PROMPT_TOKENS']}")
        parts.insert(5, f"ct{config['TARGET_COMPLETION_TOKENS']}")

    classifier_type = config.get("CLASSIFIER_TYPE", "").lower()
    if classifier_type == "lda":
        lda_params = config.get("CLASSIFIER_PARAMS", {}).get("lda", {})
        shrinkage = lda_params.get("shrinkage")
        if shrinkage is not None and shrinkage is not False:
            parts.append("shrinkage")

    return "_".join(parts)


def resolve_unique_base_name(directory: Path, base: str) -> str:
    """Generate a unique directory name by appending an index if needed."""
    run_dir = directory / base
    if not run_dir.exists():
        return base
    idx = 1
    while True:
        candidate = f"{base}_{idx}"
        if not (directory / candidate).exists():
            return candidate
        idx += 1


def write_eval(out_dir: Path, payload: dict) -> None:
    """Write evaluation results to JSON."""
    path = out_dir / "evaluation_results.json"
    txt = json.dumps(payload, ensure_ascii=False, indent=2)
    path.write_text(txt, encoding="utf-8")


def convert_metrics_for_json(metrics: dict) -> dict:
    """Convert metrics dict values to JSON-serializable formats."""
    return {
        k: np.asarray(v).astype(int).tolist() if k == "confusion_matrix" else v
        for k, v in metrics.items()
    }


