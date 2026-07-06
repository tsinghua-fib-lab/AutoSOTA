from gitbud.gitbud import inject_repo_into_sys_path
inject_repo_into_sys_path()

from dataclasses import dataclass, field, fields
import re

import globals as g
from data.synthetic import SyntheticDatasetConfig
from data.suitesparse.load import SuiteSparseDatasetConfig
from data.llm import LlmGradientsDatasetConfig, LlmWeightsDatasetConfig


@dataclass(frozen=True)
class MetricSpec:
    """Metric specification for E2E plotting."""

    name: str
    label: str
    higher_is_better: bool
    requires: list = field(default_factory=list)


@dataclass(frozen=True)
class PlotSpec:
    """Plot specification for a single dataset/task/metric."""

    name: str
    task: str
    metric: MetricSpec
    dataset: str
    dataset_name: str
    dataset_group: str | None
    output_name: str
    title: str
    d: int | None = None
    n: int | None = None


@dataclass(frozen=True)
class FigureConfig:
    """Config for camera-ready E2E figures."""

    use_latest_run: bool = True
    run_paths: list = field(default_factory=list)
    plots: list = field(default_factory=list)
    base_width_in: float = 6.8
    base_height_in: float = 3.2
    min_font_size: int = 10
    outlier_iqr_multiplier: float = 1.5
    filter_outliers: bool = False
    axis_quantile_lo: float = 0.01
    axis_quantile_hi: float = 0.99
    axis_pad_frac: float = 0.1
    ellipse_alpha: float = 0.25
    std_multiplier: float = 1.0
    method_label_fields: tuple = field(default_factory=tuple)
    method_label_bool_labels: dict = field(default_factory=dict)
    method_label_value_only_fields: tuple = field(default_factory=tuple)
    task_label_map: dict = field(default_factory=dict)
    include_legends: bool = True
    export_legend_only: bool = False
    annotate_speedup: bool = False
    speedup_k: int | None = None
    speedup_text_va: str = "bottom"
    use_auto_limits: bool = False
    speedup_text_offset: float = 0.0


METHOD_LABEL_ALWAYS_FIELDS = (
    "s",
    "kappa",
    "dtype",
    "use_csr",
    "return_contiguous",
    "use_fused",
)
METHOD_LABEL_OPTIONAL_FIELDS = (
    "block_size",
    "tc",
    "tr",
    "br",
    "d_block",
    "s_block",
    "scale",
)
METHOD_LABEL_BOOL_LABELS = {
    "use_csr": ("csr", "coo"),
    "return_contiguous": ("contig", "noncontig"),
    "use_fused": ("fused", "unfused"),
}
METHOD_LABEL_VALUE_ONLY_FIELDS = ("dtype",)

TASK_LABELS = {
    g.TASK_GRAM_APPROX: "Gram Matrix Approximation",
    g.TASK_RIDGE_REGRESSION: "Sketch-and-ridge-regression",
    g.TASK_SKETCH_SOLVE_LS: "Sketch+Solve",
    g.TASK_OSE_ERROR: "OSE Spectral Error",
    g.TASK_GRASS_LDS: "GraSS LDS",
}


def slugify(text):
    """Return a filesystem-safe slug from text."""
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", str(text)).strip("_").lower()
    return slug or "dataset"


def dataset_title(cfg):
    """Return a title string derived from a dataset config."""
    if isinstance(cfg, SyntheticDatasetConfig):
        if cfg.distribution == g.DIST_LOW_RANK:
            base = "Synthetic Low-Rank + noise"
        else:
            base = "Synthetic Gaussian"
        return f"{base} (d={cfg.d}, n={cfg.n})"
    if isinstance(cfg, SuiteSparseDatasetConfig):
        if cfg.name == "spal_004":
            base = "SuiteSparse - spal_004"
        else:
            base = cfg.name
        if cfg.auto_submatrix:
            shape = "auto-submatrix"
        else:
            shape = f"{cfg.submatrix_rows}x{cfg.submatrix_cols}"
        if shape == "auto-submatrix":
            return f"{base} ({shape})"
        return f"{base} (d={cfg.submatrix_rows}, n={cfg.submatrix_cols})"
    if isinstance(cfg, (LlmWeightsDatasetConfig, LlmGradientsDatasetConfig)):
        if isinstance(cfg, LlmWeightsDatasetConfig) and cfg.model_name == "gpt2-medium" and "weights" in cfg.name:
            if cfg.submatrix_rows is not None and cfg.submatrix_cols is not None:
                shape = f"{cfg.submatrix_rows}x{cfg.submatrix_cols}"
            else:
                shape = "full"
            if shape == "full":
                return "GPT2-medium stacked weights (full)"
            return f"GPT2-medium stacked weights (d={cfg.submatrix_rows}, n={cfg.submatrix_cols})"
        if cfg.model_name == g.LLM_MODEL_QWEN2_1P5B and "weights" in cfg.name:
            if cfg.submatrix_rows is not None and cfg.submatrix_cols is not None:
                shape = f"{cfg.submatrix_rows}x{cfg.submatrix_cols}"
            else:
                shape = "full"
            if shape == "full":
                return "Qwen2-1.5B stacked weights (full)"
            return f"Qwen2-1.5B stacked weights (d={cfg.submatrix_rows}, n={cfg.submatrix_cols})"
        if cfg.submatrix_rows is not None and cfg.submatrix_cols is not None:
            shape = f"{cfg.submatrix_rows}x{cfg.submatrix_cols}"
        else:
            shape = "full"
        if shape == "full":
            return f"{cfg.model_name} {cfg.name} (full)"
        return f"{cfg.model_name} {cfg.name} (d={cfg.submatrix_rows}, n={cfg.submatrix_cols})"
    raise ValueError(f"Unknown dataset config type: {type(cfg)}")


def _dataset_shape_suffix(cfg):
    """Return a shape suffix string for dataset config."""
    if hasattr(cfg, "submatrix_rows") and hasattr(cfg, "submatrix_cols"):
        rows = getattr(cfg, "submatrix_rows")
        cols = getattr(cfg, "submatrix_cols")
        if rows is not None and cols is not None:
            return f"{rows}x{cols}"
    if hasattr(cfg, "d") and hasattr(cfg, "n"):
        rows = getattr(cfg, "d")
        cols = getattr(cfg, "n")
        if rows is not None and cols is not None:
            return f"{rows}x{cols}"
    return "full"


def _dataset_slug(cfg, dataset_group):
    """Return a unique slug for dataset config."""
    shape = _dataset_shape_suffix(cfg)
    return slugify(f"{dataset_group or cfg.dataset}_{cfg.name}_{shape}")


def dataset_plot_spec(cfg, task, metric, output_prefix):
    """Return a PlotSpec for a dataset config."""
    if isinstance(cfg, SyntheticDatasetConfig):
        dataset_group = None
    elif isinstance(cfg, SuiteSparseDatasetConfig):
        dataset_group = cfg.group
    elif isinstance(cfg, (LlmWeightsDatasetConfig, LlmGradientsDatasetConfig)):
        dataset_group = cfg.model_name
    else:
        raise ValueError(f"Unknown dataset config type: {type(cfg)}")

    slug = _dataset_slug(cfg, dataset_group)
    d_val = None
    n_val = None
    if hasattr(cfg, "submatrix_rows") and hasattr(cfg, "submatrix_cols"):
        d_val = getattr(cfg, "submatrix_rows")
        n_val = getattr(cfg, "submatrix_cols")
    elif hasattr(cfg, "d") and hasattr(cfg, "n"):
        d_val = getattr(cfg, "d")
        n_val = getattr(cfg, "n")
    return PlotSpec(
        name=slug,
        task=task,
        metric=metric,
        dataset=cfg.dataset,
        dataset_name=cfg.name,
        dataset_group=dataset_group,
        output_name=f"{output_prefix}_{slug}_{metric.name}",
        title=dataset_title(cfg),
        d=d_val,
        n=n_val,
    )


def method_label_fields(bench_cfg):
    """Return label fields that are present in the benchmark config."""
    available = set()
    for method_cfg in bench_cfg.methods:
        available.update(field.name for field in fields(method_cfg))

    fields_out = []
    for name in METHOD_LABEL_ALWAYS_FIELDS:
        if name in available:
            fields_out.append(name)

    for name in METHOD_LABEL_OPTIONAL_FIELDS:
        if name not in available:
            continue
        values = {
            getattr(method_cfg, name)
            for method_cfg in bench_cfg.methods
            if hasattr(method_cfg, name)
        }
        values = {value for value in values if value is not None}
        if len(values) > 1:
            fields_out.append(name)

    return tuple(fields_out)


def build_config(bench_cfg, metric, task, output_prefix):
    """Return a FigureConfig for camera-ready plots."""
    plots = [dataset_plot_spec(cfg, task, metric, output_prefix) for cfg in bench_cfg.datasets]
    return FigureConfig(
        plots=plots,
        method_label_fields=(),
        method_label_bool_labels={},
        method_label_value_only_fields=(),
        task_label_map=TASK_LABELS,
    )
