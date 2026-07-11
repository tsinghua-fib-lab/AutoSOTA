from __future__ import annotations

import json
from dataclasses import dataclass
import math
from pathlib import Path

from taming_the_ito_lyon.config import Config, load_toml_config
from taming_the_ito_lyon.paper.tables_rough_vol import (
    extract_first_seed_payload,
    load_seed_payloads,
    model_pretty_name,
)


FROBENIUS_DISPLAY_SCALE: float = 1e2
RGE_DISPLAY_SCALE: float = 1.0


@dataclass
class So3Row:
    method: str
    inference_s: float | None
    frobenius_test: float | None
    rge: float | None
    inference_s_ci: float | None = None
    frobenius_test_ci: float | None = None
    rge_ci: float | None = None


def _load_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected dict JSON in {path}, got {type(payload).__name__}")
    return payload


def _expect_dict(data: dict[str, object], key: str) -> dict[str, object]:
    value = data.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Expected dict for '{key}', got {type(value).__name__}")
    return value


def _to_float(value: object, label: str) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        return float(value)
    raise ValueError(f"Expected float-compatible value for '{label}', got {value!r}")


def _mean_and_std(values: list[float]) -> tuple[float, float | None]:
    if len(values) == 0:
        raise ValueError("Cannot compute statistics for empty list.")
    mean = float(sum(values) / float(len(values)))
    if len(values) < 2:
        return mean, None
    variance = float(sum((v - mean) ** 2 for v in values) / float(len(values) - 1))
    sigma = float(math.sqrt(variance))
    return mean, sigma


def _resolve_metrics_path(run_dir: Path, preferred: str | None) -> Path:
    if preferred is not None:
        return run_dir / preferred
    test_metrics = run_dir / "test_metrics.json"
    if test_metrics.exists():
        return test_metrics
    return run_dir / "metrics.json"


def _resolve_seed_metrics_paths(run_dir: Path, seeds: list[int]) -> list[Path]:
    paths: list[Path] = []
    for seed in seeds:
        path = run_dir / f"test_metrics_seed_{seed}.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing metrics file at {path}")
        paths.append(path)
    return paths


def _load_config(run_dir: Path) -> Config | None:
    config_path = run_dir / "config.toml"
    if not config_path.exists():
        return None
    return load_toml_config(str(config_path))


def _extract_inference_s(metrics: dict[str, object]) -> float | None:
    timings = metrics.get("timings")
    if not isinstance(timings, dict):
        return None
    value = timings.get("inference_s")
    return float(value) if isinstance(value, (int, float)) else None


def _extract_frobenius_test(metrics: dict[str, object]) -> float | None:
    test_section = _expect_dict(metrics, "test")
    metric = _expect_dict(test_section, "metric")
    value = metric.get("value") if metric.get("name") == "frobenius" else None
    return float(value) if isinstance(value, (int, float)) else None


def _extract_rge(metrics: dict[str, object]) -> float | None:
    test_section = _expect_dict(metrics, "test")
    metric = _expect_dict(test_section, "metric")
    if metric.get("name") == "rge":
        value = metric.get("value")
        return float(value) if isinstance(value, (int, float)) else None
    results_dict = _expect_dict(test_section, "results_dict")
    extra_scalar = results_dict.get("extra_scalar_metrics")
    if not isinstance(extra_scalar, dict):
        return None
    value = extra_scalar.get("rge")
    return float(value) if isinstance(value, (int, float)) else None


def _format_pm(
    value: float | None, ci: float | None, scale: float, decimals: int
) -> str:
    if value is None:
        return ""
    scaled_value = float(value) * float(scale)
    if ci is None:
        return f"${scaled_value:.{decimals}f}$"
    scaled_ci = float(ci) * float(scale)
    return (
        f"${scaled_value:.{decimals}f} \\scriptstyle{{\\pm {scaled_ci:.{decimals}f}}}$"
    )


def _format_int(value: float | None) -> str:
    if value is None:
        return ""
    return f"{int(round(float(value)))}"


def _min_value(values: list[float | None]) -> float | None:
    filtered = [float(v) for v in values if v is not None]
    if len(filtered) == 0:
        return None
    return float(min(filtered))


def _bold_if_match(value: float | None, best: float | None, rendered: str) -> str:
    if value is None or best is None:
        return rendered
    if abs(float(value) - float(best)) <= 1e-12:
        if rendered.startswith("$") and rendered.endswith("$"):
            inner = rendered[1:-1]
            return f"$\\mathbf{{{inner}}}$"
        return f"\\textbf{{{rendered}}}"
    return rendered


def build_rows(
    run_dirs: list[Path],
    metrics_file: str | None,
    seeds: list[int] | None,
) -> list[So3Row]:
    rows: list[So3Row] = []
    for run_dir in run_dirs:
        config = _load_config(run_dir)

        if seeds is None:
            metrics_path = _resolve_metrics_path(run_dir, metrics_file)
            if not metrics_path.exists():
                raise FileNotFoundError(f"Missing metrics file at {metrics_path}")
            metrics = extract_first_seed_payload(_load_json(metrics_path))
            inference_s = _extract_inference_s(metrics)
            frob_test = _extract_frobenius_test(metrics)
            rge = _extract_rge(metrics)
            inference_ci = None
            frob_ci = None
            rge_ci = None
        else:
            metrics_payloads = load_seed_payloads(run_dir, seeds)
            inference_vals = [
                v
                for v in (_extract_inference_s(p) for p in metrics_payloads)
                if v is not None
            ]
            frob_vals = [
                v
                for v in (_extract_frobenius_test(p) for p in metrics_payloads)
                if v is not None
            ]
            rge_vals = [
                v for v in (_extract_rge(p) for p in metrics_payloads) if v is not None
            ]

            inference_s, inference_ci = (
                _mean_and_std([float(v) for v in inference_vals])
                if len(inference_vals) > 0
                else (None, None)
            )
            frob_test, frob_ci = (
                _mean_and_std([float(v) for v in frob_vals])
                if len(frob_vals) > 0
                else (None, None)
            )
            rge, rge_ci = (
                _mean_and_std([float(v) for v in rge_vals])
                if len(rge_vals) > 0
                else (None, None)
            )

            metrics = metrics_payloads[0]

        run_meta = metrics.get("run")
        run_name = None
        if isinstance(run_meta, dict):
            name_value = run_meta.get("name")
            if isinstance(name_value, str) and name_value.strip():
                run_name = name_value

        if run_name is None:
            model_meta = _expect_dict(metrics, "model")
            raw_model = model_meta.get("type")
            model_name = str(raw_model) if raw_model is not None else run_dir.name
            pretty_name = model_pretty_name(model_name, config)
        else:
            pretty_name = run_name

        rows.append(
            So3Row(
                method=pretty_name,
                inference_s=inference_s,
                frobenius_test=frob_test,
                rge=rge,
                inference_s_ci=inference_ci,
                frobenius_test_ci=frob_ci,
                rge_ci=rge_ci,
            )
        )
    return rows


def render_table(rows: list[So3Row]) -> str:
    rows = sorted(
        rows, key=lambda row: float("inf") if row.rge is None else float(row.rge)
    )
    best_frob = _min_value([row.frobenius_test for row in rows])
    best_rge = _min_value([row.rge for row in rows])

    lines: list[str] = [
        r"\begin{table}",
        r"\centering",
        r"\caption{SO(3) benchmark results (Frobenius error and RGE).}",
        r"\label{tab:so3_results}",
        r"\resizebox{\linewidth}{!}{",
        r"\begin{tblr}{",
        r"  colspec = {l c c c},",
        r"  hline{1,Z} = {-}{0.08em},",
        r"  hline{2} = {-}{},",
        r"}",
        r"\textbf{Method}",
        r"& \textbf{\shortstack{Inference\\Time (s)}}",
        r"& \textbf{\shortstack{Frobenius\\($\times 10^{-2}$)}}",
        r"& \textbf{RGE} \\",
    ]

    for row in rows:
        inference_cell = _format_pm(
            row.inference_s, row.inference_s_ci, scale=1.0, decimals=1
        )
        frob_rendered = _format_pm(
            row.frobenius_test,
            row.frobenius_test_ci,
            scale=FROBENIUS_DISPLAY_SCALE,
            decimals=2,
        )
        frob_cell = _bold_if_match(row.frobenius_test, best_frob, frob_rendered)
        rge_rendered = _format_pm(
            row.rge, row.rge_ci, scale=RGE_DISPLAY_SCALE, decimals=2
        )
        rge_cell = _bold_if_match(row.rge, best_rge, rge_rendered)
        lines.append(
            " & ".join([row.method, inference_cell, frob_cell, rge_cell]) + r" \\"
        )

    lines.extend([r"\end{tblr}", r"}", r"\end{table}"])
    return "\n".join(lines)


def render_so3_table(
    run_dirs: list[Path],
    metrics_file: str | None,
    seeds: list[int] | None,
) -> str:
    rows = build_rows(run_dirs=run_dirs, metrics_file=metrics_file, seeds=seeds)
    return render_table(rows=rows)
