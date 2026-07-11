from __future__ import annotations

import json
from dataclasses import dataclass
import math
from pathlib import Path

from taming_the_ito_lyon.config import Config, load_toml_config
from taming_the_ito_lyon.config.config_options import ControlInterpolationType


MODEL_NAME_LOOKUP: dict[str, str] = {
    "nrde": r"\gls{nrde}",
    "ncde": r"\gls{ncde}",
    "bnrde": r"\gls{bnrde}",
    "gru": r"\gls{gru}",
}

DISPLAY_SCALE: float = 1e2


@dataclass
class RoughVolRow:
    method: str
    training_time_s: float | None
    ito_level2_mmd2: float | None
    ks_scores: list[float | None]
    ito_level2_mmd2_ci: float | None = None
    ks_scores_ci: list[float | None] | None = None
    ks_mean: float | None = None


def model_pretty_name(
    model_name: str,
    config: Config | None,
    name_lookup: dict[str, str] | None = None,
) -> str:
    lookup = MODEL_NAME_LOOKUP if name_lookup is None else name_lookup
    base = lookup.get(model_name, model_name)
    if model_name == "ncde" and config is not None:
        ncde_config = getattr(config, "ncde_config", None)
        if (
            ncde_config is not None
            and getattr(ncde_config, "control_interpolation", None)
            == ControlInterpolationType.HERMITE_CUBIC
        ):
            return f"{base}++"
    return base


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


def _expect_list(data: dict[str, object], key: str) -> list[object]:
    value = data.get(key)
    if not isinstance(value, list):
        raise ValueError(f"Expected list for '{key}', got {type(value).__name__}")
    return value


def _to_float(value: object, label: str) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        return float(value)
    raise ValueError(f"Expected float-compatible value for '{label}', got {value!r}")


def _extract_training_time(metrics: dict[str, object]) -> float | None:
    timings = metrics.get("timings")
    if not isinstance(timings, dict):
        return None
    training_s = timings.get("training_s")
    return float(training_s) if isinstance(training_s, (int, float)) else None


def _estimate_training_time_to_best(metrics: dict[str, object]) -> float | None:
    timings = metrics.get("timings")
    if not isinstance(timings, dict):
        return None
    time_to_best = timings.get("time_to_best_epoch_s")
    return float(time_to_best) if isinstance(time_to_best, (int, float)) else None


def _extract_results_dict(metrics: dict[str, object]) -> dict[str, object]:
    test_section = _expect_dict(metrics, "test")
    return _expect_dict(test_section, "results_dict")


def _extract_ks_scores(
    results_dict: dict[str, object], times: list[int]
) -> list[float | None]:
    raw_times = _expect_list(results_dict, "results_times")
    raw_scores = _expect_list(results_dict, "results")
    time_values: list[float] = [_to_float(t, "results_times") for t in raw_times]
    score_values: list[float] = [_to_float(s, "results") for s in raw_scores]
    ks_map = {int(t): float(s) for t, s in zip(time_values, score_values)}
    return [ks_map.get(int(t)) for t in times]


def _extract_ito_level2_mmd2(results_dict: dict[str, object]) -> float | None:
    extra_scalar = results_dict.get("extra_scalar_metrics")
    if not isinstance(extra_scalar, dict):
        return None
    value = extra_scalar.get("ito_level2_mmd2")
    return float(value) if isinstance(value, (int, float)) else None


def _format_int(value: float | None) -> str:
    if value is None:
        return ""
    return f"{int(round(value))}"


def _format_float(value: float | None, decimals: int = 2) -> str:
    if value is None:
        return ""
    scaled = float(value) * DISPLAY_SCALE
    return f"{scaled:.{decimals}f}"


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


def extract_first_seed_payload(metrics: dict[str, object]) -> dict[str, object]:
    """If metrics is a combined file with seed_metrics, return the lowest-key seed's payload."""
    seed_metrics_obj = metrics.get("seed_metrics")
    if not isinstance(seed_metrics_obj, dict) or not seed_metrics_obj:
        return metrics
    first_payload = next(
        (
            payload
            for _, payload in sorted(
                seed_metrics_obj.items(), key=lambda item: int(item[0])
            )
            if isinstance(payload, dict)
        ),
        None,
    )
    return first_payload if isinstance(first_payload, dict) else metrics


def load_seed_payloads(run_dir: Path, seeds: list[int]) -> list[dict[str, object]]:
    """Load per-seed metrics payloads, preferring a combined test_metrics.json if available."""
    combined_path = run_dir / "test_metrics.json"
    if combined_path.exists():
        combined = _load_json(combined_path)
        seed_metrics_obj = combined.get("seed_metrics")
        if isinstance(seed_metrics_obj, dict):
            maybe_payloads = [seed_metrics_obj.get(str(int(seed))) for seed in seeds]
            if all(isinstance(payload, dict) for payload in maybe_payloads):
                return [
                    payload for payload in maybe_payloads if isinstance(payload, dict)
                ]
    metrics_paths = _resolve_seed_metrics_paths(run_dir, seeds)
    return [_load_json(path) for path in metrics_paths]


def _load_config(run_dir: Path) -> Config | None:
    config_path = run_dir / "config.toml"
    if not config_path.exists():
        return None
    return load_toml_config(str(config_path))


def _mean_and_std(values: list[float]) -> tuple[float, float | None]:
    if len(values) == 0:
        raise ValueError("Cannot compute statistics for empty list.")
    mean = float(sum(values) / float(len(values)))
    if len(values) < 2:
        return mean, None
    variance = float(sum((v - mean) ** 2 for v in values) / float(len(values) - 1))
    sigma = float(math.sqrt(variance))
    return mean, sigma


def _aggregate_ks_scores(
    results_dicts: list[dict[str, object]], times: list[int]
) -> tuple[list[float | None], list[float | None]]:
    per_seed_scores: list[list[float | None]] = [
        _extract_ks_scores(results_dict, times) for results_dict in results_dicts
    ]
    means: list[float | None] = []
    cis: list[float | None] = []
    for idx in range(len(times)):
        values = [
            score
            for score in (row[idx] for row in per_seed_scores)
            if score is not None
        ]
        if len(values) == 0:
            means.append(None)
            cis.append(None)
            continue
        mean, std = _mean_and_std([float(v) for v in values])
        means.append(mean)
        cis.append(std)
    return means, cis


def _aggregate_scalar(
    results_dicts: list[dict[str, object]],
) -> tuple[float | None, float | None]:
    values: list[float] = []
    for results_dict in results_dicts:
        value = _extract_ito_level2_mmd2(results_dict)
        if value is not None:
            values.append(float(value))
    if len(values) == 0:
        return None, None
    mean, std = _mean_and_std(values)
    return mean, std


def build_rows(
    run_dirs: list[Path],
    times: list[int],
    metrics_file: str | None,
    seeds: list[int] | None,
) -> list[RoughVolRow]:
    rows: list[RoughVolRow] = []
    for run_dir in run_dirs:
        if seeds is None:
            metrics_path = _resolve_metrics_path(run_dir, metrics_file)
            if not metrics_path.exists():
                raise FileNotFoundError(f"Missing metrics file at {metrics_path}")
            metrics = extract_first_seed_payload(_load_json(metrics_path))
            results_dict = _extract_results_dict(metrics)
            ks_scores = _extract_ks_scores(results_dict, times)
            ito_level2_mmd2 = _extract_ito_level2_mmd2(results_dict)
            ito_level2_mmd2_ci = None
            ks_scores_ci = None
        else:
            metrics_payloads = load_seed_payloads(run_dir, seeds)
            results_dicts = [
                _extract_results_dict(payload) for payload in metrics_payloads
            ]
            ks_scores, ks_scores_ci = _aggregate_ks_scores(results_dicts, times)
            ito_level2_mmd2, ito_level2_mmd2_ci = _aggregate_scalar(results_dicts)
            metrics = metrics_payloads[0]

        training_metrics_path = run_dir / "metrics.json"
        training_metrics = (
            _load_json(training_metrics_path)
            if training_metrics_path.exists()
            else metrics
        )
        training_time = _estimate_training_time_to_best(training_metrics)

        config = _load_config(run_dir)
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
            RoughVolRow(
                method=pretty_name,
                training_time_s=training_time,
                ito_level2_mmd2=ito_level2_mmd2,
                ks_scores=ks_scores,
                ito_level2_mmd2_ci=ito_level2_mmd2_ci,
                ks_scores_ci=ks_scores_ci,
                ks_mean=_mean_ks(ks_scores),
            )
        )
    return rows


def _format_float_pm(value: float | None, ci: float | None, decimals: int = 2) -> str:
    if value is None:
        return ""
    scaled_value = float(value) * DISPLAY_SCALE
    if ci is None:
        return f"${scaled_value:.{decimals}f}$"
    scaled_ci = float(ci) * DISPLAY_SCALE
    return (
        f"${scaled_value:.{decimals}f} \\scriptstyle{{\\pm {scaled_ci:.{decimals}f}}}$"
    )


def _mean_ks(values: list[float | None]) -> float | None:
    filtered = [float(v) for v in values if v is not None]
    if len(filtered) == 0:
        return None
    return float(sum(filtered) / float(len(filtered)))


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


def render_table(rows: list[RoughVolRow], times: list[int]) -> str:
    rows = sorted(
        rows,
        key=lambda row: float("-inf") if row.ks_mean is None else row.ks_mean,
        reverse=True,
    )
    best_ito_level2_mmd2 = _min_value([row.ito_level2_mmd2 for row in rows])
    best_ks: list[float | None] = []
    for idx in range(len(times)):
        best_ks.append(_min_value([row.ks_scores[idx] for row in rows]))

    header_times = " & ".join([f"\\textbf{{{t}}}" for t in times])
    time_cols = len(times)
    last_col = 3 + time_cols
    ks_header = " & ".join(
        ["\\textbf{KS Score ($\\times 10^{-2}$)}"] + [""] * (time_cols - 1)
    )
    lines: list[str] = [
        r"\begin{table}",
        r"\centering",
        r"\caption{It\^o level-2 MMD$^2$ and \gls{ks} across time-marginals}",
        r"\label{tab:rough_vol_results}",
        r"\resizebox{\linewidth}{!}{",
        r"\begin{tblr}{",
        rf"  colspec = {{l c c *{{{time_cols}}}{{c}}}},",
        r"  cell{1}{1} = {r=2}{c},",
        r"  cell{1}{2} = {r=2}{c},",
        r"  cell{1}{3} = {r=2}{c},",
        rf"  cell{{1}}{{4}} = {{c={time_cols}}}{{c}},",
        r"  hline{1,Z} = {-}{0.08em},",
        rf"  hline{{2}} = {{4-{last_col}}}{{}},",
        rf"  hline{{3}} = {{1-{last_col}}}{{}},",
        r"}",
        r"\textbf{Method}",
        r"& \textbf{\shortstack{Training\\Time (s)}}",
        r"& \textbf{\shortstack{It\^o level-2\\MMD$^2$ ($\times 10^{-2}$)}}",
        rf"& {ks_header} \\",
        rf"& & & {header_times} \\",
    ]

    for row in rows:
        ks_cells: list[str] = []
        ks_ci_values = row.ks_scores_ci or [None] * len(row.ks_scores)
        for idx, (val, ci) in enumerate(zip(row.ks_scores, ks_ci_values)):
            rendered = _format_float_pm(val, ci)
            ks_cells.append(_bold_if_match(val, best_ks[idx], rendered))
        ito_cell = _bold_if_match(
            row.ito_level2_mmd2,
            best_ito_level2_mmd2,
            _format_float_pm(row.ito_level2_mmd2, row.ito_level2_mmd2_ci),
        )
        lines.append(
            " & ".join(
                [
                    row.method,
                    _format_int(row.training_time_s),
                    ito_cell,
                    " & ".join(ks_cells),
                ]
            )
            + r" \\"
        )

    lines.extend([r"\end{tblr}", r"}", r"\end{table}"])
    return "\n".join(lines)


def render_rough_vol_table(
    run_dirs: list[Path],
    times: list[int],
    metrics_file: str | None,
    seeds: list[int] | None,
) -> str:
    rows = build_rows(
        run_dirs=run_dirs,
        times=times,
        metrics_file=metrics_file,
        seeds=seeds,
    )
    return render_table(rows=rows, times=times)
