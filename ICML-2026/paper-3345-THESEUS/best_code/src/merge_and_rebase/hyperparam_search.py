from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import product
from typing import Any, Literal

import torch

TargetKind = Literal["alpha", "method_param"]
ValueKind = Literal["float", "int", "discrete"]


@dataclass(frozen=True)
class SearchCandidate:
    alpha: float
    method_params: dict[str, Any]
    values: dict[str, Any]
    stage: int
    batch_index: int
    candidate_index: int


@dataclass(frozen=True)
class SearchEvaluation:
    candidate: SearchCandidate
    score: float
    avg_acc: float
    avg_norm_acc: float
    per_task_acc: list[float]
    per_task_norm_acc: list[float]


@dataclass(frozen=True)
class SearchDimension:
    name: str
    target: TargetKind
    kind: ValueKind
    values: tuple[Any, ...] | None = None
    min_value: float | None = None
    max_value: float | None = None
    step: float | None = None

    def is_variable(self) -> bool:
        if self.kind == "discrete":
            assert self.values is not None
            return len(self.values) > 1
        assert self.min_value is not None and self.max_value is not None
        if math.isclose(float(self.min_value), float(self.max_value), rel_tol=0.0, abs_tol=1e-12):
            return False
        seq = self.sequential_values()
        return len(seq) > 1

    def default_value(self) -> Any:
        if self.kind == "discrete":
            assert self.values is not None and self.values
            return self.values[0]
        assert self.min_value is not None
        return self._coerce(self.min_value)

    def sequential_values(self) -> list[Any]:
        if self.kind == "discrete":
            assert self.values is not None
            return list(self.values)
        if self.min_value is None or self.max_value is None:
            raise ValueError(f"Search dimension '{self.name}' requires min/max bounds.")
        if self.step is None or self.step <= 0:
            if math.isclose(float(self.min_value), float(self.max_value), rel_tol=0.0, abs_tol=1e-12):
                return [self._coerce(self.min_value)]
            raise ValueError(f"Sequential search dimension '{self.name}' requires step > 0.")
        out: list[Any] = []
        cur = float(self.min_value)
        eps = max(1e-12, abs(float(self.step)) * 1e-9)
        while cur <= float(self.max_value) + eps:
            out.append(self._coerce(cur))
            cur += float(self.step)
        if not out:
            out.append(self._coerce(self.min_value))
        return _unique_preserve_order(out)

    def sample(self, u01: float, *, domain: SearchDomain | None = None) -> Any:
        if self.kind == "discrete":
            values = list(domain.values if domain is not None and domain.values is not None else (self.values or ()))
            if not values:
                raise ValueError(f"Discrete dimension '{self.name}' has no values.")
            if len(values) == 1:
                return values[0]
            idx = min(int(math.floor(float(u01) * len(values))), len(values) - 1)
            return values[idx]
        lo = float(domain.min_value if domain is not None and domain.min_value is not None else self.min_value)
        hi = float(domain.max_value if domain is not None and domain.max_value is not None else self.max_value)
        if hi < lo:
            lo, hi = hi, lo
        raw = lo if math.isclose(lo, hi, rel_tol=0.0, abs_tol=1e-12) else lo + float(u01) * (hi - lo)
        if self.step is not None and self.step > 0:
            raw = lo + round((raw - lo) / float(self.step)) * float(self.step)
            raw = min(max(raw, lo), hi)
        return self._coerce(raw)

    def refine_domain(self, *, current: SearchDomain, best_value: Any, refine_factor: float) -> SearchDomain:
        if self.kind == "discrete":
            values = list(current.values if current.values is not None else (self.values or ()))
            if len(values) <= 2:
                return SearchDomain(values=tuple(values))
            if not _all_numeric(values):
                return SearchDomain(values=(best_value,))
            ordered = sorted(values, key=float)
            best_idx = min(range(len(ordered)), key=lambda idx: abs(float(ordered[idx]) - float(best_value)))
            window = max(2, int(math.ceil(len(ordered) * float(refine_factor))))
            if window >= len(ordered):
                return SearchDomain(values=tuple(ordered))
            half = window // 2
            start = max(0, min(best_idx - half, len(ordered) - window))
            end = start + window
            return SearchDomain(values=tuple(ordered[start:end]))
        lo = float(current.min_value if current.min_value is not None else self.min_value)
        hi = float(current.max_value if current.max_value is not None else self.max_value)
        base_lo = float(self.min_value)
        base_hi = float(self.max_value)
        if math.isclose(lo, hi, rel_tol=0.0, abs_tol=1e-12):
            return SearchDomain(min_value=lo, max_value=hi)
        span = max(0.0, hi - lo)
        new_span = span * float(refine_factor)
        center = float(best_value)
        new_lo = max(base_lo, center - (new_span / 2.0))
        new_hi = min(base_hi, center + (new_span / 2.0))
        if self.step is not None and self.step > 0:
            new_lo = base_lo + round((new_lo - base_lo) / float(self.step)) * float(self.step)
            new_hi = base_lo + round((new_hi - base_lo) / float(self.step)) * float(self.step)
            if new_hi < new_lo:
                new_lo, new_hi = new_hi, new_lo
            if math.isclose(new_lo, new_hi, rel_tol=0.0, abs_tol=max(1e-12, float(self.step) * 1e-9)):
                next_hi = min(base_hi, new_lo + float(self.step))
                new_hi = max(new_hi, next_hi)
        return SearchDomain(min_value=new_lo, max_value=new_hi)

    def initial_domain(self) -> SearchDomain:
        if self.kind == "discrete":
            return SearchDomain(values=self.values)
        return SearchDomain(min_value=self.min_value, max_value=self.max_value)

    def _coerce(self, value: Any) -> Any:
        if self.kind == "int":
            return int(round(float(value)))
        if self.kind == "float":
            return float(value)
        return value


@dataclass(frozen=True)
class SearchDomain:
    values: tuple[Any, ...] | None = None
    min_value: float | None = None
    max_value: float | None = None


class SearchPlanner:
    def next_batch(self) -> list[SearchCandidate] | None:
        raise NotImplementedError

    def observe(self, results: Sequence[SearchEvaluation]) -> None:
        raise NotImplementedError

    def is_multi_param(self) -> bool:
        raise NotImplementedError

    def search_summary(self) -> dict[str, Any]:
        raise NotImplementedError


class SequentialSearchPlanner(SearchPlanner):
    def __init__(
        self,
        *,
        base_method_params: Mapping[str, Any],
        method_dims: Sequence[SearchDimension],
        alpha_values: Sequence[float],
    ) -> None:
        self._base_method_params = dict(base_method_params)
        self._method_dims = list(method_dims)
        self._alpha_values = [float(v) for v in alpha_values]
        self._batches = self._build_batches()
        self._next_batch_idx = 0
        self._observed: list[SearchEvaluation] = []

    def _build_batches(self) -> list[list[SearchCandidate]]:
        dims_for_product = [dim.sequential_values() for dim in self._method_dims]
        combos = list(product(*dims_for_product)) if dims_for_product else [tuple()]
        batches: list[list[SearchCandidate]] = []
        candidate_index = 0
        for batch_index, combo in enumerate(combos):
            method_overrides = {
                dim.name: combo[idx]
                for idx, dim in enumerate(self._method_dims)
            }
            method_params = dict(self._base_method_params)
            method_params.update(method_overrides)
            batch: list[SearchCandidate] = []
            for alpha in self._alpha_values:
                values = {"alpha": float(alpha)}
                values.update(method_overrides)
                batch.append(
                    SearchCandidate(
                        alpha=float(alpha),
                        method_params=dict(method_params),
                        values=values,
                        stage=0,
                        batch_index=batch_index,
                        candidate_index=candidate_index,
                    )
                )
                candidate_index += 1
            batches.append(batch)
        return batches

    def next_batch(self) -> list[SearchCandidate] | None:
        if self._next_batch_idx >= len(self._batches):
            return None
        batch = self._batches[self._next_batch_idx]
        self._next_batch_idx += 1
        return batch

    def observe(self, results: Sequence[SearchEvaluation]) -> None:
        self._observed.extend(results)

    def is_multi_param(self) -> bool:
        return any(dim.is_variable() for dim in self._method_dims)

    def search_summary(self) -> dict[str, Any]:
        return {
            "strategy": "sequential",
            "num_batches": len(self._batches),
            "num_candidates": sum(len(batch) for batch in self._batches),
            "method_dimensions": [dim.name for dim in self._method_dims if dim.is_variable()],
            "alpha_values": [float(v) for v in self._alpha_values],
        }


class SobolSearchPlanner(SearchPlanner):
    def __init__(
        self,
        *,
        base_method_params: Mapping[str, Any],
        alpha_default: float,
        variable_dims: Sequence[SearchDimension],
        num_samples: int,
        refinement_steps: int,
        refine_factor: float,
        refinement_samples: int | None,
        seed: int,
    ) -> None:
        self._base_method_params = dict(base_method_params)
        self._alpha_default = float(alpha_default)
        self._variable_dims = list(variable_dims)
        self._num_samples = int(num_samples)
        self._refinement_steps = int(refinement_steps)
        self._refine_factor = float(refine_factor)
        self._refinement_samples = int(refinement_samples) if refinement_samples is not None else None
        self._seed = int(seed)
        self._stage = 0
        self._done = False
        self._domains = {dim.name: dim.initial_domain() for dim in self._variable_dims}
        self._observed: list[SearchEvaluation] = []
        self._current_batch: list[SearchCandidate] | None = None
        self._candidate_counter = 0

    def next_batch(self) -> list[SearchCandidate] | None:
        if self._done:
            return None
        if not self._variable_dims:
            self._done = True
            return [
                SearchCandidate(
                    alpha=float(self._alpha_default),
                    method_params=dict(self._base_method_params),
                    values={"alpha": float(self._alpha_default)},
                    stage=0,
                    batch_index=0,
                    candidate_index=0,
                )
            ]

        if self._current_batch is not None:
            return self._current_batch

        num_samples = self._num_samples if self._stage == 0 else (self._refinement_samples or self._num_samples)
        engine = torch.quasirandom.SobolEngine(
            dimension=len(self._variable_dims),
            scramble=True,
            seed=self._seed + self._stage,
        )
        points = engine.draw(num_samples).tolist()

        batch: list[SearchCandidate] = []
        seen: set[str] = set()
        for row in points:
            sampled: dict[str, Any] = {}
            for idx, dim in enumerate(self._variable_dims):
                sampled[dim.name] = dim.sample(float(row[idx]), domain=self._domains[dim.name])
            key = json.dumps(_json_safe(sampled), sort_keys=True)
            if key in seen:
                continue
            seen.add(key)
            method_params = dict(self._base_method_params)
            alpha = float(self._alpha_default)
            values = dict(sampled)
            for dim in self._variable_dims:
                value = sampled[dim.name]
                if dim.target == "alpha":
                    alpha = float(value)
                else:
                    method_params[dim.name] = value
            values["alpha"] = alpha
            batch.append(
                SearchCandidate(
                    alpha=alpha,
                    method_params=method_params,
                    values=values,
                    stage=self._stage,
                    batch_index=self._stage,
                    candidate_index=self._candidate_counter,
                )
            )
            self._candidate_counter += 1

        if not batch:
            batch = [
                SearchCandidate(
                    alpha=float(self._alpha_default),
                    method_params=dict(self._base_method_params),
                    values={"alpha": float(self._alpha_default)},
                    stage=self._stage,
                    batch_index=self._stage,
                    candidate_index=self._candidate_counter,
                )
            ]
            self._candidate_counter += 1
        self._current_batch = batch
        return batch

    def observe(self, results: Sequence[SearchEvaluation]) -> None:
        if not results:
            self._done = True
            return
        self._observed.extend(results)
        if self._stage >= self._refinement_steps:
            self._done = True
            self._current_batch = None
            return
        best = max(results, key=lambda item: float(item.score))
        next_domains: dict[str, SearchDomain] = {}
        for dim in self._variable_dims:
            next_domains[dim.name] = dim.refine_domain(
                current=self._domains[dim.name],
                best_value=best.candidate.values[dim.name],
                refine_factor=self._refine_factor,
            )
        self._domains = next_domains
        self._stage += 1
        self._current_batch = None

    def is_multi_param(self) -> bool:
        return any(dim.target == "method_param" and dim.is_variable() for dim in self._variable_dims)

    def search_summary(self) -> dict[str, Any]:
        return {
            "strategy": "sobol",
            "num_samples": self._num_samples,
            "refinement_steps": self._refinement_steps,
            "refine_factor": self._refine_factor,
            "refinement_samples": self._refinement_samples,
            "dimensions": [dim.name for dim in self._variable_dims],
        }


def build_search_planner(
    *,
    cfg: Mapping[str, Any],
    base_method_params: Mapping[str, Any],
) -> SearchPlanner:
    search_cfg = cfg.get("hyperparam_search", None)
    alpha_dim = _parse_alpha_dimension(cfg=cfg, search_cfg=search_cfg)
    alpha_values = alpha_dim.sequential_values()

    method_dims = _parse_method_dims(search_cfg=search_cfg, base_method_params=base_method_params)
    if not search_cfg:
        return SequentialSearchPlanner(
            base_method_params=base_method_params,
            method_dims=method_dims,
            alpha_values=alpha_values,
        )

    if not isinstance(search_cfg, Mapping):
        raise ValueError("hyperparam_search must be a dict when provided.")

    strategy = str(search_cfg.get("strategy", "sequential")).strip().lower()
    if strategy == "sequential":
        return SequentialSearchPlanner(
            base_method_params=base_method_params,
            method_dims=method_dims,
            alpha_values=alpha_values,
        )
    if strategy != "sobol":
        raise ValueError("hyperparam_search.strategy must be one of: sequential, sobol")

    variable_dims = [dim for dim in [alpha_dim, *method_dims] if dim.is_variable()]
    num_samples = int(search_cfg.get("num_samples", 16))
    if num_samples <= 0:
        raise ValueError("hyperparam_search.num_samples must be > 0.")
    refine_factor = float(search_cfg.get("refine_factor", 0.5))
    if not (0.0 < refine_factor <= 1.0):
        raise ValueError("hyperparam_search.refine_factor must be in (0, 1].")
    refinement_steps_raw = search_cfg.get("refinement_steps", None)
    if refinement_steps_raw is None:
        refinement_steps = infer_sobol_refinement_steps(num_samples=num_samples, num_dims=max(1, len(variable_dims)))
    else:
        refinement_steps = int(refinement_steps_raw)
    if refinement_steps < 0:
        raise ValueError("hyperparam_search.refinement_steps must be >= 0.")
    refinement_samples = search_cfg.get("refinement_samples", None)
    if refinement_samples is not None and int(refinement_samples) <= 0:
        raise ValueError("hyperparam_search.refinement_samples must be > 0.")

    return SobolSearchPlanner(
        base_method_params=base_method_params,
        alpha_default=float(alpha_dim.default_value()),
        variable_dims=variable_dims,
        num_samples=num_samples,
        refinement_steps=refinement_steps,
        refine_factor=refine_factor,
        refinement_samples=(int(refinement_samples) if refinement_samples is not None else None),
        seed=int(cfg.get("seed", 0)),
    )


def infer_sobol_refinement_steps(*, num_samples: int, num_dims: int) -> int:
    if num_samples <= 0:
        raise ValueError("num_samples must be > 0.")
    linear_resolution = float(num_samples) ** (1.0 / max(1, int(num_dims)))
    if linear_resolution < 3.0:
        return 2
    if linear_resolution < 6.0:
        return 1
    return 0


def describe_candidate(candidate: SearchCandidate) -> str:
    extras = ", ".join(
        f"{name}={_format_candidate_value(value)}"
        for name, value in candidate.values.items()
        if name != "alpha"
    )
    if extras:
        return f"alpha={candidate.alpha:.3f}, {extras}"
    return f"alpha={candidate.alpha:.3f}"


def summarize_search_results(results: Sequence[SearchEvaluation]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for result in results:
        out.append(
            {
                "stage": int(result.candidate.stage),
                "batch_index": int(result.candidate.batch_index),
                "candidate_index": int(result.candidate.candidate_index),
                "alpha": float(result.candidate.alpha),
                "method_params": _json_safe(result.candidate.method_params),
                "values": _json_safe(result.candidate.values),
                "score": float(result.score),
                "avg_acc": float(result.avg_acc),
                "avg_norm_acc": float(result.avg_norm_acc),
                "per_task_acc": [float(v) for v in result.per_task_acc],
                "per_task_norm_acc": [float(v) for v in result.per_task_norm_acc],
            }
        )
    return out


def _parse_alpha_dimension(*, cfg: Mapping[str, Any], search_cfg: Any) -> SearchDimension:
    alpha_spec = search_cfg.get("alpha", None) if isinstance(search_cfg, Mapping) else None
    if alpha_spec is None:
        if bool(cfg.get("alpha_search", False)):
            a_min = float(cfg.get("alpha_min", 0.0))
            a_max = float(cfg.get("alpha_max", 2.0))
            a_step = float(cfg.get("alpha_step", 0.1))
            if a_step <= 0:
                raise ValueError("alpha_step must be > 0.")
            return SearchDimension(
                name="alpha",
                target="alpha",
                kind="float",
                min_value=a_min,
                max_value=a_max,
                step=a_step,
            )
        return SearchDimension(
            name="alpha",
            target="alpha",
            kind="discrete",
            values=(float(cfg.get("alpha", 1.0)),),
        )
    return _parse_dimension(name="alpha", target="alpha", raw=alpha_spec)


def _parse_method_dims(
    *,
    search_cfg: Any,
    base_method_params: Mapping[str, Any],
) -> list[SearchDimension]:
    method_dims: list[SearchDimension] = []
    if not isinstance(search_cfg, Mapping):
        return method_dims
    raw_method_params = search_cfg.get("method_params", {})
    if raw_method_params is None:
        return method_dims
    if not isinstance(raw_method_params, Mapping):
        raise ValueError("hyperparam_search.method_params must be a dict when provided.")
    for name, raw in raw_method_params.items():
        dim = _parse_dimension(name=str(name), target="method_param", raw=raw)
        if dim.kind == "discrete" and len(dim.sequential_values()) == 1 and name in base_method_params:
            continue
        method_dims.append(dim)
    return method_dims


def _parse_dimension(*, name: str, target: TargetKind, raw: Any) -> SearchDimension:
    if isinstance(raw, Mapping):
        if "values" in raw:
            values = raw.get("values", None)
            if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
                raise ValueError(f"Search dimension '{name}' values must be a sequence.")
            values_list = list(values)
            if not values_list:
                raise ValueError(f"Search dimension '{name}' values must be non-empty.")
            return SearchDimension(name=name, target=target, kind="discrete", values=tuple(values_list))
        if "min" not in raw or "max" not in raw:
            raise ValueError(f"Search dimension '{name}' requires either values or min/max.")
        value_type = str(raw.get("type", "float")).strip().lower()
        if value_type not in {"float", "int"}:
            raise ValueError(f"Search dimension '{name}' type must be float or int.")
        return SearchDimension(
            name=name,
            target=target,
            kind="int" if value_type == "int" else "float",
            min_value=float(raw["min"]),
            max_value=float(raw["max"]),
            step=(None if raw.get("step", None) is None else float(raw["step"])),
        )
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        values = list(raw)
        if not values:
            raise ValueError(f"Search dimension '{name}' values must be non-empty.")
        return SearchDimension(name=name, target=target, kind="discrete", values=tuple(values))
    return SearchDimension(name=name, target=target, kind="discrete", values=(raw,))


def _all_numeric(values: Sequence[Any]) -> bool:
    for value in values:
        try:
            float(value)
        except (TypeError, ValueError):
            return False
    return True


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    return repr(value)


def _format_candidate_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _unique_preserve_order(values: Sequence[Any]) -> list[Any]:
    out: list[Any] = []
    seen: set[str] = set()
    for value in values:
        key = json.dumps(_json_safe(value), sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        out.append(value)
    return out
