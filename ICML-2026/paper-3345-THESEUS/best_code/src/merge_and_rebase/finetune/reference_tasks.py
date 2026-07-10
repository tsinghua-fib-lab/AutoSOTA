from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from merge_and_rebase.utils.helpers import parse_csv

from ..eval.datasets.vision8_14_20 import SUITES, VISION_SUPPORTED_TASKS


@dataclass(frozen=True)
class ReferenceTaskResolutionContext:
    training_tasks: tuple[str, ...]
    suite: str | None = None
    cli_reference_suite: str | None = None
    cli_reference_datasets: tuple[str, ...] | None = None


_DATASET_AWARE_REGULARIZERS = {"kfac_ggn", "ekfac_ggn"}


def parse_reference_datasets(raw: Any, *, field_name: str) -> list[str] | None:
    if raw is None:
        return None
    if isinstance(raw, str):
        values = parse_csv(raw)
        return values or None
    if isinstance(raw, list) and all(isinstance(x, str) for x in raw):
        return list(raw)
    if isinstance(raw, tuple) and all(isinstance(x, str) for x in raw):
        return list(raw)
    raise ValueError(f"{field_name} must be a comma-separated string or a list of strings.")


def validate_vision_tasks(tasks: list[str], *, field_name: str) -> list[str]:
    unknown = [task for task in tasks if task not in VISION_SUPPORTED_TASKS]
    if unknown:
        raise ValueError(f"Unknown tasks in {field_name}: {unknown}. Supported: {VISION_SUPPORTED_TASKS}")
    return list(tasks)


def build_reference_task_resolution_context(
    *,
    training_tasks: list[str],
    suite: str | None = None,
    cli_reference_suite: str | None = None,
    cli_reference_datasets: Any = None,
) -> ReferenceTaskResolutionContext:
    parsed_cli_reference_datasets = parse_reference_datasets(
        cli_reference_datasets,
        field_name="--reference-datasets",
    )
    return ReferenceTaskResolutionContext(
        training_tasks=tuple(training_tasks),
        suite=(str(suite).strip() if suite is not None else None),
        cli_reference_suite=(str(cli_reference_suite).strip() if cli_reference_suite is not None else None),
        cli_reference_datasets=(
            tuple(parsed_cli_reference_datasets) if parsed_cli_reference_datasets is not None else None
        ),
    )


def resolve_reference_tasks(
    *,
    context: ReferenceTaskResolutionContext,
    regularization_cfg: Mapping[str, Any] | None = None,
    require_reference: bool = True,
) -> tuple[list[str], bool]:
    reg_cfg = dict(regularization_cfg or {})
    explicit = False
    reference_tasks: list[str] | None = None

    if context.cli_reference_datasets is not None:
        reference_tasks = list(context.cli_reference_datasets)
        explicit = True
    elif context.cli_reference_suite:
        reference_suite = str(context.cli_reference_suite).strip()
        if reference_suite not in SUITES:
            raise ValueError(f"--reference-suite must be one of {sorted(SUITES.keys())}; got {reference_suite!r}.")
        reference_tasks = list(SUITES[reference_suite].tasks)
        explicit = True
    else:
        reference_datasets_cfg = parse_reference_datasets(
            reg_cfg.get("reference_datasets", None),
            field_name="regularization.reference_datasets",
        )
        reference_suite_cfg = reg_cfg.get("reference_suite", None)
        if reference_datasets_cfg is not None:
            reference_tasks = reference_datasets_cfg
            explicit = True
        elif reference_suite_cfg is not None:
            reference_suite = str(reference_suite_cfg).strip()
            if reference_suite not in SUITES:
                raise ValueError(
                    f"regularization.reference_suite must be one of {sorted(SUITES.keys())}; got {reference_suite!r}."
                )
            reference_tasks = list(SUITES[reference_suite].tasks)
            explicit = True

    if reference_tasks is None:
        if context.suite is not None:
            reference_tasks = list(SUITES[str(context.suite)].tasks)
        else:
            reference_tasks = list(context.training_tasks)

    reference_tasks = validate_vision_tasks(list(reference_tasks), field_name="reference tasks")
    if explicit and not reference_tasks:
        raise ValueError("Reference dataset selection resolved to an empty task list.")
    if require_reference and len(context.training_tasks) == 1 and not explicit:
        raise ValueError(
            "Training with a single dataset and dataset-aware regularization requires an explicit regularization dataset. "
            "Set --reference-suite, --reference-datasets, regularization.reference_suite, or regularization.reference_datasets."
        )
    return reference_tasks, explicit


def resolve_reference_tasks_from_kwargs(
    *,
    regularization_cfg: Mapping[str, Any] | None,
    kwargs: Mapping[str, Any],
    task: str,
    require_reference: bool = True,
) -> list[str]:
    context = kwargs.get("reference_resolution_context", None)
    if isinstance(context, ReferenceTaskResolutionContext):
        references, _ = resolve_reference_tasks(
            context=context,
            regularization_cfg=regularization_cfg,
            require_reference=require_reference,
        )
        return [ref_task for ref_task in references if ref_task != task]

    reg_cfg = dict(regularization_cfg or {})
    reference_datasets_cfg = parse_reference_datasets(
        reg_cfg.get("reference_datasets", None),
        field_name="regularization.reference_datasets",
    )
    if reference_datasets_cfg is not None:
        return [ref_task for ref_task in validate_vision_tasks(reference_datasets_cfg, field_name="reference tasks") if ref_task != task]
    reference_suite_cfg = reg_cfg.get("reference_suite", None)
    if reference_suite_cfg is not None:
        reference_suite = str(reference_suite_cfg).strip()
        if reference_suite not in SUITES:
            raise ValueError(
                f"regularization.reference_suite must be one of {sorted(SUITES.keys())}; got {reference_suite!r}."
            )
        return [ref_task for ref_task in list(SUITES[reference_suite].tasks) if ref_task != task]

    inherited = kwargs.get("reference_tasks", []) or kwargs.get("all_tasks", []) or [task]
    return [ref_task for ref_task in list(inherited) if ref_task != task]


def _safe_reference_tag(value: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9._-]+", "_", str(value).strip().lower()).strip("_")
    return text or "ref"


def _suite_reference_label(tasks: list[str]) -> str | None:
    for suite_name, suite in SUITES.items():
        suite_tasks = list(suite.tasks)
        if tasks == suite_tasks:
            if suite_name.startswith("vision") and suite_name[len("vision") :].isdigit():
                return f"{suite_name[len('vision') :]}vision"
            return _safe_reference_tag(suite_name)
    return None


def reference_tag_from_tasks(tasks: list[str]) -> str | None:
    normalized = validate_vision_tasks(list(tasks), field_name="reference tasks")
    if not normalized:
        return None
    suite_label = _suite_reference_label(normalized)
    if suite_label is not None:
        return f"{suite_label}_ref"
    if len(normalized) == 1:
        return f"{_safe_reference_tag(normalized[0])}_ref"
    return "__".join(_safe_reference_tag(task) for task in normalized) + "_ref"


def regularization_reference_tags(
    *,
    regularization_cfg: Mapping[str, Any] | None,
    context: ReferenceTaskResolutionContext,
) -> list[str]:
    cfg = dict(regularization_cfg or {})
    name = str(cfg.get("name", "")).strip()
    if not name:
        return []

    tags: list[str] = []
    if name == "composite":
        for child in list(cfg.get("regularizers", []) or []):
            if isinstance(child, Mapping):
                tags.extend(
                    regularization_reference_tags(
                        regularization_cfg=child,
                        context=context,
                    )
                )
        return list(dict.fromkeys(tag for tag in tags if tag))

    if name == "distillation":
        teacher_cfg = cfg.get("teacher", None)
        if isinstance(teacher_cfg, Mapping):
            nested = teacher_cfg.get("regularization", None)
            if isinstance(nested, Mapping):
                tags.extend(
                    regularization_reference_tags(
                        regularization_cfg=nested,
                        context=context,
                    )
                )
        return list(dict.fromkeys(tag for tag in tags if tag))

    if name not in _DATASET_AWARE_REGULARIZERS:
        return []

    references, _ = resolve_reference_tasks(
        context=context,
        regularization_cfg=cfg,
        require_reference=True,
    )
    tag = reference_tag_from_tasks(references)
    return [tag] if tag is not None else []


def apply_reference_tags_to_out_dir(
    *,
    out_dir: str,
    regularization_cfg: Mapping[str, Any] | None,
    context: ReferenceTaskResolutionContext,
) -> str:
    tags = regularization_reference_tags(regularization_cfg=regularization_cfg, context=context)
    if not tags:
        return str(out_dir)
    tag_suffix = "__".join(dict.fromkeys(tags))
    if tag_suffix in str(out_dir):
        return str(out_dir)
    return f"{out_dir}_{tag_suffix}"
