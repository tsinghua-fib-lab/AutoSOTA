import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from model_mapping import ModelMappingEntry, load_model_mapping

SCRIPT_DIR = Path(__file__).parent.resolve()
DATA_DIR = SCRIPT_DIR / "data"

MLEBENCH_SCORE_TASKS = ("above_median", "any_medal", "valid_submission")


class ModelMapper:
    """Wrap the curated model mapping and provide direct lookups."""

    def __init__(self, mapping_path: Path):
        self._records = load_model_mapping(mapping_path)
        self._id_lookup: dict[str, ModelMappingEntry] = {}
        for record in self._records.values():
            self._id_lookup[record.key.strip()] = record
            for run_id in record.run_ids:
                normalized = self._normalize(run_id)
                if normalized:
                    self._id_lookup.setdefault(normalized, record)

    def _normalize(self, value: str | None) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @property
    def restricts(self) -> bool:
        return bool(self._records)

    @property
    def records(self) -> dict[str, ModelMappingEntry]:
        return self._records

    def lookup_subject(self, subject_id: str | None) -> ModelMappingEntry | None:
        normalized = self._normalize(subject_id)
        if normalized is None:
            return None
        return self._records.get(normalized)

    def lookup_run(
        self, run_id: str | None = None, model_field: str | None = None
    ) -> ModelMappingEntry | None:
        for candidate in (model_field, run_id):
            normalized = self._normalize(candidate)
            if normalized is None:
                continue
            record = self._records.get(normalized)
            if record:
                return record
            record = self._id_lookup.get(normalized)
            if record:
                return record
        return None


def load_pyirt_file(
    path: Path,
    mapper: ModelMapper,
    sink: dict[str, dict[str, list[float]]],
    keep_unmapped_subjects: bool,
):
    with path.open() as f:
        for line in f:
            entry = json.loads(line)
            subject_id = entry["subject_id"]
            record = mapper.lookup_subject(subject_id)
            if record is None:
                if mapper.restricts and not keep_unmapped_subjects:
                    continue
                key = subject_id
            else:
                key = record.key
            responses = sink.setdefault(key, {})
            for item_id, value in entry["responses"].items():
                responses[item_id] = [float(value)]


def iter_runs(path: Path) -> Iterable[dict]:
    with path.open() as f:
        for line in f:
            yield json.loads(line)


def resolve_subject_key(
    mapper: ModelMapper,
    *,
    run_id: str | None = None,
    model_field: str | None = None,
    alias: str | None = None,
    allow_unmapped: bool = False,
) -> tuple[str | None, bool]:
    """Return a canonical subject key and whether the record was missing."""

    canonical = mapper.lookup_run(run_id=run_id, model_field=model_field)
    if canonical is None:
        if mapper.restricts and not allow_unmapped:
            return None, True
        subject_key = str(model_field or run_id or alias or "").strip()
        if not subject_key:
            subject_key = "unknown"
        return subject_key, False
    return canonical.key, False


def add_run_records(
    paths: list[Path],
    mapper: ModelMapper,
    sink: dict[str, dict[str, list[float]]],
    score_column: str,
    *,
    verbose: bool = False,
) -> tuple[int, int]:
    total_rows = 0
    missing_ids = 0
    conflicts = 0
    for path in paths:
        for record in iter_runs(path):
            total_rows += 1
            run_id = record.get("run_id")
            alias = record.get("alias")
            model_field = record.get("model")
            if model_field == "human":
                continue
            subject_key, missing = resolve_subject_key(
                mapper, run_id=run_id, model_field=model_field, alias=alias
            )
            if missing:
                missing_ids += 1
                if verbose:
                    print(model_field)
                continue
            if subject_key is None:
                continue
            score_val = record.get(score_column)
            if score_val is None:
                continue
            try:
                score = float(score_val)
            except (TypeError, ValueError):
                continue
            score = 1.0 if score >= 0.5 else 0.0
            task_id = record.get("task_id")
            if not task_id:
                continue
            responses = sink.setdefault(subject_key, {})
            scores = responses.setdefault(task_id, [])
            if scores:
                conflicts += 1
            scores.append(score)
    if verbose and missing_ids:
        if mapper.restricts:
            print(
                f"[prepare_sparse_pyirt] {missing_ids} rows skipped "
                "because they were missing from the mapping",
                file=sys.stderr,
            )
        else:
            print(
                f"[prepare_sparse_pyirt] {missing_ids} rows used fallback identifiers",
                file=sys.stderr,
            )
    if verbose and conflicts:
        print(
            "[prepare_sparse_pyirt] "
            f"{conflicts} conflicting responses resolved by mean>=0.5",
            file=sys.stderr,
        )
    return total_rows, missing_ids


def add_gdpval_results(
    path: Path,
    mapper: ModelMapper,
    sink: dict[str, dict[str, list[float]]],
    *,
    keep_unmapped_subjects: bool,
) -> tuple[int, int]:
    total_rows = 0
    missing_ids = 0
    if not path.exists():
        return total_rows, missing_ids
    for record in iter_runs(path):
        total_rows += 1
        subject_key, missing = resolve_subject_key(
            mapper,
            model_field=record.get("model"),
            allow_unmapped=keep_unmapped_subjects,
        )
        if missing:
            missing_ids += 1
            continue
        if subject_key is None:
            continue
        task_id = record.get("task_id")
        if not task_id:
            continue
        score_val = record.get("score")
        if score_val is None:
            continue
        try:
            score = float(score_val)
        except (TypeError, ValueError):
            continue
        score = 1.0 if score >= 0.5 else 0.0
        responses = sink.setdefault(subject_key, {})
        scores = responses.setdefault(task_id, [])
        scores.append(score)
    return total_rows, missing_ids


def add_mlebench_results(
    path: Path,
    mapper: ModelMapper,
    sink: dict[str, dict[str, list[float]]],
    *,
    keep_unmapped_subjects: bool,
) -> tuple[int, int, int]:
    total_rows = 0
    missing_ids = 0
    metrics_written = 0
    if not path.exists():
        return total_rows, missing_ids, metrics_written
    for record in iter_runs(path):
        total_rows += 1
        subject_key, missing = resolve_subject_key(
            mapper,
            model_field=record.get("model"),
            allow_unmapped=keep_unmapped_subjects,
        )
        if missing:
            missing_ids += 1
            continue
        if subject_key is None:
            continue
        task_id = record.get("task_id")
        if not task_id:
            continue
        score_block = record.get("score")
        if not isinstance(score_block, dict):
            continue
        responses = sink.setdefault(subject_key, {})
        for metric in MLEBENCH_SCORE_TASKS:
            value = score_block.get(metric)
            if value is None:
                continue
            try:
                score = float(value)
            except (TypeError, ValueError):
                continue
            metric_score = 1.0 if score >= 0.5 else 0.0
            metric_task_id = f"{task_id}::{metric}"
            responses.setdefault(metric_task_id, []).append(metric_score)
            metrics_written += 1
    return total_rows, missing_ids, metrics_written


def add_cybench_results(
    path: Path,
    mapper: ModelMapper,
    sink: dict[str, dict[str, list[float]]],
    *,
    keep_unmapped_subjects: bool,
    cybench_task_ids: set[str] | None = None,
) -> tuple[int, int]:
    """Add Cybench results from normalized JSONL file.

    This function processes Cybench CTF challenge results, where each task
    represents a cybersecurity challenge with a binary pass/fail outcome.
    """
    total_rows = 0
    missing_ids = 0
    if not path.exists():
        return total_rows, missing_ids
    for record in iter_runs(path):
        total_rows += 1
        subject_key, missing = resolve_subject_key(
            mapper,
            run_id=record.get("run_id"),
            model_field=record.get("model"),
            allow_unmapped=keep_unmapped_subjects,
        )
        if missing:
            missing_ids += 1
            continue
        if subject_key is None:
            continue
        task_id = record.get("task_id")
        if not task_id:
            continue
        if cybench_task_ids is not None:
            cybench_task_ids.add(task_id)
        # Cybench uses score_binarized for binary outcomes
        score_val = record.get("score_binarized")
        if score_val is None:
            score_val = record.get("score_cont", 0.0)
        try:
            score = float(score_val)
        except (TypeError, ValueError):
            continue
        score = 1.0 if score >= 0.5 else 0.0
        responses = sink.setdefault(subject_key, {})
        scores = responses.setdefault(task_id, [])
        scores.append(score)
    return total_rows, missing_ids


def write_output(path: Path, responses: dict[str, dict[str, list[float]]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for subject_id in sorted(responses):
            aggregated: dict[str, int] = {}
            for item_id, values in responses[subject_id].items():
                mean_score = sum(values) / len(values)
                aggregated[item_id] = 1 if mean_score >= 0.5 else 0
            payload = {
                "subject_id": subject_id,
                "responses": aggregated,
            }
            f.write(json.dumps(payload) + "\n")


def compute_sparsity_stats(responses: dict[str, dict[str, list[float]]]) -> dict[str, float]:
    """Return sparsity statistics for the response matrix."""

    subject_count = len(responses)
    item_ids: set[str] = set()
    filled = 0
    for subject_responses in responses.values():
        item_ids.update(subject_responses.keys())
        filled += len(subject_responses)

    total_items = len(item_ids)
    total_possible = subject_count * total_items
    sparsity = 1 - (filled / total_possible) if total_possible else 0.0
    return {
        "sparsity": sparsity,
        "subjects": subject_count,
        "items": total_items,
        "filled": filled,
        "total_possible": total_possible,
    }


def apply_target_sparsity(
    responses: dict[str, dict[str, list[float]]],
    target_sparsity: float,
    *,
    excluded_task_ids: set[str] | None = None,
) -> tuple[int, dict[str, float]]:
    """Randomly drop subject/item pairs to reach the desired sparsity."""

    rng = random.Random()
    current_stats = compute_sparsity_stats(responses)
    if current_stats["total_possible"] == 0:
        return 0, current_stats

    # Only increase sparsity; no-op if already sparse enough
    if target_sparsity <= current_stats["sparsity"]:
        return 0, current_stats

    target_density = 1 - target_sparsity
    desired_filled = int(current_stats["total_possible"] * target_density)
    desired_filled = max(0, desired_filled)

    removal_needed = current_stats["filled"] - desired_filled
    if removal_needed <= 0:
        return 0, current_stats

    candidates: list[tuple[str, str]] = []
    for subject_id, subject_responses in responses.items():
        for item_id in subject_responses:
            if excluded_task_ids and item_id in excluded_task_ids:
                continue
            candidates.append((subject_id, item_id))

    rng.shuffle(candidates)
    removed = 0
    for subject_id, item_id in candidates:
        if removed >= removal_needed:
            break
        subject_responses = responses.get(subject_id)
        if subject_responses is None:
            continue
        # Avoid deleting the last remaining response for a subject when possible
        if len(subject_responses) <= 1 and removed + 1 < removal_needed:
            continue
        subject_responses.pop(item_id, None)
        removed += 1

    new_stats = compute_sparsity_stats(responses)
    return removed, new_stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert run logs into a sparse py-IRT JSONL file."
    )
    parser.add_argument(
        "--model-mapping",
        type=Path,
        default=DATA_DIR / "model_run_mapping.csv",
        help="JSON/CSV mapping describing which models and run_ids to include.",
    )
    parser.add_argument(
        "--pyirt-input",
        type=Path,
        action="append",
        default=[],
        help="Existing py-IRT JSONL inputs to include (can be repeated).",
    )
    parser.add_argument(
        "--runs-input",
        type=Path,
        action="append",
        default=[],
        help="Raw JSONL run files (per-task rows) to include (can be repeated).",
    )
    parser.add_argument(
        "--score-column",
        default="score_binarized",
        help="Column to read from runs-input for binary correctness.",
    )
    parser.add_argument(
        "--gdpval-input",
        type=Path,
        default=None,
        help=(
            "Normalized GDPVal JSONL file to include; defaults to data/"
            "gdpval_normalized_results.jsonl."
        ),
    )
    parser.add_argument(
        "--mlebench-input",
        type=Path,
        default=None,
        help=(
            "Normalized MLEBench JSONL file to include; defaults to data/"
            "mlebench_normalized_results.jsonl."
        ),
    )
    parser.add_argument(
        "--cybench-input",
        type=Path,
        default=None,
        help=(
            "Normalized Cybench JSONL file to include; defaults to data/"
            "cybench_normalized_results.jsonl."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Where to write the combined py-IRT JSONL file.",
    )
    parser.add_argument(
        "--target-sparsity",
        type=float,
        default=None,
        help=(
            "Desired sparsity percentage for the output (e.g., 80 or 0.8). "
            "If provided and higher than the current sparsity, entries are "
            "removed at random to reach the target."
        ),
    )
    parser.add_argument(
        "--print-subject-counts",
        action="store_true",
        help="Print how many responses each subject contributed.",
    )
    parser.add_argument(
        "--keep-unmapped-pyirt-subjects",
        action="store_true",
        help=(
            "When reading py-IRT inputs, retain rows even if the subject is missing "
            "from the mapping by falling back to the recorded subject_id."
        ),
    )
    parser.add_argument(
        "--exclude-cybench-from-sparsity",
        action="store_true",
        help="When applying target sparsity, do not drop Cybench task responses.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed progress information.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.pyirt_input and not args.runs_input:
        raise SystemExit("Provide at least one --pyirt-input or --runs-input.")

    target_sparsity = args.target_sparsity
    if target_sparsity is not None and target_sparsity > 1:
        target_sparsity /= 100.0
    if target_sparsity is not None and not (0 <= target_sparsity <= 1):
        raise ValueError("--target-sparsity must be between 0 and 1 (or 0-100).")

    mapper = ModelMapper(args.model_mapping)
    combined: dict[str, dict[str, list[float]]] = defaultdict(dict)
    cybench_task_ids: set[str] = set()

    for path in args.pyirt_input:
        if not path.exists():
            raise FileNotFoundError(f"py-IRT input {path} does not exist")
        load_pyirt_file(
            path,
            mapper,
            combined,
            keep_unmapped_subjects=args.keep_unmapped_pyirt_subjects,
        )

    rows_loaded = 0
    if args.runs_input:
        for path in args.runs_input:
            if not path.exists():
                raise FileNotFoundError(f"runs input {path} does not exist")
        rows_loaded, missing = add_run_records(
            args.runs_input, mapper, combined, args.score_column, verbose=args.verbose
        )
        if args.verbose:
            print(
                f"[prepare_sparse_pyirt] processed {rows_loaded} run rows "
                f"(missing ids: {missing})"
            )

    gdpval_path = args.gdpval_input or DATA_DIR / "gdpval_normalized_results.jsonl"
    if gdpval_path:
        gdp_rows, gdp_missing = add_gdpval_results(
            gdpval_path,
            mapper,
            combined,
            keep_unmapped_subjects=args.keep_unmapped_pyirt_subjects,
        )
        if args.verbose and (gdp_rows or gdp_missing):
            print(
                "[prepare_sparse_pyirt] processed "
                f"{gdp_rows} GDPVal rows (missing ids: {gdp_missing})"
            )

    mlebench_path = args.mlebench_input or DATA_DIR / "mlebench_normalized_results.jsonl"
    if mlebench_path:
        mle_rows, mle_missing, mle_metrics = add_mlebench_results(
            mlebench_path,
            mapper,
            combined,
            keep_unmapped_subjects=args.keep_unmapped_pyirt_subjects,
        )
        if args.verbose and (mle_rows or mle_missing or mle_metrics):
            print(
                "[prepare_sparse_pyirt] processed "
                f"{mle_rows} MLEBench rows across {mle_metrics} metric tasks "
                f"(missing ids: {mle_missing})"
            )

    cybench_path = args.cybench_input or DATA_DIR / "cybench_normalized_results.jsonl"
    if cybench_path:
        cyber_rows, cyber_missing = add_cybench_results(
            cybench_path,
            mapper,
            combined,
            keep_unmapped_subjects=args.keep_unmapped_pyirt_subjects,
            cybench_task_ids=cybench_task_ids,
        )
        if args.verbose and (cyber_rows or cyber_missing):
            print(
                "[prepare_sparse_pyirt] processed "
                f"{cyber_rows} Cybench rows (missing ids: {cyber_missing})"
            )

    original_stats = compute_sparsity_stats(combined)
    if args.verbose:
        print(
            "[prepare_sparse_pyirt] original sparsity: "
            f"{original_stats['sparsity']:.2%} "
            f"({original_stats['filled']}/{original_stats['total_possible']} filled across "
            f"{original_stats['subjects']} subjects, {original_stats['items']} items)"
        )

    if target_sparsity is not None:
        excluded_tasks = cybench_task_ids if args.exclude_cybench_from_sparsity else None
        if args.verbose and excluded_tasks:
            print(
                f"[prepare_sparse_pyirt] excluding {len(excluded_tasks)} Cybench tasks "
                "from sparsity adjustments"
            )
        removed, new_stats = apply_target_sparsity(
            combined, target_sparsity, excluded_task_ids=excluded_tasks
        )
        if args.verbose:
            if removed:
                print(
                    f"[prepare_sparse_pyirt] applied target sparsity {target_sparsity:.2%}: "
                    f"removed {removed} cells; new sparsity {new_stats['sparsity']:.2%} "
                    f"({new_stats['filled']}/{new_stats['total_possible']} filled)"
                )
                if new_stats["sparsity"] + 1e-6 < target_sparsity:
                    print(
                        "[prepare_sparse_pyirt] warning: unable to reach requested sparsity "
                        "without emptying all remaining responses"
                    )
            else:
                if new_stats["sparsity"] + 1e-6 < target_sparsity and excluded_tasks:
                    print(
                        "[prepare_sparse_pyirt] warning: no eligible non-Cybench responses "
                        "available to drop to meet sparsity target"
                    )
                else:
                    print(
                        f"[prepare_sparse_pyirt] target sparsity {target_sparsity:.2%} not "
                        "applied (dataset already meets or exceeds sparsity target)"
                    )

    write_output(args.output, combined)
    if args.verbose:
        print(
            f"[prepare_sparse_pyirt] wrote {len(combined)} subjects and "
            f"{sum(len(v) for v in combined.values())} responses to {args.output}"
        )
    if args.verbose and args.print_subject_counts:
        print("[prepare_sparse_pyirt] per-subject response counts:")
        for subject_id in sorted(combined):
            print(f"  {subject_id}: {len(combined[subject_id])}")


if __name__ == "__main__":
    main()
