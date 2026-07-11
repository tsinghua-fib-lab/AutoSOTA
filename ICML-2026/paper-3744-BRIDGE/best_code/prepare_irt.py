import argparse
import json
import math
from pathlib import Path
import yaml

SCRIPT_DIR = Path(__file__).parent.resolve()
DATA_DIR = SCRIPT_DIR / "data"
VERIFIED_RESULTS_DIR = DATA_DIR / "experiments" / "evaluation" / "verified"
BASH_ONLY_RESULTS_DIR = DATA_DIR / "experiments" / "evaluation" / "bash-only"
MODEL_MAPPING_PATH = DATA_DIR / "model_run_mapping.json"


def read_jsonl(path: Path):
    """Load every JSON line into a list of dicts."""
    with path.open() as f:
        return [json.loads(line) for line in f]


MODEL_NAME_MAP = {
    "Qwen/Qwen2.5-Coder-32B-Instruct": "qwen2_5coder32b",
    "openai/gpt-oss-20b": "gpt-oss-20b",
    "openai/gpt-oss-120b": "gpt-oss-120b",
    "MiniMaxAI/MiniMax-M2": "minimaxm2",
    "Qwen/Qwen3-Coder-30B-A3B-Instruct": "qwen3-coder-30b",
    "Qwen/Qwen3-Coder-480B-A35B-Instruct": "qwen3-coder-480b",
}


def normalize_swebench_records(records):
    """Normalize swebench_normalized_results.jsonl records to the expected schema.

    Maps 'task_id' -> 'id' and 'model' -> 'model_name' so that
    build_score_matrix can consume them directly.  Also translates
    fully-qualified model names to the short names used in
    swebench_normalized_results.jsonl via MODEL_NAME_MAP.
    """
    normalized = []
    for record in records:
        model = record.get("model") or record.get("model_name")
        normalized.append({
            "id": record.get("task_id") or record.get("id"),
            "model_name": MODEL_NAME_MAP.get(model, model),
            "score": record["score"],
        })
    return normalized


def select_score(value):
    if isinstance(value, dict):
        for key in ("any_medal", "valid_submission", "above_median"):
            if key in value:
                return value[key]
        raise ValueError(f"Unable to pick score from {value}")
    return value


def build_score_matrix(records):
    required = {"id", "model_name", "score"}
    for idx, record in enumerate(records):
        missing = required - record.keys()
        if missing:
            raise ValueError(f"Record {idx} is missing fields: {missing}")
    ids = sorted({record["id"] for record in records})
    models = sorted({record["model_name"] for record in records})
    id_to_idx = {rid: idx for idx, rid in enumerate(ids)}
    model_to_idx = {name: idx for idx, name in enumerate(models)}
    matrix = [[float("nan") for _ in models] for _ in ids]
    for record in records:
        row = id_to_idx[record["id"]]
        col = model_to_idx[record["model_name"]]
        matrix[row][col] = select_score(record["score"])
    return {"ids": ids, "models": models, "array": matrix}


def matrix_to_pyirt_records(scores):
    ids = scores["ids"]
    models = scores["models"]
    arr = scores["array"]
    for col_idx, model in enumerate(models):
        responses = {}
        for row_idx, item in enumerate(ids):
            value = arr[row_idx][col_idx]
            if value is None or isinstance(value, float) and math.isnan(value):
                continue
            responses[item] = int(value)
        if responses:
            yield {"subject_id": model, "responses": responses}


def write_pyirt_jsonl(scores, out_path, verbose=False):
    records = list(matrix_to_pyirt_records(scores))
    with open(out_path, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")
    if verbose:
        print(f"Wrote {len(records)} subjects to {out_path}")


def _tokenize_verified_keywords(*values):
    """Return normalized keyword tokens used for filtering."""
    tokens = set()
    for value in values:
        if not value:
            continue
        lowered = str(value).strip().lower()
        if not lowered:
            continue
        tokens.add(lowered)
        cleaned = lowered.replace('/', ' ').replace('+', ' ').replace('-', ' ').replace('_', ' ')
        for part in cleaned.split():
            part = part.strip()
            if len(part) < 3:
                continue
            tokens.add(part)
    return tokens


def _load_run_metadata(entry):
    metadata = {}
    for name in ("metadata.yaml", "metadata.yml"):
        meta_path = entry / name
        if meta_path.exists() and yaml is not None:
            try:
                metadata = yaml.safe_load(meta_path.read_text()) or {}
            except Exception:
                pass
            break
    return metadata


def _load_run_success_ids(entry, verified_task_ids=None):
    results_path = entry / "results" / "results.json"
    success_ids = None
    if results_path.exists():
        with results_path.open() as f:
            payload = json.load(f)
        success_ids = set(payload.get("resolved") or [])
    else:
        per_instance_path = entry / "per_instance_details.json"
        if per_instance_path.exists():
            with per_instance_path.open() as f:
                payload = json.load(f)
            success_ids = {
                task_id
                for task_id, details in payload.items()
                if isinstance(details, dict) and details.get("resolved")
            }
    if success_ids is None:
        return None
    if verified_task_ids is not None:
        success_ids &= verified_task_ids
    return success_ids


def load_verified_runs(verified_root, *, extra_roots=None, verified_task_ids=None):
    """Load metadata and resolved ids for every verified evaluation run."""
    runs = []
    verified_root = Path(verified_root)
    if not verified_root.exists():
        raise FileNotFoundError(f"No verified directory found at {verified_root}")
    roots = [verified_root]
    if extra_roots:
        for root in extra_roots:
            root_path = Path(root)
            if root_path.exists():
                roots.append(root_path)
    verified_task_ids = set(verified_task_ids) if verified_task_ids is not None else None
    seen = set()
    for root in roots:
        for entry in sorted(root.iterdir()):
            if not entry.is_dir():
                continue
            run_id = entry.name
            if run_id in seen:
                continue
            success_ids = _load_run_success_ids(entry, verified_task_ids)
            if success_ids is None:
                continue
            metadata = _load_run_metadata(entry)
            tags = metadata.get("tags") or {}
            info = metadata.get("info") or {}
            model_tags = tags.get("model") or []
            org = tags.get("org")
            label = info.get("name") or run_id
            keywords = _tokenize_verified_keywords(run_id, label, org, *model_tags)
            search_terms = [run_id, label]
            if isinstance(org, (list, tuple, set)):
                search_terms.extend(org)
            else:
                search_terms.append(org)
            search_terms.extend(model_tags or [])
            search_text = " ".join(
                str(term).strip().lower() for term in search_terms if term
            ).strip()
            runs.append(
                {
                    "run_id": run_id,
                    "label": label,
                    "org": org,
                    "model_tags": model_tags,
                    "keywords": keywords,
                    "search_text": search_text,
                    "success_ids": success_ids,
                    "success_count": len(success_ids),
                }
            )
            seen.add(run_id)
    return runs


def _coerce_mapping_flag(value, default=True):
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    text = str(value).strip().lower()
    if not text:
        return default
    if text in {"1", "t", "true", "y", "yes"}:
        return True
    if text in {"0", "f", "false", "n", "no"}:
        return False
    return default


def normalize_model_mapping(mapping):
    """Normalize the editable mapping dict before using it."""
    normalized = {}
    mapping = mapping or {}
    for model_key, spec in mapping.items():
        key = str(model_key).strip()
        if not key:
            continue
        spec = spec or {}
        alias = spec.get("alias") or spec.get("display_name") or key
        runs = spec.get("run_ids") or spec.get("runs") or spec.get("run_names") or []
        if isinstance(runs, dict):
            run_values = runs.keys()
        elif isinstance(runs, (list, tuple, set)):
            run_values = runs
        else:
            run_values = [runs]
        normalized_runs = []
        seen = set()
        for run_id in run_values:
            run_text = str(run_id).strip()
            if not run_text or run_text in seen:
                continue
            seen.add(run_text)
            normalized_runs.append(run_text)
        normalized[key] = {
            "alias": str(alias).strip() or key,
            "plot": _coerce_mapping_flag(spec.get("plot"), True),
            "selected": _coerce_mapping_flag(spec.get("selected"), True),
            "run_ids": normalized_runs,
        }
    return normalized


def build_verified_records_from_mapping(runs, mapping, base_ids, *, selected_only=False, verbose=False):
    """Convert verified successes into py-IRT rows based on the curated mapping."""
    mapping = mapping or {}
    if not mapping:
        return [], []
    run_lookup = {run["run_id"]: run for run in runs}
    base_ids = list(base_ids)
    missing_runs = []
    records = []
    kept_models = 0
    for model_key, spec in mapping.items():
        if selected_only and not spec.get("selected", True):
            continue
        run_ids = spec.get("run_ids") or []
        combined_success = set()
        for run_id in run_ids:
            run = run_lookup.get(run_id)
            if not run:
                missing_runs.append(run_id)
                continue
            combined_success.update(run["success_ids"])
        if not combined_success:
            if verbose:
                print(f"[verified] {model_key}: no overlapping successes; skipping")
            continue
        kept_models += 1
        if verbose:
            print(f"[verified] {model_key}: keeping {len(combined_success)} successes from {len(run_ids)} run(s)")
        for task_id in base_ids:
            records.append({
                "id": task_id,
                "model_name": model_key,
                "score": 1 if task_id in combined_success else 0,
            })
    if verbose:
        print(f"[verified] built {kept_models} model column(s) (selected_only={selected_only})")
    if missing_runs and verbose:
        print("[verified] missing run directories:", ", ".join(sorted(set(missing_runs))))
    return records, sorted(set(missing_runs))


def build_unmapped_run_records(runs, base_ids, *, verbose=False):
    """Convert verified runs without mapping entries into py-IRT rows keyed by run id."""
    base_ids = list(base_ids)
    base_set = set(base_ids)
    if not runs:
        return []
    records = []
    kept_runs = 0
    for run in runs:
        success_ids = set(run["success_ids"]) & base_set
        if not success_ids:
            if verbose:
                print(f"[verified] {run['run_id']}: no overlapping successes; skipping")
            continue
        kept_runs += 1
        if verbose:
            print(f"[verified] Keeping unmapped run {run['run_id']} with {len(success_ids)} SWE successes")
        for task_id in base_ids:
            records.append({
                "id": task_id,
                "model_name": run["run_id"],
                "score": 1 if task_id in success_ids else 0,
            })
    if verbose:
        print(f"[verified] built {kept_runs} unmapped run column(s)")
    return records


def prepare_swe_a_pyirt(verbose=False):
    """Build swe_a_pyirt.jsonl from swebench results and verified runs."""
    # Load swebench results
    swe_results = read_jsonl(DATA_DIR / "swebench_normalized_results.jsonl")
    swe_results = normalize_swebench_records(swe_results)
    swe_scores = build_score_matrix(swe_results)
    if verbose:
        print(f"SWE-bench: {len(swe_scores['ids'])} ids x {len(swe_scores['models'])} models")

    # Load model mapping
    if not MODEL_MAPPING_PATH.exists():
        raise FileNotFoundError(f"Mapping file not found at {MODEL_MAPPING_PATH}")
    model_run_mapping = normalize_model_mapping(json.loads(MODEL_MAPPING_PATH.read_text()))
    if verbose:
        print(f"Configured {len(model_run_mapping)} model(s) in mapping")

    # Load verified runs
    extra_verified_roots = [BASH_ONLY_RESULTS_DIR]
    verified_runs = load_verified_runs(
        VERIFIED_RESULTS_DIR,
        extra_roots=extra_verified_roots,
        verified_task_ids=set(swe_scores["ids"]),
    )
    if verbose:
        print(f"Discovered {len(verified_runs)} verified run directories")
    if not verified_runs:
        raise RuntimeError("No verified runs found; ensure VERIFIED_RESULTS_DIR is correct.")

    # Build records from mapping (selected only)
    selected_run_records, missing_runs = build_verified_records_from_mapping(
        verified_runs,
        model_run_mapping,
        swe_scores["ids"],
        selected_only=True,
        verbose=verbose,
    )
    if missing_runs and verbose:
        print("Missing directories from mapping:", ", ".join(missing_runs))

    # Build unmapped run records
    run_id_to_model_key = {
        run_id: model_key
        for model_key, spec in model_run_mapping.items()
        for run_id in (spec.get("run_ids") or [])
    }
    mapped_run_ids = set(run_id_to_model_key)
    unmapped_runs = [run for run in verified_runs if run["run_id"] not in mapped_run_ids]
    if verbose:
        print(f"Found {len(unmapped_runs)} verified run(s) without mapping entries")

    unmapped_records = build_unmapped_run_records(unmapped_runs, swe_scores["ids"], verbose=verbose)

    # Combine selected and unmapped records
    combined_records = [dict(record) for record in selected_run_records]
    if unmapped_records:
        combined_records.extend(unmapped_records)

    # Remap subjects to mapping keys
    remapped_subjects = 0
    for record in combined_records:
        mapped_name = run_id_to_model_key.get(record["model_name"])
        if mapped_name and mapped_name != record["model_name"]:
            remapped_subjects += 1
            record["model_name"] = mapped_name
    if remapped_subjects and verbose:
        print(f"Remapped {remapped_subjects} subject(s) to mapping keys")

    # Build final score matrix from combined records
    if combined_records:
        combined_scores = build_score_matrix(combined_records)
    else:
        combined_scores = {"ids": [], "models": [], "array": []}

    # Build swebench pyirt records
    swe_pyirt_records = list(matrix_to_pyirt_records(swe_scores))

    # Build combined pyirt records
    combined_pyirt_records = list(matrix_to_pyirt_records(combined_scores))

    # Concatenate and write output
    all_records = swe_pyirt_records + combined_pyirt_records
    if not all_records:
        raise RuntimeError("No py-irt records found to output.")

    out_path = DATA_DIR / "swe_a_pyirt.jsonl"
    with out_path.open("w") as f:
        for record in all_records:
            f.write(json.dumps(record) + "\n")
    if verbose:
        print(f"Wrote {len(all_records)} subjects to {out_path}")

    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare swe_a_pyirt.jsonl for IRT analysis")
    parser.add_argument("--verbose", action="store_true", help="Print detailed progress information")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prepare_swe_a_pyirt(verbose=args.verbose)


if __name__ == "__main__":
    main()
