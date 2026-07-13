# Dataset loaders for the robustness/clamping and intervention experiments.
import json, random
from pathlib import Path

ROBUSTNESS_DIR = Path(__file__).resolve().parent

# ----------------------------
# Shared helpers
# ----------------------------
def _resolve_path(path):
    path = Path(path)
    if path.is_absolute() or path.exists():
        return path
    local_path = ROBUSTNESS_DIR / path
    if local_path.exists():
        return local_path
    return path

def _read_jsonl(path):
    path = _resolve_path(path)
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def _group_by_slice(data):
    grouped = {}
    for ex in data:
        grouped.setdefault(ex["slice"], []).append(ex)
    return grouped

def _pairs_from_grouped(grouped, n=None, seed=None, shuffle=True):
    if seed is not None:
        random.seed(seed)
    pairs = []
    for slice_name, examples in grouped.items():
        exs = examples
        if n is not None:
            exs = random.sample(examples, min(len(examples), n))
        for ex in exs:
            pairs.append((slice_name, ex["original"], ex["perturbed"]))
    if shuffle:
        random.shuffle(pairs)
    return pairs

# ----------------------------
# Perturbation loader
# ----------------------------
def load_perturbation_benchmark(path="perturbation_dataset/perturbation_benchmark_clean.jsonl", n=None, seed=None, shuffle=True):
    """
    Load the cleaned perturbation benchmark dataset.
    Returns list of (slice, original, perturbed).
    If n is given, samples n examples per slice.
    """
    data = _read_jsonl(path)
    grouped = _group_by_slice(data)
    return _pairs_from_grouped(grouped, n=n, seed=seed, shuffle=shuffle)

# ----------------------------
# Intervention loader
# ----------------------------
def load_intervention_benchmark(path="intervention_dataset/intervention_benchmark_clean.jsonl", n=None, seed=None, shuffle=True):
    """
    Load the intervention robustness dataset.
    Each row has: {"slice": <str>, "original": <str>, "perturbed": <str>}
    Only the tag in square brackets differs; the sentence is identical.
    Returns list of (slice, original, perturbed).
    If n is given, samples n examples per slice.
    """
    # Allow relative paths like "data/..." or absolute paths
    path = str(Path(path))
    data = _read_jsonl(path)
    grouped = _group_by_slice(data)
    return _pairs_from_grouped(grouped, n=n, seed=seed, shuffle=shuffle)

# ----------------------------
# Unified dispatcher
# ----------------------------
def load_benchmark(name="perturbation", n=None, path=None, seed=None, shuffle=True):
    """
    Unified benchmark loader.
    Returns a list of (slice, text_a, text_b).

    name:
      - "perturbation": classic perturbations (spelling, synonym, typo, etc.)
      - "intervention": interventions (gender/number/negation tags)

    path:
      - If None, uses sensible defaults:
          "perturbation" -> "perturbation_dataset/perturbation_benchmark_clean.jsonl"
          "intervention" -> "intervention_dataset/intervention_benchmark_clean.jsonl"
    """
    if name == "perturbation":
        default_path = "perturbation_dataset/perturbation_benchmark_clean.jsonl"
        return load_perturbation_benchmark(path or default_path, n=n, seed=seed, shuffle=shuffle)

    elif name == "intervention":
        default_path = "intervention_dataset/intervention_benchmark_clean.jsonl"
        return load_intervention_benchmark(path or default_path, n=n, seed=seed, shuffle=shuffle)

    else:
        raise ValueError(f"Unknown benchmark: {name}. Use 'perturbation' or 'intervention'.")
