# robustness/tasks.py
import json, random

def load_perturbation_benchmark(path="perturbation_dataset/perturbation_benchmark_clean.jsonl", n=None):
    """
    Load the cleaned perturbation benchmark dataset.
    Returns list of (slice, original, perturbed).
    If n is given, samples n examples per slice, otherwise loads all.
    """
    with open(path, "r") as f:
        data = [json.loads(line) for line in f]

    # group by slice
    grouped = {}
    for ex in data:
        grouped.setdefault(ex["slice"], []).append(ex)

    pairs = []
    for slice_name, examples in grouped.items():
        if n is not None:
            examples = random.sample(examples, min(len(examples), n))
        for ex in examples:
            pairs.append((slice_name, ex["original"], ex["perturbed"]))

    return pairs


def load_benchmark(name="perturbation",
                   n=None,
                   path="perturbation_dataset/perturbation_benchmark_clean.jsonl"):
    """
    Unified benchmark loader (currently only perturbation dataset).
    Returns a list of (slice, text_a, text_b).
    """
    if name == "perturbation":
        return load_perturbation_benchmark(path=path, n=n)
    else:
        raise ValueError(f"Unknown benchmark: {name}. Only 'perturbation' is supported.")