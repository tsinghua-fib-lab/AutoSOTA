"""TEDBench: Large-Scale Protein Fold Classification Benchmark.

Quick start::

    import tedbench

    print(tedbench.__version__)

    for m in tedbench.list_models():
        print(m["name"], m["type"], m["params"])

    model = tedbench.load_model("miae-b")        # pretrained encoder
    model = tedbench.load_model("miae-b-ft")     # fine-tuned classifier
    model = tedbench.load_model("dexiongc/tedbench-miae-b")  # full HF repo ID
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("TEDBench")
except PackageNotFoundError:
    __version__ = "unknown"

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

_HF_ORG = "TEDBench"

_MODEL_REGISTRY: dict[str, dict] = {
    # ---- Small (29M) ----
    "miae-s": {
        "repo_id": f"{_HF_ORG}/miae-s",
        "params": "29M",
        "type": "pretrained",
        "description": "MiAE-Small pretrained encoder",
    },
    "miae-s-ft": {
        "repo_id": f"{_HF_ORG}/miae-s-ft",
        "params": "29M",
        "type": "fine-tuned",
        "description": "MiAE-Small fine-tuned on TEDBench fold classification",
    },
    "miae-s-sc": {
        "repo_id": f"{_HF_ORG}/miae-s-sc",
        "params": "29M",
        "type": "from-scratch",
        "description": "MiAE-Small trained from scratch on TEDBench",
    },
    # ---- Base (102M) ----
    "miae-b": {
        "repo_id": f"{_HF_ORG}/miae-b",
        "params": "102M",
        "type": "pretrained",
        "description": "MiAE-Base pretrained encoder",
    },
    "miae-b-ft": {
        "repo_id": f"{_HF_ORG}/miae-b-ft",
        "params": "102M",
        "type": "fine-tuned",
        "description": "MiAE-Base fine-tuned on TEDBench fold classification",
    },
    "miae-b-sc": {
        "repo_id": f"{_HF_ORG}/miae-b-sc",
        "params": "102M",
        "type": "from-scratch",
        "description": "MiAE-Base trained from scratch on TEDBench",
    },
    # ---- Base + sequence input (102M) ----
    "miae-b-seq": {
        "repo_id": f"{_HF_ORG}/miae-b-seq",
        "params": "102M",
        "type": "pretrained",
        "description": "MiAE-Base+seq pretrained encoder (structure + sequence tokens)",
    },
    "miae-b-seq-ft": {
        "repo_id": f"{_HF_ORG}/miae-b-seq-ft",
        "params": "102M",
        "type": "fine-tuned",
        "description": "MiAE-Base+seq fine-tuned on TEDBench fold classification",
    },
    "miae-b-seq-sc": {
        "repo_id": f"{_HF_ORG}/miae-b-seq-sc",
        "params": "102M",
        "type": "from-scratch",
        "description": "MiAE-Base+seq trained from scratch on TEDBench",
    },
    # ---- Large (339M) ----
    "miae-l": {
        "repo_id": f"{_HF_ORG}/miae-l",
        "params": "339M",
        "type": "pretrained",
        "description": "MiAE-Large pretrained encoder",
    },
    "miae-l-ft": {
        "repo_id": f"{_HF_ORG}/miae-l-ft",
        "params": "339M",
        "type": "fine-tuned",
        "description": "MiAE-Large fine-tuned on TEDBench fold classification",
    },
    "miae-l-sc": {
        "repo_id": f"{_HF_ORG}/miae-l-sc",
        "params": "339M",
        "type": "from-scratch",
        "description": "MiAE-Large trained from scratch on TEDBench",
    },
}


def list_models() -> list[dict]:
    """Return metadata for all available TEDBench pretrained models.

    Returns:
        List of dicts with keys ``name``, ``repo_id``, ``params``, ``type``,
        ``description``.

    Example::

        for m in tedbench.list_models():
            print(f"{m['name']:20s}  {m['type']:12s}  {m['params']}")
    """
    return [{"name": k, **v} for k, v in _MODEL_REGISTRY.items()]


def load_model(name: str):
    """Load a TEDBench model by short name or HuggingFace repo ID.

    Short names are resolved via the built-in registry.  Any string containing
    ``"/"`` is treated as a full HuggingFace repo ID or a local directory path
    and passed through directly::

        model = tedbench.load_model("miae-b")               # pretrained encoder
        model = tedbench.load_model("miae-b-ft")            # fine-tuned classifier
        model = tedbench.load_model("dexiongc/tedbench-miae-b")  # full HF repo ID
        model = tedbench.load_model("/path/to/local/dir")   # local directory

    Args:
        name: Short model name from :func:`list_models`, a full HuggingFace
            repo ID, or a local directory path.

    Returns:
        The loaded model in eval mode (``MiAE`` or ``MiAEClassifier``).

    Raises:
        KeyError: If *name* is not in the registry and does not contain ``"/"``.
    """
    from tedbench.utils.io import load_from_hf

    if "/" in str(name):
        return load_from_hf(name)

    if name not in _MODEL_REGISTRY:
        available = ", ".join(sorted(_MODEL_REGISTRY))
        raise KeyError(
            f"Unknown model {name!r}. Available short names:\n  {available}\n"
            "Pass a full HuggingFace repo ID (e.g. 'dexiongc/tedbench-miae-b') "
            "to bypass the registry."
        )

    return load_from_hf(_MODEL_REGISTRY[name]["repo_id"])
