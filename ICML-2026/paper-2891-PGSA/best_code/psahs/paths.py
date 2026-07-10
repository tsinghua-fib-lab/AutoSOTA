"""Repository path helpers (independent of current working directory)."""
from __future__ import annotations

import os
from pathlib import Path


def get_repo_root() -> str:
    """Return the PSAHS-release repository root."""
    return str(Path(__file__).resolve().parent.parent)


def dataset_root() -> str:
    return os.path.join(get_repo_root(), "dataset")


def output_dir(args) -> str:
    """Checkpoints and logs: ``outputs/<dataset>/``."""
    path = os.path.join(get_repo_root(), "outputs", _dataset_slug(args))
    os.makedirs(path, exist_ok=True)
    return path


def figures_dir(args) -> str:
    """Figures: ``figures/<dataset>/``."""
    path = os.path.join(get_repo_root(), "figures", _dataset_slug(args))
    os.makedirs(path, exist_ok=True)
    return path


def _dataset_slug(args) -> str:
    ds = (getattr(args, "dataset", None) or "run").strip() or "run"
    return "".join(c if (c.isalnum() or c in "-_") else "_" for c in ds)


def _fs_token(value, max_len: int = 120) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text:
        return ""
    out = "".join(c if (c.isalnum() or c in "-_") else "_" for c in text)
    return out[:max_len] if len(out) > max_len else out


def checkpoint_suffix(args) -> str:
    """Checkpoint basename suffix: ``_{src}_{tgt}`` and optional ``_{num_nodes}``."""
    parts = []
    if (getattr(args, "dataset", "") or "").strip() != "Noncircle":
        src = _fs_token(getattr(args, "src_name", None))
        tgt = _fs_token(getattr(args, "tgt_name", None))
        if src or tgt:
            parts.append(f"_{src or 'src'}_{tgt or 'tgt'}")
    if (getattr(args, "dataset", "") or "").strip() == "Noncircle":
        n = int(getattr(args, "num_nodes", 4000) or 4000)
        parts.append(f"_{n}")
    return "".join(parts)


def noncircle_default_pickle_paths(args):
    """Default (source_pkl, target_pkl) for the Noncircle benchmark."""
    root = get_repo_root()
    n = int(getattr(args, "num_nodes", 4000) or 4000)
    src = os.path.join(root, "dataset", "noncircle", f"source_graph_ICLR_{n}.pkl")
    tgt = os.path.join(
        root, "dataset", "noncircle", f"ICLR_target_graphs_{n}", "target_graph_h0.3.pkl"
    )
    return src, tgt
