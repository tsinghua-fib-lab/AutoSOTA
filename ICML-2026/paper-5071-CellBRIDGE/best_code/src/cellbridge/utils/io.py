from pathlib import Path

from hydra.core.hydra_config import HydraConfig


def get_project_root() -> Path:
    """Return the repository root for source-tree installs."""
    return Path(__file__).resolve().parents[3]


def resolve_path(path: str | Path, *bases: str | Path | None) -> Path:
    """Resolve a config path from caller bases, cwd, or the repository root."""
    p = Path(path)
    if p.is_absolute():
        return p

    candidates: list[Path] = []
    for base in bases:
        if base is not None:
            candidates.append(Path(base) / p)
    candidates.extend([Path.cwd() / p, get_project_root() / p])

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[-1]


def get_run_dir() -> Path:
    """Absolute Hydra run directory (falls back to CWD if not under Hydra)."""
    try:
        return Path(HydraConfig.get().runtime.output_dir)
    except Exception:
        return Path.cwd()
