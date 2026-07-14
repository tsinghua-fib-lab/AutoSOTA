"""Path helpers for running the MTLR code locally or on Google Colab.

The intended Colab location is
`/content/drive/MyDrive/Colab Notebooks/MTLR_Codes`.
Local runs are resolved relative to this file or the current working directory.
"""

from __future__ import annotations

import os
import sys
import zipfile
from pathlib import Path
from typing import Iterable


CODE_MARKERS = ("synthetic_sweeps.py", "real_data_har.py", "ARMUL.py", "MTL.py")
COLAB_CODE_CANDIDATES = (
    Path("/content/drive/MyDrive/Colab Notebooks/MTLR_Codes"),
    Path("/content/drive/My Drive/Colab Notebooks/MTLR_Codes"),
    Path("/content/drive/MyDrive/MTLR_Codes"),
    Path("/content/drive/My Drive/MTLR_Codes"),
)


def running_in_colab() -> bool:
    """Return True when running inside Google Colab."""
    try:
        import google.colab  # type: ignore  # noqa: F401
    except Exception:
        return False
    return True


def mount_drive_if_colab(mount_point: str = "/content/drive") -> None:
    """Mount Google Drive when running in Colab; no-op locally."""
    if not running_in_colab():
        return

    drive_root = Path(mount_point)
    if (drive_root / "MyDrive").exists() or (drive_root / "My Drive").exists():
        return

    from google.colab import drive  # type: ignore

    drive.mount(mount_point)


def _looks_like_code_dir(path: Path) -> bool:
    return path.is_dir() and all((path / marker).exists() for marker in CODE_MARKERS)


def _candidate_code_dirs(start: Path | None = None) -> Iterable[Path]:
    start = Path.cwd() if start is None else Path(start)
    yield start
    yield start / "MTLR_Codes"
    for parent in [start, *start.parents]:
        yield parent / "MTLR_Codes"
    yield Path(__file__).resolve().parent
    yield from COLAB_CODE_CANDIDATES


def find_code_dir(start: Path | None = None, mount_drive: bool = True) -> Path:
    """Locate the `MTLR_Codes` directory and add it to `sys.path`."""
    if mount_drive:
        mount_drive_if_colab()

    seen: set[Path] = set()
    for candidate in _candidate_code_dirs(start):
        candidate = candidate.expanduser()
        key = candidate.resolve() if candidate.exists() else candidate
        if key in seen:
            continue
        seen.add(key)
        if _looks_like_code_dir(candidate):
            code_dir = candidate.resolve()
            if str(code_dir) not in sys.path:
                sys.path.insert(0, str(code_dir))
            return code_dir

    searched = "\n".join(f"  - {p}" for p in _candidate_code_dirs(start))
    raise FileNotFoundError(
        "Could not locate the MTLR_Codes directory. "
        "On Colab, upload the folder to `MyDrive/Colab Notebooks/MTLR_Codes`.\n"
        f"Searched:\n{searched}"
    )


def default_figure_dir(code_dir: Path) -> Path:
    """Choose a figure directory that works for both full-repo and Colab runs."""
    code_dir = Path(code_dir).resolve()
    repo_root = code_dir.parent
    if (repo_root / "Images").exists():
        return repo_root / "Images"
    return code_dir / "Images"


def setup_project_paths(chdir: bool = True) -> tuple[Path, Path, Path]:
    """Return `(CODE_DIR, PROJECT_ROOT, FIGURE_DIR)` and prepare imports."""
    code_dir = find_code_dir()
    project_root = code_dir.parent
    figure_dir = default_figure_dir(code_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    if chdir:
        os.chdir(code_dir)
    return code_dir, project_root, figure_dir


def _has_har_files(root: Path) -> bool:
    return (
        (root / "train" / "X_train.txt").exists()
        and (root / "train" / "subject_train.txt").exists()
        and (root / "train" / "y_train.txt").exists()
        and (root / "test" / "X_test.txt").exists()
        and (root / "test" / "subject_test.txt").exists()
        and (root / "test" / "y_test.txt").exists()
    )


def _extract_har_zip(zip_path: Path) -> None:
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(zip_path.parent)


def find_har_dataset(code_dir: Path | None = None, extract_if_needed: bool = True) -> Path:
    """Locate the extracted UCI HAR dataset, optionally extracting the local zip."""
    code_dir = find_code_dir(mount_drive=False) if code_dir is None else Path(code_dir).resolve()
    candidates = (
        code_dir / "UCI_HAR_Dataset",
        code_dir / "HAR Data" / "UCI HAR Dataset",
        Path.cwd() / "UCI_HAR_Dataset",
        Path.cwd() / "HAR Data" / "UCI HAR Dataset",
    )

    for root in candidates:
        if _has_har_files(root):
            return root.resolve()

    zip_candidates = (
        code_dir / "HAR Data" / "UCI HAR Dataset.zip",
        code_dir / "UCI HAR Dataset.zip",
    )
    if extract_if_needed:
        for zip_path in zip_candidates:
            if zip_path.exists():
                _extract_har_zip(zip_path)
                extracted = zip_path.parent / "UCI HAR Dataset"
                if _has_har_files(extracted):
                    return extracted.resolve()

    expected = code_dir / "UCI_HAR_Dataset"
    raise FileNotFoundError(
        "Could not locate the UCI HAR dataset. Place the extracted dataset at\n"
        f"  {expected}\n"
        "so that `train/X_train.txt` and `test/X_test.txt` exist. "
        "If using Colab, upload the dataset folder inside `MTLR_Codes`."
    )
