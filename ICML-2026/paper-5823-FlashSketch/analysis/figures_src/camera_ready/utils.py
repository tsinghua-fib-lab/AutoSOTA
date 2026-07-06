from gitbud.gitbud import inject_repo_into_sys_path
inject_repo_into_sys_path()

from pathlib import Path
import re
import shutil

import torch

import globals as g
from io_utils import ensure_dir


def maybe_copy_pdf_to_paper(pdf_path, subdir=None):
    """Copy a PDF into paper/figures (optionally a subdir) if enabled."""
    if not g.ENV_FLAG_TRUE(g.GET_ENV_VAR(g.ENV_CAMERA_READY_PDF_TO_PAPER)):
        return
    target_dir = g.PAPER_FIGURES_DIR()
    if subdir:
        target_dir = target_dir / subdir
    ensure_dir(target_dir)
    pdf_path = Path(pdf_path)
    shutil.copy2(pdf_path, target_dir / pdf_path.name)


def _slugify_gpu_name(name):
    """Return a filesystem-safe GPU slug."""
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", str(name)).strip("_").lower()
    return slug or "unknown_gpu"


def get_gpu_info():
    """Return a dict with GPU name and slug for the current device."""
    name = "cpu"
    try:
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
    except Exception:
        name = "unknown"
    return {"name": name, "slug": _slugify_gpu_name(name)}
