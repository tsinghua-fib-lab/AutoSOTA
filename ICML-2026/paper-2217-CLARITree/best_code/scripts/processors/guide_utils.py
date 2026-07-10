# scripts/processors/guide_utils.py
from __future__ import annotations
import re, subprocess
from pathlib import Path

R_APPEND = r'''
# ---- R^2 + MSE computation (auto-appended) ----
y_true <- as.numeric(newdata[[ncol(newdata)]])
y_pred <- pred
ss_res <- sum((y_true - y_pred)^2, na.rm = TRUE)
ss_tot <- sum((y_true - mean(y_true, na.rm = TRUE))^2, na.rm = TRUE)
r2 <- 1 - ss_res/ss_tot
mse <- mean((y_true - y_pred)^2, na.rm = TRUE)
cat("R^2 on data:", r2, "\n")
cat("MSE on data:", mse, "\n")
'''

MSE_APPEND = r'''
# ---- MSE computation (auto-appended) ----
y_true <- as.numeric(newdata[[ncol(newdata)]])
y_pred <- pred
mse <- mean((y_true - y_pred)^2, na.rm = TRUE)
cat("MSE on data:", mse, "\n")
'''

NEW_DATA_LINE_RE = re.compile(
    r'''^(?P<head>\s*newdata\s*<-\s*read\.csv\(\s*["'].*?/splits/outer_(\d+)/)train\.csv(?P<tail>["']\s*,\s*header\s*=\s*TRUE\s*,\s*colClasses\s*=\s*["']character["']\s*\).*)$'''
)

def update_guide_model_r(r_path: Path) -> bool:
    txt = r_path.read_text(encoding="utf-8", errors="ignore")
    orig = txt
    lines = txt.splitlines()
    changed = False
    for i, line in enumerate(lines):
        m = NEW_DATA_LINE_RE.match(line)
        if m:
            lines[i] = line.replace("/train.csv", "/test.csv")
            changed = True
    txt = "\n".join(lines)
    has_r2 = "R^2 on data:" in txt or "R^2 on test data:" in txt
    has_mse = "MSE on data:" in txt
    if not has_r2:
        if not txt.endswith("\n"):
            txt += "\n"
        txt += R_APPEND.strip() + "\n"
        changed = True
    elif not has_mse:
        if not txt.endswith("\n"):
            txt += "\n"
        txt += MSE_APPEND.strip() + "\n"
        changed = True
    if changed:
        r_path.write_text(txt, encoding="utf-8")
    return changed

def run_rscript(r_path, cwd=None):
    r_path = Path(r_path)
    if cwd is None:
        cwd = Path(__file__).resolve().parents[2]
    proc = subprocess.run(
        ["Rscript", str(r_path)],
        capture_output=True, text=True, cwd=str(cwd)
    )
    out = proc.stdout + "\n" + proc.stderr
    if proc.returncode != 0:
        raise RuntimeError(f"Rscript failed ({proc.returncode}) for {r_path}:\n{out}")
    return out

def parse_r2(stdout: str) -> float:
    m = re.search(r"R\^2 on(?:.*)data:\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)", stdout)
    if not m:
        raise ValueError(f"Cannot find R^2 in output:\n{stdout}")
    return float(m.group(1))

def parse_mse(stdout: str) -> float:
    m = re.search(r"MSE on(?:.*)data:\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)", stdout)
    if not m:
        raise ValueError(f"Cannot find MSE in output:\n{stdout}")
    return float(m.group(1))


# ============================= training R^2 + elapsed parser (robust) =============================
import re
from pathlib import Path
from typing import Tuple

_R2_TRAIN_PATTERNS = [
    r"Proportion of variance\s*\(\s*R-?\s*squared\s*\)\s*explained by (?:tree\s+)?model\s*[:=]\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)",
    r"Proportion of variance\s*explained\s*[:=]\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)",
    r"R-?\s*squared\s*[:=]\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)",
]
_R2_TRAIN_RES = [re.compile(p, re.IGNORECASE) for p in _R2_TRAIN_PATTERNS]

_ELAPSED_RE = re.compile(
    r"Elapsed\s+time\s+in\s+seconds\s*[:=]\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)",
    re.IGNORECASE,
)

def _extract_r2_elapsed(text: str) -> Tuple[float, float]:
    r2_train = float("nan")
    elapsed = float("nan")
    for rx in _R2_TRAIN_RES:
        m = rx.search(text)
        if m:
            try:
                r2_train = float(m.group(1))
                break
            except Exception:
                pass
    m2 = _ELAPSED_RE.search(text)
    if m2:
        try:
            elapsed = float(m2.group(1))
        except Exception:
            pass
    return r2_train, elapsed

def parse_guide_train_r2_and_elapsed(text_or_path) -> tuple[float, float]:
    try:
        p = Path(text_or_path)
        if p.exists():
            content = p.read_text(encoding="utf-8", errors="ignore")
        else:
            content = str(text_or_path)
    except Exception:
        content = str(text_or_path)
    return _extract_r2_elapsed(content)

def find_guide_training_out(work_dir: Path):
    work_dir = Path(work_dir)
    candidates = []
    common_names = [
        "guide_model.out", "guide.out", "model.out",
        "guide_model.lst", "guide.lst",
        "guide_model.log", "guide.log",
        "guide_output.txt", "output.txt",
    ]
    for name in common_names:
        p = work_dir / name
        if p.exists():
            candidates.append(p)
    if not candidates:
        for ext in ("*.out", "*.lst", "*.log", "*.txt"):
            candidates.extend(work_dir.glob(ext))
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]
