from __future__ import annotations

import inspect
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
METHODS = ROOT / "methods"
LOGS = ROOT / "outputs" / "logs"
sys.path.insert(0, str(METHODS))


def main() -> None:
    LOGS.mkdir(parents=True, exist_ok=True)

    import score_only_methods

    cls_src = inspect.getsource(score_only_methods.SplitConformalRealPlusOTScore)
    banned = [
        "s_aug",
        "concatenate([s_min, s_ot])",
        "ot_map_scores(source_scores=s_maj, target_scores=s_min",
        "S_mapped_real",
        "apply_scale_ot_map",
        "cum_weights",
        "w_r <-",
        "w_s <-",
        "mean scaling",
        "weighted quantile with real weight",
    ]

    hits = [pat for pat in banned if pat in cls_src]
    lines = [
        "Old-method check for SplitConformalRealPlusOTScore",
        "",
        "PASS: legacy ot_map_scores may exist only as an unused helper for older baselines.",
    ]
    if hits:
        lines.append("FAIL: banned old-method patterns found in the active RSA-CP class:")
        lines.extend(f"  - {hit}" for hit in hits)
    else:
        lines.append("PASS: no old OT-score augmentation / weighted-quantile / mean-scaling logic found in the active RSA-CP class.")

    path = LOGS / "old_method_check.txt"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    if hits:
        raise SystemExit(f"Old-method check failed. See {path}")


if __name__ == "__main__":
    main()
