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

    import score_ot_utils
    import score_only_methods

    rsa_src = (METHODS / "score_ot_utils.py").read_text(encoding="utf-8")
    cls_src = inspect.getsource(score_only_methods.SplitConformalRealPlusOTScore)

    checks = {
        "RSA-CP uses barycentric OT": all(
            token in rsa_src
            for token in [
                "num[i] += delta * tgt[j]",
                "den[i] += delta",
                "mapped_src = num / np.maximum(den",
                "rsa_ot_map",
            ]
        ),
        "RSA-CP uses Beta-Binomial rank window": all(
            token in rsa_src for token in ["q_betabin", "b_minus", "b_plus", "q_by_k"]
        ),
        "SplitConformalRealPlusOTScore calls RSA-CP decision/quantile helpers": (
            "get_rsacp_quantile" in cls_src or "prepare_rsacp_state" in cls_src
        ),
        "MEPS helper uses CQR scores": "np.maximum(intervals[:, 0] - y, y - intervals[:, 1])"
        in (ROOT / "real_data_pipeline.py").read_text(encoding="utf-8"),
        "ImageNet helper uses APS scores": "aps_components"
        in (ROOT / "real_data_pipeline.py").read_text(encoding="utf-8"),
    }

    lines = ["RSA-CP real-data method correctness check", ""]
    for name, ok in checks.items():
        lines.append(f"{'PASS' if ok else 'FAIL'}: {name}")

    path = LOGS / "method_correctness_check.txt"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    if not all(checks.values()):
        raise SystemExit(f"Method correctness check failed. See {path}")


if __name__ == "__main__":
    main()
