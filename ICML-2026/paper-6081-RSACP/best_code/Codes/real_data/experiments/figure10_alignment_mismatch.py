"""Optional score-level alignment mismatch figure.

The current real-data release focuses on Figures 4, 5, 8, and 9.  This script is
kept as an optional helper for reproducing the earlier Figure 10 stress test.
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import real_data_pipeline as pipe


if __name__ == "__main__":
    raw = pipe.run_figure10_alignment()
    summary = pipe.summarize(raw, ["figure", "dataset", "Method", "alpha", "delta", "m", "N", "n_test"])
    raw.to_csv(pipe.RAW_DIR / "figure10_alignment_mismatch_raw.csv", index=False)
    summary.to_csv(pipe.SUMMARY_DIR / "figure10_alignment_mismatch_summary.csv", index=False)
    pipe.plot_figure10(summary, pipe.FIGURE_DIR / "figure10_alignment_mismatch")
