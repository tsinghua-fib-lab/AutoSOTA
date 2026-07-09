"""Extract per-round metrics from CPC-LLM sweep outputs on the Modal volume.

Reads accepted sample scores from each round and writes a JSON file with
per-(alpha, seed, round) records containing fraction infeasible, mean score,
and max score.

Usage:
    modal run analysis/extract_sweep_data.py
    # writes sweep_data.json to the current directory
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import modal

app = modal.App("cpc-llm-extract")
outputs_volume = modal.Volume.from_name("cpc-llm-outputs")
img = modal.Image.debian_slim().pip_install("numpy", "pandas")

OUTPUTS_PATH = "/vol/outputs"


@app.function(
    image=img,
    timeout=300,
    volumes={OUTPUTS_PATH: outputs_volume},
)
def extract_sweep_data(
    alphas: list[float] | None = None,
    seeds: list[int] | None = None,
) -> list[dict]:
    """Extract per-round accepted-sample metrics from sweep outputs.

    For round 0, all samples from the initial SFT generation are treated as
    accepted.  For rounds 1+, the AR-accepted samples
    (``accepted_uLiks_*.jsonl``) are used.

    Args:
        alphas: Alpha values to include. None means all found.
        seeds: Seed values to include. None means all found.

    Returns:
        List of dicts, one per (alpha, seed, round_idx) with keys:
        ``seed``, ``alpha``, ``round_idx``, ``frac_infeasible``,
        ``mean_score``, ``max_score``, ``n_total``, ``n_feasible``.
    """
    import numpy as np
    import pandas as pd

    outputs_volume.reload()
    root = Path(OUTPUTS_PATH)

    alpha_set = set(alphas) if alphas else None
    seed_set = set(seeds) if seeds else None

    def score_stats(df: pd.DataFrame) -> dict[str, float | int | None]:
        """Compute feasibility and score statistics from a scored DataFrame.

        Args:
            df: DataFrame with a ``score`` column. Infeasible samples have
                non-finite scores (NaN or ±inf).

        Returns:
            Dict with ``frac_infeasible``, ``mean_score``, ``max_score``,
            ``n_total``, and ``n_feasible``.
        """
        scores = df["score"].to_numpy(dtype=float)
        finite = scores[np.isfinite(scores)]
        return {
            "frac_infeasible": 1.0 - len(finite) / len(scores)
            if len(scores) > 0
            else None,
            # Raw scores are negative (closer to 0 is better); negate for display
            "mean_score": float(-np.mean(finite)) if len(finite) else None,
            "max_score": float(-np.min(finite)) if len(finite) else None,
            "n_total": len(scores),
            "n_feasible": len(finite),
        }

    results: list[dict] = []
    for d in sorted(root.iterdir()):
        if not d.is_dir() or not d.name.startswith("cpc_llm_"):
            continue
        sp = d / "smaller_pythia"
        if not sp.exists():
            continue

        # Identify seed and alpha from output filenames
        data_files = [
            f.name for f in sp.glob("*") if "seed" in f.name and "alpha" in f.name
        ]
        if not data_files:
            continue
        fname = data_files[0]
        alpha_m = re.search(r"alpha([\d.]+)", fname)
        seed_m = re.search(r"seed(\d+)", fname)
        if not alpha_m or not seed_m:
            continue
        alpha = float(alpha_m.group(1))
        seed = int(seed_m.group(1))
        if alpha_set and alpha not in alpha_set:
            continue
        if seed_set and seed not in seed_set:
            continue

        # Round 0: all samples from init SFT generation (treated as all accepted)
        sft_dir = sp / "smaller_pythia_sft_init_r0"
        if sft_dir.exists():
            gen_files = list(sft_dir.glob("gens_likelihood_*_temp1.0_*seqs.jsonl"))
            if gen_files:
                gen_file = max(gen_files, key=lambda f: f.stat().st_size)
                try:
                    df = pd.read_json(gen_file, orient="records", lines=True)
                    row = {"seed": seed, "alpha": alpha, "round_idx": 0}
                    row.update(score_stats(df))
                    results.append(row)
                except Exception as e:
                    print(f"Error r0 seed={seed} alpha={alpha}: {e}")

        # Rounds 1+: AR-accepted samples
        for round_dir in sorted(sp.iterdir()):
            if not round_dir.is_dir():
                continue
            rm = re.search(r"_r(\d+)$", round_dir.name)
            if not rm:
                continue
            round_idx = int(rm.group(1))
            if round_idx == 0:
                continue

            accepted_files = list(round_dir.glob("accepted_uLiks_*.jsonl"))
            if not accepted_files:
                continue
            try:
                df = pd.read_json(
                    sorted(accepted_files)[0], orient="records", lines=True
                )
                if "score" not in df.columns:
                    continue
                row = {"seed": seed, "alpha": alpha, "round_idx": round_idx}
                row.update(score_stats(df))
                results.append(row)
            except Exception as e:
                print(f"Error r{round_idx} seed={seed} alpha={alpha}: {e}")

    return results


@app.local_entrypoint()
def main() -> None:
    results = extract_sweep_data.remote()
    out_path = Path("sweep_data.json")
    out_path.write_text(json.dumps(results, indent=2))
    print(f"Wrote {len(results)} records to {out_path}")
