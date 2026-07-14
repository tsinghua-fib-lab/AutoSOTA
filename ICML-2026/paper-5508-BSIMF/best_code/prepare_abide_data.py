#!/usr/bin/env python3
"""
Prepare ABIDE I data: compute functional connectivity from AAL ROI time series
and extract behavioral scores, then save in the merged format expected by the model.
"""
import os
import sys
import pickle
import warnings
from typing import Dict

import numpy as np
import pandas as pd

# --- Configuration ---
ROIS_DIR = "/datasets/abide/rois_aal"
PHENO_CSV = "/datasets/abide/phenotypic.csv"
OUTPUT_PKL = "/datasets/abide/abide_i_merged.pkl"
N_REGIONS = 116  # AAL atlas

# The 12 behavioral scores we target (matching the paper's selection)
# We pick scores from ADI-R, ADOS, SRS, and SCQ batteries
# 12 behavioral scores matching paper's "n_ratings=12"
BEHAVIORAL_COLS = [
    "ADI_R_SOCIAL_TOTAL_A",
    "ADI_R_VERBAL_TOTAL_BV",
    "ADI_RRB_TOTAL_C",
    "ADI_R_ONSET_TOTAL_D",
    "ADOS_TOTAL",
    "ADOS_COMM",
    "ADOS_SOCIAL",
    "ADOS_STEREO_BEHAV",
    "SRS_RAW_TOTAL",
    "SRS_AWARENESS",
    "SRS_COGNITION",
    "SCQ_TOTAL",
]

MIN_SCORES_PER_SUBJECT = 3
# Sentinel values used in ABIDE for missing data
SENTINEL_VALUES = [-9999.0, -9999, -9998, -9997]
# Fisher z-transform connectivity? Helps with normality
USE_FISHER_Z = True


def compute_connectivity(roi_ts: np.ndarray) -> np.ndarray:
    """
    Compute functional connectivity (Pearson correlation) from ROI time series.

    Args:
        roi_ts: (T, N) time series matrix, T=timepoints, N=regions

    Returns:
        conn: (N, N) symmetric correlation matrix. Any NaN (from zero-variance)
              regions are filled with 0.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        corr = np.corrcoef(roi_ts.T)  # (N, N)

    # Handle NaN from zero-variance regions
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)

    if USE_FISHER_Z:
        # Fisher z-transform: atanh(r), with clipping to avoid inf
        corr = np.clip(corr, -0.999999, 0.999999)
        corr = np.arctanh(corr)

    # Ensure symmetry
    corr = 0.5 * (corr + corr.T)
    return corr.astype(np.float32)


def main():
    print("=" * 60)
    print("Loading phenotypic data...")
    pheno = pd.read_csv(PHENO_CSV)
    print(f"  {len(pheno)} subjects in phenotypic CSV")

    # Determine which subjects have behavioral data
    avail_cols = [c for c in BEHAVIORAL_COLS if c in pheno.columns]
    missing_cols = [c for c in BEHAVIORAL_COLS if c not in pheno.columns]
    if missing_cols:
        print(f"  WARNING: Missing columns in CSV: {missing_cols}")
    print(f"  Using {len(avail_cols)} behavioral score columns")

    # Clean sentinel values: replace sentinels with NaN
    for col in avail_cols:
        pheno[col] = pd.to_numeric(pheno[col], errors="coerce")
        for sentinel in SENTINEL_VALUES:
            pheno.loc[pheno[col] == sentinel, col] = np.nan
        # Also treat very negative values (< -100) as sentinels
        pheno.loc[pheno[col] < -100, col] = np.nan

    # Count available scores per subject
    score_counts = pheno[avail_cols].notna().sum(axis=1)
    good_mask = score_counts >= MIN_SCORES_PER_SUBJECT
    good_pheno = pheno[good_mask].copy()
    print(f"  {good_mask.sum()} subjects with >= {MIN_SCORES_PER_SUBJECT} valid behavioral scores")

    # Check available ROI files
    if not os.path.isdir(ROIS_DIR):
        print(f"\nERROR: ROI directory not found: {ROIS_DIR}")
        sys.exit(1)

    available_rois = set(os.listdir(ROIS_DIR))
    print(f"  {len(available_rois)} ROI files in {ROIS_DIR}")

    # Build merged dataset
    big_dict: Dict[str, Dict] = {}
    n_missing_roi = 0
    n_bad_roi = 0
    n_good = 0

    for _, row in good_pheno.iterrows():
        sid = str(int(row["SUB_ID"]))
        roi_fname = f"{sid}_rois_aal.1D"

        if roi_fname not in available_rois:
            n_missing_roi += 1
            continue

        # Load ROI time series
        roi_path = os.path.join(ROIS_DIR, roi_fname)
        try:
            roi_ts = np.loadtxt(roi_path, dtype=np.float32)
        except Exception as e:
            print(f"  WARNING: Could not load {roi_fname}: {e}")
            n_bad_roi += 1
            continue

        # Validate shape: should be (T, 116) or (116, T)
        if roi_ts.ndim != 2:
            n_bad_roi += 1
            continue

        # AAL has 116 regions. The ROI files from CPAC have shape (T, N)
        if roi_ts.shape[1] != N_REGIONS and roi_ts.shape[0] == N_REGIONS:
            roi_ts = roi_ts.T
        elif roi_ts.shape[1] != N_REGIONS:
            # Some files might have a different number of regions
            print(f"  WARNING: {roi_fname} has shape {roi_ts.shape}, expected 116 regions")
            n_bad_roi += 1
            continue

        # Compute connectivity
        conn = compute_connectivity(roi_ts)

        # Extract behavioral scores
        scores = np.full(len(avail_cols), np.nan, dtype=np.float32)
        for i, col in enumerate(avail_cols):
            val = row[col]
            if pd.notna(val):
                scores[i] = float(val)

        # Get group label
        dx_group = int(row.get("DX_GROUP", 0))
        # DX_GROUP: 1=autism, 2=control -> map to 0,1
        group = 0 if dx_group == 2 else 1 if dx_group == 1 else dx_group

        big_dict[sid] = {
            "conn": conn,
            "scores": scores,
            "assignedGroup": group,
        }
        n_good += 1

    print(f"\n--- Summary ---")
    print(f"  Subjects with valid data: {n_good}")
    print(f"  Missing ROI files: {n_missing_roi}")
    print(f"  Bad ROI files: {n_bad_roi}")

    # Report score availability
    if n_good > 0:
        score_array = np.stack([big_dict[s]["scores"] for s in big_dict])
        print(f"\n  Behavioral score availability (out of {n_good} subjects):")
        for i, col in enumerate(avail_cols):
            present = np.sum(~np.isnan(score_array[:, i]))
            print(f"    {col}: {present} subjects")

    # Save merged dataset
    print(f"\nSaving to {OUTPUT_PKL}...")
    os.makedirs(os.path.dirname(OUTPUT_PKL), exist_ok=True)
    with open(OUTPUT_PKL, "wb") as f:
        pickle.dump(big_dict, f)

    # Also save the column order for reference
    col_info_path = OUTPUT_PKL.replace(".pkl", "_columns.txt")
    with open(col_info_path, "w") as f:
        f.write("Behavioral score columns (order in scores vector):\n")
        for i, col in enumerate(avail_cols):
            f.write(f"  [{i}] {col}\n")

    print(f"Done! {n_good} subjects saved.")
    print(f"Column info: {col_info_path}")


if __name__ == "__main__":
    main()
