# src/turkey_prelim.py
import argparse, json, os, math
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --------------------------- utilities ---------------------------

def index_images(root: Path) -> pd.DataFrame:
    """Index all images under fold1..fold5 and return rel & basename."""
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    paths = [p for p in root.rglob("*") if p.suffix.lower() in exts]
    rows = [{"rel": str(p.relative_to(root)), "basename": p.name} for p in paths]
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError(f"No image files found under {root}")
    return df

def parse_annotations_turkey(root: Path) -> pd.DataFrame:
    """
    Parse data/Turkey/annotations.json into rows: rel, basename, annotator, label(0/1).
    Map class_label=='plumage_injury' -> 1, anything else -> 0.
    """
    ann_path = root / "annotations.json"
    if not ann_path.exists():
        raise FileNotFoundError(f"Missing {ann_path}")

    obj = json.load(open(ann_path, "r"))
    if not isinstance(obj, list):
        raise ValueError("annotations.json top-level is not a list (unexpected format)")

    rows = []
    for annot in obj:
        items = annot.get("annotations", [])
        annotator = annot.get("user_mail") or annot.get("name") or "unknown"
        for rec in items:
            ip = rec.get("image_path")
            cl = (rec.get("class_label") or "").strip().lower()
            if not ip:
                continue
            # Normalize path: strip leading "Turkey/" if present and keep foldX/filename
            rel = ip.split("Turkey/")[-1].lstrip("/\\")
            base = os.path.basename(rel)
            lab = 1 if cl in {"plumage_injury", "injury", "injured", "positive", "1", "true", "yes"} else 0
            rows.append({"rel": rel, "basename": base, "annotator": annotator, "label": lab, "raw_label": cl})

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("Parsed 0 rows from annotations.json — unexpected key names?")
    return df

def build_cost_table(root: Path) -> pd.DataFrame:
    """
    Return one row per image with: rel, basename, n_yes, n_no, n, delta, abs_delta, y_star.
    """
    df_idx = index_images(root)   # columns: rel, basename
    df_ann = parse_annotations_turkey(root)  # columns: rel, basename, annotator, label, raw_label

    # 1) Try to join on rel (most reliable)
    df = df_idx.merge(df_ann[["rel","basename","annotator","label"]], on="rel", how="inner")

    # 2) If the match rate is low (paths differ), fall back to basename join
    if len(df) < 0.5 * len(df_ann):
        df = df_idx.merge(
            df_ann[["rel","basename","annotator","label"]],
            on="basename",
            how="inner",
            suffixes=("_idx", "_ann"),
        )

        # Normalize columns so downstream code always has 'rel' and 'basename'
        if "rel_idx" in df.columns:
            df["rel"] = df["rel_idx"]
        elif "rel_ann" in df.columns:
            df["rel"] = df["rel_ann"]

    # Ensure we have rel + some basename column
    if "rel" not in df.columns:
        # if we still don't, try to recover from possible suffixing
        rel_cols = [c for c in df.columns if c.startswith("rel")]
        if rel_cols:
            df["rel"] = df[rel_cols[0]]
        else:
            raise ValueError("Could not determine 'rel' column after merges.")

    base_cols = [c for c in df.columns if c.startswith("basename")]
    if not base_cols:
        # worst case, rebuild basename from rel
        df["basename"] = df["rel"].map(lambda s: os.path.basename(str(s)))
        base_cols = ["basename"]

    # 3) Aggregate votes per image
    g = df.groupby("rel")["label"]
    out = g.agg(n_yes="sum", n="count").reset_index()
    out["n_no"] = out["n"] - out["n_yes"]

    # 4) Δ with +1 smoothing
    out["delta"] = np.log((out["n_yes"] + 1) / (out["n_no"] + 1))
    out["abs_delta"] = out["delta"].abs()
    out["y_star"] = (out["delta"] >= 0).astype(int)

    # 5) Map a single basename back
    base_col = base_cols[0]
    base_map = df.drop_duplicates("rel").set_index("rel")[base_col]
    out["basename"] = out["rel"].map(base_map).fillna(out["rel"].map(lambda s: os.path.basename(str(s))))

    return out


def summarize_and_save(df_cost: pd.DataFrame, out_dir: Path, sample: int | None):
    out_dir.mkdir(parents=True, exist_ok=True)
    if sample and len(df_cost) > sample:
        df_cost = df_cost.sample(sample, random_state=42).reset_index(drop=True)

    # Basic stats
    mean_votes = df_cost["n"].mean()
    med_votes  = df_cost["n"].median()
    min_votes  = int(df_cost["n"].min())
    max_votes  = int(df_cost["n"].max())

    mean_abs_delta = df_cost["abs_delta"].mean()
    max_abs_delta  = df_cost["abs_delta"].max()

    print("\n=== Turkey: annotation & Δ summary ===")
    print(f"Images with any votes    : {len(df_cost):,}")
    print(f"Votes per image          : mean={mean_votes:.2f}, median={med_votes:.0f}, min={min_votes}, max={max_votes}")
    print(f"|Δ|                       : mean={mean_abs_delta:.3f}, max={max_abs_delta:.3f}")

    # Save tables
    df_cost.to_csv(out_dir / "turkey_cost_table.csv", index=False)

    annot_hist = (
        df_cost["n"].value_counts()
        .sort_index()
        .rename_axis("num_annotations")
        .reset_index(name="image_count")
    )
    annot_hist.to_csv(out_dir / "turkey_annotations_hist.csv", index=False)

    # Histogram of signed Δ
    plt.figure()
    plt.hist(df_cost["delta"], bins=50)
    plt.title("Turkey: Δ distribution")
    plt.xlabel("Δ = log((n_yes+1)/(n_no+1))")
    plt.ylabel("Count (images)")
    plt.tight_layout()
    plt.savefig(out_dir / "turkey_delta_hist.png", dpi=150)
    plt.close()

    print(f"\nArtifacts written to: {out_dir}")

# --------------------------- CLI ---------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True, help="Path to Turkey dataset root (contains fold1..fold5, annotations.json)")
    ap.add_argument("--sample", type=int, default=None, help="Optional number of images to sample for quick looks")
    args = ap.parse_args()

    root = Path(args.root)
    df_cost = build_cost_table(root)
    out_dir = Path("results") / "turkey"
    summarize_and_save(df_cost, out_dir, args.sample)

if __name__ == "__main__":
    main()

