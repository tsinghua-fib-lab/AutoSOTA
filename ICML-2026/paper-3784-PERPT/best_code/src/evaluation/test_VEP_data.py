#!/usr/bin/env python3
import argparse
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple, List

import numpy as np
import pandas as pd
from datasets import load_dataset

# ----------------------------
# label normalization
# ----------------------------

_POS_STRINGS = {
    "pathogenic", "likely_pathogenic", "pathogenic/likely_pathogenic",
    "deleterious", "damaging", "positive", "pos", "true", "1", "high", "causal", "case", "rare",
}
_NEG_STRINGS = {
    "benign", "likely_benign", "benign/likely_benign",
    "neutral", "negative", "neg", "false", "0", "low", "noncausal", "control", "common",
}

def normalize_binary_label(series: pd.Series, *, dataset_name: str) -> np.ndarray:
    s = series
    if s.dtype == bool:
        return s.astype(np.int8).to_numpy()

    if np.issubdtype(s.dtype, np.number):
        vals = pd.to_numeric(s, errors="coerce")
        uniq = pd.unique(vals.dropna())
        if set(map(int, uniq)) <= {0, 1}:
            return vals.fillna(0).astype(np.int8).to_numpy()
        raise ValueError(f"[{dataset_name}] numeric label not binary 0/1. uniques={sorted(list(set(uniq)))[:20]}")

    ss = s.astype(str).str.strip().str.lower()
    ss = ss.str.replace(r"\s+", "_", regex=True)

    y = np.full(len(ss), -1, dtype=np.int8)
    y[ss.isin(_POS_STRINGS)] = 1
    y[ss.isin(_NEG_STRINGS)] = 0

    unset = y < 0
    if unset.any():
        y[unset & ss.str.contains("pathogenic", na=False)] = 1
        y[unset & ss.str.contains("benign", na=False)] = 0

    if (y < 0).any():
        bad = pd.Series(s.iloc[np.where(y < 0)[0]]).value_counts().head(12)
        raise ValueError(f"[{dataset_name}] couldn't map some label strings. top unmapped:\n{bad}")

    return y


# ----------------------------
# helpers
# ----------------------------

def load_df(hf: str, split: str = "test", subset: Optional[str] = None) -> pd.DataFrame:
    ds = load_dataset(hf, subset, split=split)
    return ds.to_pandas()

def norm_consequence(s: pd.Series) -> pd.Series:
    return (
        s.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"\s+", "_", regex=True)
    )

def schema_debug(df: pd.DataFrame, name: str) -> None:
    print(f"\n--- schema debug: {name} ---")
    print("columns:", list(df.columns))
    if "label" in df.columns:
        print(f"col=label: dtype={df['label'].dtype}, n_unique={df['label'].nunique(dropna=False)}")
        print("top label value_counts:")
        vc = df["label"].value_counts(dropna=False).head(8)
        for k, v in vc.items():
            print(f"  {k}: {int(v)}")
    if "consequence" in df.columns:
        print("\ncol=consequence (top 16):")
        vc = df["consequence"].value_counts(dropna=False).head(16)
        for k, v in vc.items():
            print(f"  {k}: {int(v)}")
    if "tss_dist" in df.columns:
        print(f"\ncol=tss_dist: dtype={df['tss_dist'].dtype}, n_unique={df['tss_dist'].nunique(dropna=False)}")

def count_pos_neg(y: np.ndarray) -> Tuple[int, int]:
    return int((y == 1).sum()), int((y == 0).sum())

def mismatch_status(pos: int, neg: int, exp_pos: Optional[int], exp_neg: Optional[int]) -> str:
    if exp_pos is None or exp_neg is None:
        return "NO_EXPECTED_COUNTS"
    return "OK" if (pos == exp_pos and neg == exp_neg) else f"MISMATCH (expected {exp_pos}/{exp_neg})"


# ----------------------------
# promoter derived benchmark
# ----------------------------

PROMOTER_TAGSETS: Dict[str, set] = {
    "PROMOTER_CORE(pls)": {"pls"},
    "PROMOTER_PLUS(pls+flank)": {"pls", "pls_flank"},
    "PROMOTER_CCRE(broad)": {
        "pls", "pls_flank",
        "pels", "pels_flank",
        "dnase-h3k4me3", "dnase-h3k4me3_flank",
        "ctcf-only", "ctcf-only_flank",
    },
}

TSS_WINDOWS = [None, 50, 100, 200, 300, 500, 750, 1000, 1500, 2000, 3000, 5000]
K_NEG_GRID = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12]

def sample_by_match_group(df: pd.DataFrame, y: np.ndarray, k_neg: int) -> Tuple[pd.DataFrame, np.ndarray]:
    if "match_group" not in df.columns:
        return df, y

    d = df.copy()
    d["_y"] = y

    keep = []
    for _, sub in d.groupby("match_group", sort=False):
        pos = sub[sub["_y"] == 1]
        neg = sub[sub["_y"] == 0]
        if len(pos) == 0 or len(neg) == 0:
            continue
        keep.append(pos.head(1))
        keep.append(neg.head(min(k_neg, len(neg))))

    if not keep:
        out = d.drop(columns=["_y"])
        return out.iloc[0:0], np.array([], dtype=np.int8)

    out = pd.concat(keep, axis=0).drop(columns=["_y"])
    y2 = normalize_binary_label(out["label"], dataset_name=f"ukb_nc_promoter_kneg{k_neg}")
    return out, y2

def promoter_filter_df(
    df_nc: pd.DataFrame,
    tagset_name: str,
    tss_bp: Optional[int],
    use_match_group: bool,
    k_neg: int,
) -> Tuple[pd.DataFrame, np.ndarray]:
    d = df_nc.copy()
    d["_cons_norm"] = norm_consequence(d["consequence"]) if "consequence" in d.columns else ""

    tagset = PROMOTER_TAGSETS[tagset_name]
    d = d[d["_cons_norm"].isin(tagset)].copy()

    if tss_bp is not None:
        d = d[d["tss_dist"].abs() <= tss_bp].copy()

    y = normalize_binary_label(d["label"], dataset_name="ukb_nc_promoter_subset")

    if use_match_group:
        d, y = sample_by_match_group(d, y, k_neg=k_neg)

    return d, y

def sweep_promoter_rules(df_nc: pd.DataFrame, target_pos: int, target_neg: int) -> pd.DataFrame:
    rows: List[dict] = []
    for tagset_name in PROMOTER_TAGSETS.keys():
        for tss in TSS_WINDOWS:
            for use_mg in (False, True):
                k_grid = [None] if not use_mg else K_NEG_GRID
                for k_neg in k_grid:
                    d, y = promoter_filter_df(
                        df_nc, tagset_name=tagset_name, tss_bp=tss,
                        use_match_group=use_mg, k_neg=(k_neg or 0)
                    )
                    pos, neg = count_pos_neg(y)
                    err = abs(pos - target_pos) + abs(neg - target_neg)
                    rows.append(dict(
                        tagset=tagset_name,
                        tss_bp=float(tss) if tss is not None else np.nan,
                        use_match_group=use_mg,
                        k_neg=float(k_neg) if k_neg is not None else np.nan,
                        pos=pos, neg=neg, total=pos+neg, err=err
                    ))
    return pd.DataFrame(rows).sort_values(["err", "total", "tagset", "tss_bp", "k_neg"])


# ----------------------------
# benchmark specs
# ----------------------------

@dataclass
class Spec:
    name: str
    hf: str
    metric: str
    expected_pos: Optional[int]
    expected_neg: Optional[int]
    split: str = "test"
    subset: Optional[str] = None
    postprocess: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None

SPECS = [
    Spec("A_clinvar_pathogenic_vs_benign_missense", "songlab/clinvar_vs_benign", "AUROC", 21204, 26845),
    Spec("B_cosmic_frequent_in_cancer_vs_common_missense", "songlab/cosmic", "AUPRC", 182, 15080),
    Spec("C_gwas_finemapped_missense_causal_vs_matched", "songlab/ukb_finemapped_coding", "AUPRC", 224, 1997),
    Spec("E_omim_noncoding_pathogenic_vs_common", "songlab/omim_traitgym", "AUPRC", 338, 2968),
    Spec("G_gwas_finemapped_noncoding_causal_vs_matched", "songlab/ukb_finemapped_nc_traitgym", "AUPRC", 1113, 10036),
]


def verify_spec(spec: Spec, debug: bool) -> Tuple[int, int, int, str]:
    df = load_df(spec.hf, split=spec.split, subset=spec.subset)
    if spec.postprocess is not None:
        df = spec.postprocess(df)

    if debug:
        schema_debug(df, spec.name)

    y = normalize_binary_label(df["label"], dataset_name=spec.name)
    pos, neg = count_pos_neg(y)
    total = len(y)
    status = mismatch_status(pos, neg, spec.expected_pos, spec.expected_neg)

    print(f"\n=== {spec.name} ===")
    print(f"dataset: {spec.hf}")
    print(f"metric: {spec.metric}")
    print(f"pos={pos}, neg={neg}, total={total}")
    print(f"status: {status}")

    return pos, neg, total, status


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--debug_schema", action="store_true")

    # promoter target from paper figure
    ap.add_argument("--promoter_target_pos", type=int, default=119)
    ap.add_argument("--promoter_target_neg", type=int, default=880)
    ap.add_argument("--promoter_topk", type=int, default=20)

    # fixed promoter rule (default: your current best-ish)
    ap.add_argument("--promoter_tagset", type=str, default="PROMOTER_CCRE(broad)", choices=list(PROMOTER_TAGSETS.keys()))
    ap.add_argument("--promoter_tss_bp", type=int, default=1500)  # set to 1500 from your best row
    ap.add_argument("--promoter_use_match_group", action="store_true", default=True)
    ap.add_argument("--promoter_k_neg", type=int, default=10)

    # if you want to recompute best rule every run
    ap.add_argument("--sweep_promoter", action="store_true")

    args = ap.parse_args()

    # 1) verify A/B/C/E/G
    results = []
    for spec in SPECS:
        pos, neg, total, status = verify_spec(spec, debug=args.debug_schema)
        results.append(dict(name=spec.name, hf=spec.hf, metric=spec.metric, pos=pos, neg=neg, total=total, status=status))

    # 2) build H as derived promoter benchmark from ukb_finemapped_nc_traitgym
    df_nc = load_df("songlab/ukb_finemapped_nc_traitgym", split="test")
    if args.debug_schema:
        schema_debug(df_nc, "PROMOTER_SOURCE: songlab/ukb_finemapped_nc_traitgym")

    if args.sweep_promoter:
        sweep = sweep_promoter_rules(df_nc, target_pos=args.promoter_target_pos, target_neg=args.promoter_target_neg)
        print("\n=== promoter dial-in sweep (within ukb_finemapped_nc_traitgym) ===")
        print(f"target: pos={args.promoter_target_pos}, neg={args.promoter_target_neg}, total={args.promoter_target_pos + args.promoter_target_neg}")
        print(sweep.head(args.promoter_topk).to_string(index=False))
        best = sweep.iloc[0].to_dict()
        print("\n=== best promoter rule (by L1 error to target) ===")
        print(best)

    dfH, yH = promoter_filter_df(
        df_nc,
        tagset_name=args.promoter_tagset,
        tss_bp=args.promoter_tss_bp,
        use_match_group=args.promoter_use_match_group,
        k_neg=args.promoter_k_neg,
    )
    posH, negH = count_pos_neg(yH)
    totalH = posH + negH
    statusH = mismatch_status(posH, negH, args.promoter_target_pos, args.promoter_target_neg)

    nameH = "H_promoter_variants_derived_from_ukb_nc"
    print(f"\n=== {nameH} ===")
    print("source dataset: songlab/ukb_finemapped_nc_traitgym")
    print(f"rule: tagset={args.promoter_tagset}, tss_bp={args.promoter_tss_bp}, "
          f"use_match_group={args.promoter_use_match_group}, k_neg={args.promoter_k_neg}")
    print("metric: AUPRC")
    print(f"pos={posH}, neg={negH}, total={totalH}")
    print(f"status: {statusH}")

    results.append(dict(
        name=nameH,
        hf="songlab/ukb_finemapped_nc_traitgym (derived subset)",
        metric="AUPRC",
        pos=posH, neg=negH, total=totalH, status=statusH
    ))

    # 3) summary
    print("\n=== summary ===")
    summary = pd.DataFrame(results)
    print(summary[["name", "metric", "pos", "neg", "total", "status"]].to_string(index=False))


if __name__ == "__main__":
    main()
