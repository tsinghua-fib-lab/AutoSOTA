#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd

DEFAULT_ROOTS = [
    "/home/mica/gamba/other-models/final_representations/gamba_onepass",
    "/home/mica/gamba/other-models/final_representations/all_tasks",
]

PAIR_ID_CANDIDATES = ["pair_id", "pairid", "pairID", "pair_id_feature", "pair_id_control"]
ROLE_CANDIDATES = ["role", "region_type", "regiontype", "region_kind", "regionkind"]

TASK_CANDIDATES  = ["task", "pair_type", "comparison", "dataset", "mode"]
CAT_CANDIDATES   = ["category", "cat", "roi_category"]
SCOPE_CANDIDATES = ["scope", "window", "context", "region_scope"]
GROUP_CANDIDATES = ["group", "split", "fold", "partition"]


def pick_col(cols, candidates):
    cols_l = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in cols_l:
            return cols_l[cand.lower()]
    return None


def normalize_role(x: str) -> str:
    s = str(x)
    s = s.replace("random_noannot", "random-noannot")
    s = s.replace("random-no-annot", "random-noannot")
    s = s.replace("random_no_annot", "random-noannot")
    return s


def infer_model_name(meta_path: Path) -> str:
    parts = meta_path.parts
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] in ("gamba_onepass", "all_tasks"):
            if i + 1 < len(parts):
                return parts[i + 1]
    return meta_path.parents[1].name


def infer_family(meta_path: Path) -> str:
    parts = set(meta_path.parts)
    if "gamba_onepass" in parts:
        return "gamba_onepass"
    if "all_tasks" in parts:
        return "all_tasks"
    return "unknown"


def infer_group_from_path(meta_path: Path) -> str:
    name = meta_path.name
    for g in ("all", "train", "test", "val", "valid"):
        if f"_{g}_" in name or name.endswith(f"_{g}_meta.parquet"):
            return g
    for part in meta_path.parts:
        if part in ("all", "train", "test", "val", "valid"):
            return part
    return "all"


def load_cache_meta(p: Path) -> pd.DataFrame:
    df = pd.read_parquet(p)
    df.columns = [c.strip() for c in df.columns]
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", action="append", default=[], help="root(s) to search recursively")
    ap.add_argument("--cache", action="append", default=[], help="explicit meta parquet path(s)")
    ap.add_argument("--only_all", action="store_true", help="only include *_all_meta.parquet caches")
    ap.add_argument("--no_labelset_check", action="store_true", help="skip pair_id -> {labels} check")
    args = ap.parse_args()

    roots = args.root if args.root else DEFAULT_ROOTS

    meta_paths = [Path(p) for p in args.cache]
    for r in roots:
        rr = Path(r)
        meta_paths += list(rr.rglob("cache_*_meta.parquet"))
        meta_paths += list(rr.rglob("*cache*_meta.parquet"))

    meta_paths = sorted({p.resolve() for p in meta_paths if p.exists()})
    if args.only_all:
        meta_paths = [p for p in meta_paths if "_all_meta.parquet" in p.name]

    if not meta_paths:
        raise SystemExit("found no cache_*_meta.parquet files under given roots/paths")

    rows = []
    included_models = set()
    included_paths = []

    for mp in meta_paths:
        try:
            df = load_cache_meta(mp)
        except Exception as e:
            print(f"[skip] {mp}: failed to read parquet: {e}")
            continue

        pair_col = pick_col(df.columns, PAIR_ID_CANDIDATES)
        if pair_col is None:
            print(f"[skip] {mp}: no pair_id-like column. cols={list(df.columns)[:20]} ...")
            continue

        role_col = pick_col(df.columns, ROLE_CANDIDATES)
        if role_col is None:
            print(f"[skip] {mp}: missing role/region_type column. cols={list(df.columns)[:20]} ...")
            continue

        task_col  = pick_col(df.columns, TASK_CANDIDATES)
        cat_col   = pick_col(df.columns, CAT_CANDIDATES)
        scope_col = pick_col(df.columns, SCOPE_CANDIDATES)
        group_col = pick_col(df.columns, GROUP_CANDIDATES)

        model = infer_model_name(mp)
        family = infer_family(mp)

        included_models.add(model)
        included_paths.append(str(mp))

        df["_model"] = model
        df["_family"] = family
        df["_pair_id"] = df[pair_col].astype(str)
        df["_label"] = df[role_col].astype(str).map(normalize_role)

        df["_task"] = df[task_col].astype(str) if task_col else "__notask__"
        df["_category"] = df[cat_col].astype(str) if cat_col else "__nocat__"
        df["_scope"] = df[scope_col].astype(str) if scope_col else "__noscope__"
        df["_group"] = df[group_col].astype(str) if group_col else infer_group_from_path(mp)

        gcols = ["_family", "_model", "_task", "_category", "_scope", "_group", "_label"]
        for keys, sub in df.groupby(gcols, dropna=False):
            pair_ids = set(sub["_pair_id"].unique().tolist())
            rows.append({
                "family": keys[0],
                "model": keys[1],
                "task": keys[2],
                "category": keys[3],
                "scope": keys[4],
                "group": keys[5],
                "label": keys[6],
                "n_pairs": len(pair_ids),
                "pair_ids": pair_ids,
                "meta_path": str(mp),
            })

    if not rows:
        raise SystemExit("no usable cache meta found with pair_id columns")

    print("\n=== included models ===")
    print(sorted(included_models))
    print("\n=== included meta paths (first 30) ===")
    for p in included_paths[:30]:
        print(p)

    summary = pd.DataFrame(rows)

    # 1) set equality per label-bucket, within family
    report = []
    bucket_cols = ["family", "task", "category", "scope", "group", "label"]
    for bucket, sub in summary.groupby(bucket_cols):
        sub = sub.sort_values("model")
        ref_model = sub["model"].iloc[0]
        ref_set = sub["pair_ids"].iloc[0]
        any_mismatch = False

        for _, r in sub.iterrows():
            s = r["pair_ids"]
            if s != ref_set:
                any_mismatch = True
                report.append({
                    **dict(zip(bucket_cols, bucket)),
                    "status": "mismatch",
                    "ref_model": ref_model,
                    "model": r["model"],
                    "n_ref": len(ref_set),
                    "n_model": len(s),
                    "n_missing": len(ref_set - s),
                    "n_extra": len(s - ref_set),
                    "example_missing": ",".join(list(sorted(ref_set - s))[:5]),
                    "example_extra": ",".join(list(sorted(s - ref_set))[:5]),
                    "meta_path": r["meta_path"],
                })

        if not any_mismatch:
            report.append({
                **dict(zip(bucket_cols, bucket)),
                "status": "all_match",
                "ref_model": ref_model,
                "model": "__ALL__",
                "n_ref": len(ref_set),
                "n_model": len(ref_set),
                "n_missing": 0,
                "n_extra": 0,
                "example_missing": "",
                "example_extra": "",
                "meta_path": "",
            })

    rep = pd.DataFrame(report).sort_values(["status"] + bucket_cols + ["model"])
    pd.set_option("display.max_rows", 5000)
    pd.set_option("display.max_colwidth", 120)
    print("\n=== pair_id set equality across models (within family; by task/category/scope/group/label) ===")
    print(rep.to_string(index=False))

    # 2) correct check: pair_id -> set(labels) consistency within a bucket (family, task, category, scope, group)
    if not args.no_labelset_check:
        print("\n=== pair_id -> label-set consistency (within family/task/category/scope/group) ===")

        labelset_rows = []
        for (family, task, category, scope, group), sub in summary.groupby(["family", "task", "category", "scope", "group"]):
            for model, msub in sub.groupby("model"):
                # build pid -> set(labels)
                pid_to_labels = {}
                for _, r in msub.iterrows():
                    lab = r["label"]
                    for pid in r["pair_ids"]:
                        pid_to_labels.setdefault(pid, set()).add(lab)
                labelset_rows.append({
                    "family": family, "task": task, "category": category, "scope": scope, "group": group,
                    "model": model,
                    "n_pair_ids": len(pid_to_labels),
                    "pid_to_labels": pid_to_labels,
                })

        ls = pd.DataFrame(labelset_rows)

        ls_report = []
        for bucket, sub in ls.groupby(["family", "task", "category", "scope", "group"]):
            sub = sub.sort_values("model")
            ref_model = sub["model"].iloc[0]
            ref_map = sub["pid_to_labels"].iloc[0]

            for _, r in sub.iterrows():
                m = r["pid_to_labels"]
                inter = set(ref_map.keys()) & set(m.keys())
                diffs = [pid for pid in inter if ref_map[pid] != m[pid]]
                ls_report.append({
                    "family": bucket[0], "task": bucket[1], "category": bucket[2], "scope": bucket[3], "group": bucket[4],
                    "ref_model": ref_model, "model": r["model"],
                    "n_ref_pair_ids": len(ref_map),
                    "n_model_pair_ids": len(m),
                    "n_labelset_diffs_on_intersection": len(diffs),
                    "example_diff_pair_ids": ",".join(diffs[:5]),
                })

        ls_rep = pd.DataFrame(ls_report).sort_values(["n_labelset_diffs_on_intersection"], ascending=False)
        print(ls_rep.to_string(index=False))
        ls_rep.to_csv(
            "/home/mica/gamba/other-models/final_representations/pair_id_labelset_mapping_check.csv",
            index=False,
        )
    # ---------------------------------------------------------------------
    # 1b) cross-family ROI equality: compare all_tasks vs gamba_onepass
    # ---------------------------------------------------------------------
    print("\n=== cross-family ROI pair_id equality (all_tasks vs gamba_onepass) ===")

    cf = summary.copy()

    # focus on ROIs only; adjust if your ROI label differs
    cf = cf[cf["label"] == "roi"].copy()

    # aggregate per (family, task, category, scope, group, label) -> union of pair_ids
    cf_bucket_cols = ["family", "task", "category", "scope", "group", "label"]
    cf_agg = (
        cf.groupby(cf_bucket_cols, dropna=False)["pair_ids"]
        .apply(lambda xs: set().union(*xs))
        .reset_index()
    )

    # pivot families into columns
    pivot_cols = ["task", "category", "scope", "group", "label"]
    cf_piv = cf_agg.pivot_table(
        index=pivot_cols,
        columns="family",
        values="pair_ids",
        aggfunc="first",
    )

    def _as_set(x):
        return x if isinstance(x, set) else set()

    rows_cf = []
    for idx, row in cf_piv.iterrows():
        s_all = _as_set(row.get("all_tasks"))
        s_gop = _as_set(row.get("gamba_onepass"))

        missing_in_gop = s_all - s_gop
        extra_in_gop = s_gop - s_all

        status = "match" if (not missing_in_gop and not extra_in_gop) else "mismatch"
        rows_cf.append({
            "task": idx[0],
            "category": idx[1],
            "scope": idx[2],
            "group": idx[3],
            "label": idx[4],
            "status": status,
            "n_all_tasks": len(s_all),
            "n_gamba_onepass": len(s_gop),
            "n_missing_in_gamba_onepass": len(missing_in_gop),
            "n_extra_in_gamba_onepass": len(extra_in_gop),
            "example_missing": ",".join(sorted(list(missing_in_gop))[:5]),
            "example_extra": ",".join(sorted(list(extra_in_gop))[:5]),
        })

    cf_rep = pd.DataFrame(rows_cf).sort_values(
        ["status", "task", "category", "scope", "group", "label"]
    )
    pd.set_option("display.max_rows", 5000)
    pd.set_option("display.max_colwidth", 120)
    print(cf_rep.to_string(index=False))

    # optional: fail hard if any mismatch
    if (cf_rep["status"] == "mismatch").any():
        raise SystemExit("cross-family ROI mismatch detected")



if __name__ == "__main__":
    main()
