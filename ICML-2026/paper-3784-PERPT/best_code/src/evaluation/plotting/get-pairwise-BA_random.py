#!/usr/bin/env python3
import os
import math
import argparse
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import confusion_matrix

# ---------------- config ----------------

CATEGORIES = [
    "vista_enhancer",
    "UCNE",
    "repeats",
    "exons",
    "introns",
    "noncoding_regions",
    "coding_regions",
    "upstream_TSS",
    "UTR5",
    "UTR3",
    "promoters",
]

# for NT we currently only have random ROI, not full
NT_SCOPES = ["roi"]

NT_MODELS = [
    "hyenaDNA-random-init",
    "hyenaDNA",
    "phyloGPN",
    "nt-ms",
    "nt-human",
    "phyloGPN-random-init",
    "nt-ms-random-init",
    "nt-human-random-init",
    "caduceus-theirs",
    "caduceus-theirs-random-init",
    "evo2"
]

# feature embeddings (multiclass global reps)
NT_FEATURE_ROOT = "/home/mica/NucleotideTransformer/final_representations/random_pairs/"

# random-only embeddings for NT
NT_RANDOM_ROOT = "/home/mica/NucleotideTransformer/final_representations/random_pairs/"

# baselines (kmer6, phylop) – already combined feature+random per category
BASELINE_MODELS = ["kmer6", "phylop"]
BASELINE_ROOT = (
    "/home/mica/gamba/data_processing/data/240-mammalian/"
    "final_representations/random_pairs/baseline"
)
# baselines usually have both scopes
BASELINE_SCOPES = ["roi", "full"]


# ---------------- core metrics ----------------

def compute_ba_and_se(X: np.ndarray, y: np.ndarray):
    """
    LOO 1-NN balanced accuracy and SE (binary or multiclass).
    X: [N, D], y: [N]
    """
    X = np.asarray(X)
    y = np.asarray(y)

    if X.shape[0] < 3:
        raise ValueError(f"too few samples: {X.shape[0]}")

    nn = NearestNeighbors(n_neighbors=2, metric="cosine")
    nn.fit(X)
    _, idx = nn.kneighbors(X)
    # idx[:, 0] is self, idx[:, 1] is nearest neighbor excluding self
    y_pred = y[idx[:, 1]]

    classes = np.unique(y)
    cm = confusion_matrix(y, y_pred, labels=classes)
    n_per_class = cm.sum(axis=1).astype(float)
    correct_per_class = np.diag(cm).astype(float)

    # avoid div by zero for empty classes
    with np.errstate(divide="ignore", invalid="ignore"):
        recalls = np.divide(
            correct_per_class,
            np.where(n_per_class == 0, 1.0, n_per_class),
        )

    K = len(classes)
    ba = float(np.nanmean(recalls))  # 0–1

    # variance from per-class binomial approximation
    var = np.nansum(
        recalls * (1.0 - recalls) / np.where(n_per_class == 0, np.inf, n_per_class)
    ) / (K ** 2)
    se = math.sqrt(max(var, 0.0))

    return ba * 100.0, se * 100.0  # percent


def compute_ba_and_se_from_npz(npz_path: str):
    """convenience wrapper for baseline npz files that already contain labels."""
    z = np.load(npz_path, allow_pickle=True)
    X = np.asarray(z["embeddings"])
    y = np.asarray(z["labels"])
    return compute_ba_and_se(X, y)


# ---------------- loaders: NT feature + random ----------------
def load_nt_feature_category(
    model: str,
    category: str,
    scope: str,
    group_name: str = "all",
):
    """
    load feature embeddings for a single category from per-category NT reps.

    expected files:
      NT_FEATURE_ROOT/<model>/
        reps_<model>_<group>_<category>_<scope>.npz
        reps_<model>_<group>_<category>_<scope>_meta.parquet
        OR:
        reps_<model>_<group>_<category>_<scope>.npz
        reps_<model>_<group>_<category>_<scope>_meta.parquet
    """
    model_dir = os.path.join(NT_FEATURE_ROOT, model)
    if not os.path.isdir(model_dir):
        print(f"[warn] missing NT feature dir: {model_dir}")
        return None, None

    base = os.path.join(
        model_dir,
        f"reps_{model}_{group_name}_{category}_random_{scope}",
    )
    npz_path = base + ".npz"
    meta_path = base + "_meta.parquet"

    if not os.path.exists(npz_path) or not os.path.exists(meta_path):
        # Try the alternative naming convention
        base = os.path.join(
            model_dir,
            f"reps_{model}_{group_name}_{category}_{scope}",
        )
        npz_path = base + ".npz"
        meta_path = base + "_meta.parquet"

        if not os.path.exists(npz_path) or not os.path.exists(meta_path):
            print(f"[warn] missing NT random files: {npz_path} / {meta_path}")
            return None, None

    z = np.load(npz_path, allow_pickle=True)

    X = np.asarray(z["embeddings"])
    y = np.asarray(z["labels"])

    # boolean mask for nonrandom labels
    mask = y != "random"

    # filtered embeddings + labels
    X_random = X[mask]
    y_random = y[mask]

    meta = pd.read_parquet(meta_path)

    if X.shape[0] < 3:
        print(
            f"[warn] NT {model} {scope} {category}: "
            f"too few feature samples ({X.shape[0]})"
        )
        return None, None

    # everything in this file is this category, so the label is just "feature"
    y = np.full(X.shape[0], "feature", dtype=object)
    return X, y


def load_nt_random_category(
    model: str,
    category: str,
    scope: str,
    group_name: str = "all",
):
    """
    load random embeddings for a single category/model/scope.

    expected files (from your random script):
      NT_RANDOM_ROOT/<model>/
        reps_<model>_<group>_<category>_random_<scope>.npz
        reps_<model>_<group>_<category>_random_<scope>_meta.parquet
        OR:
        reps_<model>_<group>_<category>_<scope>.npz
        reps_<model>_<group>_<category>_<scope>_meta.parquet
    """
    model_dir = os.path.join(NT_RANDOM_ROOT, model)
    if not os.path.isdir(model_dir):
        print(f"[warn] missing NT random dir: {model_dir}")
        return None, None

    base = os.path.join(
        model_dir,
        f"reps_{model}_{group_name}_{category}_random_{scope}",
    )
    npz_path = base + ".npz"
    meta_path = base + "_meta.parquet"

    if not os.path.exists(npz_path) or not os.path.exists(meta_path):
        # Try the alternative naming convention
        base = os.path.join(
            model_dir,
            f"reps_{model}_{group_name}_{category}_{scope}",
        )
        npz_path = base + ".npz"
        meta_path = base + "_meta.parquet"

        if not os.path.exists(npz_path) or not os.path.exists(meta_path):
            print(f"[warn] missing NT random files: {npz_path} / {meta_path}")
            return None, None

    z = np.load(npz_path, allow_pickle=True)

    X = np.asarray(z["embeddings"])
    y = np.asarray(z["labels"])

    # boolean mask for random labels
    mask = y == "random"

    # filtered embeddings + labels
    X_random = X[mask]
    y_random = y[mask]


    if X.shape[0] < 3:
        print(f"[warn] NT {model} {scope} {category}: too few random samples ({X.shape[0]})")
        return None, None

    y = np.full(X.shape[0], "random", dtype=object)
    return X, y


# ---------------- collectors ----------------

def collect_nt_random_vs_category_rows(group_name: str = "all"):
    """
    for each NT model, category, scope:
      - load feature embeddings for that category from global reps
      - load random embeddings for that category from random reps
      - balance classes
      - compute LOO 1-NN BA + SE
    """
    rows = []
    for model in NT_MODELS:
        for scope in NT_SCOPES:
            for cat in CATEGORIES:
                X_feat, y_feat = load_nt_feature_category(model, cat, scope, group_name)
                X_rand, y_rand = load_nt_random_category(model, cat, scope, group_name)

                if X_feat is None or X_rand is None:
                    continue

                # balance classes
                n = min(X_feat.shape[0], X_rand.shape[0])
                if n < 5:
                    print(
                        f"[warn] {model} {scope} {cat}: "
                        f"too few after balancing (n_feat={X_feat.shape[0]}, "
                        f"n_rand={X_rand.shape[0]})"
                    )
                    continue

                rng = np.random.default_rng(1337)
                idx_feat = rng.choice(X_feat.shape[0], size=n, replace=False)
                idx_rand = rng.choice(X_rand.shape[0], size=n, replace=False)

                X = np.concatenate([X_feat[idx_feat], X_rand[idx_rand]], axis=0)
                y = np.concatenate([y_feat[idx_feat], y_rand[idx_rand]], axis=0)

                try:
                    ba, se = compute_ba_and_se(X, y)
                except Exception as e:
                    print(f"[skip] {model} {scope} {cat}: {e}")
                    continue

                rows.append(
                    dict(
                        Model=model,
                        Family="NT",
                        Group=group_name,
                        Category=cat,
                        Scope=scope,
                        BA_pct=ba,
                        BA_SE_pct=se,
                        N_feat=int(n),
                        N_rand=int(n),
                    )
                )
    return rows


def collect_baseline_rows():
    """
    baselines live under:
      BASELINE_ROOT/<model>/all/<category>/
        reps_<model>_all_<category>_<scope>.npz

    and each npz already has combined feature+random labels.
    """
    rows = []
    for model in BASELINE_MODELS:
        for scope in BASELINE_SCOPES:
            for cat in CATEGORIES:
                npz_path = os.path.join(
                    BASELINE_ROOT,
                    model,
                    "all",
                    cat,
                    f"reps_{model}_all_{cat}_{scope}.npz",
                )
                if not os.path.exists(npz_path):
                    print(f"[warn] missing baseline npz: {npz_path}")
                    continue

                try:
                    ba, se = compute_ba_and_se_from_npz(npz_path)
                except Exception as e:
                    print(f"[skip] baseline {model} {scope} {cat}: {e}")
                    continue

                rows.append(
                    dict(
                        Model=model,
                        Family="baseline",
                        Group="all",
                        Category=cat,
                        Scope=scope,
                        BA_pct=ba,
                        BA_SE_pct=se,
                        # N_feat / N_rand unknown from combined npz; you can add later if needed
                        N_feat=np.nan,
                        N_rand=np.nan,
                    )
                )
    return rows


# ---------------- aggregation ----------------

def aggregate_global(df: pd.DataFrame) -> pd.DataFrame:
    """aggregate per-category BA into per-model global BA across categories."""
    summaries = []
    for (model, group, scope), sub in df.groupby(["Model", "Group", "Scope"]):
        ba_vals = sub["BA_pct"].to_numpy()
        se_vals = sub["BA_SE_pct"].to_numpy()

        K = len(ba_vals)
        if K == 0:
            continue

        ba_global = float(np.mean(ba_vals))

        # combine SEs assuming independence
        var_i = (se_vals / 100.0) ** 2
        var_global = float(np.sum(var_i) / (K ** 2))
        se_global = math.sqrt(max(var_global, 0.0)) * 100.0

        std_across_cats = float(np.std(ba_vals, ddof=1)) if K > 1 else 0.0

        summaries.append(
            dict(
                Model=model,
                Group=group,
                Scope=scope,
                N_Categories=K,
                GlobalBalancedAccuracyPct=ba_global,
                GlobalBalancedAccuracyStdPct=std_across_cats,
                GlobalBalancedAccuracySEPct=se_global,
            )
        )
    return pd.DataFrame(summaries)


# ---------------- main ----------------

def main():
    ap = argparse.ArgumentParser(
        description="feature vs random 1-NN BA using NT global reps + random reps + baselines"
    )
    ap.add_argument(
        "-o",
        "--outdir",
        default=(
            "/home/mica/gamba/data_processing/data/240-mammalian/"
            "global_balacc_random"
        ),
        help="output directory for per-category and global summary TSVs",
    )
    ap.add_argument("--group_name", type=str, default="all")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    rows_nt = collect_nt_random_vs_category_rows(group_name=args.group_name)
    rows_bl = collect_baseline_rows()

    rows = rows_nt + rows_bl
    if not rows:
        raise SystemExit("no rows collected; check paths and config.")

    df = pd.DataFrame(rows)
    percat_path = os.path.join(
        args.outdir,
        "global_balacc_random_per_model.tsv",
    )
    df.to_csv(percat_path, sep="\t", index=False)
    print(f"[info] wrote per-category table: {percat_path}")

    df_global = aggregate_global(df)
    global_path = os.path.join(
        args.outdir,
        "global_balacc_random.tsv",
    )
    df_global.to_csv(global_path, sep="\t", index=False)
    print(f"[info] wrote global table: {global_path}")


if __name__ == "__main__":
    main()
