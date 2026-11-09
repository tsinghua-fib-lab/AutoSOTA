import hashlib
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd

from . import nullrec
from .folding import bprna_string, compute_bpps, eternafold_mfe, load_bpps
from .paths import DATA, ERRORS, MODEL_COLS, TARGETS

FEAT_CACHE = os.path.join(DATA, "features")
MODEL_ERRORS = [ERRORS[TARGETS.index(c)] for c in MODEL_COLS]


def load_train():
    return pd.read_json(os.path.join(DATA, "train.json"), lines=True)


def load_test():
    return pd.read_json(os.path.join(DATA, "test.json"), lines=True)


def stack_targets(df, cols, n=68):
    return np.stack([np.stack([np.asarray(v, dtype=np.float32)[:n] for v in df[c]]) for c in cols], axis=-1)


def train_targets(df, err_max=10.0, snr_min=1.5):
    """[N, 68, 5] targets/errors in MODEL_COLS order. Positions with error > err_max and
    value/error < snr_min are NaN-ed (1st place writeup, 'Data preprocessing' #1)."""
    y = stack_targets(df, MODEL_COLS)
    e = stack_targets(df, MODEL_ERRORS)
    with np.errstate(invalid="ignore", divide="ignore"):
        bad = (e > err_max) & (y / e < snr_min)
    y = np.where(bad, np.nan, y)
    e = np.where(np.isnan(y), 0, e)
    return y, e


def _features_chunk(args):
    seqs, structs, loops, ids = args
    return nullrec.build_inputs(seqs, structs, loops, [load_bpps(i) for i in ids])


def feature_paths(df, name):
    h = hashlib.md5("".join(df.id + df.sequence + df.structure + df.predicted_loop_type).encode()).hexdigest()[:10]
    return (os.path.join(FEAT_CACHE, "%s_%s_X.npy" % (name, h)),
            os.path.join(FEAT_CACHE, "%s_%s_A.npy" % (name, h)))


def build_features(df, name, processes=64, chunk=64):
    """Cached (X [N, L+2, F], A [N, L+2, L+2, 8]) for an equal-length frame."""
    os.makedirs(FEAT_CACHE, exist_ok=True)
    fx, fa = feature_paths(df, name)
    if os.path.exists(fx) and os.path.exists(fa):
        return np.load(fx), np.load(fa, mmap_mode="r")
    jobs = []
    for s in range(0, len(df), chunk):
        g = df.iloc[s:s + chunk]
        jobs.append((list(g.sequence), list(g.structure), list(g.predicted_loop_type), list(g.id)))
    with Pool(processes) as p:
        res = p.map(_features_chunk, jobs)
    x = np.concatenate([r[0] for r in res])
    a = np.concatenate([r[1] for r in res])
    np.save(fx, x)
    np.save(fa, a)
    return x, np.load(fa, mmap_mode="r")


def prepare_frame(df, loop_source):
    df = df.copy()
    if loop_source == "bprna":
        df["predicted_loop_type"] = [bprna_string(s) for s in df.structure]
    return df


# ---------------------------------------------------------------- clustering / weights


def edit_distance_clusters(seqs, threshold):
    """Single-linkage clusters of sequences whose Levenshtein distance <= threshold."""
    from rapidfuzz.distance import Levenshtein
    from rapidfuzz.process import cdist

    d = cdist(seqs, seqs, scorer=Levenshtein.distance, workers=-1)
    n = len(seqs)
    parent = np.arange(n)

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    ii, jj = np.where(np.triu(d <= threshold, 1))
    for i, j in zip(ii, jj):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj
    roots = np.array([find(i) for i in range(n)])
    _, labels = np.unique(roots, return_inverse=True)
    return labels


def cluster_weights(labels):
    counts = np.bincount(labels)
    return 1.0 / np.sqrt(counts[labels])


# ---------------------------------------------------------------- random sequences for pseudo-labelling

PREFIX = "GGAAA"
SUFFIX = "AAAAGAAACAACAACAACAAC"


def random_sequences(n, length, templates, mutate_frac=0.5, rate=(0.1, 0.5), seed=0):
    """Half mutated copies of real designs, half uniform random, with Kaggle's constant flanks."""
    rng = np.random.default_rng(seed)
    out = set()
    core = length - len(PREFIX) - len(SUFFIX)
    templates = [t for t in templates if len(t) == length]
    while len(out) < n:
        if templates and rng.random() < mutate_frac:
            t = list(templates[rng.integers(len(templates))])
            r = rng.uniform(*rate)
            for i in range(len(PREFIX), length - len(SUFFIX)):
                if rng.random() < r:
                    t[i] = "ACGU"[rng.integers(4)]
            s = "".join(t)
        else:
            s = PREFIX + "".join(rng.choice(list("ACGU"), core)) + SUFFIX
        out.add(s)
    return sorted(out)


def make_random_frame(n, length, templates, seed=0, processes=100):
    seqs = random_sequences(n, length, templates, seed=seed)
    with Pool(processes) as p:
        structs = p.map(eternafold_mfe, seqs)
    df = pd.DataFrame({
        "id": ["id_rand%d_%06d" % (seed, i) for i in range(len(seqs))],
        "sequence": seqs,
        "structure": structs,
        "seq_length": length,
    })
    df["predicted_loop_type"] = [bprna_string(s) for s in structs]
    compute_bpps(zip(df.id, df.sequence))
    return df
