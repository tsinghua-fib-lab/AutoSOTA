"""Evolutionary (MSA) expert for S2F scoring — IDEA-601 (round 6).

Builds a position-specific amino-acid frequency profile (weighted PSSM) from the
assay's ProteinGym alignment (DMS_msa_files/*.a2m, subMSA restricted to the
assayed region) and scores a mutation at position i by

    log f(mt, i) - log f(wt, i)

which is mutation-independent per position, so it caches per protein exactly like
the wild-type ESM marginal does.

Coordinate mapping (verified empirically on 2 assays, 524/524 and 161/161 exact
column matches, plus a hard assert in build_pssm for every assay):
  - The first a2m row is the query and spans target positions
    [MSA_start, MSA_end] with one character per position.
  - A column where the query character is UPPERCASE is a real focus column:
    raw column j -> target position MSA_start + j. (Insertion columns have a
    lowercase query character and contain no uppercase residue in any row.)
  - In other rows, lowercase letters are insertions relative to the query and
    dots are gap-like; both are stripped. '-' gaps are kept.

Sequence weights:
  ProteinGym ships precomputed weights (DMS_msa_weights/), but those were
  computed on the FULL-length alignment (identity threshold theta chosen for
  full-length comparisons); on the subMSA's few focus columns the same theta
  degenerates (measured on NEIME: 99.5% of sequence pairs are neighbors at
  theta=0.2, weights collapse to ~uniform). Uniform weights are a known-worse
  evolutionary baseline, so we reweight ourselves with the standard
  Henikoff & Henikoff (1994) position-based scheme, which needs no threshold:

      w_i = sum_c 1 / (r_c * n_c)

  where r_c is the number of distinct residues at column c and n_c the number
  of rows covering c. Rows carrying non-standard residues (X, B, Z, ...) are
  excluded from the counts. The query row is included (EVE convention).

Frequencies get a small Dirichlet-style pseudocount (0.5 per amino acid) so an
unobserved mutant residue gets a finite log-odds instead of -inf.

Positions with no aligned column (insertion columns, or outside
[MSA_start, MSA_end]) have NO coverage; the caller must fall back to the
model-only score for such mutations (a PSSM log-odds of 0 would corrupt the
ranking).

Caching: one npz per assay under dataset/DMS_pssm_cache/ (gitignored), storing
(positions (N,), logf (N, 20)) with the amino-acid order AA_ORDER.
"""
import os
import string

import numpy as np

AA_ORDER = sorted("ACDEFGHIKLMNPQRSTVWY")
AA2IDX = {aa: i for i, aa in enumerate(AA_ORDER)}
NONSTANDARD = set("XBUZJO")

DEFAULT_CACHE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "dataset", "DMS_pssm_cache")

_STD = set(AA_ORDER)
_STRIP = dict.fromkeys(string.ascii_lowercase + ".", None)
_STRIP_TRANS = str.maketrans(_STRIP)


def load_a2m(path):
    """Return the alignment rows (sequences only, headers discarded)."""
    rows = []
    with open(path) as f:
        for line in f:
            if line.startswith(">"):
                rows.append("")
            else:
                rows[-1] += line.rstrip()
    return rows


def strip_insertions(seq):
    """Delete lowercase letters and dots (insertions); keep '-' gaps."""
    return seq.translate(_STRIP_TRANS)


def henikoff_weights(arr):
    """Henikoff & Henikoff (1994) position-based sequence weights.

    arr: (N, L) uint8 array over AA_ORDER + '-'.
    """
    n_rows, n_cols = arr.shape
    gap = arr == ord("-")
    n_c = (~gap).sum(axis=0)
    w = np.zeros(n_rows, dtype=np.float64)
    for c in range(n_cols):
        if n_c[c] == 0:
            continue
        r_c = len(np.unique(arr[~gap[:, c], c]))
        w[~gap[:, c]] += 1.0 / (r_c * n_c[c])
    return w


def build_pssm(assay_meta, msa_root, cache_dir=None, verbose=False):
    """Build (positions, logf) for one assay; cache to cache_dir.

    assay_meta: dict row from DMS_substitutions.csv with keys
        DMS_id, MSA_filename, MSA_start, MSA_end, target_seq.
    Returns (positions: np.ndarray (N,) 1-indexed, logf: np.ndarray (N, 20),
             neff: float). Returns (None, None, 0.0) when the assay has no MSA.
    """
    cache_dir = cache_dir or DEFAULT_CACHE_DIR
    msa_name = assay_meta["MSA_filename"]
    cache_path = os.path.join(cache_dir, assay_meta["DMS_id"] + ".npz")

    if os.path.isfile(cache_path):
        z = np.load(cache_path)
        return z["positions"], z["logf"], float(z["neff"])

    msa_path = os.path.join(msa_root, msa_name)
    if not os.path.isfile(msa_path):
        return None, None, 0.0

    rows = load_a2m(msa_path)
    query = rows[0]
    start = int(assay_meta["MSA_start"])
    target = assay_meta["target_seq"]

    # Focus columns in the STRIPPED query: one char per query column that is
    # not an insertion (lowercase/dot); each maps to target position
    # start + j (1-indexed) where j is its raw column index.
    jmap = [j for j, c in enumerate(query) if not (c.islower() or c == ".")]
    qstrip = strip_insertions(query)
    if len(qstrip) != len(jmap):
        raise ValueError("strip/column mismatch for %s" % msa_name)

    focus = [(k, start + jmap[k], c) for k, c in enumerate(qstrip)
             if c in _STD]
    positions = np.array([p for _, p, _ in focus], dtype=np.int64)
    n_pos = len(focus)

    # Mandatory mapping verification: the query residue at every focus column
    # must equal the target-sequence residue at the mapped position.
    n_bad = sum(1 for _k, p, c in focus
                if p - 1 >= len(target) or target[p - 1] != c)
    if n_bad:
        raise ValueError(
            "PSSM mapping broken for %s: %d/%d query residues disagree with "
            "the target sequence" % (assay_meta["DMS_id"], n_bad, n_pos))

    # Keep rows without non-standard residues; strip insertions from all rows.
    kept = []
    for r in rows:
        rs = strip_insertions(r)
        if len(rs) != len(qstrip):
            raise ValueError(
                "row length %d != query %d in %s — insertion columns are "
                "supposed to be consistent" % (len(rs), len(qstrip), msa_name))
        if not (set(rs) & NONSTANDARD):
            kept.append(rs)

    arr = np.frombuffer("".join(kept).encode("ascii"),
                        dtype=np.uint8).reshape(len(kept), len(qstrip))
    weights = henikoff_weights(arr)
    n_eff = float(weights.sum())

    # Weighted frequency per focus column + pseudocount 0.5 (Dirichlet-style).
    pseudo = 0.5
    focus_arr = arr[:, [k for k, _p, _c in focus]]  # (N_kept, n_pos)
    aa_ids = np.array([ord(a) for a in AA_ORDER], dtype=np.uint8)
    counts = np.zeros((n_pos, 20), dtype=np.float64)
    col_weight = (focus_arr != ord("-")).T.astype(np.float64) @ weights
    for a_i, a_id in enumerate(aa_ids):
        counts[:, a_i] = (focus_arr == a_id).T.astype(np.float64) @ weights
    freq = (counts + pseudo) / (col_weight[:, None] + 20 * pseudo)
    logf = np.log(freq)

    os.makedirs(cache_dir, exist_ok=True)
    np.savez(cache_path, positions=positions, logf=logf, neff=n_eff)
    if verbose:
        print("[pssm] %s: %d positions, neff=%.1f (%d rows kept of %d)" %
              (assay_meta["DMS_id"], n_pos, n_eff, len(kept), len(rows)))
    return positions, logf, n_eff


def row_scores(mutations, positions, logf):
    """Per-row PSSM log-odds and coverage for a list of mutation rows.

    mutations: list of (sites (0-indexed ints), muts (e.g. ['A2D']), target)
        in evaluate.py's per-row order.
    Returns (scores: np.ndarray (n_rows,), coverage: np.ndarray (n_rows,)).
    Coverage = fraction of the row's mutated sites that have an aligned column;
    uncovered sites contribute 0 to the row score.
    """
    pos2row = {int(p): i for i, p in enumerate(positions)}
    scores = np.zeros(len(mutations))
    cov = np.zeros(len(mutations))
    for r, (_sites, muts, _tar) in enumerate(mutations):
        acc = 0.0
        n_cov = 0
        for m in muts:
            p = int(m[1:-1])  # 'A123D' -> 123 (1-indexed target position)
            k = pos2row.get(p)
            if k is None:
                continue
            mt, wt = m[-1], m[0]
            if mt in AA2IDX and wt in AA2IDX:
                acc += logf[k, AA2IDX[mt]] - logf[k, AA2IDX[wt]]
                n_cov += 1
        if n_cov:
            scores[r] = acc
            cov[r] = n_cov / len(muts)
    return scores, cov
