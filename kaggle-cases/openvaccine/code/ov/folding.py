import os
import re
import subprocess
import tempfile
from multiprocessing import Pool

import numpy as np

from .paths import BPPS, ETERNAFOLD

CONTRAFOLD = os.path.join(ETERNAFOLD, "src", "contrafold")
EF_PARAMS = os.path.join(ETERNAFOLD, "parameters", "EternaFoldParams.v1")


def _run_contrafold(seq, extra):
    with tempfile.TemporaryDirectory() as d:
        fa = os.path.join(d, "s.fa")
        with open(fa, "w") as f:
            f.write(">s\n%s\n" % seq)
        out = os.path.join(d, "out.txt")
        cmd = [CONTRAFOLD, "predict", fa, "--params", EF_PARAMS] + extra
        if "--posteriors" in extra:
            cmd.append(out)
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            with open(out) as f:
                return f.read()
        res = subprocess.run(cmd, check=True, capture_output=True, text=True)
        return res.stdout


def eternafold_bpps(seq):
    txt = _run_contrafold(seq, ["--posteriors", "0.00001"])
    n = len(seq)
    m = np.zeros((n, n), dtype=np.float32)
    for line in txt.strip().split("\n"):
        parts = line.split()
        i = int(parts[0]) - 1
        for p in parts[2:]:
            j, v = p.split(":")
            m[i, int(j) - 1] = float(v)
    return np.maximum(m, m.T)


def eternafold_mfe(seq):
    txt = _run_contrafold(seq, ["--viterbi"])
    return txt.strip().split("\n")[-1].strip()


def bpps_path(key):
    return os.path.join(BPPS, "%s.npy" % key)


def _compute_one(args):
    key, seq = args
    path = bpps_path(key)
    if not os.path.exists(path):
        np.save(path, eternafold_bpps(seq))
    return key


def compute_bpps(pairs, processes=100):
    """pairs: iterable of (key, sequence); caches to data/bpps/<key>.npy"""
    os.makedirs(BPPS, exist_ok=True)
    todo = [(k, s) for k, s in pairs if not os.path.exists(bpps_path(k))]
    if todo:
        with Pool(processes) as p:
            for _ in p.imap_unordered(_compute_one, todo, chunksize=8):
                pass
    return len(todo)


def load_bpps(key):
    return np.load(bpps_path(key))


def _bps(secstruct):
    bps = []
    for l, r in zip("([{", ")]}"):
        stack = []
        for i, c in enumerate(secstruct):
            if c == l:
                stack.append(i)
            elif c == r:
                bps.append([stack.pop(), i])
    return bps


def bprna_string(dbn):
    """Approximate bpRNA loop-type string (S/H/B/I/M/E/X) from dot-bracket.

    Adapted from KaggleOpenVaccine/scripts/nullrecurrent_inference.py (H. Wayment-Steele)."""
    partner = -np.ones(len(dbn), dtype=int)
    for i, j in _bps(dbn):
        partner[i], partner[j] = j, i
    s = ["u"] * len(dbn)
    for k, c in enumerate(dbn):
        if c != ".":
            s[k] = "S"
    while "u" in s:
        obj = re.search(r"uu*", "".join(s))
        a, b = obj.start(), obj.end()
        n_open = dbn[:a].count(")") - dbn[:a].count("(")
        if n_open == 0:
            s[a:b] = ("E" if a == 0 or b == len(dbn) else "X") * (b - a)
        else:
            last = int(partner[a - 1])
            nxt = int(partner[b])
            if last == b:
                s[a:b] = "H" * (b - a)
            elif last - 1 == nxt:
                s[a:b] = "B" * (b - a)
            elif dbn[a - 1] == ")" and dbn[b] == "(":
                s[a:b] = "M" * (b - a)
            elif dbn[nxt + 1:last] == "." * (last - nxt - 1):
                s[a:b] = "I" * (b - a)
                s[nxt + 1:last] = "I" * (last - nxt - 1)
            else:
                s[a:b] = "M" * (b - a)
    return "".join(s)
