"""Rule extraction and rule-based inference from trained TT-Sparse models."""

from __future__ import annotations

import operator as _op
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

from tt_sparse.encoder import TabularEncoder
from tt_sparse.model import TTSparseModel
from tt_sparse.qmc import (
    QuineMcCluskey,
    enumerate_truth_table,
    implicants_to_dnf,
    simplify_negations,
    _FLIP,
)


@dataclass
class RuleSet:
    """Extracted rules forming a linear rule model.

    Attributes:
        rules: List of dicts with keys 'r' (rule string), 'w' (weight dict),
               and 'kind' ('intercept', 'ltt', or 'skip').
        task_type: One of 'binary', 'multiclass', 'regression'.
        class_names: Target class labels.
        complexity: Total rule complexity (literals + connectives).
    """

    rules: list[dict]
    task_type: str
    class_names: list[str]
    complexity: int


def explain(
    model: TTSparseModel,
    encoder: TabularEncoder,
    training_data: pd.DataFrame | None = None,
    *,
    min_observations: int = 1,
    use_xor: bool = True,
) -> RuleSet:
    """Extract human-readable Boolean rules from a trained TT-Sparse model.

    Args:
        model: Trained (and optionally pruned) TTSparseModel.
        encoder: The TabularEncoder used to encode training data.
        training_data: Original DataFrame for don't-care computation (optional).
        min_observations: Minimum times a pattern must appear to not be a don't-care.
        use_xor: Enable XOR/XNOR simplification in Quine-McCluskey (default True).

    Returns:
        RuleSet containing weighted Boolean rules.
    """
    if not model.is_frozen:
        model.freeze_connections()

    manifest = encoder.get_feature_manifest()
    class_names = encoder.class_names
    is_bin = model.task_type == "binary"
    is_reg = model.task_type == "regression"

    model.eval()
    W, b = model.get_folded_classifier()
    n_nodes = model.num_nodes
    skip_size = model.skip_size

    node_W = W[:, :n_nodes]
    skip_W = W[:, n_nodes:n_nodes + skip_size] if skip_size > 0 else None

    indices_list, weights_list = model.get_active_connections()
    bias_vals = model.ltt_bias.detach().cpu().numpy()
    qm = QuineMcCluskey(use_xor=use_xor)

    X_ltt: np.ndarray | None = None
    if training_data is not None:
        enc_data = encoder.transform(training_data)
        X_ltt = enc_data["X_ltt"]

    node_rules: list[str] = []
    for ni in range(n_nodes):
        idx = indices_list[ni]
        w = weights_list[ni].tolist()
        bias = float(bias_vals[ni])
        k = len(idx)

        if k == 0:
            node_rules.append("True" if bias > 0 else "False")
            continue
        if k > 16:
            node_rules.append("False")
            continue

        minterms = enumerate_truth_table(k, w, bias)
        if not minterms:
            node_rules.append("False")
            continue
        if len(minterms) == 2**k:
            node_rules.append("True")
            continue

        dc: list[int] = []
        if X_ltt is not None and min_observations > 0:
            dc = _dont_cares(X_ltt, idx, k, min_observations)

        imps = qm.simplify(minterms, dc=dc, num_bits=k)
        if not imps:
            node_rules.append("False")
            continue

        expr = implicants_to_dnf(imps)
        for i, binary_idx in enumerate(idx):
            desc = manifest.get(binary_idx, f"x{i}")
            if " " in desc:
                desc = f"({desc})"
            expr = re.sub(rf"\bx{i}\b", desc, expr)
        expr = simplify_negations(expr)
        node_rules.append(expr)

    # Assemble rule list
    rules: list[dict] = []
    complexity = 0

    intercept_w = _weight_dict(b, class_names, is_bin, is_reg)
    rules.append({"r": "True", "w": intercept_w, "kind": "intercept"})
    complexity += 1

    if skip_W is not None:
        skip_names = encoder.get_skip_feature_names()
        for si, name in enumerate(skip_names):
            wd = _weight_dict(skip_W[:, si], class_names, is_bin, is_reg)
            if wd:
                rules.append({"r": name, "w": wd, "kind": "skip"})
                complexity += 1

    for ri, expr in enumerate(node_rules):
        if not expr or expr == "False":
            continue
        wd = _weight_dict(node_W[:, ri], class_names, is_bin, is_reg)
        if not wd:
            continue
        if expr == "True":
            for cls_name, v in wd.items():
                rules[0]["w"][cls_name] = rules[0]["w"].get(cls_name, 0.0) + v
            continue
        rules.append({"r": expr, "w": wd, "kind": "ltt"})
        complexity += 1 + expr.count("&") + expr.count("|") + expr.count("^")

    return RuleSet(rules=rules, task_type=model.task_type, class_names=class_names, complexity=complexity)


def predict_rules(
    rules: RuleSet,
    df: pd.DataFrame,
    encoder: TabularEncoder | None = None,
) -> np.ndarray:
    """Predict using extracted rules (no neural network needed).

    Args:
        rules: RuleSet from explain().
        df: DataFrame with the same feature columns as training data.
        encoder: Required for exact equivalence with the model.

    Returns:
        Probabilities (binary/multiclass) or scores (regression).
    """
    n = len(df)

    skip_values: dict[str, np.ndarray] = {}
    ltt_bits: dict[str, np.ndarray] = {}
    if encoder is not None:
        data = encoder.transform(df)
        X_ltt = data["X_ltt"]
        X_skip = data["X_skip"]
        manifest = encoder.get_feature_manifest()
        skip_names = encoder.get_skip_feature_names()
        skip_values = {name: X_skip[:, i] for i, name in enumerate(skip_names)}
        ltt_bits = {desc: X_ltt[:, idx].astype(bool) for idx, desc in manifest.items()}

    if rules.task_type == "multiclass":
        class_to_idx = {cn: i for i, cn in enumerate(rules.class_names)}
        scores = np.zeros((n, len(rules.class_names)))
    else:
        scores = np.zeros(n)

    for rule in rules.rules:
        weights = rule["w"]
        if not weights:
            continue

        kind = rule.get("kind", "ltt")
        expr = rule["r"].strip()

        if kind == "skip":
            if expr in skip_values:
                act = skip_values[expr]
            else:
                continue
        elif kind == "intercept" or expr == "True":
            act = np.ones(n)
        else:
            act = _eval_rule(expr, ltt_bits, df, n).astype(np.float64)

        if rules.task_type == "multiclass":
            for cn, w in weights.items():
                if cn in class_to_idx:
                    scores[:, class_to_idx[cn]] += act * w
        else:
            for _, w in weights.items():
                scores += act * w

    if rules.task_type == "binary":
        return 1.0 / (1.0 + np.exp(-scores))
    elif rules.task_type == "regression":
        return scores
    else:
        e = np.exp(scores - scores.max(1, keepdims=True))
        return e / e.sum(1, keepdims=True)


# =============================================================================
# Internals
# =============================================================================


def _dont_cares(X_ltt: np.ndarray, indices: list[int], k: int, min_obs: int) -> list[int]:
    sel = X_ltt[:, indices]
    powers = 2 ** np.arange(k - 1, -1, -1)
    ints = (sel @ powers).astype(int)
    if min_obs <= 1:
        return sorted(set(range(2**k)) - set(ints.tolist()))
    counts = np.bincount(ints, minlength=2**k)
    return [i for i in range(2**k) if counts[i] < min_obs]


def _weight_dict(
    w: np.ndarray, class_names: list[str], is_bin: bool, is_reg: bool
) -> dict[str, float]:
    d: dict[str, float] = {}
    if is_reg:
        if w[0] != 0:
            d["output"] = float(w[0])
    elif is_bin:
        if w[0] != 0:
            d[class_names[1]] = float(w[0])
    else:
        for i, cn in enumerate(class_names):
            if w[i] != 0:
                d[cn] = float(w[i])
    return d


def _strip_outer_parens(s: str) -> str:
    """Remove matched outer parentheses only when they wrap the entire expression."""
    s = s.strip()
    while s.startswith("(") and s.endswith(")"):
        depth = 0
        for i, ch in enumerate(s):
            if ch == '(':
                depth += 1
            elif ch == ')':
                depth -= 1
            if depth == 0:
                if i == len(s) - 1:
                    s = s[1:-1].strip()
                    break
                else:
                    return s
    return s


def _eval_rule(
    expr: str, ltt_bits: dict[str, np.ndarray], df: pd.DataFrame, n: int
) -> np.ndarray:
    """Evaluate a Boolean DNF expression (may contain XOR terms)."""
    if expr == "True":
        return np.ones(n, dtype=bool)
    if expr == "False":
        return np.zeros(n, dtype=bool)

    terms = _split_or(expr)
    result = np.zeros(n, dtype=bool)
    for term in terms:
        term = _strip_outer_parens(term)
        if " ^ " in term and "&" not in term:
            result |= _eval_xor_term(term, ltt_bits, df, n)
        else:
            lits = _split_and(term)
            mask = np.ones(n, dtype=bool)
            for lit in lits:
                lit = _strip_outer_parens(lit)
                if " ^ " in lit:
                    mask &= _eval_xor_term(lit, ltt_bits, df, n)
                else:
                    mask &= _resolve_literal(lit, ltt_bits, df)
            result |= mask
    return result


def _eval_xor_term(expr: str, ltt_bits: dict[str, np.ndarray], df: pd.DataFrame, n: int) -> np.ndarray:
    """Evaluate an XOR expression like '(a ^ b)' or '~(a ^ b)'."""
    negated = False
    inner = expr.strip()
    if inner.startswith("~(") and inner.endswith(")"):
        negated = True
        inner = inner[2:-1]
    elif inner.startswith("(") and inner.endswith(")"):
        inner = inner[1:-1]

    parts = [p.strip().strip("()") for p in inner.split(" ^ ")]
    xor_result = np.zeros(n, dtype=bool)
    for part in parts:
        xor_result ^= _resolve_literal(part, ltt_bits, df)

    return ~xor_result if negated else xor_result


def _split_at_operator(expr: str, op: str) -> list[str]:
    """Split top-level terms at a given operator (e.g. ' | ', ' & ') respecting parentheses."""
    depth = 0
    parts: list[str] = []
    current: list[str] = []
    op_len = len(op)
    i = 0
    while i < len(expr):
        ch = expr[i]
        if ch == '(':
            depth += 1
            current.append(ch)
        elif ch == ')':
            depth -= 1
            current.append(ch)
        elif depth == 0 and expr[i:i + op_len] == op:
            parts.append("".join(current))
            current = []
            i += op_len
            continue
        else:
            current.append(ch)
        i += 1
    if current:
        parts.append("".join(current))
    return parts


def _split_or(expr: str) -> list[str]:
    """Split top-level OR terms respecting parentheses."""
    return _split_at_operator(expr, " | ")


def _split_and(expr: str) -> list[str]:
    """Split top-level AND terms respecting parentheses."""
    return _split_at_operator(expr, " & ")


def _resolve_literal(
    lit: str, ltt_bits: dict[str, np.ndarray], df: pd.DataFrame
) -> np.ndarray:
    """Resolve a single literal to a boolean array."""
    lit = lit.strip()
    if lit in ltt_bits:
        return ltt_bits[lit]

    negated = _get_negated_form(lit)
    if negated and negated in ltt_bits:
        return ~ltt_bits[negated]

    if ltt_bits:
        for key in ltt_bits:
            if _get_negated_form(key) == lit:
                return ~ltt_bits[key]

    return _eval_comparison(lit, df)


def _get_negated_form(lit: str) -> str | None:
    for op, flipped in _FLIP.items():
        if f" {op} " in lit:
            parts = lit.split(f" {op} ", 1)
            return f"{parts[0]} {flipped} {parts[1]}"
    return None


_COMPARISON_OPS = {
    ">=": _op.ge,
    "<=": _op.le,
    "!=": _op.ne,
    "==": _op.eq,
    ">": _op.gt,
    "<": _op.lt,
}


def _eval_comparison(lit: str, df: pd.DataFrame) -> np.ndarray:
    """Evaluate a comparison literal against a DataFrame."""
    n = len(df)
    for op_str, op_fn in _COMPARISON_OPS.items():
        if f" {op_str} " in lit:
            feat, val = lit.split(f" {op_str} ", 1)
            feat, val = feat.strip(), val.strip().strip("'\"")
            if feat not in df.columns:
                return np.zeros(n, dtype=bool)
            col = df[feat]
            try:
                v: float | str = float(val)
                col = pd.to_numeric(col, errors="coerce")
            except (ValueError, TypeError):
                v = val
            return op_fn(col, v).values
    return np.ones(n, dtype=bool)
