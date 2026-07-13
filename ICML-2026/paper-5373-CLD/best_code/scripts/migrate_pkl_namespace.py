#!/usr/bin/env python
"""Rebuild legacy CLD model pickles as fresh ``jaxcld`` heads.

The trained heads were pickled when this package was named ``cld`` (and, for the
oldest ``lr_exp`` artifacts, ``solve``). A pickle stores each object's class by its
*module path*, so after the ``cld -> jaxcld`` rename ``CVXNNLangDetectHead.load``
(i.e. ``pickle.load``) dies with ``ModuleNotFoundError: No module named 'cld'``.

Rather than re-serialize the old object graph, this script reads the old pickle,
**constructs a brand-new** ``jaxcld.models.cvx_relu_mlp.CVX_ReLU_MLP`` and copies only
the weights/config that inference needs (``stacked_predict`` is just
``relu(X @ W1) @ W2``):

    theta1, theta2            trained per-class weights  (required)
    n_classes, P_S, beta, rho, seed   config scalars

The bulky training-only state embedded in the old pickle (the full ``X``/``Xtst``
matrices, ADMM duals ``u``/``v``/``s``, ``d_diags``/``e_diags``) is dropped, so the new
artifact is far smaller and carries no training data.

To *read* the old pickle we temporarily alias the legacy import names (``cld``,
``solve`` and their submodules) onto the live ``jaxcld`` package in ``sys.modules``,
then restore ``sys.modules`` in a ``finally``. (A literal on-disk copy of ``jaxcld``
to ``cld``/``solve`` also works, but the repo already has a real ``solve/`` dir, so
the in-process alias avoids clobbering it and leaves nothing to clean up on disk.)

Usage:
    python scripts/migrate_pkl_namespace.py            # rewrite every legacy pkl in data/ (.bak kept)
    python scripts/migrate_pkl_namespace.py --dry-run  # report only, write nothing
    python scripts/migrate_pkl_namespace.py f1.pkl ... # only the given files

Run under the env that has jax/torch (e.g. `conda run -n cld`). After migrating,
re-upload with scripts/push_models_to_hf.py / push_lr_models_to_hf.py.
"""
import argparse
import glob
import importlib
import os
import pickle
import pickletools
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)  # so `import jaxcld` resolves when run as scripts/...
LEGACY_PREFIXES = ("cld", "solve")  # old top-level package names; all aliased to jaxcld

# trained weights inference requires (stacked_predict = relu(X @ theta1) @ theta2)
REQUIRED_ATTRS = ("theta1", "theta2")


class IncompatibleSchema(Exception):
    """Raised for legacy artifacts whose weight layout the current head can't use."""
# config scalars carried over when present; everything else in the old pickle is
# training-only state (full X/Xtst, ADMM duals u/v/s, d_diags/e_diags) and is dropped.
OPTIONAL_ATTRS = ("n_classes", "P_S", "beta", "rho", "seed")


def install_legacy_aliases(files):
    """Alias the legacy module paths used by `files` onto live jaxcld modules.

    Returns a restore callback that undoes every sys.modules change.
    """
    needed = set()
    for p in files:
        needed.update(legacy_modules(p))

    saved = {}      # name -> previous sys.modules entry (KeyError sentinel = absent)
    SENTINEL = object()
    for mod in needed:
        parts = mod.split(".")
        for i in range(1, len(parts) + 1):
            name = ".".join(parts[:i])
            if name in saved:
                continue
            jx = "jaxcld" + ("" if i == 1 else "." + ".".join(parts[1:i]))
            saved[name] = sys.modules.get(name, SENTINEL)
            sys.modules[name] = importlib.import_module(jx)

    def restore():
        for name, prev in saved.items():
            if prev is SENTINEL:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = prev

    return restore


def legacy_modules(path):
    with open(path, "rb") as f:
        data = f.read()
    return sorted({a for _op, a, _pos in pickletools.genops(data)
                   if isinstance(a, str) and a.split(".", 1)[0] in LEGACY_PREFIXES})


def rebuild(old):
    """Make a fresh jaxcld CVX_ReLU_MLP carrying only the inference weights/config."""
    from jaxcld.models.cvx_relu_mlp import CVX_ReLU_MLP

    missing = [a for a in REQUIRED_ATTRS if getattr(old, a, None) is None]
    if missing:
        raise RuntimeError(f"old object missing trained weights {missing}; cannot rebuild")
    # current head expects stacked per-class weights: theta1 (C, d, m), theta2 (C, m).
    # Older binary artifacts stored 2-D theta1; those don't fit stacked_predict.
    if getattr(old.theta1, "ndim", None) != 3:
        raise IncompatibleSchema(
            f"theta1 has shape {getattr(old.theta1, 'shape', None)} (expected 3-D stacked weights)")

    new = CVX_ReLU_MLP.__new__(CVX_ReLU_MLP)  # skip __init__ so we needn't pass X/y
    for attr in REQUIRED_ATTRS:
        setattr(new, attr, getattr(old, attr))
    for attr in OPTIONAL_ATTRS:
        if hasattr(old, attr):
            setattr(new, attr, getattr(old, attr))
    # null out the training-only fields the class normally holds
    for attr in ("X", "y", "Xtst", "ytst", "d_diags", "e_diags"):
        setattr(new, attr, None)
    return new


def verify(old, new):
    """The fresh head must reproduce the original logits on arbitrary input."""
    import numpy as np
    import jax.numpy as jnp

    d = int(old.theta1.shape[1])
    x = jnp.asarray(np.random.default_rng(0).standard_normal((8, d)).astype(np.float32))
    lo = np.asarray(old.stacked_predict(x, old.theta1, old.theta2))
    ln = np.asarray(new.stacked_predict(x, new.theta1, new.theta2))
    if not np.allclose(lo, ln, atol=1e-5):
        raise RuntimeError("rebuilt head logits differ from original")


def migrate(path, dry_run):
    mods = legacy_modules(path)
    if not mods:
        print(f"[skip]  {path}  (already clean)")
        return False
    print(f"[found] {path}  -> {', '.join(mods)}")
    if dry_run:
        return True

    with open(path, "rb") as f:
        old = pickle.load(f)
    try:
        new = rebuild(old)
    except IncompatibleSchema as e:
        print(f"[skip]  {path}  (incompatible legacy schema: {e}; not an HF head, left as-is)")
        return False
    verify(old, new)

    tmp = path + ".migrated"
    with open(tmp, "wb") as f:
        pickle.dump(new, f, protocol=pickle.HIGHEST_PROTOCOL)
    remaining = legacy_modules(tmp)
    if remaining:
        os.remove(tmp)
        raise RuntimeError(f"rebuilt pickle still references {remaining}; aborting {path}")

    shutil.copy2(path, path + ".bak")
    os.replace(tmp, path)
    old_mb = os.path.getsize(path + ".bak") / 1e6
    new_mb = os.path.getsize(path) / 1e6
    print(f"[done]  {path}  ({old_mb:.1f}MB -> {new_mb:.1f}MB; original saved as {path}.bak)")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*", help="pkl files (default: data/**/*.pkl)")
    ap.add_argument("--dry-run", action="store_true", help="report only, write nothing")
    args = ap.parse_args()

    files = args.files or sorted(
        p for p in glob.glob(os.path.join(ROOT, "data", "**", "*.pkl"), recursive=True)
        if not os.path.basename(p).startswith("._")
    )

    restore = (lambda: None)
    try:
        if not args.dry_run:
            restore = install_legacy_aliases(files)
        changed = sum(migrate(p, args.dry_run) for p in files)
    finally:
        restore()

    verb = "would migrate" if args.dry_run else "migrated"
    print(f"\n{verb} {changed} file(s); {len(files) - changed} skipped (already clean or incompatible).")


if __name__ == "__main__":
    main()
