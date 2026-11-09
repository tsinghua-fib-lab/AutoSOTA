#!/usr/bin/env python3
"""Compare vectorized inputs to the runner-up's original notebook function."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from atomic_model import TYPES

ROOT = Path(__file__).resolve().parents[1]


def main():
    test = pd.read_csv(ROOT / "data/raw/test.csv")
    structures = pd.read_csv(ROOT / "data/raw/structures.csv")
    names = np.sort(test.molecule_name.unique())
    selected_names = names[np.linspace(0, len(names) - 1, 32, dtype=int)]
    original = test[test.molecule_name.isin(selected_names)].copy()
    original["ext"] = False
    reverse = original.rename(columns={"atom_index_0": "atom_index_1", "atom_index_1": "atom_index_0"}).copy()
    reverse["ext"] = True
    extended = pd.concat([original, reverse], ignore_index=True)
    # Supply only the two expected dataframes. Unexpected reads fail closed.
    def read_csv(filename):
        if str(filename) == "sample.csv":
            return extended.copy()
        if str(filename) == "reference/../data/structures.csv":
            return structures.copy()
        raise ValueError("Unexpected reference data access: " + str(filename))
    namespace = {"np": np, "pd": SimpleNamespace(read_csv=read_csv, factorize=pd.factorize),
                 "atoms": "CFHNO", "SCRIPT_DIR": "reference", "tqdm": lambda x: x}
    notebook = ROOT / "repos/champs_reference/solutions/2/transformer-experiments.ipynb"
    source = "".join(json.loads(notebook.read_text())["cells"][4]["source"])
    exec(compile(source, str(notebook) + "#cell4", "exec"), namespace)
    xyz, coupling, extension, atom, _, pair_ids = namespace["preprocess"]("sample.csv", type_index=TYPES)
    global_names = np.load(ROOT / "data/atomic/molecule_names.npy")
    mi = pd.Index(global_names).get_indexer(test.molecule_name)
    keys = np.unique(np.r_[mi * 32 + test.atom_index_0.values, mi * 32 + test.atom_index_1.values])
    groups = extended.groupby(["molecule_name", "atom_index_0"], sort=True).size().index
    sample_keys = pd.Index(global_names).get_indexer(groups.get_level_values(0)) * 32 + groups.get_level_values(1).values
    selected = np.searchsorted(keys, sample_keys)
    vectorized = {name: np.load(ROOT / ("data/atomic/test_" + name + ".npy"), mmap_mode="r")[selected]
                  for name in ("xyz", "type", "atom")}
    checks = {"sample_molecules": len(selected_names), "sample_source_groups": len(selected),
              "xyz_exact": bool(np.array_equal(xyz, vectorized["xyz"])),
              "type_exact": bool(np.array_equal(coupling[:, 0], vectorized["type"])),
              "atom_exact": bool(np.array_equal(atom[:, 0], vectorized["atom"])),
              "xyz_max_absolute_difference": float(np.max(np.abs(xyz - vectorized["xyz"]))),
              "reference": str(notebook.relative_to(ROOT)) + "#cell4", "test_labels_used": False}
    original_ids = pair_ids[~extension[:, 0]]
    checks["original_pair_ids_exact"] = bool(np.array_equal(np.sort(original_ids), np.sort(original.id.values)))
    if not all(checks[key] for key in ("xyz_exact", "type_exact", "atom_exact", "original_pair_ids_exact")):
        raise AssertionError(checks)
    (ROOT / "artifacts/atomic_preprocessing_verification.json").write_text(json.dumps(checks, indent=2))
    print(json.dumps(checks, indent=2), flush=True)


if __name__ == "__main__":
    main()
