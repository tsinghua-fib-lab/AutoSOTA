#!/usr/bin/env python3
"""Verify SubsetShrakeRupley against Bio.PDB.SASA.ShrakeRupley on real AF2 models."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from Bio.PDB import PDBParser
from Bio.PDB.SASA import ShrakeRupley

sys.path.insert(0, "/repo/sota")
from sasa_fast import SubsetShrakeRupley, heavy_atoms_and_radii  # noqa: E402

KV = Path("/paper_data/kvigly_test_mutation_pdbs")
SUF = "_unrelaxed_rank_1_model_3.pdb"

files = ["WT" + SUF, "A101E" + SUF, "Q157F" + SUF, "K65Q" + SUF, "S182F" + SUF, "P42M" + SUF]
parser = PDBParser(QUIET=1)
fast = SubsetShrakeRupley()
worst = 0.0
for fn in files:
    st = parser.get_structure("s", str(KV / fn))
    chain = list(st[0].get_chains())[0]
    res = [r for r in chain if r.id[0] == " "]

    sr = ShrakeRupley()
    sr.compute(st, level="R")
    ref = {r.id[1]: r.sasa for r in res}

    atoms, coords, elements = heavy_atoms_and_radii(st)
    idx_of = {id(a): k for k, a in enumerate(atoms)}
    for target in (1, 50, 101, 150, 200, 221):
        rr = [r for r in res if r.id[1] == target]
        if not rr:
            continue
        subset = [idx_of[id(a)] for a in rr[0].get_atoms()]
        got = sum(fast.atom_sasa(coords, elements, subset).values())
        d = abs(got - ref[target])
        worst = max(worst, d)
        print(f"{fn:12s} res {target:3d}  ref={ref[target]:9.4f} fast={got:9.4f}  |d|={d:.3e}")
print(f"\nworst |difference| over {len(files)} structures x 6 residues: {worst:.3e}")
print("EXACT" if worst < 1e-9 else "NOT EXACT")
