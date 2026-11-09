#!/usr/bin/env python3
"""Exact single-residue Shrake-Rupley SASA.

`Bio.PDB.SASA.ShrakeRupley.compute` walks every atom of the structure, which is
88% of the runtime of the NESP structural feature extraction, while the pipeline
only ever needs the SASA of the *mutated* residue. The occluded-point count of
atom i depends only on atom i's sphere and the coordinates of its neighbours, so
restricting the loop to the atoms of one residue reproduces Biopython's values
exactly (the residue SASA is just the sum over its atoms).

`sasa_fast.py::residue_sasa` is verified against `ShrakeRupley` on real AF2 models
by `sota/verify_sasa_fast.py`.
"""
from __future__ import annotations

import numpy as np
from Bio.PDB.SASA import ShrakeRupley
from Bio.PDB.kdtrees import KDTree


class SubsetShrakeRupley:
    def __init__(self, n_points: int = 100, probe_radius: float = 1.4, radii_dict=None):
        ref = ShrakeRupley(n_points=n_points, probe_radius=probe_radius,
                           radii_dict=radii_dict)
        # keep the reference sphere exactly as Biopython stores it (float32)
        self.sphere = ref._sphere
        self.n_points = n_points
        self.probe_radius = probe_radius
        self.radii_dict = ref.radii_dict

    def atom_sasa(self, coords, elements, subset):
        """coords (N,3) float64; elements list[str]; subset = atom indices to compute.

        Returns {atom_index: asa}."""
        if not len(coords):
            return {}
        kdt = KDTree(coords, 10)
        radii = np.array([self.radii_dict[e] for e in elements], dtype=np.float64)
        radii += self.probe_radius
        twice_maxradii = float(np.max(radii)) * 2
        ptset = set(range(self.n_points))
        f = radii * radii * (4 * np.pi / self.n_points)
        out = {}
        for i in subset:
            r_i = radii[i]
            s_on_i = (np.array(self.sphere, copy=True) * r_i) + coords[i]
            available_set = ptset.copy()
            kdt_sphere = KDTree(s_on_i, 10)
            for jj in kdt.search(coords[i], twice_maxradii):
                j = jj.index
                if i == j:
                    continue
                if jj.radius < (r_i + radii[j]):
                    available_set -= {pt.index for pt in kdt_sphere.search(coords[j], radii[j])}
            out[i] = len(available_set) * f[i]
        return out


def heavy_atoms_and_radii(structure):
    """Return (coords, elements) for every atom of chain 0, in Biopython's order."""
    atoms = list(list(structure[0].get_chains())[0].get_atoms())
    coords = np.array([a.coord for a in atoms], dtype=np.float64)
    elements = [a.element for a in atoms]
    return atoms, coords, elements
