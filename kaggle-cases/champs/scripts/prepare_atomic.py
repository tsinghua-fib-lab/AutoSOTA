#!/usr/bin/env python3
"""Prepare the original symmetry-extended, source-centered atomic inputs.

Reads only the training targets, test atom pairs, and molecular coordinates.
No test targets or supplementary quantum properties are read.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from numpy.lib.format import open_memmap
from sklearn.model_selection import train_test_split

from atomic_model import TYPES

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/atomic"


def prepare(split, molecule_names, dense_xyz, dense_atom, raw_dir=None):
    started = time.time()
    frame = pd.read_csv((raw_dir or ROOT / "data/raw") / (split + ".csv"))
    molecule = pd.Index(molecule_names).get_indexer(frame.molecule_name)
    if (molecule < 0).any():
        raise ValueError("Missing molecule geometry")
    a0 = frame.atom_index_0.to_numpy(dtype=np.int64)
    a1 = frame.atom_index_1.to_numpy(dtype=np.int64)
    types = pd.Index(TYPES).get_indexer(frame.type).astype(np.int8)
    if (types < 0).any():
        raise ValueError("Unknown coupling type")
    # Sorting unique keys matches groupby(molecule_name, atom_index_0)
    # after adding symmetry-reversed pairs in the original notebook.
    keys, inverse = np.unique(np.r_[molecule * 32 + a0, molecule * 32 + a1], return_inverse=True)
    source_molecule, source_atom = keys // 32, keys % 32
    n = len(keys)
    print(json.dumps({"split": split, "source_groups": n, "pairs": len(frame)}), flush=True)
    prefix = OUT / split
    xyz = open_memmap(str(prefix) + "_xyz.npy", mode="w+", dtype=np.float32, shape=(n, 3, 29))
    atoms = open_memmap(str(prefix) + "_atom.npy", mode="w+", dtype=np.int8, shape=(n, 29))
    coupling = open_memmap(str(prefix) + "_type.npy", mode="w+", dtype=np.int8, shape=(n, 29))
    coupling[:] = -1
    for start in range(0, n, 10000):
        stop = min(start + 10000, n)
        m, a = source_molecule[start:stop], source_atom[start:stop]
        valid = dense_atom[m] >= 0
        positions = dense_xyz[m] - dense_xyz[m, a][:, None, :]
        positions[~valid] = 0
        xyz[start:stop] = positions.transpose(0, 2, 1)
        atoms[start:stop] = dense_atom[m]
    original_group = inverse[:len(frame)]
    reverse_group = inverse[len(frame):]
    coupling[original_group, a1] = types
    coupling[reverse_group, a0] = types
    np.save(str(prefix) + "_molecule.npy", source_molecule.astype(np.int32))
    np.save(str(prefix) + "_pair_group.npy", original_group.astype(np.int32))
    np.save(str(prefix) + "_pair_atom.npy", a1.astype(np.int8))
    np.save(str(prefix) + "_pair_id.npy", frame.id.to_numpy(dtype=np.int64))
    np.save(str(prefix) + "_pair_type.npy", types)
    if split == "train":
        target = open_memmap(str(prefix) + "_target.npy", mode="w+", dtype=np.float32, shape=(n, 29))
        target[:] = 0
        y = frame.scalar_coupling_constant.to_numpy(dtype=np.float32)
        target[original_group, a1] = y
        target[reverse_group, a0] = y
        target.flush()
        # Strict molecule holdout for future training. Published checkpoints
        # trained on all train molecules must not be evaluated on this as OOF.
        unique_molecules = np.unique(source_molecule)
        _, valid_molecules = train_test_split(unique_molecules, test_size=0.1, random_state=13)
        np.save(str(prefix) + "_valid_mask.npy", np.isin(source_molecule, valid_molecules))
        # Match the original NumPy float32 normalization expression exactly.
        mean = xyz.mean(axis=(0, 2), keepdims=True)
        std = xyz.std(axis=(0, 2), keepdims=True)
        np.savez(OUT / "normalization.npz", mean=mean, std=std)
        print(json.dumps({"normalization_mean": mean.tolist(), "normalization_std": std.tolist()}), flush=True)
    xyz.flush(); atoms.flush(); coupling.flush()
    print(json.dumps({"split": split, "seconds": round(time.time() - started, 2), "complete": True}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data/raw")
    parser.add_argument("--out-dir", type=Path, default=OUT)
    parser.add_argument("--split", choices=("train", "test", "both"), default="both")
    args = parser.parse_args()
    OUT = args.out_dir.resolve()
    OUT.mkdir(parents=True, exist_ok=True)
    structures = pd.read_csv(args.raw_dir / "structures.csv")
    molecule_names = np.sort(structures.molecule_name.unique())
    np.save(OUT / "molecule_names.npy", molecule_names.astype(str))
    index = pd.Index(molecule_names).get_indexer(structures.molecule_name)
    atom_index = structures.atom_index.to_numpy()
    dense_xyz = np.zeros((len(molecule_names), 29, 3), dtype=np.float64)
    dense_atom = np.full((len(molecule_names), 29), -1, dtype=np.int8)
    dense_xyz[index, atom_index] = structures[["x", "y", "z"]].to_numpy()
    dense_atom[index, atom_index] = pd.Index(list("CFHNO")).get_indexer(structures.atom)
    if (dense_atom[index, atom_index] < 0).any():
        raise ValueError("Unknown element")
    del structures
    for split in (("train", "test") if args.split == "both" else (args.split,)):
        prepare(split, molecule_names, dense_xyz, dense_atom, args.raw_dir)
