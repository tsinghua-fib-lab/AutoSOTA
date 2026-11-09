#!/usr/bin/env python3
"""Extract label-free structural features for the 2413 NESP test rows from the
AlphaFold2 mutant models in /paper_data/kvigly_test_mutation_pdbs/.

No test labels are used anywhere here: every feature is a function of the input
structures and the mutation identity only.

Outputs a CSV with one row per test seq_id.
"""
from __future__ import annotations

import argparse
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from Bio.PDB import PDBParser
from Bio.PDB.SASA import ShrakeRupley
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sasa_fast import SubsetShrakeRupley  # noqa: E402

KV = "/paper_data/kvigly_test_mutation_pdbs"
WT_PDB = os.path.join(KV, "WT_unrelaxed_rank_1_model_3.pdb")
SUFFIX = "_unrelaxed_rank_1_model_3.pdb"

WT_SEQUENCE = (
    "VPVNPEPDATSVENVALKTGSGDSQSDPIKADLEVKGQSALPFDVDCWAILCKGAPNVLQRVNEKTKNSNRDRSGANK"
    "GPFKDPQKWGIKALPPKNPSWSAQDFKSPEEYAFASSLQGGTNAILAPVNLASQNSQGGVLNGFYSANKVAQFDPSK"
    "PQQTKGTWFQITKFTGAAGPYCKALGSNDKSVCDKNKNIAGDWGFDPAKWAYQYDEKNNKFNYVGK"
)

BACKBONE = {"N", "CA", "C", "O", "OXT"}

HYDROPHOBIC = set("AVLIMFW")
NAME3_TO_1 = {"VAL": "V", "PRO": "P", "ASN": "N", "GLU": "E", "ASP": "D", "ALA": "A",
              "THR": "T", "SER": "S", "LEU": "L", "LYS": "K", "GLY": "G", "GLN": "Q",
              "ILE": "I", "PHE": "F", "CYS": "C", "TRP": "W", "ARG": "R", "TYR": "Y",
              "HIS": "H", "MET": "M"}

_G = {}


# --------------------------------------------------------------- PDB helpers
def ca_and_bf(structure):
    chain = list(structure[0].get_chains())[0]
    coords, bf, resseq = [], [], []
    for res in chain:
        if res.id[0] != " " or "CA" not in res:
            continue
        coords.append(res["CA"].coord)
        bf.append(res["CA"].bfactor)
        resseq.append(res.id[1])
    return np.asarray(coords, float), np.asarray(bf, float), np.asarray(resseq, int)


def structure_arrays(structure):
    """(coords, heavy, res_number, sidechain, is_ca) for every atom of chain 0,
    in `chain.get_atoms()` order."""
    atoms = list(list(structure[0].get_chains())[0].get_atoms())
    coords = np.array([a.coord for a in atoms], dtype=np.float64)
    elements = [a.element for a in atoms]
    res_idx = np.array([a.get_parent().id[1] for a in atoms], dtype=int)
    side = np.array([a.get_name().strip() not in BACKBONE for a in atoms])
    is_ca = np.array([a.get_name().strip() == "CA" for a in atoms])
    heavy = np.array([e != "H" for e in elements])
    return coords, heavy, res_idx, side, is_ca


def kabsch(A, B):
    Ac = A - A.mean(0)
    Bc = B - B.mean(0)
    U, _, Vt = np.linalg.svd(Ac.T @ Bc)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return R, B.mean(0) - (A.mean(0) @ R.T)


def all_atom_contacts(chain, resseq_id, cutoff=8.0):
    """Reference contact count (Biopython chain walk) backing the original
    `contacts_mut` / `contacts_wt` features."""
    target = [r for r in chain if r.id[0] == " " and r.id[1] == resseq_id]
    if not target or "CA" not in target[0]:
        return 0
    tres = target[0]
    txyz = np.array([a.coord for a in tres if a.element != "H"], float)
    if txyz.size == 0:
        return 0
    tca = tres["CA"].coord
    n = 0
    for res in chain:
        if res.id[0] != " " or res.id[1] == resseq_id or "CA" not in res:
            continue
        if np.linalg.norm(res["CA"].coord - tca) > (cutoff + 8.0):
            continue
        xyz = np.array([a.coord for a in res if a.element != "H"], float)
        if xyz.size == 0:
            continue
        d = np.linalg.norm(txyz[:, None, :] - xyz[None, :, :], axis=-1)
        n += int((d < cutoff).sum())
    return n


def contacts_of_residue(coords, heavy, res_idx, side, res_number, cutoff,
                        sidechain_only=False):
    """Atoms of *other* residues within `cutoff` of any heavy atom of residue
    `res_number`. With sidechain_only both sides are restricted to side-chain
    heavy atoms."""
    own = res_idx == res_number
    if not own.any():
        return 0
    keep = heavy.copy()
    if sidechain_only:
        keep &= side
    target = own & keep
    partner = keep & (~own)
    if not target.any() or not partner.any():
        return 0
    tree = cKDTree(coords[partner])
    return int(sum(len(c) for c in tree.query_ball_point(coords[target], cutoff)))


def radius_of_gyration(coords):
    if len(coords) == 0:
        return float("nan")
    c = coords.mean(0)
    return float(np.sqrt(((coords - c) ** 2).sum(1).mean()))


# ------------------------------------------------------------------ workers
def init_worker(wt_pdb):
    parser = PDBParser(QUIET=1)
    wt = parser.get_structure("wt", wt_pdb)
    ca_wt, bf_wt, resseq_wt = ca_and_bf(wt)
    sr = ShrakeRupley()
    sr.compute(wt, level="R")
    chain = list(wt[0].get_chains())[0]
    sasa_by_res = {r.id[1]: r.sasa for r in chain if r.id[0] == " "}
    resnames = {r.id[1]: r.resname for r in chain if r.id[0] == " "}
    _G.update(parser=parser, ca_wt=ca_wt, bf_wt=bf_wt, resseq_wt=resseq_wt,
              chain=chain, sasa_by_res=sasa_by_res, resnames=resnames,
              fastsr=SubsetShrakeRupley(), wt_arrays=structure_arrays(wt))


def one_row(rec):
    parser = _G["parser"]
    ca_wt, bf_wt, resseq_wt, chain = _G["ca_wt"], _G["bf_wt"], _G["resseq_wt"], _G["chain"]
    sasa_by_res, resnames = _G["sasa_by_res"], _G["resnames"]
    wcoords, wheavy, wres, wside, wca = _G["wt_arrays"]

    mut_seq = rec["protein_sequence"]
    pos = 0
    for pos, (a, b) in enumerate(zip(mut_seq, WT_SEQUENCE)):
        if a != b:
            break
    aa1 = WT_SEQUENCE[pos]
    aa2 = mut_seq[pos] if len(mut_seq) == len(WT_SEQUENCE) else "_"
    pos1 = pos + 1

    feat = {"seq_id": rec["seq_id"], "position": pos, "AA1": aa1, "AA2": aa2,
            "struct_res_at_site": resnames.get(pos1, "UNK")}
    feat["sasa_wt_site"] = float(sasa_by_res.get(pos1, np.nan))
    feat["burial"] = feat["sasa_wt_site"] / 183.27

    # ---- WT-side packing features (no mutant model needed) ------------------
    n_shell = 0
    try:
        feat["cm_wt_8"] = contacts_of_residue(wcoords, wheavy, wres, wside, pos1, 8.0, True)
        for cutoff in (6.0, 10.0, 12.0):
            feat[f"cm_wt_{int(cutoff)}"] = contacts_of_residue(
                wcoords, wheavy, wres, wside, pos1, cutoff, True)
        feat["rgyr_wt"] = radius_of_gyration(wcoords[wheavy])
        site_ca_w = wcoords[(wres == pos1) & wca][0]
        n_shell = int(np.unique(wres[np.linalg.norm(wcoords - site_ca_w, axis=1) < 10.0]).size)
        feat["shell_res_w10"] = n_shell

        # --- environment descriptors around the mutation site (WT model) -----
        site_atoms = wcoords[(wres == pos1) & wheavy]
        ca_all = wcoords[wca]
        ca_res = wres[wca]
        d_ca = np.linalg.norm(ca_all - site_ca_w, axis=1)
        for cutoff in (8.0, 12.0, 14.0):
            feat[f"shell_res_w{int(cutoff)}"] = int((d_ca < cutoff).sum()) - 1
            feat[f"ca_w{int(cutoff)}"] = int((d_ca < cutoff).sum()) - 1

        # minimum heavy-atom distance from the site residue to every other residue
        dist_by_res = {}
        for rr in np.unique(wres):
            if rr == pos1:
                continue
            other = wcoords[(wres == rr) & wheavy]
            if not len(other) or not len(site_atoms):
                continue
            dist_by_res[int(rr)] = float(cKDTree(other).query(site_atoms, k=1)[0].min())

        near = {r: d for r, d in dist_by_res.items() if d < 12.0}
        feat["dwd12"] = float(sum(np.exp(-d / 6.0) for d in near.values()))
        feat["nres12"] = len(near)
        hyd = sum(1 for r, d in dist_by_res.items()
                  if d < 10.0 and NAME3_TO_1.get(resnames.get(r, "UNK"), "X") in HYDROPHOBIC)
        n10 = sum(1 for d in dist_by_res.values() if d < 10.0)
        feat["hydfrac10"] = (hyd / n10) if n10 else float("nan")
        close = {r: d for r, d in dist_by_res.items() if d < 8.0}
        feat["contact_order"] = (min(abs(r - pos1) for r in close) if close else 999)
        feat["centdist"] = float(np.linalg.norm(site_ca_w - ca_all.mean(0)))
    except Exception:
        pass

    mut_path = os.path.join(KV, f"{aa1}{pos1}{aa2}{SUFFIX}")
    if not os.path.exists(mut_path):
        feat["pdb_found"] = 0
        return feat
    feat["pdb_found"] = 1

    try:
        mst = parser.get_structure("m", mut_path)
        ca_m, bf_m, resseq_m = ca_and_bf(mst)
    except Exception:
        feat["pdb_found"] = -1
        return feat

    common = sorted(set(resseq_wt.tolist()) & set(resseq_m.tolist()))
    idx_w = {r: k for k, r in enumerate(resseq_wt)}
    idx_m = {r: k for k, r in enumerate(resseq_m)}
    iw = np.array([idx_w[r] for r in common])
    im = np.array([idx_m[r] for r in common])
    A, B = ca_m[im], ca_wt[iw]
    R, t = kabsch(A, B)
    dev = np.linalg.norm(A @ R.T + t - B, axis=1)
    feat["grmsd"] = float(np.sqrt((dev ** 2).mean()))

    kk = idx_w.get(pos1)
    if kk is not None:
        jj = [j for j, r in enumerate(common) if r == pos1]
        if jj:
            feat["dev_site"] = float(dev[jj[0]])
        for w in (2, 4, 8, 12):
            lo, hi = max(0, kk - w), min(len(ca_wt), kk + w + 1)
            sel = [j for j, r in enumerate(common) if lo <= idx_w[r] < hi]
            if sel:
                feat[f"locdev{w}"] = float(np.sqrt(np.mean(dev[sel] ** 2)))

        # pLDDT (AF2 confidence) deltas
        d = bf_m[im] - bf_wt[iw]
        if jj:
            feat["dplddt_site"] = float(d[jj[0]])
        for w in (3, 6, 10):
            lo, hi = max(0, kk - w), min(len(ca_wt), kk + w + 1)
            sel = [j for j, r in enumerate(common) if lo <= idx_w[r] < hi]
            if sel:
                feat[f"dplddt{w}"] = float(np.mean(d[sel]))
        lo, hi = max(0, kk - 6), min(len(ca_wt), kk + 7)
        sel = [j for j, r in enumerate(common) if lo <= idx_w[r] < hi]
        if sel:
            feat["plddt_min_mut_w6"] = float(np.min(bf_m[im][sel]))
            feat["plddt_min_wt_w6"] = float(np.min(bf_wt[iw][sel]))

        # AF2 confidence of the mutated residue itself, and relative to the WT
        jj2 = [j for j, r in enumerate(common) if r == pos1]
        if jj2:
            bm_site = float(bf_m[im][jj2[0]])
            bw_site = float(bf_wt[iw][jj2[0]])
            feat["mut_plddt_site"] = bm_site
            feat["rel_plddt_site"] = bm_site / bw_site if bw_site else float("nan")

    # SASA of the mutated residue in the mutant model (exact single-residue
    # Shrake-Rupley -- see sota/sasa_fast.py).
    try:
        mc = list(mst[0].get_chains())[0]
        atoms = list(mc.get_atoms())
        coords_m = np.array([a.coord for a in atoms], dtype=np.float64)
        elements_m = [a.element for a in atoms]
        subset = [k for k, a in enumerate(atoms)
                  if a.get_parent().id[0] == " " and a.get_parent().id[1] == pos1]
        if subset:
            vals = _G["fastsr"].atom_sasa(coords_m, elements_m, subset)
            site_sasa = float(sum(vals.values()))
            feat["sasa_mut_site"] = site_sasa
            feat["dsasa_site"] = site_sasa - feat["sasa_wt_site"]
    except Exception:
        pass

    # the two original contact features (reference Biopython walk)
    try:
        mc = list(mst[0].get_chains())[0]
        feat["contacts_mut"] = all_atom_contacts(mc, pos1)
        feat["contacts_wt"] = all_atom_contacts(chain, pos1)
        feat["dcontacts"] = feat["contacts_mut"] - feat["contacts_wt"]
    except Exception:
        pass

    # extra packing features (KD-tree, side-chain heavy atoms)
    try:
        mcoords, mheavy, mres, mside, mca = structure_arrays(mst)
        feat["cm_mut_8"] = contacts_of_residue(mcoords, mheavy, mres, mside, pos1, 8.0, True)
        for cutoff in (6.0, 10.0, 12.0):
            feat[f"cm_mut_{int(cutoff)}"] = contacts_of_residue(
                mcoords, mheavy, mres, mside, pos1, cutoff, True)
        feat["rgyr_mut"] = radius_of_gyration(mcoords[mheavy])
        feat["drgyr"] = feat["rgyr_mut"] - feat["rgyr_wt"]

        # local CA deviation restricted to the contact shell around the site
        try:
            site_ca_m = mcoords[(mres == pos1) & mca][0]
            shell = set(int(x) for x in np.unique(
                wres[np.linalg.norm(wcoords - site_ca_m, axis=1) < 10.0]).tolist())
            sel = [j for j, r in enumerate(common) if r in shell]
            if sel:
                feat["locdev_shell"] = float(np.sqrt(np.mean(dev[sel] ** 2)))
        except Exception:
            pass
    except Exception:
        pass

    return feat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper-data", default="/paper_data")
    ap.add_argument("--out", default="/repo/outputs/struct_features.csv")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    test = pd.read_csv(Path(args.paper_data) / "test.csv")
    recs = test.to_dict("records")
    if args.limit:
        recs = recs[:args.limit]

    with Pool(args.workers, initializer=init_worker, initargs=(WT_PDB,)) as pool:
        rows = pool.map(one_row, recs, chunksize=8)

    out = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print("wrote", args.out, out.shape, flush=True)
    print(out.describe().T.to_string())


if __name__ == "__main__":
    main()
