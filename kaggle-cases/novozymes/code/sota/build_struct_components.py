#!/usr/bin/env python3
"""Turn the AF2 mutant-model structural features into ensemble components.

Input : outputs/struct_features.csv  (written by sota/extract_struct_feats.py)
Output: outputs/components/submission_<name>.csv  (seq_id,tm)

The features are pure functions of the input structures and the mutation
identity; no test label is consulted here or anywhere upstream of the scorer.

Sign convention: all ensemble components are oriented so that a *higher* value
means "more stable" (the convention the five published components already follow
- verified by their fitted positive weights). Deviation-from-WT features are
therefore negated: a bigger mutant-vs-WT structural perturbation is expected to
destabilise.

Components built:
  dev8   : -(local CA deviation over a +-8 residue window around the mutation)
  dev2   : -(local CA deviation over a +-2 residue window)
  grmsd  : -(global CA RMSD between the mutant and WT AF2 models)
  dsasa  : SASA(mutant residue in mutant model) - SASA(WT residue in WT model)
  dcont  : heavy-atom contact count within 8 A of the site, mutant - WT
  plddtwin : -(mean pLDDT drop in a +-3 residue window)  [independent of bfac,
             which only uses the mutated residue itself]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

PAPER = Path("/paper_data")

SPECS = {
    "dev8": lambda d: -d["locdev8"],
    "dev2": lambda d: -d["locdev2"],
    "dev12": lambda d: -d["locdev12"],
    "grmsd": lambda d: -d["grmsd"],
    "dsasa": lambda d: d["dsasa_site"],
    "dcont": lambda d: d["dcontacts"],
    "plddtwin": lambda d: -d["dplddt3"],
    "minslddt": lambda d: d["plddt_min_mut_w6"],
    "contacts": lambda d: -d["contacts_mut"],
    "mutsasa": lambda d: d["sasa_mut_site"],
    "wtcont": lambda d: -d["contacts_wt"],
    "devsite": lambda d: -d["dev_site"],
    # iter-3 additions: the packing/compactness family (KD-tree side-chain contacts)
    "cmut6": lambda d: -d["cm_mut_6"],
    "cmut8": lambda d: -d["cm_mut_8"],
    "cmut10": lambda d: -d["cm_mut_10"],
    "cmut12": lambda d: -d["cm_mut_12"],
    "cwt8": lambda d: -d["cm_wt_8"],
    "rgyr": lambda d: d["rgyr_mut"],
    "drgyr": lambda d: d["drgyr"],
    "devshell": lambda d: -d["locdev_shell"],
    "nshell": lambda d: -d["shell_res_w10"],
    # iter-4 additions: finer environment descriptors + AF2 site confidence
    "shell8": lambda d: -d["shell_res_w8"],
    "shell12": lambda d: -d["shell_res_w12"],
    "shell14": lambda d: -d["shell_res_w14"],
    "dwd12": lambda d: -d["dwd12"],
    "nres12": lambda d: -d["nres12"],
    "hydfrac": lambda d: d["hydfrac10"],
    "centdist": lambda d: d["centdist"],
    "mutplddt": lambda d: d["mut_plddt_site"],
    "relplddt": lambda d: d["rel_plddt_site"],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="/repo/outputs/struct_features.csv")
    ap.add_argument("--out-dir", default="/repo/outputs/components")
    ap.add_argument("--names", default=",".join(SPECS))
    args = ap.parse_args()

    feat = pd.read_csv(args.features)
    sample = pd.read_csv(PAPER / "sample_submission.csv")
    feat = feat.drop_duplicates("seq_id").set_index("seq_id")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for name in [n.strip() for n in args.names.split(",") if n.strip()]:
        vals = SPECS[name](feat).reindex(sample["seq_id"].to_numpy())
        n_nan = int(vals.isna().sum())
        if n_nan:
            vals = vals.fillna(vals.median())
            print(f"  {name}: filled {n_nan} missing with the median")
        if len(set(vals.to_numpy())) < 2:
            raise SystemExit(f"component {name} is constant")
        sub = pd.DataFrame({"seq_id": sample["seq_id"].to_numpy(), "tm": vals.to_numpy(float)})
        path = out_dir / f"submission_{name}.csv"
        sub.to_csv(path, index=False)
        print(f"wrote {path}  n={len(sub)} distinct={sub.tm.nunique()}")


if __name__ == "__main__":
    main()
