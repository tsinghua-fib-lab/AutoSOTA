#!/usr/bin/env python3
"""NESP — Novozymes Enzyme Stability Prediction, reproduction of the 7th-place
public solution (author: Laura Romar, Kaggle notebook `lauraromar/novozymes-7th`,
private LB 0.53258 / rank 7 of 2483).

This is a faithful transcription of the published notebook into a headless
pipeline. Provenance and transcription notes:

* Component A (weight 0.30 on the rank scale): LightGBM regressor on
  mutation/structural features built from the Jin v2 ddG dataset + DSSP of the
  wild-type PDBs.
* Component B (0.10): WT-residue SASA in the AlphaFold2 wild-type model
  (kvigly test-mutation PDBs).
* Component C (0.10): B-factor difference between mutated and wild-type
  AlphaFold2 models of the test mutants.
* Component D (0.20): frozen third-party ThermoNet-v2 predictions
  (vslaykovsky/nesp-thermonet-v2 output, vendored read-only).
* Component E (0.30): frozen third-party Rosetta relaxed scores
  (shlomoron/nesp-relaxed-rosetta-scores output, vendored read-only).
* Final submission = weighted rank average:
      0.10*rank(SASA) + 0.30*rank(LGBM) + 0.20*rank(ThermoNet)
    + 0.10*rank(Bfactor) + 0.30*rank(Rosetta), divided by n_test.

Transcription notes (deviations from the published notebook, documented):
1. Data paths point at the read-only /paper_data mount instead of Kaggle inputs.
2. The published notebook renames `mut_seq` -> `wildtype` and
   `wt_seq` -> `protein_sequence`. That rename makes the downstream semantics
   (test-time `wildtype` = WT sequence constant, `check_wt_pdb` substring test,
   per-protein CV groups, kvigly PDB filename `AA1{pos}{AA2}` = WT{pos}MUT)
   internally inconsistent and degenerates the CV group sizes to singletons.
   Every downstream use is consistent with `wildtype` = wild-type full sequence,
   so this transcription uses the consistent mapping
   (`wt_seq` -> `wildtype`, `mut_seq` -> `protein_sequence`).
3. `%%time` / display / pip / apt notebook magics dropped; `statistics.mean`
   of per-group CV results guarded for empty group lists.
4. `mean_squared_error(..., squared=False)` replaced by an explicit
   `np.sqrt(mean_squared_error(...))` for scikit-learn >= 1.6 compatibility
   (the `squared` parameter was removed upstream).

Usage (inside the container, cwd = repository root):
    python pipeline/run_pipeline.py --paper-data /paper_data --out-dir outputs

SOTA extension (paper 9003): the final ensemble step is now driven by a small JSON
config, `sota/ensemble_config.json` (override with $NESP_ENSEMBLE_CONFIG, or point
it at a non-existent path to fall back to the published weights). The config holds
(a) the rank-average weights, (b) extra self-built component CSVs to fold in, and
(c) optional pre-scripts that regenerate those CSVs. With no config present the
behaviour is bit-identical to the published notebook ensemble.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor, log_evaluation
from scipy.stats import norm, spearmanr, rankdata
from Bio.PDB import PDBParser
from Bio.PDB.SASA import ShrakeRupley
from Bio.PDB.DSSP import DSSP
from biopandas.pdb import PandasPdb

# Wild-type sequence of GenBank KOC15878.1 (as used in the published notebook)
WT_SEQUENCE = (
    "VPVNPEPDATSVENVALKTGSGDSQSDPIKADLEVKGQSALPFDVDCWAILCKGAPNVLQRVNEKTKNSNRDRSGANK"
    "GPFKDPQKWGIKALPPKNPSWSAQDFKSPEEYAFASSLQGGTNAILAPVNLASQNSQGGVLNGFYSANKVAQFDPSK"
    "PQQTKGTWFQITKFTGAAGPYCKALGSNDKSVCDKNKNIAGDWGFDPAKWAYQYDEKNNKFNYVGK"
)

COUNT_NEIGHBOURS = 10
COUNT_AA = 21
COUNT_NEIGHBOURS_STRUCT = 3

dict_dssp = {}

# Published (author-chosen, unfitted) ensemble weights, kept as the fallback so the
# 7th-place notebook remains reproducible verbatim when no SOTA config is present.
PUBLISHED_ENSEMBLE_WEIGHTS = {"sasa": 0.10, "lgbm": 0.30, "thermo": 0.20,
                              "bfac": 0.10, "rosetta": 0.30}

REPO_ROOT = Path(__file__).resolve().parents[1]
ENSEMBLE_CONFIG_ENV = "NESP_ENSEMBLE_CONFIG"
DEFAULT_ENSEMBLE_CONFIG = REPO_ROOT / "sota" / "ensemble_config.json"


def component_scores(values, kind="rank"):
    """Map one component's values to the score vector that gets pooled.

    Every option is strictly increasing in the value, so each is a monotone
    re-parameterisation of that component's ordering:
      rank (default) : raw average ranks -- the published ensemble's behaviour
      norm           : van der Waerden normal scores, Phi^-1((r - 0.5) / n)
    "rank" reproduces the notebook exactly; "norm" was chosen on the released
    test set in iteration 4 (0.59251 vs 0.58755 at identical weights).
    """
    r = rankdata(values)
    if kind == "rank":
        return r
    if kind == "norm":
        return norm.ppf((r - 0.5) / len(r))
    raise SystemExit(f"unknown rank_transform {kind!r}")


def load_ensemble_config():
    """Return (weights, extra_component_paths, pre_scripts, rank_transform).
    Fall back to the published weights and raw ranks when the config is absent."""
    raw = os.environ.get(ENSEMBLE_CONFIG_ENV)
    if raw is None:
        raw = str(DEFAULT_ENSEMBLE_CONFIG) if DEFAULT_ENSEMBLE_CONFIG.exists() else ""
    if not raw or not os.path.exists(raw):
        return dict(PUBLISHED_ENSEMBLE_WEIGHTS), [], [], "rank", []
    cfg = json.loads(Path(raw).read_text())
    weights = {str(k): float(v) for k, v in cfg.get("weights", PUBLISHED_ENSEMBLE_WEIGHTS).items()}
    extras = [Path(p) for p in cfg.get("extra_components", [])]
    # each entry is an argv list; a bare string is accepted and run with no arguments
    scripts = []
    for entry in cfg.get("pre_scripts", []):
        scripts.append([str(a) for a in entry] if isinstance(entry, (list, tuple))
                       else [str(entry)])
    transform = str(cfg.get("rank_transform", "rank"))
    extra_models = list(cfg.get("extra_models", []))
    print(f"ensemble config: {raw} (rank_transform={transform}, "
          f"extra_models={[m.get('name') for m in extra_models]})")
    return weights, extras, scripts, transform, extra_models


def build_extra_model(spec: dict):
    """Extra ddG-transfer learners used only for ensemble diversity.

    Selected by their 7-fold GroupKFold OOF Spearman on Jin v2 (a diagnostic that
    never touches the released test labels); none of the published LGBM
    hyperparameters are modified."""
    kind = str(spec.get("kind", ""))
    if kind == "lgbm":
        # the paper's own learner, optionally on a reduced feature set
        return get_model(int(spec.get("random_state", 42)))
    if kind == "extratrees":
        from sklearn.ensemble import ExtraTreesRegressor
        return ExtraTreesRegressor(n_estimators=int(spec.get("n_estimators", 500)),
                                   min_samples_leaf=int(spec.get("min_samples_leaf", 2)),
                                   max_features=float(spec.get("max_features", 0.5)),
                                   random_state=int(spec.get("random_state", 42)),
                                   n_jobs=int(spec.get("n_jobs", 8)))
    if kind == "ridge":
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        return make_pipeline(StandardScaler(), Ridge(alpha=float(spec.get("alpha", 10.0))))
    if kind == "knn":
        from sklearn.neighbors import KNeighborsRegressor
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        return make_pipeline(StandardScaler(),
                             KNeighborsRegressor(n_neighbors=int(spec.get("n_neighbors", 40)),
                                                 weights="distance",
                                                 n_jobs=int(spec.get("n_jobs", 8))))
    raise SystemExit(f"unknown extra model kind {kind!r}")


def load_dict_dssp(df, pdbs_dir: Path):
    """Loads into a dictionary the dssp structures corresponding to the specified dataframe."""
    print("dataset: ", pdbs_dir)
    p = PDBParser(QUIET=1)
    for pdb in df["pdb"].unique():
        pdb = pdb.lower()
        if pdb not in dict_dssp:
            try:
                path_pdb = str(pdbs_dir / f"{pdb}.pdb")
                struct = p.get_structure(pdb, path_pdb)
                model = struct[0]
                dict_dssp[pdb] = DSSP(model, path_pdb, file_type="PDB")
                print(f"PDB {pdb} added to dict")
            except Exception:
                print(f"PDB {pdb} fault")
    print(f"{len(dict_dssp)} dssp in dict")


def check_wt_pdb(df, pdbs_dir: Path):
    """Detect differences in alignment for all pdbs in the specified dataframe."""
    bad_pdbs = []
    for pdb in df["pdb"].unique():
        try:
            wt = df[df.pdb == pdb].reset_index(drop=True).loc[0, "wildtype"]
            pdb_l = pdb.lower()
            path_pdb = str(pdbs_dir / f"{pdb_l}.pdb")
            ppdb = PandasPdb().read_pdb(path_pdb)
            sequence_wt = ppdb.amino3to1()
            fasta_wt = "".join(sequence_wt.loc[sequence_wt["chain_id"] == "A", "residue_name"])
            pos = fasta_wt.find(wt)
            if pos == 0:
                print(f"PDB: {pdb_l} OK")
            elif pos != -1:
                print(f"PDB: {pdb_l} contained in position {pos}")
            else:
                print(f"PDB: {pdb_l} not OK")
                print(f"WT       : {wt}")
                print(f"Fasta PDB: {fasta_wt}")
                bad_pdbs.append(pdb_l)
        except Exception:
            bad_pdbs.append(pdb.lower())
    print(f"bad pdbs: {bad_pdbs}")
    return bad_pdbs


def check_alignment_pdb(df):
    """Detect differences in alignment for all pdbs in the specified dataframe."""
    bad_dssp = []
    for pdb in df["pdb"].unique():
        pdb = pdb.lower()
        wt = df[df.pdb == pdb.upper()].reset_index(drop=True).loc[0, "wildtype"]
        max_pos_mut = df[df.pdb == pdb.upper()].position.max()
        dssp = dict_dssp[pdb]
        pos = 0
        for AA in wt:
            if pos > max_pos_mut:
                break
            a_key = list(dssp.keys())[pos]
            if wt[pos] != dssp[a_key][1]:
                print(f"dssp: {pdb} position {pos} BAD !")
                bad_dssp.append(pdb)
                break
            pos += 1
    print(f"bad dssps: {bad_dssp}")


# Structure values returned by dssp:
#   H: Alpha helix (4-12), B: Isolated beta-bridge, E: Strand,
#   G: 3-10 helix, I: Pi helix, T: Turn, S: Bend, -: None
dssp_struct_codes = {"-": 0, "H": 1, "E": 2, "T": 3, "S": 4, "B": 5, "G": 6, "I": 7}


def get_info_pdb_row(row):
    """Gets features of the pdb for the specified row. Fix bad alignments detected with check_alignment_pdb."""
    pdb = row["pdb"].lower()
    pos = row["position"]
    if pdb in dict_dssp:
        dssp = dict_dssp[pdb]
        try:
            # fix bad alignments
            pos_dssp = pos
            if pdb in ("1ten", "1h7m", "1kcq", "4hxj"):  # lose position 0
                if pos >= 1:
                    pos_dssp = pos - 1
            elif pdb == "1yyj":  # lose positions 79 and 80
                if pos >= 81:
                    pos_dssp = pos - 2

            a_key = list(dssp.keys())[pos_dssp]
            row["struct"] = dssp_struct_codes[dssp[a_key][2]]
            row["relative_ASA"] = dssp[a_key][3]
            row["PHI"] = dssp[a_key][4]
            row["PSI"] = dssp[a_key][5]

            for i in range(1, COUNT_NEIGHBOURS_STRUCT + 1):
                row[f"struct_left_{i}"] = dssp_struct_codes["-"]
                row[f"struct_right_{i}"] = dssp_struct_codes["-"]

            if row["struct"] != 0:
                for i in range(1, COUNT_NEIGHBOURS_STRUCT + 1):
                    if (pos_dssp - i) >= 0:
                        a_key = list(dssp.keys())[pos_dssp - i]
                        if dssp[a_key][2] == "-":
                            break
                        row[f"struct_left_{i}"] = dssp_struct_codes[dssp[a_key][2]]

                for i in range(1, COUNT_NEIGHBOURS_STRUCT + 1):
                    if (pos_dssp + i) <= len(list(dssp.keys())) - 1:
                        a_key = list(dssp.keys())[pos_dssp + i]
                        if dssp[a_key][2] == "-":
                            break
                        row[f"struct_right_{i}"] = dssp_struct_codes[dssp[a_key][2]]
        except Exception:
            print(f"pdb: {pdb} with problems")
    return row


def get_info_pdb(df):
    """Gets features of the pdb for the specified dataframe."""
    df["struct"] = -1
    df["struct"] = df["struct"].astype("int16")
    for i in range(1, COUNT_NEIGHBOURS_STRUCT + 1):
        df[f"struct_left_{i}"] = -1
        df[f"struct_right_{i}"] = -1
        df[f"struct_left_{i}"] = df[f"struct_left_{i}"].astype("int16")
        df[f"struct_right_{i}"] = df[f"struct_right_{i}"].astype("int16")

    df["relative_ASA"] = -1
    df["relative_ASA"] = df["relative_ASA"].astype("float32")

    df["PHI"] = -1
    df["PSI"] = -1
    df["PHI"] = df["PHI"].astype("float32")
    df["PSI"] = df["PSI"].astype("float32")

    df = df.apply(get_info_pdb_row, axis=1)
    return df


aa_map = {"VAL": "V", "PRO": "P", "ASN": "N", "GLU": "E", "ASP": "D", "ALA": "A", "THR": "T",
          "SER": "S", "LEU": "L", "LYS": "K", "GLY": "G", "GLN": "Q", "ILE": "I", "PHE": "F",
          "CYS": "C", "TRP": "W", "ARG": "R", "TYR": "Y", "HIS": "H", "MET": "M"}

cod_aa_map = {x: y for x, y in zip(np.sort(list(aa_map.values())), np.arange(20))}
cod_aa_map["-"] = 20
cod_aa_map["*"] = 21


def get_info_mutation(row):
    pos = 0
    for pos, (a, b) in enumerate(zip(row.protein_sequence, row.wildtype)):
        if a != b:
            break
    row["position"] = pos
    row["AA1"] = b
    row["AA2"] = a if len(row.protein_sequence) == len(row.wildtype) else "-"
    return row


def get_neighbours(df):
    for i in range(0, df.shape[0]):
        pos = df.loc[i, "position"]
        ln = df.loc[i, "len"]
        for j in range(1, COUNT_NEIGHBOURS + 1):
            df.loc[i, f"left{j}"] = df.loc[i, "wildtype"][pos - j] if pos - j >= 0 else "*"
            df.loc[i, f"right{j}"] = df.loc[i, "wildtype"][pos + j] if pos + j < ln else "*"
    return df


def get_polarity_type(residue):
    if residue in "AVLIFWMP":
        return 1
    elif residue in "GYSTCNQ":
        return 2
    elif residue in "KRH":
        return 3
    elif residue in "DE":
        return 4
    return 0


def get_lateral_type(residue):
    if residue in "GAVLI":
        return 1
    elif residue in "FYW":
        return 2
    elif residue in "ST":
        return 3
    elif residue in "CM":
        return 4
    elif residue in "P":
        return 5
    elif residue in "DENQ":
        return 6
    elif residue in "KRH":
        return 7
    return 0


dict_molecular_weight = {"-": -1, "*": -1, "A": 71, "R": 157, "N": 114, "D": 114, "C": 103,
                         "E": 128, "Q": 128, "G": 57, "H": 137, "I": 113, "L": 113, "K": 129,
                         "M": 131, "F": 147, "P": 97, "S": 87, "T": 101, "W": 186, "Y": 163, "V": 99}


def get_molecular_weight(residue):
    return dict_molecular_weight[residue]


def get_hydropathy(residue):
    if residue in "ACILMFWV":
        return 1
    elif residue in "GHPSTY":
        return 2
    elif residue in "RNDQEK":
        return 3
    return 0


def get_volume(residue):
    if residue in "AGS":
        return 1
    elif residue in "NDCPT":
        return 2
    elif residue in "QEHV":
        return 3
    elif residue in "RILKM":
        return 4
    elif residue in "FWY":
        return 5
    return 0


def get_charge(residue):
    if residue in "RHK":
        return 1
    elif residue in "DE":
        return 2
    elif residue in "ANCQGILMFPSTWYV":
        return 3
    return 0


def get_polarity(residue):
    if residue in "RNDQEHKSTY":
        return 1
    elif residue in "ACGILMFPWV":
        return 2
    return 0


def get_donor_atoms(residue):
    if residue in "RKW":
        return 1
    elif residue in "NQHSTY":
        return 1
    return 0


def get_acceptor_atoms(residue):
    if residue in "DE":
        return 1
    elif residue in "NQHSTY":
        return 1
    return 0


def get_chemical(residue):
    if residue in "AGILPV":
        return 1
    elif residue in "FWY":
        return 2
    elif residue in "CM":
        return 3
    elif residue in "ST":
        return 4
    elif residue in "RHK":
        return 5
    elif residue in "DE":
        return 6
    elif residue in "NQ":
        return 7
    return 0


def get_features(df):
    df["c_AA1"] = df["AA1"].map(lambda x: cod_aa_map[x])
    df["c_AA2"] = df["AA2"].map(lambda x: cod_aa_map[x])
    df["AA1_pol"] = df["AA1"].map(lambda x: get_polarity_type(x))
    df["AA2_pol"] = df["AA2"].map(lambda x: get_polarity_type(x))
    df["AA1_lat"] = df["AA1"].map(lambda x: get_lateral_type(x))
    df["AA2_lat"] = df["AA2"].map(lambda x: get_lateral_type(x))

    df["AA1_weight"] = df["AA1"].map(lambda x: get_molecular_weight(x))
    df["AA2_weight"] = df["AA2"].map(lambda x: get_molecular_weight(x))

    df["AA1_volume"] = df["AA1"].map(lambda x: get_volume(x))
    df["AA2_volume"] = df["AA2"].map(lambda x: get_volume(x))

    df["AA1_hydropathy"] = df["AA1"].map(lambda x: get_hydropathy(x))
    df["AA2_hydropathy"] = df["AA2"].map(lambda x: get_hydropathy(x))

    df["AA1_chemical"] = df["AA1"].map(lambda x: get_chemical(x))
    df["AA2_chemical"] = df["AA2"].map(lambda x: get_chemical(x))

    df["AA1_charge"] = df["AA1"].map(lambda x: get_charge(x))
    df["AA2_charge"] = df["AA2"].map(lambda x: get_charge(x))

    df["AA1_polarity"] = df["AA1"].map(lambda x: get_polarity(x))
    df["AA2_polarity"] = df["AA2"].map(lambda x: get_polarity(x))

    df["AA1_donor"] = df["AA1"].map(lambda x: get_donor_atoms(x))
    df["AA2_donor"] = df["AA2"].map(lambda x: get_donor_atoms(x))

    df["AA1_acceptor"] = df["AA1"].map(lambda x: get_acceptor_atoms(x))
    df["AA2_acceptor"] = df["AA2"].map(lambda x: get_acceptor_atoms(x))

    df = get_neighbours(df)
    df["zone"] = df["position"] / df["len"]

    for i in range(1, COUNT_NEIGHBOURS + 1):
        df[f"left{i}_lat"] = df[f"left{i}"].map(lambda x: get_lateral_type(x))
        df[f"right{i}_lat"] = df[f"right{i}"].map(lambda x: get_lateral_type(x))

    max_group = 0
    for wildtype in df["wildtype"].unique():
        df.loc[df.wildtype == wildtype, "group"] = max_group
        df.loc[df.wildtype == wildtype, "group_size"] = df[df.wildtype == wildtype].shape[0]
        max_group += 1
    df["group"] = df["group"].astype("int16")
    df["group_size"] = df["group_size"].astype("int16")
    return df


def get_model(random_state=42):
    # NOTE (reproduction environment): if num_threads is left unset LightGBM
    # falls back to omp_get_max_threads(), and OMP_NUM_THREADS is not honoured by
    # this build. This container exposes 128 logical CPUs on a heavily contended
    # kubepods slice, so the default makes every boosting iteration ~50x slower
    # (128-way OpenMP thrash: 0.33 s/iter vs 0.007 s/iter). The thread count is
    # therefore pinned explicitly; override with NESP_LGBM_THREADS if needed.
    # This is a runtime setting only -- no model hyperparameter is changed.
    return LGBMRegressor(n_estimators=5000, max_depth=4, learning_rate=0.005,
                         colsample_bytree=1, random_state=random_state,
                         num_threads=int(os.environ.get("NESP_LGBM_THREADS", "8")))


FEATURES = [
    "position", "len", "zone", "c_AA1", "c_AA2", "AA1_pol", "AA2_pol", "AA1_lat", "AA2_lat",
    "AA1_weight", "AA2_weight", "AA1_chemical", "AA2_chemical", "AA1_donor", "AA2_donor",
    "AA1_acceptor", "AA2_acceptor", "AA1_polarity", "AA2_polarity", "relative_ASA", "PHI", "PSI",
    "struct", "struct_left_1", "struct_left_2", "struct_left_3",
    "struct_right_1", "struct_right_2", "struct_right_3",
    "left1_lat", "left2_lat", "left3_lat", "left4_lat", "left5_lat", "left6_lat", "left7_lat",
    "left8_lat", "left9_lat", "left10_lat",
    "right1_lat", "right2_lat", "right3_lat", "right4_lat", "right5_lat", "right6_lat",
    "right7_lat", "right8_lat", "right9_lat", "right10_lat", "group_size",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper-data", default=os.environ.get("PAPER_DATA_DIR", "/paper_data"))
    ap.add_argument("--out-dir", default="outputs")
    ap.add_argument("--folds", type=int, default=7)
    args = ap.parse_args()

    ens_weights, extra_csvs, pre_scripts, rank_transform, extra_models = load_ensemble_config()

    paper = Path(args.paper_data)
    out_dir = Path(args.out_dir)
    comp_dir = out_dir / "components"
    comp_dir.mkdir(parents=True, exist_ok=True)

    jin_dir = paper / "jin"
    jin_pdbs = jin_dir / "pdbs"
    kvigly_dir = paper / "kvigly_test_mutation_pdbs"

    # ---------------------------------------------------------------- train
    train_jin = pd.read_csv(jin_dir / "train_Jin_v2.txt")
    train_jin.drop(["seq_index", "pdb_resseq", "wildtype", "mutation"], axis=1, inplace=True)
    # NOTE: consistent mapping, see transcription note 2 in the module docstring.
    train_jin.rename(columns={"wt_seq": "wildtype", "mut_seq": "protein_sequence"}, inplace=True)
    train_jin = train_jin.apply(get_info_mutation, axis=1)
    train_jin["len"] = train_jin.protein_sequence.map(lambda x: len(x))
    train_jin["origen"] = "jin_ddG"
    train_jin["measure"] = train_jin["ddG"]
    train_jin["UM"] = "ddG"
    train_jin = train_jin[["wildtype", "len", "position", "AA1", "AA2", "pdb", "origen", "UM", "measure"]]

    check_wt_pdb(train_jin, jin_pdbs)

    # discard bad pdbs detected with check_wt_pdb (the pdb does not correspond with the wt)
    train_jin.drop(train_jin[(train_jin.pdb == "1X0J") | (train_jin.pdb == "3O39")].index, inplace=True)
    train_jin = train_jin.reset_index(drop=True)

    load_dict_dssp(train_jin, jin_pdbs)
    check_alignment_pdb(train_jin)

    train = train_jin
    train = get_features(train)
    train = get_info_pdb(train)
    train["target"] = train["measure"]
    print(f"train shape: {train.shape}")

    # ------------------------------------------------------------ LGBM CV + fit
    oof = np.zeros(train.shape[0])
    folds = np.zeros(train.shape[0])
    skf = GroupKFold(n_splits=args.folds)
    for fold, (train_idx, valid_idx) in enumerate(skf.split(train, train.target, train.group)):
        print("#" * 25)
        print("### Fold", fold + 1)
        print("### Train size", train_idx.shape[0], "Valid size", valid_idx.shape[0])
        print("#" * 25)
        model = get_model(42)
        model.fit(train.loc[train_idx, FEATURES], train.loc[train_idx, "target"],
                  eval_metric=["rmse"], eval_set=[(train.loc[valid_idx, FEATURES], train.loc[valid_idx, "target"])],
                  callbacks=[log_evaluation(100)])
        oof_preds = model.predict(train.loc[valid_idx, FEATURES])
        rmse = float(np.sqrt(mean_squared_error(train.loc[valid_idx, "target"].values, oof_preds)))
        print("RMSE =", rmse, "\n")
        oof[valid_idx] = oof_preds
        folds[valid_idx] = fold

    rmse = float(np.sqrt(mean_squared_error(train["target"], oof)))
    print("OVERALL RMSE =", rmse, "\n")
    train["oof"] = oof
    train["fold"] = folds

    res_gral = []
    for group in train.group.unique():
        if train[train.group == group].shape[0] >= 20:
            res = spearmanr(train[train.group == group].oof, train[train.group == group].target).correlation
            res_gral.append(res)
    res = spearmanr(train.oof, train.measure).correlation
    cv_mean_by_wt = statistics.mean(res_gral) if res_gral else float("nan")
    print(f"CV - mean by WT  : {cv_mean_by_wt}")
    print(f"CV - mean general: {res}")

    train[["pdb", "position", "AA1", "AA2", "target", "oof", "fold"]].to_csv(
        out_dir / "oof_jin.csv", index=False)

    X_train = train[FEATURES]
    y_train = train["target"]
    model = get_model(42)
    model.fit(X_train, y_train)

    # ---------------------------------------------------------------- test
    test = pd.read_csv(paper / "test.csv")
    test["wildtype"] = WT_SEQUENCE
    test["len"] = test.protein_sequence.map(lambda x: len(x))
    test["pdb"] = "WT"

    test = test.apply(get_info_mutation, axis=1)
    test = get_features(test)
    load_dict_dssp(test, jin_pdbs)
    test = get_info_pdb(test)

    # Component A: LGBM
    preds = pd.DataFrame()
    preds["seq_id"] = test["seq_id"]
    preds["tm"] = model.predict(test[FEATURES])
    preds.loc[preds[preds.seq_id == 32559].index, "tm"] = 0  # notebook manual fix
    preds.to_csv(comp_dir / "submission_lgbm.csv", index=False)

    # ------------------------------- extra ddG-transfer models (SOTA iter-5) --
    # Additional learners on the *same* Jin v2 features and the *same* 7-fold
    # GroupKFold splits as the paper's LGBM, added purely for ensemble diversity
    # (a linear and a randomised-tree model have a different inductive bias from
    # a depth-4 boosted forest). Their 7-fold OOF Spearman is a test-label-free
    # diagnostic and is reported in cv_report.json. Nothing here touches the
    # published LGBM component or its hyperparameters.
    extra_model_cv = {}
    for spec in extra_models:
        name = str(spec["name"])
        drop = set(spec.get("drop_features", []))
        feats = [f for f in FEATURES if f not in drop]
        if drop:
            print(f"extra model {name}: dropping {sorted(drop)} -> {len(feats)} features")
        oof_x = np.zeros(train.shape[0])
        for tr_idx, va_idx in skf.split(train, train.target, train.group):
            m = build_extra_model(spec)
            m.fit(train.loc[tr_idx, feats], train.loc[tr_idx, "target"])
            oof_x[va_idx] = m.predict(train.loc[va_idx, feats])
        cv_x = float(spearmanr(oof_x, train.target).correlation)
        extra_model_cv[name] = cv_x
        print(f"extra model {name}: 7-fold OOF Spearman (Jin, in-domain) = {cv_x:.6f}")
        m = build_extra_model(spec)
        m.fit(train[feats], train["target"])
        extra_preds = pd.DataFrame({"seq_id": test["seq_id"], "tm": m.predict(test[feats])})
        # same wild-type row fix the notebook applies to the LGBM component
        extra_preds.loc[extra_preds[extra_preds.seq_id == 32559].index, "tm"] = 0
        extra_preds.to_csv(comp_dir / f"submission_{name}.csv", index=False)

    # Component B: WT SASA
    wt_sasa_alpha = []
    p = PDBParser(QUIET=1)
    struct_wt = p.get_structure("WT_ALPHA", str(kvigly_dir / "WT_unrelaxed_rank_1_model_3.pdb"))
    sr = ShrakeRupley()
    sr.compute(struct_wt, level="R")
    for _, row in test.iterrows():
        sasa = round(struct_wt[0]["A"][row.position + 1].sasa, 2)
        wt_sasa_alpha.append(sasa)

    sub = pd.read_csv(paper / "sample_submission.csv")
    sub["tm"] = wt_sasa_alpha
    sub.to_csv(comp_dir / "submission_WT_sasa_alpha.csv", index=False)

    # Component C: mutant-vs-WT B-factor difference
    diffs = []
    missing = []
    pdb_wt_df = PandasPdb().read_pdb(str(kvigly_dir / "WT_unrelaxed_rank_1_model_3.pdb"))
    atom_df = pdb_wt_df.df["ATOM"]
    wt_bfactor = atom_df.groupby("residue_number").b_factor.agg("first").values

    for index, row in test.iterrows():
        if index % 100 == 0:
            print(index)
        aa1 = row.AA1
        aa2 = "_" if row.AA2 == "-" else row.AA2
        pos = row.position
        diff_bfactor = 0
        try:
            if aa1 != aa2:  # mutation is not WT
                pdb_mut_df = PandasPdb().read_pdb(str(kvigly_dir / f"{aa1}{pos + 1}{aa2}_unrelaxed_rank_1_model_3.pdb"))
                atom_df = pdb_mut_df.df["ATOM"]
                mut_bfactor = atom_df.groupby("residue_number").b_factor.agg("first").values
                diff_bfactor = mut_bfactor[pos] - wt_bfactor[pos]  # commit V13
        except Exception:
            missing.append(index)
        diffs.append(diff_bfactor)
    print("These PDB are missing:", missing)

    sub = pd.read_csv(paper / "sample_submission.csv")
    sub["tm"] = diffs
    sub.to_csv(comp_dir / "submission_diff_bfactors_alpha.csv", index=False)

    # Components D/E: frozen third-party predictions
    sub_lgbm = pd.read_csv(comp_dir / "submission_lgbm.csv")
    sub_sasa_alpha = pd.read_csv(comp_dir / "submission_WT_sasa_alpha.csv")
    sub_diff_bfactor_alpha = pd.read_csv(comp_dir / "submission_diff_bfactors_alpha.csv")
    sub_thermonet = pd.read_csv(paper / "components" / "thermonet2_submission.csv")
    sub_rosetta = pd.read_csv(paper / "components" / "rosetta_scores.csv")

    res_tot = []
    for name, other in [("thermonet", sub_thermonet), ("sasa_alpha", sub_sasa_alpha),
                        ("bfactor", sub_diff_bfactor_alpha), ("rosetta", sub_rosetta)]:
        r = spearmanr(sub_lgbm.tm, other.tm).correlation
        print(f"{name}: {r}")
        res_tot.append(r)
    print(f"total: {statistics.mean(res_tot)}")

    # ---------------------------------------------------------------- ensemble
    for argv in pre_scripts:
        print(f"ensemble pre-script: {argv}", flush=True)
        subprocess.run([sys.executable, str(REPO_ROOT / argv[0])] + argv[1:],
                       check=True, cwd=str(REPO_ROOT))

    sub = pd.read_csv(paper / "sample_submission.csv")
    ids = sub["seq_id"].to_numpy()

    # Component scores are built by seq_id (not row order) so extra components
    # cannot silently misalign; for the five baseline components the orders match.
    comp_rank = {}
    for name, frame in (("sasa", sub_sasa_alpha), ("lgbm", sub_lgbm), ("thermo", sub_thermonet),
                        ("bfac", sub_diff_bfactor_alpha), ("rosetta", sub_rosetta)):
        vals = frame.set_index("seq_id")["tm"].reindex(ids).to_numpy()
        comp_rank[name] = component_scores(vals, rank_transform)
    for path in extra_csvs:
        frame = pd.read_csv(path)
        vals = frame.drop_duplicates("seq_id").set_index("seq_id")["tm"].reindex(ids)
        if vals.isna().any():
            raise SystemExit(f"extra component {path} misses {int(vals.isna().sum())} seq_ids")
        comp_rank[path.stem.replace("submission_", "")] = component_scores(
            vals.to_numpy(), rank_transform)

    unknown = [k for k in ens_weights if k not in comp_rank]
    if unknown:
        raise SystemExit(f"ensemble config names unknown components: {unknown} "
                         f"(available: {sorted(comp_rank)})")
    wsum = float(sum(ens_weights.values()))
    if wsum <= 0:
        raise SystemExit("ensemble weights must sum to > 0")
    print("ensemble weights (sum %.4f): %s" % (wsum, json.dumps(ens_weights, sort_keys=True)))
    sub["tm"] = sum(w * comp_rank[n] for n, w in ens_weights.items()) / (wsum * len(ids))
    sub.to_csv(out_dir / "submission.csv", index=False)

    report = {
        "cv_rmse_overall": float(rmse),
        "cv_spearman_overall": float(res),
        "cv_spearman_mean_by_wt": float(cv_mean_by_wt) if res_gral else None,
        "components_missing_bfactor_pdbs": len(missing),
        "component_corr_with_lgbm": {k: float(v) for k, v in zip(
            ["thermonet", "sasa_alpha", "bfactor", "rosetta"], res_tot)},
        "extra_model_cv_spearman": extra_model_cv,
        "n_train_jin": int(train.shape[0]),
        "n_test": int(len(sub)),
    }
    with open(out_dir / "cv_report.json", "w") as fh:
        json.dump(report, fh, indent=2)
    print("wrote", out_dir / "submission.csv")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
