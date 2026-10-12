#!/usr/bin/env python3
"""S2F evaluation wrapper for autosota (multi-GPU parallel).

Runs S2F evaluate.py on a list of assays, sharding across multiple GPUs,
parses the per-GPU results.csv files, and outputs the primary metric
(Avg Spearman).

Usage:
  python scripts/s2f_eval.py [--gpus 0,1,2,3] [--subset dev|full|--assay-ids ...]
"""
import argparse
import csv
import datetime
import os
import subprocess
import sys
import tempfile
import yaml

# Layout: <project_root>/{S3F,autosota-util}; this file is autosota-util/scripts/.
# Derived rather than hardcoded so the tree can be relocated without edits.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))

# This host has driver CUDA 13; the s3f env is torch 1.8.1+cu111. Child
# evaluate.py processes inherit these so torch-sparse / KeOps can find
# CUDA 11 libs and headers even if the caller forgot to export them.
_conda = os.environ.get("CONDA_PREFIX") or "/opt/conda/envs/s3f"
_lib = os.path.join(_conda, "lib")
if os.path.isdir(_lib):
    _old = os.environ.get("LD_LIBRARY_PATH", "")
    if _lib not in _old.split(":"):
        os.environ["LD_LIBRARY_PATH"] = _lib + (":" + _old if _old else "")
os.environ.setdefault("CUDA_PATH", os.path.join(_conda, "cuda11"))
os.environ.setdefault("KEOPS_CACHE_FOLDER",
                      os.path.expanduser("~/.cache/keops2.3"))

# Summary log: every eval appends one row here for iteration tracking
SUMMARY_CSV = os.path.join(PROJECT_ROOT,
                           "autosota-util/results/s2f_eval_summary.csv")

S3F_REPO = os.environ.get("S2F_REPO_DIR", os.path.join(PROJECT_ROOT, "S3F"))
DATADIR = os.path.join(S3F_REPO, "dataset/DMS_ProteinGym_substitutions")
STRUCTDIR = os.path.join(S3F_REPO, "dataset/ProteinGym_AF2_structures")
DEFAULT_CKPT = os.path.abspath(os.path.join(S3F_REPO, "dataset/s2f.pth"))
BASE_CONFIG = os.path.join(S3F_REPO, "config/evaluate/s2f.yaml")
OUTPUT_ROOT = os.path.join(S3F_REPO, "results")

# 40 fixed assays for dev-subset eval, stratified by Selection Type
# (OrganismalFitness 15 / Stability 13 / Activity 9 / Expression 3) to mirror
# the full 204 non-binding set. FIXED — never re-sampled between iterations,
# so score changes are attributable to code changes, not sampling noise.
# Binding/interface assays excluded (user preference: no interface data).
DEV_SUBSET = [
    "A0A140D2T1_ZIKV_Sourisseau_2019",
    "A0A1I9GEU1_NEIME_Kennouche_2019",
    "ANCSZ_Hobbs_2022",
    "BCHB_CHLTE_Tsuboyama_2023_2KRU",
    "BLAT_ECOLX_Deng_2012",
    "BRCA1_HUMAN_Findlay_2018",
    "CAS9_STRP1_Spencer_2017_positive",
    "CBPA2_HUMAN_Tsuboyama_2023_1O6X",
    "ENV_HV1B9_DuenasDecamp_2016",
    "ENV_HV1BR_Haddox_2016",
    "GAL4_YEAST_Kitzman_2015",
    "HECD1_HUMAN_Tsuboyama_2023_3DKM",
    "HSP82_YEAST_Mishra_2016",
    "HXK4_HUMAN_Gersing_2022_activity",
    "HXK4_HUMAN_Gersing_2023_abundance",
    "KCNE1_HUMAN_Muhammad_2023_expression",
    "KCNH2_HUMAN_Kozek_2020",
    "KCNJ2_MOUSE_Coyote-Maestas_2022_function",
    "KKA2_KLEPN_Melnikov_2014",
    "LGK_LIPST_Klesmith_2015",
    "MBD11_ARATH_Tsuboyama_2023_6ACV",
    "MET_HUMAN_Estevam_2023",
    "MK01_HUMAN_Brenan_2016",
    "PITX2_HUMAN_Tsuboyama_2023_2L7M",
    "POLG_HCVJF_Qi_2014",
    "POLG_PESV_Tsuboyama_2023_2MXD",
    "RBP1_HUMAN_Tsuboyama_2023_2KWH",
    "RD23A_HUMAN_Tsuboyama_2023_1IFY",
    "RDRP_I33A0_Li_2023",
    "RFAH_ECOLI_Tsuboyama_2023_2LCL",
    "RL40A_YEAST_Roscoe_2014",
    "RPC1_LAMBD_Li_2019_low-expression",
    "SAV1_MOUSE_Tsuboyama_2023_2YSB",
    "SHOC2_HUMAN_Kwon_2022",
    "SRBS1_HUMAN_Tsuboyama_2023_2O2W",
    "TADBP_HUMAN_Bolognesi_2019",
    "TPMT_HUMAN_Matreyek_2018",
    "UBC9_HUMAN_Weile_2017",
    "UBE4B_HUMAN_Tsuboyama_2023_3L1X",
    "VILI_CHICK_Tsuboyama_2023_1YU5",
]

# 24 assays for INTERNAL model selection (epoch picking, blend weights, ...).
# Construction: disjoint from DEV_SUBSET, drawn from the 206 known-good
# non-binding assays, stratified by Selection Type in dev-40's proportions
# (OrgFitness 9 / Stability 8 / Activity 5 / Expression 2), every sequence
# <= 1022 residues (so the 1022-window logic never kicks in) and <= 3000 total
# mutants. Total 26,878 mutants vs dev-40's 164,821. Measured: 51 s on 4 GPUs,
# all 24 assays load cleanly.
#
# NOTE total (not single) mutant count is what drives runtime: load_dataset()
# reads every row of the assay CSV, multi-mutants included.
#
# ★ sel-24 is NEVER the reported score. It exists so that tuning decisions stop
# being made on dev-40 (which would overfit the headline metric) or on CATH
# validation accuracy (empirically anti-correlated with downstream Spearman).
# A sel-24 run deliberately emits differently-named METRICS_JSON keys so it
# cannot be mistaken for a primary-metric result.
SEL_SUBSET = [
    "AACC1_PSEAI_Dandage_2018",
    "AICDA_HUMAN_Gajula_2014_3cycles",
    "CCDB_ECOLI_Adkar_2012",
    "IF1_ECOLI_Kelsic_2016",
    "LYAM1_HUMAN_Elazar_2016",
    "MAFG_MOUSE_Tsuboyama_2023_1K1V",
    "ODP2_GEOSE_Tsuboyama_2023_1W4G",
    "OPSD_HUMAN_Wan_2019",
    "OTC_HUMAN_Lo_2023",
    "OTU7A_HUMAN_Tsuboyama_2023_2L2D",
    "P53_HUMAN_Kotler_2018",
    "PR40A_HUMAN_Tsuboyama_2023_1UZC",
    "Q837P4_ENTFA_Meier_2023",
    "Q837P5_ENTFA_Meier_2023",
    "RAF1_HUMAN_Zinkus-Boltz_2019",
    "RL40A_YEAST_Roscoe_2013",
    "SOX30_HUMAN_Tsuboyama_2023_7JJK",
    "SUMO1_HUMAN_Weile_2017",
    "TAT_HV1BR_Fernandes_2016",
    "THO1_YEAST_Tsuboyama_2023_2WQG",
    "TNKS2_HUMAN_Tsuboyama_2023_5JRT",
    "TPOR_HUMAN_Bridgford_2020",
    "TRPC_SACS2_Chan_2017",
    "YAIA_ECOLI_Tsuboyama_2023_2KVT",
]

# IDEA-407: internal selection set for multi-window scoring. All non-binding
# assays with seq_len > 1022 that are NOT in dev-40 (10 assays, 26,692 single
# mutants). sel-24 contains no long proteins by construction, so it cannot
# select windowing changes; this set fills that gap. Disjoint from both
# dev-40 and sel-24.
SEL_LONG_SUBSET = [
    "BRCA2_HUMAN_Erwood_2022_HEK293T",
    "CAR11_HUMAN_Meitlis_2020_gof",
    "CAR11_HUMAN_Meitlis_2020_lof",
    "ERBB2_HUMAN_Elazar_2016",
    "NPC1_HUMAN_Erwood_2022_HEK293T",
    "NPC1_HUMAN_Erwood_2022_RPE1",
    "POLG_CXB3N_Mattenberger_2021",
    "SCN5A_HUMAN_Glazer_2019",
    "SPIKE_SARS2_Starr_2020_expression",
    "UBE4B_MOUSE_Starita_2013",
]

# IDEA-501: 80-assay internal selection set (round 5). Built to fix sel-24's
# resolution problem: sel-24's 24-assay mean has a standard error comparable to
# the effects being chased (F8: +0.0004 on sel-24 was -0.0030 on dev-40).
#
# Construction (deterministic, seed 501):
#   - pool: all assays in DMS_substitutions.csv with coarse_selection_type !=
#     Binding, not in dev-40 / sel-24 / sel-long, with per-assay CSV and AF2
#     pdb on disk, DMS_total_number_mutants <= 8000, seq_len <= 1022
#   - stratified draw to dev-40's coarse-type proportions
#     (OrganismalFitness 30 / Stability 26 / Activity 18 / Expression 6 = 80)
#   - random.shuffle(seed=501) within type after sorting by id; reserves kept
#     from the pool leftovers for replacing any assay that fails to evaluate
# Disjoint from dev-40, sel-24 and sel-long (asserted below). Total 253,897
# mutant rows (219,209 single) — a full pass costs more than sel-24 but stays
# in the tens of minutes; use sel-24 for coarse sweeps and sel-80 to arbitrate.
SEL80_SUBSET = [
    "OXDA_RHOTO_Vanella_2023_activity",  # Activity
    "DYR_ECOLI_Thompson_2019",  # OrganismalFitness
    "NUSG_MYCTU_Tsuboyama_2023_2MI6",  # Stability
    "GLPA_HUMAN_Elazar_2016",  # Expression
    "CASP3_HUMAN_Roychowdhury_2020",  # Activity
    "DYR_ECOLI_Nguyen_2023",  # OrganismalFitness
    "CASP7_HUMAN_Roychowdhury_2020",  # Activity
    "CATR_CHLRE_Tsuboyama_2023_2AMI",  # Stability
    "TCRG1_MOUSE_Tsuboyama_2023_1E0L",  # Stability
    "CCDB_ECOLI_Tripathi_2016",  # OrganismalFitness
    "CALM1_HUMAN_Weile_2017",  # OrganismalFitness
    "OXDA_RHOTO_Vanella_2023_expression",  # Expression
    "BLAT_ECOLX_Stiffler_2015",  # OrganismalFitness
    "P53_HUMAN_Giacomelli_2018_WT_Nutlin",  # OrganismalFitness
    "SRC_HUMAN_Ahler_2019",  # Activity
    "PIN1_HUMAN_Tsuboyama_2023_1I6C",  # Stability
    "RCD1_ARATH_Tsuboyama_2023_5OAO",  # Stability
    "PTEN_HUMAN_Mighell_2018",  # Activity
    "PA_I34A1_Wu_2015",  # OrganismalFitness
    "KCNE1_HUMAN_Muhammad_2023_function",  # Activity
    "ESTA_BACSU_Nutschel_2020",  # Stability
    "HEM3_HUMAN_Loggerenberg_2023",  # Activity
    "SRC_HUMAN_Nguyen_2022",  # OrganismalFitness
    "A0A2Z5U3Z0_9INFA_Wu_2014",  # OrganismalFitness
    "PAI1_HUMAN_Huttinger_2021",  # Activity
    "MTH3_HAEAE_RockahShmuel_2015",  # OrganismalFitness
    "RASH_HUMAN_Bandaru_2017",  # Activity
    "TRPC_THEMA_Chan_2017",  # OrganismalFitness
    "REV_HV1H2_Fernandes_2016",  # OrganismalFitness
    "MLAC_ECOLI_MacRae_2023",  # OrganismalFitness
    "RPC1_LAMBD_Li_2019_high-expression",  # Activity
    "AMFR_HUMAN_Tsuboyama_2023_4G3O",  # Stability
    "GDIA_HUMAN_Silverstein_2021",  # OrganismalFitness
    "BLAT_ECOLX_Jacquier_2013",  # OrganismalFitness
    "CBS_HUMAN_Sun_2020",  # OrganismalFitness
    "F7YBW8_MESOW_Ding_2023",  # OrganismalFitness
    "RL40A_YEAST_Mavor_2016",  # OrganismalFitness
    "YNZC_BACSU_Tsuboyama_2023_2JVD",  # Stability
    "RCRO_LAMBD_Tsuboyama_2023_1ORC",  # Stability
    "SDA_BACSU_Tsuboyama_2023_1PV0",  # Stability
    "SRC_HUMAN_Chakraborty_2023_binding-DAS_25uM",  # Activity
    "ENVZ_ECOLI_Ghose_2023",  # Activity
    "R1AB_SARS2_Flynn_2022",  # OrganismalFitness
    "CSN4_MOUSE_Tsuboyama_2023_1UFM",  # Stability
    "P53_HUMAN_Giacomelli_2018_Null_Nutlin",  # OrganismalFitness
    "OBSCN_HUMAN_Tsuboyama_2023_1V1C",  # Stability
    "PSAE_PICP2_Tsuboyama_2023_1PSE",  # Stability
    "KCNJ2_MOUSE_Coyote-Maestas_2022_surface",  # Expression
    "VKOR1_HUMAN_Chiasson_2020_abundance",  # Expression
    "SPG2_STRSG_Tsuboyama_2023_5UBS",  # Stability
    "ADRB2_HUMAN_Jones_2020",  # Activity
    "Q59976_STRSQ_Romero_2015",  # Activity
    "DLG4_HUMAN_Faure_2021",  # OrganismalFitness
    "AMIE_PSEAE_Wrenbeck_2017",  # Activity
    "UBR5_HUMAN_Tsuboyama_2023_1I2T",  # Stability
    "SR43C_ARATH_Tsuboyama_2023_2N88",  # Stability
    "ILF3_HUMAN_Tsuboyama_2023_2L33",  # Stability
    "ISDH_STAAW_Tsuboyama_2023_2LHR",  # Stability
    "HSP82_YEAST_Cote-Hammarlof_2020_growth-H2O2",  # OrganismalFitness
    "SERC_HUMAN_Xie_2023",  # OrganismalFitness
    "CP2C9_HUMAN_Amorosi_2021_abundance",  # Expression
    "RS15_GEOSE_Tsuboyama_2023_1A32",  # Stability
    "CBX4_HUMAN_Tsuboyama_2023_2K28",  # Stability
    "PPM1D_HUMAN_Miller_2022",  # OrganismalFitness
    "RL20_AQUAE_Tsuboyama_2023_1GYZ",  # Stability
    "TPK1_HUMAN_Weile_2017",  # OrganismalFitness
    "SPA_STAAU_Tsuboyama_2023_1LP1",  # Stability
    "SYUA_HUMAN_Newberry_2020",  # OrganismalFitness
    "A0A247D711_LISMN_Stadelmann_2021",  # Activity
    "VKOR1_HUMAN_Chiasson_2020_activity",  # Activity
    "A4GRB6_PSEAI_Chen_2020",  # OrganismalFitness
    "P53_HUMAN_Giacomelli_2018_Null_Etoposide",  # OrganismalFitness
    "BLAT_ECOLX_Firnberg_2014",  # OrganismalFitness
    "RNC_ECOLI_Weeks_2023",  # Activity
    "NUSA_ECOLI_Tsuboyama_2023_1WCL",  # Stability
    "NKX31_HUMAN_Tsuboyama_2023_2L9R",  # Stability
    "DNJA1_HUMAN_Tsuboyama_2023_2LO1",  # Stability
    "PKN1_HUMAN_Tsuboyama_2023_1URF",  # Stability
    "Q53Z42_HUMAN_McShan_2019_expression",  # Expression
    "NRAM_I33A0_Jiang_2016",  # OrganismalFitness
]

assert not (set(DEV_SUBSET) & set(SEL_SUBSET)), \
    "sel-24 must stay disjoint from dev-40 or selection leaks into the score"
assert not (set(DEV_SUBSET) & set(SEL_LONG_SUBSET)), \
    "sel-long must stay disjoint from dev-40 or selection leaks into the score"
assert not (set(SEL_SUBSET) & set(SEL_LONG_SUBSET)), \
    "sel-long must stay disjoint from sel-24 or selection leaks into the score"
assert not (set(DEV_SUBSET) & set(SEL80_SUBSET)), \
    "sel-80 must stay disjoint from dev-40 or selection leaks into the score"
assert not (set(SEL_SUBSET) & set(SEL80_SUBSET)), \
    "sel-80 must stay disjoint from sel-24 or selection leaks into the score"
assert not (set(SEL_LONG_SUBSET) & set(SEL80_SUBSET)), \
    "sel-80 must stay disjoint from sel-long or selection leaks into the score"
assert len(SEL80_SUBSET) == 80, "sel-80 must stay 80 assays (stratified draw)"


def _load_selection_types():
    """Load {DMS_id: Selection_Type} from the paper's ProteinGym summary CSV."""
    sel = {}
    csv_path = os.path.join(S3F_REPO,
                            "asset/DMS_substitutions_Spearman_DMS_level.csv")
    if not os.path.isfile(csv_path):
        return sel
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            sel[row["DMS ID"]] = row.get("Selection Type", "")
    return sel


def _filter_interface(assay_ids, selection_types):
    """Drop binding/interface assays (protein-protein interaction data)."""
    filtered = [a for a in assay_ids
                if selection_types.get(a, "") != "Binding"]
    dropped = [a for a in assay_ids if a not in filtered]
    if dropped:
        print(f"[s2f_eval] Dropped {len(dropped)} binding/interface assays: "
              f"{dropped}", flush=True)
    return filtered


def make_config(assay_ids, output_path, ckpt_path, datadir, structdir):
    """Create a concrete S2F eval YAML with the given assay id_list."""
    with open(BASE_CONFIG, "r") as f:
        raw = f.read()

    from jinja2 import Template
    template = Template(raw)
    rendered = template.render(
        datadir=datadir,
        structdir=structdir,
        surfdir="",  # S2F doesn't use surfaces
        ckpt=ckpt_path,
    )
    base = yaml.safe_load(rendered)

    base["id_list"] = assay_ids
    base.pop("surface_path", None)
    # Evaluation outputs go under S3F/results/ (gitignored), not ~/scratch
    base["output_dir"] = OUTPUT_ROOT

    with open(output_path, "w") as f:
        yaml.dump(base, f)
    return output_path


def run_eval_on_gpu(config_path, gpu_id, job_id=None):
    """Run S2F evaluate.py on one GPU; returns its results.csv path.

    job_id differentiates working dirs when multiple GPUs run in parallel
    (evaluate.py uses SLURM_JOB_ID as a directory component). When None,
    the inherited SLURM_JOB_ID (set by the multi-GPU parent) is used.
    """
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    if job_id is not None:
        env["SLURM_JOB_ID"] = str(job_id)

    cmd = [
        sys.executable,
        os.path.join(S3F_REPO, "script/evaluate.py"),
        "-c", config_path,
    ]
    print(f"[s2f_eval] GPU {gpu_id} (job {job_id}): {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, cwd=S3F_REPO, env=env,
                          capture_output=True, text=True, timeout=86400)
    if result.returncode != 0:
        print(f"[s2f_eval] GPU {gpu_id} ERROR: {result.stderr[-1500:]}", flush=True)
        return None

    # results.csv lives under .../FusionNetwork/<job_id>/<timestamp>/results.csv
    effective_job_id = job_id if job_id is not None else env.get("SLURM_JOB_ID")
    job_dir = os.path.join(OUTPUT_ROOT, "ResidueTypePrediction", "DMSAssay",
                           "FusionNetwork", str(effective_job_id))
    if os.path.isdir(job_dir):
        subdirs = sorted(
            (d for d in os.listdir(job_dir) if d.startswith("20")),
            reverse=True,
        )
        for subdir in subdirs:
            csv_path = os.path.join(job_dir, subdir, "results.csv")
            if os.path.isfile(csv_path):
                return csv_path

    # A tagged run must never pick up a result from a different job.
    if effective_job_id is not None:
        return None
    # Fallback: newest results.csv anywhere under the output root
    best_match, best_mtime = None, 0
    for root, dirs, files in os.walk(OUTPUT_ROOT):
        if "results.csv" in files:
            p = os.path.join(root, "results.csv")
            mtime = os.path.getmtime(p)
            if mtime > best_mtime:
                best_mtime, best_match = mtime, p
    return best_match


def _append_summary(avg_spearman, count, method, assayset):
    """Append one row to the iteration-tracking summary CSV."""
    try:
        os.makedirs(os.path.dirname(SUMMARY_CSV), exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        row = [ts, method, "", assayset,
               f"{avg_spearman:.4f}", str(count), ""]
        if not os.path.isfile(SUMMARY_CSV):
            with open(SUMMARY_CSV, "w", newline="") as f:
                csv.writer(f).writerow(
                    ["timestamp", "method", "checkpoint", "assayset",
                     "avg_spearman", "num_assays", "notes"])
        with open(SUMMARY_CSV, "a", newline="") as f:
            csv.writer(f).writerow(row)
    except OSError as e:
        print(f"[s2f_eval] WARNING: could not append summary: {e}", flush=True)


def parse_results(results_csv):
    """Parse a results.csv; returns (avg_spearman, metrics_dict, count)."""
    spearman_values = []
    metrics = {}
    with open(results_csv, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                sp = float(row["spearmanr"])
                spearman_values.append(sp)
                metrics[row["DMS_id"]] = {
                    "spearmanr": sp,
                    "pearsonr": float(row.get("pearsonr", 0)),
                    "mae": float(row.get("mae", 0)),
                    "rmse": float(row.get("rmse", 0)),
                }
            except (ValueError, KeyError):
                continue

    avg = sum(spearman_values) / len(spearman_values) if spearman_values else 0.0
    return avg, metrics, len(spearman_values)


def main():
    parser = argparse.ArgumentParser(description="S2F eval wrapper for autosota")
    parser.add_argument("--ckpt", default=DEFAULT_CKPT, help="S2F checkpoint path")
    parser.add_argument("--subset", default="dev", choices=["dev", "sel", "sel-long", "sel80", "full"],
                       help="Assay subset: dev (40, the reported metric), "
                            "sel (24 held-out, internal selection only), "
                            "sel-long (10 long-protein held-out, internal "
                            "selection only), sel80 (80 held-out, internal "
                            "selection only), or full (all non-binding)")
    parser.add_argument("--gpus", default="0,1,2,3",
                       help="Comma-separated GPU IDs to shard across")
    parser.add_argument("--method", default=None,
                       help="Method name recorded in the summary CSV "
                            "(default: checkpoint basename)")
    parser.add_argument("--assay-ids", help="Comma-separated assay IDs (overrides --subset)")
    parser.add_argument("--assay-file", help="File with one assay ID per line")
    args = parser.parse_args()

    # Determine assay list
    if args.assay_ids:
        assay_ids = [a.strip() for a in args.assay_ids.split(",") if a.strip()]
    elif args.assay_file:
        with open(args.assay_file) as f:
            assay_ids = [l.strip() for l in f if l.strip() and not l.startswith("#")]
    elif args.subset == "dev":
        assay_ids = DEV_SUBSET
    elif args.subset == "sel":
        assay_ids = SEL_SUBSET
    elif args.subset == "sel-long":
        assay_ids = SEL_LONG_SUBSET
    elif args.subset == "sel80":
        assay_ids = SEL80_SUBSET
    else:
        with open(os.path.join(S3F_REPO, "config/evaluate/s2f.yaml")) as f:
            cfg = yaml.safe_load(f)
        assay_ids = cfg["id_list"]

    # Drop binding/interface assays (user: no interface data needed)
    selection_types = _load_selection_types()
    if selection_types and os.environ.get("S2F_INCLUDE_BINDING", "0") != "1":
        assay_ids = _filter_interface(assay_ids, selection_types)

    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    ckpt_path = os.path.abspath(args.ckpt)
    datadir = os.path.abspath(DATADIR)
    structdir = os.path.abspath(STRUCTDIR)

    # Single-GPU mode: run directly in this process (also used by shard children)
    if len(gpus) == 1:
        gpu = gpus[0]
        fd, cfg_path = tempfile.mkstemp(suffix=".yaml")
        os.close(fd)
        make_config(assay_ids, cfg_path, ckpt_path, datadir, structdir)
        try:
            results_csv = run_eval_on_gpu(cfg_path, gpu)
            if results_csv is None:
                print(f"[s2f_eval] GPU {gpu}: no results.csv produced", flush=True)
                sys.exit(1)
            avg, metrics, count = parse_results(results_csv)
            if count != len(assay_ids) or set(metrics) != set(assay_ids):
                raise RuntimeError(
                    f"Incomplete assay results: expected {len(assay_ids)}, got {count}; "
                    f"missing={sorted(set(assay_ids) - set(metrics))}")
            for assay_id, m in sorted(metrics.items()):
                print(f"  {assay_id}: spearmanr={m['spearmanr']:.4f} "
                      f"pearsonr={m['pearsonr']:.4f} mae={m['mae']:.4f} "
                      f"rmse={m['rmse']:.4f}", flush=True)
            print(f"[s2f_eval] GPU {gpu}: {count} assays, "
                  f"Avg Spearman {avg:.4f}", flush=True)
        finally:
            try:
                os.unlink(cfg_path)
            except OSError:
                pass
        return

    # Shard assays across GPUs (balanced: first shards get the extra)
    n = len(assay_ids)
    base, extra = divmod(n, len(gpus))
    shards = []
    idx = 0
    for i in range(len(gpus)):
        size = base + (1 if i < extra else 0)
        shards.append(assay_ids[idx:idx + size])
        idx += size

    print(f"[s2f_eval] {n} assays over {len(gpus)} GPUs: "
          f"{[len(s) for s in shards]} per shard", flush=True)

    # Launch one subprocess per GPU (each with its own temp config)
    procs = []
    configs = []
    for i, (gpu, shard) in enumerate(zip(gpus, shards)):
        if not shard:
            procs.append(None)
            configs.append(None)
            continue
        fd, cfg_path = tempfile.mkstemp(suffix=".yaml")
        os.close(fd)
        make_config(shard, cfg_path, ckpt_path, datadir, structdir)
        configs.append(cfg_path)
        procs.append(subprocess.Popen(
            [sys.executable, __file__, "--subset", args.subset,
             "--gpus", gpu, "--ckpt", ckpt_path,
             "--assay-ids", ",".join(shard)],
            cwd=S3F_REPO,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": gpu,
                 "SLURM_JOB_ID": str(i)},
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True,
        ))

    # Collect output and results from each shard
    all_metrics = {}
    for i, proc in enumerate(procs):
        if proc is None:
            continue
        out, _ = proc.communicate(timeout=86400)
        # Find this shard's results.csv (newest under its job_id dir)
        job_dir = os.path.join(OUTPUT_ROOT, "ResidueTypePrediction", "DMSAssay",
                               "FusionNetwork", str(i))
        results_csv = None
        if os.path.isdir(job_dir):
            subdirs = sorted((d for d in os.listdir(job_dir) if d.startswith("20")),
                             reverse=True)
            for subdir in subdirs:
                p = os.path.join(job_dir, subdir, "results.csv")
                if os.path.isfile(p):
                    results_csv = p
                    break
        if results_csv:
            _, metrics, _ = parse_results(results_csv)
            all_metrics.update(metrics)
        print(out, flush=True)

    for cfg_path in configs:
        if cfg_path:
            try:
                os.unlink(cfg_path)
            except OSError:
                pass

    # Aggregate
    spearman_values = [m["spearmanr"] for m in all_metrics.values()]
    avg_spearman = (sum(spearman_values) / len(spearman_values)
                    if spearman_values else 0.0)
    count = len(spearman_values)

    print(f"\n[s2f_eval] Done. {count} assays evaluated.", flush=True)
    print(f"[s2f_eval] Avg Spearman: {avg_spearman:.4f}", flush=True)
    for assay_id, m in sorted(all_metrics.items()):
        print(f"  {assay_id}: spearmanr={m['spearmanr']:.4f} "
              f"pearsonr={m['pearsonr']:.4f} mae={m['mae']:.4f} "
              f"rmse={m['rmse']:.4f}", flush=True)

    import json
    if args.subset == "sel":
        # Deliberately different key names: a selection run must never be
        # parsed as the primary metric by the optimizer harness.
        summary = {
            "ProteinGym_Sel24_Avg_Spearman": round(avg_spearman, 4),
            "ProteinGym_Sel24_Num_Assays": count,
        }
    elif args.subset == "sel-long":
        # Same convention for the long-protein selection set (IDEA-407).
        summary = {
            "ProteinGym_SelLong_Avg_Spearman": round(avg_spearman, 4),
            "ProteinGym_SelLong_Num_Assays": count,
        }
    elif args.subset == "sel80":
        # Same convention for the 80-assay selection set (IDEA-501).
        summary = {
            "ProteinGym_Sel80_Avg_Spearman": round(avg_spearman, 4),
            "ProteinGym_Sel80_Num_Assays": count,
        }
    else:
        summary = {
            "ProteinGym_Avg_Spearman": round(avg_spearman, 4),
            "ProteinGym_Num_Assays": count,
        }
    print(f"\nMETRICS_JSON: {json.dumps(summary)}", flush=True)

    if count < n:
        print(f"[s2f_eval] WARNING: only {count}/{n} assays evaluated", flush=True)

    # Append to iteration-tracking summary CSV
    method = args.method or os.path.basename(args.ckpt)
    assayset = {"dev": "dev-40", "sel": "sel-24", "sel-long": "sel-long",
                "sel80": "sel-80"}.get(args.subset, "full")
    _append_summary(avg_spearman, count, method, assayset)


if __name__ == "__main__":
    main()
