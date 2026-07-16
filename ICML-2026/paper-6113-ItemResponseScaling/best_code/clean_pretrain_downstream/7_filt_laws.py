import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import pickle
import os
import multiprocessing as mp
import sys
from scipy.special import expit
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent
BASE_DIR = PROJECT_ROOT / "data"
sys.path.append(str(PROJECT_ROOT.parent))
from utils import (
    MODEL2PARA,
    recursive_defaultdict,
    fn_step1_classic,
    fit_step1_classic,
    fn_step2_classic,
    fit_step2_classic,
    fn_step1_irt,
    fit_step1_irt,
)

# function forms
# - classic:
#   - step1: 
#     - correct_bpb_sub VS FLOP: f1, fit_step1_classic
#   - step2:
#     - acc_per_char_sub VS correct_bpb_sub: f21, fit_step2_classic
#     - p_correct_choice_sub VS correct_bpb_sub: f22, fit_step2_classic
# - IRT:
#   - step1:
#     - theta_binary_1pl VS FLOP: g11, fit_step1_irt
#     - theta_beta_1pl VS FLOP: g12, fit_step1_irt
#     - theta_binary_2pl VS FLOP: g13, fit_step1_irt
#     - theta_beta_2pl VS FLOP: g14, fit_step1_irt

# output_dict = {
#     bench (str): {
#         model_data_mix (str): {
#             max_model_size (int): {
#                 "max_size_flop": float,
#                 "classic": {
#                     "data": {
#                         "step1": [(FLOP (float), bpb (float)), ...],
#                         "step2": [(bpb (float), acc (float), prob (float)), ...],
#                     },
#                     "paras": {
#                         "step1": f1_paras (list of float, len = 3),
#                         "step2_acc": f21_paras (list of float, len = 4),
#                         "step2_prob": f22_paras (list of float, len = 4),
#                     },
#                     "pred_1B": {
#                         "acc": float,
#                         "prob": float,
#                     }
#                 },
#                 "irt": {
#                     "data": {
#                         "step1": [(FLOP (float), theta_binary_1pl (float), theta_beta_1pl (float), theta_binary_2pl (float), theta_beta_2pl (float)), ...],
#                     },
#                     "paras": {
#                         "step1_binary_1pl": g11_paras (list of float, len = 2),
#                         "step1_beta_1pl": g12_paras (list of float, len = 2),
#                         "step1_binary_2pl": g13_paras (list of float, len = 2),
#                         "step1_beta_2pl": g14_paras (list of float, len = 2),
#                     },
#                     "pred_1B": {
#                         "binary_1pl": float,
#                         "beta_1pl": float,
#                         "binary_2pl": float,
#                         "beta_2pl": float,
#                     }
#                 }
#             }
#         }
#     }
# }

def process_fit(args):
    mix, max_size, bench, df_size, max_flop, max_size_flop, difficulty_dict = args
    binary_difficulty_1pl = np.asarray(difficulty_dict[bench]["binary_difficulty_1pl"], dtype=np.float32)
    beta_difficulty_1pl = np.asarray(difficulty_dict[bench]["beta_difficulty_1pl"], dtype=np.float32)
    binary_difficulty_2pl = np.asarray(difficulty_dict[bench]["binary_difficulty_2pl"], dtype=np.float32)
    beta_difficulty_2pl = np.asarray(difficulty_dict[bench]["beta_difficulty_2pl"], dtype=np.float32)
    binary_discrimination_2pl = np.asarray(difficulty_dict[bench]["binary_discrimination_2pl"], dtype=np.float32)
    beta_discrimination_2pl = np.asarray(difficulty_dict[bench]["beta_discrimination_2pl"], dtype=np.float32)

    # fit classic step 1
    data_classic_step1 = df_size.loc[
        df_size["FLOP"].notna() & df_size[f"correct_bpb_sub_{bench}"].notna(),
        ["FLOP", f"correct_bpb_sub_{bench}"],
    ]
    f1_paras = fit_step1_classic(
        flops=data_classic_step1["FLOP"].tolist(),
        bpbs=data_classic_step1[f"correct_bpb_sub_{bench}"].tolist(),
    )

    # fit irt step 1
    data_irt_step1 = df_size.loc[
        df_size["FLOP"].notna(),
        [
            "FLOP",
            f"ability_{bench}_binary_1pl",
            f"ability_{bench}_beta_1pl",
            f"ability_{bench}_binary_2pl",
            f"ability_{bench}_beta_2pl",
        ],
    ]
    g11_paras = fit_step1_irt(
        flops=data_irt_step1["FLOP"].tolist(),
        thetas=data_irt_step1[f"ability_{bench}_binary_1pl"].tolist(),
    )
    g12_paras = fit_step1_irt(
        flops=data_irt_step1["FLOP"].tolist(),
        thetas=data_irt_step1[f"ability_{bench}_beta_1pl"].tolist(),
    )
    g13_paras = fit_step1_irt(
        flops=data_irt_step1["FLOP"].tolist(),
        thetas=data_irt_step1[f"ability_{bench}_binary_2pl"].tolist(),
    )
    g14_paras = fit_step1_irt(
        flops=data_irt_step1["FLOP"].tolist(),
        thetas=data_irt_step1[f"ability_{bench}_beta_2pl"].tolist(),
    )
    
    # fit classic step 2
    data_classic_step2 = df_size.loc[
        df_size[f"correct_bpb_sub_{bench}"].notna(),
        [f"correct_bpb_sub_{bench}", f"acc_sub_{bench}", f"p_correct_choice_sub_{bench}"],
    ]
    f21_paras = fit_step2_classic(
        bpbs=data_classic_step2[f"correct_bpb_sub_{bench}"].tolist(),
        metrics=data_classic_step2[f"acc_sub_{bench}"].tolist(),
    )
    f22_paras = fit_step2_classic(
        bpbs=data_classic_step2[f"correct_bpb_sub_{bench}"].tolist(),
        metrics=data_classic_step2[f"p_correct_choice_sub_{bench}"].tolist(),
    )

    # extrapolate classic
    pred_acc_classic = fn_step2_classic(
        bpb=fn_step1_classic(flop=max_flop, paras=f1_paras),
        paras=f21_paras,
    )
    pred_prob_classic = fn_step2_classic(
        bpb=fn_step1_classic(flop=max_flop, paras=f1_paras),
        paras=f22_paras,
    )
    
    # extrapolate irt
    pred_theta_binary_1pl = fn_step1_irt(flop=max_flop, paras=g11_paras)
    pred_theta_beta_1pl = fn_step1_irt(flop=max_flop, paras=g12_paras)
    pred_theta_binary_2pl = fn_step1_irt(flop=max_flop, paras=g13_paras)
    pred_theta_beta_2pl = fn_step1_irt(flop=max_flop, paras=g14_paras)
    pred_acc_irt_binary_1pl = expit(pred_theta_binary_1pl + binary_difficulty_1pl).mean()
    pred_prob_irt_beta_1pl = expit(pred_theta_beta_1pl + beta_difficulty_1pl).mean()
    pred_acc_irt_binary_2pl = expit(binary_discrimination_2pl * (pred_theta_binary_2pl - binary_difficulty_2pl)).mean()
    pred_prob_irt_beta_2pl = expit(beta_discrimination_2pl * (pred_theta_beta_2pl - beta_difficulty_2pl)).mean()

    return {
        "bench": bench,
        "mix": mix,
        "max_size": int(max_size),
        "max_size_flop": float(max_size_flop),
        "classic_step1_data": list(data_classic_step1.itertuples(index=False, name=None)),
        "classic_step2_data": list(data_classic_step2.itertuples(index=False, name=None)),
        "irt_step1_data": list(data_irt_step1.itertuples(index=False, name=None)),
        "f1_paras": f1_paras,
        "f21_paras": f21_paras,
        "f22_paras": f22_paras,
        "g11_paras": g11_paras,
        "g12_paras": g12_paras,
        "g13_paras": g13_paras,
        "g14_paras": g14_paras,
        "pred_acc_classic": float(pred_acc_classic),
        "pred_prob_classic": float(pred_prob_classic),
        "pred_acc_irt_binary_1pl": float(pred_acc_irt_binary_1pl),
        "pred_prob_irt_beta_1pl": float(pred_prob_irt_beta_1pl),
        "pred_acc_irt_binary_2pl": float(pred_acc_irt_binary_2pl),
        "pred_prob_irt_beta_2pl": float(pred_prob_irt_beta_2pl),
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=BASE_DIR)
    args = parser.parse_args()

    data_root = args.data_root
    data_root.mkdir(parents=True, exist_ok=True)
    input_path = data_root / "6_long.parquet"
    diff_input_path = data_root / "6_difficulty.pkl"
    output_path = data_root / "7_filt_laws.pkl"

    output_dict = recursive_defaultdict()
    
    with open(diff_input_path, "rb") as f:
        difficulty_dict = pickle.load(f)
    
    df_input = pd.read_parquet(input_path)
    df_input["model_size"] = df_input["model_size"].map(MODEL2PARA).astype(int)
    unique_mixes = sorted(df_input["model_data_mix"].unique())
    unique_model_sizes = sorted(df_input["model_size"].unique())
    binary_theta_cols = [c for c in df_input.columns if c.startswith("ability_") and c.endswith("_binary_1pl")]
    unique_benches = sorted({c[len("ability_") : -len("_binary_1pl")] for c in binary_theta_cols})
    max_flop = df_input["FLOP"].max() # 1B, final step
    
    tasks = []
    for mix in unique_mixes:
        df_mix = df_input[df_input["model_data_mix"] == mix]
        for max_size in unique_model_sizes[1:-1]:  # removing first and last
            df_size = df_mix[df_mix["model_size"] <= max_size]
            max_size_flop = df_size["FLOP"].max()
            for bench in unique_benches:
                tasks.append(
                    (mix, max_size, bench, df_size, max_flop, max_size_flop, difficulty_dict)
                )

    n_cpus = int(os.cpu_count() * 0.8)
    with mp.Pool(processes=n_cpus) as pool:
        results = list(tqdm(pool.imap(process_fit, tasks), total=len(tasks), desc="fitting"))

    for res in results:
        bench = res["bench"]
        mix = res["mix"]
        max_size = res["max_size"]
        output_dict[bench][mix][max_size]["max_size_flop"] = res["max_size_flop"]
        output_dict[bench][mix][max_size]["classic"]["data"]["step1"] = res["classic_step1_data"]
        output_dict[bench][mix][max_size]["classic"]["data"]["step2"] = res["classic_step2_data"]
        output_dict[bench][mix][max_size]["irt"]["data"]["step1"] = res["irt_step1_data"]
        output_dict[bench][mix][max_size]["classic"]["paras"]["step1"] = res["f1_paras"]
        output_dict[bench][mix][max_size]["classic"]["paras"]["step2_acc"] = res["f21_paras"]
        output_dict[bench][mix][max_size]["classic"]["paras"]["step2_prob"] = res["f22_paras"]
        output_dict[bench][mix][max_size]["irt"]["paras"]["step1_binary_1pl"] = res["g11_paras"]
        output_dict[bench][mix][max_size]["irt"]["paras"]["step1_beta_1pl"] = res["g12_paras"]
        output_dict[bench][mix][max_size]["irt"]["paras"]["step1_binary_2pl"] = res["g13_paras"]
        output_dict[bench][mix][max_size]["irt"]["paras"]["step1_beta_2pl"] = res["g14_paras"]
        output_dict[bench][mix][max_size]["classic"]["pred_1B"]["acc"] = res["pred_acc_classic"]
        output_dict[bench][mix][max_size]["classic"]["pred_1B"]["prob"] = res["pred_prob_classic"]
        output_dict[bench][mix][max_size]["irt"]["pred_1B"]["binary_1pl"] = res["pred_acc_irt_binary_1pl"]
        output_dict[bench][mix][max_size]["irt"]["pred_1B"]["beta_1pl"] = res["pred_prob_irt_beta_1pl"]
        output_dict[bench][mix][max_size]["irt"]["pred_1B"]["binary_2pl"] = res["pred_acc_irt_binary_2pl"]
        output_dict[bench][mix][max_size]["irt"]["pred_1B"]["beta_2pl"] = res["pred_prob_irt_beta_2pl"]
    
    with open(output_path, "wb") as f:
        pickle.dump(output_dict, f)
