import os
import pandas as pd
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import RDKFingerprint
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import pickle

def get_fingerprints(smiles_list):
    fps = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            fps.append(RDKFingerprint(mol))
    return fps

def max_tanimoto_similarity(fp, fps_list):
    if len(fps_list) == 0:
        return 0.0
    sims = [DataStructs.TanimotoSimilarity(fp, ref_fp) for ref_fp in fps_list]
    return max(sims)


def load_KBTS_smiles(base_dir, num_targets=100):

    init_smiles_list_all = []
    bo_smiles_list_all = []

    for i in range(num_targets):
        target_dir = os.path.join(base_dir, str(i))
        init_path = os.path.join(target_dir, "init_score.csv")
        if not os.path.exists(init_path):
            init_smiles_list_all.append([])
            bo_smiles_list_all.append([])
            continue

        df_init = pd.read_csv(init_path)
        init_smiles = df_init["smile"].tolist()

        bo_smiles = []
        for seed in range(1, 2):
            bo_path = os.path.join(target_dir, f"{seed}_result.csv")
            if not os.path.exists(bo_path):
                continue
            df_bo = pd.read_csv(bo_path)
            bo_smiles.extend(df_bo["SMILES"].tolist())

        init_smiles_list_all.append(init_smiles)
        bo_smiles_list_all.append(bo_smiles)

    return init_smiles_list_all, bo_smiles_list_all


def load_ELILLM_smiles(base_dir, num_targets=100, seed_range=(1, 2)):

    init_smiles_list_all = []
    bo_smiles_list_all = []

    for i in range(num_targets):
        target_dir = os.path.join(base_dir, str(i))
        init_path = os.path.join(target_dir, "init_score.csv")


        if not os.path.exists(init_path):
            init_smiles_list_all.append([])
            bo_smiles_list_all.append([])
            continue

        df_init = pd.read_csv(init_path)
        init_smiles = df_init["smile"].tolist()
        init_smiles_list_all.append(init_smiles)

        bo_smiles = []
        for seed in range(*seed_range):
            bo_path = os.path.join(target_dir, f"{seed}_result.csv")
            if not os.path.exists(bo_path):
                continue
            df_bo = pd.read_csv(bo_path)
            bo_smiles.extend(df_bo["Molecule"].tolist())

        bo_smiles_list_all.append(bo_smiles)

    return init_smiles_list_all, bo_smiles_list_all


def load_LMLF_smiles(base_dir, num_targets=100, seed_range=(1, 2)):

    init_smiles_list_all = []
    lmlf_smiles_list_all = []

    for i in range(num_targets):
        target_dir = os.path.join(base_dir, str(i))
        init_path = os.path.join(target_dir, "init_score.csv")

        if not os.path.exists(init_path):
            init_smiles_list_all.append([])
            lmlf_smiles_list_all.append([])
            continue

        df_init = pd.read_csv(init_path)
        init_smiles = df_init["smile"].tolist()
        init_smiles_list_all.append(init_smiles)

        lmlf_smiles = []
        for seed in range(*seed_range):
            lmlf_path = os.path.join(target_dir, f"{seed}_result_ori.csv")
            if not os.path.exists(lmlf_path):
                continue
            df_lmlf = pd.read_csv(lmlf_path)
            lmlf_smiles.extend(df_lmlf["Molecule"].tolist())

        lmlf_smiles_list_all.append(lmlf_smiles)

    return init_smiles_list_all, lmlf_smiles_list_all

def compute_similarity_curve(init_smiles_list_all, bo_smiles_list_all, step=10, max_mols=100):

    bucket_scores = {k: [] for k in range(step, max_mols+1, step)}

    for init_smiles, bo_smiles in zip(init_smiles_list_all, bo_smiles_list_all):
        if len(init_smiles) == 0 or len(bo_smiles) == 0:
            continue

        init_fps = get_fingerprints(init_smiles)
        bo_fps = get_fingerprints(bo_smiles)
        sim_scores = [max_tanimoto_similarity(fp, init_fps) for fp in bo_fps]

        max_k = min(len(sim_scores), max_mols)
        for k in range(step, max_k+1, step):
            bucket = sim_scores[k-step:k]
            bucket_scores[k].append(np.mean(bucket))


    x, y = [], []
    for k in sorted(bucket_scores.keys()):
        if len(bucket_scores[k]) == 0:
            continue
        x.append(k)
        y.append(np.mean(bucket_scores[k]))
    return x, y


def plot_multiple_curves(curves_dict, save_path=None):
    plt.figure(figsize=(8, 6))
    plt.rcParams['font.weight'] = 'bold'
    plt.rcParams['axes.labelweight'] = 'bold'
    ax = plt.gca()
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:6.2f}"))
    plt.subplots_adjust(left=0.18, right=0.95, top=0.95, bottom=0.15)

    markers = ['o', 's', '^', 'D', 'v', '*', 'p', 'H', 'X', '+']  # 可根据曲线数扩展

    for idx, (name, (x, y)) in enumerate(curves_dict.items()):
        plt.plot(
            x, y,
            marker=markers[idx % len(markers)],  # 循环使用 marker
            linewidth=2.5,  # 线更粗
            markersize=8,   # 点更大
            label=name
        )

    plt.xlabel("Iteration (number of generated molecules)", fontsize=18, fontweight='bold')
    plt.ylabel("Average max Tanimoto similarity", fontsize=18, fontweight='bold')
    plt.legend(
        fontsize=16,
        frameon=True,
        framealpha=0.9,
        edgecolor="black",
        handlelength=2.5,
        handletextpad=0.8,
        loc='best'
    )
    plt.grid(True)

    if save_path:
        plt.savefig(save_path,
                    format='pdf',
                    dpi=300,
                    bbox_inches='tight',
                    pad_inches=0
                    )

    plt.show()


if __name__ == "__main__":

    curve_cache_path = "similarity_curves.pkl"


    if os.path.exists(curve_cache_path):
        print(f"[INFO] Load cached curves from {curve_cache_path}")
        with open(curve_cache_path, "rb") as f:
            curves = pickle.load(f)
    else:
        print("[INFO] No cached curves found, computing...")

        curves = {}


        kbts_base_dir = "../results/rand"
        init_kbts, bo_kbts = load_KBTS_smiles(
            kbts_base_dir, num_targets=100
        )
        curves["K-BTS-rand"] = compute_similarity_curve(
            init_kbts, bo_kbts
        )


        elillm_base_dir = "../baselines/ELILLM-rand"
        init_elillm, bo_elillm = load_ELILLM_smiles(
            elillm_base_dir, num_targets=100
        )
        curves["ELILLM-rand"] = compute_similarity_curve(
            init_elillm, bo_elillm
        )


        lmlf_base_dir = "../baselines/LMLF-rand"
        init_lmlf, bo_lmlf = load_LMLF_smiles(
            lmlf_base_dir, num_targets=100
        )
        curves["LMLF-rand"] = compute_similarity_curve(
            init_lmlf, bo_lmlf
        )


        with open(curve_cache_path, "wb") as f:
            pickle.dump(curves, f)

        print(f"[INFO] Curves saved to {curve_cache_path}")


    plot_multiple_curves(
        curves,
        save_path="similarity_curve.pdf"
    )
