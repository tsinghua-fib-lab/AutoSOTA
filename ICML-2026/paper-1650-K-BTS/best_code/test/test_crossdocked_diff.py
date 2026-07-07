import pandas as pd
import numpy as np
import os
from utils.rdkit_tools import internal_diversity
from rdkit import RDLogger
import pickle
logger = RDLogger.logger()
logger.setLevel(RDLogger.CRITICAL)

# modified from ELILLM:test_crossdocked
if __name__ == "__main__":

    bo_diversity_list = []
    init_diversity_list = []
    init_qed_list = []
    init_sa_list = []
    init_logp_list = []
    init_mw_list = []

    init_qed_median_list = []
    init_sa_median_list = []
    init_logp_median_list = []
    init_mw_median_list = []

    bo_qed_list = []
    bo_sa_list = []
    bo_logp_list = []
    bo_mw_list = []

    bo_qed_median_list = []
    bo_sa_median_list = []
    bo_logp_median_list = []
    bo_mw_median_list = []
    dir_path = "results/diff"
    # dir_path = "results/diff_wo_knowledge"
    # dir_path = "results/diff_wo_lower"
    # dir_path = "results/diff_wo_upper"
    # dir_path = "results/diff_wo_warmstart"
    init_smiles_list = []
    bo_smiles_list = []
    topk_result = {}
    for i in range(10):
        name_protein = os.path.join(dir_path, str(i))
        result_path = f"../{name_protein}/init_scores.csv"
        #result_path = f"../{name_protein}/init_score.csv"
        result_df = pd.read_csv(result_path)
        init_docking_scores = result_df["rdkit3d_docking_score"].tolist()
        # init_docking_scores = result_df["docking_scores"].tolist()
        init_qed = result_df["qed_score"].tolist()
        init_sa = result_df["sascore"].tolist()
        init_logp = result_df["logp"].tolist()
        init_mw = result_df["mw"].tolist()

        # init_qed = result_df["QED"].tolist()
        # init_sa = result_df["SAS"].tolist()
        # init_logp = result_df["logp"].tolist()
        # init_mw = result_df["mw"].tolist()

        init_qed_list.append(np.mean(init_qed))
        init_sa_list.append(np.mean(init_sa))
        init_logp_list.append(np.mean(init_logp))
        init_mw_list.append(np.mean(init_mw))
        
        init_qed_median_list.append(np.median(init_qed))
        init_sa_median_list.append(np.median(init_sa))
        init_logp_median_list.append(np.median(init_logp))
        init_mw_median_list.append(np.median(init_mw))

        init_smiles = result_df["smiles"].tolist()
        # init_smiles = result_df["smile"].tolist()
        init_smiles_list.append(init_smiles)

        init_diversity = internal_diversity(init_smiles)
        init_diversity_list.append(init_diversity)
        init_docking_scores = np.sort(init_docking_scores)
        bo_dockings = []
        bo_smiles = []
        ori_dockings = []
        bo_qed = []
        bo_sa = []
        bo_logp = []
        bo_mw = []
        for seed in range(1, 2):
            bo_name = f"../{name_protein}/{seed}_result.csv"
            df = pd.read_csv(bo_name)
            bo_dockings.extend(df["docking_score"].tolist())
            bo_smiles.extend(df["SMILES"].tolist())
            bo_qed.extend(df["qed"].tolist())
            bo_sa.extend(df["sa"].tolist())
            bo_logp.extend(df["logp"].tolist())
            bo_mw.extend(df["mw"].tolist())


        if len(bo_dockings) != 100:
            print(f"{i}:{len(bo_dockings)}")
        bo_smiles_list.append(bo_smiles)
        bo_dockings = np.sort(bo_dockings).tolist()
        bo_qed_list.append(np.mean(bo_qed))
        bo_sa_list.append(np.mean(bo_sa))
        bo_logp_list.append(np.mean(bo_logp))
        bo_mw_list.append(np.mean(bo_mw))
        
        bo_qed_median_list.append(np.median(bo_qed))
        bo_sa_median_list.append(np.median(bo_sa))
        bo_logp_median_list.append(np.median(bo_logp))
        bo_mw_median_list.append(np.median(bo_mw))

        bo_diversity = internal_diversity(bo_smiles)
        bo_diversity_list.append(bo_diversity)
        for k in [1, 5, 10, 20]:
            mean_init = np.mean(init_docking_scores[:k])
            median_init = np.median(init_docking_scores[:k])
            mean_bo = np.mean(bo_dockings[:k])
            median_bo = np.median(bo_dockings[:k])
            if k==1 and mean_init > mean_bo:
                print(f"{i}:{mean_init - mean_bo}")
            if f"init_mean_list_{k}" in topk_result:
                topk_result[f"init_mean_list_{k}"].append(mean_init)
                topk_result[f"init_median_list_{k}"].append(median_init)
            else:
                topk_result[f"init_mean_list_{k}"] = [mean_init]
                topk_result[f"init_median_list_{k}"] = [median_init]
            if f"bo_mean_list_{k}" in topk_result:
                topk_result[f"bo_mean_list_{k}"].append(mean_bo)
                topk_result[f"bo_median_list_{k}"].append(median_bo)
            else:
                topk_result[f"bo_mean_list_{k}"] = [mean_bo]
                topk_result[f"bo_median_list_{k}"] = [median_bo]
        pass


        # df = pd.DataFrame(topk)
        # df.to_csv(f"../{name_protein}/top{k}.csv", index=False)

    result = {}
    for k in [1,5,10,20]:
        result[f"init_mean_list_{k}"] = topk_result[f"init_mean_list_{k}"]
        result[f"init_median_list_{k}"] = topk_result[f"init_median_list_{k}"]
        result[f"bo_mean_list_{k}"] = topk_result[f"bo_mean_list_{k}"]
        result[f"bo_median_list_{k}"] = topk_result[f"bo_median_list_{k}"]
        result[f"init_top{k}_mean"] = np.mean(result[f"init_mean_list_{k}"])
        result[f"init_top{k}_median"] = np.mean(result[f"init_median_list_{k}"])
        print(f"init_top{k}_mean:", result[f"init_top{k}_mean"])
        print(f"init_top{k}_median:", result[f"init_top{k}_median"])
        result[f"bo_top{k}_mean"] = np.mean(result[f"bo_mean_list_{k}"])
        result[f"bo_top{k}_median"] = np.mean(result[f"bo_median_list_{k}"])
        print(f"bo_top{k}_mean:", result[f"bo_top{k}_mean"])
        print(f"bo_top{k}_median:", result[f"bo_top{k}_median"])
    result["init_qed_list"] = init_qed_list
    result["init_sa_list"] = init_sa_list
    result["init_logp_list"] = init_logp_list
    result["init_mw_list"] = init_mw_list
    result["init_diversity_list"] = init_diversity_list

    result["init_mean_qed"] = np.mean(init_qed_list)
    result["init_mean_sa"] = np.mean(init_sa_list)
    result["init_mean_logp"] = np.mean(init_logp_list)
    result["init_mean_mw"] = np.mean(init_mw_list)
    result["init_mean_diversity"] = np.mean(init_diversity_list)

    result["init_median_qed"] = np.mean(init_qed_median_list)
    result["init_median_sa"] = np.mean(init_sa_median_list)
    result["init_median_logp"] = np.mean(init_logp_median_list)
    result["init_median_mw"] = np.mean(init_mw_median_list)
    result["init_median_diversity"] = np.median(init_diversity_list)

    result["bo_qed_list"] = bo_qed_list
    result["bo_sa_list"] = bo_sa_list
    result["bo_logp_list"] = np.mean(bo_logp_list)
    result["bo_mw_list"] = np.mean(bo_mw_list)
    result["bo_diversity_list"] = bo_diversity_list
    
    result["bo_mean_qed"] = np.mean(bo_qed_list)
    result["bo_mean_sa"] = np.mean(bo_sa_list)
    result["bo_mean_logp"] = np.mean(bo_logp_list)
    result["bo_mean_mw"] = np.mean(bo_mw_list)
    result["bo_mean_diversity"] = np.mean(bo_diversity_list)
    
    result["bo_median_qed"] = np.mean(bo_qed_median_list)
    result["bo_median_sa"] = np.mean(bo_sa_median_list)
    result["bo_median_logp"] = np.mean(bo_logp_median_list)
    result["bo_median_mw"] = np.mean(bo_mw_median_list)
    result["bo_median_diversity"] = np.median(bo_diversity_list)

    result["init_smiles_list"] = init_smiles_list
    result["bo_smiles_list"] = bo_smiles_list
    with open(f"../{dir_path}/mean_result.pickle", "wb") as f:
        pickle.dump(result, f)
    print("init_qed:", result["init_mean_qed"])
    print("init_sa:", result["init_mean_sa"])
    print("init_logp:", result["init_mean_logp"])
    print("init_mw:", result["init_mean_mw"])
    print("init_diversity:", result["init_mean_diversity"])
    print("\n")
    
    print("init_qed_median:", result["init_median_qed"])
    print("init_sa_median:", result["init_median_sa"])
    print("init_logp_median:", result["init_median_logp"])
    print("init_mw_median:", result["init_median_mw"])
    print("init_diversity_median:", result["init_median_diversity"])
    print("\n")
    print("bo_qed:", result["bo_mean_qed"])
    print("bo_sa:", result["bo_mean_sa"])
    print("bo_logp:", result["bo_mean_logp"])
    print("bo_mw:", result["bo_mean_mw"])
    print("bo_diversity:", result["bo_mean_diversity"])
    print("\n")
    print("bo_qed_median:", result["bo_median_qed"])
    print("bo_sa_median:", result["bo_median_sa"])
    print("bo_logp_median:", result["bo_median_logp"])
    print("bo_mw_median:", result["bo_median_mw"])
    print("bo_diversity_median:", result["bo_median_diversity"])
    
    # data = {
    #     'mean_init': mean_init_list,
    #     'mean_ori': mean_ori_list,
    #     'mean_bo': mean_bo_list
    # }
    # df = pd.DataFrame(data, index=name_proteins)
    # df.to_csv(f"../{dir_path}/mean_result_score.csv", index_label='protein')


    pass