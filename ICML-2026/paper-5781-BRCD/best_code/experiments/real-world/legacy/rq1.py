import argparse
import glob
import math
import itertools
import json
import os
import pickle
import warnings
from datetime import datetime, timedelta
from tempfile import TemporaryDirectory
from os.path import abspath, basename, dirname, exists, join

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.score.LocalScoreFunction import local_score_BIC
from tqdm import tqdm

from RCAEval.benchmark.evaluation import Evaluator
from RCAEval.benchmark.metrics import F1, SHD, F1_Skeleton
from RCAEval.classes.graph import MemoryGraph, Node
from RCAEval.graph_heads import finalize_directed_adj
from RCAEval.io.time_series import drop_constant, drop_extra, drop_time
from RCAEval.utility import (
    dump_json,
    download_syn_rcd_dataset,
    download_syn_circa_dataset,
    download_syn_causil_dataset,
    is_py310,
    load_json,
)



if is_py310():
    from causallearn.search.ConstraintBased.FCI import fci
    from causallearn.search.ConstraintBased.PC import pc
    from causallearn.search.FCMBased.lingam import DirectLiNGAM, ICALiNGAM, VARLiNGAM
    from causallearn.search.ScoreBased.GES import ges
    from causallearn.utils.cit import chisq, fisherz, gsq, kci, mv_fisherz
    from RCAEval.graph_construction.granger import granger
    from RCAEval.graph_construction.pcmci import pcmci
    from RCAEval.graph_construction.cmlp import cmlp
    try:
        from RCAEval.graph_construction.dag_gnn import dag_gnn
        from RCAEval.graph_construction.dag_gnn import notears_low_rank as ntlr
        from RCAEval.graph_construction.notears import notears
    except Exception as e:
        print(e)
    
AVAILABLE_METHODS = sorted(
    [
        "pc",
        "ppc",
        "pcmci",
        "fci",
        "notears",
        "ntlr",
        "DirectLiNGAM",
        "VARLiNGAM",
        "ICALiNGAM",
        "ges",
        "granger",
    ]
)


def parse_args():
    parser = argparse.ArgumentParser(description="RCAEval evaluation")
    # for data
    parser.add_argument("--dataset", type=str, default="data", help="Choose a dataset.",
        choices=["circa10", "circa50", "rcd10", "rcd50", "causil10", "causil50"])
    parser.add_argument("--method", type=str, help="Method name")
    parser.add_argument("--length", type=int, default=None, help="length of time series")
    parser.add_argument("--test", action="store_true", help="Perform smoke test on certain methods without fully run")
    args = parser.parse_args()

    if args.method not in globals():
        raise ValueError(f"{args.method=} not defined. Available: {AVAILABLE_METHODS}")

    if args.dataset not in ["circa10", "circa50", "rcd10", "rcd50", "causil10", "causil50"]:
        print(f"{args.dataset=} not defined. Available: circa10, circa50, rcd10, rcd50, causil10, causil50")
        exit()

    return args


args = parse_args()


# download dataset
if "circa" in args.dataset:
    download_syn_circa_dataset()
elif "rcd" in args.dataset:
    download_syn_rcd_dataset()
elif "causil" in args.dataset:
    download_syn_causil_dataset()
DATASET_MAP = {
    "circa10": "data/syn_circa/10",
    "circa50": "data/syn_circa/50",
    "causil10": "data/syn_causil/10",
    "causil50": "data/syn_causil/50",
    "rcd10": "data/syn_rcd/10",
    "rcd50": "data/syn_rcd/50"
}
dataset = DATASET_MAP[args.dataset]



output_path = TemporaryDirectory().name
report_path = join(output_path, "report.xlsx")
result_path = join(output_path, "results")
os.makedirs(result_path, exist_ok=True)


# ==== PROCESS TO GENERATE JSON ====
data_paths = list(glob.glob(os.path.join(dataset, "**/data.csv"), recursive=True))
if args.test is True:
    data_paths = data_paths[:2]


def evaluate():
    eval_data = {
        "Case": [],
        "Precision": [],
        "Recall": [],
        "F1-Score": [],
        "Precision-Skel": [],
        "Recall-Skel": [],
        "F1-Skel": [],
        "SHD": [],
    }

    for data_path in data_paths:
        if "circa" in data_path or "rcd" in data_path:
            num_node = int(basename(dirname(dirname(dirname(dirname(data_path))))))
            graph_idx = int(basename(dirname(dirname(dirname(data_path)))))
            case_idx = int(basename(dirname(data_path)))

        if "causil" in data_path:
            graph_idx = int(basename(dirname(data_path))[-1:])
            case_idx = 0

        # ===== READ RESULT =====
        est_graph_name = f"{graph_idx}_{case_idx}_est_graph.json"
        est_graph_path = join(result_path, est_graph_name)

        if not exists(est_graph_path):
            continue
        est_graph = MemoryGraph.load(est_graph_path)

        # ====== READ TRUE GRAPH =====
        if "circa" in data_path:
            true_graph_path = join(dirname(dirname(dirname(data_path))), "graph.json")
            true_graph = MemoryGraph.load(true_graph_path)

        if "causil" in data_path:
            dag_gt = pickle.load(open(join(dirname(data_path), "DAG.gpickle"), "rb"))
            true_graph = MemoryGraph(dag_gt)

        if "rcd" in data_path:
            dag_gt = pickle.load(
                open(join(dirname(dirname(dirname(data_path))), "g_graph.pkl"), "rb")
            )
            true_graph = MemoryGraph(dag_gt)
            true_graph = MemoryGraph.load(
                join(dirname(dirname(dirname(data_path))), "true_graph.json")
            )

        e = F1(true_graph, est_graph)
        e_skel = F1_Skeleton(true_graph, est_graph)
        shd = SHD(true_graph, est_graph)

        eval_data["Case"].append(est_graph_name)
        eval_data["Precision"].append(e["precision"])
        eval_data["Recall"].append(e["recall"])
        eval_data["F1-Score"].append(e["f1"])
        eval_data["Precision-Skel"].append(e_skel["precision"])
        eval_data["Recall-Skel"].append(e_skel["recall"])
        eval_data["F1-Skel"].append(e_skel["f1"])
        eval_data["SHD"].append(shd)

    avg_precision = np.mean(eval_data["Precision"])
    avg_recall = np.mean(eval_data["Recall"])
    avg_f1 = np.mean(eval_data["F1-Score"])
    avg_precision_skel = np.mean(eval_data["Precision-Skel"])
    avg_recall_skel = np.mean(eval_data["Recall-Skel"])
    avg_f1_skel = np.mean(eval_data["F1-Skel"])
    avg_shd = np.mean(eval_data["SHD"])

    print(f"F1:   {avg_f1:.2f}")
    print(f"F1-S: {avg_f1_skel:.2f}")
    print(f"SHD:  {math.floor(avg_shd)}")



def process(data_path):
    if "circa" in data_path:
        num_node = int(basename(dirname(dirname(dirname(dirname(data_path))))))
        graph_idx = int(basename(dirname(dirname(dirname(data_path)))))
        case_idx = int(basename(dirname(data_path)))

    if "causil" in data_path:
        num_node = int(basename(dirname(dirname(dirname(data_path)))).split("_")[0])
        graph_idx = int(basename(dirname(data_path))[-1:])
        case_idx = 0

    if "rcd" in data_path:
        num_node = int(basename(dirname(dirname(dirname(dirname(data_path))))))
        graph_idx = int(basename(dirname(dirname(dirname(data_path)))))
        case_idx = int(basename(dirname(data_path)))

    if "circa" in data_path: 
        data = pd.read_csv(data_path, header=None)
        data.header = list(map(str, range(0, data.shape[1])))
    else:
        data = pd.read_csv(data_path)


    # == PROCESS ==
    data = data.fillna(method="ffill")
    data = data.fillna(value=0)
    np_data = np.absolute(data.to_numpy().astype(float))

    if args.length is not None:
        np_data = np_data[: args.length, :]

    adj = []
    G = None

    st = datetime.now()
    try:
        if args.method == "pc":
            adj = pc(
                np_data,
                stable=False,
                show_progress=False,
            ).G.graph
        elif args.method == "fci":
            adj = fci(
                np_data,
                show_progress=False,
                verbose=False,
            )[0].graph
        elif args.method == "ICALiNGAM":
            model = ICALiNGAM()
            model.fit(np_data)
            adj = model.adjacency_matrix_
            adj = adj.astype(bool).astype(int)
        elif args.method == "VARLiNGAM":
            raise NotImplementedError
        elif args.method == "DirectLiNGAM":
            model = DirectLiNGAM()
            model.fit(np_data)
            adj = model.adjacency_matrix_
            adj = adj.astype(bool).astype(int)
        elif args.method == "ges":
            record = ges(np_data)
            adj = record["G"].graph
        elif args.method == "granger":
            adj = granger(data)
        elif args.method == "pcmci":
            adj = pcmci(pd.DataFrame(np_data))
        elif args.method == "ntlr":
            adj = ntlr(pd.DataFrame(np_data))
        else:
            raise ValueError(f"{args.method=} not defined. Available: {AVAILABLE_METHODS}")

        if "circa" in data_path:
            est_graph = MemoryGraph.from_adj(
                adj, nodes=[Node("SIM", str(i)) for i in range(len(adj))]
            )
        else:
            est_graph = MemoryGraph.from_adj(adj, nodes=data.columns.to_list())

        est_graph.dump(join(result_path, f"{graph_idx}_{case_idx}_est_graph.json"))

    except Exception as e:
        raise e
        print(f"{args.method=} failed on {data_path=}")
        est_graph = MemoryGraph.from_adj([], nodes=[])
        est_graph.dump(join(result_path, f"{graph_idx}_{case_idx}_failed.json"))


start_time = datetime.now()

for data_path in tqdm(data_paths):
    output = process(data_path)

end_time = datetime.now()
time_taken = end_time - start_time
avg_speed = round(time_taken.total_seconds() / len(data_paths), 2)


evaluate()
print("Avg speed:", avg_speed)
