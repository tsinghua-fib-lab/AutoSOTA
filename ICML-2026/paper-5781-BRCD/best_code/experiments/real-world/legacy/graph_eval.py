import argparse
import glob
import math
import itertools
import json
import os
import pickle
import warnings
from datetime import datetime, timedelta
from multiprocessing import Pool
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
from RCAEval.utility import dump_json, is_py310, load_json
from RCAEval.utility.visualization import draw_adj, draw_digraph, draw_mem_graph

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


def adj2generalgraph(adj):
    # convert adj to GeneralGraph
    # add 1, since causallearn CausaGraph also do so,
    # take it out later in score_g function
    G = GeneralGraph(nodes=[f"X{i + 1}" for i in range(len(adj))])
    for row_idx in range(len(adj)):
        for col_idx in range(len(adj)):
            if adj[row_idx, col_idx] == 1:
                G.add_directed_edge(f"X{col_idx + 1}", f"X{row_idx + 1}")
    return G


def score_g(Data, G, parameters=None):
    # calculate the score for the current G
    # here G is a DAG
    parameters = {"lambda_value": 0}

    score = 0
    for i, node in enumerate(G.get_nodes()):
        PA = G.get_parents(node)

        # for granger
        if len(PA) > 0 and isinstance(PA[0], str):
            pass  # already in str format
        else:
            PA = [p.name for p in PA]

        if len(PA) > 0 and isinstance(PA[0], str):
            # this is for FCI, bc it doesn't have node_names param
            # remove X from list ['X6', 'X10']
            PA = [int(p[1:]) - 1 for p in PA]


        delta_score = local_score_BIC(Data, i, PA, parameters)

        # delta_score is nan, ignore
        if np.isnan(delta_score):
            continue

        score = score + delta_score
    return score.sum()


def parse_args():
    parser = argparse.ArgumentParser(description="RCAEval evaluation")
    # for data
    parser.add_argument("-i", "--input-path", type=str, default="data", help="path to data")
    parser.add_argument(
        "-o", "--output-path", type=str, default="output", help="for results and reports"
    )
    # length
    parser.add_argument("--length", type=int, default=None, help="length of time series")

    # for method
    parser.add_argument("-m", "--model", type=str, default="pc_pagerank", help="func name")
    parser.add_argument("-t", "--test", type=str, default=None, help="granger test or pc test")
    parser.add_argument("-a", "--alpha", type=float, default=0.05)
    parser.add_argument("--tau", type=float, default=3)
    parser.add_argument("--stable", action="store_true")

    # for evaluation
    parser.add_argument("-w", "--worker-num", type=int, default=1, help="number of workers")
    parser.add_argument("--iter-num", type=int, default=1)
    parser.add_argument("--eval-step", type=int, default=None)
    parser.add_argument("--bic", action="store_true")

    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--useful", action="store_true")
    parser.add_argument("--tuning", action="store_true")
    parser.add_argument("--small", action="store_true")
    parser.add_argument("--large", action="store_true")

    args = parser.parse_args()

    assert args.alpha in [0.005, 0.01, 0.05, 0.1, 0.2]

    # assert args.tau in [3, 6, 10]

    # check if args.model is defined here
    if args.model not in globals():
        raise ValueError(f"{args.model=} not defined. Available: {AVAILABLE_METHODS}")

    if args.verbose:
        print(json.dumps(vars(args), indent=2, sort_keys=True))
    return args


args = parse_args()

SEARCH_SPACE = {  # large space as default
    "indep_list": ["fisherz", "gsq", "chisq"],
    "alpha_list": [0.005, 0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5],
    "stable_list": [True, False],
}


if args.small is True:
    SEARCH_SPACE = {
        "indep_list": ["gsq", "chisq"],
        "alpha_list": [0.005, 0.01, 0.05, 0.1, 0.2],
        "stable_list": [True, False],
    }
elif args.large is True:
    SEARCH_SPACE = {
        "indep_list": ["fisherz", "gsq", "chisq"],
        "alpha_list": [0.005, 0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5],
        "stable_list": [True, False],
    }


# ==== PREPARE PATHS ====
if args.tuning is True:
    scale = "small" if args.small else "large"
    art_name = f"{basename(args.input_path)}_{args.model.replace('_', '-')}_tuning_{scale}"
else:
    art_name = f"{basename(args.input_path)}_l{args.length}_{args.model.replace('_', '-')}_stable-{args.stable}_{args.test}_{args.alpha}"

output_path = f"{args.output_path}/{art_name}"
report_path = join(output_path, f"{art_name}.xlsx")
result_path = join(output_path, "results")

if not exists(args.output_path):
    os.makedirs(args.output_path)

# if exists(result_path):
#     shutil.rmtree(result_path)
os.makedirs(result_path, exist_ok=True)

dump_json(filename=join(output_path, "args.json"), data=vars(args))


# ==== PROCESS TO GENERATE JSON ====
data_paths = list(glob.glob(os.path.join(args.input_path, "**/data.csv"), recursive=True))


def evaluate():
    eval_data = {
        "Case": [],
        "Precision": [],
        "Recall": [],
        "F1-Score": [],
        "Precision-Skel": [],
        "Recall-Skel": [],
        "F1-Skel": [],
        "BIC": [],
        "SHD": [],
    }

    # print("Evaluate...")
    for data_path in data_paths:
        # for circa and rcd
        if "_circa" in data_path or "_rcd" in data_path:
            num_node = int(basename(dirname(dirname(dirname(dirname(data_path))))))
            graph_idx = int(basename(dirname(dirname(dirname(data_path)))))
            case_idx = int(basename(dirname(data_path)))

        # === for CausIL
        if "syn_causil" in data_path:
            graph_idx = int(basename(dirname(data_path))[-1:])
            case_idx = 0

        # ===== READ RESULT =====
        est_graph_name = f"{graph_idx}_{case_idx}_est_graph.json"
        est_graph_path = join(result_path, est_graph_name)

        # for real circa pc
        # est_graph_name = f"../CIRCA/results/{graph_idx}_{case_idx}.json"
        # est_graph_path = est_graph_name

        if not exists(est_graph_path):
            continue
        est_graph = MemoryGraph.load(est_graph_path)

        # ====== READ TRUE GRAPH =====
        # for circa
        # /home/luan/ws/RCAEval/data/syn_circa/10/0/cases/0/data.csv
        if "_circa" in data_path:
            true_graph_path = join(dirname(dirname(dirname(data_path))), "graph.json")
            true_graph = MemoryGraph.load(true_graph_path)

        # for causil
        # /home/luan/ws/RCAEval/data/syn_causil/10_services/synthetic/Graph0/data.csv
        if "syn_causil" in data_path:
            dag_gt = pickle.load(open(join(dirname(data_path), "DAG.gpickle"), "rb"))
            true_graph = MemoryGraph(dag_gt)
            # draw_digraph(dag_gt, figsize=(8, 8))

        # for rcd
        if "_rcd" in data_path:
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

        if args.bic:
            bic_path = join(result_path, f"{graph_idx}_{case_idx}_bic.txt")
            if exists(bic_path):
                bic = float(open(bic_path).read())
                eval_data["BIC"].append(bic)

    avg_precision = np.mean(eval_data["Precision"])
    avg_recall = np.mean(eval_data["Recall"])
    avg_f1 = np.mean(eval_data["F1-Score"])
    avg_precision_skel = np.mean(eval_data["Precision-Skel"])
    avg_recall_skel = np.mean(eval_data["Recall-Skel"])
    avg_f1_skel = np.mean(eval_data["F1-Skel"])

    avg_shd = np.mean(eval_data["SHD"])
    avg_bic = np.mean(eval_data["BIC"])

    eval_data["Case"].insert(0, "Average")
    eval_data["Precision"].insert(0, avg_precision)
    eval_data["Recall"].insert(0, avg_recall)
    eval_data["F1-Score"].insert(0, avg_f1)
    eval_data["Precision-Skel"].insert(0, avg_precision_skel)
    eval_data["Recall-Skel"].insert(0, avg_recall_skel)
    eval_data["F1-Skel"].insert(0, avg_f1_skel)
    eval_data["BIC"].insert(0, avg_bic)
    eval_data["SHD"].insert(0, avg_shd)

    # print Average
    # print("================================")
    print(f"F1:   {avg_f1:.2f}")
    print(f"F1-S: {avg_f1_skel:.2f}")
    print(f"SHD:  {math.floor(avg_shd)}")

    if args.bic is not True:
        eval_data.pop("BIC")

    report_df = pd.DataFrame(eval_data)
    report_df.to_excel(report_path, index=False)

    # print(f"Results are saved to {result_path}")
    # print(f"Report is saved to {abspath(report_path)}")


def process(data_path):
    # return

    # === for CIRCA
    # /home/luan/ws/RCAEval/data/syn_circa/10/0/cases/0/data.csv
    # graph_0_case_0
    if "_circa" in data_path:
        num_node = int(basename(dirname(dirname(dirname(dirname(data_path))))))
        graph_idx = int(basename(dirname(dirname(dirname(data_path)))))
        case_idx = int(basename(dirname(data_path)))

    # === for CausIL
    if "syn_causil" in data_path:
        # /home/luan/ws/RCAEval/data/syn_causil/10_services/synthetic/Graph0/data.csv
        # graph_0
        num_node = int(basename(dirname(dirname(dirname(data_path)))).split("_")[0])
        graph_idx = int(basename(dirname(data_path))[-1:])
        case_idx = 0

    # === for rcd
    # /home/luan/ws/RCAEval/data/syn_rcd/10/0/cases/0/data.csv
    if "_rcd" in data_path:
        num_node = int(basename(dirname(dirname(dirname(dirname(data_path))))))
        graph_idx = int(basename(dirname(dirname(dirname(data_path)))))
        case_idx = int(basename(dirname(data_path)))

    if "syn_circa" in data_path:  # rca_circa has headers, no worries
        data = pd.read_csv(
            data_path,
            header=None,
        )
        data.header = list(map(str, range(0, data.shape[1])))
    else:
        data = pd.read_csv(
            data_path,
        )
        # print(data.shape)

        if args.bic is True:
            # get the last 30% rows for evaluation
            eval_data = data.iloc[int(data.shape[0] * 0.7) :, :]
            eval_data = eval_data.to_numpy().astype(float)
            # eval_data = data.to_numpy().astype(float)
            eval_data = np.mat(eval_data)

            # print("Only get the first 70% rows to do causal graph construction")
            data = data.iloc[: int(data.shape[0] * 0.7), :]

    # == PROCESS ==
    # func = globals()[args.model]
    data = data.fillna(method="ffill")
    data = data.fillna(value=0)
    np_data = np.absolute(data.to_numpy().astype(float))

    # select the first 10000 rows
    # np_data = np_data[:50000, :]

    if args.length is not None:
        np_data = np_data[: args.length, :]

    # print(np_data.shape)

    adj = []
    G = None

    st = datetime.now()
    try:
        if args.model == "pc":
            assert args.test in [None, "fisherz", "gsq", "chisq", "mv_fisherz", "kci"]
            if args.test is None:
                args.test = "fisherz"
            indep_test = globals()[args.test]

            if args.tuning is True:  # tuning mode, ignore args.test and alpha
                # select 70% np_data for train, 30% for eval
                train_data = np_data[: int(np_data.shape[0] * 0.7), :]
                eval_data = np_data[int(np_data.shape[0] * 0.7) :, :]
                best_bic = np.inf
                best_config = None

                # indep_list = ["fisherz", "gsq", "chisq"]
                # indep_list = ["gsq", "chisq"]
                # alpha_list = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
                # alpha_list = [0.005, 0.01, 0.05, 0.1, 0.2]
                # stable_list = [True, False]
                indep_list = SEARCH_SPACE["indep_list"]
                alpha_list = SEARCH_SPACE["alpha_list"]
                stable_list = SEARCH_SPACE["stable_list"]
                if (
                    "_circa" in data_path and "chisq" in indep_list
                ):  # remove chisq out of indep_list, Out of memory
                    indep_list.remove("chisq")
                if (
                    "syn_causil" in data_path and "chisq" in indep_list
                ):  # remove chisq out of indep_list, Out of memory
                    indep_list.remove("chisq")

                configs = list(itertools.product(indep_list, alpha_list, stable_list))

                for indep_test, alpha, stable in configs:
                    print(f"{indep_test=} {alpha=} {stable=}")
                    G = pc(
                        train_data,
                        alpha=alpha,
                        indep_test=indep_test,
                        stable=stable,
                        show_progress=False,
                        node_names=range(np_data.shape[1]),
                    ).G

                    bic = score_g(eval_data, G)
                    if bic < best_bic:
                        best_bic = bic
                        best_config = (indep_test, alpha, stable)
                        print(f"{graph_idx}_{case_idx}: {best_bic=}, {best_config=}")

                indep_test, alpha, stable = best_config
                # write down best param
                with open(join(result_path, f"{graph_idx}_{case_idx}_best_config.txt"), "w") as f:
                    f.write(f"{indep_test=}_{alpha=}_{stable=}")

                # train the graph using whole data and best config
                G = pc(
                    np_data,
                    alpha=best_config[1],
                    indep_test=best_config[0],
                    stable=best_config[2],
                    show_progress=False,
                    node_names=range(np_data.shape[1]),
                ).G
                adj = G.graph

            elif args.bic is True:
                G = pc(
                    np_data,
                    alpha=args.alpha,
                    indep_test=indep_test,
                    stable=args.stable,
                    show_progress=False,
                    node_names=range(np_data.shape[1]),
                ).G
                adj = G.graph

                bic = score_g(eval_data, G)
                with open(join(result_path, f"{graph_idx}_{case_idx}_bic.txt"), "w") as f:
                    f.write(f"{bic}")

            else:
                adj = pc(
                    np_data,
                    alpha=args.alpha,
                    indep_test=indep_test,
                    stable=args.stable,
                    show_progress=False,
                ).G.graph

        elif args.model == "fci":
            assert args.test in [None, "fisherz", "gsq", "chisq", "mv_fisherz", "kci"]
            if args.test is None:
                args.test = "fisherz"

            indep_test = globals()[args.test]

            if args.tuning is True:  # tuning mode, ignore args.test and alpha
                # select 70% np_data for train, 30% for eval
                train_data = np_data[: int(np_data.shape[0] * 0.7), :]
                eval_data = np_data[int(np_data.shape[0] * 0.7) :, :]
                best_bic = np.inf
                best_config = None

                indep_list = SEARCH_SPACE["indep_list"]
                alpha_list = SEARCH_SPACE["alpha_list"]
                stable_list = SEARCH_SPACE["stable_list"]
                if (
                    "_circa" in data_path and "chisq" in indep_list
                ):  # remove chisq out of indep_list, Out of memory
                    indep_list.remove("chisq")
                if (
                    "syn_causil" in data_path and "chisq" in indep_list
                ):  # remove chisq out of indep_list, Out of memory
                    indep_list.remove("chisq")

                configs = list(itertools.product(indep_list, alpha_list, stable_list))

                for indep_test, alpha, stable in configs:
                    print(f"{indep_test=} {alpha=} {stable=}")
                    G = fci(
                        train_data,
                        independence_test_method=indep_test,
                        alpha=alpha,
                        show_progress=False,
                        verbose=False,
                        node_names=range(np_data.shape[1]),
                    )[0]
                    bic = score_g(eval_data, G)
                    if bic < best_bic:
                        best_bic = bic
                        best_config = (indep_test, alpha, stable)
                        print(f"{graph_idx}_{case_idx}: {best_bic=}, {best_config=}")

                # write down best param
                indep_test, alpha, stable = best_config
                with open(join(result_path, f"{graph_idx}_{case_idx}_best_config.txt"), "w") as f:
                    f.write(f"{indep_test=}_{alpha=}_{stable=}")

                # train the graph using whole data and best config
                G = fci(
                    np_data,
                    independence_test_method=best_config[0],
                    alpha=best_config[1],
                    stable=best_config[2],
                    show_progress=False,
                    verbose=False,
                    node_names=range(np_data.shape[1]),
                )[0]
                adj = G.graph
                bic = score_g(eval_data, G)
                with open(join(result_path, f"{graph_idx}_{case_idx}_bic.txt"), "w") as f:
                    f.write(f"{bic}")

            elif args.bic is True:
                G = fci(
                    np_data,
                    independence_test_method=indep_test,
                    alpha=args.alpha,
                    show_progress=False,
                    verbose=False,
                    node_names=range(np_data.shape[1]),
                )[0]
                adj = G.graph
                bic = score_g(eval_data, G)
                with open(join(result_path, f"{graph_idx}_{case_idx}_bic.txt"), "w") as f:
                    f.write(f"{bic}")

            else:
                adj = fci(
                    np_data,
                    independence_test_method=indep_test,
                    alpha=args.alpha,
                    show_progress=False,
                    verbose=False,
                )[0].graph

        elif args.model == "ICALiNGAM":
            if args.tuning is True:  # tuning mode
                train_data = np_data[: int(np_data.shape[0] * 0.7), :]
                eval_data = np_data[int(np_data.shape[0] * 0.7) :, :]
                best_bic = np.inf
                best_config = None

                # for max_iter in [100, 200, 500, 1000, 1500]:
                for max_iter in range(200, 1200, 200):
                    print(f"{max_iter=}")
                    model = ICALiNGAM(max_iter=max_iter)
                    model.fit(np_data)
                    adj = model.adjacency_matrix_
                    adj = adj.astype(bool).astype(int)

                    G = adj2generalgraph(adj)
                    bic = score_g(eval_data, G)
                    if bic < best_bic:
                        best_bic = bic
                        best_config = max_iter
                        print(f"{graph_idx}_{case_idx}: {best_bic=}, {best_config=}")

                # write down best param
                with open(join(result_path, f"{graph_idx}_{case_idx}_best_config.txt"), "w") as f:
                    f.write(f"{best_config=}")

                model = ICALiNGAM(max_iter=best_config)
                model.fit(np_data)
                adj = model.adjacency_matrix_
                adj = adj.astype(bool).astype(int)
            else:
                model = ICALiNGAM()
                model.fit(np_data)
                adj = model.adjacency_matrix_
                adj = adj.astype(bool).astype(int)

        elif args.model == "VARLiNGAM":
            raise NotImplementedError
        elif args.model == "DirectLiNGAM":
            if args.tuning is True:  # tuning mode
                train_data = np_data[: int(np_data.shape[0] * 0.7), :]
                eval_data = np_data[int(np_data.shape[0] * 0.7) :, :]
                best_bic = np.inf
                best_measure = None

                for measure in ["pwling", "kernel"]:
                    print(f"{measure=}")
                    model = DirectLiNGAM(measure=measure)
                    model.fit(train_data)
                    adj = model.adjacency_matrix_
                    adj = adj.astype(bool).astype(int)

                    G = adj2generalgraph(adj)
                    bic = score_g(eval_data, G)
                    if bic < best_bic:
                        best_bic = bic
                        best_measure = measure
                        print(f"{graph_idx}_{case_idx}: {best_bic=}, {best_measure=}")

                # write down best param
                with open(join(result_path, f"{graph_idx}_{case_idx}_best_config.txt"), "w") as f:
                    f.write(f"{best_measure=}")

                model = DirectLiNGAM(measure=best_measure)
                model.fit(train_data)
                adj = model.adjacency_matrix_
                adj = adj.astype(bool).astype(int)

            else:
                model = DirectLiNGAM()
                model.fit(np_data)
                adj = model.adjacency_matrix_
                adj = adj.astype(bool).astype(int)
        elif args.model == "ges":
            if args.tuning is True:  # tuning mode
                train_data = np_data[: int(np_data.shape[0] * 0.7), :]
                eval_data = np_data[int(np_data.shape[0] * 0.7) :, :]
                best_bic = np.inf
                best_config = None

                score_func_list = [
                    # 'local_score_CV_general',
                    "local_score_marginal_general",
                    # 'local_score_CV_multi',
                    # 'local_score_marginal_multi',
                    "local_score_BIC",
                    "local_score_BDeu",
                ]
                maxP_list = np.linspace(1, train_data.shape[1] // 2, 4, dtype=int)
                lambda_list = np.linspace(0, 1, 4)

                configs = list(itertools.product(score_func_list, maxP_list, lambda_list))

                for score_func, maxP, lambd in configs:
                    print(f"{score_func=}, {maxP=}, {lambd=}")
                    record = ges(
                        train_data, score_func=score_func, maxP=maxP, parameters={"lambda": lambd}
                    )
                    adj = record["G"].graph

                    G = adj2generalgraph(adj)
                    bic = score_g(eval_data, G)
                    if bic < best_bic:
                        best_bic = bic
                        best_config = (score_func, maxP, lambd)
                        print(f"{graph_idx}_{case_idx}: {best_bic=}, {best_config=}")

                # write down best param
                with open(join(result_path, f"{graph_idx}_{case_idx}_best_config.txt"), "w") as f:
                    f.write(f"{best_config=}")

                score_func, maxP, lambd = best_config
                record = ges(
                    np_data, score_func=score_func, maxP=maxP, parameters={"lambda": lambd}
                )
                adj = record["G"].graph

            else:
                record = ges(np_data)
                adj = record["G"].graph

        elif args.model == "granger":
            assert args.test in [None, "ssr_ftest", "ssr_chi2test", "lrtest", "params_ftest"]

            if args.tuning is True:  # tuning mode, ignore args.test and alpha
                # we have test, alpha, tau

                # select 70% np_data for train, 30% for eval
                train_data = np_data[: int(np_data.shape[0] * 0.7), :]
                eval_data = np_data[int(np_data.shape[0] * 0.7) :, :]
                best_bic = np.inf
                best_config = None

                test_list = ["ssr_ftest", "ssr_chi2test", "lrtest", "params_ftest"]
                p_val_list = [0.01, 0.05, 0.1, 0.2]  # p-value
                tau_list = [1, 3, 5]  # p-value

                configs = list(itertools.product(test_list, p_val_list, tau_list))

                for test, p_val, tau in configs:
                    print(f"{test=} {p_val=} {tau=}")
                    adj = granger(
                        pd.DataFrame(train_data), test=test, maxlag=tau, p_val_threshold=p_val
                    )

                    # convert adj to GeneralGraph
                    # add 1, since causallearn CausaGraph also do so, minus later in score_g function
                    G = GeneralGraph(nodes=[f"X{i + 1}" for i in range(np_data.shape[1])])
                    for row_idx in range(len(adj)):
                        for col_idx in range(len(adj)):
                            if adj[row_idx, col_idx] == 1:
                                G.add_directed_edge(f"X{col_idx + 1}", f"X{row_idx + 1}")

                    bic = score_g(eval_data, G)
                    if bic < best_bic:
                        best_bic = bic
                        best_config = (test, p_val, tau)
                        print(f"{graph_idx}_{case_idx}: {best_bic=}, {best_config=}")

                # write down best param
                test, p_val, tau = best_config
                with open(join(result_path, f"{graph_idx}_{case_idx}_best_config.txt"), "w") as f:
                    f.write(f"{test=}_{p_val=}_{tau=}")

                # train the graph using whole data and best config
                adj = granger(pd.DataFrame(np_data), test=test, maxlag=tau, p_val_threshold=p_val)

            else:
                adj = granger(data, test=args.test, maxlag=args.tau, p_val_threshold=args.alpha)

        elif args.model == "pcmci":
            if args.tuning is True:
                # what we can tune
                # tau and alpha, get it from granger
                train_data = np_data[: int(np_data.shape[0] * 0.7), :]
                eval_data = np_data[int(np_data.shape[0] * 0.7) :, :]
                best_bic = np.inf
                best_config = None

                alpha_list = [0.005, 0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5]
                tau_list = [1, 3, 5]
                configs = list(itertools.product(alpha_list, tau_list))

                for alpha, tau in configs:
                    print(f"{alpha=} {tau=}")
                    adj = pcmci(pd.DataFrame(train_data), tau_max=tau, alpha=alpha)
                    G = adj2generalgraph(adj)

                    bic = score_g(eval_data, G)
                    if bic < best_bic:
                        best_bic = bic
                        best_config = (alpha, tau)
                        print(f"{graph_idx}_{case_idx}: {best_bic=}, {best_config=}")

                # write down best param
                alpha, tau = best_config
                with open(join(result_path, f"{graph_idx}_{case_idx}_best_config.txt"), "w") as f:
                    f.write(f"{alpha=}_{tau=}")

                adj = pcmci(pd.DataFrame(np_data), tau_max=tau, alpha=alpha)

            else:
                adj = pcmci(pd.DataFrame(np_data))
        elif args.model == "cmlp":
            adj = cmlp(pd.DataFrame(np_data))
        elif args.model == "notears":
            adj = notears(pd.DataFrame(np_data))
        elif args.model == "ntlr":
            adj = ntlr(pd.DataFrame(np_data))

        elif args.model == "dag_gnn":
            adj = dag_gnn(pd.DataFrame(np_data))

        else:
            raise ValueError(f"{args.model=} not defined. Available: {AVAILABLE_METHODS}")

        # for CIRCA
        if "_circa" in data_path:
            est_graph = MemoryGraph.from_adj(
                adj, nodes=[Node("SIM", str(i)) for i in range(len(adj))]
            )
        else:
            # for others
            est_graph = MemoryGraph.from_adj(adj, nodes=data.columns.to_list())

        est_graph.dump(join(result_path, f"{graph_idx}_{case_idx}_est_graph.json"))
        draw_mem_graph(
            est_graph,
            figsize=(10, 10),
            filename=join(result_path, f"{graph_idx}_{case_idx}_est_graph.png"),
        )

    except Exception as e:
        raise e
        print(f"{args.model=} failed on {data_path=}")
        print(e)
        est_graph = MemoryGraph.from_adj([], nodes=[])
        est_graph.dump(join(result_path, f"{graph_idx}_{case_idx}_failed.json"))

    with open(join(result_path, f"{graph_idx}_{case_idx}_time_taken.txt"), "w") as f:
        tt = datetime.now() - st
        tt = tt - timedelta(microseconds=tt.microseconds)

        f.write(f"Time taken: {tt}")

        # check if time taken (tt) over 1 hour ?
        if tt > timedelta(hours=1):
            return 2


start_time = datetime.now()

if args.worker_num > 1:
    with Pool(min(args.worker_num, os.cpu_count() - 2)) as p:
        list(tqdm(p.imap(process, data_paths), total=len(data_paths)))
else:  # single worker
    cnt = 0
    over_time = 0
    for data_path in tqdm(data_paths):
        output = process(data_path)

        cnt += 1
        if args.eval_step is not None and cnt % args.eval_step == 0:
            evaluate()

        # one case take too long (over 1 hour, terminate this loop after just 1 evaluation)
        if output == 2:
            over_time += 1

        if over_time == 2:
            print(f"Terminated at {cnt=} due to time taken over 1 hour")
            break

end_time = datetime.now()
time_taken = end_time - start_time
# remove set milliseconds to 0
time_taken = time_taken - timedelta(microseconds=time_taken.microseconds)
with open(join(output_path, "time_taken.txt"), "w") as f:
    s = f"Time taken: {time_taken}"
    print(s)
    f.write(s)


evaluate()