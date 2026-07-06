from typing import Set, Tuple, FrozenSet, Iterable
import numpy as np
import pandas as pd
import random

from pyciphod.utils.scms.dynamic_scm import create_random_linear_dt_dynamic_scm, DtDynamicSCM, create_random_linear_dt_dynamic_scm_from_ftadmg
from pyciphod.utils.time_series.data_format import DTimeVar
from pyciphod.utils.graphs.partially_specified_graphs import CompletedPartiallyDirectedAcyclicDifferenceGraph
from pyciphod.causal_discovery.basic.constraint_based import PC, RestPC, FCI
from pyciphod.utils.graphs.temporal_graphs import create_random_ft_dag, FtDirectedAcyclicGraph
from pyciphod.utils.stat_tests.independence_tests import LinearRegressionCoefficientTTest, FisherZTest, CopulaTest
from pyciphod.utils.stat_tests.dependency_measures import Copula, compute_copula_fit


def f1_score_o(g, g_hat):
    def arrowhead_set(graph):
        out = set()
        vertices = list(graph.get_vertices())
        for u in vertices:
            for v in vertices:
                if u == v:
                    continue
                if graph.is_adjacent(u, v) and graph.is_pointed_edge(u, v):
                    out.add((u, v))
        return out

    true_set = arrowhead_set(g)
    pred_set = arrowhead_set(g_hat)
    TP = len(pred_set & true_set)
    FP = len(pred_set - true_set)
    FN = len(true_set - pred_set)
    prec = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    rec = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    if prec + rec == 0.0:
        return 0.0
    return 2 * (prec * rec) / (prec + rec)


def f1_score_a(g, g_hat):
    def adjacency_set(graph):
        out = set()
        vertices = list(graph.get_vertices())
        for u in vertices:
            for v in graph.get_adjacencies(u):
                if u == v:
                    continue
                out.add(frozenset((u, v)))
        return out

    # dir= g.get_directed_edges()
    # undir = g.get_undirected_edges()
    # true_set = dir.union(undir)
    # dir= g_hat.get_directed_edges()
    # undir = g_hat.get_undirected_edges()
    # pred_set = dir.union(undir)
    #
    # TP = set()
    # for p in list(true_set):
    #     if p in list(pred_set):
    #         if p not in TP:
    #             TP.add(p)
    #
    # FP = set()
    # for p in list(pred_set):
    #     if p not in list(true_set):
    #         if p not in FP:
    #             FP.add(p)
    #
    # FN = set()
    # for p in list(true_set):
    #     if p not in list(pred_set):
    #         if p not in FP:
    #             FN.add(p)
    # TP = len(TP)
    # FP = len(FP)
    # FN = len(FN)

    true_set = adjacency_set(g)
    pred_set = adjacency_set(g_hat)

    TP = len(pred_set & true_set)
    FP = len(pred_set - true_set)
    FN = len(true_set - pred_set)

    prec = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    rec = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    if prec + rec == 0.0:
        return 0.0
    return 2 * (prec * rec) / (prec + rec)


def fork_structure(n):
    X1 = DTimeVar(f"X", 1)
    X2 = DTimeVar(f"X", 2)
    X3 = DTimeVar(f"X", 3)
    Y1 = DTimeVar(f"Y", 1)
    Y2 = DTimeVar(f"Y", 2)
    Y3 = DTimeVar(f"Y", 3)
    Z1 = DTimeVar(f"Z", 1)
    Z2 = DTimeVar(f"Z", 2)
    Z3 = DTimeVar(f"Z", 3)

    azz = random.choice([1, -1]) * random.uniform(-1, -0.5)
    axx = random.choice([1, -1]) * random.uniform(-1, -0.5)
    azx = random.choice([1, -1]) * random.uniform(-1, -0.5)
    ayy = random.choice([1, -1]) * random.uniform(-1, -0.5)
    azy = random.choice([1, -1]) * random.uniform(-1, -0.5)


    z1 = np.random.normal(size=n)
    z2 = azz * z1 + np.random.normal(size=n)
    z3 = azz * z2 + np.random.normal(size=n)


    x1 = np.random.normal(size=n)
    x2 = axx * x1 + azx * z1 + np.random.normal(size=n)
    x3 = axx * x2 + azx * z2 + np.random.normal(size=n)

    y1 = np.random.normal(size=n)
    y2 = ayy*y1 + azy * z1 + np.random.normal(size=n)
    y3 = ayy*y2 + azy * z2 + np.random.normal(size=n)

    df = pd.DataFrame(
        {X3: x3,
         Y3: y3,
         Z3: z3
         }
    )

    ge = CompletedPartiallyDirectedAcyclicDifferenceGraph()
    ge.add_vertices([X3, Y3, Z3])
    ge.add_undirected_edge(Z3, X3)
    ge.add_undirected_edge(Z3, Y3)
    ge.add_undirected_edge(X3, Y3)

    gt = CompletedPartiallyDirectedAcyclicDifferenceGraph()
    gt.add_vertices([X3, Y3, Z3])
    gt.add_directed_edge(Z3, X3)
    gt.add_directed_edge(Z3, Y3)

    return df, ge, gt


def chain_structure(n):
    X1 = DTimeVar(f"X", 1)
    X2 = DTimeVar(f"X", 2)
    X3 = DTimeVar(f"X", 3)
    Y1 = DTimeVar(f"Y", 1)
    Y2 = DTimeVar(f"Y", 2)
    Y3 = DTimeVar(f"Y", 3)
    Z1 = DTimeVar(f"Z", 1)
    Z2 = DTimeVar(f"Z", 2)
    Z3 = DTimeVar(f"Z", 3)

    azz = random.choice([1, -1]) * random.uniform(-1, -0.5)
    axx = random.choice([1, -1]) * random.uniform(-1, -0.5)
    axz = random.choice([1, -1]) * random.uniform(-1, -0.5)
    ayy = random.choice([1, -1]) * random.uniform(-1, -0.5)
    azy = random.choice([1, -1]) * random.uniform(-1, -0.5)

    x1 = np.random.normal(size=n)
    x2 = axx * x1 + np.random.normal(size=n)
    x3 = axx * x2 + np.random.normal(size=n)

    z1 = np.random.normal(size=n)
    z2 = azz * z1 + axz * x1 + np.random.normal(size=n)
    z3 = azz * z2 + axz * x2 + np.random.normal(size=n)

    y1 = np.random.normal(size=n)
    y2 = ayy * y1 + azy * z1 + np.random.normal(size=n)
    y3 = ayy * y2 + azy * z2 + np.random.normal(size=n)

    df = pd.DataFrame(
        {X3: x3,
         Y3: y3,
         Z3: z3
         }
    )

    ge = CompletedPartiallyDirectedAcyclicDifferenceGraph()
    ge.add_vertices([X3, Y3, Z3])
    ge.add_undirected_edge(Z3, X3)
    ge.add_undirected_edge(Z3, Y3)
    ge.add_undirected_edge(X3, Y3)

    gt = CompletedPartiallyDirectedAcyclicDifferenceGraph()
    gt.add_vertices([X3, Y3, Z3])
    gt.add_directed_edge(Z3, X3)
    gt.add_directed_edge(Z3, Y3)
    return df, ge, gt

def v_structure(n):
    X1 = DTimeVar(f"X", 1)
    X2 = DTimeVar(f"X", 2)
    X3 = DTimeVar(f"X", 3)
    Y1 = DTimeVar(f"Y", 1)
    Y2 = DTimeVar(f"Y", 2)
    Y3 = DTimeVar(f"Y", 3)
    Z1 = DTimeVar(f"Z", 1)
    Z2 = DTimeVar(f"Z", 2)
    Z3 = DTimeVar(f"Z", 3)


    azz = random.choice([1, -1]) * random.uniform(0.5, 1)
    axx = random.choice([1, -1]) * random.uniform(0.5, 1)
    axz = random.choice([1, -1]) * random.uniform(0.5, 1)
    ayy = random.choice([1, -1]) * random.uniform(0.5, 1)
    ayz = random.choice([1, -1]) * random.uniform(0.5, 1)

    x1 = np.random.normal(size=n)
    x2 = axx * x1 + np.random.normal(size=n)
    x3 = axx * x2 + np.random.normal(size=n)

    y1 = np.random.normal(size=n)
    y2 = ayy * y1 + np.random.normal(size=n)
    y3 = ayy * y2 + np.random.normal(size=n)

    z1 = np.random.normal(size=n)
    z2 = azz * z1 + axz * x1 + ayz * y1 + np.random.normal(size=n)
    z3 = azz * z2 + axz * x1 + ayz * y1 + np.random.normal(size=n)

    df = pd.DataFrame(
        {X3: x3,
         Y3: y3,
         Z3: z3
         }
    )

    ge = CompletedPartiallyDirectedAcyclicDifferenceGraph()
    ge.add_vertices([X3, Y3, Z3])
    ge.add_directed_edge(X3, Z3)
    ge.add_directed_edge(Y3, Z3)

    gt = CompletedPartiallyDirectedAcyclicDifferenceGraph()
    gt.add_vertices([X3, Y3, Z3])
    gt.add_directed_edge(X3, Z3)
    gt.add_directed_edge(Y3, Z3)
    return df, ge, gt


def chain_v_structure(n):
    X1 = DTimeVar(f"X", 1)
    X2 = DTimeVar(f"X", 2)
    X3 = DTimeVar(f"X", 3)
    Y1 = DTimeVar(f"Y", 1)
    Y2 = DTimeVar(f"Y", 2)
    Y3 = DTimeVar(f"Y", 3)
    Z1 = DTimeVar(f"Z", 1)
    Z2 = DTimeVar(f"Z", 2)
    Z3 = DTimeVar(f"Z", 3)

    U1 = DTimeVar(f"U", 1)
    U2 = DTimeVar(f"U", 2)
    U3 = DTimeVar(f"U", 3)
    W1 = DTimeVar(f"W", 1)
    W2 = DTimeVar(f"W", 2)
    W3 = DTimeVar(f"W", 3)



    azz = random.choice([1, -1]) * random.uniform(-1, -0.5)
    axx = random.choice([1, -1]) * random.uniform(-1, -0.5)
    axz = random.choice([1, -1]) * random.uniform(-1, -0.5)
    ayy = random.choice([1, -1]) * random.uniform(-1, -0.5)
    ayz = random.choice([1, -1]) * random.uniform(-1, -0.5)

    auu = random.choice([1, -1]) * random.uniform(-1, -0.5)
    aux = random.choice([1, -1]) * random.uniform(-1, -0.5)
    aww = random.choice([1, -1]) * random.uniform(-1, -0.5)
    awy = random.choice([1, -1]) * random.uniform(-1, -0.5)

    u1 = np.random.normal(size=n)
    u2 = auu * u1 + np.random.normal(size=n)
    u3 = auu * u2 + np.random.normal(size=n)
    w1 = np.random.normal(size=n)
    w2 = aww * w1 + np.random.normal(size=n)
    w3 = aww * w2 + np.random.normal(size=n)

    x1 = np.random.normal(size=n)
    x2 = axx * x1 + aux * u1 + np.random.normal(size=n)
    x3 = axx * x2 + aux * u2 + np.random.normal(size=n)

    y1 = np.random.normal(size=n)
    y2 = ayy * y1 + awy * w1 + np.random.normal(size=n)
    y3 = ayy * y2 + awy * w2 + np.random.normal(size=n)

    z1 = np.random.normal(size=n)
    z2 = azz * z1 + axz * x1 + ayz * y1 + np.random.normal(size=n)
    z3 = azz * z2 + axz * x2 + ayz * y2 + np.random.normal(size=n)

    df = pd.DataFrame(
        {X3: x3,
         Y3: y3,
         Z3: z3,
         U3: u3,
         W3: w3
         }
    )

    ge = CompletedPartiallyDirectedAcyclicDifferenceGraph()
    ge.add_vertices([X3, Y3, Z3, U3, W3])
    ge.add_directed_edge(U3, X3)
    ge.add_directed_edge(U3, Z3)
    ge.add_directed_edge(W3, Y3)
    ge.add_directed_edge(W3, Z3)
    ge.add_directed_edge(X3, Z3)
    ge.add_directed_edge(Y3, Z3)
    ge.add_undirected_edge(U3, X3)
    ge.add_undirected_edge(W3, Y3)

    gt = CompletedPartiallyDirectedAcyclicDifferenceGraph()
    gt.add_vertices([X3, Y3, Z3])
    gt.add_directed_edge(X3, Z3)
    gt.add_directed_edge(Y3, Z3)
    gt.add_directed_edge(U3, X3)
    gt.add_directed_edge(W3, Y3)
    return df, ge, gt


if __name__ == '__main__':
    structure = "chain_v"
    n_sample = 5000

    list_pc_f1_a_t = []
    list_pc_f1_o_t = []
    list_pc_f1_a_e = []
    list_pc_f1_o_e = []

    list_fci_f1_a_t = []
    list_fci_f1_o_t = []
    list_fci_f1_a_e = []
    list_fci_f1_o_e = []

    list_restpc_f1_a_t = []
    list_restpc_f1_o_t = []
    list_restpc_f1_a_e = []
    list_restpc_f1_o_e = []

    for iter in range(1000):
        print("####### iter ", iter, " #######")
        if structure == "fork":
            data, ge, gt = fork_structure(n_sample)
        elif structure == "v":
            data, ge, gt = v_structure(n_sample)
        elif structure == "chain":
            data, ge, gt = chain_structure(n_sample)
        elif structure == "chain_v":
            data, ge, gt = chain_v_structure(n_sample)

        pc = PC(sparsity=0.05, ci_test=FisherZTest)
        pc.run(data)
        fci = FCI(sparsity=0.05, ci_test=FisherZTest)
        fci.run(data)
        restpc = RestPC(sparsity=0.05, ci_test=FisherZTest)
        restpc.run(data)

        ft_pc_a = f1_score_a(gt, pc.g_hat)
        list_pc_f1_a_t.append(ft_pc_a)
        ft_pc_o = f1_score_o(gt, pc.g_hat)
        list_pc_f1_o_t.append(ft_pc_o)

        fe_pc_a = f1_score_a(ge, pc.g_hat)
        list_pc_f1_a_e.append(fe_pc_a)
        fe_pc_o = f1_score_o(ge, pc.g_hat)
        list_pc_f1_o_e.append(fe_pc_o)


        ft_fci_a = f1_score_a(gt, fci.g_hat)
        list_fci_f1_a_t.append(ft_fci_a)
        ft_fci_o = f1_score_o(gt, fci.g_hat)
        list_fci_f1_o_t.append(ft_fci_o)

        fe_fci_a = f1_score_a(ge, fci.g_hat)
        list_fci_f1_a_e.append(fe_fci_a)
        fe_fci_o = f1_score_o(ge, fci.g_hat)
        list_fci_f1_o_e.append(fe_fci_o)


        ft_restpc_a = f1_score_a(gt, restpc.g_hat)
        list_restpc_f1_a_t.append(ft_restpc_a)
        ft_restpc_o = f1_score_o(gt, restpc.g_hat)
        list_restpc_f1_o_t.append(ft_restpc_o)

        fe_restpc_a = f1_score_a(ge, restpc.g_hat)
        list_restpc_f1_a_e.append(fe_restpc_a)
        fe_restpc_o = f1_score_o(ge, restpc.g_hat)
        list_restpc_f1_o_e.append(fe_restpc_o)

        print(ft_pc_o, fe_pc_o)
        print(ft_restpc_o, fe_restpc_o)

    print("################")

    print("PC")
    print("F1-t-a", np.mean(list_pc_f1_a_t), np.var(list_pc_f1_a_t))
    print("F1-t-o", np.mean(list_pc_f1_o_t), np.var(list_pc_f1_o_t))
    print("F1-e-a", np.mean(list_pc_f1_a_e), np.var(list_pc_f1_a_e))
    print("F1-e-o", np.mean(list_pc_f1_o_e), np.var(list_pc_f1_o_e))

    print("FCI")
    print("F1-t-a", np.mean(list_fci_f1_a_t), np.var(list_fci_f1_a_t))
    print("F1-t-o", np.mean(list_fci_f1_o_t), np.var(list_fci_f1_o_t))
    print("F1-e-a", np.mean(list_fci_f1_a_e), np.var(list_fci_f1_a_e))
    print("F1-e-o", np.mean(list_fci_f1_o_e), np.var(list_fci_f1_o_e))

    print("RestPC")
    print("F1-t-a", np.mean(list_restpc_f1_a_t), np.var(list_restpc_f1_a_t))
    print("F1-t-o", np.mean(list_restpc_f1_o_t), np.var(list_restpc_f1_o_t))
    print("F1-e-a", np.mean(list_restpc_f1_a_e), np.var(list_restpc_f1_a_e))
    print("F1-e-o", np.mean(list_restpc_f1_o_e), np.var(list_restpc_f1_o_e))
