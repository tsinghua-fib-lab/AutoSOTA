from typing import Set, Tuple, FrozenSet, Iterable
import numpy as np
import pandas as pd
import random

from pyciphod.utils.graphs.partially_specified_graphs import CompletedPartiallyDirectedAcyclicDifferenceGraph
from pyciphod.causal_discovery.basic.constraint_based import PC, RestPC
from pyciphod.utils.stat_tests.independence_tests import KernelPartialCorrelationTest, CopulaTest, CIMhTest



def f1_score_o(g, g_hat):
    true_set = g.get_directed_edges()
    pred_set = g_hat.get_directed_edges()
    TP = len(pred_set & true_set)
    FP = len(pred_set - true_set)
    FN = len(true_set - pred_set)
    prec = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    rec = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    if prec + rec == 0.0:
        return 0.0
    return 2 * (prec * rec) / (prec + rec), prec, rec


def f1_score_a(g, g_hat):
    dir= g.get_directed_edges()
    undir = g.get_undirected_edges()
    true_set = dir.union(undir)
    dir= g_hat.get_directed_edges()
    undir = g_hat.get_undirected_edges()
    pred_set = dir.union(undir)

    TP = set()
    for p in list(true_set):
        if p in list(pred_set):
            if p not in TP:
                TP.add(p)

    FP = set()
    for p in list(pred_set):
        if p not in list(true_set):
            if p not in FP:
                TP.add(p)

    FN = set()
    for p in list(true_set):
        if p not in list(pred_set):
            if p not in FP:
                FN.add(p)
    TP = len(TP)
    FP = len(FP)
    FN = len(FN)

    prec = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    rec = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    if prec + rec == 0.0:
        return 0.0
    return 2 * (prec * rec) / (prec + rec), prec, rec



def fork_structure(n):
    X3 = "X"
    Y3 = "Y"
    Z3 = "Z"

    azx = random.choice([1, -1]) * random.uniform(-1, -0.5)
    azy = random.choice([1, -1]) * random.uniform(-1, -0.5)

    z3 = np.random.normal(size=n)
    x3 = azx * z3 + np.random.normal(size=n)
    y3 = azy * z3 + np.random.normal(size=n)

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
    X3 ="X"
    Y3 = "Y"
    Z3 = "Z"

    axz = random.choice([1, -1]) * random.uniform(-1, -0.5)
    azy = random.choice([1, -1]) * random.uniform(-1, -0.5)

    x3 = np.random.normal(size=n)
    z3 = axz * x3 + np.random.normal(size=n)
    y3 = azy * z3 + np.random.normal(size=n)

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
    return df, gt, ge

def v_structure(n):
    X3 ="X"
    Y3 = "Y"
    Z3 = "Z"

    axz = random.choice([1, -1]) * random.uniform(-1, -0.5)
    ayz = random.choice([1, -1]) * random.uniform(-1, -0.5)

    x3 = np.random.normal(size=n)
    y3 = np.random.normal(size=n)
    z3 = axz * x3 + ayz * y3 + np.random.normal(size=n)

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
    X3 ="X"
    Y3 = "Y"
    Z3 = "Z"
    U3 = "U"
    W3 = "W"

    axz = random.choice([1, -1]) * random.uniform(-1, -0.5)
    ayz = random.choice([1, -1]) * random.uniform(-1, -0.5)
    aux = random.choice([1, -1]) * random.uniform(-1, -0.5)
    awy = random.choice([1, -1]) * random.uniform(-1, -0.5)

    u3 = np.random.normal(size=n)
    w3 = np.random.normal(size=n)
    x3 = aux * u3 + np.random.normal(size=n)
    y3 = awy * w3 + np.random.normal(size=n)
    z3 = axz * x3 + ayz * y3 + np.random.normal(size=n)

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

    gt = CompletedPartiallyDirectedAcyclicDifferenceGraph()
    gt.add_vertices([X3, Y3, Z3])
    gt.add_directed_edge(X3, Z3)
    gt.add_directed_edge(Y3, Z3)
    ge.add_directed_edge(U3, X3)
    ge.add_directed_edge(W3, Y3)
    return df, ge, gt




def graph_10(n):
    X3 ="X"
    Y3 = "Y"
    Z3 = "Z"
    U3 = "U"
    W3 = "W"

    V1 = "V1"
    V2 = "V2"
    V3 = "V3"
    V4 = "V4"
    V5 = "V5"



    axz = random.choice([1, -1]) * random.uniform(-1, -0.5)
    ayz = random.choice([1, -1]) * random.uniform(-1, -0.5)
    aux = random.choice([1, -1]) * random.uniform(-1, -0.5)
    awy = random.choice([1, -1]) * random.uniform(-1, -0.5)

    u3 = np.random.normal(size=n)
    w3 = np.random.normal(size=n)
    x3 = aux * u3 + np.random.normal(size=n)
    y3 = awy * w3 + np.random.normal(size=n)
    z3 = axz * x3 + ayz * y3 + np.random.normal(size=n)


    v1 = np.random.normal(size=n)
    v2 = np.random.normal(size=n)
    v3 = np.random.normal(size=n)

    v4 = np.random.normal(size=n)
    v5 = np.random.normal(size=n)



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

    gt = CompletedPartiallyDirectedAcyclicDifferenceGraph()
    gt.add_vertices([X3, Y3, Z3])
    gt.add_directed_edge(X3, Z3)
    gt.add_directed_edge(Y3, Z3)
    ge.add_directed_edge(U3, X3)
    ge.add_directed_edge(W3, Y3)
    return df, ge, gt


def generate_graph_with_min_two_independent_parents(n):
    1

    return 1


if __name__ == '__main__':
    structure = "chain_v"
    n_sample = 1000

    list_pc_f1_a_t = []
    list_pc_f1_o_t = []
    list_pc_f1_a_e = []
    list_pc_f1_o_e = []

    list_restpc_f1_a_t = []
    list_restpc_f1_o_t = []
    list_restpc_f1_a_e = []
    list_restpc_f1_o_e = []

    for iter in range(10):
        print("####### iter ", iter, " #######")
        if structure == "fork":
            data, ge, gt = fork_structure(n_sample)
        elif structure == "v":
            data, ge, gt = v_structure(n_sample)
        elif structure == "chain":
            data, ge, gt = chain_structure(n_sample)
        elif structure == "chain_v":
            data, ge, gt = chain_v_structure(n_sample)

        pc = PC(sparsity=0.05, ci_test=CIMhTest)
        pc.run(data)
        restpc = RestPC(sparsity=0.05, ci_test=CIMhTest)
        restpc.run(data)

        _, _, ft_pc_a = f1_score_a(gt, pc.g_hat)
        list_pc_f1_a_t.append(ft_pc_a)
        _, _, ft_pc_o = f1_score_o(gt, pc.g_hat)
        list_pc_f1_o_t.append(ft_pc_o)

        _, _, fe_pc_a = f1_score_a(ge, pc.g_hat)
        list_pc_f1_a_e.append(fe_pc_a)
        _, _, fe_pc_o = f1_score_o(ge, pc.g_hat)
        list_pc_f1_o_e.append(fe_pc_o)

        _, _, ft_restpc_a = f1_score_a(gt, restpc.g_hat)
        list_restpc_f1_a_t.append(ft_restpc_a)
        _, _, ft_restpc_o = f1_score_o(gt, restpc.g_hat)
        list_restpc_f1_o_t.append(ft_restpc_o)

        _, _, fe_restpc_a = f1_score_a(ge, restpc.g_hat)
        list_restpc_f1_a_e.append(fe_restpc_a)
        print(list_pc_f1_a_e)
        _, _, fe_restpc_o = f1_score_o(ge, restpc.g_hat)
        list_restpc_f1_o_e.append(fe_restpc_o)

        print(ft_pc_o, fe_pc_o)
        print(ft_restpc_o, fe_restpc_o)

    print("################")

    print("PC")
    print("Recall-t-a", np.mean(list_pc_f1_a_t), np.var(list_pc_f1_a_t))
    print("Recall-t-o", np.mean(list_pc_f1_o_t), np.var(list_pc_f1_o_t))
    print("Recall-e-a", np.mean(list_pc_f1_a_e), np.var(list_pc_f1_a_e))
    print("Recall-e-o", np.mean(list_pc_f1_o_e), np.var(list_pc_f1_o_e))

    print("RestPC")
    print("Recall-t-a", np.mean(list_restpc_f1_a_t), np.var(list_restpc_f1_a_t))
    print("Recall-t-o", np.mean(list_restpc_f1_o_t), np.var(list_restpc_f1_o_t))
    print("Recall-e-a", np.mean(list_restpc_f1_a_e), np.var(list_restpc_f1_a_e))
    print("Recall-e-o", np.mean(list_restpc_f1_o_e), np.var(list_restpc_f1_o_e))
