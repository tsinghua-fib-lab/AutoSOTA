from typing import Set, Tuple, FrozenSet, Iterable
import numpy as np
import pandas as pd
import random

from pyciphod.utils.scms.dynamic_scm import create_random_linear_dt_dynamic_scm, DtDynamicSCM, create_random_linear_dt_dynamic_scm_from_ftadmg
from pyciphod.utils.time_series.data_format import DTimeVar
from pyciphod.utils.graphs.partially_specified_graphs import CompletedPartiallyDirectedAcyclicDifferenceGraph
from pyciphod.causal_discovery.basic.constraint_based import PC, RestPC, FCI
from pyciphod.utils.graphs.temporal_graphs import create_random_ft_dag, FtDirectedAcyclicGraph
from pyciphod.utils.stat_tests.independence_tests import LinearRegressionCoefficientTTest, FisherZTest, CopulaTest, CIMhTest, KernelPartialCorrelationTest
from pyciphod.utils.stat_tests.dependency_measures import Copula, compute_copula_fit

import time

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



# def chain_v_structure_1(n, name_prefix=""):
#     X1 = DTimeVar(f"{name_prefix}X", 1)
#     X2 = DTimeVar(f"{name_prefix}X", 2)
#     X3 = DTimeVar(f"{name_prefix}X", 3)
#     Y1 = DTimeVar(f"{name_prefix}Y", 1)
#     Y2 = DTimeVar(f"{name_prefix}Y", 2)
#     Y3 = DTimeVar(f"{name_prefix}Y", 3)
#     Z1 = DTimeVar(f"{name_prefix}Z", 1)
#     Z2 = DTimeVar(f"{name_prefix}Z", 2)
#     Z3 = DTimeVar(f"{name_prefix}Z", 3)
#
#     U1 = DTimeVar(f"{name_prefix}U", 1)
#     U2 = DTimeVar(f"{name_prefix}U", 2)
#     U3 = DTimeVar(f"{name_prefix}U", 3)
#     W1 = DTimeVar(f"{name_prefix}W", 1)
#     W2 = DTimeVar(f"{name_prefix}W", 2)
#     W3 = DTimeVar(f"{name_prefix}W", 3)
#
#     D1 = DTimeVar(f"{name_prefix}D1", 3)
#     D2 = DTimeVar(f"{name_prefix}D2", 3)
#     D3 = DTimeVar(f"{name_prefix}D3", 3)
#     D4 = DTimeVar(f"{name_prefix}D4", 3)
#     D5 = DTimeVar(f"{name_prefix}D5", 3)
#
#
#     azz = random.choice([1, -1]) * random.uniform(-1, -0.5)
#     axx = random.choice([1, -1]) * random.uniform(-1, -0.5)
#     axz = random.choice([1, -1]) * random.uniform(-1, -0.5)
#     ayy = random.choice([1, -1]) * random.uniform(-1, -0.5)
#     ayz = random.choice([1, -1]) * random.uniform(-1, -0.5)
#     add = random.choice([1, -1]) * random.uniform(-1, -0.5)
#     azd = random.choice([1, -1]) * random.uniform(-1, -0.5)
#
#
#     auu = random.choice([1, -1]) * random.uniform(-1, -0.5)
#     aux = random.choice([1, -1]) * random.uniform(-1, -0.5)
#     aww = random.choice([1, -1]) * random.uniform(-1, -0.5)
#     awy = random.choice([1, -1]) * random.uniform(-1, -0.5)
#
#     u1 = np.random.normal(size=n)
#     u2 = auu * u1 + np.random.normal(size=n)
#     u3 = auu * u2 + np.random.normal(size=n)
#     w1 = np.random.normal(size=n)
#     w2 = aww * w1 + np.random.normal(size=n)
#     w3 = aww * w2 + np.random.normal(size=n)
#
#     x1 = np.random.normal(size=n)
#     x2 = axx * x1 + aux * u1 + np.random.normal(size=n)
#     x3 = axx * x2 + aux * u2 + np.random.normal(size=n)
#
#     y1 = np.random.normal(size=n)
#     y2 = ayy * y1 + awy * w1 + np.random.normal(size=n)
#     y3 = ayy * y2 + awy * w2 + np.random.normal(size=n)
#
#     z1 = np.random.normal(size=n)
#     z2 = azz * z1 + axz * x1 + ayz * y1 + np.random.normal(size=n)
#     z3 = azz * z2 + axz * x2 + ayz * y2 + np.random.normal(size=n)
#
#     d_1_1 = np.random.normal(size=n)
#     d_1_2 = add * d_1_1 + azd * z1 + np.random.normal(size=n)
#     d_1_3 = add * d_1_2 + azd * z2 + np.random.normal(size=n)
#
#     d_2_1 = np.random.normal(size=n)
#     d_2_2 = add * d_2_1 + azd * z1 + np.random.normal(size=n)
#     d_2_3 = add * d_2_2 + azd * z2 + np.random.normal(size=n)
#
#     d_3_1 = np.random.normal(size=n)
#     d_3_2 = add * d_3_1 + azd * z1 + np.random.normal(size=n)
#     d_3_3 = add * d_3_2 + azd * z2 + np.random.normal(size=n)
#
#     d_4_1 = np.random.normal(size=n)
#     d_4_2 = add * d_4_1 + azd * z1 + np.random.normal(size=n)
#     d_4_3 = add * d_4_2 + azd * z2 + np.random.normal(size=n)
#
#     d_5_1 = np.random.normal(size=n)
#     d_5_2 = add * d_5_1 + azd * z1 + np.random.normal(size=n)
#     d_5_3 = add * d_5_2 + azd * z2 + np.random.normal(size=n)
#
#
#     df = pd.DataFrame(
#         {X3: x3,
#          Y3: y3,
#          Z3: z3,
#          U3: u3,
#          W3: w3,
#          D1: d_1_3,
#          D2: d_2_3,
#          D3: d_3_3,
#          D4: d_4_3,
#          D5: d_5_3}
#     )
#
#     ge = CompletedPartiallyDirectedAcyclicDifferenceGraph()
#     ge.add_vertices([X3, Y3, Z3, U3, W3, D1, D2, D3, D4, D5])
#     ge.add_directed_edge(U3, X3)
#     ge.add_directed_edge(U3, Z3)
#     ge.add_directed_edge(W3, Y3)
#     ge.add_directed_edge(W3, Z3)
#     ge.add_directed_edge(X3, Z3)
#     ge.add_directed_edge(Y3, Z3)
#     ge.add_undirected_edge(U3, X3)
#     ge.add_undirected_edge(W3, Y3)
#
#
#     ge.add_directed_edge(U3, D1)
#     ge.add_directed_edge(W3, D1)
#     ge.add_directed_edge(X3, D1)
#     ge.add_directed_edge(Y3, D1)
#
#     ge.add_directed_edge(U3, D2)
#     ge.add_directed_edge(W3, D2)
#     ge.add_directed_edge(X3, D2)
#     ge.add_directed_edge(Y3, D2)
#
#     ge.add_directed_edge(U3, D3)
#     ge.add_directed_edge(W3, D3)
#     ge.add_directed_edge(X3, D3)
#     ge.add_directed_edge(Y3, D3)
#
#     ge.add_directed_edge(U3, D4)
#     ge.add_directed_edge(W3, D4)
#     ge.add_directed_edge(X3, D4)
#     ge.add_directed_edge(Y3, D4)
#
#     ge.add_directed_edge(U3, D5)
#     ge.add_directed_edge(W3, D5)
#     ge.add_directed_edge(X3, D5)
#     ge.add_directed_edge(Y3, D5)
#
#     ge.add_undirected_edge(Z3, D1)
#     ge.add_undirected_edge(Z3, D2)
#     ge.add_undirected_edge(Z3, D3)
#     ge.add_undirected_edge(Z3, D4)
#     ge.add_undirected_edge(Z3, D5)
#     ge.add_undirected_edge(D1, D2)
#     ge.add_undirected_edge(D1, D3)
#     ge.add_undirected_edge(D1, D4)
#     ge.add_undirected_edge(D1, D5)
#     ge.add_undirected_edge(D2, D3)
#     ge.add_undirected_edge(D2, D4)
#     ge.add_undirected_edge(D2, D5)
#     ge.add_undirected_edge(D3, D4)
#     ge.add_undirected_edge(D3, D5)
#     ge.add_undirected_edge(D4, D5)
#
#
#     gt = CompletedPartiallyDirectedAcyclicDifferenceGraph()
#     gt.add_vertices([X3, Y3, Z3, U3, W3, D1, D2, D3, D4, D5])
#     gt.add_directed_edge(X3, Z3)
#     gt.add_directed_edge(Y3, Z3)
#     gt.add_directed_edge(U3, X3)
#     gt.add_directed_edge(W3, Y3)
#
#
#     gt.add_directed_edge(Z3, D1)
#     gt.add_directed_edge(Z3, D2)
#     gt.add_directed_edge(Z3, D3)
#     gt.add_directed_edge(Z3, D4)
#     gt.add_directed_edge(Z3, D5)
#
#     return df, ge, gt


def chain_v_structure_10(n, name_prefix=""):
    X5 = DTimeVar(f"{name_prefix}X", 5)
    Y5 = DTimeVar(f"{name_prefix}Y", 5)
    Z5 = DTimeVar(f"{name_prefix}Z", 5)

    U5 = DTimeVar(f"{name_prefix}U", 5)
    W5 = DTimeVar(f"{name_prefix}W", 5)

    D1 = DTimeVar(f"{name_prefix}D1", 5)
    D2 = DTimeVar(f"{name_prefix}D2", 5)
    D3 = DTimeVar(f"{name_prefix}D3", 5)

    V1 = DTimeVar(f"{name_prefix}V1", 5)
    V2 = DTimeVar(f"{name_prefix}V2", 5)


    azz = random.choice([1, -1]) * random.uniform(-1, -0.5)
    axx = random.choice([1, -1]) * random.uniform(-1, -0.5)
    axz = random.choice([1, -1]) * random.uniform(-1, -0.5)
    ayy = random.choice([1, -1]) * random.uniform(-1, -0.5)
    ayz = random.choice([1, -1]) * random.uniform(-1, -0.5)
    add1 = random.choice([1, -1]) * random.uniform(-1, -0.5)
    azd1 = random.choice([1, -1]) * random.uniform(-1, -0.5)
    add2 = random.choice([1, -1]) * random.uniform(-1, -0.5)
    azd2= random.choice([1, -1]) * random.uniform(-1, -0.5)
    add3 = random.choice([1, -1]) * random.uniform(-1, -0.5)
    azd3 = random.choice([1, -1]) * random.uniform(-1, -0.5)
    avv1 = random.choice([1, -1]) * random.uniform(-1, -0.5)
    avv2 = random.choice([1, -1]) * random.uniform(-1, -0.5)
    azv1 = random.choice([1, -1]) * random.uniform(-1, -0.5)
    azv2 = random.choice([1, -1]) * random.uniform(-1, -0.5)


    auu = random.choice([1, -1]) * random.uniform(-1, -0.5)
    aux = random.choice([1, -1]) * random.uniform(-1, -0.5)
    aww = random.choice([1, -1]) * random.uniform(-1, -0.5)
    awy = random.choice([1, -1]) * random.uniform(-1, -0.5)

    u0 = np.random.normal(size=n)
    u1 = auu * u0 + np.random.normal(size=n)
    u2 = auu * u1 + np.random.normal(size=n)
    u3 = auu * u2 + np.random.normal(size=n)
    u4 = auu * u3 + np.random.normal(size=n)
    u5 = auu * u4 + np.random.normal(size=n)

    w0 = np.random.normal(size=n)
    w1 = aww * w0 + np.random.normal(size=n)
    w2 = aww * w1 + np.random.normal(size=n)
    w3 = aww * w2 + np.random.normal(size=n)
    w4 = aww * w3 + np.random.normal(size=n)
    w5 = aww * w4 + np.random.normal(size=n)

    x0 = np.random.normal(size=n)
    x1 = axx * x0 + aux * u0 + np.random.normal(size=n)
    x2 = axx * x1 + aux * u1 + np.random.normal(size=n)
    x3 = axx * x2 + aux * u2 + np.random.normal(size=n)
    x4 = axx * x3 + aux * u3 + np.random.normal(size=n)
    x5 = axx * x4 + aux * u4 + np.random.normal(size=n)

    y0 = np.random.normal(size=n)
    y1 = ayy * y0 + awy * w0 + np.random.normal(size=n)
    y2 = ayy * y1 + awy * w1 + np.random.normal(size=n)
    y3 = ayy * y2 + awy * w2 + np.random.normal(size=n)
    y4 = ayy * y3 + awy * w3 + np.random.normal(size=n)
    y5 = ayy * y4 + awy * w4 + np.random.normal(size=n)

    z0 = np.random.normal(size=n)
    z1 = azz * z0 + axz * x0 + ayz * y0 + np.random.normal(size=n)
    z2 = azz * z1 + axz * x1 + ayz * y1 + np.random.normal(size=n)
    z3 = azz * z2 + axz * x2 + ayz * y2 + np.random.normal(size=n)
    z4 = azz * z3 + axz * x3 + ayz * y3 + np.random.normal(size=n)
    z5 = azz * z4 + axz * x4 + ayz * y4 + np.random.normal(size=n)

    d_1_0 = np.random.normal(size=n)
    d_1_1 = add1 * d_1_0 + azd1 * z0 + np.random.normal(size=n)
    d_1_2 = add1 * d_1_1 + azd1 * z1 + np.random.normal(size=n)
    d_1_3 = add1 * d_1_2 + azd1 * z2 + np.random.normal(size=n)
    d_1_4 = add1 * d_1_3 + azd1 * z3 + np.random.normal(size=n)
    d_1_5 = add1 * d_1_4 + azd1 * z4 + np.random.normal(size=n)

    d_2_0 = np.random.normal(size=n)
    d_2_1 = add2 * d_2_0 + azd2 * z0 + np.random.normal(size=n)
    d_2_2 = add2 * d_2_1 + azd2 * z1 + np.random.normal(size=n)
    d_2_3 = add2 * d_2_2 + azd2 * z2 + np.random.normal(size=n)
    d_2_4 = add2 * d_2_3 + azd2 * z3 + np.random.normal(size=n)
    d_2_5 = add2 * d_2_4 + azd2 * z4 + np.random.normal(size=n)

    d_3_0 = np.random.normal(size=n)
    d_3_1 = add3 * d_3_0 + azd3 * z0 + np.random.normal(size=n)
    d_3_2 = add3 * d_3_1 + azd3 * z1 + np.random.normal(size=n)
    d_3_3 = add3 * d_3_2 + azd3 * z2 + np.random.normal(size=n)
    d_3_4 = add3 * d_3_3 + azd3 * z3 + np.random.normal(size=n)
    d_3_5 = add3 * d_3_4 + azd3 * z4 + np.random.normal(size=n)

    v_1_0 = np.random.normal(size=n)
    v_1_1 = avv1 * v_1_0 + azv1 * x0 + np.random.normal(size=n)
    v_1_2 = avv1 * v_1_1 + azv1 * x1 + np.random.normal(size=n)
    v_1_3 = avv1 * v_1_2 + azv1 * x2 + np.random.normal(size=n)
    v_1_4 = avv1 * v_1_3 + azv1 * x3 + np.random.normal(size=n)
    v_1_5 = avv1 * v_1_4 + azv1 * x4 + np.random.normal(size=n)

    v_2_0 = np.random.normal(size=n)
    v_2_1 = avv2 * v_2_0 + azv2 * y0 + np.random.normal(size=n)
    v_2_2 = avv2 * v_2_1 + azv2 * y1 + np.random.normal(size=n)
    v_2_3 = avv2 * v_2_2 + azv2 * y2 + np.random.normal(size=n)
    v_2_4 = avv2 * v_2_3 + azv2 * y3 + np.random.normal(size=n)
    v_2_5 = avv2 * v_2_4 + azv2 * y4 + np.random.normal(size=n)

    df = pd.DataFrame(
        {X5: x5,
         Y5: y5,
         Z5: z5,
         U5: u5,
         W5: w5,
         D1: d_1_5,
         D2: d_2_5,
         D3: d_3_5,
         V1: v_1_5,
         V2: v_2_5}
    )

    ge = CompletedPartiallyDirectedAcyclicDifferenceGraph()
    ge.add_vertices([X5, Y5, Z5, U5, W5, D1, D2, D3, V1, V2])
    ge.add_directed_edge(U5, X5)
    ge.add_directed_edge(U5, Z5)
    ge.add_directed_edge(W5, Y5)
    ge.add_directed_edge(W5, Z5)
    ge.add_directed_edge(X5, Z5)
    ge.add_directed_edge(Y5, Z5)
    ge.add_undirected_edge(U5, X5)
    ge.add_undirected_edge(W5, Y5)

    ge.add_undirected_edge(U5, V1)
    ge.add_undirected_edge(X5, V1)
    ge.add_undirected_edge(W5, V2)
    ge.add_undirected_edge(Y5, V2)

    ge.add_directed_edge(U5, D1)
    ge.add_directed_edge(W5, D1)
    ge.add_directed_edge(X5, D1)
    ge.add_directed_edge(Y5, D1)
    ge.add_directed_edge(V1, D1)
    ge.add_directed_edge(V2, D1)

    ge.add_directed_edge(U5, D2)
    ge.add_directed_edge(W5, D2)
    ge.add_directed_edge(X5, D2)
    ge.add_directed_edge(Y5, D2)
    ge.add_directed_edge(V1, D1)
    ge.add_directed_edge(V2, D1)

    ge.add_directed_edge(U5, D3)
    ge.add_directed_edge(W5, D3)
    ge.add_directed_edge(X5, D3)
    ge.add_directed_edge(Y5, D3)
    ge.add_directed_edge(V1, D1)
    ge.add_directed_edge(V2, D1)

    ge.add_undirected_edge(Z5, D1)
    ge.add_undirected_edge(Z5, D2)
    ge.add_undirected_edge(Z5, D3)
    ge.add_undirected_edge(D1, D2)
    ge.add_undirected_edge(D1, D3)
    ge.add_undirected_edge(D2, D3)


    gt = CompletedPartiallyDirectedAcyclicDifferenceGraph()
    gt.add_vertices([X5, Y5, Z5, U5, W5, D1, D2, D3, V1, V2])
    gt.add_directed_edge(X5, Z5)
    gt.add_directed_edge(Y5, Z5)
    gt.add_directed_edge(U5, X5)
    gt.add_directed_edge(W5, Y5)


    gt.add_directed_edge(Z5, D1)
    gt.add_directed_edge(Z5, D2)
    gt.add_directed_edge(Z5, D3)
    gt.add_directed_edge(X5, V1)
    gt.add_directed_edge(Y5, V2)

    return df, ge, gt



def chain_v_structure_20(n, name_prefix=""):
    X5 = DTimeVar(f"{name_prefix}X", 5)
    Y5 = DTimeVar(f"{name_prefix}Y", 5)
    Z5 = DTimeVar(f"{name_prefix}Z", 5)

    U5 = DTimeVar(f"{name_prefix}U", 5)
    W5 = DTimeVar(f"{name_prefix}W", 5)

    D1 = DTimeVar(f"{name_prefix}D1", 5)
    D2 = DTimeVar(f"{name_prefix}D2", 5)
    D3 = DTimeVar(f"{name_prefix}D3", 5)

    D4 = DTimeVar(f"{name_prefix}D4", 5)
    D5 = DTimeVar(f"{name_prefix}D5", 5)
    D6 = DTimeVar(f"{name_prefix}D6", 5)
    D7 = DTimeVar(f"{name_prefix}D7", 5)
    D8 = DTimeVar(f"{name_prefix}D8", 5)
    D9 = DTimeVar(f"{name_prefix}D9", 5)


    V1 = DTimeVar(f"{name_prefix}V1", 5)
    V2 = DTimeVar(f"{name_prefix}V2", 5)

    V3 = DTimeVar(f"{name_prefix}V3", 5)
    V4 = DTimeVar(f"{name_prefix}V4", 5)
    V5 = DTimeVar(f"{name_prefix}V5", 5)
    V6 = DTimeVar(f"{name_prefix}V6", 5)


    azz = random.choice([1, -1]) * random.uniform(-1, -0.5)
    axx = random.choice([1, -1]) * random.uniform(-1, -0.5)
    axz = random.choice([1, -1]) * random.uniform(-1, -0.5)
    ayy = random.choice([1, -1]) * random.uniform(-1, -0.5)
    ayz = random.choice([1, -1]) * random.uniform(-1, -0.5)
    add1 = random.choice([1, -1]) * random.uniform(-1, -0.5)
    azd1 = random.choice([1, -1]) * random.uniform(-1, -0.5)
    add2 = random.choice([1, -1]) * random.uniform(-1, -0.5)
    azd2= random.choice([1, -1]) * random.uniform(-1, -0.5)
    add3 = random.choice([1, -1]) * random.uniform(-1, -0.5)
    azd3 = random.choice([1, -1]) * random.uniform(-1, -0.5)
    avv1 = random.choice([1, -1]) * random.uniform(-1, -0.5)
    avv2 = random.choice([1, -1]) * random.uniform(-1, -0.5)
    azv1 = random.choice([1, -1]) * random.uniform(-1, -0.5)
    azv2 = random.choice([1, -1]) * random.uniform(-1, -0.5)

    add = random.choice([1, -1]) * random.uniform(-1, -0.5)
    azd = random.choice([1, -1]) * random.uniform(-1, -0.5)
    avv = random.choice([1, -1]) * random.uniform(-1, -0.5)
    azv = random.choice([1, -1]) * random.uniform(-1, -0.5)


    auu = random.choice([1, -1]) * random.uniform(-1, -0.5)
    aux = random.choice([1, -1]) * random.uniform(-1, -0.5)
    aww = random.choice([1, -1]) * random.uniform(-1, -0.5)
    awy = random.choice([1, -1]) * random.uniform(-1, -0.5)

    u0 = np.random.normal(size=n)
    u1 = auu * u0 + np.random.normal(size=n)
    u2 = auu * u1 + np.random.normal(size=n)
    u3 = auu * u2 + np.random.normal(size=n)
    u4 = auu * u3 + np.random.normal(size=n)
    u5 = auu * u4 + np.random.normal(size=n)

    w0 = np.random.normal(size=n)
    w1 = aww * w0 + np.random.normal(size=n)
    w2 = aww * w1 + np.random.normal(size=n)
    w3 = aww * w2 + np.random.normal(size=n)
    w4 = aww * w3 + np.random.normal(size=n)
    w5 = aww * w4 + np.random.normal(size=n)

    x0 = np.random.normal(size=n)
    x1 = axx * x0 + aux * u0 + np.random.normal(size=n)
    x2 = axx * x1 + aux * u1 + np.random.normal(size=n)
    x3 = axx * x2 + aux * u2 + np.random.normal(size=n)
    x4 = axx * x3 + aux * u3 + np.random.normal(size=n)
    x5 = axx * x4 + aux * u4 + np.random.normal(size=n)

    y0 = np.random.normal(size=n)
    y1 = ayy * y0 + awy * w0 + np.random.normal(size=n)
    y2 = ayy * y1 + awy * w1 + np.random.normal(size=n)
    y3 = ayy * y2 + awy * w2 + np.random.normal(size=n)
    y4 = ayy * y3 + awy * w3 + np.random.normal(size=n)
    y5 = ayy * y4 + awy * w4 + np.random.normal(size=n)

    z0 = np.random.normal(size=n)
    z1 = azz * z0 + axz * x0 + ayz * y0 + np.random.normal(size=n)
    z2 = azz * z1 + axz * x1 + ayz * y1 + np.random.normal(size=n)
    z3 = azz * z2 + axz * x2 + ayz * y2 + np.random.normal(size=n)
    z4 = azz * z3 + axz * x3 + ayz * y3 + np.random.normal(size=n)
    z5 = azz * z4 + axz * x4 + ayz * y4 + np.random.normal(size=n)

    d_1_0 = np.random.normal(size=n)
    d_1_1 = add1 * d_1_0 + azd1 * z0 + np.random.normal(size=n)
    d_1_2 = add1 * d_1_1 + azd1 * z1 + np.random.normal(size=n)
    d_1_3 = add1 * d_1_2 + azd1 * z2 + np.random.normal(size=n)
    d_1_4 = add1 * d_1_3 + azd1 * z3 + np.random.normal(size=n)
    d_1_5 = add1 * d_1_4 + azd1 * z4 + np.random.normal(size=n)

    d_2_0 = np.random.normal(size=n)
    d_2_1 = add2 * d_2_0 + azd2 * z0 + np.random.normal(size=n)
    d_2_2 = add2 * d_2_1 + azd2 * z1 + np.random.normal(size=n)
    d_2_3 = add2 * d_2_2 + azd2 * z2 + np.random.normal(size=n)
    d_2_4 = add2 * d_2_3 + azd2 * z3 + np.random.normal(size=n)
    d_2_5 = add2 * d_2_4 + azd2 * z4 + np.random.normal(size=n)

    d_3_0 = np.random.normal(size=n)
    d_3_1 = add3 * d_3_0 + azd3 * z0 + np.random.normal(size=n)
    d_3_2 = add3 * d_3_1 + azd3 * z1 + np.random.normal(size=n)
    d_3_3 = add3 * d_3_2 + azd3 * z2 + np.random.normal(size=n)
    d_3_4 = add3 * d_3_3 + azd3 * z3 + np.random.normal(size=n)
    d_3_5 = add3 * d_3_4 + azd3 * z4 + np.random.normal(size=n)



    d_4_0 = np.random.normal(size=n)
    d_4_1 = add * d_4_0 + azd * z0 + np.random.normal(size=n)
    d_4_2 = add * d_4_1 + azd * z1 + np.random.normal(size=n)
    d_4_3 = add * d_4_2 + azd * z2 + np.random.normal(size=n)
    d_4_4 = add * d_4_3 + azd * z3 + np.random.normal(size=n)
    d_4_5 = add * d_4_4 + azd * z4 + np.random.normal(size=n)

    d_5_0 = np.random.normal(size=n)
    d_5_1 = add * d_5_0 + azd * z0 + np.random.normal(size=n)
    d_5_2 = add * d_5_1 + azd * z1 + np.random.normal(size=n)
    d_5_3 = add * d_5_2 + azd * z2 + np.random.normal(size=n)
    d_5_4 = add * d_5_3 + azd * z3 + np.random.normal(size=n)
    d_5_5 = add * d_5_4 + azd * z4 + np.random.normal(size=n)

    d_6_0 = np.random.normal(size=n)
    d_6_1 = add * d_6_0 + azd * z0 + np.random.normal(size=n)
    d_6_2 = add * d_6_1 + azd * z1 + np.random.normal(size=n)
    d_6_3 = add * d_6_2 + azd * z2 + np.random.normal(size=n)
    d_6_4 = add * d_6_3 + azd * z3 + np.random.normal(size=n)
    d_6_5 = add * d_6_4 + azd * z4 + np.random.normal(size=n)

    d_7_0 = np.random.normal(size=n)
    d_7_1 = add * d_7_0 + azd * z0 + np.random.normal(size=n)
    d_7_2 = add * d_7_1 + azd * z1 + np.random.normal(size=n)
    d_7_3 = add * d_7_2 + azd * z2 + np.random.normal(size=n)
    d_7_4 = add * d_7_3 + azd * z3 + np.random.normal(size=n)
    d_7_5 = add * d_7_4 + azd * z4 + np.random.normal(size=n)

    d_8_0 = np.random.normal(size=n)
    d_8_1 = add * d_8_0 + azd * z0 + np.random.normal(size=n)
    d_8_2 = add * d_8_1 + azd * z1 + np.random.normal(size=n)
    d_8_3 = add * d_8_2 + azd * z2 + np.random.normal(size=n)
    d_8_4 = add * d_8_3 + azd * z3 + np.random.normal(size=n)
    d_8_5 = add * d_8_4 + azd * z4 + np.random.normal(size=n)

    d_9_0 = np.random.normal(size=n)
    d_9_1 = add * d_9_0 + azd * z0 + np.random.normal(size=n)
    d_9_2 = add * d_9_1 + azd * z1 + np.random.normal(size=n)
    d_9_3 = add * d_9_2 + azd * z2 + np.random.normal(size=n)
    d_9_4 = add * d_9_3 + azd * z3 + np.random.normal(size=n)
    d_9_5 = add * d_9_4 + azd * z4 + np.random.normal(size=n)

    v_1_0 = np.random.normal(size=n)
    v_1_1 = avv1 * v_1_0 + azv1 * x0 + np.random.normal(size=n)
    v_1_2 = avv1 * v_1_1 + azv1 * x1 + np.random.normal(size=n)
    v_1_3 = avv1 * v_1_2 + azv1 * x2 + np.random.normal(size=n)
    v_1_4 = avv1 * v_1_3 + azv1 * x3 + np.random.normal(size=n)
    v_1_5 = avv1 * v_1_4 + azv1 * x4 + np.random.normal(size=n)

    v_2_0 = np.random.normal(size=n)
    v_2_1 = avv2 * v_2_0 + azv2 * y0 + np.random.normal(size=n)
    v_2_2 = avv2 * v_2_1 + azv2 * y1 + np.random.normal(size=n)
    v_2_3 = avv2 * v_2_2 + azv2 * y2 + np.random.normal(size=n)
    v_2_4 = avv2 * v_2_3 + azv2 * y3 + np.random.normal(size=n)
    v_2_5 = avv2 * v_2_4 + azv2 * y4 + np.random.normal(size=n)

    v_3_0 = np.random.normal(size=n)
    v_3_1 = avv * v_3_0 + azv * y0 + np.random.normal(size=n)
    v_3_2 = avv * v_3_1 + azv * y1 + np.random.normal(size=n)
    v_3_3 = avv * v_3_2 + azv * y2 + np.random.normal(size=n)
    v_3_4 = avv * v_3_3 + azv * y3 + np.random.normal(size=n)
    v_3_5 = avv * v_3_4 + azv * y4 + np.random.normal(size=n)

    v_4_0 = np.random.normal(size=n)
    v_4_1 = avv * v_4_0 + azv * y0 + np.random.normal(size=n)
    v_4_2 = avv * v_4_1 + azv * y1 + np.random.normal(size=n)
    v_4_3 = avv * v_4_2 + azv * y2 + np.random.normal(size=n)
    v_4_4 = avv * v_4_3 + azv * y3 + np.random.normal(size=n)
    v_4_5 = avv * v_4_4 + azv * y4 + np.random.normal(size=n)

    v_5_0 = np.random.normal(size=n)
    v_5_1 = avv * v_5_0 + azv * y0 + np.random.normal(size=n)
    v_5_2 = avv * v_5_1 + azv * y1 + np.random.normal(size=n)
    v_5_3 = avv * v_5_2 + azv * y2 + np.random.normal(size=n)
    v_5_4 = avv * v_5_3 + azv * y3 + np.random.normal(size=n)
    v_5_5 = avv * v_5_4 + azv * y4 + np.random.normal(size=n)

    v_6_0 = np.random.normal(size=n)
    v_6_1 = avv * v_6_0 + azv * y0 + np.random.normal(size=n)
    v_6_2 = avv * v_6_1 + azv * y1 + np.random.normal(size=n)
    v_6_3 = avv * v_6_2 + azv * y2 + np.random.normal(size=n)
    v_6_4 = avv * v_6_3 + azv * y3 + np.random.normal(size=n)
    v_6_5 = avv * v_6_4 + azv * y4 + np.random.normal(size=n)


    df = pd.DataFrame(
        {X5: x5,
         Y5: y5,
         Z5: z5,
         U5: u5,
         W5: w5,
         D1: d_1_5,
         D2: d_2_5,
         D3: d_3_5,
         D4: d_4_5,
         D5: d_5_5,
         D6: d_6_5,
         D7: d_7_5,
         D8: d_8_5,
         D9: d_9_5,
         V1: v_1_5,
         V2: v_2_5,
         V3: v_3_5,
         V4: v_4_5,
         V5: v_5_5,
         V6: v_6_5,
         }
    )

    ge = CompletedPartiallyDirectedAcyclicDifferenceGraph()
    ge.add_vertices([X5, Y5, Z5, U5, W5, D1, D2, D3, V1, V2, D4, D5, D6, D7, D8, D9, V3, V4, V5, V6])
    ge.add_directed_edge(U5, X5)
    ge.add_directed_edge(U5, Z5)
    ge.add_directed_edge(W5, Y5)
    ge.add_directed_edge(W5, Z5)
    ge.add_directed_edge(X5, Z5)
    ge.add_directed_edge(Y5, Z5)
    ge.add_undirected_edge(U5, X5)
    ge.add_undirected_edge(W5, Y5)

    ge.add_undirected_edge(U5, V1)
    ge.add_undirected_edge(X5, V1)
    ge.add_undirected_edge(W5, V2)
    ge.add_undirected_edge(Y5, V2)

    ge.add_directed_edge(U5, D1)
    ge.add_directed_edge(W5, D1)
    ge.add_directed_edge(X5, D1)
    ge.add_directed_edge(Y5, D1)
    ge.add_directed_edge(V1, D1)
    ge.add_directed_edge(V2, D1)

    ge.add_directed_edge(U5, D2)
    ge.add_directed_edge(W5, D2)
    ge.add_directed_edge(X5, D2)
    ge.add_directed_edge(Y5, D2)
    ge.add_directed_edge(V1, D1)
    ge.add_directed_edge(V2, D1)

    ge.add_directed_edge(U5, D3)
    ge.add_directed_edge(W5, D3)
    ge.add_directed_edge(X5, D3)
    ge.add_directed_edge(Y5, D3)
    ge.add_directed_edge(V1, D1)
    ge.add_directed_edge(V2, D1)

    ge.add_undirected_edge(Z5, D1)
    ge.add_undirected_edge(Z5, D2)
    ge.add_undirected_edge(Z5, D3)
    ge.add_undirected_edge(D1, D2)
    ge.add_undirected_edge(D1, D3)
    ge.add_undirected_edge(D2, D3)


    gt = CompletedPartiallyDirectedAcyclicDifferenceGraph()
    gt.add_vertices([X5, Y5, Z5, U5, W5, D1, D2, D3, V1, V2])
    gt.add_directed_edge(X5, Z5)
    gt.add_directed_edge(Y5, Z5)
    gt.add_directed_edge(U5, X5)
    gt.add_directed_edge(W5, Y5)


    gt.add_directed_edge(Z5, D1)
    gt.add_directed_edge(Z5, D2)
    gt.add_directed_edge(Z5, D3)
    gt.add_directed_edge(X5, V1)
    gt.add_directed_edge(Y5, V2)

    return df, ge, gt



def two_chain_v_structure(n):
    df1, ge1, gt1 = chain_v_structure_10(n)
    df2, ge2, gt2 = chain_v_structure_10(n, name_prefix="s_")

    df = pd.concat([df1, df2], axis=1)
    ge = CompletedPartiallyDirectedAcyclicDifferenceGraph()
    ge.add_vertices(ge1.get_vertices())
    ge.add_vertices(ge2.get_vertices())
    for u, v in ge1.get_directed_edges():
        ge.add_directed_edge(u, v)
    for u, v in ge2.get_directed_edges():
        ge.add_directed_edge(u, v)
    for u, v in ge1.get_undirected_edges():
        ge.add_undirected_edge(u, v)
    for u, v in ge2.get_undirected_edges():
        ge.add_undirected_edge(u, v)

    gt = CompletedPartiallyDirectedAcyclicDifferenceGraph()
    gt.add_vertices(gt1.get_vertices())
    gt.add_vertices(gt2.get_vertices())
    for u, v in gt1.get_directed_edges():
        gt.add_directed_edge(u, v)
    for u, v in gt2.get_directed_edges():
        gt.add_directed_edge(u, v)

    return df, ge, gt


def three_chain_v_structure(n):
    df1, ge1, gt1 = chain_v_structure_10(n)
    df2, ge2, gt2 = chain_v_structure_10(n, name_prefix="s_")
    df3, ge3, gt3 = chain_v_structure_10(n, name_prefix="t_")

    df = pd.concat([df1, df2], axis=1)
    ge = CompletedPartiallyDirectedAcyclicDifferenceGraph()
    ge.add_vertices(ge1.get_vertices())
    ge.add_vertices(ge2.get_vertices())
    ge.add_vertices(ge3.get_vertices())
    for u, v in ge1.get_directed_edges():
        ge.add_directed_edge(u, v)
    for u, v in ge2.get_directed_edges():
        ge.add_directed_edge(u, v)
    for u, v in ge3.get_directed_edges():
        ge.add_directed_edge(u, v)
    for u, v in ge1.get_undirected_edges():
        ge.add_undirected_edge(u, v)
    for u, v in ge2.get_undirected_edges():
        ge.add_undirected_edge(u, v)
    for u, v in ge3.get_undirected_edges():
        ge.add_undirected_edge(u, v)

    gt = CompletedPartiallyDirectedAcyclicDifferenceGraph()
    gt.add_vertices(gt1.get_vertices())
    gt.add_vertices(gt2.get_vertices())
    gt.add_vertices(gt3.get_vertices())
    for u, v in gt1.get_directed_edges():
        gt.add_directed_edge(u, v)
    for u, v in gt2.get_directed_edges():
        gt.add_directed_edge(u, v)
    for u, v in gt3.get_directed_edges():
        gt.add_directed_edge(u, v)
    return df, ge, gt


if __name__ == '__main__':
    nb_var = 10
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

    time_pc = []
    time_fci = []
    time_restpc = []
    for iter in range(10):
        print("####### iter ", iter, " #######")
        if nb_var == 10:
            data, ge, gt = chain_v_structure_10(n_sample)
        elif nb_var == 20:
            data, ge, gt = two_chain_v_structure(n_sample)
        elif nb_var == 30:
            data, ge, gt = three_chain_v_structure(n_sample)

        start_pc = time.time()
        pc = PC(sparsity=0.05, ci_test=FisherZTest)
        pc.run(data)
        end_pc = time.time()
        time_pc.append(end_pc - start_pc)
        start_fci = time.time()
        fci = FCI(sparsity=0.05, ci_test=FisherZTest)
        fci.run(data)
        end_fci = time.time()
        time_fci.append(end_fci - start_fci)
        start_restpc = time.time()
        restpc = RestPC(sparsity=0.05, ci_test=FisherZTest)
        restpc.run(data)
        end_restpc = time.time()
        time_restpc.append(end_restpc - start_restpc)

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

        # print(ft_pc_o, fe_pc_o)
        # print(ft_restpc_o, fe_restpc_o)

    print("################")

    print("PC")
    print("F1-t-a", np.mean(list_pc_f1_a_t), np.var(list_pc_f1_a_t))
    print("F1-t-o", np.mean(list_pc_f1_o_t), np.var(list_pc_f1_o_t))
    print("F1-e-a", np.mean(list_pc_f1_a_e), np.var(list_pc_f1_a_e))
    print("F1-e-o", np.mean(list_pc_f1_o_e), np.var(list_pc_f1_o_e))
    print(f"Average time:", np.mean(time_pc), np.std(time_pc))

    print("FCI")
    print("F1-t-a", np.mean(list_fci_f1_a_t), np.var(list_fci_f1_a_t))
    print("F1-t-o", np.mean(list_fci_f1_o_t), np.var(list_fci_f1_o_t))
    print("F1-e-a", np.mean(list_fci_f1_a_e), np.var(list_fci_f1_a_e))
    print("F1-e-o", np.mean(list_fci_f1_o_e), np.var(list_fci_f1_o_e))
    print(f"Average time:", np.mean(time_fci), np.std(time_fci))

    print("RestPC")
    print("F1-t-a", np.mean(list_restpc_f1_a_t), np.var(list_restpc_f1_a_t))
    print("F1-t-o", np.mean(list_restpc_f1_o_t), np.var(list_restpc_f1_o_t))
    print("F1-e-a", np.mean(list_restpc_f1_a_e), np.var(list_restpc_f1_a_e))
    print("F1-e-o", np.mean(list_restpc_f1_o_e), np.var(list_restpc_f1_o_e))
    print(f"Average time:", np.mean(time_restpc), np.std(time_restpc))
