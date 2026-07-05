from typing import Set, Tuple, FrozenSet, Iterable
import numpy as np
import pandas as pd
import random

from pyciphod.utils.scms.dynamic_scm import create_random_linear_dt_dynamic_scm, DtDynamicSCM, create_random_linear_dt_dynamic_scm_from_ftadmg
from pyciphod.utils.time_series.data_format import DTimeVar
from pyciphod.utils.graphs.partially_specified_graphs import CompletedPartiallyDirectedAcyclicDifferenceGraph
from pyciphod.causal_discovery.basic.constraint_based import PC, RestPC
from pyciphod.utils.graphs.temporal_graphs import create_random_ft_dag, FtDirectedAcyclicGraph
from pyciphod.utils.stat_tests.independence_tests import LinearRegressionCoefficientTTest, FisherZTest, CopulaTest
from pyciphod.utils.stat_tests.dependency_measures import Copula, compute_copula_fit


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
    return 2 * (prec * rec) / (prec + rec)

def f1_score_a(g, g_hat):
    true_set = g.get_undirected_edges()
    pred_set = g_hat.get_undirected_edges()
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

    np.random.randint(3)
    bin_var = random.choice(["X", "Y", "Z"])

    if "Z" == bin_var:
        z1 = (np.random.normal(size=n) > 0).astype(int)
        z2 = (-z1 + np.random.normal(size=n) > 0).astype(int)
        z3 = (-z2 + np.random.normal(size=n) > 0).astype(int)
    else:
        z1 = np.random.normal(size=n)
        z2 = -z1 + np.random.normal(size=n)
        z3 = -z2 + np.random.normal(size=n)

    if "X" == bin_var:
        x1 = (np.random.normal(size=n) > 0).astype(int)
        x2 = (-0.2*x1 + 0.7*z1 + np.random.normal(size=n) > 0).astype(int)
        x3 = (-0.2*x2 + 0.7*z2 + np.random.normal(size=n) > 0).astype(int)
    else:
        x1 = np.random.normal(size=n)
        x2 = -0.2*x1 + 0.7 * z1 + np.random.normal(size=n)
        x3 = -0.2*x2 + 0.7 * z2 + np.random.normal(size=n)

    if "Y" == bin_var:
        y1 = (np.random.normal(size=n) > 0).astype(int)
        y2 = (-0.2*y1 + 0.2*z1 + np.random.normal(size=n) > 0).astype(int)
        y3 = (-0.2*y2 + 0.2*z2 + np.random.normal(size=n) > 0).astype(int)
    else:
        y1 = np.random.normal(size=n)
        y2 = -0.2*y1 + 0.6 * z1 + np.random.normal(size=n)
        y3 = -0.2*y2 + 0.6 * z2 + np.random.normal(size=n)

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

def chain_structure():
    1


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

    np.random.randint(3)
    bin_var = random.choice(["X", "Y", "Z"])
    print("bin", bin_var)

    if "X" == bin_var:
        x1 = (np.random.normal(size=n) > 0).astype(int)
        x2 = (-x1 + np.random.normal(size=n) > 0).astype(int)
        x3 = (-x2 + np.random.normal(size=n) > 0).astype(int)
    else:
        x1 = np.random.normal(size=n)
        x2 = -x1 + np.random.normal(size=n)
        x3 = -x2+ np.random.normal(size=n)

    if "Y" == bin_var:
        y1 = (np.random.normal(size=n) > 0).astype(int)
        y2 = (-0.8 * y1 + np.random.normal(size=n) > 0).astype(int)
        y3 = (-0.8 * y2 + np.random.normal(size=n) > 0).astype(int)
    else:
        y1 = np.random.normal(size=n)
        y2 = -0.8 * y1 + np.random.normal(size=n)
        y3 = -0.8 * y2 + np.random.normal(size=n)

    if "Z" == bin_var:
        z1 = (np.random.normal(size=n) > 0).astype(int)
        z2 = (-0.2*z1 + 0.7*x1 + y1 + np.random.normal(size=n) > 0).astype(int)
        z3 = (-0.2*z2 + 0.7*x2 + y2 + np.random.normal(size=n) > 0).astype(int)
    else:
        z1 = np.random.normal(size=n)
        z2 = -0.2*z1 + 0.8*x1 + y1 + np.random.normal(size=n)
        z3 = -0.2*z2 + 0.8*x1 + y1 + np.random.normal(size=n)

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


def chain_v_structure():
    1


if __name__ == '__main__':
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
        data, ge, gt = v_structure(n_sample)

        copula_fit = compute_copula_fit(
            data,
            cols=data.columns,
            n_iter=1000,
            burn_in=50,
            thin=5,
            random_state=0,
        )
        m = copula_fit["copula_matrix"]

        pc = PC(sparsity=0.05, ci_test=CopulaTest)
        pc.run(data, dependence_matrix=m)
        restpc = RestPC(sparsity=0.05, ci_test=CopulaTest)
        restpc.run(data, dependence_matrix=m)

        ft_pc_a = f1_score_a(gt, pc.g_hat)
        list_pc_f1_a_t.append(ft_pc_a)
        ft_pc_o = f1_score_o(gt, pc.g_hat)
        list_pc_f1_o_t.append(ft_pc_o)

        fe_pc_a = f1_score_a(ge, pc.g_hat)
        list_pc_f1_a_e.append(fe_pc_a)
        fe_pc_o = f1_score_o(ge, pc.g_hat)
        list_pc_f1_o_e.append(fe_pc_o)

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

    print("RestPC")
    print("F1-t-a", np.mean(list_restpc_f1_a_t), np.var(list_restpc_f1_a_t))
    print("F1-t-o", np.mean(list_restpc_f1_o_t), np.var(list_restpc_f1_o_t))
    print("F1-e-a", np.mean(list_restpc_f1_a_e), np.var(list_restpc_f1_a_e))
    print("F1-e-o", np.mean(list_restpc_f1_o_e), np.var(list_restpc_f1_o_e))
