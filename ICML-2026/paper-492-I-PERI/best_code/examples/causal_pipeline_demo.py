import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from itertools import combinations

from pyciphod.utils import DirectedAcyclicGraph
from pyciphod.utils import LinearSCM, create_random_linear_scm_from_dag

from pyciphod.causal_estimation import GComputation, LinearOutcomeRegression
from pyciphod.causal_reasoning import back_door_criterion, front_door_criterion
from pyciphod.causal_discovery import PC


def build_target_dag1():
    dag = DirectedAcyclicGraph()
    vertices = ["A", "Y", "Z1", "Z2", "Z3", "Z4", "Z5", "Z6", "Z7", "Z8", "Z9"]
    for v in vertices:
        dag.add_vertex(v)
    # Edges as described
    dag.add_directed_edge("A", "Z1")
    dag.add_directed_edge("Y", "Z1")
    dag.add_directed_edge("Z1", "Z2")
    dag.add_directed_edge("A", "Z3")
    dag.add_directed_edge("Z3", "Y")
    dag.add_directed_edge("Z4", "A")
    dag.add_directed_edge("Z6", "Y")
    dag.add_directed_edge("Z4", "Z5")
    dag.add_directed_edge("Z6", "Z5")
    dag.add_directed_edge("Z7", "A")
    dag.add_directed_edge("Z9", "Y")
    dag.add_directed_edge("Z8", "Z7")
    dag.add_directed_edge("Z8", "Z9")
    return dag


def make_data1(dag, n=5000, seed=10):
    scm, coeffs, intercepts = create_random_linear_scm_from_dag(dag, seed=seed)
    tce = coeffs[("A", "Z3")]*coeffs[("Z3", "Y")]
    # data= scm.generate_data(n)
    # data.to_csv("./data1.csv", index=False)

    data = pd.read_csv("./data1.csv")
    return data, tce


def build_target_dag2():
    dag = DirectedAcyclicGraph()
    vertices = ["A", "Y", "Z1", "Z2", "Z3", "Z4", "Z5", "Z6", "Z7", "Z8", "Z9"]
    for v in vertices:
        dag.add_vertex(v)
    # Edges as described
    dag.add_directed_edge("A", "Z1")
    dag.add_directed_edge("Y", "Z1")
    dag.add_directed_edge("Z1", "Z2")
    dag.add_directed_edge("A", "Z3")
    dag.add_directed_edge("Z3", "Y")
    dag.add_directed_edge("Z4", "A")
    dag.add_directed_edge("Z6", "Y")
    dag.add_directed_edge("Z4", "Z5")
    dag.add_directed_edge("Z6", "Z5")
    dag.add_directed_edge("Z7", "A")
    dag.add_directed_edge("Z9", "Y")
    dag.add_directed_edge("Z8", "Z7")
    dag.add_directed_edge("Z8", "Z9")
    return dag


def make_data2(dag, n=5000,  seed=1):
    # rng = np.random.RandomState(seed)
    # Z1 = rng.normal(size=n)
    # Z2 = rng.normal(size=n)
    # Z3 = rng.normal(size=n)
    #
    # logits_t = 0.6 * Z1 - 0.3 * Z2
    # p_t = 1.0 / (1.0 + np.exp(-logits_t))
    # A = (rng.rand(n) < p_t).astype(int)
    #
    # logits_y = -0.5 + 1.2 * A + 0.3 * Z1 - 0.2 * Z2 + 0.5 * Z3
    # p_y = 1.0 / (1.0 + np.exp(-logits_y))
    # y = (rng.rand(n) < p_y).astype(int)
    # data = pd.DataFrame({'Z1': Z1, 'Z2': Z2, 'Z3': Z3, 'A': A, 'Y': y})
    # data.to_csv("./data2.csv", index=False)

    data = pd.read_csv("./data2.csv")
    #print(data)
    return data, 1


setting = "d"
if setting == "c":
    dag = build_target_dag1()

    # dag.draw_graph()
    data, te = make_data1(dag, n=5000)
    print("True causal effect:", te)

    adj_list1 = ["Z9"]
    adj_list2 = ["Z9", "Z6"]
    adj_list3 = ["Z7", "Z8", "Z9", "Z6"]
    adj_list4 = ["Z7", "Z8", "Z9", "Z6", "Z2"]
    front_adj_list1 = ["Z3"]

else:
    dag = build_target_dag2()
    #dag.draw_graph()
    data, _ = make_data2(dag, n=5000)
    adj_list1 = ["Z1", "Z2"]

pc = PC()
pc.run(data)

# all_covariates = list(data.columns)
# all_covariates.remove("A")
# all_covariates.remove("Y")

pc.g_hat.draw_graph()

if setting == "c":
    bd_test = back_door_criterion(pc.g_hat, ["A"], ["Y"], adj_list1)
    print(bd_test)
    bd_test = back_door_criterion(pc.g_hat, ["A"], ["Y"], adj_list2)
    print(bd_test)
    bd_test = back_door_criterion(pc.g_hat, ["A"], ["Y"], adj_list3)
    print(bd_test)
    bd_test = back_door_criterion(pc.g_hat, ["A"], ["Y"], adj_list4)
    print(bd_test)

    fd_test = front_door_criterion(pc.g_hat, ["A"], ["Y"], front_adj_list1)
    print(fd_test)
else:
    bd_test = back_door_criterion(pc.g_hat, ["A"], ["Y"], adj_list1)
    print(bd_test)

print("############# run")

list_rf = []
for i in range(10):
    if setting == "c":
        gcomp = GComputation("A", "Y", z=adj_list1, w=None, model=LinearRegression(), seed=0)
    else:
        gcomp = GComputation("A", "Y", z=adj_list1, w=None, model=RandomForestClassifier(), seed=0)
    res = gcomp.run(data)
    est = float(res.get("cate"))
    list_rf.append(est)
print(list_rf)
print(np.mean(list_rf))

# print(data)

# print("#############  Z9 Z6")
# gcomp = LinearOutcomeRegression("X", "Y", z=adj_list2, w=None, seed=0)
# res = gcomp.run(data)
# est = float(res.get("cate"))
# print(est)
#
# list_rf = []
# for i in range(10):
#     gcomp = GComputation("X", "Y", z=adj_list2, w=None, model=LinearRegression(), seed=0)
#     res = gcomp.run(data)
#     est = float(res.get("cate"))
#     list_rf.append(est)
# print(np.mean(list_rf))
#
#
# print("############# Z7, Z8, Z9, Z6")
# gcomp = LinearOutcomeRegression("X", "Y", z=adj_list3, w=None, seed=0)
# res = gcomp.run(data)
# est = float(res.get("cate"))
# print(est)
#
# list_rf = []
# for i in range(10):
#     gcomp = GComputation("X", "Y", z=adj_list3, w=None, model=LinearRegression(), seed=0)
#     res = gcomp.run(data)
#     est = float(res.get("cate"))
#     list_rf.append(est)
# print(np.mean(list_rf))
