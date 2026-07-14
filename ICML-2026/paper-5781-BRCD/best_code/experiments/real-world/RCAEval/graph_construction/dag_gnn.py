import numpy as np
import pandas as pd
from castle.algorithms import DAG_GNN, NotearsLowRank
from castle.common import GraphDAG
from castle.datasets import DAG, IIDSimulation
from castle.metrics import MetricsDAG


def dag_gnn(data: pd.DataFrame):
    # rl learn
    gnn = DAG_GNN()
    gnn.learn(np.array(data))

    return np.array(gnn.causal_matrix)


def notears_low_rank(data: pd.DataFrame):
    # rl learn
    notears = NotearsLowRank()
    # notears.learn(np.array(data), rank=data.shape[1])
    notears.learn(np.array(data), rank=10)

    return np.array(notears.causal_matrix)
