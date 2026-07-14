import pandas as pd
import numpy as np
from castle.common import GraphDAG
from castle.metrics import MetricsDAG
from castle.datasets import IIDSimulation, DAG
from castle.algorithms import Notears

def notears(data : pd.DataFrame):
    # notears learn
    nt = Notears()
    nt.learn(np.array(data))

    return np.array(nt.causal_matrix)
