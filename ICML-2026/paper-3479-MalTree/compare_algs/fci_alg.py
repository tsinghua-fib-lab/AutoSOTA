import numpy as np
import pandas as pd
from compare_algs.causallearn_package.search.ConstraintBased.FCI import fci
from compare_algs.causallearn_package.utils.cit import fisherz
import time

def causal_learn_to_pcalg(adj: pd.DataFrame) -> pd.DataFrame:
    """
    Convert a causal-learn style adjacency matrix to pcalg style.
    causal-learn: TAIL=-1, ARROW=1, CIRCLE=2, NULL=0
    pcalg:        TAIL=3,  ARROW=2, CIRCLE=1, NULL=0
    """
    mapping = {-1: 3, 1: 2, 2: 1, 0: 0}
    return adj.replace(mapping)

def my_fci(data:pd.DataFrame, alpha:float, maxk:int=4) -> dict:
    """
    in casual-learn:
    TAIL = -1
    NULL = 0
    ARROW = 1
    CIRCLE = 2
    graph[j,i]=1 and graph[i,j]=-1 indicates  i --> j ,
    graph[i,j] = graph[j,i] = -1 indicates i --- j,
    graph[i,j] = graph[j,i] = 1 indicates i <-> j,
    graph[j,i]=1 and graph[i,j]=2 indicates  i o-> j
    """
    
    data_array = data.to_numpy()
    data_labels = data.columns.tolist()
    time_start = time.time()
    graph, ntest = fci(data_array, fisherz, alpha, depth=maxk, node_names=data_labels, verbose=False, show_progress=False)
    time_end = time.time()
    graph_node_name = [node.get_name() for node in graph.node_map.keys()]
    pag_graph_df = pd.DataFrame(graph.graph.T, columns=graph_node_name, index=graph_node_name)
    
    return {"PAG.DataFrame": causal_learn_to_pcalg(pag_graph_df), "CI_num": ntest, "runtime_sec": time_end - time_start}



if __name__ == "__main__":
    # Example usage
    # A->C, B->C, C-D
    np.random.seed(42)
    A = np.random.randn(1000)
    B = np.random.randn(1000)
    C = A + 0.5 * B + np.random.randn(1000)
    D = C + np.random.randn(1000)
    data = pd.DataFrame(
        {'A': A,
         'B': B,
         'C': C,
         'D': D}
    )
    alpha = 0.05
    result = my_fci(data, alpha)
    print("FCI Learned PAG adjacency matrix:")
    print(result['PAG.DataFrame'])
    print(f"Number of conditional independence tests performed: {result['CI_num']}")