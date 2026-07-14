import numpy as np

# TODO: WIP


def finalize_directed_adj(adj: np.ndarray) -> np.ndarray:
    """
    In causallearn, there are multiples edge types to allow unmeasured variables.
    since there is no left step, we will convert the adj as follows
    1. nul  to nul
    2. -->  to  -->
    3. ---  to  <->
    4. <->  to  <->
    5. o->  to  -->

    . --o  to  not_determined_yet
    . o-o  to  not_determined_yet

    # rules for output_adj
    - i --> j : adj[j, i] == 1 and adj[i, j] == 0
    - i <-- j : adj[j, i] == 0 and adj[i, j] == 1
    - i <-> j : adj[j, i] == 1 and adj[i, j] == 1

    IMPORTATN NOTE:
    # edge direction here is from cause --> effect
    # before passing to pagerank, we need to do adj.T to convert cause <-- effect.
    """
    # # rules for output_adj
    # - i --> j : adj[j, i] == 1 and adj[i, j] == 0
    # - i <-- j : adj[j, i] == 0 and adj[i, j] == 1
    # - i <-> j : adj[j, i] == 1 and adj[i, j] == 1
    # """
    output_adj = np.zeros_like(adj)
    for i in range(adj.shape[0]):
        for j in range(adj.shape[1]):
            # case 1: no edge: i j
            if adj[i, j] == adj[j, i] == 0:
                pass

            # case 2.1: fully directed i-->j
            elif adj[j, i] == 1 and adj[i, j] == -1:
                output_adj[j, i] = 1

            # case 2.2: fully directed i<--j
            elif adj[j, i] == -1 and adj[i, j] == 1:
                output_adj[i, j] = 1

            # case 2.3: fully directed i-->j, preprocessed case
            elif adj[j, i] == 1 and adj[i, j] == 0:
                output_adj[j, i] = 1

            # case 2.4: fully directed i<--j, preprocessed case
            elif adj[j, i] == 0 and adj[i, j] == 1:
                output_adj[i, j] = 1


            # case 3: i---j  to  i<->j
            elif adj[i, j] == adj[j, i] == -1:
                output_adj[i, j] = output_adj[j, i] = 1

            # case 4: i<->j  to  i<->j
            elif adj[i, j] == adj[j, i] == 1:
                output_adj[i, j] = output_adj[j, i] = 1

            # case 5.1: io->j  to  i-->j, in FCI, tricky
            elif adj[i, j] == 2 and adj[j, i] == 1:
                output_adj[j, i] = 1

            # case 5.2: i<-oj  to  i<--j, in FCI, tricky
            elif adj[i, j] == 1 and adj[j, i] == 2:
                output_adj[i, j] = 1

            # case 6: o-o  to  <->
            elif adj[i, j] == adj[j, i] == 2:
                output_adj[i, j] = output_adj[j, i] = 1

            else:
                raise ValueError(f"Unexpected value: {adj[i, j]=}, {adj[j, i]=}")
    
    return output_adj

# def finalize_undirected_adj(adj : np.ndarray) -> np.ndarray:
#     pass