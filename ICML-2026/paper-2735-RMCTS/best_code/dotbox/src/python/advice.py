import numpy as np

N = 4
NUMACTIONS = 2*N*(N-1)

def hint(G):
    assert len(G.shape) == 2
    assert G.shape[1] == NUMACTIONS + (N-1)*(N-1) + 4
    m = G.shape[0]
    pi = np.zeros((m,NUMACTIONS),dtype=np.float32)
    pi[:] = 1.0 / NUMACTIONS
    value = np.zeros((m,1),dtype=np.float32)
    return pi, value


