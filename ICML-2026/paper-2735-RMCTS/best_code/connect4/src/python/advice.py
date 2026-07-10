import numpy as np

pi0 = np.array([3.,4.,5.,7.,5.,4.,3.], dtype=np.float32)
pi0 /= np.sum(pi0)

def hint(G):
    assert len(G.shape) == 2
    assert G.shape[1] == 45
    m = G.shape[0]
    pi = np.zeros((m,7),dtype=np.float32)
    pi[:] = pi0
    value = np.zeros((m,1),dtype=np.float32)
    return pi,value

