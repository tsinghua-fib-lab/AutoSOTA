import numpy as np

N = 8
# compute distance to central point
sigma = 4.0
a = np.arange(64)
x = (a // N) - (N//2)
y = (a % N) - (N//2)
sq_dist_to_center = np.maximum(x**2, y**2)
pi0 = np.exp(-sq_dist_to_center / (2.0 * (sigma**2)))
pi0 /= np.sum(pi0)
pi0 = np.float32(pi0)

def hint(G):
    assert len(G.shape) == 2
    assert G.shape[1] == 67
    m = G.shape[0]
    pi = np.zeros((m,64),dtype=np.float32)
    pi[:] = pi0
    value = np.zeros((m,1),dtype=np.float32)
    return pi, value

