# this part is to calculate the equilibrium parameter adjustment matrix
from itertools import count
import numpy as np
#,
def L_parameters(state, state_p, n=10):
    l = np.ones(n)
    if n is None:
        for i in count():
            l1 = (state[i] - state_p[i] + l)/2
            if l1 == l:
                return l
            else:
                l = l1
                return l

    elif isinstance(n, (int, float)):
        for i in range(n):
            l = (state[i] - state_p[i] + l) / 2
        return l






    #for i in range(n)

    return l

print(L_parameters(n=4))





