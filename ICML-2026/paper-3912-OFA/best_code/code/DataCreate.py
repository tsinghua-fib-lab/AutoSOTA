import numpy as np


def random_data(num_parameter, size_data):
    a = []
    for i in range(size_data):
        b = np.random.dirichlet(np.ones(num_parameter), size=1)
        b.sort()
        b = np.around(b, 4).tolist()
        b = b[0]
        b = b

        a.append(b)

    return a


def order_state(data, size_data):
    x = data
    b = []
    for i in range(size_data):
        a = state_num(x[i])
        b.append(a)

    order = 0  # need rewrite

    return order


def state_num(data):
    x = data
    a = []
    for i in range(x.shape[0]):
        b = max(x[i]) - min(x[i])
        a.append(b)
    c = np.sum(a) / x.shape[0]

    return c
