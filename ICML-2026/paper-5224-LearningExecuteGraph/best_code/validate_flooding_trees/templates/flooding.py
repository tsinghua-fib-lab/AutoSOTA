import numpy as np
from .flooding_utils import flatten_sample, nested_dict

def get_sample(l,d):
    sample = {
        'message': [0]*l,
        'sent_message': [[0]*l for _ in range(d)],
        'my_slot': [[0]*l for _ in range(d)],
        'message_pipe': [[0]*l for _ in range(d)],
        'message_slot': [[0]*l for _ in range(d)]
    }
    return nested_dict(2), sample


def unflatten_sample(flat, l, d):
    sample = {
        'message': flat[:l],
        'sent_message': [flat[l + i*l : l + (i+1)*l] for i in range(d)],
        'my_slot': [flat[l + d*l + i*l : l + d*l + (i+1)*l] for i in range(d)],
        'message_pipe': [flat[l + 2*d*l + i*l : l + 2*d*l + (i+1)*l] for i in range(d)],
        'message_slot': [flat[l + 3*d*l + i*l : l + 3*d*l + (i+1)*l] for i in range(d)],
    }
    return sample


def execute(x, Y_train, l):
    non_zero_indices = np.nonzero(x)[0]
    y = np.sum(Y_train[non_zero_indices], axis=0)
    y = np.where(y > 0, 1, 0)
    return unflatten_sample(y, l)


def get_dataset(l,d):
    T, Y = [], []
    # flooding samples
    for i in range(l):
        x, y = get_sample(l,d)
        x['message'][i] = 1
        y['message'][i] = 1
        for t in range(d): y['sent_message'][t][i] = 1
        T.append(x)
        Y.append(flatten_sample(y))

        for s in range(d):

            x, y = get_sample(l,d)
            x['sent_message'][s][i] = 1
            x['my_slot'][s][i] = 1
            y['message_slot'][s][i] = 1
            y['my_slot'][s][i] = 1
            T.append(x)
            Y.append(flatten_sample(y))

            x, y = get_sample(l,d)
            x['sent_message'][s][i] = 0
            x['my_slot'][s][i] = 1
            y['my_slot'][s][i] = 1
            T.append(x)
            Y.append(flatten_sample(y))

            x, y = get_sample(l,d)
            x['message_slot'][s][i] = 1
            y['message_pipe'][s][i] = 1
            T.append(x)
            Y.append(flatten_sample(y))

            x, y = get_sample(l,d)
            x['message_pipe'][s][i] = 1
            if s < d-1:
                y['message_pipe'][s+1][i] = 1
            else:
                y['message'][i] = 1
            T.append(x)
            Y.append(flatten_sample(y))
    
    return T, np.array(Y)


