import numpy as np
from .flooding_utils import flatten_sample, nested_dict, match_sample

def get_sample(l,d):
    sample = {
        'message': [0]*l,
        'final_message':[0]*l,
        'sent_message': [[0]*l for _ in range(d)],
        'my_slot': [[0]*l for _ in range(d)],
        'message_pipe': [[0]*l for _ in range(d)],
        'message_slot': [[0]*l for _ in range(d)]
    }
    return nested_dict(2), sample


def unflatten_sample(flat, l, d):

    sample = {
        'message': flat[:l],
        'final_message':flat[l:2*l],
        'sent_message': [flat[2*l + i*l : 2*l + (i+1)*l] for i in range(d)],
        'my_slot': [flat[2*l + d*l + i*l : 2*l + d*l + (i+1)*l] for i in range(d)],
        'message_pipe': [flat[2*l + 2*d*l + i*l : 2*l + 2*d*l + (i+1)*l] for i in range(d)],
        'message_slot': [flat[2*l + 3*d*l + i*l : 2*l + 3*d*l + (i+1)*l] for i in range(d)],
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
            x['sent_message'][s][i] = 1
            x['my_slot'][s][i] = 0
            y['message_slot'][s][i] = 0
            y['my_slot'][s][i] = 0
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
                y['final_message'][i] = 1
            T.append(x)
            Y.append(flatten_sample(y))
    
    return T, np.array(Y)


def get_dataset_uncoded(l,d):
    T, Y = [], []
    # flooding samples
    for i in range(l):
        _, x = get_sample(l,d)
        _, y = get_sample(l,d)
        x['message'][i] = 1
        y['message'][i] = 1
        for t in range(d): y['sent_message'][t][i] = 1
        T.append(flatten_sample(x, normalize=True))
        Y.append(flatten_sample(y))

        for s in range(d):

            _, x = get_sample(l,d)
            _, y = get_sample(l,d)
            x['sent_message'][s][i] = 1
            x['my_slot'][s][i] = 1
            y['message_slot'][s][i] = 1
            y['my_slot'][s][i] = 1
            T.append(flatten_sample(x, normalize=True))
            Y.append(flatten_sample(y))

            _, x = get_sample(l,d)
            _, y = get_sample(l,d)
            x['sent_message'][s][i] = 1
            x['my_slot'][s][i] = 0
            y['message_slot'][s][i] = 0
            y['my_slot'][s][i] = 0
            T.append(flatten_sample(x, normalize=True))
            Y.append(flatten_sample(y))

            _, x = get_sample(l,d)
            _, y = get_sample(l,d)
            x['sent_message'][s][i] = 0
            x['my_slot'][s][i] = 1
            y['my_slot'][s][i] = 1
            T.append(flatten_sample(x, normalize=True))
            Y.append(flatten_sample(y))

            _, x = get_sample(l,d)
            _, y = get_sample(l,d)
            x['message_slot'][s][i] = 1
            y['message_pipe'][s][i] = 1
            T.append(flatten_sample(x, normalize=True))
            Y.append(flatten_sample(y))

            _, x = get_sample(l,d)
            _, y = get_sample(l,d)
            x['message_pipe'][s][i] = 1
            if s < d-1:
                y['message_pipe'][s+1][i] = 1
            else:
                y['final_message'][i] = 1
            T.append(flatten_sample(x, normalize=True))
            Y.append(flatten_sample(y))
    
    return np.array(T), np.array(Y)

def star_graph_adjacency_matrix(n):
    """
    Generate adjacency matrix for a star graph with n nodes.
    
    Args:
        n (int): Number of nodes (including the center node)
    
    Returns:
        numpy.ndarray: n x n adjacency matrix with self-loops
    """
    if n < 1:
        raise ValueError("Number of nodes must be at least 1")
    
    # Create zero matrix
    adj_matrix = np.zeros((n, n), dtype=int)
    
    # Add self-loops (diagonal elements)
    np.fill_diagonal(adj_matrix, 1)
    
    # Connect all nodes to center node (node 0)
    # Connect center to all other nodes
    for i in range(1, n):
        adj_matrix[0, i] = 1  # center to peripheral
        adj_matrix[i, 0] = 1  # peripheral to center
    
    return adj_matrix

def create_node_features(n, s, t, l, d, message, local_ids):
    X_test_list = []
    for node in range(n):
        _, sample = get_sample(l, d)
        # if node is source, set sample['message'] to the binary message
        if node != s:
            sample['message'] = message.copy()
        # set node's my_slot at index local_id-1 to all ones
        jid = local_ids[node] - 1  # zero-based
        sample['my_slot'][jid] = [1]*l
        X_test_list.append(sample)
    return X_test_list


def calculate_accuracy(a1, a2, b):
    """
    Calculate accuracy where a sample is correct only if:
    - vector i in a1 equals vector i in a2
    - AND vector i in a1 equals vector i in b
    
    Parameters:
    a1, a2, b: lists of numpy vectors
    
    Returns:
    accuracy: float between 0 and 1
    """
    if len(a1) != len(a2) or len(a1) != len(b):
        raise ValueError("All three lists must have the same length")
    
    correct_count = 0
    total_count = len(a1)
    
    for i in range(total_count):
        # Check if a1[i] equals a2[i] AND a1[i] equals b[i]
        if np.array_equal(a1[i], a2[i]) and np.array_equal(a1[i], b[i]):
            correct_count += 1
    
    accuracy = correct_count / total_count
    return accuracy


def encode_data(xhat, T, pad):
    x = np.zeros(len(T)+pad)
    matches = match_sample(xhat, T)
    x[matches] = 1
    no_norm_x = x
    if x.sum() > 0:
        x = x/np.sqrt(x.sum())
    return no_norm_x, x
