import numpy as np

class dataNode(object):
    """
    node for the last dimension
    """
    def __init__(self, weight, noise, children=None):
        self.weight = weight
        self.noise = noise
        self.children = [None, None] if children is None else children


class splitNode(object): 
    """
    nodes for the first (d-1) dimensions
    """
    def __init__(self, next_root=None, children=None):
        self.next_root = next_root
        self.children = [None, None] if children is None else children

def insertData(root, attr, l, r, mag):
    """
    :param root: current root of dataNode 
    :param attr: attribute vector + weight at the end
    :param l: left boundary of the current range
    :param r: right boundary of the current range
    :param mag: magnitude of the noise
    return the updated current node
    """
    if root is None:
        root = dataNode(0, np.random.laplace(0, mag))
    node = root
    while True:
        node.weight += attr[-1]
        if l == r:
            break
        mid = l + r >> 1
        idx = 0 if attr[-2] <= mid else 1
        if node.children[idx] is None:
            node.children[idx] = dataNode(0, np.random.laplace(0, mag))
        node = node.children[idx]
        if idx == 0:
            r = mid
        else:
            l = mid + 1
    return root



def make_consistent_data(root, mag):
    """Bottom-up consistency enforcement for a dataNode tree.
    
    Applies the Hay et al. VLDB 2010 constrained inference:
    For each parent-child triplet, redistributes noise so that
    parent.noisy = left_child.noisy + right_child.noisy,
    minimizing L2 distance to original noisy values.
    
    This is a post-processing step that does NOT consume privacy budget
    (post-processing immunity of differential privacy).
    """
    if root is None:
        return
    make_consistent_data(root.children[0], mag)
    make_consistent_data(root.children[1], mag)
    
    left = root.children[0]
    right = root.children[1]
    
    if left is not None and right is not None:
        z_p = root.weight + root.noise
        z_l = left.weight + left.noise
        z_r = right.weight + right.noise
        
        # L2-optimal consistent estimates (Hay et al. VLDB 2010)
        zp_new = (2.0 * z_p + z_l + z_r) / 3.0
        zl_new = (z_p + 2.0 * z_l - z_r) / 3.0
        zr_new = (z_p - z_l + 2.0 * z_r) / 3.0
        
        root.noise = zp_new - root.weight
        left.noise = zl_new - left.weight
        right.noise = zr_new - right.weight

def make_consistent_split(root, mag):
    """Apply consistency to all dataNode trees in the splitNode hierarchy."""
    if root is None:
        return
    make_consistent_data(root.next_root, mag)
    make_consistent_split(root.children[0], mag)
    make_consistent_split(root.children[1], mag)

def queryData(node, l, r, x, y, mag):
    """
    :param node: current dataNode 
    :param l: left boundary of the current range
    :param r: right boundary of the current range
    :param x: left boundary of the query range
    :param y: right boundary of the query range
    :param mag: magnitude of the noise
    :return: (weight, noise) of the range [x, y]
    """
    if (y < l or x > r):
        return (None, 0, 0)
    if (node is None):
        node = dataNode(0, np.random.laplace(0, mag))
    if (l >= x and r <= y):
        return (node, node.weight, node.noise)
    mid = l + r >> 1
    if (y <= mid):
        (node.children[0], weight, noise) = queryData(node.children[0], l, mid, x, y, mag)
        return (node, weight, noise)
    elif (x > mid):
        (node.children[1], weight, noise) = queryData(node.children[1], mid + 1, r, x, y, mag)
        return (node, weight, noise)
    else:
        (node.children[0], lweight, lnoise) = queryData(node.children[0], l, mid, x, mid, mag)
        (node.children[1], rweight, rnoise) = queryData(node.children[1], mid + 1, r, mid + 1, y, mag)
        return (node, lweight + rweight, lnoise + rnoise)

def insertSplit(root, attr, l, r, k, n, last, mag):
    """
    :param root: current root of splitNode 
    :param attr: attribute vector + weight at the end
    :param l: left boundary of the current range
    :param r: right boundary of the current range
    :param k: current dimension
    :param n: the maximum length of each dimension
    :param last: the last dimension
    :param mag: magnitude of the noise
    return the updated current node
    """
    if root is None:
        root = splitNode()
    node = root
    while True:
        next_k = k + 1
        if (next_k == last):
            node.next_root = insertData(node.next_root, attr, 0, n - 1, mag)
        else:
            node.next_root = insertSplit(node.next_root, attr, 0, n - 1, next_k, n, last, mag)
        if l == r:
            break 
        mid = l + r >> 1
        idx = 0 if attr[k] <= mid else 1
        if node.children[idx] is None:
            node.children[idx] = splitNode()
        node = node.children[idx]
        if idx == 0:
            r = mid
        else:
            l = mid + 1
    return root

def querySplit(node, l, r, q, k, n, last, mag):
    """
    :param node: current splitNode 
    :param l: left boundary of the current range
    :param r: right boundary of the current range
    :param q: query ranges for all dimensions
    :param k: current dimension
    :param n: the maximum length of each dimension
    :param last: the last dimension
    :param mag: magnitude of the noise
    :return: (weight, noise) of the range [x, y]
    """
    if (q[k][1] < l or q[k][0] > r):
        return (None, 0, 0)
    if (node is None):
        node = splitNode()
    if (l >= q[k][0] and r <= q[k][1]):
        next_k = k + 1
        if (next_k == last):
            (node.next_root, weight, noise) = queryData(node.next_root, 0, n - 1, q[-1][0], q[-1][1], mag)
        else:
            (node.next_root, weight, noise) = querySplit(node.next_root, 0, n - 1, q, next_k, n, last, mag)
        return (node, weight, noise)
    mid = l + r >> 1
    if (q[k][1] <= mid):
        (node.children[0], weight, noise) = querySplit(node.children[0], l, mid, q, k, n, last, mag)   
        return (node, weight, noise)
    elif (q[k][0] > mid):
        (node.children[1], weight, noise) = querySplit(node.children[1], mid + 1, r, q, k, n, last, mag)
        return (node, weight, noise)
    else:
        lq = q[:]
        lq[k] = [q[k][0], mid]
        rq = q[:]
        rq[k] = [mid + 1, q[k][1]]
        (node.children[0], lweight, lnoise) = querySplit(node.children[0], l, mid, lq, k, n, last, mag)
        (node.children[1], rweight, rnoise) = querySplit(node.children[1], mid + 1, r, rq, k, n, last, mag)
        return (node, lweight + rweight, lnoise + rnoise)