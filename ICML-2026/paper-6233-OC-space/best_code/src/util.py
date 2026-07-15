import prada
import veritas
import math
import numba
from datetime import datetime
import numpy as np


SEED = 5823

TIMEOUT = 30*60
NUM_ADV_EX = 500
GUARD = 1e-5
NFOLDS = 5


DNAMES_REGRESSION = [
    "WineQuality",
    "Houses",
    "Ailerons",
    "Abalone",
    "CpuSmall",
    "Elevators",
    "House16H"
]

DNAMES_CLASSIFICATION = [
    "Electricity",
    "MiniBooNE",
    "Jannis",
    "Credit",
    "California",
    "CompasTwoYears",
    "Vehicle",
    "Spambase",
    "Phoneme",
    "Adult",
    "Ijcnn1",
    "Mnist[2v4]",
    "DryBean[6vRest]",
    "Volkert[2v7]",
]

DNAMES_SUB = [
    "California",
    "Adult",
    "Spambase",
    "Phoneme"
]

def nowstr():
    return datetime.now().strftime("%Y-%m-%d-%H:%M:%S")


def get_dataset(dname, seed, fold, silent):
    d = prada.get_dataset(dname, seed=seed, silent=silent)
    d.load_dataset()
    d.robust_normalize()
    #d.transform_target()
    d.scale_target()
    d.astype(veritas.FloatT)

    if d.is_binary():
        d.use_balanced_accuracy()

    dtrain, dtest = d.train_and_test_fold(fold, nfolds=NFOLDS)
    dtrain, dvalid = dtrain.split(0, nfolds=NFOLDS-1)

    return d, dtrain, dvalid, dtest


def pareto_front_xy(x, y):
    yperm = np.argsort(y)[::-1]
    x = x[yperm]
    y = y[yperm]
    xperm = np.argsort(x, kind="stable")
    x = x[xperm]
    y = y[xperm]
    perm = yperm[xperm]

    n = len(x)
    onfront = np.zeros(n, dtype=bool)

    i = 0
    while i < n:
        onfront[i] = True
        j = n
        for k in range(i+1, n):
            if y[k] > y[i]:
                j = k
                break
        if j < n:
            i = j
        else:
            break

    # Check if on convex hull
    from scipy.spatial import ConvexHull
    xy = np.hstack((x.reshape(-1, 1), y.reshape(-1, 1)))
    ch = ConvexHull(xy)
    onhull = np.zeros_like(onfront)
    onhull[ch.vertices] = True
    onhull &= onfront

    onfront_inv_perm = np.zeros_like(onfront)
    onfront_inv_perm[perm] = onfront
    onhull_inv_perm = np.zeros_like(onfront)
    onhull_inv_perm[perm] = onhull
    return onfront_inv_perm, onhull_inv_perm

def extract_guard(at):
    """ check if we need guard smaller than GUARD """
    min_diff = 1
    for attribute, split_values in at.get_splits().items():
        #print("\n", attribute, split_values)
        if len(split_values)>1:
            diffs = np.diff(split_values)
            #print(diffs)
            if np.min(diffs) < min_diff:
                #print("*** HERE ***")
                min_diff = np.min(diffs)
    
    assert min_diff > 0
    guard = 10**(int(math.floor(math.log10(min_diff))))
    guard = min(guard, GUARD)
    # print(f"Running attacks with guard {guard}.\n")
    return guard

def intersect(ibox, obox):
    """ intersect two boxes, return None if they don't intersect """
    new_box = ibox.copy()
    for i in range(ibox.shape[1]):
        new_box[0, i] = max(ibox[0, i], obox[0, i])
        new_box[1, i] = min(ibox[1, i], obox[1, i])
        if new_box[0, i] >= new_box[1, i]:
            return None
    return new_box

def relax_box(box, idx):
    """ relax a box in the dimension idx, i.e. set the bounds to -inf and inf """
    relaxed_box = box.copy()
    relaxed_box[0, idx] = float('-inf')
    relaxed_box[1, idx] = float('inf')
    return relaxed_box

def get_sub_addtree(at, depth):
    sub_at = veritas.AddTree(at.num_leaf_values(), at.get_type())
    for ti in range(depth):
        sub_at.add_tree(at[ti])
    return sub_at


@numba.njit(cache=True, fastmath=True)
def linf(example, box, feat_map):
    dist = 0.0
    for idx, i in enumerate(feat_map):
        x = example[i]
        lo, hi = box[0, idx], box[1, idx]

        d = 0
        t = lo - x
        if t > 0.0:
            d = t
        t = x - hi
        if t > d:
            d = t

        if d > dist:
            dist = d
    return dist

@numba.njit(cache=True, fastmath=True)
def dist_to_boxes(example, boxes, feat_map):
    dist = np.zeros((boxes.shape[0],), dtype=np.float64)
    for s in range(boxes.shape[0]):
        max_dist = 0.0
        los = boxes[s, 0]
        his = boxes[s, 1]
        for idx, i in enumerate(feat_map):
            x = example[i]
            lo, hi = los[idx], his[idx]

            d = 0
            t = lo - x
            if t > 0.0:
                d = t
            t = x - hi
            if t > d:
                d = t

            if d > max_dist:
                max_dist = d
        dist[s] = max_dist
    return dist

@numba.njit(cache=True, fastmath=True)
def dist_to_boxes_l1(example, boxes, feat_map):
    dist = np.zeros((boxes.shape[0],), dtype=np.float64)
    for s in range(boxes.shape[0]):
        los = boxes[s, 0]
        his = boxes[s, 1]
        for idx, i in enumerate(feat_map):
            x = example[i]
            lo, hi = los[idx], his[idx]

            d = 0.0
            t = lo - x
            if t > 0.0:
                d = t
            t = x - hi
            if t > d:
                d = t

            dist[s] += d
    return dist

@numba.njit(cache=True, fastmath=True)
def l1(example, box, feat_map):
    dist = 0.0
    for idx, i in enumerate(feat_map):
        x = example[i]
        lo, hi = box[0, idx], box[1, idx]

        d = 0.0
        t = lo - x
        if t > 0.0:
            d = t
        t = x - hi
        if t > d:
            d = t

        dist += d
    return dist


@numba.njit(cache=True, fastmath=True)
def dist_to_closest_box(example, boxes, feat_map):
    min_dist = 1e18
    for s in range(boxes.shape[0]):
        max_dist = 0.0
        los = boxes[s, 0]
        his = boxes[s, 1]
        for idx, i in enumerate(feat_map):
            x = example[i]
            lo, hi = los[idx], his[idx]

            d = 0
            t = lo - x
            if t > 0.0:
                d = t
            t = x - hi
            if t > d:
                d = t

            if d > max_dist:
                max_dist = d
                if max_dist >= min_dist:
                    break
        if max_dist < min_dist:
            min_dist = max_dist
    return min_dist

@numba.njit(cache=True, fastmath=True)
def min_dist(b1, b2):
    """L_inf distance between two boxes: max per-dimension gap."""
    max_gap = 0.0
    for i in range(b1.shape[1]):
        lo1, hi1 = b1[0, i], b1[1, i]
        lo2, hi2 = b2[0, i], b2[1, i]

        gap = 0.0
        if hi1 < lo2:
            gap = lo2 - hi1
        elif hi2 < lo1:
            gap = lo1 - hi2

        if gap > max_gap:
            max_gap = gap

    return max_gap

@numba.njit(cache=True, fastmath=True)
def overlaps_batch(obox, boxes):
    # boxes: (N, 2, D)
    out = np.empty(boxes.shape[0], dtype=np.bool_)
    for i in range(boxes.shape[0]):
        ok = True
        for d in range(obox.shape[1]):
            if obox[0, d] >= boxes[i, 1, d] or boxes[i, 0, d] >= obox[1, d]:
                ok = False
                break
        out[i] = ok
    return out

    
@numba.njit(cache=True, fastmath=True)
def overlaps(obox, ibox):
    for d in range(obox.shape[1]):
        if obox[0, d] >= ibox[1, d] or ibox[0, d] >= obox[1, d]:
            return False
    return True

@numba.njit(cache=True, fastmath=True)
def equals(obox, ibox):
    for d in range(obox.shape[1]):
        if obox[0, d] != ibox[0, d] or obox[1, d] != ibox[1, d]:
            return False
    return True
