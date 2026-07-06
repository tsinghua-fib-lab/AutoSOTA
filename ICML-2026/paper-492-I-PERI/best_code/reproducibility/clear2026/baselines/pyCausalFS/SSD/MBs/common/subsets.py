import itertools
from itertools import combinations, chain

def subsets(nbrs, k):
    return set(combinations(nbrs, k))

