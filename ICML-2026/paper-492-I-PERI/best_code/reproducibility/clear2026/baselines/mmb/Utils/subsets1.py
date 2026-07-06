from itertools import combinations
def subsets1(Canlist, cutSetSize):
    #Take out all subsets of size cutSetSize from Canlist.
    subsets_tuples = list(combinations(Canlist, cutSetSize))
    subsets_lists = [list(subset) for subset in subsets_tuples]
    return subsets_lists

