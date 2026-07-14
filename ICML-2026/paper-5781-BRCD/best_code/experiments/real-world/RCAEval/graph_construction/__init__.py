import numpy as np

from .fci import fci_default
from .ges import ges
from .granger import granger
from .lingam import DirectLiNGAM, ICALiNGAM
from .pc import pc_default


def normalize_adj(adj):
    norm_adj = np.zeros_like(adj)

    node_num = len(adj)

    for a in range(node_num):
        for b in range(node_num):
            # case 1 no edge: a b
            if adj[a, b] == adj[b, a] == 0:
                pass

            # case 2 undirected a -- b
            elif adj[a, b] == adj[b, a] == -1:
                norm_adj[a, b] = norm_adj[b, a] = 1

            # case 3 directed a -> b
            elif adj[a, b] == 1 and adj[b, a] == -1:
                norm_adj[a, b] = 1
                # pr_input[b, a] = 0

            # case 4 directed a <- b
            elif adj[a, b] == -1 and adj[b, a] == 1:
                # pr_input[a, b] = 0
                norm_adj[b, a] = 1
            elif adj[a, b] == 0 and adj[b, a] == 1:
                # already ok
                norm_adj[a, b] = 0
                norm_adj[b, a] = 1
            elif adj[a, b] == 1 and adj[b, a] == 0:
                # already ok
                norm_adj[a, b] = 1
                norm_adj[b, a] = 0
            elif adj[a, b] == 1 and adj[b, a] == 1:
                # a <-> b, in FCI
                norm_adj[a, b] = 1
                norm_adj[b, a] = 1
            elif adj[a, b] == 2 and adj[b, a] == 1:
                # a o-> b, in FCI
                # hmm, we will make it a->b
                norm_adj[a, b] = 1
                norm_adj[b, a] = 0
            elif adj[a, b] == 1 and adj[b, a] == 2:
                # a <-0 b, in FCI
                # hmm, we will make it a<-b
                norm_adj[a, b] = 0
                norm_adj[b, a] = 1

            elif adj[a, b] == 2 and adj[b, a] == 2:
                # a o-o b, in FCI
                # hmm, we will make it a<->b
                norm_adj[a, b] = 1
                norm_adj[b, a] = 1

            else:
                # 1 and 1 ?? what the hell?
                raise ValueError(f"Unexpected value: {adj[a, b]}, {adj[b, a]}")
    return norm_adj