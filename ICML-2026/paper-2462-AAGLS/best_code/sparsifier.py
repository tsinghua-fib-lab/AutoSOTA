import networkx as nx
import metis
from scipy.sparse.linalg import eigsh
import numpy as np
import math
import bisectalgos
import helper_functions
import sys

class Sparsifier:
    def __init__(self, G, alg="spectral"):
        self.tree = []
        self.mapping = dict()

        self.alg=alg

        self.Gorig = G
        self.G = nx.convert_node_labels_to_integers(G, label_attribute="orig")
        
        self.root = self.decomp(self.G)
        self.weight(self.root)


    def bisect(self, G):
        if (self.alg=="spectral"):
            return bisectalgos.spectral_bisect(G)
        if (self.alg=="spectralbalanced_10"):
            return bisectalgos.spectral_bisect(G, balancedfactor=0.1)
        if (self.alg=="spectralbalanced_1"):
            return bisectalgos.spectral_bisect(G, balancedfactor=0.01)
        if (self.alg=="metismulti_0"):
            return bisectalgos.metismulti_bisect(G, m=10)
        if (self.alg=="metismulti_1"):
            return bisectalgos.metismulti_bisect(G, m=max(int(round(1*math.sqrt(G.number_of_nodes()))), 4))
        if (self.alg=="metismulti_10"):
            return bisectalgos.metismulti_bisect(G, m=max(int(round(10*math.sqrt(G.number_of_nodes()))), 4))
        if (self.alg=="metismulti_100"):
            return bisectalgos.metismulti_bisect(G, m=max(int(round(100*math.sqrt(G.number_of_nodes()))), 4))

        raise(Exception("Specified bisectalgo not found!"))

    def decomp(self, G):
        if G.number_of_nodes() == 1:
            index = len(self.tree)
            self.mapping[index] = G.nodes[list(G.nodes)[0]]["orig"]
            self.tree.append([])
            return index
        else:
            L,R = self.bisect(G)
            l = self.decomp(L)
            r = self.decomp(R)

            index = len(self.tree)
            self.tree.append([[l], [r]])

            return index

    def weight(self, v):
        if len(self.tree[v]) == 0:
            return len(list(nx.edge_boundary(self.Gorig, [self.mapping[v]]))), [self.mapping[v]]
        else: 
            l,L = self.weight(self.tree[v][0][0])
            r,R = self.weight(self.tree[v][1][0])
            self.tree[v][0].append(l)
            self.tree[v][1].append(r)

            return l+r-len(list(nx.edge_boundary(self.Gorig, L, R)))*2, L+R


if __name__ == "__main__":
    import argparse
    import pickle

    parser = argparse.ArgumentParser()
    parser.add_argument("graph", help="Path to graph file")
    parser.add_argument("bisectalgo", help="bisectalgorithm to use")
    args = parser.parse_args()

    G = helper_functions.read_graph(args.graph)
    bisectalgo = args.bisectalgo

    s = Sparsifier(G, alg=bisectalgo)

    payload = pickle.dumps(s)
    sys.stdout.buffer.write(payload)
    sys.stdout.buffer.flush()
