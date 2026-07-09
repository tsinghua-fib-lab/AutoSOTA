import networkx as nx
import math
import helper_functions

INF = 1e18

class TreeDP:
    def __init__(self, T, tau, k):
        self.T = T
        self.tau = tau
        self.k = k
        self.n = len(T)
        self.DP = [[-INF] * self.n for _ in range(k+1)]
        self.backtrack = [[0] * self.n for _ in range(k+1)]
        self.L = []


    def _edge_bound(self, w, x):
        if x < -w:
            return -INF
        if x > w:
            return w
        return x


    def dfs(self, v):
        if len(self.T[v]) == 0:
            self.DP[0][v] = -self.tau
            for i in range(1,self.k+1): self.DP[i][v] = INF
            return 1

        
        leftsubtree = self.dfs(self.T[v][0][0])
        rightsubtree = 0
        if (len(self.T[v])==2): rightsubtree = self.dfs(self.T[v][1][0])
        subtreesize = leftsubtree + rightsubtree

        for l in range(min(self.k, subtreesize)+1):
            if len(self.T[v]) == 1:
                child, weight = self.T[v][0]
                self.DP[l][v] = max(
                    self.DP[l][v], self._edge_bound(weight, self.DP[l][child])
                )

            if len(self.T[v]) == 2:
                (left_child, left_w), (right_child, right_w) = self.T[v]

                upper = min(leftsubtree+1, l+1)
                lower = max(0, l - rightsubtree)

                for a in range(lower,upper):
                    tmp = (self._edge_bound(left_w, self.DP[a][left_child])
                            + self._edge_bound(right_w, self.DP[l - a][right_child]))

                    if (self.DP[l][v] < tmp):
                        self.DP[l][v] = tmp
                        self.backtrack[l][v] = a

            if len(self.T[v]) > 2:
                assert False
        return subtreesize



    def solve(self, root=0):
        return self.dfs(root)

    def recoverL(self, v, k):
        if len(self.T[v]) == 0:
            if k > 0: self.L.append(v)
            return
        
        if len(self.T[v]) == 1:
            self.recoverL(self.T[v][0][0], k)
            return
        
        if len(self.T[v]) == 2:
            self.recoverL(self.T[v][0][0], self.backtrack[k][v])
            self.recoverL(self.T[v][1][0], k-self.backtrack[k][v])
            return

        assert(False)


def solveGivenK(T, k, root=0):
    # We expect: a Tree, given as adjacency list, 0-indexed. And some integer k.
    # T[a][0] = [b, w]

    l = 0
    r = 10
    while (r - l > 1e-4):
        # print(l,r, flush=True)
        tau = (l+r)/2

        solver = TreeDP(T, tau, k)
        n = solver.solve(root=root)
        if (solver.DP[k][root]<0): r = tau
        else: l = tau

    solver = TreeDP(T, l, k)
    solver.solve(root=root)
    solver.recoverL(root, min(k, n))
    # print(solver.DP)

    return solver.L


if __name__ == "__main__":
    import argparse
    import pickle
    import __main__
    import sparsifier
    __main__.Sparsifier = sparsifier.Sparsifier

    n_processors = 32
    parser = argparse.ArgumentParser()
    parser.add_argument("graph", help="Path to graph file")
    parser.add_argument("sparsifier", help="Path to sparsifier file")
    parser.add_argument("k", type=int, help="k")
    args = parser.parse_args()

    G = helper_functions.read_graph(args.graph)
    with open(args.sparsifier, "rb") as f:
        s = pickle.load(f)
    k = args.k

    if k >= G.number_of_nodes():
        raise Exception("k must be smaller than n!")

    L = solveGivenK(s.tree, k, root=s.root)
    Lmapped = [res for key,res in s.mapping.items() if key in L]

    tau,cut = helper_functions.cut_set(G, Lmapped)
    print(tau)