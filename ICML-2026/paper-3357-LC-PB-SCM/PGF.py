import numpy as np
import sympy as sp
from sympy import Symbol as S
from sympy import Basic, Poly,  pprint, init_printing

# Generate the PGF expression from the graph structure.
# input: graph: np.ndarray, graph[i][j]: double denotes the edge weight from i to j.
class PGF():
    @staticmethod
    def thin(expr:Basic, s:S, rho:S, rho_s:S) -> Basic:
        return expr.subs(s, s*(rho*rho_s+1-rho))

    @staticmethod
    def pois(l:S, s:S):
        return sp.exp(l*(s-1))


    @staticmethod
    def topo(graph):
        from collections import deque
        in_degree = {node: 0 for node in range(len(graph))}  # Initialize each node's in-degree to 0.
        for u in range(len(graph)):
            for v in range(len(graph)):
                if graph[u][v] != 0 :in_degree[v] += 1  # Compute each node's in-degree.
        queue = deque([node for node in range(len(graph)) if in_degree[node] == 0])  # Add nodes with in-degree 0 to the queue.
        result = []

        while queue:
            u = queue.popleft()
            result.append(u)

            for v in range(len(graph)):
                if graph[u][v] != 0:
                    in_degree[v] -= 1  # Remove the current node's outgoing edge and decrement the neighbor's in-degree.
                    if in_degree[v] == 0:  # Add the neighbor to the queue if its in-degree becomes 0.
                        queue.append(v)

        return result

    def __init__(self, graph:np.ndarray):
        self.graph = graph
        self.n = len(graph)

        # s_1, ..., s_n denote the symbolic variables corresponding to the random variables.
        # mu_1, ..., mu_n denote the Poisson noise parameters corresponding to the random variables.
        # e_12, ..., e_n,n-1 denote edge weights.
        self.mu = sp.symbols(' '.join([f'mu_{i+1}' for i in range(self.n)]))
        self.s = sp.symbols(' '.join([f's_{i+1}' for i in range(self.n)]))
        self.e = [[sp.Symbol(f'e_{i+1},{j+1}')
                    for j in range(self.n)]
                    for i in range(self.n)]
        self.expr:Basic = 0


        for i in range(self.n):
            self.expr += self.mu[i] * (self.s[i]-1)

        order = self.topo(graph)
        for i in range(self.n):
            for j in range(i+1, self.n):
                u, v = order[i], order[j]
                if graph[u][v] != 0:
                    self.expr = self.thin(self.expr, self.s[u], self.e[u][v], self.s[v])
                    # tmp = Poly(self.thin(self.expr, self.s[u], self.e[u][v], self.s[v]), self.s)
                    # self.expr = tmp.as_expr()


    def compute(self, mu:list, s:list=None, e:list=None) -> Poly:
        if e is None:
            e = self.graph
        subs_dict = {}
        subs_dict.update({self.s[i]: s[i] for i in range(self.n)} if s != None else {})
        subs_dict.update({self.mu[i]: mu[i] for i in range(self.n)})
        subs_dict.update({self.e[i][j]: e[i][j] for i in range(self.n) for j in range(self.n) if e[i][j] != 0})
        rec = self.expr
        if isinstance(rec, sp.Mul):
            rec = sum(arg.args[0] for arg in rec.args if isinstance(arg, sp.exp))
        elif isinstance(rec, sp.exp):
            rec = rec.args[0]
        poly = Poly(rec.subs(subs_dict), self.s)
        return poly


if __name__ == '__main__':
    graph = [
        [0, 0, 0, 0, 0],
        [0, 0, 0.5, 0, 0],
        [0.5, 0, 0, 0, 0],
        [0, 0.5, 0.5, 0, 0],
    ]
    pgf = PGF(graph).compute([0.3, 0.5, 0.4, 0.2])
    print(pgf.as_expr())
