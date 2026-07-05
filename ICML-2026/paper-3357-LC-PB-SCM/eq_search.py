import numpy as np
from itertools import product
from util import mag2graph
from copy import copy, deepcopy
import re


# 0 not adjacent, 1/-1 direct edge, 2 bi-direct edge, 3 uncertained
def vertex_state(u:int, v:int, edge_mat) -> int:
    if 2 in (edge_mat[u][v], edge_mat[v][u]):
        return 3

    if edge_mat[u][v] == 1 and edge_mat[v][u] == -1:
        return 1
    if edge_mat[u][v] == -1 and edge_mat[v][u] == 1:
        return -1

    if edge_mat[u][v] == 1 and edge_mat[v][u] == 1:
        return 2

    if edge_mat[u][v] == 0 and edge_mat[v][u] == 0:
        return 0

def list2term(l:list[tuple[int, int]]) -> frozenset:
    ll = l.copy()
    rec = {}
    for s, n in ll:
        rec.update({s: n})
    return frozenset(rec.items())

def pgf2terms(pgf:str) -> set[frozenset]:
    # Just for test
    pattern = r"(?<![\w\*])[+\-]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+\-]?\d+)?\s*\*?\s*|(?<=^)[+\-]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+\-]?\d+)?\s*$|[\s\+\-]\s*[+\-]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+\-]?\d+)?\s*$"
    pgf = re.sub(pattern, "", pgf)
    pgf = pgf.strip()
    pgf = pgf.replace(" ", "")
    pgf = pgf.replace("**", "^")

    terms = set()
    for term in pgf.split('+'):
        mterm = {}
        for item in term.split('*'):
            if not item.startswith('s_'):
                continue
            if '^' in item:
                var, order = item.split('^')
                order = int(order)
            else:
                var, order = item, 1
            var = int(var[2:])
            mterm[var] = order
        terms.add(frozenset(mterm.items()))
    return terms

def eq_search_v2(n:int, terms:set[frozenset]):
    '''
    This function takes a set of terms and returns a mixed graph as a list of lists of integers.

    Parameters
    ----------
    n : int
        The number of variables in the polynomial.
    terms : set[dict[int, int]]
        A set of terms, where each term is represented as a dictionary mapping variable indices to their exponents.

    Returns
    -------
    list[list[int]]
        A adjacency list representation of a mixed graph.
    '''

    rec = np.zeros((n, n), dtype=np.int32)
    # rule1, skeleton search
    for u in range(n):
        for v in range(u+1, n):
            if list2term([(u+1, 1), (v+1, 1)]) in terms:
                rec[u][v] = 2
                rec[v][u] = 2\

    # rule2, si*sj^2
    # 1:u->v, -1:v->u, 0: not triangle
    def r2(u:int, v:int) -> int:
        if list2term([(u, 1), (v, 2)]) in terms:
            return 1
        elif list2term([(u, 2), (v, 1)]) in terms:
            return -1
        else:
            return 0

    for u in range(n):
        for v in range(u+1, n):
            if r2(u+1, v+1) == 1:
                rec[u][v], rec[v][u] = 1, -1
            elif r2(u+1, v+1) == -1:
                rec[v][u], rec[u][v] = 1, -1

    # unshield collider rule
    flag = True
    while flag:
        flag = False
        for u, v, w in product(range(n), repeat=3):
            # u->v.-.w   uw is not adjacent
            if rec[u][v]==1 and rec[v][u]==-1 and rec[w][v]==2 and (rec[u][w]|rec[w][u])==0:
                if list2term([(u+1, 1), (v+1, 1), (w+1, 1)]) in terms:
                    rec[v][w], rec[w][v] = 1, -1
                    flag = True
                else:
                    rec[w][v] = 1

    # shield collider
    for u, v, w in product(range(n), repeat=3):
        if vertex_state(u, v, rec) == 3 and vertex_state(v, w, rec) == 3 and vertex_state(u, w, rec) == 3:
            if list2term([(u+1, 1), (v+1, 1), (w+1, 2)]) in terms:
                rec[u][w] = 1
                rec[v][w] = 1

    # rule 3
    for u, v, w in product(range(n), repeat=3):
        if u == v or u == w or v == w:
            continue
        if 2 not in (rec[u][v], rec[v][w], rec[w][u], rec[u][w], rec[v][u], rec[w][v]):
            continue
        for d, dd in product(range(1, 3), range(1, 3)):
            if list2term([(u+1, d), (v+1, dd), (w+1, 3)]) in terms:
                # .-> The acyclicity assumption rules out one edge, namely the case
                # where u, v, and w share the same confounder.
                if r2(u+1, v+1)==1 and r2(v+1, w+1)==1:
                    rec[u][w] = 1
                    continue

                if r2(v+1, u+1)==1 and r2(u+1, w+1)==1:
                    rec[v][w] = 1
                    continue

                # .-. The start point cannot be determined, similar to a purely observed triangle structure.
                if r2(u+1, w+1)==1 and r2(v+1, w+1)==1:
                    continue

                # The remaining cases are fully identifiable from the degrees; the guard avoids conflicts.
                if rec[u][w] != -1 and rec[w][u] != 1:
                    rec[u][w], rec[w][u] = 1, -1
                if rec[v][w] != -1 and rec[w][v] != 1:
                    rec[v][w], rec[w][v] = 1, -1

                if dd > d:
                    if rec[u][v] != -1 and rec[v][u] != 1:
                        rec[u][v], rec[v][u] = 1, -1
                elif d > dd:
                    if rec[v][u] != -1 and rec[u][v] != 1:
                        rec[v][u], rec[u][v] = 1, -1
                else:
                    # Triangle structure plus a confounder between the start and end nodes.
                    if r2(u+1, w+1)==1 and r2(v+1, w+1)==0 and r2(u+1, v+1)==0:
                        rec[u][v], rec[v][u] = 1, -1
                        rec[v][w], rec[w][v] = 1, -1
                    if r2(v+1, w+1)==1 and r2(u+1, w+1)==0 and r2(u+1, v+1)==0:
                        rec[v][u], rec[u][v] = 1, -1
                        rec[u][w], rec[w][u] = 1, -1

    # rule 4 (derived from r2)
    for u in range(n):
        for v in range(u+1, n):
            st = r2(u+1, v+1)
            if st == 0:
                continue
            # u,v is directed by rule2
            if st == 1:
                l = [(u+1, 1), (v+1, 2)]
            else:
                l = [(u+1, 2), (v+1, 1)]


            for w in range(n):
                if w == u or w == v:
                    continue

                # Adjacent to the endpoint.
                l.append((w+1, 2))
                # If it is not a child, one edge can be ruled out.
                if list2term(l) not in terms:
                    e_out = False
                    if st == 1 and rec[w][v] == 2:
                        rec[w][v] = 1
                    elif st == -1 and rec[w][u] == 2:
                        rec[w][u] = 1
                # If it is a child, the edge can be determined.
                else:
                    if st == 1 and 2 in (rec[w][v], rec[v][w]):
                        rec[v][w] = 1
                        # If this is not a three-node cycle, it can be determined as a directed edge.
                        if vertex_state(w, u, rec) == 0:
                            rec[w][v] = -1
                        elif r2(u+1, w+1) != 1:
                            rec[w][v] = -1
                    elif st == -1 and 2 in (rec[w][u], rec[u][w]):
                        rec[u][w] = 1
                        if vertex_state(w, v, rec) == 0:
                            rec[w][u] = -1
                        elif r2(v+1, w+1) != 1:
                            rec[w][u] = -1
                    pass
                l.pop()

                # Adjacent to the start point.
                # If it is not a child, one edge can be ruled out.
                l.append((w+1, 1))
                if list2term(l) not in terms:
                    s_out = False
                    if st == 1 and rec[w][u] == 2:
                        rec[w][u] = 1
                    elif st == -1 and rec[w][v] == 2:
                        rec[w][v] = 1
                # If it is a child, the edge can be determined.
                else:
                    if st == 1 and rec[u][w] == 2 and vertex_state(v, w, rec) == 0:
                        rec[u][w], rec[w][u] = 1, -1
                    elif st == -1 and rec[v][w] == 2 and vertex_state(u, w, rec) == 0:
                        rec[v][w], rec[w][v] = 1, -1
                l.pop()

                # If outgoing edges from both the start and end points are ruled out,
                # there are two <-> edges.
                # Another interpretation is that there is no ternary quadratic term (112).
                if st == 1:
                    if vertex_state(u, w, rec)==3 and vertex_state(w, v, rec)==3 and list2term([(u+1, 1), (v+1, 2), (w+1, 1)]) not in terms:
                        rec[u][w], rec[w][u] = 1, 1
                        rec[v][w], rec[w][v] = 1, 1
                else:
                    if vertex_state(u, w, rec)==3 and vertex_state(w, v, rec)==3 and list2term([(u+1, 2), (v+1, 1), (w+1, 1)]) not in terms:
                        rec[u][w], rec[w][u] = 1, 1
                        rec[v][w], rec[w][v] = 1, 1


    # One <-> edge.
    for u, v, w in product(range(n), repeat=3):
        if u == v or u == w or v == w:
            continue

        if vertex_state(u, w, rec) != 3:
            continue

        # One <-> edge, case 1.
        if r2(u+1, v+1)==1 and vertex_state(v, w, rec)==1 and r2(u+1, w+1)==0 and vertex_state(u, w, rec)==3:
            # if list2term([(u+1, 1), (v+1, 1), (w+1, 3)]) in terms:
            #     continue
            # if list2term([(u+1, 2), (v+1, 1), (w+1, 3)]) in terms:
            #     continue
            if list2term([(u+1, 1), (v+1, 2), (w+1, 3)]) in terms:
                continue

            rec[u][w], rec[w][u] = 1, 1

        # One <-> edge, case 2.
        if r2(u+1, v+1)==1 and r2(v+1, w+1)==-1 and r2(u+1, w+1)==0 and vertex_state(u, w, rec)==3:
            if list2term([(u+1, 1), (v+1, 3), (w+1, 1)]) in terms:
                continue

            rec[u][w], rec[w][u] = 1, 1

    for u, v, w in product(range(n), repeat=3):
        if u == v or u == w or v == w:
            continue
        # if rec[u][v]|rec[v][u]|rec[u][w]|rec[w][u]|rec[v][w]|rec[w][v]:
        #     continue
        # Three <-> edges.
        if vertex_state(u, v, rec)==3 and vertex_state(u, w, rec)==3 and vertex_state(w, v, rec)==3:
            if list2term([(u+1, 1), (v+1, 1), (w+1, 1)]) not in terms:
                rec[u][v], rec[v][u] = 1, 1
                rec[u][w], rec[w][u] = 1, 1
                rec[v][w], rec[w][v] = 1, 1
        # unshielded collider
        if vertex_state(u, v, rec)==3 and vertex_state(u, w, rec)==3 and vertex_state(w, v, rec)==0:
            if list2term([(u+1, 1), (v+1, 1), (w+1, 1)]) not in terms:
                rec[v][u] = 1
                rec[w][u] = 1

    return rec


if __name__ == '__main__':
    # print(list2term([(0, 1), (1, 2), (2, 3)]))
    pgf = 's_1*s_2*s_3**2 + s_1*s_2*s_3 + s_1*s_2 + s_1*s_3 + s_1 + s_2*s_3**2 + s_2*s_3 + s_2 + s_3'
    terms = pgf2terms(pgf)
    mag = eq_search_v2(3, terms)
    print(mag)
    print(mag2graph(mag))
