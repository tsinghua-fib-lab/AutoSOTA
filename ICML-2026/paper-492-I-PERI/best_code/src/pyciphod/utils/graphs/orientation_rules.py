from pyciphod.utils.time_series.data_format import DTimeVar


def uc_rule(g, sepset):
    """
        Apply the Unshielded Collider (UC) rule:
        For each unshielded triple x - y - z with x and z not adjacent,
        orient x -> y <- z if y not in sepset(x, z).
    """
    nodes = g.get_vertices()
    adj = {x: g.get_adjacencies(x) for x in nodes}

    for x in nodes:
        for y in sorted(adj[x]):
            if y.time != 0:
                continue

            for z in sorted(adj[y]):
                if z.time != 0:
                    continue

                if z == x or z in adj[x]:
                    continue

                if y not in sepset.get((x, z), []):
                    g.remove_undirected_edge(x, y)
                    g.add_directed_edge(x, y)

                    g.remove_undirected_edge(z, y)
                    g.add_directed_edge(z, y)

def meek_rule_1(g, X, Y, Z):
    """
    :param X: a vertex in g
    :param Y:  a vertex in g
    :param Z:  a vertex in g
    :return:  True if the conditions of Meek's Rule 1 are satisfied for the triple (X, Y, Z) in g, False otherwise.
     Meek's Rule 1 states that if there is a directed edge from X to Y (X -> Y) and an undirected edge between Y and Z (Y - Z), and there is no edge between X and Z (X and Z are not adjacent), then we can orient the edge between Y and Z as Y -> Z.
    """
    if ((X, Y) in g.get_directed_edges()) and ((
    Y, Z) in g.get_undirected_edges()) and (Z not in g.get_adjacencies(X)):
        return True
    return False


def meek_rule_2(g, X, Y, Z):
    """
    :param X: a vertex in g
    :param Y:  a vertex in g
    :param Z:  a vertex in g
    :return:  True if the conditions of Meek's Rule 2 are satisfied for the triple (X, Y, Z) in g, False otherwise.
     Meek's Rule 2 states that if there is a directed edge from X to Y (X -> Y) and a directed edge from Y to Z (Y -> Z), and there is an undirected edge between X and Z (X - Z), then we can orient the edge between X and Z as X -> Z.
    """
    if (X, Y) in g.get_directed_edges() and (Y, Z) in g.get_directed_edges() and (
    X, Z) in g.get_undirected_edges():
        return True
    return False


def meek_rule_3(g, X, Y, Z, W):
    """
    :param g: a pattern of a CPDAG
    :param X: a vertex in g
    :param Y:  a vertex in g
    :param Z:  a vertex in g
    :return:  True if the conditions of Meek's Rule 3 are satisfied for the triple (X, Y, Z) and vertex W in g, False otherwise.
     Meek's Rule 3 states that if there is a directed edge from X to Y (X -> Y) and a directed edge from Z to Y (Z -> Y), and there is an undirected edge between X and Z (X - Z), and there exists a vertex W such that there are undirected edges between W and Y (W - Y), W and X (W - X), and W and Z (W - Z), then we can orient the edge between X and Z as X -> Z.
    """
    if (X, Y) in g.get_directed_edges() and (
    Z, Y) in g.get_directed_edges() and Z not in g.get_adjacencies(X):
        if (W, Y) in g.get_undirected_edges() and (W, X) in g.get_undirected_edges() and (
                W, Z) in g.get_undirected_edges():
            return True
    return False


def meek_rule_4(g, X, Y, Z, W):
    """
    :param X: a vertex in g
    :param Y: a vertex in g
    :param Z: a vertex in g
    :param W: a vertex in g
    :return: True if the conditions of Meek's Rule 4 are satisfied for the quadruple (X, Y, Z, W) in g, False otherwise.
     Meek's Rule 4 states that if there is an undirected edge between X and Y (X - Y), a directed edge from X to W (X -> W),
     an undirected edge between Y and W (Y - W), an undirected edge between Y and Z (Y - Z), and a directed edge from W to Z
     (W -> Z), then we can orient the edge between Y and Z as Y -> Z.
    """
    if (X, Y) in g.get_undirected_edges() and (X, W) in g.get_directed_edges() and (
    Y, W) in g.get_undirected_edges() and (Y, Z) in g.get_undirected_edges() and (W, Z) in g.get_directed_edges():
        return True
    return False


def is_contemporaneous(x: DTimeVar, y: DTimeVar) -> bool:
    """
    Return True if x and y belong to the same time slice.
    """
    if not isinstance(x, DTimeVar) or not isinstance(y, DTimeVar):
        return False

    return x.time == y.time


def time_orientation(g):
    """Orient lagged edges according to temporal order."""
    for x, y in list(g.get_undirected_edges()):
        if x.time < y.time:
            g.remove_undirected_edge(x, y)
            g.add_directed_edge(x, y)

        elif y.time < x.time:
            g.remove_undirected_edge(x, y)
            g.add_directed_edge(y, x)

def ts_meek_rule_1(d, X, Y, Z):
    """
    :param X: a vertex in d
    :param Y:  a vertex in d
    :param Z:  a vertex in d
    :return:  True if the conditions of Meek's Rule 1 are satisfied for the triple (X, Y, Z) in d, False otherwise.
     Meek's Rule 1 states that if there is a directed edge from X to Y (X -> Y) and an undirected edge between Y and Z (Y - Z), and there is no edge between X and Z (X and Z are not adjacent), then we can orient the edge between Y and Z as Y -> Z.
     In the temporal setting, this rule is only applied when Y and Z belong to the same time slice.
    """
    return is_contemporaneous(Y, Z) and meek_rule_1(d, X, Y, Z)


def ts_meek_rule_2(d, X, Y, Z):
    """
    :param X: a vertex in d
    :param Y:  a vertex in d
    :param Z:  a vertex in d
    :return:  True if the conditions of Temporal Meek's Rule 2 are satisfied for the triple (X, Y, Z) in d, False otherwise.
     Meek's Rule 2 states that if there is a directed edge from X to Y (X -> Y) and a directed edge from Y to Z (Y -> Z), and there is an undirected edge between X and Z (X - Z), then we can orient the edge between X and Z as X -> Z.
     In the temporal setting, this rule is only applied when X and Z belong to the same time slice.
    """
    return is_contemporaneous(X, Z) and meek_rule_2(d, X, Y, Z)


def ts_meek_rule_3(d, X, Y, Z, W):
    """
    :param X: a vertex in d
    :param Y:  a vertex in d
    :param Z:  a vertex in d
    :return:  True if the conditions of Temporal Meek's Rule 3 are satisfied for the triple (X, Y, Z) and vertex W in d, False otherwise.
     Meek's Rule 3 states that if there is a directed edge from X to Y (X -> Y) and a directed edge from Z to Y (Z -> Y), and there is an undirected edge between X and Z (X - Z), and there exists a vertex W such that there are undirected edges between W and Y (W - Y), W and X (W - X), and W and Z (W - Z), then we can orient the edge between X and Z as X -> Z.
     In the temporal setting, this rule is only applied when W and Y belong to the same time slice.
    """
    return is_contemporaneous(W, Y) and meek_rule_3(d, X, Y, Z, W)


def ts_meek_rule_4(d, X, Y, Z, W):
    """
    :param X: a vertex in d
    :param Y: a vertex in d
    :param Z: a vertex in d
    :param W: a vertex in d
    :return: True if the conditions of Temporal Meek's Rule 4 are satisfied for the quadruple (X, Y, Z, W) in d, False otherwise.
     Meek's Rule 4 states that if there is an undirected edge between X and Y (X - Y), a directed edge from X to W (X -> W),
     an undirected edge between Y and W (Y - W), an undirected edge between Y and Z (Y - Z), and a directed edge from W to Z
     (W -> Z), then we can orient the edge between Y and Z as Y -> Z.
     In the temporal setting, this rule is only applied when Y and Z belong to the same time slice.
    """
    return is_contemporaneous(Y, Z) and meek_rule_4(d, X, Y, Z, W)


def apply_ts_meek_rules(g):
    """
        Apply Meek's orientation rules iteratively using only edge sets:
        Rule 1, Rule 2, Rule 3, Rule 4 for propagating orientations.
        Returns True if any edge was oriented, False otherwise.
    """
    nodes = g.get_vertices()
    adj = {x: g.get_adjacencies(x) for x in nodes}
    changed = False

    for x in nodes:
        for y in sorted(adj[x]):
            for z in sorted(set(adj[y]) - set(adj[x])):
                if ts_meek_rule_1(g, x, y, z):
                    changed = True
                    g.remove_undirected_edge(y, z)
                    g.add_directed_edge(y, z)

            for z in sorted(set(adj[y]) & set(adj[x])):
                if ts_meek_rule_2(g, x, y, z):
                    changed = True
                    g.remove_undirected_edge(x, z)
                    g.add_directed_edge(x, z)

    for x in nodes:
        for y in sorted(adj[x]):
            if (x, y) not in g.get_directed_edges():
                continue

            for z in sorted(set(adj[y]) - set(adj[x])):
                if (z, y) not in g.get_directed_edges():
                    continue

                for w in sorted(set(adj[x]) & set(adj[y]) & set(adj[z])):
                    if ts_meek_rule_3(g, x, y, z, w):
                        changed = True
                        g.remove_undirected_edge(w, y)
                        g.add_directed_edge(w, y)

    for x in nodes:
        for y in sorted(adj[x]):
            if not (x, y) in g.get_undirected_edges():
                continue

            for w in sorted(set(adj[x]) & set(adj[y])):
                if not (x, w) in g.get_directed_edges():
                    continue

                if not (y, w) in g.get_undirected_edges():
                    continue

                for z in sorted(set(adj[y]) & set(adj[w])):
                    if ts_meek_rule_4(g, x, y, z, w):
                        changed = True
                        g.remove_undirected_edge(y, z)
                        g.add_directed_edge(y, z)

    return changed