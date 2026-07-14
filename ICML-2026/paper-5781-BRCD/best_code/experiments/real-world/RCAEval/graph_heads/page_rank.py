import networkx as nx
import numpy as np
from causallearn.graph.GraphClass import CausalGraph
from sknetwork.ranking import PageRank


def page_rank_preprocess(adj):
    pr_input = np.zeros_like(adj)

    node_num = len(adj)

    for a in range(node_num):
        for b in range(node_num):
            # case 1 no edge: a b
            if adj[a, b] == adj[b, a] == 0:
                pass

            # case 2 undirected a -- b
            elif adj[a, b] == adj[b, a] == -1:
                pr_input[a, b] = pr_input[b, a] = 1

            # case 3 directed a -> b
            elif adj[a, b] == 1 and adj[b, a] == -1:
                pr_input[a, b] = 1
                # pr_input[b, a] = 0

            # case 4 directed a <- b
            elif adj[a, b] == -1 and adj[b, a] == 1:
                # pr_input[a, b] = 0
                pr_input[b, a] = 1
            elif adj[a, b] == 0 and adj[b, a] == 1:
                # already ok
                pr_input[a, b] = 0
                pr_input[b, a] = 1
            elif adj[a, b] == 1 and adj[b, a] == 0:
                # already ok
                pr_input[a, b] = 1
                pr_input[b, a] = 0
            elif adj[a, b] == 1 and adj[b, a] == 1:
                # a <-> b, in FCI
                pr_input[a, b] = 1
                pr_input[b, a] = 1
            elif adj[a, b] == 2 and adj[b, a] == 1:
                # a o-> b, in FCI
                # hmm, we will make it a->b
                pr_input[a, b] = 1
                pr_input[b, a] = 0
            elif adj[a, b] == 1 and adj[b, a] == 2:
                # a <-0 b, in FCI
                # hmm, we will make it a<-b
                pr_input[a, b] = 0
                pr_input[b, a] = 1

            elif adj[a, b] == 2 and adj[b, a] == 2:
                # a o-o b, in FCI
                # hmm, we will make it a<->b
                pr_input[a, b] = 1
                pr_input[b, a] = 1

            else:
                # 1 and 1 ?? what the hell?
                raise ValueError(f"Unexpected value: {adj[a, b]}, {adj[b, a]}")
    return pr_input


def page_rank(adj, node_names=None, damping_factor=0.85, solver="piteration", n_iter=10, tol=1e-6):
    if node_names is None:
        node_names = [f"X{i}" for i in range(len(adj))]

    pr_input = page_rank_preprocess(adj)
    # # Change the adj to graph
    # G = nx.DiGraph()
    # for i in range(len(adj)):
    #     for j in range(len(adj)):
    #         if adj[i,j] == -1:
    #             G.add_edge(i,j)
    #         if adj[i,j] == 1:
    #             G.add_edge(j,i)
    # nodes = sorted(G.nodes())
    # pr_input = np.asarray(nx.to_numpy_matrix(G, nodelist=nodes))
    # print(f"Number of element equal to -1, 0, 1 in pr_input: ")
    # print(np.sum(pr_input == -1), np.sum(pr_input == 0), np.sum(pr_input == 1))
    # print(pr_input.shape)

    pr = PageRank(damping_factor=damping_factor, solver=solver, n_iter=n_iter, tol=tol)
    # pr = PageRank()

    # transpose before fit
    scores = pr.fit_transform(pr_input)

    # merge scores and node names, sort by scores
    output = list(zip(node_names, scores))
    output.sort(key=lambda x: x[1], reverse=True)
    return output
