import io

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from causallearn.graph.GraphClass import CausalGraph
from causallearn.utils.GraphUtils import GraphUtils

from RCAEval.classes.graph import MemoryGraph
from RCAEval.io.time_series import drop_time


def customize_topological_generations(G):
    """
    - This function just roughly show the graph, not the true graph
    - This function is a customize version of networkx.topological_generations to accept circles
    - Better check on kiali dashboard for the precise graph
    """
    if not G.is_directed():
        raise nx.NetworkXError("Topological sort not defined on undirected graphs.")

    multigraph = G.is_multigraph()
    indegree_map = {v: d for v, d in G.in_degree() if d > 0}
    zero_indegree = [v for v, d in G.in_degree() if d == 0]

    while zero_indegree:
        this_generation = zero_indegree
        zero_indegree = []
        for node in this_generation:
            for child in G.neighbors(node):
                indegree_map[child] -= len(G[node][child]) if multigraph else 1
                if indegree_map[child] == 0:
                    zero_indegree.append(child)
                    del indegree_map[child]
        yield this_generation

    if indegree_map:
        for node in G.nodes():
            if node in indegree_map:
                zero_indegree = [node]
                this_generation = []
                while zero_indegree:
                    node = zero_indegree.pop()
                    this_generation.append(node)
                    for child in G.neighbors(node):
                        if child not in this_generation:
                            zero_indegree.append(child)
                yield this_generation


def draw_digraph(digraph, filename=None, service=None, metric=None, figsize=None):
    if service is None:
        service = ""
    if metric is None:
        metric = ""
    if figsize is None:
        figsize = (20, 20)

    G = digraph

    node_names = [n for n in G.nodes()]

    PRECISE_GRAPH = True

    try:
        for layer, nodes in enumerate(nx.topological_generations(G)):
            for node in nodes:
                G.nodes[node]["layer"] = layer
    except Exception:
        PRECISE_GRAPH = False
        for layer, nodes in enumerate(customize_topological_generations(G)):
            for node in nodes:
                G.nodes[node]["layer"] = layer

    # Compute the multipartite_layout using the "layer" node attribute
    pos = nx.multipartite_layout(G, subset_key="layer")

    node_color = []
    for n in node_names:
        if service in str(n) and metric in str(n):
            node_color.append("red")
        else:
            node_color.append("blue")

    fig, ax = plt.subplots(figsize=figsize)
    nx.draw_networkx(G, pos=pos, ax=ax, node_color=node_color)
    if PRECISE_GRAPH is True:
        ax.set_title("DAG layout")
    else:
        ax.set_title("The graph has CIRCLE.")

    fig.tight_layout()

    if filename is None:
        plt.show()
    else:
        plt.savefig(filename)
    # close plt figure
    plt.close()


def draw_graph(cg, labels=None):
    """
    cg can be CausalGraph
    or
    cg.G
    """
    if hasattr(cg, "G") and labels is not None:
        pyd = GraphUtils.to_pydot(cg.G, labels=labels)
    elif hasattr(cg, "G"):
        pyd = GraphUtils.to_pydot(cg.G)
    elif labels is not None:
        pyd = GraphUtils.to_pydot(cg, labels=labels)
    else:
        pyd = GraphUtils.to_pydot(cg)
    tmp_png = pyd.create_png(f="png")
    fp = io.BytesIO(tmp_png)
    img = mpimg.imread(fp, format="png")
    # plt.rcParams["figure.figsize"] = [20, 12]
    plt.rcParams["figure.autolayout"] = True
    plt.axis("off")
    plt.imshow(img)
    plt.show()


def draw_adj(adj, node_names=None, filename=None, service=None, metric=None, figsize=None):
    if node_names is None:
        node_names = [str(i) for i in range(len(adj))]

    if service is None:
        service = ""
    if metric is None:
        metric = ""
    if figsize is None:
        figsize = (20, 20)

    G = nx.DiGraph()
    G.add_nodes_from(node_names)
    for i in range(len(adj)):
        for j in range(len(adj)):
            if adj[i, j] == 1:
                G.add_edge(node_names[j], node_names[i])

    PRECISE_GRAPH = True

    try:
        for layer, nodes in enumerate(nx.topological_generations(G)):
            for node in nodes:
                G.nodes[node]["layer"] = layer
    except Exception:
        PRECISE_GRAPH = False
        for layer, nodes in enumerate(customize_topological_generations(G)):
            for node in nodes:
                G.nodes[node]["layer"] = layer

    # Compute the multipartite_layout using the "layer" node attribute
    pos = nx.multipartite_layout(G, subset_key="layer")

    node_color = []
    for n in node_names:
        if service in n and metric in n:
            node_color.append("red")
        else:
            node_color.append("blue")

    fig, ax = plt.subplots(figsize=figsize)

    if PRECISE_GRAPH is False:
        nx.draw_networkx(G, ax=ax, node_color=node_color)
    else:
        nx.draw_networkx(G, pos=pos, ax=ax, node_color=node_color)
    if PRECISE_GRAPH is True:
        ax.set_title("DAG layout")
    else:
        ax.set_title("The graph has CIRCLE.")

    fig.tight_layout()

    if filename is None:
        plt.show()
    else:
        plt.savefig(filename)
    # close plt figure
    plt.close()


def draw_mem_graph(mem_graph: MemoryGraph, filename=None, service=None, metric=None, figsize=None):
    node_names = []
    for n in mem_graph.nodes:
        try:
            node_names.append(f"{n.entity}_{n.metric}")
        except Exception:
            node_names.append(str(n))

    if service is None:
        service = ""
    if metric is None:
        metric = ""
    if figsize is None:
        figsize = (20, 20)

    G = nx.DiGraph()
    G.add_nodes_from(node_names)

    for n in mem_graph.nodes:
        childrens = mem_graph.children(n)
        for c in childrens:
            try:
                G.add_edge(f"{n.entity}_{n.metric}", f"{c.entity}_{c.metric}")
            except Exception:
                G.add_edge(str(n), str(c))

    PRECISE_GRAPH = True

    try:
        for layer, nodes in enumerate(nx.topological_generations(G)):
            for node in nodes:
                G.nodes[node]["layer"] = layer
    except Exception:
        PRECISE_GRAPH = False
        for layer, nodes in enumerate(customize_topological_generations(G)):
            for node in nodes:
                G.nodes[node]["layer"] = layer

    # Compute the multipartite_layout using the "layer" node attribute
    pos = nx.multipartite_layout(G, subset_key="layer")

    node_color = []
    for n in node_names:
        if service in n and metric in n:
            node_color.append("red")
        else:
            node_color.append("blue")

    fig, ax = plt.subplots(figsize=figsize)
    nx.draw_networkx(G, pos=pos, ax=ax, node_color=node_color)
    if PRECISE_GRAPH is True:
        ax.set_title("DAG layout")
    else:
        ax.set_title("The graph has CIRCLE.")

    fig.tight_layout()

    if filename is None:
        plt.show()
    else:
        plt.savefig(filename)
    # close plt figure
    plt.close()


def visualize_metrics(data: pd.DataFrame, filename=None, figsize=None):
    """
    Visualize the metrics of the sockshop dataset.
    """
    if figsize is None:
        figsize = (25, 25)

    data = drop_time(data)
    services = []
    metrics = []
    for c in data.columns:
        try:
            service, metric_name = c.split("_", 1)
        except Exception as e:
            print(f"Can not parse {c}")
            continue  # ignore
            # raise e
        if service not in services:
            services.append(service)
        if metric_name not in metrics:
            metrics.append(metric_name)

    n_services = len(services)
    n_metrics = len(metrics)

    fig, axs = plt.subplots(n_services, n_metrics, figsize=figsize)
    fig.tight_layout(pad=3.0)
    for i, service in enumerate(services):
        for j, metric in enumerate(metrics):
            # print(f"{service}_{metric}")
            try:
                axs[i, j].plot(data[f"{service}_{metric}"])
            except Exception:
                pass
            axs[i, j].set_title(f"{service}_{metric}")

    if filename is None:
        plt.show()
    else:
        plt.savefig(filename)

    # close the figure
    plt.close(fig)


if __name__ == "__main__":
    a = pd.read_csv("data/sock-shop/carts-cpu/1/normal.csv")
    b = pd.read_csv("data/sock-shop/carts-cpu/1/anomalous.csv")
    c = pd.concat([a, b], ignore_index=True)
    visualize_metrics(c)
    # print(c.head())
