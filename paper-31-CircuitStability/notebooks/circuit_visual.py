import networkx as nx
import matplotlib as mpl
import matplotlib.pyplot as plt


def get_top_attn_heads(mat, pct):
    """Gets the attentionh heads that are in the top `pct` of influence."""
    heads = []
    for lay_idx in range(len(mat)):
        for attn_idx in range(len(mat[lay_idx])):
            heads.append((lay_idx, attn_idx, mat[lay_idx, attn_idx]))
    heads.sort(key=lambda x: x[2], reverse=True)
    top_heads = heads[: int(len(heads) * pct)]
    return top_heads


def attn_heads_multipartite(heads):
    """Sorts the attention heads into multipartite layers and inserts MLP
    layers between them. Assumes that the heads are a list of tuples
    (layer, head, influence)."""
    min_lay_idx = min(heads, key=lambda x: x[0])[0]
    max_lay_idx = max(heads, key=lambda x: x[0])[0]

    edges = []
    all_nodes = []
    vals = {}

    counter = 1
    for lay_idx in range(min_lay_idx, max_lay_idx + 1):
        # get all of the attn heads in that layer
        lay_nodes = list(filter(lambda x: x[0] == lay_idx, heads))
        # sort the nodes by their index
        lay_nodes.sort(key=lambda x: x[1])
        for node in lay_nodes:
            head_name = f"{node[0]}.{node[1]}"
            mlp_name = f"MLP {node[0]}"
            all_nodes.append((head_name, counter))
            edges.append((head_name, mlp_name))
            vals[head_name] = node[2]
            if lay_idx != min_lay_idx:
                edges.append((f"MLP {node[0] - 1}", head_name))
        if len(lay_nodes) != 0:
            counter += 1
        else:
            edges.append((f"MLP {lay_idx-1}", f"MLP {lay_idx}"))
        all_nodes.append((f"MLP {lay_idx}", counter))
        counter += 1
    return all_nodes, edges, vals


def draw_rounded_node(ax, pos, node, node_color, width=0.02, height=0.05):
    x, y = pos[node]
    # Create a FancyBboxPatch with round corners
    box = mpl.patches.Rectangle(
        (x - width / 2, y - height / 2),
        width,
        height,
        linewidth=1,
        facecolor=node_color,
        edgecolor="black",
    )
    ax.add_patch(box)

    if (
        "MLP" in node
        or (node_color[0] * 0.299 + node_color[1] * 0.587 + node_color[2] * 0.114)
        > 0.73
    ):
        text_color = "black"
    else:
        text_color = "white"

    ax.text(
        x,
        y,
        node,
        horizontalalignment="center",
        verticalalignment="center",
        fontsize=7,
        weight="bold",
        color=text_color,
    )


def make_circuit_graph(nodes, edges, vals, color="viridis", scale=1):
    # Create a directed graph
    G = nx.DiGraph()
    for node, subset in nodes:
        G.add_node(node, subset=subset)
    G.add_edges_from(edges)

    # Normalize scalar values to a range between 0 and 1
    norm = plt.Normalize(vmin=min(vals.values()), vmax=max(vals.values()))

    # Use a colormap (e.g., 'viridis') to map scalar values to colors
    cmap = plt.get_cmap(color)
    vals = vals.copy()
    for k, v in vals.items():
        vals[k] = cmap(norm(v))

    pos = nx.multipartite_layout(
        G, subset_key="subset", align="horizontal", scale=scale
    )
    return G, pos, vals


# plt.figure(figsize=(3, 10))
# ax = plt.gca()
# nx.draw_networkx_edges(G, pos, arrows=True, width=1.5, min_source_margin=5, min_target_margin=5)

# for node, node_color in zip(G.nodes(), node_colors):
#     draw_rounded_node(ax, pos, node, node_color)

# # ax.set_aspect("equal")
# plt.autoscale()
