import networkx as nx
import matplotlib.pyplot as plt
import random
from networkx.algorithms.chains import chain_decomposition
import re

random.seed(42)  # seed for reproducibility

def rainbow_colors(n):
    # color generator for n colors in HSV space, then convert to RGB
    hsv_colors = [(i / n, 1, 1) for i in range(n)]
    rgb_colors = [plt.cm.hsv(h) for h, _, _ in hsv_colors]
    return rgb_colors

def is_2_vertex_connected(G):
    # validate if G is 2-vertex-connected
    if not nx.is_connected(G):
        return False
    cut_vertices = list(nx.articulation_points(G))
    return len(cut_vertices) == 0

# Step 1: generate a random 2-connected graph G with n vertices, for testing purpose
def create_random_biconnected_graph(n):
    G = nx.Graph()
    G.add_nodes_from(range(n))
    while not is_2_vertex_connected(G):
        i,j = random.sample(range(n), 2)
        G.add_edge(i, j)
    return G

# Step 2: construct a minimal 2-connected spanning subgraph G_minimal of G by iteratively removing edges while maintaining 2-vertex-connectivity
def construct_minimal_biconnected_subgraph(G, verbose):
    G_minimal = G.copy()
    for e in list(G_minimal.edges()):
        G_minimal.remove_edge(*e)
        if not is_2_vertex_connected(G_minimal):
            G_minimal.add_edge(*e)
    return G_minimal

# Step 3: Ear decomposition
def ear_decomposition(G_minimal, verbose):
    chains = list(chain_decomposition(G_minimal))
    chains_nodes = [[u for u, _ in ch] + [ch[-1][1]] for ch in chains]
    nodes = {chains_nodes[0][0], chains_nodes[0][1]}
    for chain in chains_nodes:
        if len(nodes & set(chain)) != 2:
            raise ValueError("Ear decomposition failed: each ear must connect to the first ear at two nodes.")
        nodes |= set(chain)
    return chains, chains_nodes

# Step 4: Identify x and y^i defined in the paper
def identify_x_yi(G_minimal, chains_nodes, verbose):
    degree_dict = dict(G_minimal.degree())
    P0 = chains_nodes[0]
    x = next(v for v in P0 if degree_dict[v] == 2)

    def identify_y0(P0, x, degree_dict):
        for v in P0:
            if v != x and degree_dict[v] == 2:
                return v
        return None

    y0 = identify_y0(P0, x, degree_dict)
    ears = chains_nodes[1:]
    ys = [next(v for v in e[1:-1] if degree_dict[v] == 2) for e in ears]
    ys_full = [y0] + ys
    return x, ys_full

# Step 5: Construct the Eulerian graph GG by doubling edges according to the ears and removing certain edges as specified in the paper
def construct_eulerian_graph(G_minimal, chains_nodes, ys_full, verbose):
    GG = nx.MultiGraph()
    GG.add_nodes_from(G_minimal.nodes)
    removed_edges = []

    reversed_ears = chains_nodes[::-1]
    reversed_ys = ys_full[::-1]

    for i, ear in enumerate(reversed_ears):
        for u, v in zip(ear[:-1], ear[1:]):
            GG.add_edge(u, v, color=G_minimal.edges[u, v].get('color', (0, 0, 0, 1)))
        y_i = reversed_ys[i]
        j_y = ear.index(y_i)
        if verbose:
            print(f"Processing ear {i}: {ear}, y_i = {y_i}; index_y = {j_y}")  # デバッグ用
        for j in range(1, len(ear) - 1):
            u, v = ear[j], ear[j + 1]
            if GG.degree(u) % 2 == 1:
                if verbose:
                    print(f"Doubled edge: ({u}, {v}), GG.adj[{u}]={list(GG.adj[u].keys())}, GG.degree({u})={GG.degree(u)}")  # デバッグ用
                GG.add_edge(u, v,  color=GG.edges[u, v, 0].get('color', (0, 0, 0, 1)))
        u_last, v_last = ear[-2], ear[-1]
        if GG.number_of_edges(u_last, v_last) == 2:
            GG.remove_edges_from([(u_last, v_last)] * 2)
            removed_edges.append((u_last, v_last))
            if verbose:
                print(f"Removed edge: ({u_last}, {v_last})")
        else:
            if GG.number_of_edges(ear[j_y], ear[j_y+1]) == 2: # y_iとy_iの次のノードの間のエッジが2本ならば削除
                GG.remove_edges_from([(ear[j_y], ear[j_y+1])] * 2)
                removed_edges.append((ear[j_y], ear[j_y+1]))
                if verbose:
                    print(f"Removed edge: ({ear[j_y]}, {ear[j_y+1]})")

    # check if GG is Eulerian
    if not nx.is_eulerian(GG):
        raise ValueError("Constructed graph GG is not Eulerian.")

    # check if each vertex has even degree
    for node in GG.nodes:
        if GG.degree(node) % 2 != 0:
            raise ValueError(f"Node {node} has odd degree {GG.degree(node)} in GG.")

    return GG, removed_edges

# Step 6: Orient the edges of GG to get a directed graph DG such that each vertex has in-degree 2, by orienting the edges according to the ears and the y^i's as specified in the paper
def orient_edges_by_ears(GG, chains_nodes, ys_full, removed_edges, verbose):
    DG = nx.DiGraph()
    DG.add_nodes_from(GG.nodes)

    reversed_chains = chains_nodes[::-1]
    reversed_ys = ys_full[::-1]

    for i, ear in enumerate(reversed_chains):
        y_i = reversed_ys[i]
        u_last, v_last = ear[-2], ear[-1]

        def add_directed_edges(u, v):
            if GG.has_edge(u, v):
                for _ in range(GG.number_of_edges(u, v)):
                    DG.add_edge(u, v)

        # Case (a)
        if ((u_last, v_last) in removed_edges or (v_last, u_last) in removed_edges):
            for j in range(len(ear) - 1):
                u, v = ear[j], ear[j + 1]
                add_directed_edges(u, v)
                if GG.number_of_edges(u,v) == 2:
                    add_directed_edges(v, u)
        else:
            # Case (b) or (c)
            y_index = ear.index(y_i)
            for j in range(0, y_index):
                u, v = ear[j], ear[j + 1]
                add_directed_edges(u, v)
                if GG.number_of_edges(u,v) == 2:
                    add_directed_edges(v, u)
            for j in range(len(ear) - 1, y_index, -1):
                u, v = ear[j], ear[j - 1]
                add_directed_edges(u, v)
                if GG.number_of_edges(u,v) == 2:
                    add_directed_edges(v, u)

    # check max in_degree is not more than 2
    for node in DG.nodes:
        if DG.in_degree(node) > 2:
            raise ValueError(f"Node {node} has in-degree {DG.in_degree(node)}, which is greater than 2.")
    return DG

# Step 7: Contract the directed graph DG by contracting each node w with in-degree 2 into an edge between its two in-neighbors, to get a contracted graph GG_c. The contracted graph GG_c should be Eulerian and have a one-to-one correspondence between its edges and the edges of DG that are not incident to x.
def contract_digraph(DG, x, verbose):
    GG_c = nx.MultiGraph()
    GG_c.add_nodes_from(DG.nodes)
    GG_c.add_edges_from(DG.edges)
    contraction_map = {}

    for w in DG.nodes:
        if w == x:
            continue
        in_edges = list(DG.in_edges(w))
        if len(in_edges) == 2:
            (u, w), (v, w2) = in_edges
            u, v = min(u, v), max(u, v)
            if w2 != w: continue

            if (u,v) not in contraction_map:
                GG_c.add_edge(u, v)
                contraction_map[(u, v)] = [w]
                GG_c.remove_edges_from([(u, w), (v, w)])
            else:
                GG_c.add_edge(u, v)
                contraction_map[(u, v)].append(w)
                GG_c.remove_edges_from([(u, w), (v, w)])
            if verbose:
                print(f"Contracted edge: ({u} -> {w} <- {v}) to ({u}, {v})")
    return GG_c, contraction_map

# Step 8: Find an Eulerian cycle on the contracted graph GG_c, and then lift it back to a cycle J on the original graph DG by replacing each contracted edge with the corresponding path through the contracted node w. The lifted cycle J should be a closed walk that visits each vertex at least once, and we will later show that it can be shortcut to a Hamiltonian cycle in G^2.
def find_eulerian_cycle_on_contracted_graph(GG, GG_c, DG, contraction_map, x, verbose):
    DG_for_check = DG.copy()
    GG_c.remove_nodes_from([n for n, d in GG_c.degree() if d == 0])
    if not nx.is_eulerian(GG_c):
        raise ValueError("Graph GG_c is not Eulerian.")
    J_c = list(nx.eulerian_circuit(GG_c, source=x))
    J = []
    J_str = f"{J_c[0][0]}"
    v_pre = None
    if verbose:
        print(J_c[0][0], end="")
    for u, v in J_c:
        if (u, v) in contraction_map or (v, u) in contraction_map:
            u, v = (v, u) if (u, v) not in contraction_map else (u, v)
            w = contraction_map[u,v][0]
            contraction_map[u,v] = contraction_map[u,v][1:]
            if verbose:
                print(f"processing {u} -> {w} <- {v}")
            if v_pre == v: (u, v) = (v, u) 
            J.append((u, w))
            J.append((w, v))
            direction_uw = "->"
            direction_wv = "<-"
            if (u,w) in DG_for_check.edges():
                DG_for_check.remove_edge(u,w)
                J.append((u,w))
            else:
                direction_uw = "<-"
                DG_for_check.remove_edge(w,u)
                J.append((w,u))
            if (v,w) in DG_for_check.edges():
                DG_for_check.remove_edge(v,w)
                J.append((v,w))
            else:
                direction_wv = "->"
                DG_for_check.remove_edge(w,v)
                J.append((w,v))
            if verbose:
                print(f" {direction_uw} {w} {direction_wv} {v}", end="")
            J_str += f" {direction_uw} {w} {direction_wv} {v}"
            v_pre = v
        else:
            if v_pre == v: (u, v) = (v, u)
            direction_uv = "->"
            if (u,v) in DG_for_check.edges():
                DG_for_check.remove_edge(u,v)
                if verbose:
                    print(f"Delete {u}->{v}")
                J.append((u, v))
            else:
                direction_uv = "<-"
                if verbose:
                    print(f"Delete {v}->{u}")
                DG_for_check.remove_edge(v,u)
                J.append((v, u))
            if verbose:
                print(f" {direction_uv} {v}", end="")
            J_str += f" {direction_uv} {v}"
            v_pre = v
    if verbose:
        print()
    GG_colored = nx.MultiDiGraph()
    GG_colored.add_nodes_from(GG.nodes)
    n = len(J)
    rainbow = rainbow_colors(n)
    for i, (u, v) in enumerate(J):
        GG_colored.add_edge(u, v, color=rainbow[i])
    return GG_colored, J_str

# Step 9: Lift the cycle J to a cycle H on the square graph G^2, by replacing each edge (u, v) in J with the edge (u, v) in G^2 if it exists, or with the path (u, w, v) where w is the contracted node corresponding to the edge (u, v) if (u, v) was contracted. The resulting cycle H should be a Hamiltonian cycle in G^2.
def lift_edges(J, x, verbose):
    pattern = r"<-\s*(\d+)\s*->"
    replaced = re.sub(pattern, r"<->", J)

    numbers = re.findall(r"\d+", replaced)
    H_list = list(map(int, numbers))
    n = len(H_list)-1
    rainbow = rainbow_colors(n)
    H = nx.DiGraph()
    for i, (u,v) in enumerate(zip(H_list, H_list[1:])):
        H.add_edge(u,v, color=rainbow[i])
    return H, H_list

# Step final: visualization
def plot_graphs_side_by_side_colored_and_multiedges(Gs, titles, pos, x=None, ys=None, chains=None, color_from_step=2):
    fig, axes = plt.subplots(1, len(Gs), figsize=(6 * len(Gs), 6))
    colors = plt.cm.get_cmap('tab10', len(chains) if chains else 1)

    for i, (G, title) in enumerate(zip(Gs, titles)):
        ax = axes[i]
        node_colors = []
        for node in G.nodes():
            if i >= color_from_step:
                if node == x:
                    node_colors.append('red')
                elif ys and node in ys:
                    node_colors.append('cyan')
                else:
                    node_colors.append('lightgray')
            else:
                node_colors.append('lightgray')

        nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, edgecolors='k', node_size=500)
        nx.draw_networkx_labels(G, pos, ax=ax)

        if title in {"Step 1: Original Graph", "Step 2: Minimal 2-Connected"}:
            nx.draw_networkx_edges(G, pos, edge_color='gray', width=1.5, ax=ax)

        elif title == "Step 3–4: Ear Decomposition with x, y^i" and chains:
            for j, chain in enumerate(chains):
                edge_list = list(nx.utils.pairwise([u for u, _ in chain] + [chain[-1][1]]))
                nx.draw_networkx_edges(G, pos, edgelist=edge_list, edge_color=[colors(j)], width=2.5, ax=ax)

        elif title in {"Step 5: Eulerian Graph $G_G$", "Step 7: Contracted Oriented Graph $GG_c$", "Step 8: Eulerian Cycle on $GG$"}:
            offset = 0.3
            drawn = set()
            for u, v, k in G.edges(keys=True):
                if (u, v, k) not in drawn:
                    edge_color = G.edges[u, v, k].get('color', (0,0,0,1))
                    rad = offset * (k - 0.5)
                    nx.draw_networkx_edges(
                        G, pos, edgelist=[(u, v)],
                        connectionstyle=f'arc3,rad={rad}',
                        edge_color=edge_color, ax=ax, width=2
                    )
                    drawn.add((u, v, k))

        elif title in {"Step 6: Oriented Graph $\\vec{G}_G$", "Step 9: Hamiltonian Cycle on $G^2$"}:
            drawn_edges = set()
            for u, v in G.edges():
                edge_color = G.edges[u, v].get('color', (0,0,0,1))
                if (v, u) in G.edges() and (v, u) not in drawn_edges:
                    nx.draw_networkx_edges(
                        G, pos, edgelist=[(u, v)],
                        connectionstyle="arc3,rad=0.2",
                        edge_color=edge_color, arrows=True,
                        arrowstyle='-|>', width=2, ax=ax
                    )
                    nx.draw_networkx_edges(
                        G, pos, edgelist=[(v, u)],
                        connectionstyle="arc3,rad=0.2",
                        edge_color=edge_color, arrows=True,
                        arrowstyle='-|>', width=2, ax=ax
                    )
                    drawn_edges.add((u, v))
                    drawn_edges.add((v, u))
                elif (u, v) not in drawn_edges:
                    nx.draw_networkx_edges(
                        G, pos, edgelist=[(u, v)],
                        connectionstyle="arc3,rad=0",
                        edge_color=edge_color, arrows=True,
                        arrowstyle='-|>', width=2, ax=ax
                    )
                    drawn_edges.add((u, v))
        else:
            pass

        ax.set_title(title)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig("debug.png", bbox_inches='tight', dpi=300)

def find_hamiltonian_cycle(G=None, pos=None, vis=False, verbose=False):
    if G == None:
        G = nx.Graph()
        G.add_edges_from([(0, 1), (0, 2), (0, 3), (0, 4), 
                          (0, 5), (0, 6), (0, 8), (0, 9), 
                          (0, 10), (0, 11), (0, 12), (0, 13), 
                          (0, 14), (1, 2), (1, 3), (1, 4), 
                          (1, 5), (1, 6), (1, 9), (1, 10), 
                          (1, 11), (1, 12), (1, 14), (2, 3), 
                          (2, 5), (2, 6), (2, 8), (2, 9), 
                          (2, 11), (2, 12), (2, 13), (2, 14), 
                          (3, 5), (3, 6), (3, 8), (3, 9), (3, 10), 
                          (3, 11), (3, 12), (3, 13), (3, 14), 
                          (4, 5), (4, 9), (4, 10), (4, 14), (5, 6), 
                          (5, 9), (5, 10), (5, 11), (5, 14), (6, 8), 
                          (6, 9), (6, 10), (6, 11), (6, 12), (6, 13), 
                          (6, 14), (7, 12), (7, 14), (8, 11), (8, 12), 
                          (8, 13), (9, 10), (9, 11), (9, 14), (10, 11), 
                          (11, 12), (11, 13), (11, 14), (12, 13), (12, 14)])
    else:
        assert is_2_vertex_connected(G), "Input graph G must be 2-vertex-connected."
    if verbose:
        print(f"G has {G.number_of_edges()} edges")
    if pos == None:
        pos = nx.spring_layout(G, seed=42)

    # Step 2
    G_minimal = construct_minimal_biconnected_subgraph(G, verbose)
    if verbose:
        print(f"Step 2: G_minimal has {G_minimal.number_of_edges()} edges")

    # Step 3
    chains, chains_nodes = ear_decomposition(G_minimal, verbose)

    if verbose:
        print("chains =", "\n".join([str(chain) for chain in chains]))
    G_minimal_colored = G_minimal.copy()
    colors = plt.cm.get_cmap('tab10', len(chains) if chains else 1)
    for i, chain in enumerate(chains):
        for u, v in chain:
            G_minimal_colored.edges[u, v]['color'] = colors(i)

    # Step 4: x, y^i
    x, ys_full = identify_x_yi(G_minimal, chains_nodes, verbose)
    if verbose:
        print(f"Step 4: x={x}, ys_full={ys_full}")

    # Step 5: Construct GG
    GG, removed_edges = construct_eulerian_graph(G_minimal_colored, chains_nodes, ys_full, verbose)
    if verbose:
        print(f"Step 5: GG has {GG.number_of_edges()} edges")
    # Step 6: Orient edges to get DG
    DG = orient_edges_by_ears(GG, chains_nodes, ys_full, removed_edges, verbose)
    if verbose:
        print(f"Step 6: DG has {DG.number_of_edges()} edges")

    # # Step 7: Contract DG to get GG_c
    GG_c, contraction_map = contract_digraph(DG, x, verbose)
    if verbose:
        print(f"Step 7: GG_c has {GG_c.edges()} edges")

    # Step 8: Find Eulerian cycle on GG_c and lift to DG
    GG_colored, J = find_eulerian_cycle_on_contracted_graph(GG, GG_c, DG, contraction_map, x, verbose)
    if verbose:
        print(f"Eulerian cycle on GG: {GG_colored.edges()}")

    # Step 9: Lift cycle J to cycle
    H, H_list = lift_edges(J, x, verbose)
    if verbose:
        print("Hamiltonian cycle order in G^2:", H_list)

    # validation
    if not vis: 
        if len(set(H_list)) < len(G.nodes):
            raise ValueError("Hamiltonian cycle does not visit all nodes exactly once.")
        elif len(H_list) != len(G.nodes) + 1:
            plot_graphs_side_by_side_colored_and_multiedges([G_minimal, DG, GG_c, GG_colored, H], ["Step 2: Minimal 2-Connected","Step 6: Oriented Graph $\\vec{G}_G$","Step 7: Contracted Oriented Graph $GG_c$", "Step 8: Eulerian Cycle on $GG$","Step 9: Hamiltonian Cycle on $G^2$"],pos,x=x,ys=ys_full,chains=chains,color_from_step=2)
            raise ValueError("Hamiltonian cycle length is incorrect.")
        return H_list
    plot_graphs_side_by_side_colored_and_multiedges(
        [
            G_minimal, 
            DG, 
            GG_c, 
            GG_colored,
            H
        ],
        [
            "Step 2: Minimal 2-Connected",
            "Step 6: Oriented Graph $\\vec{G}_G$",
            "Step 7: Contracted Oriented Graph $GG_c$", 
            "Step 8: Eulerian Cycle on $GG$",
            "Step 9: Hamiltonian Cycle on $G^2$"
        ],
        pos, x=x, ys=ys_full, chains=chains, color_from_step=2
    )

if __name__ == "__main__":
    find_hamiltonian_cycle(vis=True, verbose=False)