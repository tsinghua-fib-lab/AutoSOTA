from typing import Optional
import numpy as np
import networkx as nx

from pyciphod.utils.graphs.graphs import Graph, DirectedAcyclicGraph, AcyclicDirectedMixedGraph
from pyciphod.utils.time_series.data_format import DTimeVar
from pyciphod.utils.graphs.partially_specified_graphs import SummaryCausalGraph


class FtGraph(Graph):
    def __init__(self):
        super().__init__()
        self.t_min = None
        self.t_max = None

    def add_vertex(self, var):
        if not isinstance(var, DTimeVar):
            raise TypeError("All vertices must be instances of DTimeVar")
        super().add_vertex(var)
        t = var.time
        if self.t_min is None or t < self.t_min:
            self.t_min = t
        if self.t_max is None or t > self.t_max:
            self.t_max = t

    def _check_temporal_order(self, src, tgt):
        ts = src.time
        tt = tgt.time
        if ts > tt:
            raise ValueError(f"Causal edge from {src} to {tgt} violates temporal order: {ts} > {tt}")

    def add_directed_edge(self, src, tgt):
        self._check_temporal_order(src, tgt)
        return super().add_directed_edge(src, tgt)

    def get_summary_causal_graph(self) -> SummaryCausalGraph:
        """Create a SummaryCausalGraph by collapsing all time-variables with the same name into a single node.
        Edges in the summary graph are determined by the presence of any edge between any time-variables of the corresponding names in the original graph.
        """
        summary = SummaryCausalGraph()
        # Add summary vertices (one per unique variable name)
        var_names = {v.name for v in self.get_vertices()}
        for name in var_names:
            summary.add_vertex(name)

        # Add edges to summary graph based on original graph
        lag_max = 0
        for (u, v) in self.get_directed_edges():
            summary.add_directed_edge(u.name, v.name)
            lag_max = max(lag_max, abs(u.time-v.time))

        for (u, v) in self.get_confounded_edges():
            summary.add_confounded_edge(u.name, v.name)
            lag_max = max(lag_max, abs(u.time-v.time))

        for (u, v) in self.get_undirected_edges():
            summary.add_undirected_edge(u.name, v.name)
            lag_max = max(lag_max, abs(u.time-v.time))

        for (u, v) in self.get_uncertain_edges():
            summary.add_uncertain_edge(u.name, v.name)
            lag_max = max(lag_max, abs(u.time-v.time))
        summary.add_lag_max(lag_max)

        return summary

    def draw_graph(self, treatment: set = None, outcome: set = None):
        """
        Draw the temporal graph arranging nodes so that all time-variables of the same series
        (e.g. X_1, X_2, X_3) are aligned horizontally, and variables at the same time
        (e.g. X_1 and Y_1) are aligned vertically.

        Layout convention used here:
        - x axis = time (times ordered left to right)
        - y axis = variable name (each variable has a horizontal row)
        """
        import matplotlib.pyplot as plt
        import numpy as _np
        from matplotlib.patches import FancyArrowPatch

        # Colors and styles (same as base)
        vertex_color = "lightblue"
        font_color = "black"
        directed_edge_color = "gray"
        confounded_edge_color = "black"
        undirected_edge_color = "black"
        uncertain_edge_color = "#F7B617"
        treatment_color = "#c82804"
        outcome_color = "#4851a1"

        all_nodes = list(self._g.nodes)
        outcome = set(outcome or [])
        treatment = set(treatment or [])

        if not all_nodes:
            return

        # Map times to x positions (left to right) and variable names to y positions
        times = sorted({n.time for n in all_nodes})
        time_to_x = {t: float(i) for i, t in enumerate(times)}

        var_names = sorted({n.name for n in all_nodes})
        # place first variable at top; higher index => lower on plot, so invert to have first at top
        name_to_y = {name: float(len(var_names) - 1 - i) for i, name in enumerate(var_names)}

        # Build positions: x=time, y=variable name
        pos = {n: _np.array([time_to_x[n.time], name_to_y[n.name]]) for n in all_nodes}

        # Labels: name_time (same as before)
        labels = {n: f"{n.name}_{n.time}" for n in all_nodes}

        fig, ax = plt.subplots()

        # Draw nodes: respect treatment/outcome coloring
        nx.draw_networkx_nodes(self._g, pos, ax=ax, nodelist=list(treatment), node_color=treatment_color)
        nx.draw_networkx_nodes(self._g, pos, ax=ax, nodelist=list(outcome), node_color=outcome_color)

        set_vertices = set(self._g.nodes) - treatment - outcome
        nx.draw_networkx_nodes(self._g, pos, ax=ax, nodelist=list(set_vertices), node_color=vertex_color)
        nx.draw_networkx_labels(self._g, pos, labels=labels, ax=ax, font_color=font_color)

        # Edges
        acyclic_edges = [edge for edge in self.get_directed_edges() if (edge[1], edge[0]) not in self.get_directed_edges()]
        cyclic_edges = [edge for edge in self.get_directed_edges() if (edge[1], edge[0]) in self.get_directed_edges()]
        confounded_edges = self.get_confounded_edges()
        undirected_edges = self.get_undirected_edges()

        nx.draw_networkx_edges(self._g, pos, ax=ax, edgelist=acyclic_edges, edge_color=directed_edge_color, arrowstyle='->')
        nx.draw_networkx_edges(self._g, pos, ax=ax, edgelist=cyclic_edges, edge_color=directed_edge_color, arrowstyle='->')

        # Draw confounded edges as curved bidirectional arrows
        try:
            confounded_list = list(confounded_edges)
        except Exception:
            confounded_list = []
        for idx, (u, v) in enumerate(confounded_list):
            if u not in pos or v not in pos:
                continue
            rad = 0.18 if (idx % 2 == 0) else -0.18
            # Ensure xy tuples are plain floats (avoid type checker warnings)
            xyA = (float(pos[u][0]), float(pos[u][1]))
            xyB = (float(pos[v][0]), float(pos[v][1]))
            shrink_pts = 8
            arrow = FancyArrowPatch(xyA, xyB,
                                    connectionstyle=f"arc3,rad={rad}",
                                    arrowstyle='<->',
                                    mutation_scale=18,
                                    color=confounded_edge_color,
                                    linewidth=1.5,
                                    shrinkA=shrink_pts, shrinkB=shrink_pts,
                                    linestyle='dashed')
            arrow.set_zorder(3)
            arrow.set_clip_on(False)
            ax.add_patch(arrow)

        nx.draw_networkx_edges(self._g, pos, ax=ax, edgelist=undirected_edges, arrowstyle='-', edge_color=undirected_edge_color)

        dashed_arrow = [(u, v) for (u, v, t) in self.get_edges() if t == '-->']
        nx.draw_networkx_edges(self._g, pos, ax=ax, edgelist=dashed_arrow, edge_color=uncertain_edge_color,
                            style='dashed', arrowstyle='->')

        arrow_double_bar = [(u, v) for (u, v, t) in self.get_edges() if t == '-||']
        nx.draw_networkx_edges(self._g, pos, ax=ax, edgelist=arrow_double_bar, edge_color="red",
                            style='solid', arrowstyle='-[')

        # Axis limits and ticks: show times on x and variable names on y
        ax.set_xlim(min(time_to_x.values()) - 0.5, max(time_to_x.values()) + 0.5)
        ax.set_ylim(min(name_to_y.values()) - 0.5, max(name_to_y.values()) + 0.5)

        ax.set_xticks([time_to_x[t] for t in times])
        ax.set_xticklabels([str(t) for t in times])

        # y ticks at variable rows, show variable names
        y_ticks = [name_to_y[n] for n in var_names]
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(var_names)

        # Hide x grid lines if cluttered, remove axis frame
        ax.set_xlabel('time')
        ax.set_ylabel('variable')
        plt.axis('off')
        plt.show()


class FtAcyclicDirectedMixedGraph(FtGraph, AcyclicDirectedMixedGraph):
    def __init__(self):
        super().__init__()

class FtDirectedAcyclicGraph(FtGraph, DirectedAcyclicGraph):
    def __init__(self):
        super().__init__()


def create_random_ft_admg(
    num_ts: int,
    p_edge: float = 0.2,
    causally_stationary: bool = True,
    max_delay: Optional[int] = None,
    allow_instantaneous: bool = True,
    allow_unmeasured_confounding: bool = True,
    seed: Optional[int] = None,
) -> FtAcyclicDirectedMixedGraph:
    """
    Generate a random FT-ADMG where `num_v` is interpreted as the number of series
    (distinct variable names).

    For each series we create `max_delay + 1` time-variables (times 0..max_delay).

    Directed edges are only allowed from time t_src to time t_tgt with t_src <= t_tgt.
    Confounded edges are only allowed at the same time (lag 0), between different series.

    If causally_stationary=True:
        - directed edge existence is sampled once per (src_name, tgt_name, lag)
          and then repeated for all compatible times;
        - confounded edge existence is sampled once per unordered pair of series
          and repeated across all time slices.

    If causally_stationary=False:
        - every admissible pair of nodes is sampled independently.
    """
    rng = np.random.default_rng(seed)

    if num_ts < 1:
        raise ValueError("num_v must be >= 1")
    if not (0.0 <= p_edge <= 1.0):
        raise ValueError("p_edge must be in [0, 1]")

    if max_delay is None:
        max_delay = max(num_ts - 1, 1)
    if max_delay < 0:
        raise ValueError("max_delay must be >= 0")

    series_names = [f"V{i}" for i in range(num_ts)]
    times = list(range(max_delay + 1))

    order = list(series_names)
    rng.shuffle(order)


    ftadmg = FtAcyclicDirectedMixedGraph()

    # Create nodes
    nodes = [DTimeVar(name, t) for name in series_names for t in times]
    for node in nodes:
        ftadmg.add_vertex(node)

    # Useful lookup
    node_lookup = {(node.name, node.time): node for node in nodes}

    # ------------------------------------------------------------------
    # Stationary edge templates
    # ------------------------------------------------------------------
    directed_template = {}
    confounded_template = {}

    if causally_stationary:
        # Directed templates: one Bernoulli draw per ordered pair of series and lag
        for src_name in series_names:
            for tgt_name in series_names:
                for lag in range(1, max_delay + 1):
                    # if lag == 0 and not allow_instantaneous:
                    #     continue
                    directed_template[(src_name, tgt_name, lag)] = (rng.random() < p_edge)
        if allow_instantaneous:
            for i in range(len(order)):
                for j in range(i + 1, len(order)):
                    src_name = order[i]
                    tgt_name = order[j]
                    directed_template[(src_name, tgt_name, 0)] = (rng.random() < p_edge)

        # Ensure at least one directed template edge exists
        if not any(directed_template.values()):
            admissible_keys = list(directed_template.keys())
            if len(admissible_keys) > 0:
                chosen = admissible_keys[rng.integers(len(admissible_keys))]
                directed_template[chosen] = True

        # Confounding templates: one Bernoulli draw per unordered pair of distinct series
        if allow_unmeasured_confounding and allow_instantaneous:
            for i in range(num_ts):
                for j in range(i + 1, num_ts):
                    a = series_names[i]
                    b = series_names[j]
                    confounded_template[(a, b)] = (rng.random() < p_edge)

    # ------------------------------------------------------------------
    # Add directed edges
    # ------------------------------------------------------------------
    for src_name in series_names:
        for tgt_name in series_names:
            for lag in range(max_delay + 1):
                if lag == 0 and not allow_instantaneous:
                    continue

                if causally_stationary:
                    add_this_family = directed_template.get((src_name, tgt_name, lag), False)
                    if not add_this_family:
                        continue

                    # repeat across all compatible times
                    for t_tgt in times:
                        t_src = t_tgt - lag
                        if t_src < 0:
                            continue

                        src = node_lookup[(src_name, t_src)]
                        tgt = node_lookup[(tgt_name, t_tgt)]

                        try:
                            ftadmg.add_directed_edge(src, tgt)
                        except ValueError:
                            # skip invalid edges if parent class rejects them
                            continue

                else:
                    # independently sample each admissible node pair
                    for t_tgt in times:
                        t_src = t_tgt - lag
                        if t_src < 0:
                            continue

                        if rng.random() >= p_edge:
                            continue

                        src = node_lookup[(src_name, t_src)]
                        tgt = node_lookup[(tgt_name, t_tgt)]

                        try:
                            ftadmg.add_directed_edge(src, tgt)
                        except ValueError:
                            continue

    # ------------------------------------------------------------------
    # Add confounded edges (instantaneous only)
    # ------------------------------------------------------------------
    if allow_unmeasured_confounding and allow_instantaneous:
        for i in range(num_ts):
            for j in range(i + 1, num_ts):
                a = series_names[i]
                b = series_names[j]

                if causally_stationary:
                    add_this_family = confounded_template.get((a, b), False)
                    if not add_this_family:
                        continue

                    for t in times:
                        u = node_lookup[(a, t)]
                        v = node_lookup[(b, t)]
                        ftadmg.add_confounded_edge(u, v)

                else:
                    for t in times:
                        if rng.random() >= p_edge:
                            continue
                        u = node_lookup[(a, t)]
                        v = node_lookup[(b, t)]
                        ftadmg.add_confounded_edge(u, v)

    return ftadmg


def create_random_ft_dag(num_ts: int,
                             p_edge: float = 0.2, causally_stationary: bool = True, max_delay: Optional[int] = None, allow_instantaneous: bool = True,
                             seed: Optional[int] = None) -> FtDirectedAcyclicGraph:
    """
    Generate a random FT-DAG (temporal DAG) where vertices are DTimeVar and directed edges
    never go from a future time to a past time.

    This mirrors `create_random_ft_admg` but only produces directed edges. The
    `allow_instantaneous` parameter controls whether edges with lag==0 are allowed.
    """
    # First generate an ADMG without confounding, then copy directed structure into a DAG object
    ftadmg = create_random_ft_admg(num_ts=num_ts, p_edge=p_edge, causally_stationary=causally_stationary,
                                   max_delay=max_delay, allow_instantaneous=allow_instantaneous,
                                   allow_unmeasured_confounding=False, seed=seed)

    ftdag = FtDirectedAcyclicGraph()
    # copy vertices
    for v in ftadmg.get_vertices():
        ftdag.add_vertex(v)
    # copy directed edges
    for (u, v) in ftadmg.get_directed_edges():
        try:
            ftdag.add_directed_edge(u, v)
        except ValueError:
            # should not happen because ftadmg was acyclic in directed part
            continue

    return ftdag


if __name__ == '__main__':
    ftadmg = create_random_ft_admg(num_ts=3, p_edge=0.5, causally_stationary=True, max_delay=2, seed=1)
    print("Random FT-ADMG:")
    ftadmg.draw_graph()

    ftdag = create_random_ft_dag(num_ts=3, p_edge=0.3, causally_stationary=True, max_delay=2, seed=1)
    print("\nRandom FT-DAG:")
    ftdag.draw_graph()
    print(ftdag.get_vertices())




