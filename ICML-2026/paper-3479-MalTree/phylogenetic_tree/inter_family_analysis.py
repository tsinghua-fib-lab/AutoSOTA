"""
Inter-Family Evolutionary Analysis

Implements Algorithms 5-9 from Appendix E for analyzing evolutionary
relationships between malware families.
"""
import json
import numpy as np
from typing import Dict, List, Tuple, Set, Optional
from collections import defaultdict
from ete3 import Tree
import networkx as nx


def load_tree_and_aggregate_leaves(tree_path: str) -> Tuple[Tree, Dict[str, List[str]]]:
    """
    Load tree and aggregate leaves by family.

    Implements Algorithm 6 from Appendix H.

    Args:
        tree_path: Path to tree file (Newick format)

    Returns:
        Tuple of (tree, mapping from family to leaves)
    """
    tree = Tree(tree_path)

    family_to_leaves = defaultdict(list)

    for leaf in tree.get_leaves():
        # Extract family name from leaf name (assumes format: family_index or family)
        leaf_name = leaf.name
        # Try to extract family part (before underscore if present)
        family = leaf_name.rsplit('_', 1)[0] if '_' in leaf_name else leaf_name
        family_to_leaves[family].append(leaf_name)

    return tree, dict(family_to_leaves)


def get_mrca(tree: Tree, leaves: List[str]) -> Tree:
    """Get the Most Recent Common Ancestor of a set of leaves."""
    if not leaves:
        return tree.get_tree_root()

    leaf_nodes = []
    for leaf_name in leaves:
        nodes = tree.search_nodes(name=leaf_name)
        if nodes:
            leaf_nodes.append(nodes[0])

    if not leaf_nodes:
        return tree.get_tree_root()

    return tree.get_common_ancestor(leaf_nodes)


def calculate_inter_family_distances(tree: Tree,
                                      family_to_leaves: Dict[str, List[str]]) -> Dict:
    """
    Calculate inter-family distances using median path lengths to MRCA.

    Implements Algorithm 7 from Appendix H.

    Args:
        tree: Phylogenetic tree
        family_to_leaves: Mapping from family name to leaf names

    Returns:
        Distance matrix as dict: {(family1, family2): (median_d1, median_d2)}
    """
    distances = {}
    families = list(family_to_leaves.keys())

    for i, family1 in enumerate(families):
        for family2 in families[i+1:]:
            # Get MRCA of both families
            all_leaves = family_to_leaves[family1] + family_to_leaves[family2]
            mrca = get_mrca(tree, all_leaves)

            # Calculate distances from MRCA to each family's leaves
            distances_f1 = []
            distances_f2 = []

            for leaf in family_to_leaves[family1]:
                leaf_node = tree.search_nodes(name=leaf)
                if leaf_node:
                    distances_f1.append(leaf_node[0].get_distance(mrca))

            for leaf in family_to_leaves[family2]:
                leaf_node = tree.search_nodes(name=leaf)
                if leaf_node:
                    distances_f2.append(leaf_node[0].get_distance(mrca))

            # Calculate medians
            median_d1 = np.median(distances_f1) if distances_f1 else float('inf')
            median_d2 = np.median(distances_f2) if distances_f2 else float('inf')

            distances[(family1, family2)] = (median_d1, median_d2)

    return distances


def build_directed_graph(distances: Dict) -> nx.DiGraph:
    """
    Build directed graph from inter-family distances.

    Implements Algorithm 8 from Appendix H.
    Edge direction: from earlier-diverging (shorter distance) to later-diverging family.
    Edge weight: source family's median distance to MRCA.

    Args:
        distances: Distance dict from calculate_inter_family_distances

    Returns:
        Directed graph with weighted edges
    """
    G = nx.DiGraph()

    for (family1, family2), (d1, d2) in distances.items():
        if d1 < d2:
            # family1 diverged earlier, edge from family1 to family2
            G.add_edge(family1, family2, weight=d1)
        elif d2 < d1:
            # family2 diverged earlier, edge from family2 to family1
            G.add_edge(family2, family1, weight=d2)
        # If equal, no edge (can't determine direction)

    return G


def simplify_graph(G: nx.DiGraph) -> nx.DiGraph:
    """
    Simplify graph by keeping only minimum-weight outgoing edge per node.

    Implements Algorithm 9 from Appendix H.

    Args:
        G: Full directed graph

    Returns:
        Simplified graph with at most one outgoing edge per node
    """
    G_simple = nx.DiGraph()

    for node in G.nodes():
        # Get all outgoing edges
        out_edges = list(G.out_edges(node, data=True))

        if out_edges:
            # Find minimum weight edge
            min_edge = min(out_edges, key=lambda x: x[2].get('weight', float('inf')))
            G_simple.add_edge(min_edge[0], min_edge[1], weight=min_edge[2].get('weight', 0))

    return G_simple


def analyze_phylogenetic_tree(tree_path: str,
                               output_path: Optional[str] = None,
                               graphml_prefix: Optional[str] = None) -> Dict:
    """
    Complete inter-family analysis pipeline.

    Implements Algorithm 5 from Appendix H.

    Args:
        tree_path: Path to phylogenetic tree file
        output_path: Optional path to save results
        graphml_prefix: Optional path prefix; writes ``<prefix>_full.graphml``
            and ``<prefix>_simplified.graphml`` for visualization tools.

    Returns:
        Analysis results including graph structure
    """
    print("Loading tree and aggregating leaves...")
    tree, family_to_leaves = load_tree_and_aggregate_leaves(tree_path)

    print(f"Found {len(family_to_leaves)} families")

    print("Calculating inter-family distances...")
    distances = calculate_inter_family_distances(tree, family_to_leaves)

    print("Building directed graph...")
    G = build_directed_graph(distances)

    print("Simplifying graph...")
    G_simple = simplify_graph(G)

    if graphml_prefix:
        full_path = f"{graphml_prefix}_full.graphml"
        simple_path = f"{graphml_prefix}_simplified.graphml"
        nx.write_graphml(G, full_path)
        nx.write_graphml(G_simple, simple_path)
        print(f"Graphs saved to {full_path} and {simple_path}")

    # Prepare results
    results = {
        'num_families': len(family_to_leaves),
        'families': list(family_to_leaves.keys()),
        'family_sizes': {k: len(v) for k, v in family_to_leaves.items()},
        'full_graph': {
            'num_nodes': G.number_of_nodes(),
            'num_edges': G.number_of_edges(),
            'edges': [(u, v, d['weight']) for u, v, d in G.edges(data=True)]
        },
        'simplified_graph': {
            'num_nodes': G_simple.number_of_nodes(),
            'num_edges': G_simple.number_of_edges(),
            'edges': [(u, v, d['weight']) for u, v, d in G_simple.edges(data=True)]
        }
    }

    # Find hub families (nodes with many incoming edges in simplified graph)
    in_degrees = dict(G_simple.in_degree())
    hubs = sorted(in_degrees.items(), key=lambda x: x[1], reverse=True)[:10]
    results['top_hubs'] = hubs

    if output_path:
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {output_path}")

    return results


def extract_family_subgraph(G: nx.DiGraph,
                            root_family: str,
                            max_depth: int = 2) -> nx.DiGraph:
    """
    Extract subgraph centered on a specific family.

    Useful for case studies like the Mirai analysis in Section 5.4.

    Args:
        G: Full or simplified graph
        root_family: Family to center subgraph on
        max_depth: Maximum distance from root to include

    Returns:
        Subgraph containing root and connected families
    """
    if root_family not in G:
        return nx.DiGraph()

    # BFS to find all nodes within max_depth
    nodes_to_include = {root_family}
    current_level = {root_family}

    for _ in range(max_depth):
        next_level = set()
        for node in current_level:
            # Add successors (outgoing)
            next_level.update(G.successors(node))
            # Add predecessors (incoming)
            next_level.update(G.predecessors(node))
        nodes_to_include.update(next_level)
        current_level = next_level

    return G.subgraph(nodes_to_include).copy()


def analyze_mirai_lineage(tree_path: str) -> Dict:
    """
    Specific analysis for Mirai botnet family as in Section 5.4.

    Returns validated and unvalidated relationships.
    """
    tree, family_to_leaves = load_tree_and_aggregate_leaves(tree_path)
    distances = calculate_inter_family_distances(tree, family_to_leaves)
    G = build_directed_graph(distances)
    G_simple = simplify_graph(G)

    # Extract Mirai subgraph
    mirai_subgraph = extract_family_subgraph(G_simple, 'Mirai', max_depth=1)

    # Known validated Mirai descendants
    validated_descendants = {'Bashlite', 'Okiru', 'MooBot', 'Gafgyt', 'RapperBot'}

    results = {
        'mirai_connections': [],
        'validated': [],
        'unvalidated': []
    }

    for u, v, data in mirai_subgraph.edges(data=True):
        connection = {
            'source': u,
            'target': v,
            'weight': data.get('weight', 0)
        }
        results['mirai_connections'].append(connection)

        if u == 'Mirai':
            if v in validated_descendants:
                results['validated'].append(connection)
            else:
                results['unvalidated'].append(connection)

    return results


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Analyze inter-family evolutionary relationships')
    parser.add_argument('--tree', required=True, help='Path to phylogenetic tree (Newick format)')
    parser.add_argument('--output', default='inter_family_analysis.json',
                       help='Path to save analysis results')
    parser.add_argument('--family', help='Specific family to analyze (e.g., Mirai)')
    parser.add_argument('--depth', type=int, default=2, help='Max depth for family subgraph')
    parser.add_argument('--graphml-prefix',
                        help='Write <prefix>_full.graphml and <prefix>_simplified.graphml')

    args = parser.parse_args()

    if args.family:
        print(f"\nAnalyzing {args.family} family relationships...")
        if args.family.lower() == 'mirai':
            results = analyze_mirai_lineage(args.tree)
        else:
            tree, family_to_leaves = load_tree_and_aggregate_leaves(args.tree)
            distances = calculate_inter_family_distances(tree, family_to_leaves)
            G = build_directed_graph(distances)
            G_simple = simplify_graph(G)
            subgraph = extract_family_subgraph(G_simple, args.family, args.depth)
            results = {
                'family': args.family,
                'connections': [(u, v, d['weight']) for u, v, d in subgraph.edges(data=True)]
            }
    else:
        print("\nRunning full inter-family analysis...")
        results = analyze_phylogenetic_tree(args.tree, args.output, args.graphml_prefix)

    print("\n=== Analysis Results ===")
    print(json.dumps(results, indent=2))
