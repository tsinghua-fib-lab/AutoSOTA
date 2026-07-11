"""
Outlier Detection for Phylogenetic Trees

Implements Algorithms 1 and 2 from Appendix C: IQR-based outlier detection
to remove samples that distort tree topology.
"""
import json
import numpy as np
from typing import Dict, List, Set, Tuple
from collections import defaultdict
from ete3 import Tree


def compute_lateral_distance(tree: Tree, leaf1: str, leaf2: str) -> float:
    """
    Compute lateral distance between two leaves.

    Lateral distance = d(leaf1, MRCA) + d(MRCA, leaf2)

    Args:
        tree: Phylogenetic tree
        leaf1: Name of first leaf
        leaf2: Name of second leaf

    Returns:
        Lateral distance (sum of path lengths through MRCA)
    """
    try:
        node1 = tree.search_nodes(name=leaf1)
        node2 = tree.search_nodes(name=leaf2)

        if not node1 or not node2:
            return float('inf')

        return node1[0].get_distance(node2[0])
    except Exception:
        return float('inf')


def compute_leaf_to_leaf_distances(tree: Tree, leaves: List[str]) -> Dict[str, List[float]]:
    """
    Compute all pairwise lateral distances within a set of leaves.

    Args:
        tree: Phylogenetic tree
        leaves: List of leaf names

    Returns:
        Dictionary mapping each leaf to its list of distances to other leaves
    """
    distances = defaultdict(list)

    for i, leaf1 in enumerate(leaves):
        for leaf2 in leaves[i+1:]:
            dist = compute_lateral_distance(tree, leaf1, leaf2)
            if dist != float('inf'):
                distances[leaf1].append(dist)
                distances[leaf2].append(dist)

    return distances


def compute_median_lateral_distance(distances: List[float]) -> float:
    """Compute median of a list of distances."""
    if not distances:
        return 0.0
    return float(np.median(distances))


def compute_iqr_threshold(median_distances: List[float], multiplier: float = 1.5) -> float:
    """
    Compute IQR-based outlier threshold.

    Threshold = Q3 + multiplier * IQR

    Args:
        median_distances: List of median lateral distances for all samples
        multiplier: IQR multiplier (default 1.5 for standard outlier detection)

    Returns:
        Threshold value above which samples are considered outliers
    """
    if len(median_distances) < 4:
        return float('inf')

    q1 = np.percentile(median_distances, 25)
    q3 = np.percentile(median_distances, 75)
    iqr = q3 - q1

    return q3 + multiplier * iqr


def detect_outliers_from_tree(tree: Tree,
                               family_labels: Dict[str, str],
                               iqr_multiplier: float = 1.5) -> Set[str]:
    """
    Detect outliers using IQR-based method on lateral distances.

    Implements Algorithm 1 from Appendix C.

    Args:
        tree: Initial phylogenetic tree
        family_labels: Mapping from leaf name to family name
        iqr_multiplier: Multiplier for IQR threshold

    Returns:
        Set of leaf names identified as outliers
    """
    outliers = set()

    # Group leaves by family
    families = defaultdict(list)
    for leaf in tree.get_leaves():
        if leaf.name in family_labels:
            families[family_labels[leaf.name]].append(leaf.name)

    # Process each family
    for family_name, family_leaves in families.items():
        if len(family_leaves) < 2:
            continue

        # Compute all pairwise distances within family
        distances = compute_leaf_to_leaf_distances(tree, family_leaves)

        # Compute median lateral distance for each sample
        median_distances = {}
        for leaf in family_leaves:
            if leaf in distances and distances[leaf]:
                median_distances[leaf] = compute_median_lateral_distance(distances[leaf])

        if len(median_distances) < 4:
            continue

        # Compute IQR threshold
        median_values = list(median_distances.values())
        threshold = compute_iqr_threshold(median_values, iqr_multiplier)

        # Flag outliers
        for leaf, median_dist in median_distances.items():
            if median_dist > threshold:
                outliers.add(leaf)

    return outliers


def filter_distance_matrix(distance_matrix: np.ndarray,
                           labels: List[str],
                           outliers: Set[str]) -> Tuple[np.ndarray, List[str]]:
    """
    Remove outliers from distance matrix.

    Args:
        distance_matrix: Original n x n distance matrix
        labels: List of n labels corresponding to matrix rows/columns
        outliers: Set of labels to remove

    Returns:
        Filtered distance matrix and updated label list
    """
    # Find indices to keep
    keep_indices = [i for i, label in enumerate(labels) if label not in outliers]

    # Filter matrix
    filtered_matrix = distance_matrix[np.ix_(keep_indices, keep_indices)]

    # Filter labels
    filtered_labels = [labels[i] for i in keep_indices]

    return filtered_matrix, filtered_labels


def two_stage_tree_construction(distance_matrix: np.ndarray,
                                 labels: List[str],
                                 family_labels: Dict[str, str],
                                 tree_method: str = 'nj',
                                 iqr_multiplier: float = 1.5) -> Tuple[np.ndarray, List[str], Set[str]]:
    """
    Two-stage tree construction with outlier removal.

    Implements Algorithms 1 and 2 from Appendix C:
    1. Build initial tree from all samples
    2. Detect outliers using IQR method
    3. Rebuild tree without outliers

    Args:
        distance_matrix: Original distance matrix
        labels: Sample labels
        family_labels: Family membership mapping
        tree_method: Tree construction method ('nj' or 'upgma')
        iqr_multiplier: IQR threshold multiplier

    Returns:
        Tuple of (clean distance matrix, clean labels, outlier set)
    """
    from scipy.cluster.hierarchy import linkage, to_tree
    from scipy.spatial.distance import squareform

    # Stage 1: Build initial tree
    print("Stage 1: Building initial tree...")

    # Convert to condensed distance matrix for scipy
    condensed = squareform(distance_matrix)

    if tree_method == 'upgma':
        Z = linkage(condensed, method='average')
    else:  # nj - approximate with ward for scipy
        Z = linkage(condensed, method='ward')

    # Convert to ete3 tree for analysis
    # Create a simple Newick representation
    def linkage_to_newick(Z, labels):
        """Convert scipy linkage to Newick format."""
        n = len(labels)
        nodes = {i: labels[i] for i in range(n)}

        for i, (left, right, dist, count) in enumerate(Z):
            left, right = int(left), int(right)
            new_node = n + i
            nodes[new_node] = f"({nodes[left]}:{dist/2},{nodes[right]}:{dist/2})"

        return nodes[2*n - 2] + ";"

    newick = linkage_to_newick(Z, labels)
    initial_tree = Tree(newick, format=1)

    # Stage 1: Detect outliers
    print("Detecting outliers...")
    outliers = detect_outliers_from_tree(initial_tree, family_labels, iqr_multiplier)
    print(f"Found {len(outliers)} outliers")

    # Stage 2: Filter and return clean data
    if outliers:
        clean_matrix, clean_labels = filter_distance_matrix(distance_matrix, labels, outliers)
    else:
        clean_matrix, clean_labels = distance_matrix, labels

    return clean_matrix, clean_labels, outliers


def save_outlier_report(outliers: Set[str],
                        family_labels: Dict[str, str],
                        output_path: str):
    """Save outlier detection report to file."""
    report = {
        'total_outliers': len(outliers),
        'outliers_by_family': defaultdict(list)
    }

    for outlier in outliers:
        family = family_labels.get(outlier, 'Unknown')
        report['outliers_by_family'][family].append(outlier)

    report['outliers_by_family'] = dict(report['outliers_by_family'])

    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"Outlier report saved to {output_path}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Detect and remove outliers from phylogenetic analysis')
    parser.add_argument('--embeddings', required=True, help='Path to embeddings JSON file')
    parser.add_argument('--distance-matrix', help='Path to distance matrix NPY file')
    parser.add_argument('--families', required=True, help='Path to family labels JSON file')
    parser.add_argument('--iqr-multiplier', type=float, default=1.5, help='IQR multiplier for threshold')
    parser.add_argument('--output-matrix', help='Path to save filtered distance matrix')
    parser.add_argument('--output-report', default='outlier_report.json', help='Path to save outlier report')

    args = parser.parse_args()

    # Load data
    with open(args.embeddings, 'r') as f:
        embeddings_data = json.load(f)

    with open(args.families, 'r') as f:
        family_labels = json.load(f)

    labels = list(embeddings_data.keys())

    # Load or compute distance matrix
    if args.distance_matrix:
        distance_matrix = np.load(args.distance_matrix)
    else:
        print("Computing distance matrix from embeddings...")
        embeddings = np.array([embeddings_data[label]['embedding'] for label in labels])
        from scipy.spatial.distance import cdist
        distance_matrix = cdist(embeddings, embeddings, metric='euclidean')

    # Run two-stage construction
    clean_matrix, clean_labels, outliers = two_stage_tree_construction(
        distance_matrix, labels, family_labels, iqr_multiplier=args.iqr_multiplier
    )

    # Save results
    if args.output_matrix:
        np.save(args.output_matrix, clean_matrix)
        print(f"Clean distance matrix saved to {args.output_matrix}")

        # Also save clean labels
        labels_path = args.output_matrix.replace('.npy', '_labels.json')
        with open(labels_path, 'w') as f:
            json.dump(clean_labels, f)

    save_outlier_report(outliers, family_labels, args.output_report)

    print(f"\nSummary:")
    print(f"  Original samples: {len(labels)}")
    print(f"  Outliers removed: {len(outliers)}")
    print(f"  Clean samples: {len(clean_labels)}")
