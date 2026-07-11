"""
Tree I/O Utilities

Provides comprehensive support for reading and writing phylogenetic trees
in multiple formats: PHYLIP, Newick, NHX, and ete3 internal formats.

Supports the tree construction pipeline described in Section 4.3 of the paper.
"""
import json
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import re


# =============================================================================
# PHYLIP Format Support
# =============================================================================

def read_phylip_distance_matrix(filepath: str) -> Tuple[np.ndarray, List[str]]:
    """
    Read distance matrix from PHYLIP format file.

    PHYLIP format:
    - First line: number of taxa
    - Subsequent lines: taxon_name followed by distances

    Args:
        filepath: Path to PHYLIP format file

    Returns:
        Tuple of (distance_matrix, taxon_names)
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()

    # Parse number of taxa
    num_taxa = int(lines[0].strip())

    taxon_names = []
    distances = []

    for line in lines[1:num_taxa + 1]:
        parts = line.strip().split()
        if not parts:
            continue

        # Relaxed PHYLIP: the taxon name is the first whitespace-delimited
        # token; the remaining tokens are distances. Names have no length limit.
        taxon_names.append(parts[0])
        distances.append([float(d) for d in parts[1:]])

    distance_matrix = np.array(distances)

    # Handle lower triangular format
    if distance_matrix.shape[1] < num_taxa:
        # Convert lower triangular to full matrix
        full_matrix = np.zeros((num_taxa, num_taxa))
        for i in range(num_taxa):
            for j in range(len(distances[i])):
                full_matrix[i][j] = distances[i][j]
                full_matrix[j][i] = distances[i][j]
        distance_matrix = full_matrix

    return distance_matrix, taxon_names


def write_phylip_distance_matrix(distance_matrix: np.ndarray,
                                  taxon_names: List[str],
                                  output_path: str,
                                  lower_triangular: bool = False):
    """
    Write distance matrix to PHYLIP format file.

    Args:
        distance_matrix: Square distance matrix
        taxon_names: List of taxon names
        output_path: Output file path
        lower_triangular: If True, write only lower triangle
    """
    num_taxa = len(taxon_names)

    if distance_matrix.shape != (num_taxa, num_taxa):
        raise ValueError("Distance matrix dimensions must match number of taxa")

    with open(output_path, 'w') as f:
        f.write(f"{num_taxa}\n")

        for i in range(num_taxa):
            # Relaxed PHYLIP: full taxon name, whitespace-separated from distances.
            name = taxon_names[i]

            if lower_triangular:
                distances = distance_matrix[i, :i+1]
            else:
                distances = distance_matrix[i, :]

            distance_str = ' '.join(f"{d:.6f}" for d in distances)
            f.write(f"{name} {distance_str}\n")

    print(f"PHYLIP distance matrix saved to: {output_path}")


# =============================================================================
# Newick Format Support
# =============================================================================

def read_newick_tree(filepath: str) -> 'Tree':
    """
    Read phylogenetic tree from Newick format file.

    Args:
        filepath: Path to Newick file

    Returns:
        ete3 Tree object
    """
    from ete3 import Tree

    with open(filepath, 'r') as f:
        newick_str = f.read().strip()

    # Try different format specifiers
    formats_to_try = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    for fmt in formats_to_try:
        try:
            return Tree(newick_str, format=fmt)
        except Exception:
            continue

    # If all fail, try with default
    return Tree(newick_str)


def write_newick_tree(tree: 'Tree',
                       output_path: str,
                       format: int = 0):
    """
    Write tree to Newick format file.

    Args:
        tree: ete3 Tree object
        output_path: Output file path
        format: Newick format specifier (0-9)
    """
    newick_str = tree.write(format=format)

    with open(output_path, 'w') as f:
        f.write(newick_str)

    print(f"Newick tree saved to: {output_path}")


def parse_newick_string(newick_str: str) -> 'Tree':
    """
    Parse Newick string to tree object.

    Args:
        newick_str: Newick format string

    Returns:
        ete3 Tree object
    """
    from ete3 import Tree

    # Clean the string
    newick_str = newick_str.strip()

    # Ensure ends with semicolon
    if not newick_str.endswith(';'):
        newick_str += ';'

    return Tree(newick_str, format=1)


# =============================================================================
# NHX (New Hampshire eXtended) Format Support
# =============================================================================

def read_nhx_tree(filepath: str) -> 'Tree':
    """
    Read tree from NHX format file.

    NHX extends Newick with metadata in [&&NHX:key=value] format.

    Args:
        filepath: Path to NHX file

    Returns:
        ete3 Tree object with features parsed
    """
    from ete3 import Tree

    with open(filepath, 'r') as f:
        nhx_str = f.read().strip()

    tree = Tree(nhx_str, format=1)

    # Parse NHX features
    nhx_pattern = r'\[&&NHX:([^\]]+)\]'

    for node in tree.traverse():
        if hasattr(node, 'name') and node.name:
            match = re.search(nhx_pattern, node.name)
            if match:
                features_str = match.group(1)
                # Clean node name
                node.name = re.sub(nhx_pattern, '', node.name).strip()

                # Parse features
                for feature in features_str.split(':'):
                    if '=' in feature:
                        key, value = feature.split('=', 1)
                        node.add_feature(key, value)

    return tree


def write_nhx_tree(tree: 'Tree',
                    output_path: str,
                    features: List[str] = None):
    """
    Write tree to NHX format with specified features.

    Args:
        tree: ete3 Tree object
        output_path: Output file path
        features: List of feature names to include
    """
    if features is None:
        features = []

    def format_nhx_node(node):
        nhx_parts = []
        for feat in features:
            if hasattr(node, feat):
                nhx_parts.append(f"{feat}={getattr(node, feat)}")

        if nhx_parts:
            return f"[&&NHX:{':'.join(nhx_parts)}]"
        return ""

    # Build NHX string
    def to_nhx(node):
        if node.is_leaf():
            return f"{node.name}{format_nhx_node(node)}:{node.dist}"
        else:
            children = ','.join(to_nhx(c) for c in node.children)
            return f"({children}){node.name}{format_nhx_node(node)}:{node.dist}"

    nhx_str = to_nhx(tree) + ";"

    with open(output_path, 'w') as f:
        f.write(nhx_str)

    print(f"NHX tree saved to: {output_path}")


# =============================================================================
# Tree Construction from Distance Matrix
# =============================================================================

def build_upgma_tree(distance_matrix: np.ndarray,
                      taxon_names: List[str]) -> 'Tree':
    """
    Build UPGMA tree from distance matrix.

    UPGMA assumes a molecular clock (constant evolutionary rate).

    Args:
        distance_matrix: Square distance matrix
        taxon_names: List of taxon names

    Returns:
        ete3 Tree object
    """
    from scipy.cluster.hierarchy import linkage, to_tree
    from scipy.spatial.distance import squareform
    from ete3 import Tree

    # Convert to condensed distance matrix
    condensed = squareform(distance_matrix)

    # Build UPGMA tree using average linkage
    Z = linkage(condensed, method='average')

    # Convert to Newick format
    def linkage_to_newick(Z, labels):
        n = len(labels)
        nodes = {i: labels[i] for i in range(n)}

        for i, (left, right, dist, count) in enumerate(Z):
            left, right = int(left), int(right)
            new_node = n + i

            left_dist = dist / 2
            right_dist = dist / 2

            nodes[new_node] = f"({nodes[left]}:{left_dist:.6f},{nodes[right]}:{right_dist:.6f})"

        return nodes[2*n - 2] + ";"

    newick = linkage_to_newick(Z, taxon_names)
    return Tree(newick, format=1)


def build_nj_tree(distance_matrix: np.ndarray,
                   taxon_names: List[str]) -> 'Tree':
    """
    Build Neighbor-Joining tree from distance matrix.

    NJ does not assume a molecular clock.

    Note: For large datasets (>10k samples), use RapidNJ external tool
    for better performance as described in Section 4.3.

    Args:
        distance_matrix: Square distance matrix
        taxon_names: List of taxon names

    Returns:
        ete3 Tree object
    """
    from Bio.Phylo.TreeConstruction import DistanceMatrix, DistanceTreeConstructor
    from Bio import Phylo
    from io import StringIO
    from ete3 import Tree

    n = len(taxon_names)

    # Convert to Biopython DistanceMatrix format (lower triangular)
    matrix_list = []
    for i in range(n):
        row = [distance_matrix[i][j] for j in range(i + 1)]
        matrix_list.append(row)

    dm = DistanceMatrix(taxon_names, matrix_list)

    # Build NJ tree
    constructor = DistanceTreeConstructor()
    bio_tree = constructor.nj(dm)

    # Convert to Newick
    output = StringIO()
    Phylo.write(bio_tree, output, 'newick')
    newick_str = output.getvalue()

    return Tree(newick_str, format=1)


# =============================================================================
# Tree Rooting Methods
# =============================================================================

def root_tree_outgroup(tree: 'Tree', outgroup_name: str) -> 'Tree':
    """
    Root tree using outgroup method.

    This is the recommended rooting method achieving 87.1% temporal
    consistency as described in Section 4.4.

    Args:
        tree: Unrooted ete3 Tree object
        outgroup_name: Name of outgroup taxon

    Returns:
        Rooted tree
    """
    tree = tree.copy()

    outgroup_nodes = tree.search_nodes(name=outgroup_name)
    if not outgroup_nodes:
        # Try partial match
        for leaf in tree.get_leaves():
            if outgroup_name in leaf.name:
                outgroup_nodes = [leaf]
                break

    if outgroup_nodes:
        tree.set_outgroup(outgroup_nodes[0])

    return tree


def root_tree_midpoint(tree: 'Tree') -> 'Tree':
    """
    Root tree using midpoint method.

    Places root at midpoint of longest path.

    Args:
        tree: Unrooted ete3 Tree object

    Returns:
        Rooted tree
    """
    tree = tree.copy()

    # Find the two most distant leaves
    max_dist = 0
    leaf1, leaf2 = None, None

    leaves = tree.get_leaves()
    for i, l1 in enumerate(leaves):
        for l2 in leaves[i+1:]:
            dist = l1.get_distance(l2)
            if dist > max_dist:
                max_dist = dist
                leaf1, leaf2 = l1, l2

    if leaf1 and leaf2:
        # Find midpoint
        midpoint_dist = max_dist / 2

        # Get path between leaves
        ancestor = tree.get_common_ancestor(leaf1, leaf2)

        # Root at the midpoint
        dist_to_ancestor = leaf1.get_distance(ancestor)
        if dist_to_ancestor < midpoint_dist:
            # Midpoint is on leaf2's side
            tree.set_outgroup(leaf2)
        else:
            tree.set_outgroup(leaf1)

    return tree


# =============================================================================
# Label Mapping Utilities
# =============================================================================

def create_taxon_mapping(embeddings_path: str,
                          class_mapping_path: Optional[str] = None) -> Dict[str, Dict]:
    """
    Create mapping between short taxon names (for PHYLIP) and full sample info.

    Args:
        embeddings_path: Path to embeddings JSON
        class_mapping_path: Optional path to class name mapping

    Returns:
        Dictionary mapping short names to {sha, family, index}
    """
    with open(embeddings_path, 'r') as f:
        embeddings = json.load(f)

    # Load class mapping if provided
    class_mapping = {}
    if class_mapping_path:
        with open(class_mapping_path, 'r') as f:
            content = f.read()
            try:
                class_mapping = eval(content)
            except Exception:
                class_mapping = json.loads(content)

        # Reverse mapping (index -> name)
        class_mapping = {str(v): k for k, v in class_mapping.items()}

    mapping = {}
    family_counters = {}

    for sha, data in embeddings.items():
        # Get family name
        if isinstance(data, dict):
            label = data.get('label', 'Unknown')
            if isinstance(label, (int, float)):
                label = class_mapping.get(str(int(label)), f'Family{int(label)}')
        else:
            label = 'Unknown'

        # Build a unique taxon label from the family name and a counter.
        family_short = label.replace(' ', '')

        if family_short not in family_counters:
            family_counters[family_short] = 0

        counter = family_counters[family_short]
        family_counters[family_short] += 1

        short_name = f"{family_short}_{counter}"

        mapping[short_name] = {
            'sha': sha,
            'family': label,
            'index': counter
        }

    return mapping


def save_taxon_mapping(mapping: Dict[str, Dict], output_path: str):
    """Save taxon mapping to JSON file."""
    with open(output_path, 'w') as f:
        json.dump(mapping, f, indent=2)
    print(f"Taxon mapping saved to: {output_path}")


def load_taxon_mapping(mapping_path: str) -> Dict[str, Dict]:
    """Load taxon mapping from JSON file."""
    with open(mapping_path, 'r') as f:
        return json.load(f)


# =============================================================================
# Tree Format Detection and Conversion
# =============================================================================

def detect_tree_format(filepath: str) -> str:
    """
    Detect the format of a tree file.

    Args:
        filepath: Path to tree file

    Returns:
        Format string: 'newick', 'nhx', 'phylip', or 'unknown'
    """
    with open(filepath, 'r') as f:
        content = f.read(1000)  # Read first 1000 chars

    # Check for NHX
    if '[&&NHX:' in content:
        return 'nhx'

    # Check for Newick (contains parentheses and semicolon)
    if '(' in content and ');' in content:
        return 'newick'

    # Check for PHYLIP (first line is a number)
    first_line = content.split('\n')[0].strip()
    if first_line.isdigit():
        return 'phylip'

    return 'unknown'


def convert_tree_format(input_path: str,
                        output_path: str,
                        output_format: str = 'newick'):
    """
    Convert tree file between formats.

    Args:
        input_path: Input tree file path
        output_path: Output file path
        output_format: Target format ('newick', 'nhx', 'phylip')
    """
    input_format = detect_tree_format(input_path)

    # Read tree
    if input_format == 'newick':
        tree = read_newick_tree(input_path)
    elif input_format == 'nhx':
        tree = read_nhx_tree(input_path)
    elif input_format == 'phylip':
        matrix, names = read_phylip_distance_matrix(input_path)
        tree = build_nj_tree(matrix, names)
    else:
        raise ValueError(f"Unknown input format for {input_path}")

    # Write tree
    if output_format == 'newick':
        write_newick_tree(tree, output_path)
    elif output_format == 'nhx':
        write_nhx_tree(tree, output_path)
    else:
        raise ValueError(f"Unsupported output format: {output_format}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Tree I/O utilities')
    subparsers = parser.add_subparsers(dest='command')

    # Read PHYLIP
    read_phy = subparsers.add_parser('read-phylip', help='Read PHYLIP distance matrix')
    read_phy.add_argument('input', help='Input PHYLIP file')

    # Build tree
    build = subparsers.add_parser('build', help='Build tree from distance matrix')
    build.add_argument('input', help='Input PHYLIP or NPY file')
    build.add_argument('--method', choices=['upgma', 'nj'], default='nj')
    build.add_argument('--output', default='tree.nwk')
    build.add_argument('--labels', help='Labels JSON file')

    # Convert
    convert = subparsers.add_parser('convert', help='Convert tree format')
    convert.add_argument('input', help='Input tree file')
    convert.add_argument('output', help='Output tree file')
    convert.add_argument('--format', choices=['newick', 'nhx'], default='newick')

    # Root
    root = subparsers.add_parser('root', help='Root tree')
    root.add_argument('input', help='Input tree file')
    root.add_argument('--method', choices=['outgroup', 'midpoint'], default='outgroup')
    root.add_argument('--outgroup', help='Outgroup taxon name')
    root.add_argument('--output', default='rooted_tree.nwk')

    # View
    view = subparsers.add_parser('view', help='Print a Newick (.nwk) tree as ASCII')
    view.add_argument('input', help='Input Newick tree file')
    view.add_argument('--max-leaves', type=int, default=200,
                      help='Skip ASCII art above this leaf count (default: 200)')

    args = parser.parse_args()

    if args.command == 'read-phylip':
        matrix, names = read_phylip_distance_matrix(args.input)
        print(f"Read {len(names)} taxa")
        print(f"Matrix shape: {matrix.shape}")

    elif args.command == 'build':
        if args.input.endswith('.npy'):
            matrix = np.load(args.input)
            with open(args.labels, 'r') as f:
                labels = json.load(f)
            if isinstance(labels, dict):
                labels = list(labels.keys())
        else:
            matrix, labels = read_phylip_distance_matrix(args.input)

        if args.method == 'upgma':
            tree = build_upgma_tree(matrix, labels)
        else:
            tree = build_nj_tree(matrix, labels)

        write_newick_tree(tree, args.output)

    elif args.command == 'convert':
        convert_tree_format(args.input, args.output, args.format)

    elif args.command == 'root':
        tree = read_newick_tree(args.input)
        if args.method == 'outgroup' and args.outgroup:
            rooted = root_tree_outgroup(tree, args.outgroup)
        else:
            rooted = root_tree_midpoint(tree)
        write_newick_tree(rooted, args.output)

    elif args.command == 'view':
        tree = read_newick_tree(args.input)
        leaves = tree.get_leaves()
        nodes = sum(1 for _ in tree.traverse())
        print(f"{args.input}: {len(leaves)} leaves, {nodes} nodes")
        if len(leaves) <= args.max_leaves:
            print(tree.get_ascii())
        else:
            print(f"Tree has {len(leaves)} leaves (> --max-leaves={args.max_leaves}); "
                  "skipping ASCII art.")
            print("Open it in an interactive viewer (iTOL, IcyTree, FigTree), "
                  "or see inter_family_vis.html for the family-level view.")
