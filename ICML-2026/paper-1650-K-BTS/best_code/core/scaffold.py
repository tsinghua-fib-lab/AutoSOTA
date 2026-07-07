from collections import defaultdict

def cluster_molecules(molecule_list):
    clusters = defaultdict(list)
    for m in molecule_list:
        if m.scaffold_smiles:
            clusters[m.scaffold_smiles].append(m)
    return {k: v for k, v in clusters.items() if len(v) > 1}