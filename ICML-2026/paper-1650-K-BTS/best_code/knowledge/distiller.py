from rdkit import Chem

def distill_transformation(mol_poor, mol_best):

    s_mol = mol_poor.scaffold_mol
    if not s_mol: return []

    # labelByIndex=True
    core_p = Chem.ReplaceCore(mol_poor.mol, s_mol, labelByIndex=True)
    core_b = Chem.ReplaceCore(mol_best.mol, s_mol, labelByIndex=True)

    if not core_p or not core_b:
        return []

    def get_fragment_map(core_obj):
        mapping = {}
        for frag in Chem.GetMolFrags(core_obj, asMols=True):
            for atom in frag.GetAtoms():

                if atom.GetSymbol() == '*':
                    anchor = atom.GetIsotope()

                    mapping[anchor] = Chem.MolToSmiles(frag, canonical=True)
        return mapping

    map_p = get_fragment_map(core_p)
    map_b = get_fragment_map(core_b)

    results = []
    # compare same anchor
    all_anchors = set(map_p.keys()) | set(map_b.keys())
    for anchor in all_anchors:
        s_p = map_p.get(anchor, "[*][H]") # 默认氢原子
        s_b = map_b.get(anchor, "[*][H]")
        if s_p != s_b:
            results.append({
                "anchor": anchor,
                "from_smiles": s_p,
                "to_smiles": s_b,
                "delta_score": mol_poor.score - mol_best.score
            })
    return results