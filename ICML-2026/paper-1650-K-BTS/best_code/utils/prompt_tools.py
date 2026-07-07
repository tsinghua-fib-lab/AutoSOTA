from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

def get_system_instruction():
    return (
        "You are an expert medicinal chemist specializing in structure-based drug design (SBDD). "
        "Your goal is to optimize a lead compound to improve its binding affinity (lowering docking scores). "
        "You are proficient in SMILES/SMARTS notation, bioisosteric replacement, and medicinal chemistry principles. "
        "Always ensure that the generated molecules are chemically valid and maintain reasonable drug-likeness."
    )

# get Hm
def get_similar_history(m_seed_smiles, history_set, top_n=5):
    try:
        seed_mol = Chem.MolFromSmiles(m_seed_smiles)
        if not seed_mol: return []
        seed_scaffold = Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(seed_mol))

        similar_mols = []

        for hist_smiles in list(history_set)[::-1]:
            if hist_smiles == m_seed_smiles: continue

            hist_mol = Chem.MolFromSmiles(hist_smiles)
            if not hist_mol: continue
            hist_scaffold = Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(hist_mol))

            if hist_scaffold == seed_scaffold:
                similar_mols.append(hist_smiles)
            if len(similar_mols) >= top_n: break
        return similar_mols
    except:
        return []


def generate_user_task_prompt(m_seed, action_info, history_set):

    current_context = (
        f"### Current Lead Molecule\n"
        f"- SMILES: {m_seed.smiles}\n"
        f"- Current Docking Score: {m_seed.score:.2f}\n"
        f"- Scaffold: {m_seed.scaffold_smiles}\n"
    )

    similar_history = get_similar_history(m_seed.smiles, history_set)
    forbidden_section = ""
    if similar_history:
        forbidden_section = "### Forbidden Results (Already Tested)\n"
        forbidden_section += "The following derivatives of this scaffold have already been evaluated and must NOT be repeated:\n"
        for sml in similar_history:
            forbidden_section += f"- {sml}\n"
        forbidden_section += "\n"

    if action_info:
        action_key = action_info["action_key"]
        utility = action_info["expected_utility"]
        is_exact = action_info["is_exact_match"]
        sim = action_info.get("similarity", 1.0)

        parts = action_key.split(">>")
        from_grp = parts[0].replace("[SCAFFOLD]", "")
        to_grp = parts[1].replace("[SCAFFOLD]", "")

        match_desc = "previously successful for THIS scaffold" if is_exact else f"successful for a SIMILAR scaffold (Similarity: {sim:.2f})"
        suggestion_header = "### Strategic Inspiration (From Historical Success)"

        if "[SCAFFOLD]" in action_key:
            evidence_detail = f"Transitioning from core [{from_grp}] to [{to_grp}] improved binding (Utility: {utility:.2f})."
            instruction = "ACTION: Perform a core replacement or structural reorganization inspired by this scaffold hop."
        else:
            evidence_detail = f"Replacing group [{from_grp}] with [{to_grp}] was highly effective (Utility: {utility:.2f})."
            if any(to_grp in s for s in similar_history):
                instruction = (
                    f"CRITICAL ACTION: We already tried adding [{to_grp}] to this scaffold (see Forbidden Results). "
                    f"DO NOT return the same molecule. Instead, use the logic of [{to_grp}] (e.g., strong electron-withdrawing, "
                    f"hydrophobicity) to design a NEW bioisostere like -OCF3, -SF5, or -SO2CF3, or shift the position."
                )
            else:
                instruction = f"ACTION: Design a modification inspired by the shift toward [{to_grp}]."

        expert_guidance = (
            f"{suggestion_header}\n"
            f"- Evidence: This strategy was {match_desc}.\n"
            f"- Insight: {evidence_detail}\n"
            f"- Task Guidance: {instruction}\n"
        )
    else:
        expert_guidance = "### Optimization Strategy\n- No specific history found. Identify improvements based on medicinal chemistry expertise.\n"

    output_constraint = (
        "### Task Requirements\n"
        "1. Provide exactly ONE new SMILES string.\n"
        f"2. MANDATORY: The result MUST be different from the Seed and all Forbidden Results.\n"
        "3. Ensure chemical validity and explain your rationale: How does it relate to the inspiration and bypass the forbidden list?\n"
        "\nResponse Format:\n"
        "Rationale: <Your explanation>\n"
        "New SMILES: <The new SMILES string>"
    )

    return f"{current_context}\n{forbidden_section}{expert_guidance}\n{output_constraint}"