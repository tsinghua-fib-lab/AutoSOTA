""" 
Uses individual scores for agents to calculate compound scores for combined reagents
"""

import json

def load_tox_scores(file_path: str) -> dict:
    """Load toxicity scores from a JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)
    
def calculate_combined_tox_score(components: str, tox_scores: dict) -> float:
    """Calculate the combined toxicity score for a list of components."""
    total_score = 0.0
    for component in components.split("."):
        score = tox_scores.get(component, 100)  # Default to 100, substitute manually afterwards
        total_score = max(score, total_score)  # Use the highest score among components
    return total_score if components else 0.0

if __name__ == "__main__":
    tox_scores = load_tox_scores("chat_scoring/individual_components.json")
    with open("agent_encoder_list.json", 'r') as f:
        agents = json.load(f)
    # create dict between original smiles and their scores
    smiles_to_score = {item['original_smiles']: item['toxicity_score'] for item in tox_scores}
    total_score = {a: calculate_combined_tox_score(a, smiles_to_score) for a in agents}
    with open("agents_with_scores.json", 'w') as f:
        json.dump(total_score, f, indent=2)