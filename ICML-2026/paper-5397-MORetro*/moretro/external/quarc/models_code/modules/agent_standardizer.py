import json
from pathlib import Path


class AgentStandardizer:
    """Applies conversion rules to standardize agents."""

    def __init__(self, conv_rules: dict | str, other_dict: dict | str):
        if isinstance(conv_rules, str) or isinstance(conv_rules, Path):
            with open(conv_rules, "r") as f:
                conv_rules = json.load(f)

        if isinstance(other_dict, str) or isinstance(other_dict, Path):
            with open(other_dict, "r") as f:
                other_dict = json.load(f)

        self.conv_rules = conv_rules
        self.other_dict = other_dict

    def apply_conv_rules(self, agent_smiles: str) -> str:
        return self.conv_rules.get(agent_smiles, agent_smiles)

    def map_to_other(self, agent_smiles: str) -> str:
        return self.other_dict.get(agent_smiles, agent_smiles)

    def standardize(self, agent_smiles: list[str]) -> list[str]:
        if not isinstance(agent_smiles, list):
            agent_smiles = [agent_smiles]
        return [self.map_to_other(self.apply_conv_rules(smi)) for smi in agent_smiles]
