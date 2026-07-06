import json

class AgentEncoder:
    """Encodes a list of agents [str] to a list of indices [int] and vice versa."""

    def __init__(self, class_path: str | None = None, classes: list[str] | None = None):
        if not (class_path is None or classes is None):
            raise ValueError("Either class_path or classes should be provided")

        if class_path is not None:
            with open(class_path) as f:
                self.classes_ = json.load(f)
                self.classes_.insert(0, "eos")
        else:
            self.classes_ = classes
            self.classes_.insert(0, "eos")

        self.class_to_index = {agent: index for index, agent in enumerate(self.classes_)}

    def __len__(self) -> int:
        """Length is true num_class + 1 because of the eos index."""
        return len(self.classes_)

    def __call__(self, agents: list[str]) -> list[int]:
        return self.encode(agents)

    def encode(self, agents: list[str]) -> list[int]:
        missing_agents = set(agents) - set(self.classes_)
        if missing_agents:
            raise ValueError(f"Agents {missing_agents} not found in agent classes.")
        return [self.class_to_index[agent] for agent in set(agents)]

    def decode(self, indices: list[int]) -> list[str]:
        if not len(indices) > 0:
            return []

        if max(indices) >= len(self.classes_):
            raise IndexError("One or more indices are out of bounds.")
        return [self.classes_[index] for index in indices]
