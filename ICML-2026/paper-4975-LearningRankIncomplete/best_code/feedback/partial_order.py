from .rank_breakable import RankBreakableFeedback
from .preference import PreferenceFeedback
from collections import defaultdict

class PartialOrderFeedback(RankBreakableFeedback):
    def __init__(self, preferences: list[PreferenceFeedback]) -> None:
        self.preferences = preferences
        self._validate_partial_order()

    def _validate_partial_order(self) -> None:
        # Build an adjacency list mapping a predecessor ID to a set of successor IDs
        adj = defaultdict(set)
        for pref in self.preferences:
            adj[pref.prec_id].add(pref.succ_id)

        # 1. Check for cycles (Asymmetry/Irreflexivity) using Depth First Search
        visited = set()
        rec_stack = set()

        def has_cycle(node_id):
            visited.add(node_id)
            rec_stack.add(node_id)

            for neighbor in adj.get(node_id, []):
                if neighbor not in visited:
                    if has_cycle(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True
            
            rec_stack.remove(node_id)
            return False

        for node in list(adj.keys()):
            if node not in visited:
                if has_cycle(node):
                    raise ValueError("The provided preferences contain a cycle and do not form a valid partial order.")

        # 2. Check for explicit transitivity
        for a in adj:
            for b in adj[a]:
                for c in adj.get(b, []):
                    if c not in adj[a]:
                        raise ValueError(f"Missing transitive relation: {a} < {b} and {b} < {c}, but {a} < {c} is missing.")

    def rank_breaking(self) -> list[PreferenceFeedback]:
        # Since the input is already a validated list of preference feedback, 
        # we just return it.
        return self.preferences
