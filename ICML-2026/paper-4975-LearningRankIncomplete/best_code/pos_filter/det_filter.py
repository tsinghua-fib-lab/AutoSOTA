from typing import List, Tuple
from collections import defaultdict
from .pos_filter import PositionalFilter
from feedback import PreferenceFeedback
# Assuming PartialOrderFeedback is importable from your module structure
from feedback import PartialOrderFeedback 

class DeterministicPositionalFilter(PositionalFilter):
    def __init__(self, n: int, positions: List[Tuple[int, int]]) -> None:
        super().__init__(n=n)
        
        # Automatically handle duplicates by converting to a set
        self.positions = list(set(positions))
        self._validate_and_close_partial_order()

    def _validate_and_close_partial_order(self) -> None:
        adj = defaultdict(set)
        
        for p1, p2 in self.positions:
            # Check bounds
            if not (0 <= p1 < self.n and 0 <= p2 < self.n):
                raise ValueError(f"Positions must be between 0 and {self.n - 1}. Found ({p1}, {p2}).")
            
            # Check irreflexivity directly
            if p1 == p2:
                raise ValueError(f"Strict inequality violation: Position cannot precede itself ({p1}).")
            
            adj[p1].add(p2)

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
                    raise ValueError("The provided position pairs contain a cycle and do not form a valid partial order.")

        # 2. Automatically compute and apply the transitive closure
        closure_edges = set(self.positions)
        
        def get_reachable(start_node):
            reachable = set()
            stack = [start_node]
            while stack:
                curr = stack.pop()
                for neighbor in adj.get(curr, []):
                    if neighbor not in reachable:
                        reachable.add(neighbor)
                        stack.append(neighbor)
            return reachable

        for node in range(self.n):
            if node in adj:
                reachable_nodes = get_reachable(node)
                for r_node in reachable_nodes:
                    closure_edges.add((node, r_node))
                    # Keep the adjacency list updated with the new edges
                    adj[node].add(r_node)

        # Overwrite self.positions with the fully transitively closed set
        self.positions = list(closure_edges)

    def filter(self, ranking: List[int]) -> PartialOrderFeedback:
        # Validate ranking length and uniqueness
        if len(ranking) != self.n or len(set(ranking)) != self.n:
            raise ValueError(f"The positional filter expects rankings with exactly {self.n} unique elements.")
        
        # Validate ranking item bounds
        if not all(0 <= item < self.n for item in ranking):
            raise ValueError(f"All item IDs in the ranking must be between 0 and {self.n - 1}.")

        preferences = []
        for p1, p2 in self.positions:
            # ranking maps position index to item ID
            prec_item_id = ranking[p1]
            succ_item_id = ranking[p2]
            
            preferences.append(PreferenceFeedback(prec_id=prec_item_id, succ_id=succ_item_id))
            
        return PartialOrderFeedback(preferences)
