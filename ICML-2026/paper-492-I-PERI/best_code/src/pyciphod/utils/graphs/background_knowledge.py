class BackgroundKnowledge:
    def __init__(self):
        self._mandatory_edges = set()
        self._forbidden_edges = set()
        self._mandatory_orientations = set()
        self._forbidden_orientations = set()
        self._non_descendants = dict()

    # -------------------- Mandatory edges --------------------
    def add_mandatory_edge(self, u, v):
        self._mandatory_edges.add((u, v))

    def add_mandatory_edges_from(self, edges):
        self._mandatory_edges.update(edges)

    # -------------------- Forbidden edges --------------------
    def add_forbidden_edge(self, u, v):
        self._forbidden_edges.add((u, v))

    def add_forbidden_edges_from(self, edges):
        self._forbidden_edges.update(edges)

    # -------------------- Mandatory orientations --------------------
    def add_mandatory_orientation(self, u, v):
        self._mandatory_orientations.add((u, v))

    def add_mandatory_orientations_from(self, edges):
        self._mandatory_orientations.update(edges)

    # -------------------- Forbidden orientations --------------------
    def add_forbidden_orientation(self, u, v):
        self._forbidden_orientations.add((u, v))

    def add_forbidden_orientations_from(self, edges):
        self._forbidden_orientations.update(edges)

    # -------------------- Non-descendant constraints --------------------
    def add_non_descendant(self, node, non_descendant):
        if non_descendant not in self._non_descendants:
            self._non_descendants[non_descendant] = set()
        self._non_descendants[non_descendant].add(node)

        # Automatically forbid orientation non_descendant <- node
        self.add_forbidden_orientation(node, non_descendant)

    def add_non_descendants_from(self, node, nodes):
        if node not in self._non_descendants:
            self._non_descendants[node] = set()
        self._non_descendants[node].update(nodes)

        # Automatically forbid orientation for each non-descendant
        for nd in nodes:
            self.add_forbidden_orientation(node, nd)

    def remove_non_descendant(self, node, non_descendant):
        if node in self._non_descendants:
            self._non_descendants[node].discard(non_descendant)
            self._forbidden_orientations.discard((non_descendant, node))
            if not self._non_descendants[node]:
                del self._non_descendants[node]

    # -------------------- Getters --------------------
    def get_mandatory_edges(self):
        """Return a copy of all mandatory edges (undirected)."""
        return self._mandatory_edges.copy()

    def get_forbidden_edges(self):
        """Return a copy of all forbidden edges (undirected)."""
        return self._forbidden_edges.copy()

    def get_mandatory_orientations(self):
        """Return a copy of all mandatory orientations (u -> v)."""
        return self._mandatory_orientations.copy()

    def get_forbidden_orientations(self):
        """Return a copy of all forbidden orientations (u -> v)."""
        return self._forbidden_orientations.copy()

    def get_non_descendants(self):
        """Return a copy of the non-descendant constraints {node: set(non_descendants)}."""
        return {k: v.copy() for k, v in self._non_descendants.items()}