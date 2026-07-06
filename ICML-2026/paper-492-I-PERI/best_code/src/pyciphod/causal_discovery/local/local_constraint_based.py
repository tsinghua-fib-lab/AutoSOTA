from abc import ABC, abstractmethod
from typing import Type, Optional
import pandas as pd
from itertools import combinations

from pyciphod.causal_discovery.basic.constraint_based import ConstraintBased
from pyciphod.causal_discovery.basic.constraint_based import PC
from pyciphod.utils.graphs.partially_specified_graphs import LocalEssentialGraph
from pyciphod.utils.graphs.orientation_rules import meek_rule_1, meek_rule_2, meek_rule_3
from pyciphod.utils.stat_tests.independence_tests import CiTests, FisherZTest as FisherZ
from pyciphod.utils.graphs.background_knowledge import BackgroundKnowledge


class LocalConstraintBased:
    """
    Base class for local constraint-based causal discovery algorithms.
    """
    def __init__(self, target, sparsity: float = 0.05, ci_test: Type[CiTests] = FisherZ, background_knowledge: Optional[BackgroundKnowledge] = None, twd: Optional[bool] = False):
        # super().__init__(sparsity=sparsity, ci_test=ci_test, background_knowledge=background_knowledge, twd=twd)
        self._sparsity = sparsity
        self._ci_test = ci_test
        self._bk = background_knowledge if background_knowledge is not None else BackgroundKnowledge()
        self._twd = twd
        self.performed_tests = set()
        self.nb_ci_tests = 0
        self.sepset = dict()
        self.g_hat = self._initialize_graph() # Graph representation (e.g., LEG for PC)

        # Normalise `target` pour qu'il soit toujours une liste de nœuds.
        # Evite que 'X' soit interprété comme ['X'] vs ['X1','X2'] et surtout évite list('X') -> ['X']
        if isinstance(target, str):
            self._target = [target]
        elif isinstance(target, (list, set, tuple)):
            self._target = list(target)
        else:
            # fallback: wrap into a single-element list
            self._target = [target]

        self.neighborhood_h = set()
        # self._visited = set()

    @abstractmethod
    def _initialize_graph(self):
        """Return the graph object used by the algorithm."""
        pass


    @abstractmethod
    def _apply_background_knowledge(self):
        """Apply background knowledge constraints to the graph structure."""
        pass


    @abstractmethod
    def _skeleton(self, data: pd.DataFrame = None, max_sepset_size: int = None):
        """Construct the skeleton of the graph using CI tests."""
        pass

    @abstractmethod
    def _orientation(self):
        """Orient edges using rules based on the skeleton and separation sets."""
        pass

    @abstractmethod
    def run(self, data: pd.DataFrame, hop: int = 0):
        pass


class LocPC(LocalConstraintBased):
    def __init__(self, target, sparsity: float = 0.05, ci_test: Type[CiTests] = FisherZ, background_knowledge: Optional[BackgroundKnowledge] = None, twd: bool = False):
        # Initialize base class which sets up common fields
        super().__init__(target=target, sparsity=sparsity, ci_test=ci_test, background_knowledge=background_knowledge, twd=twd)

        # LocPC-specific initialization
        self._knowntests = {}
        self.neighborhood_h = set()

        # graph representation used by LocPC
        self.performed_tests = set()
        self.nb_ci_tests = 0
        self.sepset = dict()
        self.g_hat = self._initialize_graph()  # Graph representation (e.g., LEG for LocPC)

    def _skeleton(self, data: pd.DataFrame = None, max_sepset_size: int = None):
        """Construct the skeleton of the graph using CI tests."""
        # This method is integrated into the run method for LocPC, as it iteratively updates the skeleton while expanding the neighborhood.
        return 1

    def _initialize_graph(self):
        """Return the graph object used by the algorithm."""
        return LocalEssentialGraph()

    def _update_skeleton(self, D_set, data: pd.DataFrame = None, max_sepset_size: int = None):
        bk_nd = self._bk.get_non_descendants()
        nodes = list(data.columns)
        bk_non_descendants = {node: bk_nd.get(node, set()) for node in nodes}

        for d in D_set: 
            for b in [x for x in nodes if (x not in self._visited) and x!=d]:
                self.g_hat.add_undirected_edge(d,b)
        data_test = data
        s = 0
        repeat = True 
        while repeat: 
            repeat = False       
            adj = {x: self.g_hat.get_adjacencies(x) for x in nodes}
            for d in D_set: 
                self._visited.add(d)
                if len(adj[d]) - 1 >= s:
                    repeat = True
                    for b in [x for x in sorted(adj[d]) if not (x in self._visited and d in bk_non_descendants[x])]:
                        for S in combinations([a for a in sorted(adj[d]) if a != b], s):
                            if (d,b,S) in self._knowntests:
                                p_val = self._knowntests[(d,b,S)]
                            else:
                                test = self._ci_test(d,b,list(S), self._twd)
                                self.performed_tests.add((d,b,S))
                                self.nb_ci_tests += 1
                                if self._twd:
                                    data_test = data.dropna(subset=[d, b] + list(S))
                                p_val = test.get_pvalue(data_test)
                                self._knowntests[(d,b,S)] = self._knowntests[(b,d,S)] = p_val
                            if p_val > self._sparsity:
                                self.g_hat.remove_undirected_edge(d,b)
                                self.sepset[(d, b)] = self.sepset[(b, d)] = S
                                break
            s += 1

    def _apply_background_knowledge(self):
        """
        Apply background knowledge constraints to the leg:
        1) Remove forbidden edges and add mandatory edges in the skeleton.
        2) Orient edges according to mandatory and forbidden orientations if present.
        """
        if not self._bk:
            return 

        # --- Step 1: enforce mandatory and forbidden edges ---

        # Add mandatory edges if missing
        for u, v in self._bk.get_mandatory_edges():
            if (u, v) not in self.g_hat.get_undirected_edges() and (v, u) not in self.g_hat.get_undirected_edges():
                self.g_hat.add_undirected_edge(u, v)

        # --- Step 2: enforce orientations ---
        # Mandatory orientations
        for u, v in self._bk.get_mandatory_orientations():
            if (u, v) in self.g_hat.get_undirected_edges() or (v, u) in self.g_hat.get_undirected_edges():
                self.g_hat.remove_undirected_edge(u, v)
                self.g_hat.remove_undirected_edge(v, u)
                self.g_hat.add_directed_edge(u, v)

        # Forbidden orientations
        for u, v in self._bk.get_forbidden_orientations():
            if (u, v) in self.g_hat.get_undirected_edges():
                # Orient as v -> u instead
                self.g_hat.remove_undirected_edge(u, v)
                self.g_hat.add_directed_edge(v, u)
            elif (v, u) in self.g_hat.get_undirected_edges():
                # Orient as u -> v instead
                self.g_hat.remove_undirected_edge(v, u)
                self.g_hat.add_directed_edge(u, v)


    def _uc_rule(self):
        nodes = self.g_hat.get_vertices()
        adj = {x: self.g_hat.get_adjacencies(x) for x in nodes}
        for x in nodes:
            for y in sorted(adj[x]):
                for z in sorted(adj[y]):
                    if (y,x) in self.g_hat.get_directed_edges() or (y, z) in self.g_hat.get_directed_edges():
                        continue 
                    if z == x or z in sorted(adj[x]) or any(node not in self.neighborhood_h for node in (x, y, z)):
                        continue
                    if y not in self.sepset.get((x, z), []):
                        self.g_hat.remove_undirected_edge(x, y)
                        self.g_hat.remove_undirected_edge(y, z)
                        self.g_hat.add_directed_edges_from([(x, y), (z, y)])


    def _apply_meek_rules(self):
        nodes = self.g_hat.get_vertices()
        changed = False
        adj = {x: self.g_hat.get_adjacencies(x) for x in nodes}

        for x in self._visited:
            for y in sorted(set(adj[x])):
                # Rule 1:
                for z in sorted(set(adj[y]) - set(adj[x])):
                    if any(node not in self.neighborhood_h for node in (x, y, z)):
                        continue
                    if (x,y) in self.g_hat.get_directed_edges() and (y,z) in self.g_hat.get_undirected_edges():
                        changed = True
                        self.g_hat.remove_undirected_edge(z, y)
                        self.g_hat.add_directed_edge(y, z)
                # Rule 2:
                for z in sorted(set(adj[y]) & set(adj[x])):
                    if any(node not in self.neighborhood_h for node in (x, y, z)):
                        continue
                    if (x,y) in self.g_hat.get_directed_edges() and (y,z) in self.g_hat.get_directed_edges() and (x,z) in self.g_hat.get_undirected_edges():
                        changed = True
                        self.g_hat.remove_undirected_edge(x, z)
                        self.g_hat.add_directed_edge(x, z)

        # Rule 3:
        for x in self._visited:
            for y in sorted(set(adj[x])):
                if (x, y) not in self.g_hat.get_directed_edges():
                    continue
                for z in sorted(set(adj[y]) - set(adj[x])):
                    if (z, y) not in self.g_hat.get_directed_edges():
                        continue
                    for w in sorted(set(adj[x]) & set(adj[y]) & set(adj[z])):
                        if any(node not in self.neighborhood_h for node in (x, y, z, w)):
                            continue
                        undirected_triplet = [(w, y), (w, x), (z, w)]
                        if all(edge in self.g_hat.get_undirected_edges() for edge in undirected_triplet):
                            changed = True
                            self.g_hat.remove_undirected_edge(w, y)
                            self.g_hat.add_directed_edge(w, y)

        return changed
    
    def _nnc_rule(self):
        nodes = self.g_hat.get_vertices()
        for d in self._visited:
            for a in sorted(set(self.g_hat.get_adjacencies(d)) - self._visited):
                if (d,a) not in self.g_hat.get_undirected_edges():
                    continue
                nnc = True
                outside_nodes = set(nodes) - self._visited - set(self.g_hat.get_adjacencies(d))
                if len(outside_nodes) == 0:
                    continue
                for w in outside_nodes:
                    if a in self.sepset[(d,w)]:
                        nnc = False
                        break
                if nnc:
                    self.g_hat.remove_undirected_edge(d,a)
                    self.g_hat.add_uncertain_edge(d,a,'-||')

    def _orientation(self):
        self._uc_rule()
        repeat = True
        while repeat:
            repeat = self._apply_meek_rules()
        self._nnc_rule()
        
    def run(self, data, hop: int = 0):
        bk_nd = self._bk.get_non_descendants()
        nodes = list(data.columns)
        bk_non_descendants = {node: bk_nd.get(node, set()) for node in nodes}

        # target = [self._target]
        nodes = list(data.columns)
        self.g_hat.add_vertices(nodes)
        self._visited = set()
        # self._target = [self._target]
        k = 0
        D_set = set(list(self._target))
        new_neighbors = set(list(self._target))
        while k <= hop:
            self.neighborhood_h.update(new_neighbors)
            self._update_skeleton(D_set, data)
            if self._visited == set(nodes):
                break

            D_new = {
                n
                for d in D_set
                for n in self.g_hat.get_adjacencies(d)
                if n not in bk_non_descendants[d]
            }

            new_neighbors = set()
            for d in D_set:
                for n in self.g_hat.get_adjacencies(d):
                    if n in bk_non_descendants[d]:
                        new_neighbors.add(n)
                    if n in self._visited:
                        self.neighborhood_h.add(d)

            D_set = D_new
            k+=1
        self._apply_background_knowledge()
        self._orientation()

    def _find_subset_NOC(self):
        # Add assert if no target to run the algorithm before
        subset = set(list(self._target))
        queue = list(self._target)
        while queue:
            current = queue.pop(0)
            neighbors = set(self.g_hat.get_adjacencies(current)) & self._visited
            for neighbor in neighbors:
                if ((current, neighbor) not in self.g_hat.get_directed_edges() and
                        (neighbor, current) not in self.g_hat.get_directed_edges()):
                    if neighbor not in subset:
                        subset.add(neighbor)
                        queue.append(neighbor)
        return set(subset)

    def _non_orientability_criterion(self, subset):
        for d in subset:
            set_condition_2 = []
            for a in set(self.g_hat.get_adjacencies(d)) - subset:
                # Condition 1:
                if (d, a) in self.g_hat.get_undirected_edges():
                    return False
                # Condition 2:
                if (d, a) in self.g_hat.get_uncertain_edges():
                    set_condition_2.append(a)
                    if len(set_condition_2) > 1:
                        return False
        return True

    def run_locPC_CDE(self, treatment: str, outcome: str, data) -> dict:
        nodes = list(data.columns)
        h = 0
        identifiability_CDE = False

        # Run the algorithm incrementally with hops
        while h <= len(nodes):
            self.run(data, hop=h)
            subset_noc = self._find_subset_NOC()
            if self._non_orientability_criterion(subset_noc):
                break  # Stop if non-orientability criterion met
            if (outcome, treatment) in self.g_hat.get_directed_edges() or treatment not in self.g_hat.get_adjacencies(
                    outcome):
                identifiability_CDE = True
                break  # Check on the relation between treatment and outcome
            if len(self._visited) == len(nodes):
                break  # Stop if all nodes visited
            h += 1

        if not identifiability_CDE:  # Extra identifiability check if not flagged
            for n in self.g_hat.get_adjacencies(outcome):
                if (n, outcome) not in self.g_hat.get_directed_edges() and (
                outcome, n) not in self.g_hat.get_directed_edges():
                    break
            else:
                identifiability_CDE = True  # All adjacencies oriented, CDE identifiable

        adjustment_set = [p for (p, x) in self.g_hat.get_directed_edges() if
                          x == outcome] if identifiability_CDE else None  # Compute adjustment set if identifiable

        return {
            "treatment": treatment,
            "outcome": outcome,
            "identifiability": identifiability_CDE,
            "adjustment_set": adjustment_set
        }

class LocalConstraintBasedForCausalEffect(ABC):
    def __init__(self, treatment: str, outcome: str, sparsity: float = 0.05, ci_test: Type[CiTests] = FisherZ, background_knowledge: Optional[BackgroundKnowledge] = None, twd: bool = False):
        self._treatment = treatment
        self._outcome = outcome
        self._sparsity = sparsity
        self._ci_test = ci_test
        self._bk = background_knowledge if background_knowledge is not None else BackgroundKnowledge()
        self._twd = twd

        # self.local_constraint_based_algorithm = LocalConstraintBased(target=target, sparsity=sparsity, ci_test=ci_test, background_knowledge=background_knowledge, twd=twd)

        self.local_constraint_based_algorithm = self._initialize_local_constraint_based_algorithm()

    @abstractmethod
    def _initialize_local_constraint_based_algorithm(self):
        """Initialize the local constraint-based algorithm instance."""
        pass

    @abstractmethod
    def _non_orientability_criterion(self, subset):
        pass

    @abstractmethod
    def run(self, target, data, hop: int = 0):
        pass


class LocPC_CDE(LocalConstraintBasedForCausalEffect):
    def __init__(self, treatment: str, outcome: str, sparsity: float = 0.05, ci_test: Type[CiTests] = FisherZ, background_knowledge: Optional[BackgroundKnowledge] = None, twd: bool = False):
        super().__init__(treatment, outcome, sparsity, ci_test, background_knowledge, twd)

    def _initialize_local_constraint_based_algorithm(self):
        return LocPC(target=self._outcome, sparsity=self._sparsity, ci_test=self._ci_test, background_knowledge=self._bk, twd=self._twd)

    def _find_subset_NOC(self):
        # Add assert if no target to run the algorithm before
        # handle both string and iterable outcomes
        if isinstance(self._outcome, str):
            subset = {self._outcome}
            queue = [self._outcome]
        elif isinstance(self._outcome, (list, set, tuple)):
            subset = set(self._outcome)
            queue = list(self._outcome)
        else:
            subset = {self._outcome}
            queue = [self._outcome]
        while queue:
            current = queue.pop(0)
            neighbors = set(self.local_constraint_based_algorithm.g_hat.get_adjacencies(current)) & self.local_constraint_based_algorithm._visited
            for neighbor in neighbors:
                if ((current, neighbor) not in self.local_constraint_based_algorithm.g_hat.get_directed_edges() and
                    (neighbor, current) not in self.local_constraint_based_algorithm.g_hat.get_directed_edges()):
                    if neighbor not in subset:
                        subset.add(neighbor)
                        queue.append(neighbor)
        return set(subset)
    
        
    def _non_orientability_criterion(self, subset):
        for d in subset: 
            set_condition_2 = []
            for a in set(self.local_constraint_based_algorithm.g_hat.get_adjacencies(d)) - subset:
                # Condition 1:
                if (d,a) in self.local_constraint_based_algorithm.g_hat.get_undirected_edges():
                    return False
                # Condition 2:
                if (d,a) in self.local_constraint_based_algorithm.g_hat.get_uncertain_edges():
                    set_condition_2.append(a)
                    if len(set_condition_2) > 1:
                        return False
        return True   

    def run(self, data: pd.DataFrame) -> dict:
        h = 0
        identifiability_CDE = False
        
        # Run the algorithm incrementally with hops
        nodes = list(data.columns)
        while h <= len(nodes):
            self.local_constraint_based_algorithm.run(data, hop=h)
            subset_noc = self._find_subset_NOC()
            if self._non_orientability_criterion(subset_noc):
                break # Stop if non-orientability criterion met
            if (self._outcome, self._treatment) in self.local_constraint_based_algorithm.g_hat.get_directed_edges() or self._treatment not in self.local_constraint_based_algorithm.g_hat.get_adjacencies(self._outcome):
                identifiability_CDE = True
                break # Check on the relation between treatment and outcome
            if len(self.local_constraint_based_algorithm._visited) == len(nodes):
                break # Stop if all nodes visited
            h += 1
        
        if not identifiability_CDE: # Extra identifiability check if not flagged
            for n in self.local_constraint_based_algorithm.g_hat.get_adjacencies(self._outcome):
                if (n, self._outcome) not in self.local_constraint_based_algorithm.g_hat.get_directed_edges() and (self._outcome, n) not in self.local_constraint_based_algorithm.g_hat.get_directed_edges():
                    break
            else:
                identifiability_CDE = True # All adjacencies oriented, CDE identifiable
        
        adjustment_set = [p for (p, x) in self.local_constraint_based_algorithm.g_hat.get_directed_edges() if x == self._outcome] if identifiability_CDE else None # Compute adjustment set if identifiable
        
        return {
            "treatment": self._treatment,
            "outcome": self._outcome,
            "identifiability": identifiability_CDE,
            "adjustment_set": adjustment_set
     }
