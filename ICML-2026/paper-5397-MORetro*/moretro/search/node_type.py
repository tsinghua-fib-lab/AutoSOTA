from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from itertools import product

import numpy as np

type Vector = np.ndarray
type Path = list[MolNode | RxnNode]
type PathCost = dict[tuple[float, ...], Path]


logger = logging.getLogger(__name__)


def zero_vector(length: int) -> np.ndarray:
    return np.zeros(length)


def compute_crowding_distance(costs: np.ndarray) -> np.ndarray:
    """
    Compute crowding distance for a set of cost vectors.

    Parameters
    ----------
    costs : np.ndarray
        2D array of shape (n_solutions, n_objectives)
    Returns
    -------
    np.ndarray
        Crowding distances for each solution.
    """
    n, m = costs.shape
    crowd_dists = np.zeros(n)
    for i in range(m):
        sorted_idx = np.argsort(costs[:, i])
        crowd_dists[sorted_idx[0]] = np.inf
        crowd_dists[sorted_idx[-1]] = np.inf
        if n > 2:
            obj_range = costs[sorted_idx[-1], i] - costs[sorted_idx[0], i]
            if obj_range > 0:
                for j in range(1, n - 1):
                    crowd_dists[sorted_idx[j]] += (
                        costs[sorted_idx[j + 1], i] - costs[sorted_idx[j - 1], i]
                    ) / obj_range
    return crowd_dists


def filter_pareto_with_dominated(
    self,
    all_solutions: PathCost,
    max_dominated: int = 5,
    use_crowding: bool = False,
) -> PathCost:
    """Return Pareto-optimal solutions plus up to ``max_dominated`` dominated ones."""
    if not all_solutions:
        return {}

    max_keep = max(0, int(max_dominated))
    items = list(all_solutions.items())

    if len(items) == 1:
        self.local_pareto = dict(items)
        return dict(items)

    costs = np.array([cost for cost, _ in items], dtype=float)
    cost_rounded = np.round(costs, 3)
    leq = cost_rounded[:, None, :] <= cost_rounded[None, :, :]
    lt = cost_rounded[:, None, :] < cost_rounded[None, :, :]
    dominates_matrix = np.all(leq, axis=2) & np.any(lt, axis=2)
    is_dominated = dominates_matrix.any(axis=0)

    pareto_indices = np.flatnonzero(~is_dominated)
    dominated_indices = np.flatnonzero(is_dominated)

    if max_keep and dominated_indices.size:
        if use_crowding:
            crowd_dists = compute_crowding_distance(costs[dominated_indices])
            order = np.argsort(-crowd_dists)[:max_keep]
        else:
            dominated_costs = costs[dominated_indices].sum(axis=1)
            order = np.argsort(dominated_costs)[:max_keep]
        keep_dominated = set(dominated_indices[order])
    else:
        keep_dominated = set()

    keep_pareto = set(pareto_indices)

    result: PathCost = {}
    for idx, (cost, path) in enumerate(items):
        if idx in keep_pareto or idx in keep_dominated:
            result[cost] = path

    self.local_pareto = {items[idx][0]: items[idx][1] for idx in pareto_indices}

    return result


@dataclass(frozen=False, eq=False)
class MolNode:
    """
    Class to represent a molecule node in the multi-objective retrosynthesis search graph.
    This node is an OR node, meaning it only needs one reaction (child node) to be successful

    Parameters:
    -----------
    smiles : str
        Molecule in canonical SMILES format
    heuristic_fns : list[Callable[[str], float]]
        List of heuristic functions that calculate objective values for this molecule.
    depth : int
        Depth of the node in the search tree. Root (target) molecule has depth 0.
    is_known : bool
        Whether this molecule is available in the building blocks (known starting materials).
    pareto_objectives : int
        Number of Pareto objectives.
    max_dominated_solutions : int
        Maximum number of dominated solutions to keep.
    rxn_no : Vector (np.ndarray)
        Reaction number vector containing scalar values for each weight group.
    _total_value : Vector (np.ndarray)
        Total value vector propagated from parent nodes, one value for each weight group.
    best_rxn_no : Vector (np.ndarray)
        Best cost vector for each Pareto objective, tracking minimum costs along synthesis paths.
    best_total_value : Vector (np.ndarray)
        Best total value vector propagated backward, one value for each Pareto objective.
    success : bool
        Whether this node has at least one successful synthesis path to building blocks.
    success_cost : PathCost
        Dictionary mapping cost vectors (as tuples) to the corresponding synthesis paths.
        Key: tuple of objective costs, Value: list of nodes in the path.
    is_open : bool
        Whether this node is available for expansion in the search.
    is_dominated: bool
        Whether this node is currently dominated by any point in the Pareto front.
    zero_bound : bool
        Whether to use zero lower bounds for known molecules. If True, known molecules
        get zero cost estimates.
    is_target : bool
        Whether this is the target molecule.

    Attributes computed in __post_init__:
    ------------------------------------
    h_length : int
        Length of heuristic functions list.
    value_estimates : Vector (np.ndarray)
        Heuristic-based estimates for each objective.
    success_cost_estimate : Vector (np.ndarray)
        Cost estimate for successful synthesis (only set for known molecules).
    local_pareto : PathCost
        Local Pareto front of solutions.
    successor_cost_already_checked : set
        Set of costs that have already been checked for successors.
    """

    smiles: str = field(compare=True)
    heuristic_fns: list[Callable[[str], float]]
    depth: int
    is_known: bool
    pareto_objectives: int
    max_dominated_solutions: int
    rxn_no: Vector = field(default_factory=lambda: np.array([]))
    _total_value: Vector = field(default_factory=lambda: np.array([]))
    best_rxn_no: Vector = field(
        default_factory=lambda: np.array([])
    )  # for domination checking only
    best_total_value: Vector = field(
        default_factory=lambda: np.array([])
    )  # for domination checking only
    success: bool = False
    success_cost: PathCost = field(default_factory=dict)
    is_open: bool = True
    is_dominated: bool = False
    zero_bound: bool = True
    is_target: bool = False  # whether this is the target molecule

    def __post_init__(self) -> None:
        self.h_length: int = len(self.heuristic_fns)
        self.value_estimates = self._calculate_heuristics()
        self.local_pareto: PathCost = dict()
        self.successor_cost_already_checked = set()

        if self.is_known:  # initiate the objectives
            self.is_open = False
            if self.zero_bound:
                self.success_cost_estimate = zero_vector(self.pareto_objectives)
            else:
                self.success_cost_estimate = self.value_estimates[
                    : self.pareto_objectives
                ]

    def _calculate_heuristics(self) -> np.ndarray:
        objectives = []
        for heuristic in self.heuristic_fns:
            objectives.append(heuristic(self.smiles))
        return np.array(objectives)

    def objectives_to_scalar(self, weights: np.ndarray) -> np.ndarray:
        """
        Convert the objectives to a scalar using current weights and initialize rxn_no.

        Parameters
        ----------
        weights : np.ndarray
            Weight matrix for scalarization.

        Returns
        -------
        np.ndarray
            Array of scalar reaction numbers, one for each weight group.
        """
        rxn_no = []
        for weight in weights:
            rxn_no.append(np.dot(np.array(self.value_estimates), weight))
        return np.array(rxn_no)

    def uppropagate(
        self,
        children: list[RxnNode],
        weights: np.ndarray,
        child_new_success: bool,
    ) -> tuple[bool, bool]:
        """
        Propagate costs upward from reaction children (OR logic).

        Parameters
        ----------
        children : list[RxnNode]
            Reaction node children (synthesis routes).
        weights : np.ndarray
            Weight matrix for scalarization.
        child_new_success : bool
            Whether any child has new success.
        Returns
        -------
        tuple[bool, bool]
            First bool: True if node attributes were modified.
            Second bool: True if there was new success.
        """
        new_success_cost: PathCost = dict()
        success = False
        if self.is_known:  # known building block
            no_weights, _ = weights.shape
            if self.zero_bound:
                new_rxn_no = np.zeros(no_weights)
            else:
                new_rxn_no = self.objectives_to_scalar(weights)
            success = True
            new_success_cost = {tuple(self.success_cost_estimate): [self]}
            best_rxn_no = np.zeros(
                self.pareto_objectives
            )  # should have shape of objectives
        elif self.is_open:  # tip node of tree which is not a building block
            new_rxn_no = self.objectives_to_scalar(weights)
            if self.zero_bound:
                best_rxn_no = np.zeros(self.pareto_objectives)
            else:
                best_rxn_no = np.array(self.value_estimates)[: self.pareto_objectives]
        elif len(children) > 0:  # interior node with children
            children_rxn_no = np.array(
                [child.rxn_no for child in children]
            )  # shape: (n_children, n_weights)
            new_rxn_no = np.min(children_rxn_no, axis=0)  # shape: (n_weights,)
            best_rxn_no = np.array(
                [child.best_rxn_no for child in children]
            )  # shape: (n_children, n_objectives)
            best_rxn_no = np.min(best_rxn_no, axis=0)
            success = any(child.success for child in children)
            if success:
                new_success_cost = self.track_success_cost(children)
        else:  # no valid expansion
            new_rxn_no = np.full(weights.shape[0], np.inf)
            best_rxn_no = np.full(self.pareto_objectives, np.inf)

        if (
            not np.array_equal(self.rxn_no, new_rxn_no)
            or not np.array_equal(self.best_rxn_no, best_rxn_no)
            or new_success_cost
            or self.success != success
        ):  # if any of the values changed, update the node and return bool True
            self.rxn_no = new_rxn_no
            self.best_rxn_no = best_rxn_no
            self.success = success
            # Replace entire success_cost with filtered solutions (prevents unbounded growth)
            if new_success_cost:
                self.success_cost = new_success_cost
                child_new_success = True
            else:
                child_new_success = False
            return (True, child_new_success)
        return (False, False)

    def downpropagate(self, parents: list[RxnNode]) -> bool:
        """
        Propagate total values downward from parent reactions.

        Parameters
        ----------
        parents : list[RxnNode]
            Parent reaction nodes producing this molecule.

        Returns
        -------
        bool
            True if total_value was updated.
        """
        if len(parents) == 0:  # product node with no parents
            new_total_value = self.rxn_no
            best_total_value = self.best_rxn_no
        else:
            new_total_value = np.min(
                np.array([p.total_value for p in parents]), axis=0
            )  # the total value is the minimum of the parents' total values
            best_total_value = np.min(
                np.array([p.best_total_value for p in parents]), axis=0
            )
        if not np.array_equal(self.total_value, new_total_value) or not np.array_equal(
            self.best_total_value, best_total_value
        ):
            self.total_value = new_total_value
            self.best_total_value = best_total_value
            return True
        return False

    def track_success_cost(self, children: list[RxnNode]) -> PathCost:
        """
        Track successful synthesis paths from child reactions.
        Maintains local Pareto front + top N dominated solutions.
        Returns the complete filtered solution set (Pareto + top N dominated).

        Parameters
        ----------
        children : list[RxnNode]
            Child reaction nodes with successful paths.
        Returns
        -------
        PathCost
            Complete filtered solution dictionary. Empty if nothing changed.
        """
        # Collect all candidate solutions from children (including current ones)
        all_candidate_solutions = self.success_cost.copy()

        # Collect all possible costs and successors from children
        all_costs_successors = {
            cost: successor
            for child in children
            for cost, successor in child.success_cost.items()
            if successor
            and (
                cost not in self.successor_cost_already_checked
                or cost in self.local_pareto.keys()
            )
        }

        # add all successors to already checked set
        for cost in all_costs_successors.keys():
            self.successor_cost_already_checked.add(cost)

        # Group by reaction SMILES to find paths for each unique reaction
        reaction_groups = {}
        for cost, successor in all_costs_successors.items():
            if successor and hasattr(successor[-1], "smiles"):
                rxn_smiles = successor[-1].smiles if len(successor) > 0 else None
                # sort reactants in reaction smiles according to len
                if rxn_smiles:
                    reactants_part, products_part = rxn_smiles.split(">>")
                    reactants = sorted(reactants_part.split("."), key=len)
                    rxn_smiles = ".".join(reactants) + ">>" + products_part

                reactant_smiles = tuple(
                    sorted(
                        [
                            node.smiles
                            for node in successor
                            if isinstance(node, MolNode)
                        ],
                        key=len,
                    )
                )
                group_key = (rxn_smiles, reactant_smiles)
                if group_key not in reaction_groups:
                    reaction_groups[group_key] = []
                reaction_groups[group_key].append((cost, successor))

        # For each reaction group, add the path with lowest total cost to candidates
        for _, cost_successor_pairs in reaction_groups.items():
            if len(cost_successor_pairs) > 1:
                min_pair = min(
                    cost_successor_pairs,
                    key=lambda x: (np.round(sum(x[0]), 3), *x[0]),
                )
                cost, successor = min_pair
            else:
                cost, successor = cost_successor_pairs[0]

            new_path = successor + [self]
            all_candidate_solutions[cost] = new_path

        # Apply Pareto filtering (skip for target molecule to keep all solutions)
        filtered_solutions = filter_pareto_with_dominated(
            self,
            all_candidate_solutions,
            50 if self.is_target else self.max_dominated_solutions,
            use_crowding=self.is_target,
        )

        # Return complete filtered set if anything changed, empty dict otherwise
        if filtered_solutions != self.success_cost:
            return filtered_solutions

        return {}

    @property
    def total_value(self) -> Vector:
        return self._total_value

    @total_value.setter
    def total_value(self, value) -> None:
        if isinstance(value, list):
            value = np.array(value)
        condition_check = not self.is_target and not self.success
        if (
            value.size > 0 and np.all(value == 0) and condition_check
        ):  # Check if all values are zero
            logger.warning(
                f"Total value of node {self.smiles} is all zeros, this is not expected."
            )
        self._total_value = value

    def __hash__(self) -> int:
        return id(self)

    def __repr__(self) -> str:
        return f"MolNode(smiles='{self.smiles}', depth={self.depth}, success={self.success})"


@dataclass(frozen=False, eq=False)
class RxnNode:
    """
    Reaction node in multi-objective retrosynthesis search graph (AND node).

    Parameters
    ----------
    smiles : str
        Reaction in SMILES format.
    template : str
        Reaction template in SMARTS format.
    reagents : list[str]
        Required reagents/catalysts.
    temp : float
        Reaction temperature in Kelvin.
    depth : int
        Node depth in search tree.
    cost : Vector
        Multi-dimensional reaction cost.
    weight_length : int
        Number of weight samples.
    pareto_objectives : int
        Number of Pareto objectives.
    max_dominated_solutions : int
        Maximum number of dominated solutions to keep.
    total_value : Vector (np.ndarray)
        Total value vector, one for each group of weights (default: empty array).
    rxn_no : Vector (np.ndarray)
        Reaction number vector, one for each group of weights (default: empty array).
    best_rxn_no : Vector (np.ndarray)
        Best cost vector for each Pareto objective, tracking minimum costs along synthesis paths.
    best_total_value : Vector (np.ndarray)
        Best total value vector propagated backward, one value for each Pareto objective.
    success_cost : PathCost
        Dictionary mapping cost vectors to synthesis paths (default: empty dict).
    success : bool
        Whether this reaction has successful synthesis paths (default: False).

    Attributes computed in __post_init__:
    ------------------------------------
    true_cost : Vector (np.ndarray)
        Copy of original cost before adding delta offset.
    _delta_offset : float
        Small offset added to costs to ensure uniqueness.
    """

    smiles: str
    template: str
    reagents: list[str]
    temp: float
    depth: int
    cost: Vector  # Actual cost of reaction in n dimensions
    weight_length: int
    pareto_objectives: int
    max_dominated_solutions: int
    total_value: Vector = field(
        default_factory=lambda: np.array([])
    )  # one for each group of weight
    rxn_no: Vector = field(
        default_factory=lambda: np.array([])
    )  # one for each group of weights
    best_rxn_no: Vector = field(
        default_factory=lambda: np.array([])
    )  # for domination checking only
    best_total_value: Vector = field(
        default_factory=lambda: np.array([])
    )  # for domination checking only
    success_cost: PathCost = field(default_factory=dict)
    success: bool = False
    _delta_offset: float = field(default=0.0, init=False)

    def __post_init__(self) -> None:
        if not self.cost.any():
            logger.error("Reaction cost cannot be empty!")
            raise ValueError("Reaction cost must be provided")

        reaction_hash = hash((self.smiles, tuple(self.reagents)))
        normalized_hash = (abs(reaction_hash) % 100) + 1
        self._delta_offset = normalized_hash * 1e-15
        self.true_cost = self.cost.copy()
        self.cost = np.array(
            [c + self._delta_offset for c in self.cost]
        )  # ensure unique costs
        self.rxn_no = zero_vector(self.weight_length)
        self.best_rxn_no = zero_vector(self.pareto_objectives)
        self.total_value = zero_vector(self.weight_length)

    def uppropagate(
        self, children: list[MolNode], weights: np.ndarray, child_new_success: bool
    ) -> tuple[bool, bool]:
        """
        Propagate costs upward from molecule children (AND logic).

        Parameters
        ----------
        children : list[MolNode]
            Molecule node children (reactants).
        weights : np.ndarray
            Weight matrix for scalarization.
        child_new_success : bool
            Whether any child has new success.

        Returns
        -------
        tuple[bool, bool]
            First bool: True if node attributes were modified.
            Second bool: True if there was new success.

        """
        success = False
        if not children:
            logger.error(
                "Reaction node must have at least one child to propagate from!"
            )
            raise ValueError("No children provided for RxnNode")
        success = all(child.success for child in children)
        new_rxn_no = np.zeros_like(self.rxn_no)
        new_best_rxn_no = np.zeros(self.pareto_objectives)
        new_success_cost = dict()

        # Sum up costs from all children
        for child in children:
            assert len(child.rxn_no) != 0, "Rxn_no for MolNode should not be empty"
            new_rxn_no += np.array(child.rxn_no)
            new_best_rxn_no += np.array(child.best_rxn_no)
        # add reaction cost with each weight combination
        rxn_cost = weights @ self.true_cost
        new_rxn_no += rxn_cost
        best_cost = np.array(self.true_cost)[: self.pareto_objectives]
        new_best_rxn_no += np.array(best_cost)

        if success:
            new_success_cost = self.track_success_cost(children)

        if (
            not np.array_equal(self.rxn_no, new_rxn_no)
            or not np.array_equal(self.best_rxn_no, new_best_rxn_no)
            or new_success_cost
            or self.success != success
        ):
            self.rxn_no = new_rxn_no
            self.best_rxn_no = new_best_rxn_no
            self.success = success
            # Replace entire success_cost with filtered solutions (prevents unbounded growth)
            if new_success_cost:
                self.success_cost = new_success_cost
                child_new_success = True
            else:
                child_new_success = False
            return (True, child_new_success)
        return (False, False)

    def downpropagate(self, parent: MolNode) -> bool:
        """
        Propagate total values downward from parent molecule.

        Parameters
        ----------
        parent : MolNode
            Parent molecule node this reaction produces.

        Returns
        -------
        bool
            True if total_value was updated.
        """
        # check if parent molecule is infeasible
        if np.all(parent.rxn_no == float("inf")):
            new_total_value = np.full(len(self.rxn_no), np.inf)
            new_best_total_value = np.full(self.pareto_objectives, np.inf)
        else:
            new_total_value = self.rxn_no - parent.rxn_no + parent.total_value
            new_best_total_value = (
                self.best_rxn_no - parent.best_rxn_no + parent.best_total_value
            )
        if not np.array_equal(self.total_value, new_total_value) or not np.array_equal(
            self.best_total_value, new_best_total_value
        ):
            self.total_value = new_total_value
            self.best_total_value = new_best_total_value
            return True
        return False

    def track_success_cost(self, children: list[MolNode]) -> PathCost:
        """
        Track synthesis paths from child molecules (AND logic).
        Maintains local Pareto front + top N dominated solutions.
        Returns the complete filtered solution set (Pareto + top N dominated).

        Parameters
        ----------
        children : list[MolNode]
            Child molecule nodes (reactants).

        Returns
        -------
        PathCost
            Complete filtered solution dictionary. Empty if nothing changed.
        """
        # Start with empty candidate solutions
        all_candidate_solutions = {}

        children_costs = [list(child.success_cost.keys()) for child in children]
        for cost_combination in product(*children_costs):
            successor_nodes = []

            # Collect all successor nodes from children
            for i, cost in enumerate(cost_combination):
                child_successor = children[i].success_cost[cost]

                if isinstance(child_successor, tuple):
                    child_nodes, _ = child_successor
                    successor_nodes.extend(child_nodes)
                else:
                    successor_nodes.extend(child_successor)

            # Calculate total cost: sum of children costs + reaction cost
            children_total_cost = np.sum(np.array(cost_combination), axis=0)
            reaction_total_cost = (
                children_total_cost + np.array(self.cost)[: self.pareto_objectives]
            )
            reaction_total_cost = tuple(reaction_total_cost.tolist())

            # Add this reaction to the path
            successor_nodes_with_rxn = successor_nodes + [self]
            all_candidate_solutions[reaction_total_cost] = successor_nodes_with_rxn

        # Apply Pareto filtering once at the end
        filtered_solutions = filter_pareto_with_dominated(
            self, all_candidate_solutions, self.max_dominated_solutions
        )

        # Return complete filtered set if anything changed, empty dict otherwise
        if filtered_solutions != self.success_cost:
            return filtered_solutions

        return {}

    def __hash__(self) -> int:
        return id(self)

    def __repr__(self) -> str:
        return f"RxnNode(smiles='{self.smiles}', depth={self.depth}, success={self.success})"
