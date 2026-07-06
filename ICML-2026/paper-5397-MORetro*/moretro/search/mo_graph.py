import logging
from collections.abc import Callable
from typing import cast

import gin
import numpy as np
from rdkit import Chem
from scipy.stats import qmc

from moretro.inference.and_or_graph import AndOrGraph
from moretro.search.node_type import MolNode, RxnNode
from moretro.utils.bo_weight_selector import BOWeightSelector
from moretro.utils.typing_hints import (
    CostVector,
    MolNodeAndWeights,
    MolsAndWeights,
    NewSolution,
    Nodes,
    ParetoCost,
    SolutionCost,
    WeightIndices,
)

# Set up module logger
logger = logging.getLogger(__name__)


@gin.configurable(denylist=["target", "building_blocks", "heuristic_fns"])
class MOGraph:
    """
    Multi-objective retrosynthesis search graph.

    Attributes
    ----------
    target : str
        Canonicalized target molecule SMILES.
    building_blocks : set[str]
        Set of available starting materials.
    heuristic_fns : list[Callable[[str], float]]
        List of objective functions.
    pareto_objectives : int
        Number of Pareto objectives.
    max_dominated_solutions : int
        Maximum number of dominated solutions to keep.
    zero_bound : bool
        Whether to use zero-bound heuristics.
    weight_samples: int
        Total number of weight vectors to sample (used for "sobol" or "dirichlet").
    no_weights: int
        Number of active weights at a time.
    weight_initial: str
        Weight initialization strategy ("sobol", "grid", "constant").
    include_extreme: bool
        Whether to include extreme points in weight initialization.
    weight_update_strategy: str
        Strategy for updating weights ("queue" or "bo").
    """

    def __init__(
        self,
        target: str,
        building_blocks: set[str],
        heuristic_fns: list[Callable[[str], float]],
        pareto_objectives: int,
        max_dominated_solutions: int,
        zero_bound: bool = True,
        weight_samples: int = 64,
        no_weights: int = 5,
        weight_initial: str = "sobol",
        include_extreme: bool = False,
        weight_update_strategy: str = "queue",
    ):
        self.target = Chem.CanonSmiles(target)
        self.building_blocks = building_blocks
        self.heuristic_fns = heuristic_fns
        self.pareto_objectives = pareto_objectives
        self.max_dominated_solutions = max_dominated_solutions
        self.zero_bound = zero_bound
        self.open_nodes: set[MolNode] = set()
        self.weight_samples = weight_samples
        self.no_weights = no_weights
        self.weight_initial = weight_initial
        self.include_extreme = include_extreme
        self.weight_update_strategy = weight_update_strategy
        self.weights_open: np.ndarray = np.zeros(
            (self.weight_samples, len(self.heuristic_fns))
        )
        self.weight_history: list[list[float]] = []
        self.solution_cost: SolutionCost = {}
        self.pareto_front: ParetoCost = {}
        self.pareto_front_costs: np.ndarray = np.empty((0, pareto_objectives))
        self.mol_to_node: dict[str, MolNode] = {}

        # Assignment check:
        if self.pareto_objectives > len(self.heuristic_fns):
            logger.error(
                f"pareto_objectives ({self.pareto_objectives}) cannot be greater than the number of heuristic functions ({len(self.heuristic_fns)})."
            )
            raise ValueError("Invalid pareto_objectives configuration.")
        # BO bookkeeping
        self.bo_selector: BOWeightSelector | None = None
        if self.weight_update_strategy == "bo":
            self.bo_selector = BOWeightSelector(len(heuristic_fns), seed=42)

        # Create a dedicated random number generator for reproducibility
        self.rng = np.random.default_rng(seed=42)

        target_known = self.target in self.building_blocks
        if target_known:
            logger.info(f"Target {self.target} is already in the building blocks.")

        self.target_node = MolNode(
            smiles=self.target,
            heuristic_fns=heuristic_fns,
            depth=0,
            is_known=target_known,
            pareto_objectives=self.pareto_objectives,
            max_dominated_solutions=self.max_dominated_solutions,
            is_open=not target_known,
            zero_bound=self.zero_bound,
            is_target=True,  # Mark this node as the target
        )

        self.graph = AndOrGraph()
        if self.target_node.is_open:
            self.graph.add_node(self.target_node, node_type="target")
            # add open target node no_weights time to set in list
            self.open_nodes.add(self.target_node)
            self.mol_to_node[self.target] = self.target_node
            self.target_node.total_value = np.zeros(self.no_weights)

        # initialize weights depending on strategy
        if self.bo_selector:
            initial_weights_open = self.weight_initialization(
                n_obj=len(heuristic_fns), init_type="grid", include_extreme=False
            )
            bo_weights_open = self.weight_initialization(
                n_obj=len(heuristic_fns),
                init_type=self.weight_initial,
                include_extreme=include_extreme,
                second_sample=True,  # flag to generate different samples for BO
            )
            self.weights_open = np.vstack([initial_weights_open, bo_weights_open])
        else:
            self.weights_open = self.weight_initialization(
                n_obj=len(heuristic_fns),
                init_type=weight_initial,
                include_extreme=include_extreme,
            )
            self.rng.shuffle(self.weights_open)
        # pop no_weights from weights_open into self.weights
        self.weights = self.weights_open[
            : self.no_weights, :
        ]  # dimensions: (no_weights, n_obj)
        self.weights_open = self.weights_open[self.no_weights :]

        if self.bo_selector:
            self.bo_selector.add_batch(self.weights)

    def expand_graph(
        self,
        predictions: dict[MolNode, list[dict]],
        expanded_nodes: MolNodeAndWeights,
    ) -> MolsAndWeights | None:
        """
        Expand graph with new reactions and molecules.

        Parameters
        ----------
        predictions : dict[MolNode, list[dict]]
            Synthesis predictions for each node. Each prediction contains
            reactants, reagents, temperature, rxn_smiles, template, and costs.
        expanded_nodes : set[tuple[MolNode, WeightIndices]]
            Nodes to expand with their weight group indices.

        Returns
        -------
        MolsAndWeights | None
            Set of new nodes and their weight indices.
        """
        new_nodes: list[tuple[Nodes, WeightIndices]] = []
        list_expanded_nodes = sorted(expanded_nodes, key=lambda x: x[0].smiles)
        # check for reactants that appear more than once but are not in the current mol_to_node
        reactants: list[str] = []
        for node, _ in list_expanded_nodes:
            for pred in predictions[node]:
                react = pred["reactants"]
                react = [Chem.CanonSmiles(r) for r in react]
                reactants.extend(react)
        multiple_reactants = {
            r for r in reactants if reactants.count(r) > 1 and r not in self.mol_to_node
        }

        for node, weight_indices in list_expanded_nodes:
            self.open_nodes.remove(node)
            node.is_open = False
            for pred in predictions[node]:
                reactants = pred["reactants"]
                reactants = [Chem.CanonSmiles(reactant) for reactant in reactants]
                reagents = pred["reagents"]
                temp = pred["temperature"]
                rxn_smiles = pred["rxn_smiles"]
                template = pred["template"]
                costs = np.array(
                    pred["costs"]
                )  # * Costs should be calculated outside this class using ML surrogates
                rxn_node = RxnNode(
                    smiles=rxn_smiles,
                    template=template,  # In SMARTS
                    reagents=reagents,
                    temp=temp,
                    depth=node.depth + 1,
                    cost=costs,
                    weight_length=len(self.weights),
                    pareto_objectives=self.pareto_objectives,
                    max_dominated_solutions=self.max_dominated_solutions,
                )
                # Check for presence of cycles in the graph
                cycle_exists = False
                for reactant in reactants:
                    if (
                        reactant in self.mol_to_node
                        and reactant in self.graph.get_ancestors(node)
                    ):
                        cycle_exists = True
                        break

                if cycle_exists:
                    continue

                self.graph.add_node(rxn_node, node_type="reaction")
                self.graph.add_edge(node, rxn_node)

                for reactant in reactants:
                    if reactant in self.mol_to_node:
                        reactant_node = self.mol_to_node[reactant]
                        # remove it's dominated status as it will have to be reevaluated
                        reactant_node.is_dominated = False
                        reactant_node.depth = max(reactant_node.depth, node.depth + 2)
                        if reactant in multiple_reactants:
                            new_nodes.append((reactant_node, weight_indices))
                    else:
                        reactant_known = reactant in self.building_blocks
                        reactant_node = MolNode(
                            smiles=reactant,
                            heuristic_fns=self.heuristic_fns,
                            depth=node.depth + 2,
                            is_known=reactant_known,
                            pareto_objectives=self.pareto_objectives,
                            max_dominated_solutions=self.max_dominated_solutions,
                            zero_bound=self.zero_bound,
                        )
                        self.mol_to_node[reactant] = reactant_node
                        self.graph.add_node(reactant_node, node_type="molecule")
                        new_nodes.append((reactant_node, weight_indices))
                    self.graph.add_edge(rxn_node, reactant_node)

                new_nodes.append((rxn_node, weight_indices))
        return set(new_nodes)

    def update_values(self, nodes: MolsAndWeights) -> bool:
        """
        Update node values and Pareto front.

        Parameters
        ----------
        nodes : MolsAndWeights
            Nodes and weight indices to update.

        Returns
        -------
        bool
            True if Pareto front was updated.
        """
        nodes_to_update = nodes.copy()
        updated_nodes, new_solutions = self.uppropagation(nodes_to_update)
        nodes_to_update.update(updated_nodes)
        downprop_updated, _ = self.downpropagation(nodes_to_update)
        pareto_updated = self.update_solution_and_pareto(new_solutions)

        return pareto_updated

    def uppropagation(
        self,
        nodes_and_weights: MolsAndWeights,
    ) -> tuple[MolsAndWeights, NewSolution]:
        """
        Propagate values upward from leaves to root.

        Parameters
        ----------
        nodes_and_weights : MolsAndWeights
            Starting nodes and their weight indices.

        Returns
        -------
        tuple[MolsAndWeights, NewSolution]
            Tuple containing:
            - MolsAndWeights: Set of updated nodes and weight indices
            - NewSolution: Dictionary mapping cost vectors to weight indices for new solutions
        """
        updated_nodes: MolsAndWeights = set()
        new_solutions: dict[CostVector, WeightIndices] = {}

        # Group nodes by weight indices
        weight_groups: dict[WeightIndices, list[Nodes]] = {}
        for node, weight_indices in nodes_and_weights:
            if weight_indices not in weight_groups:
                weight_groups[weight_indices] = []
            weight_groups[weight_indices].append(node)

        # sort dict so the group with the lowest rxn node depth goes first
        def group_sort_key(item):
            nodes = item[1]
            rxn_depths = [n.depth for n in nodes if isinstance(n, RxnNode)]
            if rxn_depths:
                return min(rxn_depths)
            return min((n.depth for n in nodes), default=float("inf"))

        weight_groups = dict(sorted(weight_groups.items(), key=group_sort_key))
        processed_weight_indices = set()
        for weight_indices, nodes in weight_groups.items():
            old_solutions = set(self.target_node.success_cost.keys())

            copy_weight_groups = weight_groups.copy()
            # * Do not accidentally uppropagate into rxns that are chosen by other weight groups
            copy_weight_groups.pop(weight_indices)
            # Get all reaction nodes from other weight groups
            rxn_nodes: set[RxnNode] = set()
            for other_indices, other_nodes in copy_weight_groups.items():
                if other_indices not in processed_weight_indices:
                    rxn_nodes.update(
                        node for node in other_nodes if isinstance(node, RxnNode)
                    )
            # Sort nodes by depth (deepest first) and then by SMILES length within each depth level
            queue = [(-node.depth, id(node), node, False) for node in nodes]
            # sort the queue by depth and smiles length
            queue.sort(key=lambda x: (x[0], len(x[2].smiles)))
            processed = set()

            while queue:
                _, node_id, node, child_new_success = queue.pop(0)
                processed.add(node_id)

                if isinstance(node, RxnNode):
                    children = cast(list[MolNode], list(self.graph.successors(node)))
                    parents_update, child_new_success = node.uppropagate(
                        children, self.weights, child_new_success
                    )
                elif isinstance(node, MolNode):
                    children = cast(list[RxnNode], list(self.graph.successors(node)))
                    parents_update, child_new_success = node.uppropagate(
                        children,
                        self.weights,
                        child_new_success,
                    )
                else:
                    raise TypeError(
                        f"Node {node} is not of type RxnNode or MolNode, but {type(node)}"
                    )

                if parents_update:
                    updated_nodes.add((node, weight_indices))
                    for parent in list(self.graph.predecessors(node)):
                        parent = cast(RxnNode | MolNode, parent)
                        if (
                            parent not in [q[2] for q in queue]
                            and parent not in rxn_nodes
                        ):
                            queue.append(
                                (-parent.depth, id(parent), parent, child_new_success)
                            )

            current_solution = set(self.target_node.success_cost.keys())
            new_costs = current_solution.difference(old_solutions)
            if new_costs:
                new_solutions.update({cost: weight_indices for cost in new_costs})

            processed_weight_indices.add(weight_indices)

        # Filter new_solutions to keep only costs still present after all weight groups have been processed
        new_solutions = {
            cost: weight_indices
            for cost, weight_indices in new_solutions.items()
            if cost in self.target_node.success_cost
        }

        return (updated_nodes, new_solutions)

    def downpropagation(self, nodes: MolsAndWeights) -> tuple[set[Nodes], set[Nodes]]:
        """
        Propagate values downward from root to leaves.

        Parameters
        ----------
        nodes : MolsAndWeights
            Starting nodes and weight indices.

        Returns
        -------
        tuple[set[Nodes], set[Nodes]]
            Tuple of (updated_nodes, processed_nodes).
        """
        # Use list to maintain depth order
        queue = sorted([node[0] for node in nodes], key=lambda x: x.depth)
        updated_nodes = set()
        processed = set()  # Track processed nodes to avoid duplicates

        while queue:
            node = queue.pop(0)
            processed.add(node)

            if isinstance(node, RxnNode):
                parents_rxn = cast(list[MolNode], list(self.graph.predecessors(node)))
                children_update = node.downpropagate(parents_rxn[0])
            elif isinstance(node, MolNode):
                parents_mol = cast(list[RxnNode], list(self.graph.predecessors(node)))
                children_update = node.downpropagate(parents_mol)
            else:
                raise TypeError(
                    f"Node {node} is not of type RxnNode or MolNode, but {type(node)}"
                )

            if children_update:
                updated_nodes.add(node)
                for child in self.graph.successors(node):
                    if child not in queue:
                        queue.append(child)

        return updated_nodes, processed

    def update_solution_and_pareto(self, new_solutions: NewSolution) -> bool:
        """
        Update solution costs and Pareto front from new solutions.

        Parameters
        ----------
        new_solutions : NewSolution
            Dictionary mapping cost vectors to weight indices.

        Returns
        -------
        bool
            True if Pareto front was updated.
        """
        old_pareto = set(self.pareto_front.keys())
        old_pareto_costs = (
            np.array(list(self.pareto_front.keys()))
            if self.pareto_front
            else np.empty((0, self.pareto_objectives))
        )
        new_pareto = set(self.target_node.local_pareto.keys())
        pareto_points_to_remove = old_pareto - new_pareto
        new_pareto_points = new_pareto - old_pareto

        # Sync solution_cost with current target_node.success_cost
        # Remove solutions that are no longer in target_node (filtered out)
        current_target_costs = set(self.target_node.success_cost.keys())
        success_costs_to_remove = set(self.solution_cost.keys()) - current_target_costs
        success_costs_to_remove.update(pareto_points_to_remove)
        for cost in success_costs_to_remove:
            self.solution_cost.pop(cost, None)
            self.pareto_front.pop(cost, None)

        # Keep track of which local weight indices contributed new solutions
        contributing_local_weight_indices: set[int] = set()

        for cost_vector, weight_indices in new_solutions.items():
            # Get the path information from target node's success_cost
            path_nodes = self.target_node.success_cost[cost_vector]

            # Transform local weight indices to global indices
            global_weight_indices = tuple(
                len(self.weight_history) + i for i in weight_indices
            )

            # Store the new solution with path and global indices
            self.solution_cost[cost_vector] = (path_nodes, global_weight_indices)

            # Check if this should be added to Pareto front
            if cost_vector in new_pareto:
                # Record contributing local indices
                for i in weight_indices:
                    contributing_local_weight_indices.add(i)

                weights = [
                    self.weights[i].tolist()
                    for i in weight_indices
                    if i < len(self.weights)
                ]
                self.pareto_front[cost_vector] = weights

        # Now compute delta HV (hypervolume improvement)
        new_pareto_costs = (
            np.array(list(self.pareto_front.keys()))
            if self.pareto_front
            else np.empty((0, self.pareto_objectives))
        )
        if self.bo_selector and contributing_local_weight_indices:
            self.bo_selector.process_pareto_update(
                old_pareto_costs,
                new_pareto_costs,
                contributing_local_weight_indices,
            )

        if new_pareto_points or pareto_points_to_remove:
            pareto_front_costs = np.array(list(self.pareto_front.keys()))
            self.pareto_front_costs = np.round(pareto_front_costs, decimals=3)
            logger.info(
                f"Pareto front updated: {len(new_pareto_points)} points added, {len(pareto_points_to_remove)} points removed. Total: {len(self.pareto_front)} points."
            )

        return bool(new_pareto_points or pareto_points_to_remove)

    def reinitialize_graph(self) -> None:
        """
        Spawn new weights and reinitialize all node values.
        """
        if self.weight_initial == "constant":
            logger.warning(
                "Weight initialization is set to 'constant'. Reinitialization will not change weights."
            )
            return
        self.update_weights()
        open_nodes = [
            node for node in self.mol_to_node.values() if node.is_open or node.is_known
        ]
        if not open_nodes:
            logger.warning("No open nodes found in the graph for reinitialization.")
        else:
            open_nodes_with_weights = [(open_node, (0,)) for open_node in open_nodes]
            updated_nodes: MolsAndWeights = set(open_nodes_with_weights)
            new_nodes, _ = self.uppropagation(updated_nodes)
            updated_nodes.update(new_nodes)
            downprop_updated, downprop_processed = self.downpropagation(updated_nodes)
            logger.info("Reinitialization of weights completed")
            logger.info(
                f"Nodes processed vs updated during downprop: {len(downprop_processed)} vs {len(downprop_updated)}"
            )

    def update_weights(self) -> None:
        """
        Update active weights from weight pool.

        Uses BO surrogate to select weights if we've collected some data and strategy is 'bo'.
        """
        # record the current weights into history (they have already been sampled)
        self.weight_history.extend(self.weights.tolist())

        # If there are fewer remaining weights than no_weights, just take what's available
        if len(self.weights_open) <= self.no_weights:
            self.weights = self.weights_open.copy()
            self.weights_open = np.empty((0, self.weights.shape[1]))
            if self.bo_selector:
                self.bo_selector.add_batch(self.weights)
            return

        if self.bo_selector:
            # Attempt BO selection
            try:
                selected_weights, remaining_weights = (
                    self.bo_selector.select_next_weights(
                        self.weights_open, k=self.no_weights
                    )
                )
                self.weights = selected_weights
                self.weights_open = remaining_weights
                self.bo_selector.add_batch(self.weights)
                return
            except Exception as e:
                logger.warning(
                    f"BO weight selection failed ({e}), falling back to simple pop."
                )

        # Default / Fallback strategy: simple pop (queue)
        self.weights = self.weights_open[: self.no_weights, :]
        self.weights_open = self.weights_open[self.no_weights :]
        if self.bo_selector:
            self.bo_selector.add_batch(self.weights)

    def weight_initialization(
        self,
        n_obj: int,
        init_type: str,
        include_extreme: bool,
        second_sample: bool = False,
    ) -> np.ndarray:
        """
        Initialize set of weights for linear combination of objectives.

        Parameters
        ----------
        n_obj : int
            Number of objectives.
        init_type : str
            Type of weight initialization to use. Options are "sobol" or "dirichlet".
        include_extreme : bool
            Whether to include extreme points in the weight vectors.
        second_sample: bool
            Whether this is a second sample for BO (used to generate different samples).

        Returns
        -------
        np.ndarray
            Array of weight vectors with shape (weight_samples, n_obj).
        """
        if init_type == "sobol":
            return self._sobol_initialization(
                n_obj, self.weight_samples, include_extreme
            )
        elif init_type == "dirichlet":
            return self.rng.dirichlet(np.ones(n_obj), size=self.weight_samples)
        elif init_type == "grid":
            return self._grid_initialization(second_sample)
        elif init_type == "constant":
            return self._constant_initialization()
        else:
            raise ValueError(f"Unknown weight initialization type: {init_type}")

    def _sobol_initialization(
        self, n_obj: int, n_samples: int, include_extreme: bool
    ) -> np.ndarray:
        """
        Generate Sobol sequence weight vectors with extreme points.

        Parameters
        ----------
        n_obj : int
            Number of objectives.
        n_samples : int
            Total number of weight vectors.
        include_extreme : bool
            Whether to include extreme points in the weight vectors.

        Returns
        -------
        np.ndarray
            Sobol-based weight vectors with extreme points.
        """
        # Reserve space for extreme points if requested
        if include_extreme:
            sobol_samples_needed = n_samples - n_obj
        else:
            sobol_samples_needed = n_samples

        if sobol_samples_needed <= 0:
            logger.warning(
                f"Not enough samples ({n_samples}) for both Sobol and extreme points"
            )
            sobol_samples_needed = max(4, n_samples // 2)
        elif sobol_samples_needed % 2 != 0:
            logger.error(
                f"Requested Sobol samples ({sobol_samples_needed}) must be a power of 2."
            )
            raise ValueError("Please re-adjust the number of samples")

        sobol = qmc.Sobol(d=n_obj, scramble=True, rng=self.rng)
        logger.info(
            "Sobol Sobol sequence (scramble=True) introduces randomness which may affect reproducibility."
        )
        m = int(np.log2(sobol_samples_needed))
        raw_samples = sobol.random_base2(m=m)
        sobol_weights = raw_samples / raw_samples.sum(
            axis=1, keepdims=True
        )  # Normalize to sum to 1

        # Create final weights array
        if include_extreme:
            extreme_points = np.eye(n_obj)
            weights = np.vstack([sobol_weights, extreme_points])
            logger.info(
                f"Generated {sobol_samples_needed} Sobol samples + {n_obj} extreme points = {n_samples} total weight vectors"
            )
        else:
            weights = sobol_weights
            logger.info(f"Generated {sobol_samples_needed} Sobol samples.")
        return weights

    def _grid_initialization(self, second_sample: bool) -> np.ndarray:
        """
        Generate grid-based weight vectors.

        Returns
        -------
        np.ndarray
            Grid-based weight vectors.
        second_sample: bool
            Whether this is a second sample for BO (used to generate different samples with finer grid).
        """
        if len(self.heuristic_fns) <= 3 or (self.bo_selector and not second_sample):
            # Generate grid points with a step size of 0.25 (0, 0.25, 0.5, 0.75, 1) that sum to 1
            steps = [0.0, 0.25, 0.5, 0.75, 1.0]
        elif second_sample:
            steps = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        else:
            # Coarser steps for higher dim
            steps = [0.0, 1 / 3, 2 / 3, 1.0]
        grid_points = np.array(
            np.meshgrid(*[steps] * len(self.heuristic_fns))
        ).T.reshape(-1, len(self.heuristic_fns))
        # Filter points that sum to 1
        grid_weights = grid_points[np.isclose(grid_points.sum(axis=1), 1.0)]
        # * Enhancement: this is the point to improve - what should initial samples be?
        if self.bo_selector and not second_sample:
            grid_weights = grid_weights[grid_weights[:, -1] >= 0.5]
        # spawn dummy weights s.t. len(weights) % no_weights == 0
        random_weights = self.rng.dirichlet(np.ones(len(self.heuristic_fns)), size=20)
        i = 0
        while len(grid_weights) % self.no_weights != 0:
            grid_weights = np.vstack([grid_weights, random_weights[i]])
            i += 1
        logger.info(f"Generated {len(grid_weights)} grid-based weight vectors.")
        return grid_weights

    def _constant_initialization(self) -> np.ndarray:
        """
        Generate constant weight vectors (equal weights for all objectives).

        Returns
        -------
        np.ndarray
            Constant weight vectors.
        """
        constant_weights = [0.2, 0.2, 0.2, 0.4]
        weights = np.array([constant_weights for _ in range(self.weight_samples)])
        logger.info(f"Generated {len(weights)} constant weight vectors.")
        print(f"Constant weights: {weights[0]}")
        return weights
