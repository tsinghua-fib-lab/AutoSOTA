import logging
import time
from collections import defaultdict
from collections.abc import Callable

import gin
import numpy as np
import torch  # type: ignore

from moretro.inference.retro_prediction import OneStepModel
from moretro.search.mo_graph import MOGraph
from moretro.search.node_type import MolNode
from moretro.utils.typing_hints import MolNodeAndWeights, Nodes

logger = logging.getLogger(__name__)


@gin.configurable(
    denylist=["target", "retro_model", "building_blocks", "heuristic_fns"]
)
class MOSearch:
    """
    Multi-objective search engine for retrosynthesis planning.

    Parameters
    ----------
    target : str
        Target molecule in SMILES format to synthesize.
    retro_model : OneStepModel
        Single-step retrosynthesis prediction model.
    building_blocks : set[str]
        Set of available starting materials.
    heuristic_fns : list[Callable[[str], float]]
        List of objective functions for multi-objective optimization.
    top_n : int
        Number of top reactions to consider per expansion.
    max_depth : int
        Maximum search depth (actual graph depth will be 2*max_depth).
    iteration_budget : int
        Maximum number of search iterations.
    weight_iter_budget : int
        Number of iterations before resampling weights.
    time_budget : float, default 0.0
        Maximum time budget in seconds (0.0 means no time limit).
    single_step_call_budget : int, default 0
        Maximum number of single-step model calls allowed (0 means no limit).
    max_pareto_solutions : int, default 250
        Maximum number of Pareto solutions to find before stopping search.
    stop_on_full_pareto : bool, default False
        Whether to stop search when the full Pareto front is found.
    exclude_dominated_nodes : bool, default False
        Whether to exclude nodes dominated by the current Pareto front from expansion.
    epsilon_pruning : float, default 0.0
        A positive float added to each objective of open nodes to aggressively prune nodes.
    """

    def __init__(
        self,
        target: str,
        retro_model: OneStepModel,
        building_blocks: set[str],
        heuristic_fns: list[Callable[[str], float]],
        top_n: int,
        max_depth: int,
        iteration_budget: int,
        weight_iter_budget: int,
        time_budget: float = 0.0,
        single_step_call_budget: int = 0,
        max_pareto_solutions: int = 250,
        stop_on_full_pareto: bool = False,
        exclude_dominated_nodes: bool = False,
        epsilon_pruning: float = 0.0,
    ):
        self.max_depth = 2 * max_depth
        self.retro_model = retro_model
        self.search_graph = MOGraph(
            target=target,
            building_blocks=building_blocks,
            heuristic_fns=heuristic_fns,
            pareto_objectives=gin.REQUIRED,  # type: ignore
            max_dominated_solutions=gin.REQUIRED,  # type: ignore
            weight_samples=gin.REQUIRED,  # type: ignore
            no_weights=gin.REQUIRED,  # type: ignore
            weight_initial=gin.REQUIRED,  # type: ignore
            include_extreme=gin.REQUIRED,  # type: ignore
        )
        self.top_n = top_n
        self.weight_iter_budget = weight_iter_budget
        self.iteration_budget = iteration_budget
        self.time_budget = time_budget
        self.single_step_call_budget = single_step_call_budget
        self.max_pareto_solutions = max_pareto_solutions
        self.stop_on_full_pareto = stop_on_full_pareto
        self.exclude_dominated_nodes = exclude_dominated_nodes
        self.epsilon_pruning = epsilon_pruning
        self.weights_open: list[bool] = [True] * self.search_graph.no_weights
        self.retro_expansion_count = 0

    def can_expand_retro(self, node: Nodes) -> bool:
        """
        Check if the node can be expanded in the retrosynthetic direction.

        Checks:
        1. Node is a MolNode (not a reaction node)
        2. Node has not reached maximum depth
        3. Node has not been expanded yet (is open)

        Parameters
        ----------
        node : Nodes
            The node to check for expansion eligibility.

        Returns
        -------
        bool
            True if the node can be expanded, False otherwise.
        """
        return (
            isinstance(node, MolNode) and node.depth < self.max_depth and node.is_open
        )

    def check_node_dominance_pareto(self, node: MolNode) -> bool:
        """
        Returns True if a node is dominated by any point in the current Pareto front.
        """
        graph = self.search_graph
        best_value = np.round(np.array(node.best_total_value), 3) + self.epsilon_pruning
        graph_pareto_front = graph.pareto_front_costs
        if graph_pareto_front.size == 0:
            return False
        # A Pareto vector dominates best_value if it is <= in all objectives and
        # strictly < in at least one objective.
        dominated = np.all(graph_pareto_front <= best_value[None, :], axis=1) & np.any(
            graph_pareto_front < best_value[None, :], axis=1
        )
        return bool(np.any(dominated))

    def remove_dominated_nodes(
        self, open_nodes: list[MolNode]
    ) -> tuple[list[MolNode] | None, bool]:
        """
        Remove dominated nodes from the list of open nodes based on the current Pareto front.
        """
        if self.exclude_dominated_nodes or self.stop_on_full_pareto:
            open_nodes = [
                node for node in open_nodes if not node.is_dominated
            ]  # do not check dominance again
            all_dominated = True
            for node in open_nodes:
                dominated = self.check_node_dominance_pareto(node)
                all_dominated &= dominated
                if self.exclude_dominated_nodes:
                    node.is_dominated = dominated
            if all_dominated and self.stop_on_full_pareto:
                logger.info(
                    "All open nodes are dominated by current Pareto front. Stopping search."
                )
                return None, True
            if self.exclude_dominated_nodes:
                open_nodes = [
                    node for node in open_nodes if not node.is_dominated
                ]  # exclude dominated nodes
        if not open_nodes:
            logger.info(
                "No open nodes available for expansion after dominance check - full Pareto front found. Stopping search."
            )
            return None, True
        return open_nodes, False

    def retro_expansion(self, nodes_and_weights: MolNodeAndWeights) -> bool:
        """
        Expand graph nodes by adding predictions from the single-step model,
        and update search graph values.

        Parameters
        ----------
        nodes_and_weights : MolNodeAndWeights
            Set of tuples containing nodes to expand and their weight indices.

        Returns
        -------
        bool
            True if early resampling is triggered (all weights blocked), False otherwise.
        """
        nodes: list[MolNode] = []
        nodes_and_weights_copy = nodes_and_weights.copy()
        for node, weight in nodes_and_weights_copy:
            if not self.can_expand_retro(node) and node.depth >= self.max_depth:
                for w in weight:
                    if self.weights_open[w]:
                        self.weights_open[w] = False
                        logger.info(
                            f"Node {node.smiles} cannot be expanded with depth {int(node.depth / 2)} (max depth {int(self.max_depth / 2)})"
                        )
                        logger.warning(
                            f"Weight {w} is now blocked from expansion until resampling."
                        )
                nodes_and_weights.remove((node, weight))
            elif not self.can_expand_retro(node):
                logger.critical("Critical error in expansion logic. This is a bug.")
            else:
                nodes.append(node)

        if not nodes:
            logger.warning(
                "All weights selected for expansion cannot expand further. Early resampling triggered."
            )
            return True

        smiles = [node.smiles for node in nodes]
        raw_predictions = self.retro_model.predict(
            smiles, self.top_n
        )  # * adds predictions with costs
        predictions = {
            node: preds for node, preds in zip(nodes, raw_predictions, strict=True)
        }
        new_nodes_and_weights = self.search_graph.expand_graph(
            predictions, nodes_and_weights
        )
        if not new_nodes_and_weights:
            logger.warning(
                "No new nodes were generated during expansion. Retrosynthesis tree was not expanded further."
            )
        else:
            for new_node, _ in new_nodes_and_weights:
                if self.can_expand_retro(new_node):
                    self.search_graph.open_nodes.add(new_node)  # type: ignore

            _ = self.search_graph.update_values(
                new_nodes_and_weights.union(nodes_and_weights)
            )

        return False

    def spawn_new_weights(self, num_iter: int, early_resampling: bool) -> int:
        """
        Samples new weights to screen the Pareto front

        Parameters
        ----------
        num_iter : int
            Current number of iterations completed.
        early_resampling : bool
            Whether resampling is triggered early due to all weights being blocked.

        Returns
        -------
        int
            Updated weight iteration count (reset to 1 if new weights are sampled, incremented otherwise)
            -100 if iteration budget is reached and search should terminate.
        """
        if num_iter == self.weight_iter_budget + 1 or early_resampling:
            logger.info("Sampling new weights...")
            if self.search_graph.weights_open.shape[0] < self.search_graph.no_weights:
                return -100  # exit search
            self.search_graph.reinitialize_graph()
            return 1
        return num_iter

    def choose_next_nodes(self) -> tuple[MolNodeAndWeights, bool]:
        """
        Select nodes to expand next based on current weight preferences.
        Checks for dominance of open nodes if specified.

        For each active weight vector, identifies the most promising nodes
        (those with minimum total values) and groups weights that prefer
        the same nodes to avoid redundant expansions.

        Returns
        -------
        MolNodeAndWeights
            Set of tuples containing selected nodes and their corresponding
            weight indices for expansion.
        bool
            True if search should stop (full Pareto front found), False otherwise.
        """

        # Convert set to sorted list for consistent ordering
        open_nodes = sorted(list(self.search_graph.open_nodes), key=lambda x: x.smiles)
        open_nodes, stop_search = self.remove_dominated_nodes(open_nodes)
        if stop_search or not open_nodes:
            return set(), True

        open_nodes_values = []
        for node in open_nodes:
            open_nodes_values.append(node.total_value)

        open_values = np.array(open_nodes_values)  # dims are n_nodes x n_weights
        min_values = np.min(open_values, axis=0)
        is_min = open_values == min_values[None, :]
        indices_per_dim = []
        for i in range(open_values.shape[1]):
            dim_indices = np.where(is_min[:, i])[0].tolist()
            indices_per_dim.append(dim_indices)

        # Group identical dimensions first, then handle overlaps
        identical_groups = defaultdict(list)
        for i, indices in enumerate(indices_per_dim):
            key = tuple(sorted(indices))
            identical_groups[key].append(i)

        # Remove overlapping nodes from longer keys
        used_nodes = set()
        final_groups = {}

        for key in sorted(identical_groups.keys(), key=len):
            remaining = tuple(x for x in key if x not in used_nodes)
            if remaining:
                final_groups[remaining] = identical_groups[key]
                used_nodes.update(remaining)

        # sort final groups by index key to ensure consistent order
        final_groups = dict(sorted(final_groups.items(), key=lambda item: item[0]))
        nodes_and_weights_to_expand = set()
        for key, dims in final_groups.items():
            i = 1
            for node_idx in key:
                nodes_and_weights_to_expand.add((open_nodes[node_idx], tuple(dims)))
                node = open_nodes[node_idx]
                if i >= len(dims):
                    break
                i += 1
        self.retro_expansion_count += len(nodes_and_weights_to_expand)
        return nodes_and_weights_to_expand, False

    def run_mo_search(self) -> None:
        """
        Execute the complete multi-objective search process.

        Iteratively performs the following steps until termination conditions are met:
        1. Check if new weights should be sampled based on strategy
        2. Select promising nodes for expansion using current weights
        3. Expand selected nodes with retrosynthesis predictions
        4. Update graph values and Pareto front
        5. Handle early resampling if all weights become blocked

        The search terminates when the iteration budget is reached, single-step call or time budget
        is exceeded, no open nodes remain, or all weight vectors are exhausted.
        One can also set a flag that the search terminates as soon as the full Pareto front is found.
        """
        logger.info("Starting multi-objective search process...")
        iter_counter = 1
        weight_iter = 1
        elapsed_time = 0
        start_time = time.time()
        time_budget = np.inf if self.time_budget == 0.0 else self.time_budget
        call_budget = (
            np.inf
            if self.single_step_call_budget == 0
            else self.single_step_call_budget
        )
        while (
            iter_counter < self.iteration_budget
            and elapsed_time < time_budget
            and self.retro_expansion_count < call_budget
        ):
            torch.cuda.empty_cache()
            weight_iter = self.spawn_new_weights(weight_iter, early_resampling=False)

            # Checking for termination criteria (max pareto solutions)
            if len(self.search_graph.pareto_front) >= self.max_pareto_solutions:
                logger.info(
                    f"Reached {self.max_pareto_solutions} Pareto solutions. Stopping search."
                )
                break
            # Checking for termination criteria (no open nodes or weights left)
            break_condition = not self.search_graph.open_nodes or weight_iter < 0
            if break_condition:
                if not self.search_graph.open_nodes:
                    logger.info("No open nodes left to expand. Stopping search.")
                else:
                    logger.info(
                        "All weights have been sampled, no more weights to explore. Stopping search."
                    )
                break

            # Not terminated yet, continue with steps of search
            nodes_and_weights_to_expand, stop_search = self.choose_next_nodes()

            if stop_search:
                break

            early_resampling = self.retro_expansion(nodes_and_weights_to_expand)
            if early_resampling:
                weight_iter = self.spawn_new_weights(
                    iter_counter, early_resampling=True
                )
                self.weights_open = [True] * self.search_graph.no_weights
            if iter_counter % 10 == 0:
                logger.info(
                    f"Completed iteration {iter_counter}/{self.iteration_budget}."
                )
            iter_counter += 1
            weight_iter += 1
            elapsed_time = time.time() - start_time
