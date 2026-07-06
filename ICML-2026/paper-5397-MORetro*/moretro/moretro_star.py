import json
import logging
import logging.config as conf
import os
import pickle
import pprint
import tempfile
from pathlib import Path as PathLib
from typing import Any
from uuid import uuid4

import gin
import graphviz
import matplotlib.pyplot as plt
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw

from moretro.inference.retro_prediction import OneStepModel
from moretro.search.mo_search import MOSearch
from moretro.search.node_type import MolNode, RxnNode
from moretro.utils.base_paths import CONFIG_DIR, LOG_DIR
from moretro.utils.prepare_models import (
    prepare_cost_models,
    prepare_heuristic_fns,
    prepare_starting_mols,
)
from moretro.utils.typing_hints import Path

# set up logging in this main file
logger = logging.getLogger("moretro")


class MORetro:
    def __init__(
        self,
        target: str,
        output_dir: str = "output/my_run",
        visualize_plots: bool = True,
        save_json: bool = False,
    ):
        retro_model = OneStepModel(gin.REQUIRED)  # type: ignore
        building_blocks = prepare_starting_mols(gin.REQUIRED)  # type: ignore
        heuristic_fns = prepare_heuristic_fns(gin.REQUIRED)  # type: ignore
        prepare_cost_models(gin.REQUIRED)  # type: ignore
        self.mo_search = MOSearch(
            target, retro_model, building_blocks, heuristic_fns, gin.REQUIRED
        )  # type: ignore
        self.target = target
        self.output_dir = output_dir
        self.visualize_plots = visualize_plots
        self.save_json = save_json

    def search(self):
        """
        Run the multi-objective retrosynthesis search and plot results
        """
        try:
            self.mo_search.run_mo_search()
            logger.info("Search completed.")
        except KeyboardInterrupt:
            logger.warning("Search interrupted by user.")
        finally:
            # Save all solution costs as pickle
            safe_target_name = self._safe_smiles_dirname(self.target)
            target_dir = f"{self.output_dir}/{safe_target_name}"
            os.makedirs(target_dir, exist_ok=True)
            pickle_path = f"{target_dir}/solution_costs.pkl"
            with open(pickle_path, "wb") as f:
                pickle.dump(self.mo_search.search_graph.solution_cost, f)
            logger.info(f"Saved all solution costs to {pickle_path}")
            if self.save_json:
                self.save_all_routes_as_json(self.output_dir)
                logger.info("Saved all routes as JSON")
            if self.visualize_plots:
                self.visualize_all_solutions(self.output_dir)
                logger.info(
                    f"Creating plots for target and saving in {self.output_dir}"
                )
                self.plot_pareto_front(f"{target_dir}/pareto_front")
                self.plot_pareto_with_dominated(f"{target_dir}/pareto_with_dominated")
            solution_summary = self.get_solution_summary()
            logger.info(
                "Solution summary: \n" + pprint.pformat(solution_summary, indent=2)
            )

    def _visualize_path(self, path: Path, output_path: str, title: str):
        """
        Directly visualize a solution path using the graph structure.

        Parameters
        ----------
        path : Path
            List of nodes representing the synthesis path
        output_path : str
            File path to save the visualization
        title : str
            Title for the visualization
        """
        dot = graphviz.Digraph(format="png")
        dot.attr(label=title, labelloc="t", fontsize="16", rankdir="LR")

        with tempfile.TemporaryDirectory() as temp_img_dir:
            # Draw all nodes in the path
            for node in path:
                if isinstance(node, MolNode):
                    self._add_mol_node(node, dot, temp_img_dir)
                elif isinstance(node, RxnNode):
                    self._add_rxn_node(node, dot)

            # Add edges based on the graph structure
            graph = self.mo_search.search_graph.graph
            for node in path:
                for successor in graph.successors(node):
                    if (
                        successor in path
                    ):  # Only connect nodes that are in this solution path
                        dot.edge(node.smiles, successor.smiles, color="darkgrey")

            dot.render(output_path, cleanup=True)

    def _add_mol_node(self, node: MolNode, dot: graphviz.Digraph, temp_img_dir: str):
        """Add a molecule node to the graph."""
        mol = Chem.MolFromSmiles(node.smiles)
        escaped = node.smiles.replace("/", "_").replace("\\", "_")
        file_path = f"{temp_img_dir}/{escaped}.png"
        Draw.MolToFile(mol, file_path, size=(200, 200))

        # Determine color based on node properties
        if getattr(node, "is_target", False):
            color = "lightsalmon"
        elif node.is_known:  # Building block
            color = "springgreen3"
        else:  # Intermediate
            color = "royalblue"

        dot.node(
            node.smiles,
            label="",
            image=file_path,
            shape="box",
            color=color,
            penwidth="2",
        )

    def _add_rxn_node(self, node: RxnNode, dot: graphviz.Digraph):
        """Add a reaction node to the graph."""
        # Create label with temperature and reagents
        label_parts = []

        # Add temperature at the top
        if hasattr(node, "temp") and node.temp is not None:
            label_parts.append(f"{node.temp}°C")

        # Add reagent molecules
        if hasattr(node, "reagents") and node.reagents:
            # Split reagents by "." and create molecule images for each
            reagent_smiles = node.reagents
            for reagent in reagent_smiles:
                if reagent.strip():  # Skip empty strings
                    label_parts.append(reagent.strip())

        # Combine all parts with line breaks
        label = "\\n".join(label_parts) if label_parts else ""

        dot.node(
            node.smiles,
            label=label,
            shape="box",
            style="rounded",
            color="lightsteelblue",
            penwidth="2",
        )

    def _path_to_route_dict(self, path, cost_vector, is_pareto: bool) -> dict:
        path_set = set(path)
        node_ids = {node: str(uuid4()) for node in path}
        graph = self.mo_search.search_graph.graph

        nodes = []
        for node in path:
            if isinstance(node, MolNode):
                nodes.append(
                    {
                        "id": node_ids[node],
                        "type": "mol",
                        "smiles": node.smiles,
                        "is_target": bool(node.is_target),
                        "is_known": bool(node.is_known),
                        "depth": int(node.depth) // 2,
                    }
                )
            else:
                if node.temp is None:
                    temp_val = None
                else:
                    try:
                        temp_val = float(node.temp)
                    except (ValueError, TypeError):
                        temp_val = str(node.temp)
                nodes.append(
                    {
                        "id": node_ids[node],
                        "type": "reaction",
                        "smiles": node.smiles,
                        "template": node.template,
                        "reagents": list(node.reagents),
                        "temperature_K": temp_val,
                        "cost": node.true_cost.tolist(),
                        "depth": (int(node.depth) + 1) // 2,
                    }
                )

        edges = []
        for node in path:
            for successor in graph.successors(node):
                if successor in path_set:
                    edges.append({"from": node_ids[node], "to": node_ids[successor]})

        num_reactions = sum(1 for n in path if isinstance(n, RxnNode))
        depth = max(int(n.depth) for n in path) // 2

        return {
            "target_smiles": self.target,
            "cost_vector": list(cost_vector),
            "is_pareto": is_pareto,
            "num_reactions": num_reactions,
            "depth": depth,
            "nodes": nodes,
            "edges": edges,
        }

    def save_pareto_routes_as_json(self, output_dir: str) -> list[dict]:
        if not PathLib(output_dir).exists():
            PathLib(output_dir).mkdir(parents=True, exist_ok=True)

        pareto_front = self.mo_search.search_graph.pareto_front
        solution_cost = self.mo_search.search_graph.solution_cost

        entries = []
        for i, (cost_vector, _) in enumerate(pareto_front.items()):
            if cost_vector not in solution_cost:
                continue
            path, _ = solution_cost[cost_vector]
            route_dict = self._path_to_route_dict(path, cost_vector, is_pareto=True)
            filename = f"pareto_route_{i + 1}.json"
            filepath = str(PathLib(output_dir) / filename)
            with open(filepath, "w") as f:
                json.dump(route_dict, f, indent=2)
            entries.append(
                {
                    "file": str(PathLib(output_dir).name + "/" + filename),
                    "cost_vector": list(cost_vector),
                }
            )
            logger.debug(f"Saved Pareto route {i + 1} to {filepath}")

        return entries

    def save_dominated_routes_as_json(
        self, output_dir: str, max_solutions: int = 100
    ) -> list[dict]:
        if not PathLib(output_dir).exists():
            PathLib(output_dir).mkdir(parents=True, exist_ok=True)

        pareto_front = self.mo_search.search_graph.pareto_front
        solution_cost = self.mo_search.search_graph.solution_cost

        dominated = [
            (cost, path)
            for cost, (path, _) in solution_cost.items()
            if cost not in pareto_front
        ][:max_solutions]

        entries = []
        for i, (cost_vector, path) in enumerate(dominated):
            route_dict = self._path_to_route_dict(path, cost_vector, is_pareto=False)
            filename = f"dominated_route_{i + 1}.json"
            filepath = str(PathLib(output_dir) / filename)
            with open(filepath, "w") as f:
                json.dump(route_dict, f, indent=2)
            entries.append(
                {
                    "file": str(PathLib(output_dir).name + "/" + filename),
                    "cost_vector": list(cost_vector),
                }
            )

        return entries

    def _save_solution_summary(
        self, output_dir: str, pareto_entries: list, dominated_entries: list
    ):
        pareto_front = self.mo_search.search_graph.pareto_front
        solution_cost = self.mo_search.search_graph.solution_cost
        total = len(solution_cost)
        n_pareto = len(pareto_front)
        summary = {
            "target_smiles": self.target,
            "total_solutions": total,
            "pareto_solutions": n_pareto,
            "dominated_solutions": total - n_pareto,
            "pareto_routes": pareto_entries,
            "dominated_routes": dominated_entries,
        }
        filepath = str(PathLib(output_dir) / "solution_summary.json")
        with open(filepath, "w") as f:
            json.dump(summary, f, indent=2)
        logger.debug(f"Saved solution summary to {filepath}")

    def save_all_routes_as_json(self, output_dir: str = "output"):
        safe_target_name = self._safe_smiles_dirname(self.target)
        target_dir = str(PathLib(output_dir) / safe_target_name)
        PathLib(target_dir).mkdir(parents=True, exist_ok=True)

        pareto_dir = str(PathLib(target_dir) / "pareto")
        dominated_dir = str(PathLib(target_dir) / "dominated")

        pareto_entries = self.save_pareto_routes_as_json(pareto_dir)
        dominated_entries = self.save_dominated_routes_as_json(dominated_dir)
        self._save_solution_summary(target_dir, pareto_entries, dominated_entries)

        logger.info(f"All JSON routes saved to {target_dir}")

    def visualize_pareto_solutions(self, output_dir: str):
        """
        Visualize all Pareto-optimal synthesis routes using direct path visualization.

        Parameters
        ----------
        output_dir : str
            Directory to save the visualization files
        """
        if not PathLib(output_dir).exists():
            PathLib(output_dir).mkdir(parents=True, exist_ok=True)

        pareto_front = self.mo_search.search_graph.pareto_front
        solution_cost = self.mo_search.search_graph.solution_cost

        logger.info(f"Found {len(pareto_front)} Pareto-optimal solutions")

        for i, (cost_vector, _) in enumerate(pareto_front.items()):
            if cost_vector in solution_cost:
                path, _ = solution_cost[cost_vector]
                title = f"Pareto Solution {i + 1}\\nCost: {[f'{c:.3f}' for c in cost_vector]}"
                output_path = str(PathLib(output_dir) / f"pareto_route_{i + 1}")
                self._visualize_path(path, output_path, title)
                logger.debug(f"Saved Pareto route {i + 1} to {output_path}.png")

    def visualize_dominated_solutions(self, output_dir: str, max_solutions: int = 100):
        """
        Visualize dominated (non-Pareto) synthesis routes using direct path visualization.

        Parameters
        ----------
        output_dir : str
            Directory to save the visualization files
        max_solutions : int
            Maximum number of dominated solutions to visualize
        """
        if not PathLib(output_dir).exists():
            PathLib(output_dir).mkdir(parents=True, exist_ok=True)

        pareto_front = self.mo_search.search_graph.pareto_front
        solution_cost = self.mo_search.search_graph.solution_cost

        # Find dominated solutions
        dominated_solutions = [
            (cost, path, idx)
            for cost, (path, idx) in solution_cost.items()
            if cost not in pareto_front
        ]

        logger.info(f"Found {len(dominated_solutions)} dominated solutions")
        dominated_solutions = dominated_solutions[:max_solutions]
        logger.debug(
            f"Saving a max of {max_solutions} dominated solutions at {output_dir}"
        )

        for i, (cost_vector, path, _) in enumerate(dominated_solutions):
            title = f"Dominated Solution {i + 1}\\nCost: {[f'{c:.3f}' for c in cost_vector]}"
            output_path = str(PathLib(output_dir) / f"dominated_route_{i + 1}")
            self._visualize_path(path, output_path, title)

    def visualize_all_solutions(self, output_dir: str = "output"):
        """
        Visualize both Pareto-optimal and dominated synthesis routes using direct visualization.

        Parameters
        ----------
        output_dir : str
            Base directory to save the visualization files
        """
        # Create target-specific folder structure
        safe_target_name = self._safe_smiles_dirname(self.target)
        target_dir = str(PathLib(output_dir) / safe_target_name)

        pareto_dir = str(PathLib(target_dir) / "pareto")
        dominated_dir = str(PathLib(target_dir) / "dominated")

        logger.info("Visualizing Pareto-optimal solutions...")
        self.visualize_pareto_solutions(pareto_dir)

        logger.info("Visualizing dominated solutions...")
        self.visualize_dominated_solutions(dominated_dir)

        logger.info(f"All visualizations saved to {target_dir}")

    def get_solution_summary(self) -> dict[str, Any]:
        """
        Get a summary of all discovered solutions.

        Returns
        -------
        dict[str, Any]
            Summary containing counts and cost ranges
        """
        pareto_front = self.mo_search.search_graph.pareto_front
        solution_cost = self.mo_search.search_graph.solution_cost

        pareto_costs = list(pareto_front.keys())
        all_costs = list(solution_cost.keys())
        dominated_costs = [cost for cost in all_costs if cost not in pareto_front]

        summary: dict[str, Any] = {
            "total_solutions": len(all_costs),
            "pareto_solutions": len(pareto_costs),
            "dominated_solutions": len(dominated_costs),
        }

        if all_costs:
            all_costs_array = np.array(all_costs)
            summary["cost_ranges"] = {
                f"objective_{i}": {
                    "min": round(float(all_costs_array[:, i].min()), 2),
                    "max": round(float(all_costs_array[:, i].max()), 2),
                    "mean": round(float(all_costs_array[:, i].mean()), 2),
                }
                for i in range(all_costs_array.shape[1])
            }

        return summary

    def plot_pareto_front(
        self,
        output_path: str | None = None,
        figsize: tuple[int, int] = (10, 6),
        show_weights: bool = False,
        weight_fontsize: int = 10,
    ):
        """
        Plot the Pareto front in 2D or 3D with weight labels for each point.

        Parameters
        ----------
        output_path : str
            File path to save the plot (without extension). If None, will save to
            output/{target_smiles}/pareto_front
        figsize : tuple[int, int]
            Figure size (width, height)
        show_weights : bool
            Whether to show weight vectors as labels
        weight_fontsize : int
            Font size for weight labels
        """
        pareto_front = self.mo_search.search_graph.pareto_front

        if not pareto_front:
            logger.warning("No Pareto solutions found to plot.")
            return

        if output_path is None:
            safe_target_name = self._safe_smiles_dirname(self.target)
            output_path = f"{self.output_dir}/{safe_target_name}/pareto_front"

        if not PathLib(output_path).parent.exists():
            PathLib(output_path).parent.mkdir(parents=True, exist_ok=True)

        # Extract cost vectors and weights
        cost_vectors = list(pareto_front.keys())
        weights = list(pareto_front.values())

        costs_array = np.array(cost_vectors)
        n_objectives = costs_array.shape[1]

        if n_objectives < 2:
            logger.warning("Need at least 2 objectives to plot Pareto front.")
            return
        elif n_objectives == 2:
            self._plot_pareto_2d(
                costs_array,
                weights,
                output_path,
                figsize,
                show_weights,
                weight_fontsize,
                None,
            )
        elif n_objectives == 3:
            self._plot_pareto_3d(
                costs_array,
                weights,
                output_path,
                figsize,
                show_weights,
                weight_fontsize,
                None,
            )
        else:
            logger.info(
                f"More than 3 objectives ({n_objectives}). Plotting Pareto front for first 3 dimensions."
            )
            reduced_costs = costs_array[:, :3]
            pareto_indices = self._compute_pareto_front_indices(reduced_costs)
            reduced_costs_pareto = reduced_costs[pareto_indices]
            weights_pareto = [weights[i] for i in pareto_indices]
            self._plot_pareto_3d(
                reduced_costs_pareto,
                weights_pareto,
                output_path,
                figsize,
                show_weights,
                weight_fontsize,
                None,
            )

    def plot_pareto_with_dominated(
        self,
        output_path: str | None = None,
        figsize: tuple[int, int] = (12, 8),
        show_weights: bool = False,
        weight_fontsize: int = 10,
    ):
        """
        Plot the Pareto front with dominated solutions in the background.

        Parameters
        ----------
        output_path : str
            File path to save the plot (without extension). If None, will save to
            output/{target_smiles}/pareto_with_dominated
        figsize : tuple[int, int]
            Figure size (width, height)
        show_weights : bool
            Whether to show weight vectors as labels (only for Pareto points)
        weight_fontsize : int
            Font size for weight labels
        """
        pareto_front = self.mo_search.search_graph.pareto_front
        solution_cost = self.mo_search.search_graph.solution_cost

        if not pareto_front:
            logger.warning("No Pareto solutions found to plot.")
            return

        if output_path is None:
            safe_target_name = self._safe_smiles_dirname(self.target)
            output_path = f"{self.output_dir}/{safe_target_name}/pareto_with_dominated"

        if not PathLib(output_path).parent.exists():
            PathLib(output_path).parent.mkdir(parents=True, exist_ok=True)

        # Extract all cost vectors
        all_cost_vectors = list(solution_cost.keys())
        all_costs_array = np.array(all_cost_vectors)
        n_objectives = all_costs_array.shape[1]

        # Separate Pareto and dominated
        pareto_costs = list(pareto_front.keys())
        pareto_weights = list(pareto_front.values())
        dominated_costs = [
            cost for cost in all_cost_vectors if cost not in pareto_front
        ]

        if n_objectives < 2:
            logger.warning("Need at least 2 objectives to plot Pareto front.")
            return
        elif n_objectives == 2:
            pareto_array = np.array(pareto_costs)
            dominated_array = np.array(dominated_costs) if dominated_costs else None
            self._plot_pareto_2d(
                pareto_array,
                pareto_weights,
                output_path,
                figsize,
                show_weights,
                weight_fontsize,
                dominated_array,
            )
        elif n_objectives == 3:
            pareto_array = np.array(pareto_costs)
            dominated_array = np.array(dominated_costs) if dominated_costs else None
            self._plot_pareto_3d(
                pareto_array,
                pareto_weights,
                output_path,
                figsize,
                show_weights,
                weight_fontsize,
                dominated_array,
            )
        else:
            logger.info(
                f"More than 3 objectives ({n_objectives}). Plotting first 3 dimensions with dominated solutions."
            )
            # Reduce all to first 3 dimensions
            reduced_pareto = np.array(pareto_costs)[:, :3]
            reduced_indices = self._compute_pareto_front_indices(reduced_pareto)
            reduced_pareto = reduced_pareto[reduced_indices]
            pareto_weights = [pareto_weights[i] for i in reduced_indices]
            reduced_dominated = (
                np.array(dominated_costs)[:, :3] if dominated_costs else None
            )
            self._plot_pareto_3d(
                reduced_pareto,
                pareto_weights,
                output_path,
                figsize,
                show_weights,
                weight_fontsize,
                reduced_dominated,
            )

    def _plot_pareto_2d(
        self,
        costs_array: np.ndarray,
        weights: list,
        output_path: str,
        figsize: tuple[int, int],
        show_weights: bool,
        weight_fontsize: int,
        dominated_costs: np.ndarray | None = None,
    ):
        """Plot 2D Pareto front."""
        plt.figure(figsize=figsize)

        # Plot dominated points first (background)
        if dominated_costs is not None:
            plt.scatter(
                dominated_costs[:, 0],
                dominated_costs[:, 1],
                c="gray",
                s=50,
                alpha=0.3,
                label="Dominated Solutions",
            )

        # Plot Pareto points
        plt.scatter(
            costs_array[:, 0],
            costs_array[:, 1],
            c="red",
            s=100,
            alpha=0.7,
            edgecolors="black",
            linewidth=1,
            label="Pareto Solutions",
        )

        # Add weight labels if requested
        if show_weights:
            for cost, weight_list in zip(costs_array, weights, strict=True):
                # Format weight vector(s) for display - each nested list on a new line
                weight_parts = []
                for weight in weight_list:
                    formatted_weights = [f"{w:.2f}" for w in weight]
                    weight_parts.append(f"[{', '.join(formatted_weights)}]")

                weight_str = "\n".join(weight_parts)
                plt.annotate(
                    weight_str,
                    (cost[0], cost[1]),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=max(6, weight_fontsize - 2),  # Make font size smaller
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
                )

        plt.xlabel("Objective 1", fontsize=12)
        plt.ylabel("Objective 2", fontsize=12)
        plt.title("Pareto Front (2D)", fontsize=14, fontweight="bold")
        plt.grid(True, alpha=0.3)
        plt.legend()

        # Connect points to show Pareto front
        sorted_indices = np.argsort(costs_array[:, 0])
        sorted_costs = costs_array[sorted_indices]
        plt.plot(
            sorted_costs[:, 0],
            sorted_costs[:, 1],
            "r--",
            alpha=0.5,
            linewidth=1,
        )

        plt.tight_layout()
        plt.savefig(f"{output_path}.png", dpi=300, bbox_inches="tight")
        plt.savefig(f"{output_path}.pdf", bbox_inches="tight")
        plt.close()

    def _plot_pareto_3d(
        self,
        costs_array: np.ndarray,
        weights: list,
        output_path: str,
        figsize: tuple[int, int],
        show_weights: bool,
        weight_fontsize: int,
        dominated_costs: np.ndarray | None = None,
    ):
        """Plot 3D Pareto front."""
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")

        # Plot dominated points first (background)
        if dominated_costs is not None:
            ax.scatter(
                dominated_costs[:, 0],
                dominated_costs[:, 1],
                dominated_costs[:, 2],  # type: ignore
                c="gray",
                s=50,  # type: ignore
                alpha=0.3,
                label="Dominated Solutions",
            )

        # Plot Pareto points
        ax.scatter(
            costs_array[:, 0],
            costs_array[:, 1],
            costs_array[:, 2],  # type: ignore
            c="red",
            s=100,  # type: ignore
            alpha=0.7,
            edgecolors="black",
            label="Pareto Solutions",
        )

        # Add weight labels if requested
        if show_weights:
            for cost, weight_list in zip(costs_array, weights, strict=True):
                # Format weight vector(s) for display - each nested list on a new line
                weight_parts = []
                for weight in weight_list:
                    formatted_weights = [f"{w:.2f}" for w in weight]
                    weight_parts.append(f"[{', '.join(formatted_weights)}]")

                weight_str = "\n".join(weight_parts)
                # Add text annotation
                ax.text(
                    cost[0],
                    cost[1],
                    cost[2],
                    weight_str,  # type: ignore
                    fontsize=max(6, weight_fontsize - 2),
                )

        ax.set_xlabel("Objective 1", fontsize=12, labelpad=10)
        ax.set_ylabel("Objective 2", fontsize=12, labelpad=10)
        ax.set_zlabel("Objective 3", fontsize=12, labelpad=15)  # type: ignore
        ax.set_title("Pareto Front (3D)", fontsize=14, fontweight="bold")
        ax.legend()

        # Add grid
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        fig.subplots_adjust(
            left=0.12, right=0.88, bottom=0.12, top=0.92
        )  # Middle ground margins
        plt.savefig(f"{output_path}.png", dpi=300)
        plt.savefig(f"{output_path}.pdf")
        plt.close()

    def _compute_pareto_front_indices(self, costs: np.ndarray) -> list[int]:
        """
        Compute the indices of points on the Pareto front for minimization.

        Parameters
        ----------
        costs : np.ndarray
            Array of cost vectors, shape (n_points, n_objectives)

        Returns
        -------
        list[int]
            Indices of non-dominated points
        """
        n = len(costs)
        is_pareto = [True] * n
        for i in range(n):
            if not is_pareto[i]:
                continue
            for j in range(n):
                if i == j:
                    continue
                # Check if j dominates i (j is better or equal in all, better in at least one)
                dominates = True
                strictly_better = False
                for k in range(costs.shape[1]):
                    if costs[j, k] > costs[i, k]:
                        dominates = False
                        break
                    if costs[j, k] < costs[i, k]:
                        strictly_better = True
                if dominates and strictly_better:
                    is_pareto[i] = False
                    break
        return [i for i in range(n) if is_pareto[i]]

    def _safe_smiles_dirname(self, smiles: str) -> str:
        """
        Convert SMILES string to a safe directory name by replacing problematic characters.

        Parameters
        ----------
        smiles : str
            SMILES string to convert

        Returns
        -------
        str
            Safe directory name
        """
        # Replace problematic characters with safe alternatives
        safe_name = smiles.replace("/", "_slash_")
        safe_name = safe_name.replace("\\", "_backslash_")
        safe_name = safe_name.replace(":", "_colon_")
        safe_name = safe_name.replace("*", "_star_")
        safe_name = safe_name.replace("?", "_question_")
        safe_name = safe_name.replace('"', "_quote_")
        safe_name = safe_name.replace("<", "_lt_")
        safe_name = safe_name.replace(">", "_gt_")
        safe_name = safe_name.replace("|", "_pipe_")
        return safe_name


if __name__ == "__main__":
    import configparser
    import io
    from argparse import ArgumentParser

    import pandas as pd

    # add argument for saving paths
    parser = ArgumentParser()
    parser.add_argument(
        "--output_dir",
        type=str,
        default="my_run",
        help="Run name / output subdirectory under 'output/' for saved files",
    )
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--config_file", type=str, default="search_config.gin")
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Whether to visualize synthesis routes.",
    )
    parser.add_argument(
        "--save_json",
        action="store_true",
        help="Whether to save synthesis routes as JSON files.",
    )
    args = parser.parse_args()

    # overwrite file save location for logger config
    new_log_path = str(LOG_DIR / f"{args.output_dir}.log")
    os.makedirs(os.path.dirname(new_log_path), exist_ok=True)

    # Configure logging dynamically
    config = configparser.ConfigParser()
    config.read(CONFIG_DIR / "logging.conf")

    # Update the log file path in the configuration
    # The args are stored as a string representation of a tuple: ('path', 'mode')
    config.set("handler_fileHandler", "args", f"('{new_log_path}', 'a')")

    # Apply configuration
    with io.StringIO() as config_buffer:
        config.write(config_buffer)
        config_buffer.seek(0)
        conf.fileConfig(config_buffer, disable_existing_loggers=False)

    gin.parse_config_file(CONFIG_DIR / args.config_file)

    mol_file = pd.read_csv(args.dataset, header=None, sep=",")
    for target_smiles in mol_file[0].tolist():
        moretro = MORetro(
            target_smiles,
            output_dir=f"output/{args.output_dir}",
            visualize_plots=args.visualize,
            save_json=args.save_json,
        )
        moretro.search()
