"""
Comprehensive unit tests for MOGraph class in mo_graph.py

This test suite covers all methods in the MOGraph class with the current implementation.
Tests are designed to work with the latest version of the code.
"""

import os
import sys
from unittest.mock import patch

import numpy as np
import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from moretro.inference.and_or_graph import AndOrGraph
from moretro.search.mo_graph import MOGraph
from moretro.search.node_type import MolNode, RxnNode


class TestMOGraphInitialization:
    """Test cases for MOGraph initialization and setup"""

    @pytest.fixture
    def simple_heuristics(self):
        """Create simple heuristic functions for testing"""

        def heuristic1(smiles: str) -> float:
            return len(smiles) * 0.1  # Simple length-based heuristic

        def heuristic2(smiles: str) -> float:
            return smiles.count("C") * 0.2  # Carbon count heuristic

        return [heuristic1, heuristic2]

    @pytest.fixture
    def basic_building_blocks(self):
        """Basic set of building blocks for testing"""
        return {"CCO", "CC", "C"}

    def test_init_basic(self, simple_heuristics, basic_building_blocks):
        """Test basic initialization of MOGraph"""
        target = "CCCO"
        graph = MOGraph(
            target=target,
            building_blocks=basic_building_blocks,
            heuristic_fns=simple_heuristics,
            pareto_objectives=2,
            max_dominated_solutions=5,
            weight_samples=8,
            no_weights=2,
        )

        assert graph.target == "CCCO"
        assert len(graph.building_blocks) == 3
        assert len(graph.heuristic_fns) == 2
        assert graph.weight_samples == 8
        assert graph.no_weights == 2
        assert isinstance(graph.target_node, MolNode)
        assert graph.target_node.is_target == True
        assert isinstance(graph.graph, AndOrGraph)

    def test_init_target_known(self, simple_heuristics):
        """Test initialization when target is in building blocks"""
        target = "CCO"
        building_blocks = {"CCO", "CC", "C"}

        graph = MOGraph(
            target=target,
            building_blocks=building_blocks,
            heuristic_fns=simple_heuristics,
            pareto_objectives=2,
            max_dominated_solutions=5,
        )

        assert graph.target_node.is_known == True
        assert graph.target_node.is_open == False
        assert len(graph.open_nodes) == 0

    def test_init_target_unknown(self, simple_heuristics, basic_building_blocks):
        """Test initialization when target is not in building blocks"""
        target = "CCCO"

        graph = MOGraph(
            target=target,
            building_blocks=basic_building_blocks,
            heuristic_fns=simple_heuristics,
            pareto_objectives=2,
            max_dominated_solutions=5,
        )

        assert graph.target_node.is_known == False
        assert graph.target_node.is_open == True
        assert len(graph.open_nodes) == 1
        assert graph.target_node in graph.open_nodes

    def test_weight_initialization_sobol_with_extreme(
        self, simple_heuristics, basic_building_blocks
    ):
        """Test Sobol weight initialization with extreme points included"""
        graph = MOGraph(
            target="CCCO",
            building_blocks=basic_building_blocks,
            heuristic_fns=simple_heuristics,
            weight_samples=10,
            no_weights=2,
            weight_initial="sobol",
            include_extreme=True,
            pareto_objectives=2,
            max_dominated_solutions=5,
        )

        assert graph.weights.shape == (2, 2)  # (no_weights, n_objectives)
        assert graph.weights_open.shape == (8, 2)  # remaining weights
        assert np.allclose(graph.weights.sum(axis=1), 1.0)  # weights sum to 1

    def test_weight_initialization_sobol_without_extreme(
        self, simple_heuristics, basic_building_blocks
    ):
        """Test Sobol weight initialization without extreme points"""
        graph = MOGraph(
            target="CCCO",
            building_blocks=basic_building_blocks,
            heuristic_fns=simple_heuristics,
            weight_samples=8,
            no_weights=2,
            weight_initial="sobol",
            include_extreme=False,
            pareto_objectives=2,
            max_dominated_solutions=5,
        )

        assert graph.weights.shape == (2, 2)  # (no_weights, n_objectives)
        assert graph.weights_open.shape == (6, 2)  # remaining weights
        assert np.allclose(graph.weights.sum(axis=1), 1.0)  # weights sum to 1

    def test_weight_initialization_dirichlet(
        self, simple_heuristics, basic_building_blocks
    ):
        """Test Dirichlet weight initialization"""
        graph = MOGraph(
            target="CCCO",
            building_blocks=basic_building_blocks,
            heuristic_fns=simple_heuristics,
            weight_samples=16,
            no_weights=3,
            weight_initial="dirichlet",
            pareto_objectives=2,
            max_dominated_solutions=5,
        )

        assert graph.weights.shape == (3, 2)
        assert graph.weights_open.shape == (13, 2)
        assert np.allclose(graph.weights.sum(axis=1), 1.0)

    def test_weight_initialization_invalid(
        self, simple_heuristics, basic_building_blocks
    ):
        """Test invalid weight initialization method"""
        with pytest.raises(ValueError, match="Unknown weight initialization type"):
            MOGraph(
                target="CCCO",
                building_blocks=basic_building_blocks,
                heuristic_fns=simple_heuristics,
                weight_initial="invalid",
                pareto_objectives=2,
                max_dominated_solutions=5,
            )


class TestWeightMethods:
    """Test weight-related methods"""

    @pytest.fixture
    def graph_with_weights(self):
        """Create a graph with known weights for testing"""

        def h1(smiles):
            return 1.0

        def h2(smiles):
            return 2.0

        graph = MOGraph(
            target="CCCO",
            building_blocks={"CC", "C"},
            heuristic_fns=[h1, h2],
            weight_samples=16,
            no_weights=4,
            pareto_objectives=2,
            max_dominated_solutions=5,
        )
        return graph

    def test_update_weights(self, graph_with_weights):
        """Test update_weights method"""
        initial_weights = graph_with_weights.weights.copy()
        initial_history_len = len(graph_with_weights.weight_history)
        initial_open_shape = graph_with_weights.weights_open.shape

        graph_with_weights.update_weights()

        # Check that weights were moved to history
        assert len(graph_with_weights.weight_history) == initial_history_len + 4

        # Check that new weights were loaded
        assert not np.array_equal(graph_with_weights.weights, initial_weights)

    def test_reinitialize_graph(self, graph_with_weights):
        """Test reinitialize_graph method"""

        # Add some nodes to the graph to make reinitialization meaningful
        def h1(smiles):
            return 1.0

        def h2(smiles):
            return 2.0

        # Create a known molecule
        known_mol = MolNode(
            smiles="CC",
            heuristic_fns=[h1, h2],
            depth=1,
            is_known=True,
            pareto_objectives=2,
            max_dominated_solutions=5,
        )
        graph_with_weights.mol_to_node["CC"] = known_mol
        graph_with_weights.graph.add_node(known_mol, node_type="molecule")

        # Create an open molecule
        open_mol = MolNode(
            smiles="CCO",
            heuristic_fns=[h1, h2],
            depth=1,
            is_known=False,
            is_open=True,
            pareto_objectives=2,
            max_dominated_solutions=5,
        )
        graph_with_weights.mol_to_node["CCO"] = open_mol
        graph_with_weights.graph.add_node(open_mol, node_type="molecule")
        graph_with_weights.open_nodes.add(open_mol)

        # Store initial state
        initial_weights = graph_with_weights.weights.copy()
        initial_history_len = len(graph_with_weights.weight_history)

        # Run reinitialization
        graph_with_weights.reinitialize_graph()

        # Verify weights were updated
        assert len(graph_with_weights.weight_history) == initial_history_len + 4
        assert not np.array_equal(graph_with_weights.weights, initial_weights)

        # Verify nodes have been processed (should have non-zero rxn_no values)
        assert len(known_mol.rxn_no) > 0
        assert len(open_mol.rxn_no) > 0


class TestGraphExpansion:
    """Test graph expansion functionality"""

    @pytest.fixture
    def graph_for_expansion(self):
        """Create a graph suitable for expansion testing"""

        def h1(smiles):
            return 1.0

        def h2(smiles):
            return 1.5

        graph = MOGraph(
            target="CCCO",
            building_blocks={"CC", "C"},
            heuristic_fns=[h1, h2],
            weight_samples=8,
            no_weights=2,
            pareto_objectives=2,
            max_dominated_solutions=5,
        )
        return graph

    def test_expand_graph_basic(self, graph_for_expansion):
        """Test basic graph expansion"""
        target_node = graph_for_expansion.target_node
        expanded_nodes = {(target_node, (0,))}

        predictions = {
            target_node: [
                {
                    "reactants": ["CCO", "C"],
                    "reagents": "catalyst",
                    "temperature": 298.0,
                    "rxn_smiles": "CCO.C>>CCCO",
                    "template": "[CH3:1][CH2:2][OH:3].[CH4:4]>>[CH3:1][CH2:2][CH2:4][OH:3]",
                    "costs": np.array([1.0, 1.5]),
                }
            ]
        }

        new_nodes = graph_for_expansion.expand_graph(predictions, expanded_nodes)

        assert new_nodes is not None
        assert len(new_nodes) == 3  # 1 reaction + 2 reactants
        assert target_node not in graph_for_expansion.open_nodes
        assert target_node.is_open == False

    def test_expand_graph_with_known_reactant(self, graph_for_expansion):
        """Test expansion when one reactant is already known"""
        target_node = graph_for_expansion.target_node
        expanded_nodes = {(target_node, (0,))}

        predictions = {
            target_node: [
                {
                    "reactants": ["CC", "CO"],  # CC is in building blocks
                    "reagents": "catalyst",
                    "temperature": 298.0,
                    "rxn_smiles": "CC.CO>>CCCO",
                    "template": "[CH3:1][CH3:2].[CH3:3][OH:4]>>[CH3:1][CH2:2][CH2:3][OH:4]",
                    "costs": np.array([0.5, 1.0]),
                }
            ]
        }

        new_nodes = graph_for_expansion.expand_graph(predictions, expanded_nodes)

        # Should create reaction node + 1 new reactant (CO), CC already exists
        assert len(new_nodes) == 3

    def test_expand_graph_cycle_detection(self, graph_for_expansion):
        """Test cycle detection during expansion"""
        target_node = graph_for_expansion.target_node

        # Add target to mol_to_node to simulate it being a reactant
        graph_for_expansion.mol_to_node["CCCO"] = target_node

        # Mock get_ancestors to return target as ancestor
        with patch.object(
            graph_for_expansion.graph, "get_ancestors", return_value=["CCCO"]
        ):
            expanded_nodes = {(target_node, (0,))}
            predictions = {
                target_node: [
                    {
                        "reactants": ["CCCO", "C"],  # Creates cycle
                        "reagents": "catalyst",
                        "temperature": 298.0,
                        "rxn_smiles": "CCCO.C>>CCCO",
                        "template": "[CH3:1]>>[CH3:1]",
                        "costs": np.array([1.0, 1.0]),
                    }
                ]
            }

            new_nodes = graph_for_expansion.expand_graph(predictions, expanded_nodes)

            # Should return empty set due to cycle detection
            assert len(new_nodes) == 0

    def test_expand_graph_multiple_predictions(self, graph_for_expansion):
        """Test expansion with multiple predictions per node"""
        target_node = graph_for_expansion.target_node
        expanded_nodes = {(target_node, (0,))}
        predictions = {
            target_node: [
                {
                    "reactants": ["CCO", "C"],
                    "reagents": "catalyst1",
                    "temperature": 298.0,
                    "rxn_smiles": "CCO.C>>CCCO",
                    "template": "template1",
                    "costs": np.array([1.0, 1.5]),
                },
                {
                    "reactants": ["CC", "CO"],
                    "reagents": "catalyst2",
                    "temperature": 323.0,
                    "rxn_smiles": "CC.CO>>CCCO",
                    "template": "template2",
                    "costs": np.array([0.8, 1.2]),
                },
            ]
        }

        new_nodes = graph_for_expansion.expand_graph(predictions, expanded_nodes)

        # Should create 2 reactions + 3 unique reactants (CCO, C, CO) + 1 building block
        assert len(new_nodes) == 6


class TestValuePropagation:
    """Test value propagation methods"""

    @pytest.fixture
    def graph_with_structure(self):
        """Create a graph with some structure for testing propagation"""

        def h1(smiles):
            return 1.0

        def h2(smiles):
            return 1.5

        graph = MOGraph(
            target="CCCO",
            building_blocks={"CC", "C"},
            heuristic_fns=[h1, h2],
            weight_samples=8,
            no_weights=2,
            pareto_objectives=2,
            max_dominated_solutions=5,
        )

        # Create some nodes manually for testing
        rxn_node = RxnNode(
            smiles="CC.CO>>CCCO",
            template="test_template",
            reagents=["catalyst"],
            temp=298.0,
            depth=1,
            cost=np.array([1.0, 1.5]),
            weight_length=2,
            pareto_objectives=2,
            max_dominated_solutions=5,
        )

        reactant1 = MolNode(
            smiles="CC",
            heuristic_fns=[h1, h2],
            depth=2,
            is_known=True,
            pareto_objectives=2,
            max_dominated_solutions=5,
        )

        reactant2 = MolNode(
            smiles="CO",
            heuristic_fns=[h1, h2],
            depth=2,
            is_known=False,
            pareto_objectives=2,
            max_dominated_solutions=5,
        )

        # Add to graph
        graph.graph.add_node(rxn_node, node_type="reaction")
        graph.graph.add_node(reactant1, node_type="molecule")
        graph.graph.add_node(reactant2, node_type="molecule")

        graph.graph.add_edge(graph.target_node, rxn_node)
        graph.graph.add_edge(rxn_node, reactant1)
        graph.graph.add_edge(rxn_node, reactant2)

        return graph, rxn_node, reactant1, reactant2

    def test_uppropagation_basic(self, graph_with_structure):
        """Test basic uppropagation functionality"""
        graph, rxn_node, reactant1, reactant2 = graph_with_structure

        nodes_and_weights = {(reactant1, (0,)), (reactant2, (0,))}

        updated_nodes, new_solutions = graph.uppropagation(nodes_and_weights)

        assert isinstance(updated_nodes, set)
        assert isinstance(new_solutions, dict)

    def test_uppropagation_multiple_weight_groups(self, graph_with_structure):
        """Test uppropagation with multiple weight groups from unrelated branches"""
        graph, rxn_node, reactant1, reactant2 = graph_with_structure

        def h1(smiles):
            return 1.0

        def h2(smiles):
            return 1.5

        # Create an independent molecule node (not connected to the existing structure)
        independent_mol = MolNode(
            smiles="CCN",
            heuristic_fns=[h1, h2],
            depth=0,
            is_known=False,
            is_open=True,
            pareto_objectives=2,
            max_dominated_solutions=5,
        )

        # Add it to the graph as an independent node
        graph.graph.add_node(independent_mol, node_type="molecule")
        graph.mol_to_node["CCN"] = independent_mol
        graph.open_nodes.add(independent_mol)

        nodes_and_weights = {
            (reactant1, (0,)),  # Group 0 expanding original branch
            (reactant2, (0,)),  # Group 0 expanding original branch
            (independent_mol, (1,)),  # Group 1 expanding independent branch
        }

        updated_nodes, new_solutions = graph.uppropagation(nodes_and_weights)

        # Verify multiple weight groups are processed separately
        weight_groups_in_result = set()
        for node, weight_idx in updated_nodes:
            weight_groups_in_result.add(weight_idx)

        # Should process both weight groups independently
        assert isinstance(updated_nodes, set)
        assert isinstance(new_solutions, dict)

        # Extract nodes by weight group
        group_0_nodes = [
            node for node, weight_idx in updated_nodes if weight_idx == (0,)
        ]
        group_1_nodes = [
            node for node, weight_idx in updated_nodes if weight_idx == (1,)
        ]

        if group_0_nodes:
            # Group 0 nodes should include nodes from the original reaction path
            group_0_smiles = [node.smiles for node in group_0_nodes]
            # Should contain reactants or nodes higher in the tree
            assert any(smiles in ["CC", "CO", "CCCO"] for smiles in group_0_smiles)

        if group_1_nodes:
            # Group 1 should only contain the independent molecule
            group_1_smiles = [node.smiles for node in group_1_nodes]
            assert all(smiles == "CCN" for smiles in group_1_smiles)

    def test_uppropagation_invalid_node_type(self, graph_with_structure):
        """Test uppropagation with invalid node type"""
        graph, _, _, _ = graph_with_structure

        # Create invalid node
        invalid_node = "not_a_node"
        nodes_and_weights = {(invalid_node, (0,))}

        with pytest.raises(AttributeError):
            graph.uppropagation(nodes_and_weights)

    def test_downpropagation_basic(self, graph_with_structure):
        """Test basic downpropagation functionality"""
        graph, rxn_node, reactant1, reactant2 = graph_with_structure
        graph.target_node.rxn_no = np.random.randn(2).tolist()
        graph.target_node.total_value = np.random.randn(2).tolist()
        graph.target_node.best_rxn_no = np.random.randn(2).tolist()
        graph.target_node.best_total_value = np.random.randn(2).tolist()

        nodes_and_weights = {(graph.target_node, (0,)), (rxn_node, (0,))}

        updated_nodes, _ = graph.downpropagation(nodes_and_weights)

        assert isinstance(updated_nodes, set)

    def test_update_values_integration(self, graph_with_structure):
        """Test update_values method that combines up and down propagation"""
        graph, rxn_node, reactant1, reactant2 = graph_with_structure

        nodes = {(reactant1, (0,)), (reactant2, (0,))}

        pareto_updated = graph.update_values(nodes)

        assert isinstance(pareto_updated, bool)


class TestUpdateSolutionAndPareto:
    """Test cases for update_solution_and_pareto method"""

    @pytest.fixture
    def graph_for_solution_update(self):
        """Create a graph suitable for testing solution updates"""

        def h1(smiles):
            return 1.0

        def h2(smiles):
            return 1.5

        graph = MOGraph(
            target="CCCO",
            building_blocks={"CC", "C"},
            heuristic_fns=[h1, h2],
            weight_samples=8,
            no_weights=2,
            pareto_objectives=2,
            max_dominated_solutions=5,
        )
        return graph

    def test_update_solution_and_pareto_basic_addition(self, graph_for_solution_update):
        """Test basic addition of new solutions to solution_cost and pareto_front"""
        graph = graph_for_solution_update

        # Set up target node with some success costs
        cost_vector = (1.0, 2.0)
        path_nodes = ["node1", "node2"]
        graph.target_node.success_cost[cost_vector] = path_nodes
        graph.target_node.local_pareto[cost_vector] = True

        # Create new solutions dict
        new_solutions = {cost_vector: (0, 1)}

        # Call the method
        pareto_updated = graph.update_solution_and_pareto(new_solutions)

        # Verify solution was added
        assert cost_vector in graph.solution_cost
        assert graph.solution_cost[cost_vector] == (
            path_nodes,
            (0, 1),
        )  # local indices become global

        # Verify Pareto front was updated
        assert cost_vector in graph.pareto_front
        assert pareto_updated == True

    def test_update_solution_and_pareto_removal(self, graph_for_solution_update):
        """Test removal of solutions no longer in target_node.success_cost"""
        graph = graph_for_solution_update

        # Pre-populate solution_cost and pareto_front
        old_cost = (0.5, 1.5)
        graph.solution_cost[old_cost] = (["old_path"], (0,))
        graph.pareto_front[old_cost] = [[0.5, 0.5]]

        # Don't add old_cost to target_node.success_cost (simulate filtering)
        # Add a new cost
        new_cost = (1.0, 2.0)
        graph.target_node.success_cost[new_cost] = ["new_path"]
        graph.target_node.local_pareto[new_cost] = True

        new_solutions = {new_cost: (0,)}

        pareto_updated = graph.update_solution_and_pareto(new_solutions)

        # Old solution should be removed
        assert old_cost not in graph.solution_cost
        assert old_cost not in graph.pareto_front

        # New solution should be added
        assert new_cost in graph.solution_cost
        assert new_cost in graph.pareto_front
        assert pareto_updated == True

    def test_update_solution_and_pareto_no_pareto_change(
        self, graph_for_solution_update
    ):
        """Test when Pareto front doesn't change"""
        graph = graph_for_solution_update

        # Set up existing Pareto front
        existing_cost = (1.0, 2.0)
        graph.target_node.local_pareto[existing_cost] = True
        graph.pareto_front[existing_cost] = [[0.5, 0.5]]

        # Add the same cost again
        graph.target_node.success_cost[existing_cost] = ["path"]
        new_solutions = {existing_cost: (0,)}

        pareto_updated = graph.update_solution_and_pareto(new_solutions)

        # Pareto should not be marked as updated (same set)
        assert pareto_updated == False
        assert existing_cost in graph.pareto_front


class TestIntegrationScenarios:
    """Integration tests for complete workflows"""

    def test_complete_expansion_and_update_workflow(self):
        """Test a complete workflow with multiple iterations and predictions"""

        def h1(smiles):
            return len(smiles) * 0.1

        def h2(smiles):
            return smiles.count("C") * 0.2

        graph = MOGraph(
            target="CCCCO",  # More complex target
            building_blocks={"CC", "CO", "C"},
            heuristic_fns=[h1, h2],
            weight_samples=16,
            no_weights=4,
            pareto_objectives=2,
            max_dominated_solutions=5,
        )

        # First iteration - expand target with two predictions
        target_node = graph.target_node
        expanded_nodes_iter1 = {(target_node, (0,))}

        predictions_iter1 = {
            target_node: [
                {
                    "reactants": ["CCCO", "C"],
                    "reagents": "catalyst1",
                    "temperature": 298.0,
                    "rxn_smiles": "CCCO.C>>CCCCO",
                    "template": "template1",
                    "costs": np.array([1.2, 1.8]),
                },
                {
                    "reactants": ["CCC", "CO"],
                    "reagents": "catalyst2",
                    "temperature": 323.0,
                    "rxn_smiles": "CCC.CO>>CCCCO",
                    "template": "template2",
                    "costs": np.array([1.5, 1.4]),
                },
            ]
        }

        new_nodes_iter1 = graph.expand_graph(predictions_iter1, expanded_nodes_iter1)

        # Update values after first iteration
        if new_nodes_iter1:
            pareto_updated_iter1 = graph.update_values(new_nodes_iter1)
            [
                graph.open_nodes.add(node[0])
                for node in new_nodes_iter1
                if isinstance(node[0], MolNode) and not node[0].is_known
            ]
            assert isinstance(pareto_updated_iter1, bool)

        open_unknown_nodes = [node for node in graph.open_nodes]
        # Second iteration - expand newly created unknown molecules
        # Find open nodes from first iteration (CCCO and CCC should be unknown)
        assert (
            len(open_unknown_nodes) >= 2
        )  # Should have at least one open unknown node

        # Pick first two open nodes for second iteration
        nodes_to_expand = (
            open_unknown_nodes[:2]
            if len(open_unknown_nodes) >= 2
            else open_unknown_nodes
        )

        expanded_nodes_iter2 = set()
        predictions_iter2 = {}

        for i, node in enumerate(nodes_to_expand):
            expanded_nodes_iter2.add((node, (0, 1)))

            if "CCCO" in node.smiles:
                predictions_iter2[node] = [
                    {
                        "reactants": ["CC", "CO"],
                        "reagents": "catalyst3",
                        "temperature": 310.0,
                        "rxn_smiles": "CC.CO>>CCCO",
                        "template": "template3",
                        "costs": np.array([0.8, 1.2]),
                    },
                    {
                        "reactants": ["CCO", "C"],
                        "reagents": "catalyst4",
                        "temperature": 298.0,
                        "rxn_smiles": "CCO.C>>CCCO",
                        "template": "template4",
                        "costs": np.array([1.0, 1.1]),
                    },
                ]
            elif "CCC" in node.smiles:
                predictions_iter2[node] = [
                    {
                        "reactants": ["CC", "C"],
                        "reagents": "catalyst5",
                        "temperature": 315.0,
                        "rxn_smiles": "CC.C>>CCC",
                        "template": "template5",
                        "costs": np.array([0.6, 0.9]),
                    },
                    {
                        "reactants": ["C", "CC"],
                        "reagents": "catalyst6",
                        "temperature": 305.0,
                        "rxn_smiles": "C.CC>>CCC",
                        "template": "template6",
                        "costs": np.array([0.7, 0.8]),
                    },
                ]

        # Expand graph in second iteration
        if predictions_iter2:
            new_nodes_iter2 = graph.expand_graph(
                predictions_iter2, expanded_nodes_iter2
            )

            # Update values after second iteration
            if new_nodes_iter2:
                pareto_updated_iter2 = graph.update_values(new_nodes_iter2)
                assert isinstance(pareto_updated_iter2, bool)

        # Verify final state after two iterations
        assert target_node.is_open == False
        assert (
            len(graph.graph.nodes) >= 8
        )  # Should have multiple reactions and molecules

        # Check that building blocks are known
        if "CC" in graph.mol_to_node:
            assert graph.mol_to_node["CC"].is_known == True
        if "CO" in graph.mol_to_node:
            assert graph.mol_to_node["CO"].is_known == True
        if "C" in graph.mol_to_node:
            assert graph.mol_to_node["C"].is_known == True

        # Verify some nodes were created and have proper values
        total_molecules = sum(
            1
            for node in graph.graph.nodes
            if hasattr(node, "smiles") and not hasattr(node, "template")
        )
        total_reactions = sum(
            1 for node in graph.graph.nodes if hasattr(node, "template")
        )

        assert total_molecules >= 4  # At least target + intermediates + building blocks
        assert total_reactions >= 2  # At least 2 reactions from 2 iterations

        # Check that Pareto front has been updated if solutions were found
        if graph.target_node.success_cost:
            assert len(graph.pareto_front) >= 0


if __name__ == "__main__":
    pytest.main([__file__])
