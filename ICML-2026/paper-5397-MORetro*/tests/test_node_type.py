"""
Unit tests for MolNode and RxnNode classes in node_type.py

Tests cover core functionalities including:
- Node initialization and properties
- Value propagation (up and down)
- Success cost tracking
- Heuristic calculations
- Path management
"""

import logging
import os
import sys
from types import SimpleNamespace
from typing import cast
from unittest.mock import Mock

import numpy as np
import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from moretro.search.node_type import (
    MolNode,
    PathCost,
    RxnNode,
    filter_pareto_with_dominated,
    zero_vector,
)


class TestMolNode:
    """Test cases for MolNode class"""

    @pytest.fixture
    def heuristic_fns(self):
        """Create mock heuristic functions"""

        def h1(smiles):
            return 1.0

        def h2(smiles):
            return 2.0

        return [h1, h2]

    @pytest.fixture
    def simple_mol_node(self, heuristic_fns):
        """Create a simple MolNode for testing"""
        return MolNode(
            smiles="CCO",
            heuristic_fns=heuristic_fns,
            depth=1,
            is_known=False,
            pareto_objectives=2,
            max_dominated_solutions=5,
        )

    @pytest.fixture
    def known_mol_node(self, heuristic_fns):
        """Create a known (building block) MolNode"""
        return MolNode(
            smiles="CC",
            heuristic_fns=heuristic_fns,
            depth=2,
            is_known=True,
            pareto_objectives=2,
            max_dominated_solutions=5,
        )

    def test_initialization_basic(self, heuristic_fns):
        """Test basic MolNode initialization"""
        node = MolNode(
            smiles="CCO",
            heuristic_fns=heuristic_fns,
            depth=1,
            is_known=False,
            pareto_objectives=2,
            max_dominated_solutions=5,
        )

        assert node.smiles == "CCO"
        assert node.depth == 1
        assert not node.is_known
        assert node.is_open
        assert not node.success
        assert node.h_length == 2
        np.testing.assert_array_equal(node.value_estimates, np.array([1.0, 2.0]))
        assert node.success_cost == {}
        # Check best_rxn_no and best_total_value initialization
        np.testing.assert_array_equal(node.best_rxn_no, np.array([]))
        np.testing.assert_array_equal(node.best_total_value, np.array([]))

    def test_known_molecule_initialization(self, heuristic_fns):
        """Test initialization of known building block"""
        node = MolNode(
            smiles="CC",
            heuristic_fns=heuristic_fns,
            depth=0,
            is_known=True,
            zero_bound=True,
            pareto_objectives=2,
            max_dominated_solutions=5,
        )

        assert not node.is_open  # Known molecules are not open for expansion
        np.testing.assert_array_equal(
            node.success_cost_estimate, np.array([0.0, 0.0])
        )  # Zero bound

    def test_known_molecule_no_zero_bound(self, heuristic_fns):
        """Test known molecule without zero bound"""
        node = MolNode(
            smiles="CC",
            heuristic_fns=heuristic_fns,
            depth=0,
            is_known=True,
            zero_bound=False,
            pareto_objectives=2,
            max_dominated_solutions=5,
        )

        np.testing.assert_array_equal(
            node.success_cost_estimate, np.array([1.0, 2.0])
        )  # Heuristic values

    def test_objectives_to_scalar(self, simple_mol_node):
        """Test conversion of objectives to scalar values"""
        weights = np.array([[0.5, 0.5], [1.0, 0.0]])
        result = simple_mol_node.objectives_to_scalar(weights)
        expected = np.array([1.5, 1.0])  # [0.5*1.0 + 0.5*2.0, 1.0*1.0 + 0.0*2.0]
        np.testing.assert_array_equal(result, expected)

    def test_uppropagate_known_molecule(self, known_mol_node):
        """Test uppropagation for known building block"""
        weights = np.array([[0.6, 0.4], [0.3, 0.7]])

        result = known_mol_node.uppropagate([], weights, False)

        assert result  # Should return True for changes
        assert known_mol_node.success
        np.testing.assert_array_equal(
            known_mol_node.rxn_no, np.array([0.0, 0.0])
        )  # Zero bound
        np.testing.assert_array_equal(
            known_mol_node.best_rxn_no, np.array([0.0, 0.0])
        )  # Zero bound for best costs
        assert (
            tuple(known_mol_node.success_cost_estimate) in known_mol_node.success_cost
        )

    def test_uppropagate_open_node(self, simple_mol_node):
        """Test uppropagation for open (leaf) node"""
        weights = np.array([[0.5, 0.5]])

        result = simple_mol_node.uppropagate([], weights, False)

        assert result
        assert (
            not simple_mol_node.success
        )  # Open node without children is not successful
        np.testing.assert_array_equal(
            simple_mol_node.rxn_no, np.array([1.5])
        )  # 0.5*1.0 + 0.5*2.0

    def test_uppropagate_with_children(self, simple_mol_node):
        """Test uppropagation with successful children"""
        # Create mock children
        child1 = Mock(spec=RxnNode)
        child1.rxn_no = np.array([2.0, 3.0])
        child1.best_rxn_no = np.array([1.5, 2.5])  # Add best_rxn_no to mock
        child1.success = True
        child1.success_cost = {(1.0, 1.5): ["path1"]}

        child2 = Mock(spec=RxnNode)
        child2.rxn_no = np.array([1.5, 4.0])
        child2.best_rxn_no = np.array([1.0, 3.0])  # Add best_rxn_no to mock
        child2.success = False
        child2.success_cost = {}

        simple_mol_node.is_open = False
        weights = np.array([[1.0, 0.0], [0.0, 1.0]])

        result = simple_mol_node.uppropagate([child1, child2], weights, False)

        assert result
        assert simple_mol_node.success  # At least one child is successful
        np.testing.assert_array_equal(
            simple_mol_node.rxn_no, np.array([1.5, 3.0])
        )  # Min of children
        np.testing.assert_array_equal(simple_mol_node.best_rxn_no, np.array([1.0, 2.5]))

    def test_downpropagate_no_parents(self, simple_mol_node):
        """Test downpropagation for node with no parents"""
        simple_mol_node.rxn_no = np.array([2.0, 3.0])
        simple_mol_node.best_rxn_no = np.array([1.5, 2.5])

        result = simple_mol_node.downpropagate([])

        assert result
        np.testing.assert_array_equal(simple_mol_node.total_value, np.array([2.0, 3.0]))
        np.testing.assert_array_equal(
            simple_mol_node.best_total_value, np.array([1.5, 2.5])
        )

    def test_downpropagate_with_parents(self, simple_mol_node):
        """Test downpropagation with parent reactions"""
        parent1 = Mock(spec=RxnNode)
        parent1.total_value = np.array([1.0, 2.0])
        parent1.best_total_value = np.array([0.8, 1.5])
        parent1.linear_cost = np.array([0.0, 0.0])

        parent2 = Mock(spec=RxnNode)
        parent2.total_value = np.array([0.5, 3.0])
        parent2.best_total_value = np.array([0.3, 2.5])
        parent2.linear_cost = np.array([0.0, 0.0])

        result = simple_mol_node.downpropagate([parent1, parent2])

        assert result
        np.testing.assert_array_equal(
            simple_mol_node.total_value, np.array([0.5, 2.0])
        )  # Min of parents
        np.testing.assert_array_equal(
            simple_mol_node.best_total_value, np.array([0.3, 1.5])
        )  # Min of parents' best_total_value

    def test_track_success_cost_reaction_grouping(self, simple_mol_node):
        """Test that track_success_cost groups reactions by SMILES and keeps lowest cost"""
        # Create children with same reaction but different costs
        child1 = Mock(spec=RxnNode)
        child1.smiles = "CC.O>>CCO"
        node1 = SimpleNamespace(smiles="CC")
        node2 = SimpleNamespace(smiles="O")
        child1.success_cost = {(2.0, 3.0): [node1, node2, child1]}  # sum = 5.0

        child2 = Mock(spec=RxnNode)
        child2.smiles = "CC.O>>CCO"  # Same reaction
        child2.success_cost = {(1.5, 4.0): [node1, node2, child2]}  # sum = 5.5

        simple_mol_node.success_cost = {}

        result = simple_mol_node.track_success_cost([child1, child2])

        # Should keep only the path with lowest total cost sum (2.0, 3.0) has sum 5.0 < 5.5
        assert len(result) == 1
        assert (2.0, 3.0) in result
        path = result[(2.0, 3.0)]
        assert path[-2] is child1  # The child with lower cost sum
        assert path[-1] is simple_mol_node

    def test_track_success_cost_reactant_sorting(self, simple_mol_node):
        """Test that reactants are sorted by length for grouping"""
        child = Mock(spec=RxnNode)
        child.smiles = "O.CC>>CCO"  # Reactants in different order
        node1 = SimpleNamespace(smiles="O")
        node2 = SimpleNamespace(smiles="CC")
        child.success_cost = {(1.0, 2.0): [node1, node2, child]}

        simple_mol_node.success_cost = {}

        result = simple_mol_node.track_success_cost([child])

        assert len(result) == 1
        # Should be grouped under sorted reactants "CC.O>>CCO"

    def test_track_success_cost_pareto_filtering(self, simple_mol_node):
        """Test Pareto filtering keeps optimal and limited dominated solutions"""
        # Create multiple children with different costs
        child1 = Mock(spec=RxnNode)
        child1.smiles = "CC.O>>CCO"
        node1 = SimpleNamespace(smiles="CC")
        node2 = SimpleNamespace(smiles="O")
        child1.success_cost = {(1.0, 3.0): [node1, node2, child1]}  # Pareto optimal

        child2 = Mock(spec=RxnNode)
        child2.smiles = "C.CO>>CCO"
        node3 = SimpleNamespace(smiles="C")
        node4 = SimpleNamespace(smiles="CO")
        child2.success_cost = {(2.0, 2.0): [node3, node4, child2]}  # Pareto optimal

        child3 = Mock(spec=RxnNode)
        child3.smiles = "CC.O>>CCO"
        child3.success_cost = {(2.0, 4.0): [node1, node2, child3]}  # Dominated

        simple_mol_node.success_cost = {}

        result = simple_mol_node.track_success_cost([child1, child2, child3])

        # Should keep both Pareto optimal and 1 dominated (max_dominated_solutions=5)
        assert len(result) >= 2
        assert (1.0, 3.0) in result
        assert (2.0, 2.0) in result

    def test_track_success_cost_already_checked_filtering(self, simple_mol_node):
        """Test that already checked costs are filtered out"""
        # Pre-populate successor_cost_already_checked
        simple_mol_node.successor_cost_already_checked.add((1.0, 2.0))

        child = Mock(spec=RxnNode)
        child.smiles = "CC.O>>CCO"
        node1 = SimpleNamespace(smiles="CC")
        node2 = SimpleNamespace(smiles="O")
        child.success_cost = {(1.0, 2.0): [node1, node2, child]}  # Already checked

        simple_mol_node.success_cost = {}

        result = simple_mol_node.track_success_cost([child])

        # Should not include the already checked cost
        assert len(result) == 0

    def test_track_success_cost_local_pareto_update(self, simple_mol_node):
        """Test that local_pareto is updated with Pareto optimal solutions"""
        child = Mock(spec=RxnNode)
        child.smiles = "CC.O>>CCO"
        node1 = SimpleNamespace(smiles="CC")
        node2 = SimpleNamespace(smiles="O")
        child.success_cost = {(1.0, 3.0): [node1, node2, child]}

        simple_mol_node.success_cost = {}

        result = simple_mol_node.track_success_cost([child])

        # local_pareto should contain the Pareto optimal solution
        assert (1.0, 3.0) in simple_mol_node.local_pareto

    def test_hash_consistency(self, simple_mol_node):
        """Test that hash is consistent (uses object id)"""
        hash1 = hash(simple_mol_node)
        hash2 = hash(simple_mol_node)
        assert hash1 == hash2
        assert hash1 == id(simple_mol_node)


class TestRxnNode:
    """Test cases for RxnNode class"""

    @pytest.fixture
    def simple_rxn_node(self):
        """Create a simple RxnNode for testing"""
        return RxnNode(
            smiles="CCO>>CC.O",
            template="[C:1]-[O:2]>>[C:1].[O:2]",
            reagents=["H2SO4"],
            temp=350.0,
            depth=2,
            cost=np.array([1.0, 2.0]),
            weight_length=2,
            pareto_objectives=2,
            max_dominated_solutions=5,
        )

    def test_initialization_basic(self):
        """Test basic RxnNode initialization"""
        node = RxnNode(
            smiles="CC>>C.C",
            template="[C:1]-[C:2]>>[C:1].[C:2]",
            reagents=["heat"],
            temp=400.0,
            depth=1,
            cost=np.array([0.5, 1.5]),
            weight_length=3,
            pareto_objectives=2,
            max_dominated_solutions=5,
        )

        assert node.smiles == "CC>>C.C"
        assert node.template == "[C:1]-[C:2]>>[C:1].[C:2]"
        assert node.reagents == ["heat"]
        assert node.temp == 400.0
        assert node.depth == 1
        assert len(node.cost) == 2
        assert len(node.rxn_no) == 3
        assert len(node.total_value) == 3
        assert not node.success
        # Check best_rxn_no and best_total_value initialization
        np.testing.assert_array_equal(
            node.best_rxn_no, np.array([0 for _ in range(len(node.cost))])
        )
        np.testing.assert_array_equal(node.best_total_value, np.array([]))

    def test_initialization_empty_cost(self):
        """Test that empty cost raises ValueError"""
        with pytest.raises(ValueError, match="Reaction cost must be provided"):
            RxnNode(
                smiles="CC>>C.C",
                template="[C:1]-[C:2]>>[C:1].[C:2]",
                reagents=["heat"],
                temp=400.0,
                depth=1,
                cost=np.array([]),
                weight_length=2,
                pareto_objectives=2,
                max_dominated_solutions=5,
            )

    def test_delta_offset_uniqueness(self):
        """Test that delta offset makes costs unique"""
        node1 = RxnNode(
            smiles="CC>>C.C",
            template="template1",
            reagents=["reagent1"],
            temp=300.0,
            depth=1,
            cost=np.array([1.0, 2.0]),
            weight_length=2,
            pareto_objectives=2,
            max_dominated_solutions=5,
        )

        node2 = RxnNode(
            smiles="CC>>C.C",
            template="template1",
            reagents=["reagent2"],  # different reagent
            temp=300.0,
            depth=1,
            cost=np.array([1.0, 2.0]),
            weight_length=2,
            pareto_objectives=2,
            max_dominated_solutions=5,
        )

        # Costs should be different due to delta offset
        assert not np.array_equal(node1.cost, node2.cost)
        np.testing.assert_array_equal(node1.true_cost, np.array([1.0, 2.0]))
        np.testing.assert_array_equal(node2.true_cost, np.array([1.0, 2.0]))

    def test_uppropagate_successful_children(self, simple_rxn_node):
        """Test uppropagation with all successful children"""
        child1 = Mock(spec=MolNode)
        child1.rxn_no = np.array([1.0, 2.0])
        child1.best_rxn_no = np.array([0.8, 1.5])
        child1.success = True
        child1.success_cost = {(0.5, 1.0): ["mol1"]}

        child2 = Mock(spec=MolNode)
        child2.rxn_no = np.array([2.0, 1.0])
        child2.best_rxn_no = np.array([1.5, 0.7])
        child2.success = True
        child2.success_cost = {(1.0, 0.5): ["mol2"]}

        weights = np.array([[1.0, 0.0], [0.0, 1.0]])

        result = simple_rxn_node.uppropagate([child1, child2], weights, False)

        assert result
        assert simple_rxn_node.success
        # Expected: child1 + child2 + reaction cost
        # [1.0, 2.0] + [2.0, 1.0] + weights @ [1.0, 2.0] = [3.0, 3.0] + [1.0, 2.0] = [4.0, 5.0]
        expected_rxn_no = np.array([4.0, 5.0])
        np.testing.assert_array_almost_equal(
            simple_rxn_node.rxn_no, expected_rxn_no, decimal=10
        )
        # best_rxn_no: sum of children's best_rxn_no + reaction cost
        # [0.8, 1.5] + [1.5, 0.7] + [1.0, 2.0] = [3.3, 4.2]
        expected_best_rxn_no = np.array([3.3, 4.2])
        np.testing.assert_array_almost_equal(
            simple_rxn_node.best_rxn_no, expected_best_rxn_no, decimal=10
        )

    def test_uppropagate_unsuccessful_children(self, simple_rxn_node):
        """Test uppropagation with unsuccessful children"""
        child1 = Mock(spec=MolNode)
        child1.rxn_no = np.array([1.0, 2.0])
        child1.best_rxn_no = np.array([0.8, 1.5])
        child1.success = True
        child1.success_cost = {(0.5, 1.0): ["mol1"]}

        child2 = Mock(spec=MolNode)
        child2.rxn_no = np.array([2.0, 1.0])
        child2.best_rxn_no = np.array([1.5, 0.7])
        child2.success = False  # This child is not successful
        child2.success_cost = {}

        weights = np.array([[1.0, 0.0], [0.0, 1.0]])

        result = simple_rxn_node.uppropagate([child1, child2], weights, False)

        assert result
        assert not simple_rxn_node.success  # Not all children are successful

    def test_uppropagate_no_children(self, simple_rxn_node):
        """Test that uppropagation with no children raises error"""
        weights = np.array([[1.0, 0.0]])

        with pytest.raises(ValueError, match="No children provided for RxnNode"):
            simple_rxn_node.uppropagate([], weights, False)

    def test_downpropagate(self, simple_rxn_node):
        """Test downpropagation from parent molecule"""
        parent = Mock(spec=MolNode)
        parent.rxn_no = np.array([3.0, 4.0])
        parent.total_value = np.array([1.0, 1.5])
        parent.best_rxn_no = np.array([2.5, 3.5])
        parent.best_total_value = np.array([0.8, 1.2])

        simple_rxn_node.rxn_no = np.array([5.0, 6.0])
        simple_rxn_node.best_rxn_no = np.array([4.5, 5.5])

        result = simple_rxn_node.downpropagate(parent)

        assert result
        # Expected: rxn_no - parent.rxn_no + parent.total_value
        # [5.0, 6.0] - [3.0, 4.0] + [1.0, 1.5] = [3.0, 3.5]
        np.testing.assert_array_equal(simple_rxn_node.total_value, np.array([3.0, 3.5]))
        # Expected: best_rxn_no - parent.best_rxn_no + parent.best_total_value
        # [4.5, 5.5] - [2.5, 3.5] + [0.8, 1.2] = [2.8, 3.2]
        np.testing.assert_array_almost_equal(
            simple_rxn_node.best_total_value, np.array([2.8, 3.2])
        )

    def test_track_success_cost_cost_combination(self, simple_rxn_node):
        """Test that track_success_cost correctly combines costs from children"""
        child1 = Mock(spec=MolNode)
        child1.success_cost = {(1.0, 2.0): ["mol1a"], (1.5, 1.8): ["mol1b"]}

        child2 = Mock(spec=MolNode)
        child2.success_cost = {(0.5, 1.0): ["mol2"]}

        simple_rxn_node.success_cost = {}

        result = simple_rxn_node.track_success_cost([child1, child2])

        # Should have 2 combinations: (1.0,2.0)+(0.5,1.0) and (1.5,1.8)+(0.5,1.0)
        assert len(result) == 2

        # Check that costs are summed correctly (plus reaction cost)
        costs = list(result.keys())
        # Costs should be approximately (1.5, 3.0) and (2.0, 2.8) plus reaction cost
        # But due to delta offset, we just check the structure
        for cost in costs:
            assert len(cost) == 2
            path = result[cost]
            assert "mol1a" in path or "mol1b" in path
            assert "mol2" in path
            assert simple_rxn_node in path

    def test_track_success_cost_pareto_filtering_rxn(self, simple_rxn_node):
        """Test Pareto filtering in RxnNode track_success_cost"""
        child1 = Mock(spec=MolNode)
        child1.success_cost = {(1.0, 3.0): ["mol1"]}

        child2 = Mock(spec=MolNode)
        child2.success_cost = {(2.0, 1.0): ["mol2"]}

        simple_rxn_node.success_cost = {}

        result = simple_rxn_node.track_success_cost([child1, child2])

        # Should have at least one result from the combinations
        assert len(result) > 0
        # Verify that paths contain the expected elements
        for path in result.values():
            assert simple_rxn_node in path

    def test_track_success_cost_empty_children(self, simple_rxn_node):
        """Test track_success_cost with children having empty success_cost"""
        child1 = Mock(spec=MolNode)
        child1.success_cost = {}  # Empty

        child2 = Mock(spec=MolNode)
        child2.success_cost = {(1.0, 2.0): ["mol2"]}

        simple_rxn_node.success_cost = {}

        result = simple_rxn_node.track_success_cost([child1, child2])

        # Should only process valid combinations
        assert len(result) == 0  # No valid combinations if one child is empty

    def test_track_success_cost_single_child(self, simple_rxn_node):
        """Test track_success_cost with single child"""
        child = Mock(spec=MolNode)
        child.success_cost = {(1.0, 2.0): ["mol1"]}

        simple_rxn_node.success_cost = {}

        result = simple_rxn_node.track_success_cost([child])

        # Should have one combination: child cost + reaction cost
        assert len(result) == 1
        cost = list(result.keys())[0]
        path = result[cost]
        assert "mol1" in path
        assert simple_rxn_node in path

    def test_track_success_cost_no_change_when_same(self, simple_rxn_node):
        """Test that track_success_cost returns empty dict when no new solutions"""
        child = Mock(spec=MolNode)
        child.success_cost = {(1.0, 2.0): ["mol1"]}

        # First call should add solutions
        result1 = simple_rxn_node.track_success_cost([child])
        assert len(result1) > 0

        # Update success_cost with the result
        simple_rxn_node.success_cost.update(result1)

        # Second call with same input should return empty (no changes)
        result2 = simple_rxn_node.track_success_cost([child])
        assert result2 == {}

    def test_hash_consistency(self, simple_rxn_node):
        """Test that hash is consistent (uses object id)"""
        hash1 = hash(simple_rxn_node)
        hash2 = hash(simple_rxn_node)
        assert hash1 == hash2
        assert hash1 == id(simple_rxn_node)


class TestUtilityFunctions:
    """Test utility functions"""

    def test_zero_vector(self):
        """Test zero vector creation"""
        result = zero_vector(3)
        assert np.array_equal(result, np.array([0.0, 0.0, 0.0]))

        result = zero_vector(0)
        assert np.array_equal(result, np.array([]))

    def test_filter_pareto_with_dominated_empty(self):
        """Test filter_pareto_with_dominated with empty solutions"""
        node = MolNode(
            smiles="CC",
            heuristic_fns=[lambda x: 1.0],
            depth=0,
            is_known=True,
            pareto_objectives=1,
            max_dominated_solutions=5,
        )
        result = filter_pareto_with_dominated(node, {})
        assert result == {}
        assert node.local_pareto == {}

    def test_filter_pareto_with_dominated_single_solution(self):
        """Test filter_pareto_with_dominated with single solution"""
        node = MolNode(
            smiles="CC",
            heuristic_fns=[lambda x: 1.0],
            depth=0,
            is_known=True,
            pareto_objectives=1,
            max_dominated_solutions=5,
        )
        mock_node = SimpleNamespace(smiles="test")
        solutions = cast(PathCost, {(1.0, 2.0): [mock_node]})
        result = filter_pareto_with_dominated(node, solutions)
        assert result == solutions
        assert node.local_pareto == {(1.0, 2.0): [mock_node]}

    def test_filter_pareto_with_dominated_pareto_only(self):
        """Test filter_pareto_with_dominated with Pareto-optimal solutions only"""
        node = MolNode(
            smiles="CC",
            heuristic_fns=[lambda x: 1.0],
            depth=0,
            is_known=True,
            pareto_objectives=1,
            max_dominated_solutions=5,
        )
        # All solutions are Pareto-optimal (no one dominates another)
        mock_node1 = SimpleNamespace(smiles="test1")
        mock_node2 = SimpleNamespace(smiles="test2")
        mock_node3 = SimpleNamespace(smiles="test3")
        solutions = cast(
            PathCost,
            {
                (1.0, 3.0): [mock_node1],  # Pareto-optimal
                (2.0, 2.0): [mock_node2],  # Pareto-optimal
                (3.0, 1.0): [mock_node3],  # Pareto-optimal
            },
        )
        result = filter_pareto_with_dominated(node, solutions)
        assert result == solutions
        assert len(node.local_pareto) == 3

    def test_filter_pareto_with_dominated_with_dominated(self):
        """Test filter_pareto_with_dominated with some dominated solutions"""
        node = MolNode(
            smiles="CC",
            heuristic_fns=[lambda x: 1.0],
            depth=0,
            is_known=True,
            pareto_objectives=1,
            max_dominated_solutions=5,
        )
        mock_node1 = SimpleNamespace(smiles="test1")
        mock_node2 = SimpleNamespace(smiles="test2")
        mock_node3 = SimpleNamespace(smiles="test3")
        solutions = cast(
            PathCost,
            {
                (1.0, 1.0): [mock_node1],  # Pareto-optimal
                (2.0, 2.0): [mock_node2],  # Dominated by (1.0, 1.0)
                (3.0, 3.0): [mock_node3],  # Dominated by (1.0, 1.0) and (2.0, 2.0)
            },
        )
        result = filter_pareto_with_dominated(node, solutions, max_dominated=1)
        # Should keep Pareto-optimal + 1 dominated (the best one: (2.0, 2.0) with sum=4.0)
        expected = {
            (1.0, 1.0): [mock_node1],
            (2.0, 2.0): [mock_node2],
        }
        assert result == expected
        assert node.local_pareto == {(1.0, 1.0): [mock_node1]}

    def test_filter_pareto_with_dominated_max_dominated_zero(self):
        """Test filter_pareto_with_dominated with max_dominated=0"""
        node = MolNode(
            smiles="CC",
            heuristic_fns=[lambda x: 1.0],
            depth=0,
            is_known=True,
            pareto_objectives=1,
            max_dominated_solutions=5,
        )
        mock_node1 = SimpleNamespace(smiles="test1")
        mock_node2 = SimpleNamespace(smiles="test2")
        mock_node3 = SimpleNamespace(smiles="test3")
        solutions = cast(
            PathCost,
            {
                (1.0, 1.0): [mock_node1],  # Pareto-optimal
                (2.0, 2.0): [mock_node2],  # Dominated
                (3.0, 3.0): [mock_node3],  # Dominated
            },
        )
        result = filter_pareto_with_dominated(node, solutions, max_dominated=0)
        # Should keep only Pareto-optimal
        expected = {(1.0, 1.0): [mock_node1]}
        assert result == expected
        assert node.local_pareto == {(1.0, 1.0): [mock_node1]}

    def test_filter_pareto_with_dominated_multiple_dominated(self):
        """Test filter_pareto_with_dominated with multiple dominated solutions"""
        node = MolNode(
            smiles="CC",
            heuristic_fns=[lambda x: 1.0],
            depth=0,
            is_known=True,
            pareto_objectives=1,
            max_dominated_solutions=5,
        )
        mock_node1 = SimpleNamespace(smiles="test1")
        mock_node2 = SimpleNamespace(smiles="test2")
        mock_node3 = SimpleNamespace(smiles="test3")
        mock_node4 = SimpleNamespace(smiles="test4")
        solutions = cast(
            PathCost,
            {
                (1.0, 1.0): [mock_node1],  # Pareto-optimal
                (2.0, 2.0): [mock_node2],  # Dominated, sum=4.0
                (3.0, 1.5): [mock_node3],  # Dominated, sum=4.5
                (4.0, 1.0): [mock_node4],  # Dominated, sum=5.0
            },
        )
        result = filter_pareto_with_dominated(node, solutions, max_dominated=2)
        # Should keep Pareto-optimal + 2 best dominated (lowest sum: (2.0, 2.0) and (3.0, 1.5))
        expected = {
            (1.0, 1.0): [mock_node1],
            (2.0, 2.0): [mock_node2],
            (3.0, 1.5): [mock_node3],
        }
        assert result == expected
        assert node.local_pareto == {(1.0, 1.0): [mock_node1]}


class TestIntegration:
    """Integration tests for MolNode and RxnNode interaction"""

    @pytest.fixture
    def heuristic_fns(self):
        def h1(smiles):
            return 1.0

        def h2(smiles):
            return 2.0

        return [h1, h2]

    def test_mol_rxn_propagation_cycle(self, heuristic_fns):
        """Test a complete propagation cycle between MolNode and RxnNode"""
        # Create a known starting material
        bb_node = MolNode(
            smiles="CC",
            heuristic_fns=heuristic_fns,
            depth=2,
            is_known=True,
            pareto_objectives=2,
            max_dominated_solutions=5,
        )  # Create a reaction that uses this building block
        rxn_node = RxnNode(
            smiles="CCO>>CC.O",  # Use valid reaction SMILES
            template="template",
            reagents=["reagent"],
            temp=300.0,
            depth=1,
            cost=np.array([0.5, 1.0]),
            weight_length=2,
            pareto_objectives=2,
            max_dominated_solutions=5,
        )

        # Create target molecule
        target_node = MolNode(
            smiles="CCO",  # Use valid SMILES
            heuristic_fns=heuristic_fns,
            depth=0,
            is_known=False,
            is_target=True,
            pareto_objectives=2,
            max_dominated_solutions=5,
        )
        weights = np.array([[1.0, 0.0], [0.0, 1.0]])

        # Simulate uppropagation from building block
        bb_updated = bb_node.uppropagate([], weights, False)
        assert all(bb_updated)
        assert bb_node.success

        # Propagate to reaction
        rxn_updated = rxn_node.uppropagate([bb_node], weights, False)
        assert all(rxn_updated)
        assert rxn_node.success

        # Mark target as not open since it has children (reaction)
        target_node.is_open = False

        # Propagate to target
        target_updated = target_node.uppropagate([rxn_node], weights, True)
        assert all(target_updated)
        assert target_node.success

        # Check that costs are properly tracked
        assert len(target_node.success_cost) > 0


class TestEdgeCases:
    """Test edge cases and critical scenarios"""

    @pytest.fixture
    def heuristic_fns(self):
        def h1(smiles):
            return 1.0

        def h2(smiles):
            return 2.0

        return [h1, h2]

    def test_mol_node_no_change_uppropagate(self, heuristic_fns):
        """Test uppropagate when no actual changes occur"""
        node = MolNode(
            smiles="CCO",
            heuristic_fns=heuristic_fns,
            depth=1,
            is_known=False,
            pareto_objectives=2,
            max_dominated_solutions=5,
        )

        # Set initial state
        node.rxn_no = np.array([1.5])
        node.best_rxn_no = np.array([1.0, 1.5])
        node.success = False

        weights = np.array([[0.5, 0.5]])  # Will produce same rxn_no = [1.5]

        result = node.uppropagate([], weights, False)
        assert not all(result)  # No changes should occur

    def test_rxn_node_duplicate_uppropagate_calls(self):
        """Test that duplicate RxnNode uppropagate calls behave correctly"""
        rxn_node = RxnNode(
            smiles="CCO>>CC.O",
            template="template",
            reagents=["reagent"],
            temp=300.0,
            depth=1,
            cost=np.array([1.0, 2.0]),
            weight_length=2,
            pareto_objectives=2,
            max_dominated_solutions=5,
        )

        # Create mock children
        child = Mock(spec=MolNode)
        child.rxn_no = np.array([1.0, 2.0])
        child.best_rxn_no = np.array([0.8, 1.5])
        child.success = True
        child.success_cost = {(0.5, 1.0): ["mol1"]}

        weights = np.array([[1.0, 0.0], [0.0, 1.0]])

        # First call should update
        result1 = rxn_node.uppropagate([child], weights, False)
        assert all(result1)

        # Second call with same parameters should not update
        result2 = rxn_node.uppropagate([child], weights, False)
        assert not all(result2)  # No changes

    def test_mol_node_downpropagate_no_change(self, heuristic_fns):
        """Test downpropagate when no changes occur"""
        node = MolNode(
            smiles="CCO",
            heuristic_fns=heuristic_fns,
            depth=1,
            is_known=False,
            pareto_objectives=2,
            max_dominated_solutions=5,
        )

        # Set initial total_value
        node.total_value = np.array([1.0, 2.0])
        node.best_total_value = np.array([0.8, 1.5])

        # Create parent with same total_value
        parent = Mock(spec=RxnNode)
        parent.total_value = np.array([1.0, 2.0])
        parent.best_total_value = np.array([0.8, 1.5])

        result = node.downpropagate([parent])
        assert not result  # No change should occur

    def test_rxn_node_downpropagate_no_change(self):
        """Test RxnNode downpropagate when no changes occur"""
        rxn_node = RxnNode(
            smiles="CCO>>CC.O",
            template="template",
            reagents=["reagent"],
            temp=300.0,
            depth=1,
            cost=np.array([1.0, 2.0]),
            weight_length=2,
            pareto_objectives=2,
            max_dominated_solutions=5,
        )

        # Set up state that would result in no change
        rxn_node.rxn_no = np.array([5.0, 6.0])
        rxn_node.best_rxn_no = np.array([4.5, 5.5])
        rxn_node.total_value = np.array([3.0, 3.5])
        rxn_node.best_total_value = np.array([2.8, 3.2])

        parent = Mock(spec=MolNode)
        parent.rxn_no = np.array([3.0, 4.0])
        parent.total_value = np.array([1.0, 1.5])
        parent.best_rxn_no = np.array([2.5, 3.5])
        parent.best_total_value = np.array([0.8, 1.2])

        # This should result in same total_value: [5.0, 6.0] - [3.0, 4.0] + [1.0, 1.5] = [3.0, 3.5]
        # And same best_total_value: [4.5, 5.5] - [2.5, 3.5] + [0.8, 1.2] = [2.8, 3.2]
        result = rxn_node.downpropagate(parent)
        assert not result  # No change

    def test_track_success_cost_existing_costs(self, heuristic_fns):
        """Test that existing costs in success_cost are not overwritten"""

        node = MolNode(
            smiles="CCO",
            heuristic_fns=heuristic_fns,
            depth=1,
            is_known=False,
            pareto_objectives=2,
            max_dominated_solutions=5,
        )

        # Pre-populate success_cost with mock node
        existing_node = Mock(spec=MolNode)
        node.success_cost = {(1.0, 2.0): [existing_node]}

        # Create child with same cost vector
        child = Mock(spec=RxnNode)
        new_node = Mock(spec=MolNode)
        child.success_cost = {(1.0, 2.0): [new_node]}

        result = node.track_success_cost([child])

        # Should not add duplicate cost
        assert len(result) == 0
        assert node.success_cost[(1.0, 2.0)] == [existing_node]  # Unchanged

    def test_success_cost_with_empty_successors(self, heuristic_fns):
        """Test success cost tracking with empty successors"""
        node = MolNode(
            smiles="CCO",
            heuristic_fns=heuristic_fns,
            depth=1,
            is_known=False,
            pareto_objectives=2,
            max_dominated_solutions=5,
        )

        # Create child with empty successor
        child = Mock(spec=RxnNode)
        child.success_cost = {(1.0, 2.0): []}  # Empty successor list

        result = node.track_success_cost([child])

        # Should filter out empty successors
        assert len(result) == 0

    def test_rxn_node_assert_error_empty_rxn_no(self):
        """Test assertion error when child has empty rxn_no"""
        rxn_node = RxnNode(
            smiles="CCO>>CC.O",
            template="template",
            reagents=["reagent"],
            temp=300.0,
            depth=1,
            cost=np.array([1.0, 2.0]),
            weight_length=2,
            pareto_objectives=2,
            max_dominated_solutions=5,
        )

        # Create child with empty rxn_no
        child = Mock(spec=MolNode)
        child.rxn_no = np.array([])  # Empty rxn_no
        child.success = True
        child.success_cost = {}

        weights = np.array([[1.0, 0.0]])

        with pytest.raises(
            AssertionError, match="Rxn_no for MolNode should not be empty"
        ):
            rxn_node.uppropagate([child], weights, False)

    @pytest.mark.skip(
        reason="Logging configuration interferes with caplog in test environment"
    )
    def test_mol_node_total_value_setter_warning(self, heuristic_fns, caplog):
        """Test that total_value setter logs warning for all-zero values"""

        # Temporarily disable console handler to ensure caplog captures the warning
        logger = logging.getLogger("moretro.search.node_type")
        original_handlers = logger.handlers[:]
        for handler in logger.handlers[:]:
            if isinstance(handler, logging.StreamHandler):
                logger.removeHandler(handler)

        try:
            node = MolNode(
                smiles="CCO",
                heuristic_fns=heuristic_fns,
                depth=1,
                is_known=False,
                is_target=False,
                pareto_objectives=2,
                max_dominated_solutions=5,
            )
            node.success = False  # Ensure warning condition is met

            with caplog.at_level(logging.WARNING, logger="moretro.search.node_type"):
                node.total_value = np.array([0.0, 0.0])

            assert "Total value of node CCO is all zeros" in caplog.text
        finally:
            # Restore original handlers
            for handler in original_handlers:
                logger.addHandler(handler)
