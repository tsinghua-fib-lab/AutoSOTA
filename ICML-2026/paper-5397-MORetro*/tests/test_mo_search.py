"""
Comprehensive unit tests for MOSearch class in mo_search.py

This test suite covers all methods in the MOSearch class including:
- Initialization and configuration
- Node expansion eligibility checking
- Retrosynthesis expansion
- Weight management and resampling
- Node selection for expansion
- Complete search integration
"""

import pytest
import numpy as np
import gin
import time
import logging
from unittest.mock import Mock, patch
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from moretro.search.mo_search import MOSearch
from moretro.search.node_type import MolNode, RxnNode
from moretro.inference.retro_prediction import OneStepModel
from moretro.search.mo_graph import MOGraph


class TestMOSearchInitialization:
    """Test cases for MOSearch initialization"""

    @pytest.fixture
    def mock_retro_model(self):
        """Create a mock retrosynthesis model"""
        mock_model = Mock(spec=OneStepModel)
        mock_model.predict.return_value = [
            [
                {
                    "reactants": ["CC", "CO"],
                    "reagents": ["catalyst"],
                    "temperature": 298.0,
                    "rxn_smiles": "CC.CO>>CCO",
                    "template": "[C:1][C:2].[C:3][OH:4]>>[C:1][C:2][C:3][OH:4]",
                    "costs": np.array([1.0, 1.5]),
                }
            ]
        ]
        return mock_model

    @pytest.fixture
    def simple_heuristics(self):
        """Create simple heuristic functions for testing"""

        def h1(smiles: str) -> float:
            return len(smiles) * 0.1

        def h2(smiles: str) -> float:
            return smiles.count("C") * 0.2

        return [h1, h2]

    @pytest.fixture
    def basic_building_blocks(self):
        """Basic set of building blocks for testing"""
        return {"CC", "CO", "C"}

    def setup_gin_config(self):
        """Setup minimal gin configuration for testing"""
        gin.clear_config()
        gin.bind_parameter("MOGraph.weight_samples", 16)
        gin.bind_parameter("MOGraph.no_weights", 4)
        gin.bind_parameter("MOGraph.weight_initial", "sobol")
        gin.bind_parameter("MOGraph.include_extreme", False)
        gin.bind_parameter("MOGraph.max_dominated_solutions", 5)
        gin.bind_parameter("MOGraph.pareto_objectives", 2)

    def test_init_basic(
        self, mock_retro_model, simple_heuristics, basic_building_blocks
    ):
        """Test basic MOSearch initialization"""
        self.setup_gin_config()

        search = MOSearch(
            target="CCO",
            retro_model=mock_retro_model,
            building_blocks=basic_building_blocks,
            heuristic_fns=simple_heuristics,
            top_n=10,
            max_depth=3,
            iteration_budget=100,
            weight_iter_budget=20,
            time_budget=300.0,
        )

        assert search.max_depth == 6  # 2 * max_depth
        assert search.top_n == 10
        assert search.weight_iter_budget == 20
        assert search.iteration_budget == 100
        assert search.time_budget == 300.0
        assert isinstance(search.search_graph, MOGraph)
        assert len(search.weights_open) == 4  # no_weights from gin config


class TestCanExpandRetro:
    """Test cases for can_expand_retro method"""

    @pytest.fixture
    def search_instance(self):
        """Create a MOSearch instance for testing"""
        gin.clear_config()
        gin.bind_parameter("MOGraph.weight_samples", 8)
        gin.bind_parameter("MOGraph.no_weights", 2)
        gin.bind_parameter("MOGraph.weight_initial", "sobol")
        gin.bind_parameter("MOGraph.include_extreme", False)
        gin.bind_parameter("MOGraph.max_dominated_solutions", 5)
        gin.bind_parameter("MOGraph.pareto_objectives", 2)

        mock_model = Mock(spec=OneStepModel)
        heuristics = [lambda x: 1.0, lambda x: 2.0]

        return MOSearch(
            target="CCO",
            retro_model=mock_model,
            building_blocks={"CC", "CO"},
            heuristic_fns=heuristics,
            top_n=5,
            max_depth=3,
            iteration_budget=50,
            weight_iter_budget=10,
        )

    def test_can_expand_mol_node_open(self, search_instance):
        """Test expansion eligibility for open MolNode within depth limit"""
        node = MolNode(
            smiles="CCO",
            heuristic_fns=[lambda x: 1.0],
            depth=2,  # Within max_depth (6)
            is_known=False,
            pareto_objectives=2,
            max_dominated_solutions=5,
        )
        assert node.is_open == True

        result = search_instance.can_expand_retro(node)
        assert result == True

    def test_can_expand_mol_node_max_depth(self, search_instance):
        """Test expansion eligibility for MolNode at max depth"""
        node = MolNode(
            smiles="CCO",
            heuristic_fns=[lambda x: 1.0],
            depth=6,  # At max_depth
            is_known=False,
            pareto_objectives=2,
            max_dominated_solutions=5,
        )

        result = search_instance.can_expand_retro(node)
        assert result == False

    def test_can_expand_mol_node_closed(self, search_instance):
        """Test expansion eligibility for closed MolNode"""
        node = MolNode(
            smiles="CCO",
            heuristic_fns=[lambda x: 1.0],
            depth=2,
            is_known=False,
            pareto_objectives=2,
            max_dominated_solutions=5,
        )
        node.is_open = False

        result = search_instance.can_expand_retro(node)
        assert result == False

    def test_can_expand_rxn_node(self, search_instance):
        """Test that RxnNode cannot be expanded"""
        node = RxnNode(
            smiles="CC.CO>>CCO",
            template="template",
            reagents=["catalyst"],
            temp=298.0,
            depth=2,
            cost=np.array([1.0, 1.5]),
            weight_length=2,
            pareto_objectives=2,
            max_dominated_solutions=5,
        )

        result = search_instance.can_expand_retro(node)
        assert result == False


class TestRetroExpansion:
    """Test cases for retro_expansion method"""

    @pytest.fixture
    def search_with_mock_model(self):
        """Create MOSearch with mock model for expansion testing"""
        gin.clear_config()
        gin.bind_parameter("MOGraph.weight_samples", 8)
        gin.bind_parameter("MOGraph.no_weights", 2)
        gin.bind_parameter("MOGraph.weight_initial", "sobol")
        gin.bind_parameter("MOGraph.include_extreme", False)
        gin.bind_parameter("MOGraph.max_dominated_solutions", 5)
        gin.bind_parameter("MOGraph.pareto_objectives", 2)

        mock_model = Mock(spec=OneStepModel)
        mock_model.predict.return_value = [
            [
                {
                    "reactants": ["CC", "CO"],
                    "reagents": ["catalyst"],
                    "temperature": 298.0,
                    "rxn_smiles": "CC.CO>>CCO",
                    "template": "template",
                    "costs": [1.0, 1.5],
                }
            ]
        ]

        heuristics = [lambda x: 1.0, lambda x: 2.0]

        return MOSearch(
            target="CCO",
            retro_model=mock_model,
            building_blocks={"CC", "CO"},
            heuristic_fns=heuristics,
            top_n=5,
            max_depth=3,
            iteration_budget=50,
            weight_iter_budget=10,
        )

    def test_retro_expansion_successful(self, search_with_mock_model):
        """Test successful retrosynthesis expansion"""
        # Create an expandable node
        node = MolNode(
            smiles="CCO",
            heuristic_fns=[lambda x: 1.0, lambda x: 2.0],
            depth=2,
            is_known=False,
            pareto_objectives=2,
            max_dominated_solutions=5,
        )

        # Add node to open_nodes set since expand_graph expects it there
        search_with_mock_model.search_graph.open_nodes.add(node)

        nodes_and_weights = {(node, (0,))}

        result = search_with_mock_model.retro_expansion(nodes_and_weights)

        assert result == False  # No early resampling
        search_with_mock_model.retro_model.predict.assert_called_once()

    def test_retro_expansion_no_predictions(self, search_with_mock_model):
        """Test expansion when model returns no predictions"""
        # Mock model to return empty predictions
        search_with_mock_model.retro_model.predict.return_value = [[]]

        node = MolNode(
            smiles="CCO",
            heuristic_fns=[lambda x: 1.0, lambda x: 2.0],
            depth=2,
            is_known=False,
            pareto_objectives=2,
            max_dominated_solutions=5,
        )

        nodes_and_weights = {(node, (0,))}

        # Mock expand_graph to return None (no new nodes)
        with patch.object(
            search_with_mock_model.search_graph, "expand_graph", return_value=None
        ):
            result = search_with_mock_model.retro_expansion(nodes_and_weights)

        assert result == False  # No early resampling, just no expansion


class TestSpawnNewWeights:
    """Test cases for spawn_new_weights method"""

    @pytest.fixture
    def search_instance(self):
        """Create MOSearch instance for weight testing"""
        gin.clear_config()
        gin.bind_parameter("MOGraph.weight_samples", 16)
        gin.bind_parameter("MOGraph.no_weights", 4)
        gin.bind_parameter("MOGraph.weight_initial", "sobol")
        gin.bind_parameter("MOGraph.include_extreme", False)
        gin.bind_parameter("MOGraph.max_dominated_solutions", 5)
        gin.bind_parameter("MOGraph.pareto_objectives", 2)

        mock_model = Mock(spec=OneStepModel)
        heuristics = [lambda x: 1.0, lambda x: 2.0]

        return MOSearch(
            target="CCO",
            retro_model=mock_model,
            building_blocks={"CC", "CO"},
            heuristic_fns=heuristics,
            top_n=5,
            max_depth=3,
            iteration_budget=50,
            weight_iter_budget=10,
        )

    def test_spawn_new_weights_iterative_budget_reached(self, search_instance):
        """Test weight spawning when iteration budget is reached"""
        with patch.object(
            search_instance.search_graph, "reinitialize_graph"
        ) as mock_reinit:
            result = search_instance.spawn_new_weights(
                num_iter=11, early_resampling=False
            )

        assert result == 1  # Reset weight iteration counter
        mock_reinit.assert_called_once()

    def test_spawn_new_weights_early_resampling(self, search_instance):
        """Test weight spawning triggered by early resampling"""
        with patch.object(
            search_instance.search_graph, "reinitialize_graph"
        ) as mock_reinit:
            result = search_instance.spawn_new_weights(
                num_iter=5, early_resampling=True
            )

        assert result == 1
        mock_reinit.assert_called_once()

    def test_spawn_new_weights_no_weights_left(self, search_instance):
        """Test weight spawning when no weights are left"""
        # Simulate no weights left by making weights_open too small
        search_instance.search_graph.weights_open = np.array(
            [[1, 2], [3, 4]]
        )  # Only 2 rows, need 4

        result = search_instance.spawn_new_weights(num_iter=11, early_resampling=False)

        assert result == -100  # Exit search

    def test_spawn_new_weights_no_action_needed(self, search_instance):
        """Test weight spawning when no action is needed"""
        result = search_instance.spawn_new_weights(num_iter=5, early_resampling=False)

        assert result == 5  # No action


class TestChooseNextNodes:
    """Test cases for choose_next_nodes method"""

    @pytest.fixture
    def search_with_nodes(self):
        """Create MOSearch instance with open nodes for testing"""
        gin.clear_config()
        gin.bind_parameter("MOGraph.weight_samples", 8)
        gin.bind_parameter("MOGraph.no_weights", 2)
        gin.bind_parameter("MOGraph.weight_initial", "sobol")
        gin.bind_parameter("MOGraph.include_extreme", False)
        gin.bind_parameter("MOGraph.max_dominated_solutions", 5)
        gin.bind_parameter("MOGraph.pareto_objectives", 2)

        mock_model = Mock(spec=OneStepModel)
        heuristics = [lambda x: 1.0, lambda x: 2.0]

        search = MOSearch(
            target="CCO",
            retro_model=mock_model,
            building_blocks={"CC", "CO"},
            heuristic_fns=heuristics,
            top_n=5,
            max_depth=3,
            iteration_budget=50,
            weight_iter_budget=10,
        )

        # Add some open nodes with different total values
        node1 = MolNode(
            smiles="CCC",
            heuristic_fns=heuristics,
            depth=2,
            is_known=False,
            pareto_objectives=2,
            max_dominated_solutions=5,
        )
        node1.total_value = [1.0, 3.0]  # Min for weight 0, not min for weight 1

        node2 = MolNode(
            smiles="CCCO",
            heuristic_fns=heuristics,
            depth=2,
            is_known=False,
            pareto_objectives=2,
            max_dominated_solutions=5,
        )
        node2.total_value = [2.0, 1.0]  # Not min for weight 0, min for weight 1

        node3 = MolNode(
            smiles="CCN",
            heuristic_fns=heuristics,
            depth=2,
            is_known=False,
            pareto_objectives=2,
            max_dominated_solutions=5,
        )
        node3.total_value = [1.5, 2.0]  # Neither min

        search.search_graph.open_nodes = {node1, node2, node3}

        return search, node1, node2, node3

    def test_choose_next_nodes_basic(self, search_with_nodes):
        """Test basic node selection functionality"""
        search, node1, node2, node3 = search_with_nodes

        selected, _ = search.choose_next_nodes()

        assert isinstance(selected, set)
        assert len(selected) > 0

        # Verify that selected nodes have weight indices
        for node, weight_indices in selected:
            assert isinstance(node, MolNode)
            assert isinstance(weight_indices, tuple)
            assert len(weight_indices) > 0

    def test_choose_next_nodes_min_selection(self, search_with_nodes):
        """Test that nodes with minimum values are selected"""
        search, node1, node2, node3 = search_with_nodes

        selected, _ = search.choose_next_nodes()

        # Convert to dict for easier checking
        selected_dict = {
            node.smiles: weight_indices for node, weight_indices in selected
        }

        # node1 should be selected for weight 0 (total_value[0] = 1.0 is minimum)
        # node2 should be selected for weight 1 (total_value[1] = 1.0 is minimum)
        assert "CCC" in selected_dict  # node1
        assert "CCCO" in selected_dict  # node2

    def test_choose_next_nodes_identical_values(self, search_with_nodes):
        """Test node selection when multiple nodes have identical minimum values"""
        search, node1, node2, node3 = search_with_nodes

        # Make all nodes have the same total values
        for node in [node1, node2, node3]:
            node.total_value = [1.0, 1.0]

        selected, _ = search.choose_next_nodes()

        # Should still return valid selections
        assert len(selected) > 0
        for node, weight_indices in selected:
            assert isinstance(node, MolNode)
            assert node in [node1, node2, node3]


class TestRunMOSearchIntegration:
    """Integration test for the complete run_mo_search method"""

    def test_run_mo_search_basic_integration(self):
        """Test a basic integration of the complete search process"""
        gin.clear_config()
        gin.bind_parameter("MOGraph.weight_samples", 8)
        gin.bind_parameter("MOGraph.no_weights", 2)
        gin.bind_parameter("MOGraph.weight_initial", "sobol")
        gin.bind_parameter("MOGraph.include_extreme", False)
        gin.bind_parameter("MOGraph.max_dominated_solutions", 5)
        gin.bind_parameter("MOGraph.pareto_objectives", 2)

        # Create mock retro model with realistic predictions
        mock_model = Mock(spec=OneStepModel)
        mock_model.predict.return_value = [
            [
                {
                    "reactants": ["CC", "CO"],
                    "reagents": ["base"],
                    "temperature": 298.0,
                    "rxn_smiles": "CC.CO>>CCO",
                    "template": "[C:1][C:2].[C:3][OH:4]>>[C:1][C:2][C:3][OH:4]",
                    "costs": [0.8, 1.2],
                },
                {
                    "reactants": ["C", "CCO"],
                    "reagents": ["acid"],
                    "temperature": 350.0,
                    "rxn_smiles": "C.CCO>>CCO",
                    "template": "[C:1].[C:2][C:3][OH:4]>>[C:2][C:3][OH:4]",
                    "costs": [1.1, 0.9],
                },
            ]
        ]

        heuristics = [
            lambda smiles: len(smiles) * 0.1,  # Length-based
            lambda smiles: smiles.count("C") * 0.15,  # Carbon count-based
        ]

        search = MOSearch(
            target="CCO",
            retro_model=mock_model,
            building_blocks={"CC", "CO", "C"},
            heuristic_fns=heuristics,
            top_n=3,
            max_depth=2,  # Keep shallow for quick test
            iteration_budget=5,  # Small budget for quick test
            weight_iter_budget=3,
            time_budget=10.0,  # 10 second limit
        )

        # Mock torch.cuda.empty_cache to avoid CUDA issues in testing
        with patch("torch.cuda.empty_cache"):
            search.run_mo_search()

        # Verify the search completed without errors
        # and that the model was called
        assert mock_model.predict.called

        # Verify that some expansions occurred
        # (the exact number depends on the search dynamics)
        assert len(search.search_graph.graph.nodes) >= 1

    def test_run_mo_search_early_termination_no_open_nodes(self):
        """Test search termination when no open nodes remain"""
        gin.clear_config()
        gin.bind_parameter("MOGraph.weight_samples", 8)
        gin.bind_parameter("MOGraph.no_weights", 2)
        gin.bind_parameter("MOGraph.weight_initial", "sobol")
        gin.bind_parameter("MOGraph.include_extreme", False)
        gin.bind_parameter("MOGraph.max_dominated_solutions", 5)
        gin.bind_parameter("MOGraph.pareto_objectives", 2)

        mock_model = Mock(spec=OneStepModel)
        heuristics = [lambda x: 1.0, lambda x: 2.0]

        search = MOSearch(
            target="CC",  # Already in building blocks
            retro_model=mock_model,
            building_blocks={"CC"},
            heuristic_fns=heuristics,
            top_n=3,
            max_depth=2,
            iteration_budget=10,
            weight_iter_budget=5,
        )

        with patch("torch.cuda.empty_cache"):
            search.run_mo_search()

        # Target is already known, so no open nodes should remain
        assert len(search.search_graph.open_nodes) == 0

    def test_run_mo_search_time_budget_exceeded(self):
        """Test search termination when time budget is exceeded"""
        gin.clear_config()
        gin.bind_parameter("MOGraph.weight_samples", 8)
        gin.bind_parameter("MOGraph.no_weights", 2)
        gin.bind_parameter("MOGraph.weight_initial", "sobol")
        gin.bind_parameter("MOGraph.include_extreme", False)
        gin.bind_parameter("MOGraph.max_dominated_solutions", 5)
        gin.bind_parameter("MOGraph.pareto_objectives", 2)

        mock_model = Mock(spec=OneStepModel)
        mock_model.predict.return_value = [
            [
                {
                    "reactants": ["CC", "CO"],
                    "reagents": ["catalyst"],
                    "temperature": 298.0,
                    "rxn_smiles": "CC.CO>>CCO",
                    "template": "template",
                    "costs": [1.0, 1.5],
                }
            ]
        ]

        heuristics = [lambda x: 1.0, lambda x: 2.0]

        search = MOSearch(
            target="CCO",
            retro_model=mock_model,
            building_blocks={"CC", "CO"},
            heuristic_fns=heuristics,
            top_n=3,
            max_depth=2,
            iteration_budget=100,  # High iteration budget
            weight_iter_budget=50,
            time_budget=0.001,  # Very small time budget
        )

        start_time = time.time()

        with patch("torch.cuda.empty_cache"):
            search.run_mo_search()

        elapsed = time.time() - start_time

        # Should terminate quickly due to time budget
        assert elapsed < 1.0  # Should finish much faster than without time limit

    def test_run_mo_search_weight_exhaustion(self):
        """Test search termination when all weights are exhausted"""
        gin.clear_config()
        gin.bind_parameter("MOGraph.weight_samples", 4)  # Very small weight pool
        gin.bind_parameter("MOGraph.no_weights", 2)
        gin.bind_parameter("MOGraph.weight_initial", "sobol")
        gin.bind_parameter("MOGraph.include_extreme", False)
        gin.bind_parameter("MOGraph.max_dominated_solutions", 5)
        gin.bind_parameter("MOGraph.pareto_objectives", 2)

        mock_model = Mock(spec=OneStepModel)
        mock_model.predict.return_value = [
            [
                {
                    "reactants": ["CC", "CO"],
                    "reagents": ["catalyst"],
                    "temperature": 298.0,
                    "rxn_smiles": "CC.CO>>CCO",
                    "template": "template",
                    "costs": [1.0, 1.5],
                }
            ]
        ]

        heuristics = [lambda x: 1.0, lambda x: 2.0]

        search = MOSearch(
            target="CCO",
            retro_model=mock_model,
            building_blocks={"CC", "CO"},
            heuristic_fns=heuristics,
            top_n=3,
            max_depth=2,
            iteration_budget=100,
            weight_iter_budget=2,  # Small weight iteration budget
        )

        with patch("torch.cuda.empty_cache"):
            search.run_mo_search()

        # Should complete without errors despite weight exhaustion
        assert True  # If we reach here, no exception was raised


class TestMOSearchRealWorldScenario:
    """Real-world integration test using actual gin config"""

    def test_mo_search_with_gin_config(self):
        """Test MOSearch using the actual gin configuration file"""
        # Clear any existing gin configuration
        gin.clear_config()

        # Load the test gin configuration (doesn't require model files)
        test_dir = os.path.dirname(__file__)
        project_root = os.path.dirname(test_dir)
        config_path = os.path.join(
            project_root, "configs", "test_config.gin"
        )

        if not os.path.exists(config_path):
            pytest.skip("Test gin config file not found")

        gin.parse_config_file(config_path)

        # Create a realistic but simple mock model
        mock_model = Mock(spec=OneStepModel)

        def mock_predict(smiles_list, top_n):
            """Mock prediction that returns reasonable synthetic routes"""
            if isinstance(smiles_list, str):
                smiles_list = [smiles_list]

            predictions = []
            for smiles in smiles_list:
                mol_predictions = []

                # Create some synthetic predictions based on the molecule
                if len(smiles) > 3:  # For non-trivial molecules
                    # Prediction 1: Break into smaller fragments
                    mol_predictions.append(
                        {
                            "reactants": ["CC", "CO"],
                            "reagents": ["base"],
                            "temperature": 298.0,
                            "rxn_smiles": f"CC.CO>>{smiles}",
                            "template": "[C:1][C:2].[C:3][OH:4]>>[C:1][C:2][C:3][OH:4]",
                            "costs": [0.8, 1.2],
                        }
                    )

                    # Prediction 2: Alternative route
                    mol_predictions.append(
                        {
                            "reactants": ["C", f"C{smiles[1:]}"],
                            "reagents": ["acid"],
                            "temperature": 350.0,
                            "rxn_smiles": f"C.C{smiles[1:]}>>{smiles}",
                            "template": "[C:1].[C:2]>>[C:1][C:2]",
                            "costs": [1.1, 0.9],
                        }
                    )

                predictions.append(mol_predictions[:top_n])

            return predictions

        mock_model.predict.side_effect = mock_predict

        # Define realistic heuristic functions
        def molecular_weight_heuristic(smiles: str) -> float:
            """Estimate based on molecular weight (length proxy)"""
            return len(smiles) * 12.0  # Rough MW estimate

        def complexity_heuristic(smiles: str) -> float:
            """Estimate based on molecular complexity"""
            complexity_chars = smiles.count("(") + smiles.count("[") + smiles.count("@")
            return complexity_chars * 5.0 + len(smiles) * 1.0

        heuristics = [molecular_weight_heuristic, complexity_heuristic]

        # Use a realistic target molecule
        target = "CCc1ccc(C(=O)O)cc1"  # p-ethylbenzoic acid
        building_blocks = {
            "CC",
            "CO",
            "C",
            "c1ccccc1",
            "CCl",
            "C(=O)O",
            "Cc1ccccc1",
            "CCc1ccccc1",
        }

        try:
            search = MOSearch(
                target=target,
                retro_model=mock_model,
                building_blocks=building_blocks,
                heuristic_fns=heuristics,
                top_n=5,
                max_depth=3,
                iteration_budget=20,
                weight_iter_budget=8,
                time_budget=30.0,
            )

            # Run the search
            with patch("torch.cuda.empty_cache"):
                search.run_mo_search()

            # Verify the search ran successfully
            assert mock_model.predict.called
            assert len(search.search_graph.graph.nodes) >= 1

            # Check if any solutions were found
            pareto_solutions = len(search.search_graph.pareto_front)
            print(f"Found {pareto_solutions} Pareto-optimal solutions")

            # Verify search completed without errors
            assert True

        except Exception as e:
            pytest.fail(f"MOSearch integration test failed: {e}")

class TestRemoveDominatedNodes:
    """Test cases for remove_dominated_nodes method"""

    @pytest.fixture
    def search_instance(self):
        """Create MOSearch instance for testing"""
        gin.clear_config()
        gin.bind_parameter("MOGraph.weight_samples", 8)
        gin.bind_parameter("MOGraph.no_weights", 2)
        gin.bind_parameter("MOGraph.weight_initial", "sobol")
        gin.bind_parameter("MOGraph.include_extreme", False)
        gin.bind_parameter("MOGraph.max_dominated_solutions", 5)
        gin.bind_parameter("MOGraph.pareto_objectives", 2)

        mock_model = Mock(spec=OneStepModel)
        heuristics = [lambda x: 1.0, lambda x: 2.0]

        return MOSearch(
            target="CCO",
            retro_model=mock_model,
            building_blocks={"CC", "CO"},
            heuristic_fns=heuristics,
            top_n=5,
            max_depth=3,
            iteration_budget=50,
            weight_iter_budget=10,
        )

    def test_remove_dominated_nodes_basic(self, search_instance):
        """Test basic removal of dominated nodes"""
        # Setup Pareto front in search graph
        # Pareto front has points: (1.0, 1.0), (0.5, 2.0)
        search_instance.search_graph.pareto_front = {
            (1.0, 1.0): [[0.5, 0.5]],
            (0.5, 2.0): [[0.5, 0.5]],
        }
        # Mock pareto_front_costs property if it exists, or just rely on implementation details
        # The implementation uses graph.pareto_front_costs which seems to be a property or attribute
        # Let's mock it on the search_graph instance
        search_instance.search_graph.pareto_front_costs = np.array(
            [[1.0, 1.0], [0.5, 2.0]]
        )

        # Create nodes
        # Node 1: (2.0, 2.0) - Dominated by (1.0, 1.0)
        node1 = MolNode(
            smiles="N1",
            heuristic_fns=[],
            depth=1,
            is_known=False,
            pareto_objectives=2,
            max_dominated_solutions=5,
        )
        node1.best_total_value = np.array([2.0, 2.0])

        # Node 2: (0.4, 1.5) - Not dominated (better in obj 1 than both pareto points)
        node2 = MolNode(
            smiles="N2",
            heuristic_fns=[],
            depth=1,
            is_known=False,
            pareto_objectives=2,
            max_dominated_solutions=5,
        )
        node2.best_total_value = np.array([0.4, 1.5])

        # Node 3: (0.8, 0.8) - Not dominated (better than (1.0, 1.0))
        node3 = MolNode(
            smiles="N3",
            heuristic_fns=[],
            depth=1,
            is_known=False,
            pareto_objectives=2,
            max_dominated_solutions=5,
        )
        node3.best_total_value = np.array([0.8, 0.8])

        open_nodes = [node1, node2, node3]

        # Enable exclude_dominated_nodes
        search_instance.exclude_dominated_nodes = True

        filtered_nodes, stop_search = search_instance.remove_dominated_nodes(open_nodes)

        assert stop_search is False
        assert len(filtered_nodes) == 2
        assert node1 not in filtered_nodes
        assert node2 in filtered_nodes
        assert node3 in filtered_nodes
        assert node1.is_dominated is True
        assert node2.is_dominated is False
        assert node3.is_dominated is False

    def test_remove_dominated_nodes_stop_on_full_pareto(self, search_instance):
        """Test stopping search when all nodes are dominated and stop_on_full_pareto is True"""
        search_instance.search_graph.pareto_front_costs = np.array([[1.0, 1.0]])

        # Node 1: (2.0, 2.0) - Dominated
        node1 = MolNode(
            smiles="N1",
            heuristic_fns=[],
            depth=1,
            is_known=False,
            pareto_objectives=2,
            max_dominated_solutions=5,
        )
        node1.best_total_value = np.array([2.0, 2.0])

        open_nodes = [node1]

        search_instance.stop_on_full_pareto = True
        search_instance.exclude_dominated_nodes = False  # Even if false, stop_on_full_pareto logic checks dominance

        filtered_nodes, stop_search = search_instance.remove_dominated_nodes(open_nodes)

        assert stop_search is True
        assert filtered_nodes is None

    def test_remove_dominated_nodes_no_exclusion(self, search_instance):
        """Test that nodes are not removed if exclude_dominated_nodes is False"""
        search_instance.search_graph.pareto_front_costs = np.array([[1.0, 1.0]])

        # Node 1: (2.0, 2.0) - Dominated
        node1 = MolNode(
            smiles="N1",
            heuristic_fns=[],
            depth=1,
            is_known=False,
            pareto_objectives=2,
            max_dominated_solutions=5,
        )
        node1.best_total_value = np.array([2.0, 2.0])

        open_nodes = [node1]

        search_instance.exclude_dominated_nodes = False
        search_instance.stop_on_full_pareto = False

        filtered_nodes, stop_search = search_instance.remove_dominated_nodes(open_nodes)

        assert stop_search is False
        assert len(filtered_nodes) == 1
        assert filtered_nodes[0] == node1
