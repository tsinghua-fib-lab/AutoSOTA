from pyciphod.causal_discovery.local.local_constraint_based import LocPC, LocPC_CDE


def test_locpc_find_subset_noc_and_non_orientability():
    # Setup LocPC with a small graph
    alg = LocPC(target='A')
    # Add vertices
    for v in ['A', 'B', 'C', 'D']:
        alg.g_hat.add_vertex(v)
    # Add undirected edges: A-B, B-C, C-D
    alg.g_hat.add_undirected_edge('A', 'B')
    alg.g_hat.add_undirected_edge('B', 'C')
    alg.g_hat.add_undirected_edge('C', 'D')

    # mark B and C as visited so they can be discovered from A
    alg._visited = {'B', 'C'}

    subset = alg._find_subset_NOC()
    # Expect A plus visited neighbors reachable: A and B (B is visited and adjacent to A)
    assert 'A' in subset
    assert 'B' in subset

    # Test non-orientability: for subset containing A and B, ensure it returns False if there
    # exists an undirected edge from a node in subset to a node outside subset
    # Here subset {'A','B'} and B has adjacency C outside subset -> undirected edge present -> False
    assert alg._non_orientability_criterion({'A', 'B'}) is False

    # If we remove the undirected edge B-C and replace it by an uncertain edge (only one), it may pass
    alg.g_hat.remove_undirected_edge('B', 'C')
    # add an uncertain edge (simulate '-||') between B and C
    alg.g_hat.add_uncertain_edge('B', 'C', edge_type='-||')
    # Now non_orientability_criterion for subset {'A','B'} should be True because
    # condition 1 (no undirected edges) holds and condition 2 has only one uncertain neighbor
    assert alg._non_orientability_criterion({'A', 'B'}) is True


def test_locpc_cde_find_subset_and_non_orientability():
    # Setup LocPC_CDE which wraps LocPC internally
    cde = LocPC_CDE(treatment='T', outcome='O')
    alg = cde.local_constraint_based_algorithm

    # Build a small graph around outcome O
    for v in ['O', 'X', 'Y']:
        alg.g_hat.add_vertex(v)
    # Undirected edges O-X and X-Y
    alg.g_hat.add_undirected_edge('O', 'X')
    alg.g_hat.add_undirected_edge('X', 'Y')

    # Mark X as visited
    alg._visited = {'X'}

    subset = cde._find_subset_NOC()
    # Should find outcome O and its visited neighbor X
    assert 'O' in subset
    assert 'X' in subset

    # Non-orientability: since O has undirected edge to X outside subset? here X in subset so we need
    # to check a case where undirected edge exists to outside -> make Y outside and O connected to Y
    alg.g_hat.add_undirected_edge('O', 'Y')
    # For subset {'O','X'} presence of undirected (O,Y) where Y not in subset should lead to False
    assert cde._non_orientability_criterion({'O', 'X'}) is False

    # Remove O-Y undirected and remove X-Y undirected and add uncertain single edge O-Y -> should be True
    alg.g_hat.remove_undirected_edge('O', 'Y')
    alg.g_hat.remove_undirected_edge('X', 'Y')
    alg.g_hat.add_uncertain_edge('O', 'Y', edge_type='-||')
    assert cde._non_orientability_criterion({'O', 'X'}) is True
