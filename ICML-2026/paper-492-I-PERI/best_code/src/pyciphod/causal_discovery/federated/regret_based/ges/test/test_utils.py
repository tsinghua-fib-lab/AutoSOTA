# Copyright 2021 Juan L Gamella

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:

# 1. Redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""
Tests for the ges.utils module.
"""

import unittest
import numpy as np
import sempler
import sempler.generators
import itertools
import networkx as nx

import ges.utils

# ---------------------------------------------------------------------
# Tests


class UtilsTests(unittest.TestCase):

    # ------------------------
    # Tests of graph functions

    def test_subsets(self):
        # Test 1
        self.assertEqual([set()], ges.utils.subsets(set()))
        # Test 2
        S = {0, 1}
        subsets = ges.utils.subsets(S)
        self.assertEqual(4, len(subsets))
        for s in [set(), {0}, {1}, {0, 1}]:
            self.assertIn(s, subsets)
        # Test 3
        S = {1, 2, 3, 4}
        subsets = ges.utils.subsets(S)
        self.assertEqual(16, len(subsets))

    def test_is_dag_1(self):
        # Should be correct
        A = np.array([[0, 0, 1, 0, 0],
                      [0, 0, 1, 0, 0],
                      [0, 0, 0, 1, 1],
                      [0, 0, 0, 0, 1],
                      [0, 0, 0, 0, 0]])
        self.assertTrue(ges.utils.is_dag(A))

    def test_is_dag_2(self):
        # DAG with a cycle
        A = np.array([[0, 0, 1, 0, 0],
                      [0, 0, 1, 0, 0],
                      [0, 0, 0, 1, 1],
                      [1, 0, 0, 0, 1],
                      [0, 0, 0, 0, 0]])
        self.assertFalse(ges.utils.is_dag(A))

    def test_is_dag_3(self):
        # PDAG
        A = np.array([[0, 0, 1, 0, 0],
                      [0, 0, 1, 0, 0],
                      [0, 0, 0, 1, 1],
                      [0, 0, 0, 0, 1],
                      [0, 0, 0, 1, 0]])
        self.assertFalse(ges.utils.is_dag(A))

    def test_topological_sort_1(self):
        A = np.array([[0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0],
                      [1, 1, 0, 0, 0],
                      [0, 0, 1, 0, 0],
                      [0, 0, 1, 1, 0]]).T
        order = ges.utils.topological_ordering(A)
        possible_orders = [[0, 1, 2, 3, 4], [1, 0, 2, 3, 4]]
        self.assertIn(order, possible_orders)

    def test_topological_sort_2(self):
        A = np.array([[0, 0, 0, 1, 0],
                      [0, 0, 0, 0, 0],
                      [1, 1, 0, 0, 0],
                      [0, 0, 1, 0, 0],
                      [0, 0, 1, 1, 0]]).T
        try:
            ges.utils.topological_ordering(A)
            self.fail()
        except ValueError:
            pass

    def test_topological_sort_3(self):
        G = 100
        p = 30
        for i in range(G):
            A = sempler.generators.dag_avg_deg(p, 3, 1, 1)
            ordering = ges.utils.topological_ordering(A)
            fro, to = np.where(A != 0)
            # Test that the ordering is correct, i.e. for every edge x
            # -> y in the graph, x appears before in the ordering
            for (x, y) in zip(fro, to):
                pos_x = np.where(np.array(ordering) == x)[0][0]
                pos_y = np.where(np.array(ordering) == y)[0][0]
                self.assertLess(pos_x, pos_y)
        print("Checked topological sorting for %d DAGs" % (i+1))

    def test_neighbors(self):
        p = 4
        A = np.zeros((p, p))
        # Test 1
        for i in range(p):
            self.assertEqual(set(), ges.utils.neighbors(i, A))
        # Test 2
        A = np.array([[0, 1, 1, 0],
                      [0, 0, 1, 0],
                      [0, 1, 0, 1],
                      [0, 0, 1, 0]])
        self.assertEqual(set(), ges.utils.neighbors(0, A))
        self.assertEqual({2}, ges.utils.neighbors(1, A))
        self.assertEqual({1, 3}, ges.utils.neighbors(2, A))
        self.assertEqual({2}, ges.utils.neighbors(3, A))

    def test_adj(self):
        p = 4
        A = np.zeros((p, p))
        # Test 1
        for i in range(p):
            self.assertEqual(set(), ges.utils.adj(i, A))
        # Test 2
        A = np.array([[0, 1, 1, 0],
                      [0, 0, 1, 0],
                      [0, 1, 0, 1],
                      [0, 0, 1, 0]])
        self.assertEqual({1, 2}, ges.utils.adj(0, A))
        self.assertEqual({0, 2}, ges.utils.adj(1, A))
        self.assertEqual({0, 1, 3}, ges.utils.adj(2, A))
        self.assertEqual({2}, ges.utils.adj(3, A))

    def test_na(self):
        A = np.array([[0, 1, 1, 0],
                      [0, 0, 1, 0],
                      [0, 1, 0, 1],
                      [0, 0, 1, 0]])
        # 00
        self.assertEqual(set(), ges.utils.na(0, 0, A))
        # 01
        self.assertEqual(set(), ges.utils.na(0, 1, A))
        # 02
        self.assertEqual(set(), ges.utils.na(0, 2, A))
        # 03
        self.assertEqual(set(), ges.utils.na(0, 3, A))

        # 10
        self.assertEqual({2}, ges.utils.na(1, 0, A))
        # 11
        self.assertEqual({2}, ges.utils.na(1, 1, A))
        # 12
        self.assertEqual(set(), ges.utils.na(1, 2, A))
        # 13
        self.assertEqual({2}, ges.utils.na(1, 3, A))

        # 20
        self.assertEqual({1}, ges.utils.na(2, 0, A))
        # 21
        self.assertEqual(set(), ges.utils.na(2, 1, A))
        # 22
        self.assertEqual({1, 3}, ges.utils.na(2, 2, A))
        # 23
        self.assertEqual(set(), ges.utils.na(2, 3, A))

        # 30
        self.assertEqual({2}, ges.utils.na(3, 0, A))
        # 31
        self.assertEqual({2}, ges.utils.na(3, 1, A))
        # 32
        self.assertEqual(set(), ges.utils.na(3, 2, A))
        # 33
        self.assertEqual({2}, ges.utils.na(3, 3, A))

    def test_pa(self):
        p = 4
        A = np.zeros((p, p))
        # Test 1
        for i in range(p):
            self.assertEqual(set(), ges.utils.pa(i, A))
        # Test 2
        A = np.array([[0, 1, 1, 0],
                      [0, 0, 1, 0],
                      [0, 1, 0, 1],
                      [0, 0, 1, 0]])
        self.assertEqual(set(), ges.utils.pa(0, A))
        self.assertEqual({0}, ges.utils.pa(1, A))
        self.assertEqual({0}, ges.utils.pa(2, A))
        self.assertEqual(set(), ges.utils.pa(3, A))

    def test_ch(self):
        p = 4
        A = np.zeros((p, p))
        # Test 1
        for i in range(p):
            self.assertEqual(set(), ges.utils.ch(i, A))
        # Test 2
        A = np.array([[0, 1, 1, 0],
                      [0, 0, 1, 0],
                      [0, 1, 0, 1],
                      [0, 0, 1, 0]])
        self.assertEqual({1, 2}, ges.utils.ch(0, A))
        self.assertEqual(set(), ges.utils.ch(1, A))
        self.assertEqual(set(), ges.utils.ch(2, A))
        self.assertEqual(set(), ges.utils.ch(3, A))

    def test_is_clique(self):
        A = np.array([[0, 1, 1, 0],
                      [0, 0, 1, 0],
                      [0, 1, 0, 1],
                      [0, 0, 1, 0]])
        # |S| = 1
        for i in range(len(A)):
            self.assertTrue(ges.utils.is_clique({i}, A))
        # |S| = 2
        self.assertTrue(ges.utils.is_clique({0, 1}, A))
        self.assertTrue(ges.utils.is_clique({0, 2}, A))
        self.assertFalse(ges.utils.is_clique({0, 3}, A))
        self.assertTrue(ges.utils.is_clique({1, 2}, A))
        self.assertFalse(ges.utils.is_clique({1, 3}, A))
        self.assertTrue(ges.utils.is_clique({2, 3}, A))
        # |S| = 3
        self.assertTrue(ges.utils.is_clique({0, 1, 2}, A))
        self.assertFalse(ges.utils.is_clique({0, 1, 3}, A))
        self.assertFalse(ges.utils.is_clique({0, 2, 3}, A))
        self.assertFalse(ges.utils.is_clique({1, 2, 3}, A))
        # |S| = 4
        self.assertFalse(ges.utils.is_clique({0, 1, 2, 3}, A))

    def test_semi_directed_paths_1(self):
        A = np.array([[0, 1, 1, 0],
                      [0, 0, 1, 0],
                      [0, 1, 0, 1],
                      [0, 0, 1, 0]])
        # 0 to 1
        paths = ges.utils.semi_directed_paths(0, 1, A)
        self.assertEqual(2, len(paths))
        self.assertTrue([0, 1] in paths)
        self.assertTrue([0, 2, 1] in paths)
        # 1 to 0
        paths = ges.utils.semi_directed_paths(1, 0, A)
        self.assertEqual(0, len(paths))

        # 0 to 2
        paths = ges.utils.semi_directed_paths(0, 2, A)
        self.assertEqual(2, len(paths))
        self.assertTrue([0, 2] in paths)
        self.assertTrue([0, 1, 2] in paths)
        # 2 to 0
        paths = ges.utils.semi_directed_paths(2, 0, A)
        self.assertEqual(0, len(paths))

        # 0 to 3
        paths = ges.utils.semi_directed_paths(0, 3, A)
        self.assertEqual(2, len(paths))
        self.assertTrue([0, 2, 3] in paths)
        self.assertTrue([0, 1, 2, 3] in paths)
        # 3 to 0
        paths = ges.utils.semi_directed_paths(3, 0, A)
        self.assertEqual(0, len(paths))

        # 1 to 2
        paths = ges.utils.semi_directed_paths(1, 2, A)
        self.assertEqual(1, len(paths))
        self.assertTrue([1, 2] in paths)
        # 2 to 1
        paths = ges.utils.semi_directed_paths(2, 1, A)
        self.assertEqual(1, len(paths))
        self.assertTrue([2, 1] in paths)

        # 1 to 3
        paths = ges.utils.semi_directed_paths(1, 3, A)
        self.assertEqual(1, len(paths))
        self.assertTrue([1, 2, 3] in paths)
        # 3 to 1
        paths = ges.utils.semi_directed_paths(3, 1, A)
        self.assertEqual(1, len(paths))
        self.assertTrue([3, 2, 1] in paths)

        # 2 to 3
        paths = ges.utils.semi_directed_paths(2, 3, A)
        self.assertEqual(1, len(paths))
        self.assertTrue([2, 3] in paths)
        # 3 to 2
        paths = ges.utils.semi_directed_paths(3, 2, A)
        self.assertEqual(1, len(paths))
        self.assertTrue([3, 2] in paths)

    def test_semi_directed_paths_2(self):
        # Test vs. networkx implementation
        G = 100
        p = 30
        for i in range(G):
            A = sempler.generators.dag_avg_deg(p, 3, 1, 1)
            cpdag = ges.utils.pdag_to_cpdag(A)
            G = nx.from_numpy_matrix(cpdag, create_using=nx.DiGraph)
            for (x, y) in itertools.combinations(range(p), 2):
                # From x to y
                paths_own = ges.utils.semi_directed_paths(x, y, cpdag)
                paths_nx = list(nx.algorithms.all_simple_paths(G, x, y))
                self.assertEqual(sorted(paths_nx), sorted(paths_own))
                # From y to x
                paths_own = ges.utils.semi_directed_paths(y, x, cpdag)
                paths_nx = list(nx.algorithms.all_simple_paths(G, y, x))
                self.assertEqual(sorted(paths_nx), sorted(paths_own))
        print("Checked path enumeration for %d PDAGs" % (i+1))

    def test_semi_directed_paths_3(self):
        A = np.array([[0, 1, 0, 0],
                      [1, 0, 1, 1],
                      [0, 1, 0, 1],
                      [0, 1, 1, 0]])
        G = nx.from_numpy_matrix(A, create_using=nx.DiGraph)
        for (x, y) in itertools.combinations(range(len(A)), 2):
            # From x to y
            paths_own = ges.utils.semi_directed_paths(x, y, A)
            paths_nx = list(nx.algorithms.all_simple_paths(G, x, y))
            self.assertEqual(sorted(paths_nx), sorted(paths_own))
            # From y to x
            paths_own = ges.utils.semi_directed_paths(y, x, A)
            paths_nx = list(nx.algorithms.all_simple_paths(G, y, x))
            self.assertEqual(sorted(paths_nx), sorted(paths_own))

    def test_skeleton(self):
        # Test utils.skeleton
        skeleton = np.array([[0, 1, 1, 0],
                             [1, 0, 1, 0],
                             [1, 1, 0, 1],
                             [0, 0, 1, 0]])
        # Test 0
        self.assertTrue((ges.utils.skeleton(skeleton) == skeleton).all())
        # Test 1
        A1 = np.array([[0, 1, 1, 0],
                       [0, 0, 1, 0],
                       [0, 0, 0, 1],
                       [0, 0, 0, 0]])
        self.assertTrue((ges.utils.skeleton(A1) == skeleton).all())
        # Test 2
        A2 = np.array([[0, 1, 1, 0],
                       [1, 0, 1, 0],
                       [0, 0, 0, 1],
                       [0, 0, 1, 0]])
        self.assertTrue((ges.utils.skeleton(A2) == skeleton).all())
        # Test 3
        A3 = np.array([[0, 1, 1, 0],
                       [0, 0, 1, 0],
                       [0, 0, 0, 1],
                       [0, 0, 1, 0]])
        self.assertTrue((ges.utils.skeleton(A3) == skeleton).all())

    def test_only_directed(self):
        # Test utils.only_directed
        # Undirected graph should result in empty graph
        A = np.array([[0, 1, 1, 0],
                      [1, 0, 1, 0],
                      [1, 1, 0, 1],
                      [0, 0, 1, 0]])
        self.assertTrue((ges.utils.only_directed(A) == np.zeros_like(A)).all())
        # Directed graph should return the same graph (maintaining weights)
        A = np.array([[0, 1, 1, 0],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1],
                      [0, 0, 0, 0]]) * np.random.uniform(size=A.shape)
        self.assertTrue((ges.utils.only_directed(A) == A).all())
        # Mixed graph should result in graph with only the directed edges
        A = np.array([[0, .5, 1, 0],
                      [0, 0, 1, 0],
                      [0, 1, 0, 1],
                      [0, 0, 1, 0]])
        truth = np.zeros_like(A)
        truth[0, 1], truth[0, 2] = 0.5, 1
        self.assertTrue((ges.utils.only_directed(A) == truth).all())

    def test_only_undirected(self):
        # Test utils.only_undirected
        # Undirected graph should result in the same graph (maintaining weights)
        A = np.array([[0, 1, 1, 0],
                      [1, 0, 1, 0],
                      [1, 1, 0, 1],
                      [0, 0, 1, 0]]) * np.random.uniform(size=(4, 4))
        self.assertTrue((ges.utils.only_undirected(A) == A).all())
        # Directed graph should return an empty graph
        A = np.array([[0, 1, 1, 0],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1],
                      [0, 0, 0, 0]])
        self.assertTrue((ges.utils.only_undirected(A) == np.zeros_like(A)).all())
        # Mixed graph should result in graph with only the directed edges
        A = np.array([[0, 1, 1, 0],
                      [0, 0, 1, 0],
                      [0, 1, 0, 1],
                      [0, 0, 1, 0]])
        truth = np.array([[0, 0, 0, 0],
                          [0, 0, 1, 0],
                          [0, 1, 0, 1],
                          [0, 0, 1, 0]])
        self.assertTrue((ges.utils.only_undirected(A) == truth).all())
        # Undirected and directed should be disjoint
        union = np.logical_xor(ges.utils.only_directed(A), ges.utils.only_undirected(A))
        self.assertTrue((union == A).all())

    def test_vstructures(self):
        # Test utils.vstructures
        # TODO: These tests do not contain any cases where (i,c,j)
        # with i > j and is saved as (j,c,i) instead
        # Undirected graph should yield no v_structures
        A = np.array([[0, 1, 1, 0],
                      [1, 0, 1, 0],
                      [1, 1, 0, 1],
                      [0, 0, 1, 0]])
        self.assertEqual(set(), ges.utils.vstructures(A))
        # Fully directed graph 1
        A = np.array([[0, 1, 1, 0],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1],
                      [0, 0, 0, 0]])
        self.assertEqual(set(), ges.utils.vstructures(A))
        # Fully directed graph 2
        A = np.array([[0, 1, 1, 0],
                      [0, 0, 1, 0],
                      [0, 0, 0, 0],
                      [0, 0, 1, 0]])
        self.assertEqual({(1, 2, 3), (0, 2, 3)}, ges.utils.vstructures(A))
        # Fully directed graph 3
        A = np.array([[0, 0, 1],
                      [0, 0, 1],
                      [0, 0, 0]])
        self.assertEqual({(0, 2, 1)}, ges.utils.vstructures(A))
        # Mixed graph 1
        A = np.array([[0, 1, 1, 0],
                      [1, 0, 1, 0],
                      [0, 0, 0, 0],
                      [0, 0, 1, 0]])
        self.assertEqual({(1, 2, 3), (0, 2, 3)}, ges.utils.vstructures(A))
        # Mixed graph 2
        A = np.array([[0, 1, 1, 0],
                      [1, 0, 1, 0],
                      [0, 0, 0, 1],
                      [0, 0, 0, 0]])
        self.assertEqual(set(), ges.utils.vstructures(A))
        # Mixed graph 3
        A = np.array([[0, 1, 1, 0],
                      [1, 0, 1, 0],
                      [0, 0, 0, 1],
                      [0, 0, 1, 0]])
        self.assertEqual(set(), ges.utils.vstructures(A))
        # Mixed graph 4
        A = np.array([[0, 0, 1],
                      [0, 0, 1],
                      [0, 1, 0]])
        self.assertEqual(set(), ges.utils.vstructures(A))

    def test_is_consistent_extension_precondition(self):
        P = np.array([[0, 1, 1, 0],
                      [1, 0, 1, 0],
                      [0, 1, 0, 1],
                      [0, 0, 1, 0]])
        # Exception if A is not a DAG (has cycle)
        A = P.copy()
        A[1, 2], A[0, 1] = 0, 0
        try:
            ges.utils.is_consistent_extension(A, P)
            self.fail()
        except ValueError as e:
            print("OK:", e)
        # Exception if A is not a DAG (has undirected edges)
        A = P.copy()
        A[2, 1], A[1, 0] = 0, 0
        try:
            ges.utils.is_consistent_extension(A, P)
            self.fail()
        except ValueError as e:
            print("OK:", e)

    def test_is_consistent_extension_1(self):
        P = np.array([[0, 1, 1, 0],
                      [1, 0, 1, 0],
                      [0, 1, 0, 1],
                      [0, 0, 1, 0]])
        # Should return True
        A = P.copy()
        A[2, 1], A[1, 0], A[3, 2] = 0, 0, 0
        self.assertTrue(ges.utils.is_consistent_extension(A, P))
        # Should return False (vstructs (0,2,3) and (1,2,3))
        A = P.copy()
        A[2, 1], A[1, 0], A[2, 3] = 0, 0, 0
        self.assertFalse(ges.utils.is_consistent_extension(A, P))
        # Should return False (vstructs (0,2,3))
        A = P.copy()
        A[1, 2], A[1, 0], A[2, 3] = 0, 0, 0
        self.assertFalse(ges.utils.is_consistent_extension(A, P))
        # Should return False (different skeleton)
        A = P.copy()
        A[2, 1], A[1, 0], A[3, 2] = 0, 0, 0
        A[1, 3] = 1
        self.assertFalse(ges.utils.is_consistent_extension(A, P))
        # Should return False (different orientation)
        A = P.copy()
        A[2, 1], A[3, 2] = 0, 0
        A[0, 1] = 0
        A[0, 2], A[2, 0] = 0, 1
        self.assertFalse(ges.utils.is_consistent_extension(A, P))

    def test_is_consistent_extension_2(self):
        P = np.array([[0, 1, 1, 0],
                      [1, 0, 1, 0],
                      [0, 0, 0, 1],
                      [0, 0, 1, 0]])
        # There are four extensions, two of which are consistent (same v-structures)
        # Extension 1 (consistent)
        A = np.array([[0, 0, 1, 0],
                      [1, 0, 1, 0],
                      [0, 0, 0, 1],
                      [0, 0, 0, 0]])
        self.assertTrue(ges.utils.is_consistent_extension(A, P))
        # Extension 2 (consistent)
        A = np.array([[0, 1, 1, 0],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1],
                      [0, 0, 0, 0]])
        self.assertTrue(ges.utils.is_consistent_extension(A, P))
        # Extension 3 (not consistent)
        A = np.array([[0, 0, 1, 0],
                      [1, 0, 1, 0],
                      [0, 0, 0, 0],
                      [0, 0, 1, 0]])
        self.assertFalse(ges.utils.is_consistent_extension(A, P))
        # Extension 4 (not consistent)
        A = np.array([[0, 1, 1, 0],
                      [0, 0, 1, 0],
                      [0, 0, 0, 0],
                      [0, 0, 1, 0]])
        self.assertFalse(ges.utils.is_consistent_extension(A, P))

    def test_separates_preconditions(self):
        A = np.array([[0, 0, 1, 0, 0],
                      [0, 0, 1, 0, 0],
                      [1, 1, 0, 1, 1],
                      [0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0]])
        # S and A are not disjoint
        try:
            ges.utils.separates({1}, {1, 2}, {3, 4}, A)
            self.fail("Exception should have been thrown")
        except ValueError as e:
            print("OK:", e)
        # S and B are not disjoint
        try:
            ges.utils.separates({0, 1}, {2}, {0, 3, 4}, A)
            self.fail("Exception should have been thrown")
        except ValueError as e:
            print("OK:", e)
        # A and B are not disjoint
        try:
            ges.utils.separates({0, 1}, {2, 3}, {3, 4}, A)
            self.fail("Exception should have been thrown")
        except ValueError as e:
            print("OK:", e)
        # None are disjoint
        try:
            ges.utils.separates({0, 1}, {0, 2, 3}, {1, 3, 4}, A)
            self.fail("Exception should have been thrown")
        except ValueError as e:
            print("OK:", e)

    def test_separates_1(self):
        A = np.array([[0, 0, 1, 0, 0],
                      [0, 0, 1, 0, 0],
                      [1, 1, 0, 1, 1],
                      [0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0]])
        self.assertTrue(ges.utils.separates({2}, {0, 1}, {3, 4}, A))
        self.assertTrue(ges.utils.separates({2}, {3, 4}, {0, 1}, A))
        self.assertFalse(ges.utils.separates(set(), {0, 1}, {3, 4}, A))
        self.assertTrue(ges.utils.separates(set(), {3, 4}, {0, 1}, A))
        self.assertTrue(ges.utils.separates(set(), {3}, {0}, A))
        self.assertFalse(ges.utils.separates(set(), {0}, {3}, A))
        self.assertTrue(ges.utils.separates({2}, {0}, {3}, A))

    def test_separates_2(self):
        A = np.array([[0, 0, 1, 0, 0],
                      [0, 0, 1, 0, 1],
                      [1, 1, 0, 1, 1],
                      [0, 0, 1, 0, 1],
                      [0, 0, 1, 1, 0]])
        self.assertFalse(ges.utils.separates({2}, {0, 1}, {3, 4}, A))
        self.assertTrue(ges.utils.separates({2}, {0}, {3, 4}, A))
        self.assertTrue(ges.utils.separates({2, 1}, {0}, {3, 4}, A))
        self.assertTrue(ges.utils.separates({2, 4}, {1}, {3}, A))

    def test_chain_component_1(self):
        G = np.array([[0, 0, 1, 0, 0],
                      [0, 0, 1, 0, 0],
                      [0, 0, 0, 1, 1],
                      [0, 0, 1, 0, 0],
                      [0, 0, 1, 0, 0]])
        chain_components = [(0, {0}),
                            (1, {1}),
                            (2, {2, 3, 4}),
                            (3, {2, 3, 4}),
                            (4, {2, 3, 4})]
        for (i, truth) in chain_components:
            self.assertEqual(truth, ges.utils.chain_component(i, G))

    def test_chain_component_2(self):
        G = np.array([[0, 1, 0, 1, 0, 0, 0, 0],
                      [0, 0, 0, 1, 0, 0, 0, 0],
                      [0, 1, 0, 0, 0, 0, 0, 0],
                      [1, 1, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 1, 0, 1, 1, 1],
                      [0, 0, 0, 0, 1, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 1, 1, 1, 0]])
        chain_components = [(0, {0, 1, 3}),
                            (1, {0, 1, 3}),
                            (2, {2}),
                            (3, {0, 1, 3}),
                            (4, {4, 5, 7}),
                            (5, {4, 5, 7}),
                            (6, {6}),
                            (7, {4, 5, 7})]
        for (i, truth) in chain_components:
            self.assertEqual(truth, ges.utils.chain_component(i, G))

    def test_chain_component_3(self):
        # Check the following property in random graphs:
        # if i is in the chain component of j, then the chain
        # component of i is equal to that of j
        G = 500
        p = 20
        for i in range(G):
            A = sempler.generators.dag_avg_deg(p, 3, 1, 1)
            cpdag = ges.utils.dag_to_cpdag(A)
            for j in range(p):
                chain_component = ges.utils.chain_component(j, cpdag)
                for h in chain_component:
                    self.assertEqual(chain_component, ges.utils.chain_component(h, cpdag))

    def test_chain_component_4(self):
        # Check that in a directed graph, the chain component of each
        # node is itself
        G = 500
        p = 20
        for i in range(G):
            A = sempler.generators.dag_avg_deg(p, 3, 1, 1)
            for j in range(p):
                self.assertEqual({j}, ges.utils.chain_component(j, A))

    def test_induced_graph_1(self):
        # Test that
        # 1. The subgraph induced by an empty set of nodes should always
        # be a disconnected graph
        # 2. When the set is not empty, the returned "subgraph" is correct
        G = 500
        p = 20
        for i in range(G):
            A = sempler.generators.dag_avg_deg(p, 3, 1, 1)
            G = ges.utils.dag_to_cpdag(A)
            # Test 1
            self.assertTrue((np.zeros_like(G) == ges.utils.induced_subgraph(set(), G)).all())
            # Test 2
            for _ in range(10):
                S = set(np.random.choice(range(p), size=np.random.randint(0, p)))
                truth = G.copy()
                Sc = set(range(p)) - S
                truth[list(Sc), :] = 0
                truth[:, list(Sc)] = 0
                self.assertTrue((truth == ges.utils.induced_subgraph(S, G)).all())

    def test_induced_graph_2(self):
        G = np.array([[0, 0, 1, 0, 0],
                      [0, 0, 1, 0, 0],
                      [0, 0, 0, 1, 1],
                      [0, 0, 1, 0, 0],
                      [0, 0, 1, 0, 0]])
        # Test 0: Sets which return a graph with no edges
        for S in [set(), {0}, {1}, {2}, {3}, {4}, {0, 1}, {0, 3}, {0, 4}, {3, 4}, {1, 3}, {1, 4}]:
            self.assertTrue((np.zeros_like(G) == ges.utils.induced_subgraph(S, G)).all())
        # Test 1
        S = {0, 1, 2}
        truth = np.array([[0, 0, 1, 0, 0],
                          [0, 0, 1, 0, 0],
                          [0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0]])
        self.assertTrue((truth == ges.utils.induced_subgraph(S, G)).all())
        # Test 2
        S = {0, 2, 3}
        truth = np.array([[0, 0, 1, 0, 0],
                          [0, 0, 0, 0, 0],
                          [0, 0, 0, 1, 0],
                          [0, 0, 1, 0, 0],
                          [0, 0, 0, 0, 0]])
        self.assertTrue((truth == ges.utils.induced_subgraph(S, G)).all())
        # Test 3
        S = {1, 2, 3, 4}
        truth = np.array([[0, 0, 0, 0, 0],
                          [0, 0, 1, 0, 0],
                          [0, 0, 0, 1, 1],
                          [0, 0, 1, 0, 0],
                          [0, 0, 1, 0, 0]])
        self.assertTrue((truth == ges.utils.induced_subgraph(S, G)).all())

    # Tests for other auxiliary functions

    def test_sort_1(self):
        # Test that without order, behaviour is identical as python's
        # sorted
        for _ in range(10):
            L = np.random.permutation(range(10))
            self.assertEqual(ges.utils.sort(L), sorted(L))
        # Test that (1) when the order is a permutation of the list,
        # the result is the order itself, and (2) when applied to an
        # empty list, sorted is the identity
        for _ in range(10):
            L = np.random.permutation(100)
            order = list(np.random.permutation(100))
            self.assertEqual(order, ges.utils.sort(L, order))
            self.assertEqual([], ges.utils.sort([], order))

    def test_sort_2(self):
        # Check that ordering works as expected when order is specified
        # Test 1
        L = [3, 6, 1, 2, 0]
        order = [0, 2, 4, 6, 1, 3, 5]
        self.assertEqual([0, 2, 6, 1, 3], ges.utils.sort(L, order))
        # Test 2, with duplicated elements
        L = [0, 1, 0, 6, 1, 9, 9, 4]
        order = [0, 2, 4, 6, 8, 1, 3, 5, 7, 9]
        self.assertEqual([0, 0, 4, 6, 1, 1, 9, 9], ges.utils.sort(L, order))
        # Test 3, with different order
        L = [8, 8, 1, 9, 7, 1, 3, 0, 2, 4, 0, 1, 3, 7, 5]
        order = [7, 3, 6, 5, 0, 4, 1, 2, 6, 8, 9]
        truth = [7, 7, 3, 3, 5, 0, 0, 4, 1, 1, 1, 2, 8, 8, 9]
        self.assertEqual(truth, ges.utils.sort(L, order))
