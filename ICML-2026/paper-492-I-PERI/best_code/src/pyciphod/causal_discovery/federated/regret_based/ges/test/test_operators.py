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
"""

import unittest
import numpy as np
import sempler
import sempler.generators

import ges
import ges.utils as utils
from ges.scores.gauss_obs_l0_pen import GaussObsL0Pen

# ---------------------------------------------------------------------
# Tests for the insert operator


class InsertOperatorTests(unittest.TestCase):
    true_A = np.array([[0, 0, 1, 0, 0],
                       [0, 0, 1, 0, 0],
                       [0, 0, 0, 1, 1],
                       [0, 0, 0, 0, 1],
                       [0, 0, 0, 0, 0]])
    factorization = [(4, (2, 3)), (3, (2,)), (2, (0, 1)), (0, ()), (1, ())]
    true_B = true_A * np.random.uniform(1, 2, size=true_A.shape)
    scm = sempler.LGANM(true_B, (0, 0), (0.3, 0.4))
    p = len(true_A)
    n = 10000
    obs_data = scm.sample(n=n)

    # ------------------------------------------------------
    # Tests
    def test_insert_1(self):
        # Test behaviour of the ges.insert(x,y,T) function

        # Insert should fail on adjacent edges
        try:
            ges.insert(0, 2, set(), self.true_A)
            self.fail("Call to insert should have failed")
        except ValueError as e:
            print("OK:", e)

        # Insert should fail when T contains non-neighbors of y
        try:
            ges.insert(0, 1, {2}, self.true_A)
            self.fail("Call to insert should have failed")
        except ValueError as e:
            print("OK:", e)

        # Insert should fail when T contains adjacents of x
        A = np.zeros_like(self.true_A)
        A[2, 4], A[4, 2] = 1, 1  # x2 - x4
        A[4, 3] = 1  # x4 -> x3
        try:
            ges.insert(3, 2, {4}, A)
            self.fail("Call to insert should have failed")
        except ValueError as e:
            print("OK:", e)

        # This should work
        true_new_A = A.copy()
        true_new_A[3, 2] = 1
        new_A = ges.insert(3, 2, set(), A)
        self.assertTrue((true_new_A == new_A).all())

        # This should work
        true_new_A = A.copy()
        true_new_A[1, 2] = 1
        new_A = ges.insert(1, 2, set(), A)
        self.assertTrue((true_new_A == new_A).all())

        # This should work
        true_new_A = A.copy()
        true_new_A[1, 2] = 1
        true_new_A[4, 2], true_new_A[2, 4] = 1, 0
        new_A = ges.insert(1, 2, {4}, A)
        self.assertTrue((true_new_A == new_A).all())

        # This should work
        true_new_A = A.copy()
        true_new_A[1, 0] = 1
        new_A = ges.insert(1, 0, set(), A)
        self.assertTrue((true_new_A == new_A).all())

    def test_insert_2(self):
        G = 100
        p = 20
        for i in range(G):
            A = sempler.generators.dag_avg_deg(p, 3, 1, 1)
            cpdag = utils.dag_to_cpdag(A)
            for x in range(p):
                # Can only apply the operator to non-adjacent nodes
                adj_x = utils.adj(x, cpdag)
                Y = set(range(p)) - adj_x
                for y in Y:
                    for T in utils.subsets(utils.neighbors(y, cpdag) - adj_x):
                        # print(x,y,T)
                        output = ges.insert(x, y, T, cpdag)
                        # Verify the new vstructures
                        vstructs = utils.vstructures(output)
                        for t in T:
                            vstruct = (x, y, t) if x < t else (t, y, x)
                            self.assertIn(vstruct, vstructs)
                        # Verify whole connectivity
                        truth = cpdag.copy()
                        # Add edge x -> y
                        truth[x, y] = 1
                        # Orient t -> y
                        truth[list(T), y] = 1
                        truth[y, list(T)] = 0
                        self.assertTrue((output == truth).all())
        print("\nExhaustively checked insert operator on %i CPDAGS" % (i + 1))

    def test_valid_insert_operators_1a(self):
        # Define A and cache
        A = np.zeros_like(self.true_A)
        A[2, 4], A[4, 2] = 1, 1  # x2 - x4
        A[4, 3] = 1  # x4 -> x3
        cache = GaussObsL0Pen(self.obs_data)
        # there should only be one valid operator, as
        #   1. X1 has no neighbors in A, so T0 = {set()}
        #   2. na_yx is also an empty set, thus na_yx U T is a clique
        #   3. there are no semi-directed paths from y to x
        valid_operators = ges.score_valid_insert_operators(0, 1, A, cache, debug=False)
        self.assertEqual(1, len(valid_operators))
        _, new_A, _, _, _ = valid_operators[0]
        true_new_A = A.copy()
        true_new_A[0, 1] = 1
        self.assertTrue((new_A == true_new_A).all())

    def test_valid_insert_operators_1b(self):
        # Define A and cache
        A = np.zeros_like(self.true_A)
        A[2, 4], A[4, 2] = 1, 1  # x2 - x4
        A[4, 3] = 1  # x4 -> x3
        cache = GaussObsL0Pen(self.obs_data)
        # there should only be one valid operator, as
        #   1. X0 has no neighbors in A, so T0 = {set()}
        #   2. na_yx is also an empty set, thus na_yx U T is a clique
        #   3. there are no semi-directed paths from y to x
        valid_operators = ges.score_valid_insert_operators(1, 0, A, cache, debug=False)
        self.assertEqual(1, len(valid_operators))
        _, new_A, _, _, _ = valid_operators[0]
        true_new_A = A.copy()
        true_new_A[1, 0] = 1
        self.assertTrue((new_A == true_new_A).all())

    def test_valid_insert_operators_2a(self):
        # Define A and cache
        A = np.zeros_like(self.true_A)
        A[2, 4], A[4, 2] = 1, 1  # x2 - x4
        A[4, 3] = 1  # x4 -> x3
        cache = GaussObsL0Pen(self.obs_data)
        # there should be two valid operators, as T0 = {X4}
        #   1. insert(X0,X2,set()) should be valid
        #   2. and also insert(X0,X2,{X4}), as na_yx U T = {X4} and is a clique
        #   3. there are no semi-directed paths from y to x
        valid_operators = ges.score_valid_insert_operators(0, 2, A, cache, debug=False)
        self.assertEqual(2, len(valid_operators))
        # Test outcome of insert(0,2,set())
        _, new_A, _, _, _ = valid_operators[0]
        true_new_A = A.copy()
        true_new_A[0, 2] = 1
        self.assertTrue((new_A == true_new_A).all())
        # Test outcome of insert(0,2,4)
        _, new_A, _, _, _ = valid_operators[1]
        true_new_A = A.copy()
        true_new_A[0, 2], true_new_A[2, 4] = 1, 0
        self.assertTrue((new_A == true_new_A).all())

    def test_valid_insert_operators_2b(self):
        # Define A and cache
        A = np.zeros_like(self.true_A)
        A[2, 4], A[4, 2] = 1, 1  # x2 - x4
        A[4, 3] = 1  # x4 -> x3
        cache = GaussObsL0Pen(self.obs_data)
        # there should only be one valid operator, as
        #   1. X0 has no neighbors in A, so T0 = {set()}
        #   2. na_yx is also an empty set, thus na_yx U T is a clique
        #   3. there are no semi-directed paths from y to x
        valid_operators = ges.score_valid_insert_operators(2, 0, A, cache, debug=False)
        self.assertEqual(1, len(valid_operators))
        # Test outcome of insert(2,0,set())
        _, new_A, _, _, _ = valid_operators[0]
        true_new_A = A.copy()
        true_new_A[2, 0] = 1
        self.assertTrue((new_A == true_new_A).all())

    def test_valid_insert_operators_3a(self):
        # Define A and cache
        A = np.zeros_like(self.true_A)
        A[2, 4], A[4, 2] = 1, 1  # x2 - x4
        A[4, 3] = 1  # x4 -> x3
        cache = GaussObsL0Pen(self.obs_data)
        # there should be two valid operators, as T0 = {X4}
        #   1. insert(X1,X2,set()) should be valid
        #   2. and also insert(X1,X2,{X4}), as na_yx U T = {X4} and is a clique
        #   3. there are no semi-directed paths from y to x
        valid_operators = ges.score_valid_insert_operators(1, 2, A, cache, debug=False)
        self.assertEqual(2, len(valid_operators))
        # Test outcome of insert(0,2,set())
        _, new_A, _, _, _ = valid_operators[0]
        true_new_A = A.copy()
        true_new_A[1, 2] = 1
        self.assertTrue((new_A == true_new_A).all())
        # Test outcome of insert(1,2,4)
        _, new_A, _, _, _ = valid_operators[1]
        true_new_A = A.copy()
        true_new_A[1, 2], true_new_A[2, 4] = 1, 0
        self.assertTrue((new_A == true_new_A).all())

    def test_valid_insert_operators_3b(self):
        # Define A and cache
        A = np.zeros_like(self.true_A)
        A[2, 4], A[4, 2] = 1, 1  # x2 - x4
        A[4, 3] = 1  # x4 -> x3
        cache = GaussObsL0Pen(self.obs_data)
        # there should only be one valid operator, as
        #   1. X1 has no neighbors in A, so T0 = {set()}
        #   2. na_yx is also an empty set, thus na_yx U T is a clique
        #   3. there are no semi-directed paths from y to x
        valid_operators = ges.score_valid_insert_operators(2, 1, A, cache, debug=False)
        self.assertEqual(1, len(valid_operators))
        # Test outcome of insert(2,0,set())
        _, new_A, _, _, _ = valid_operators[0]
        true_new_A = A.copy()
        true_new_A[2, 1] = 1
        self.assertTrue((new_A == true_new_A).all())

    def test_valid_insert_operators_4a(self):
        # Define A and cache
        A = np.zeros_like(self.true_A)
        A[2, 4], A[4, 2] = 1, 1  # x2 - x4
        A[4, 3] = 1  # x4 -> x3
        cache = GaussObsL0Pen(self.obs_data)
        # there should be one valid operator, as T0 = set(), na_yx = {4}
        #   1. insert(X0,X2,set()) should be valid
        #   2. na_yx U T = {X4} should be a clique
        #   3. the semi-directed path X2-X4->X3 contains one node in na_yx U T
        valid_operators = ges.score_valid_insert_operators(3, 2, A, cache, debug=False)
        self.assertEqual(1, len(valid_operators))
        # Test outcome of insert(3,2,set())
        _, new_A, _, _, _ = valid_operators[0]
        true_new_A = A.copy()
        true_new_A[3, 2] = 1
        self.assertTrue((new_A == true_new_A).all())

    def test_valid_insert_operators_4b(self):
        # Define A and cache
        A = np.zeros_like(self.true_A)
        A[2, 4], A[4, 2] = 1, 1  # x2 - x4
        A[4, 3] = 1  # x4 -> x3
        cache = GaussObsL0Pen(self.obs_data)
        # there should be one valid operator, as T0 = set(), na_yx = set()
        #   1. insert(X2,X3,set()) should be valid
        #   2. na_yx U T = set() should be a clique
        #   3. there are no semi-directed paths between X3 and X2
        valid_operators = ges.score_valid_insert_operators(2, 3, A, cache, debug=False)
        self.assertEqual(1, len(valid_operators))
        # Test outcome of insert(2,3,set())
        _, new_A, _, _, _ = valid_operators[0]
        true_new_A = A.copy()
        true_new_A[2, 3] = 1
        self.assertTrue((new_A == true_new_A).all())

    def test_valid_insert_operators_5(self):
        # Define A and cache
        A = np.zeros_like(self.true_A)
        A[2, 4], A[4, 2] = 1, 1  # x2 - x4
        A[4, 3] = 1  # x4 -> x3
        cache = GaussObsL0Pen(self.obs_data)
        # Should fail as 2,4 are adjacent
        try:
            ges.score_valid_insert_operators(2, 4, A, cache, debug=False)
            self.fail()
        except ValueError as e:
            print("OK:", e)
        try:
            ges.score_valid_insert_operators(4, 2, A, cache, debug=False)
            self.fail()
        except ValueError as e:
            print("OK:", e)
            # Should fail as 3,4 are adjacent
        try:
            ges.score_valid_insert_operators(3, 4, A, cache, debug=False)
            self.fail()
        except ValueError as e:
            print("OK:", e)
        try:
            ges.score_valid_insert_operators(4, 3, A, cache, debug=False)
            self.fail()
        except ValueError as e:
            print("OK:", e)

    def test_valid_insert_operators_6(self):
        A = np.array([[0, 0, 1, 0],
                      [0, 0, 1, 1],
                      [1, 1, 0, 0],
                      [0, 0, 0, 0]])
        data = self.obs_data[:, 0:4]
        cache = GaussObsL0Pen(data)
        # There should be one valid operator for x3 -> x2
        #   1. na_yx = {1}, T0 = {0}
        #   2. for T=set(), na_yx U T = {1} which is a clique, and
        #   contains a node in the semi-directed path 2-1->3
        #   3. for T = {0}, na_yx U T = {0,1} which is not a clique
        valid_operators = ges.score_valid_insert_operators(3, 2, A, cache, debug=False)
        self.assertEqual(1, len(valid_operators))
        # Test outcome of insert(3,2,set())
        _, new_A, _, _, _ = valid_operators[0]
        true_new_A = A.copy()
        true_new_A[3, 2] = 1
        self.assertTrue((new_A == true_new_A).all())

    def test_valid_insert_operators_7(self):
        A = np.array([[0, 0, 1, 0],
                      [0, 0, 0, 1],
                      [1, 1, 0, 0],
                      [0, 0, 0, 0]])
        data = self.obs_data[:, 0:4]
        cache = GaussObsL0Pen(data)
        # There should no valid operator for x3 -> x2
        #   1. na_yx = set(), T0 = {0}
        #   2. for T=set(), na_yx U T = set() which is a clique, but does not
        #   contain a node in the semi-directed path 2->1->3
        #   3. for T = {0}, na_yx U T = {0} which is a clique, but does not
        #   contain a node in the semi-directed path 2->1->3
        valid_operators = ges.score_valid_insert_operators(3, 2, A, cache, debug=False)
        self.assertEqual(0, len(valid_operators))

    def test_valid_insert_operators_8(self):
        A = np.array([[0, 0, 1, 0],
                      [0, 0, 0, 1],
                      [1, 1, 0, 0],
                      [0, 0, 0, 0]])
        data = self.obs_data[:, 0:4]
        cache = GaussObsL0Pen(data)
        # There should no valid operator for x2 -> x3
        #   1. na_yx = set(), T0 = {0}
        #   2. for T=set(), na_yx U T = set() which is a clique, but does not
        #   contain a node in the semi-directed path 2->1->3
        #   3. for T = {0}, na_yx U T = {0} which is a clique, but does not
        #   contain a node in the semi-directed path 2->1->3
        valid_operators = ges.score_valid_insert_operators(3, 2, A, cache, debug=False)
        self.assertEqual(0, len(valid_operators))

# --------------------------------------------------------------------
# Tests for the delete(x,y,H) operator


class DeleteOperatorTests(unittest.TestCase):
    true_A = np.array([[0, 0, 1, 0, 0],
                       [0, 0, 1, 0, 0],
                       [0, 0, 0, 1, 1],
                       [0, 0, 0, 0, 1],
                       [0, 0, 0, 0, 0]])
    factorization = [(4, (2, 3)), (3, (2,)), (2, (0, 1)), (0, ()), (1, ())]
    true_B = true_A * np.random.uniform(1, 2, size=true_A.shape)
    scm = sempler.LGANM(true_B, (0, 0), (0.3, 0.4))
    p = len(true_A)
    n = 10000
    obs_data = scm.sample(n=n)
    # ------------------------------------------------------
    # Tests

    def test_delete_operator_preconditions(self):
        # Test that if x and y are not adjacent an exception is thrown
        try:
            ges.delete(0, 1, set(), self.true_A)
            self.fail("Call to delete should have failed")
        except ValueError as e:
            print("OK", e)
        try:
            ges.delete(3, 0, set(), self.true_A)
            self.fail("Call to delete should have failed")
        except ValueError as e:
            print("OK", e)
        # Test that if there is no edge x -> y or x - y an exception
        # is thrown
        try:
            ges.delete(3, 2, set(), self.true_A)
            self.fail("Call to delete should have failed")
        except ValueError as e:
            print("OK", e)
        try:
            ges.delete(2, 1, set(), self.true_A)
            self.fail("Call to delete should have failed")
        except ValueError as e:
            print("OK", e)
        # Test that if H is not a subset of neighbors of Y and
        # adjacents of X, an exception is thrown
        cpdag = self.true_A.copy()
        cpdag[4, 3] = 1
        try:
            ges.delete(2, 4, {1}, cpdag)
            self.fail("Call to delete should have failed")
        except ValueError as e:
            print("OK", e)
        try:
            ges.delete(2, 4, {0}, cpdag)
            self.fail("Call to delete should have failed")
        except ValueError as e:
            print("OK", e)
        try:
            ges.delete(4, 2, {3}, cpdag)
            self.fail("Call to delete should have failed")
        except ValueError as e:
            print("OK", e)

    def test_delete_operator_1(self):
        # Test the result from applying the delete operator to a
        # hand-picked matrix
        A = np.array([[0, 0, 1, 0, 0],
                      [0, 0, 1, 0, 0],
                      [0, 0, 0, 1, 1],
                      [0, 0, 0, 0, 1],
                      [0, 0, 0, 1, 0]])
        # remove the edge 2 -> 4, with H = set()
        new_A = ges.delete(2, 4, set(), A)
        truth = A.copy()
        truth[2, 4] = 0
        self.assertTrue((truth == new_A).all())
        # remove the edge 2 -> 4, with H = {3}
        new_A = ges.delete(2, 4, {3}, A)
        truth = A.copy()
        truth[2, 4] = 0
        truth[3, 4] = 0
        self.assertTrue((truth == new_A).all())

    def test_delete_operator_2(self):
        # Test the result from applying the delete operator to a
        # hand-picked matrix
        A = np.array([[0, 1, 1, 1, 0],
                      [1, 0, 1, 0, 0],
                      [1, 1, 0, 1, 0],
                      [1, 0, 1, 0, 1],
                      [0, 0, 0, 1, 0]])
        # remove the edge 0 - 1, with H = set()
        new_A = ges.delete(0, 1, set(), A)
        truth = A.copy()
        truth[1, 0], truth[0, 1] = 0, 0
        self.assertTrue((truth == new_A).all())
        # remove the edge 0 -> 1, with H = {2}
        new_A = ges.delete(0, 1, {2}, A)
        truth = A.copy()
        truth[1, 0], truth[0, 1] = 0, 0
        truth[2, 0], truth[2, 1] = 0, 0
        print(new_A)
        self.assertTrue((truth == new_A).all())
        # remove the edge 0 - 2 with H = set()
        new_A = ges.delete(0, 2, set(), A)
        truth = A.copy()
        truth[0, 2], truth[2, 0] = 0, 0
        self.assertTrue((truth == new_A).all())
        # remove the edge 0 - 2 with H = {1}
        new_A = ges.delete(0, 2, {1}, A)
        truth = A.copy()
        truth[0, 2], truth[2, 0] = 0, 0
        truth[1, 0], truth[1, 2] = 0, 0
        self.assertTrue((truth == new_A).all())
        # remove the edge 0 - 2 with H = {1,3}
        new_A = ges.delete(0, 2, {1, 3}, A)
        truth = A.copy()
        truth[0, 2], truth[2, 0] = 0, 0
        truth[1, 0], truth[1, 2] = 0, 0
        truth[3, 0], truth[3, 2], truth[1, 0], truth[1, 2] = 0, 0, 0, 0
        self.assertTrue((truth == new_A).all())

    def test_delete_operator_3(self):
        G = 100
        p = 20
        for i in range(G):
            A = sempler.generators.dag_avg_deg(p, 3, 1, 1)
            cpdag = utils.dag_to_cpdag(A)
            for x in range(p):
                # Can only apply the operator to X -> Y or X - Y
                for y in np.where(cpdag[x, :] != 0)[0]:
                    for H in utils.subsets(utils.na(y, x, cpdag)):
                        output = ges.delete(x, y, H, cpdag)
                        # Verify the new vstructures
                        vstructs = utils.vstructures(output)
                        for h in H:
                            vstruct = (x, h, y) if x < y else (y, h, x)
                            self.assertIn(vstruct, vstructs)
                        # Verify whole connectivity
                        truth = cpdag.copy()
                        # Remove edge
                        truth[x, y], truth[y, x] = 0, 0
                        # Orient y -> h
                        truth[list(H), y] = 0
                        truth[list(utils.neighbors(x, cpdag) & H), x] = 0
                        self.assertTrue((output == truth).all())
        print("\nExhaustively checked delete operator on %i CPDAGS" % (i + 1))

    def test_valid_delete_operators_preconditions(self):
        A = np.array([[0, 0, 1, 0, 0],
                      [0, 0, 1, 0, 0],
                      [0, 0, 0, 1, 1],
                      [0, 0, 1, 0, 1],
                      [0, 0, 1, 1, 0]])
        # Should fail as X0 and X1 are not adjacent
        try:
            ges.delete(0, 1, set(), A)
            self.fail("Call to delete should have failed")
        except ValueError as e:
            print("OK", e)
        # Should fail as X0 and X1 are not adjacent
        try:
            ges.delete(1, 0, set(), A)
            self.fail("Call to delete should have failed")
        except ValueError as e:
            print("OK", e)
        # Should fail as X0 and X3 are not adjacent
        try:
            ges.delete(0, 3, set(), A)
            self.fail("Call to delete should have failed")
        except ValueError as e:
            print("OK", e)
        # Should fail as there is no edge X2 -> X0 or X2 - X0
        try:
            ges.delete(2, 0, set(), A)
            self.fail("Call to delete should have failed")
        except ValueError as e:
            print("OK", e)
        # Should fail as there is no edge X2 -> X1 or X2 - X1
        try:
            ges.delete(2, 1, set(), A)
            self.fail("Call to delete should have failed")
        except ValueError as e:
            print("OK", e)

    def test_valid_delete_operators_1(self):
        A = np.array([[0, 0, 1, 0, 0],
                      [0, 0, 1, 0, 0],
                      [0, 0, 0, 1, 1],
                      [0, 0, 1, 0, 1],
                      [0, 0, 1, 1, 0]])
        cache = GaussObsL0Pen(self.obs_data)
        # Removing the edge X2 - X4 should yield two valid
        # operators, for:
        #   1. H = Ø, as NA_yx \ Ø = {X3} is a clique
        #   2. H = {3}, as NA_yx \ {X3} = Ø is a clique
        output = ges.score_valid_delete_operators(2, 4, A, cache)
        self.assertEqual(2, len(output))
        A1, A2 = A.copy(), A.copy()
        # Remove X2 - X4
        A1[2, 4], A1[4, 2], A2[2, 4], A2[4, 2] = 0, 0, 0, 0
        # Orient X2 -> X3, X4 -> X3
        A2[3, 2], A2[3, 4] = 0, 0
        self.assertTrue(utils.member([op[1] for op in output], A1) is not None)
        self.assertTrue(utils.member([op[1] for op in output], A2) is not None)

    def test_valid_delete_operators_2(self):
        A = np.array([[0, 0, 1, 0, 0],
                      [0, 0, 1, 0, 0],
                      [0, 0, 0, 1, 1],
                      [0, 0, 1, 0, 1],
                      [0, 0, 1, 1, 0]])
        cache = GaussObsL0Pen(self.obs_data)
        # Removing the edge X1 - X2 should yield one valid
        # operator, for:
        #   1. H = Ø, as NA_yx \ Ø = {X3, X4} is a clique
        output = ges.score_valid_delete_operators(1, 2, A, cache)
        self.assertEqual(1, len(output))
        true_A = A.copy()
        # Remove X1 - X2
        true_A[1, 2] = 0
        self.assertTrue((true_A == output[0][1]).all())

    def test_valid_delete_operators_3(self):
        # Check symmetry of the delete operator when X - Y
        G = 100
        p = 20
        for i in range(G):
            A = sempler.generators.dag_avg_deg(p, 3, 1, 1)
            cpdag = utils.dag_to_cpdag(A)
            W = A * np.random.uniform(1, 2, A.shape)
            obs_sample = sempler.LGANM(W, (0, 0), (0.5, 1)).sample(n=1000)
            cache = GaussObsL0Pen(obs_sample)
            fro, to = np.where(utils.only_undirected(cpdag))
            # Test the operator to all undirected edges
            for (x, y) in zip(fro, to):
                output_a = ges.score_valid_delete_operators(x, y, cpdag, cache)
                output_b = ges.score_valid_delete_operators(y, x, cpdag, cache)
                for (op_a, op_b) in zip(output_a, output_b):
                    # Check resulting state is the same
                    self.assertTrue((op_a[1] == op_b[1]).all())
                    self.assertAlmostEqual(op_a[0], op_b[0])
        print("\nChecked equality of delete operator on undirected edges in %i CPDAGS" % (i + 1))

    def test_valid_delete_operators_4(self):
        A = np.array([[0, 1, 1, 0],
                      [0, 0, 1, 0],
                      [0, 1, 0, 1],
                      [0, 0, 1, 0]])
        cache = GaussObsL0Pen(self.obs_data)
        # Removing the edge X0 - X2 should yield two valid operators
        # operators, for:
        #   1. H = Ø, as NA_yx \ Ø = {X1} is a clique
        #   2. H = {1}, as NA_yx \ {X1} = Ø is a clique
        output = ges.score_valid_delete_operators(0, 2, A, cache)
        self.assertEqual(2, len(output))
        A1, A2 = A.copy(), A.copy()
        # Remove X2 - X4
        A1[0, 2], A2[0, 2] = 0, 0
        # Orient X2 -> X1
        A2[1, 2] = 0
        self.assertTrue(utils.member([op[1] for op in output], A1) is not None)
        self.assertTrue(utils.member([op[1] for op in output], A2) is not None)

    def test_valid_delete_operators_5(self):
        A = np.array([[0, 1, 1, 1],
                      [0, 0, 1, 1],
                      [1, 1, 0, 0],
                      [1, 1, 0, 0]])
        print("out:", utils.is_clique({2, 3}, A))
        cache = GaussObsL0Pen(self.obs_data)
        # Removing the edge X0 - X1 should yield three valid operators
        # operators, for:
        #   0. Invalid H = Ø, as NA_yx \ Ø = {X2,X3} is not a clique
        #   1. H = {X2}, as NA_yx \ H = {X3} is a clique
        #   2. H = {X3}, as NA_yx \ H = {X2} is a clique
        #   3. H = {X2,X3}, as NA_yx \ H = Ø is a clique
        output = ges.score_valid_delete_operators(0, 1, A, cache)
        print(output)
        self.assertEqual(3, len(output))
        # v-structure on X2, i.e orient X0 -> X2, X1 -> X2
        A1 = np.array([[0, 0, 1, 1],
                       [0, 0, 1, 1],
                       [0, 0, 0, 0],
                       [1, 1, 0, 0]])
        # v-structure on X3, i.e. orient X0 -> X3, X1 -> X3
        A2 = np.array([[0, 0, 1, 1],
                       [0, 0, 1, 1],
                       [1, 1, 0, 0],
                       [0, 0, 0, 0]])
        # v-structures on X2 and X3
        A3 = np.array([[0, 0, 1, 1],
                       [0, 0, 1, 1],
                       [0, 0, 0, 0],
                       [0, 0, 0, 0]])
        self.assertTrue(utils.member([op[1] for op in output], A1) is not None)
        self.assertTrue(utils.member([op[1] for op in output], A2) is not None)
        self.assertTrue(utils.member([op[1] for op in output], A3) is not None)


# --------------------------------------------------------------------
# Tests for the turn(x,y,C) operator
class TurnOperatorTests(unittest.TestCase):
    true_A = np.array([[0, 0, 1, 0, 0],
                       [0, 0, 1, 0, 0],
                       [0, 0, 0, 1, 1],
                       [0, 0, 0, 0, 1],
                       [0, 0, 0, 0, 0]])
    factorization = [(4, (2, 3)), (3, (2,)), (2, (0, 1)), (0, ()), (1, ())]
    true_B = true_A * np.random.uniform(1, 2, size=true_A.shape)
    scm = sempler.LGANM(true_B, (0, 0), (0.3, 0.4))
    p = len(true_A)
    n = 10000
    obs_data = scm.sample(n=n)
    cache = GaussObsL0Pen(obs_data)
    # ------------------------------------------------------
    # Tests

    def test_turn_operator_1(self):
        A = np.array([[0, 1, 0, 0, 0],
                      [0, 0, 0, 0, 0],
                      [0, 1, 0, 1, 0],
                      [0, 1, 1, 0, 0],
                      [0, 0, 0, 0, 0]])
        output = ges.turn(1, 2, {3}, A)
        # Orient X1 -> X2 and X3 -> X2
        A[2, 1], A[1, 2] = 0, 1
        A[2, 3] = 0
        self.assertTrue((A == output).all())

    def test_turn_operator_2(self):
        A = np.array([[0, 1, 1, 0, 0],
                      [1, 0, 1, 1, 1],
                      [0, 0, 0, 0, 0],
                      [0, 1, 0, 0, 0],
                      [0, 1, 0, 0, 0]])
        # Turn edge X3 - X1 to X3 -> X1 with C = {X4, X0}
        output = ges.turn(3, 1, {0, 4}, A)
        truth = A.copy()
        truth[1, 3] = 0
        truth[1, 0], truth[1, 4] = 0, 0
        self.assertTrue((truth == output).all())
        # Turn edge X1 - X3 to X1 -> X3 with C = Ø
        output = ges.turn(1, 3, set(), A)
        truth = A.copy()
        truth[3, 1] = 0
        self.assertTrue((truth == output).all())
        # Turn edge X4 -> X1 with C = {X3}
        output = ges.turn(4, 1, {3}, A)
        truth = A.copy()
        truth[1, 4] = 0
        truth[1, 3] = 0
        self.assertTrue((truth == output).all())
        # Turn edge X2 -> X0 with C = {X1}
        output = ges.turn(2, 0, {1}, A)
        truth = A.copy()
        truth[0, 2], truth[2, 0] = 0, 1
        truth[0, 1] = 0
        self.assertTrue((truth == output).all())

    def test_turn_operator_preconditions(self):
        A = np.array([[0, 1, 1, 0, 0],
                      [1, 0, 1, 1, 1],
                      [0, 0, 0, 0, 0],
                      [0, 1, 0, 0, 0],
                      [0, 1, 0, 0, 0]])
        # Trying to turn X1 -> X2 fails as edge already exists
        try:
            ges.turn(1, 2, set(), A)
            self.fail("Exception should have been thrown")
        except ValueError as e:
            print("OK:", e)
        # Trying to turn X3 -> X4 fails as they are not adjacent
        try:
            ges.turn(3, 4, set(), A)
            self.fail("Exception should have been thrown")
        except ValueError as e:
            print("OK:", e)
        # Trying to turn X3 <- X4 fails as they are not adjacent
        try:
            ges.turn(4, 3, set(), A)
            self.fail("Exception should have been thrown")
        except ValueError as e:
            print("OK:", e)
        # Turning X0 -> X1 with C = {X3,X2} fails as X2 is not a neighbor of X1
        try:
            ges.turn(0, 1, {3, 2}, A)
            self.fail("Exception should have been thrown")
        except ValueError as e:
            print("OK:", e)
        # Turning X3 -> X1 with C = {X4,X0,X3} should fail as X3 is contained in C
        try:
            ges.turn(3, 1, {0, 3, 4}, A)
            self.fail("Exception should have been thrown")
        except ValueError as e:
            print("OK:", e)

    def test_valid_turn_operators_preconditions(self):
        # Test preconditions
        A = np.array([[0, 1, 1, 0, 0],
                      [1, 0, 1, 1, 1],
                      [0, 0, 0, 0, 0],
                      [0, 1, 0, 0, 0],
                      [0, 1, 0, 0, 0]])
        # Trying to turn X1 -> X2 fails as edge already exists
        try:
            ges.score_valid_turn_operators(1, 2, A, self.cache)
            self.fail("Exception should have been thrown")
        except ValueError as e:
            print("OK:", e)
        # Trying to turn X3 -> X4 fails as they are not adjacent
        try:
            ges.score_valid_turn_operators(3, 4, A, self.cache)
            self.fail("Exception should have been thrown")
        except ValueError as e:
            print("OK:", e)
        # Trying to turn X3 <- X4 fails as they are not adjacent
        try:
            ges.score_valid_turn_operators(4, 3, A, self.cache)
            self.fail("Exception should have been thrown")
        except ValueError as e:
            print("OK:", e)

    def test_valid_turn_operators_1(self):
        A = np.array([[0, 1, 0, 0, 0],
                      [0, 0, 0, 0, 0],
                      [0, 1, 0, 1, 0],
                      [0, 1, 1, 0, 0],
                      [0, 0, 0, 0, 0]])
        # Turning the edge X1 <- X2 should yield one valid
        # operator, for:
        #   1. T = Ø, as NA_yx U Ø = {X3} is a clique
        output = ges.score_valid_turn_operators(1, 2, A, self.cache)
        self.assertEqual(1, len(output))
        true_A = A.copy()
        # Turn X1 <- X2 (and orient X3 -> X2)
        true_A[2, 1] = 0
        true_A[1, 2] = 1
        true_A[2, 3] = 0
        self.assertTrue((true_A == output[0][1]).all())

    def test_valid_turn_operators_2(self):
        # NOTE: Same graph as previous test
        A = np.array([[0, 1, 0, 0, 0],
                      [0, 0, 0, 0, 0],
                      [0, 1, 0, 1, 0],
                      [0, 1, 1, 0, 0],
                      [0, 0, 0, 0, 0]])
        # Turning the edge X1 <- X3 should yield one valid
        # operator, for:
        #   1. T = Ø, as NA_yx U Ø = {X2} is a clique
        output = ges.score_valid_turn_operators(1, 3, A, self.cache)
        self.assertEqual(1, len(output))
        true_A = A.copy()
        # Turn X1 <- X3 (and orient X2 -> X3)
        true_A[3, 1] = 0
        true_A[1, 3] = 1
        true_A[3, 2] = 0
        self.assertTrue((true_A == output[0][1]).all())

    def test_valid_turn_operators_3(self):
        # NOTE: Same graph as two previous tests (i.e. _3 and _2)
        A = np.array([[0, 1, 0, 0, 0],
                      [0, 0, 0, 0, 0],
                      [0, 1, 0, 1, 0],
                      [0, 1, 1, 0, 0],
                      [0, 0, 0, 0, 0]])
        # Turning the edge X0 -> X1 should yield one valid
        # operator, for:
        #   1. T = Ø, as NA_yx U Ø = Ø is a clique
        output = ges.score_valid_turn_operators(1, 0, A, self.cache)
        self.assertEqual(1, len(output))
        true_A = A.copy()
        # Turn X1 <- X0
        true_A[0, 1] = 0
        true_A[1, 0] = 1
        self.assertTrue((true_A == output[0][1]).all())

    def test_valid_turn_operators_4(self):
        A = np.array([[0, 1, 1, 1, 1],
                      [0, 0, 0, 0, 0],
                      [1, 1, 0, 0, 0],
                      [1, 1, 0, 0, 0],
                      [1, 0, 0, 0, 0]])
        # Turning the edge X0 -> X1 should yield no valid
        # operators, for (note T0 = {X4})
        #   1. T = Ø, as C = NA_yx U Ø = {X2,X3} is not a clique
        #   2. T = {X4}, as C = NA_yx U {X4} = {X2,X3,X4} is not a clique
        output = ges.score_valid_turn_operators(1, 0, A, self.cache)
        self.assertEqual(0, len(output))

    def test_valid_turn_operators_5(self):
        A = np.array([[0, 1, 1, 0, 0],
                      [0, 0, 0, 0, 0],
                      [1, 0, 0, 1, 0],
                      [0, 1, 1, 0, 0],
                      [0, 0, 0, 0, 0]])
        # Turning the edge X0 -> X1 should yield one valid
        # operator, for (note T0 = {X2})
        #   1. T = Ø, as C = NA_yx U Ø = Ø is a clique, but the path
        #   X0 - X2 - X3 -> X1 does not contain a node in C
        #   2. T = {X2}, as C = NA_yx U {X2} = {X2} is a clique and
        #   satisfies the path condition
        output = ges.score_valid_turn_operators(1, 0, A, self.cache)
        self.assertEqual(1, len(output))
        # Orient X1 -> X0 and X2 -> X0
        truth = A.copy()
        truth[0, 1], truth[1, 0] = 0, 1
        truth[0, 2] = 0
        self.assertTrue((truth == output[0][1]).all())

    def test_valid_turn_operators_6(self):
        A = np.array([[0, 1, 0, 0, 0],
                      [1, 0, 1, 1, 0],
                      [0, 1, 0, 1, 0],
                      [0, 1, 1, 0, 0],
                      [0, 0, 0, 0, 0]])
        # Orienting the edge X1 -> X3 yields not valid operators, as
        # all neighbors of X1 are adjacent to X3
        output = ges.score_valid_turn_operators(1, 3, A, self.cache)
        self.assertEqual(0, len(output))

    def test_valid_turn_operators_7(self):
        A = np.array([[0, 1, 0, 0, 0],
                      [1, 0, 1, 1, 0],
                      [0, 1, 0, 1, 0],
                      [0, 1, 1, 0, 0],
                      [0, 0, 0, 0, 0]])
        output = ges.score_valid_turn_operators(3, 1, A, self.cache)
        # Orienting the edge X3 -> X1 yields only one valid operator,
        # as for (note ne(X1) = {X2, X0}
        #   C = Ø and C = {X2} condition (i) is not satisfied
        #   C = {X0, X2} is not a clique
        #   C = {X0} satisfies all three conditions
        self.assertEqual(1, len(output))
        truth = np.array([[0, 1, 0, 0, 0],
                          [0, 0, 1, 0, 0],
                          [0, 1, 0, 1, 0],
                          [0, 1, 1, 0, 0],
                          [0, 0, 0, 0, 0]])
        self.assertTrue((truth == output[0][1]).all())

    def test_valid_turn_operators_8(self):
        A = np.array([[0, 1, 0, 0, 0],
                      [1, 0, 1, 1, 0],
                      [0, 1, 0, 1, 0],
                      [0, 1, 1, 0, 0],
                      [0, 0, 0, 0, 0]])
        # For the edge X0 -> X1 three operators are valid
        #   C = Ø : does not satisfy condition 1
        #   C = {X2}, {X3}, {X2,X3} are valid
        output = ges.score_valid_turn_operators(0, 1, A, self.cache)
        self.assertEqual(3, len(output))
        truth_2 = np.array([[0, 1, 0, 0, 0],
                            [0, 0, 0, 1, 0],
                            [0, 1, 0, 1, 0],
                            [0, 1, 1, 0, 0],
                            [0, 0, 0, 0, 0]])
        truth_3 = np.array([[0, 1, 0, 0, 0],
                            [0, 0, 1, 0, 0],
                            [0, 1, 0, 1, 0],
                            [0, 1, 1, 0, 0],
                            [0, 0, 0, 0, 0]])
        truth_23 = np.array([[0, 1, 0, 0, 0],
                             [0, 0, 0, 0, 0],
                             [0, 1, 0, 1, 0],
                             [0, 1, 1, 0, 0],
                             [0, 0, 0, 0, 0]])
        for (_, new_A, _, _, C) in output:
            if C == {2}:
                self.assertTrue((new_A == truth_2).all())
            if C == {3}:
                self.assertTrue((new_A == truth_3).all())
            if C == {2, 3}:
                self.assertTrue((new_A == truth_23).all())

    def test_valid_turn_operators_9(self):
        A = np.array([[0, 1, 1, 0, 0, 1],
                      [1, 0, 1, 1, 0, 1],
                      [0, 0, 0, 0, 0, 0],
                      [0, 1, 1, 0, 0, 0],
                      [0, 0, 1, 0, 0, 0],
                      [1, 1, 0, 0, 0, 0]])
        # Orienting the edge X0 -> X1 there should be only one valid
        # operator. NA_yx = {X5} and ne(y) / {x} = {X3, X5}:
        #   C = Ø does not satisfy condition i
        #   C = {X3} is valid
        #   C = {X5} does not satisfy condition i
        #   C = {X3,X5} do not form a clique
        output = ges.score_valid_turn_operators(0, 1, A, self.cache)
        self.assertEqual(1, len(output))
        truth = np.array([[0, 1, 1, 0, 0, 1],
                          [0, 0, 1, 0, 0, 1],
                          [0, 0, 0, 0, 0, 0],
                          [0, 1, 1, 0, 0, 0],
                          [0, 0, 1, 0, 0, 0],
                          [1, 1, 0, 0, 0, 0]])
        self.assertTrue((truth == output[0][1]).all())

    def test_valid_turn_operators_10(self):
        # Check that all valid turn operators result in a different
        # essential graph
        G = 10
        p = 20
        for i in range(G):
            A = sempler.generators.dag_avg_deg(p, 3, 1, 1)
            cpdag = utils.dag_to_cpdag(A)
            W = A * np.random.uniform(1, 2, A.shape)
            obs_sample = sempler.LGANM(W, (0, 0), (0.5, 1)).sample(n=1000)
            cache = GaussObsL0Pen(obs_sample)
            fro, to = np.where(cpdag != 0)
            for (x, y) in zip(to, fro):
                valid_operators = ges.score_valid_turn_operators(x, y, cpdag, cache)
                # print(i,len(valid_operators))
                for (_, new_A, _, _, _) in valid_operators:
                    new_cpdag = ges.utils.pdag_to_cpdag(new_A)
                    self.assertFalse((cpdag == new_cpdag).all())
        print("\nChecked that valid turn operators result in different MEC for %i CPDAGs" % (i + 1))
