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
Test the PDAG to CPDAG conversion.
"""

import unittest
import numpy as np
import sempler
import sempler.generators
import ges.utils as utils

import ges.utils

# ---------------------------------------------------------------------


class PDAG_to_CPDAG_Tests(unittest.TestCase):
    # Tests to ensure that the conversion from PDAG to CPDAG
    # works

    def test_pdag_to_dag_1(self):
        # Should work
        P = np.array([[0, 0, 1, 0],
                      [0, 0, 1, 1],
                      [1, 0, 0, 0],
                      [0, 0, 0, 0]])
        A = ges.utils.pdag_to_dag(P, debug=False)
        # print(A)
        true_A = P.copy()
        true_A[0, 2] = 0
        self.assertTrue((A == true_A).all())

    def test_pdag_to_dag_2(self):
        # Same as above but different index order, should work
        P = np.array([[0, 0, 1, 0],
                      [1, 0, 0, 1],
                      [1, 0, 0, 0],
                      [0, 0, 0, 0]])
        A = ges.utils.pdag_to_dag(P, debug=False)
        # print(A)
        true_A = P.copy()
        true_A[2, 0] = 0
        self.assertTrue((A == true_A).all())

    def test_pdag_to_dag_3(self):
        # Should work
        P = np.array([[0, 0, 1, 0],
                      [0, 0, 0, 1],
                      [1, 1, 0, 0],
                      [0, 0, 0, 0]])
        A = ges.utils.pdag_to_dag(P, debug=False)
        # print(A)
        true_A1, true_A2 = P.copy(), P.copy()
        true_A1[0, 2], true_A2[2, 0] = 0, 0
        self.assertTrue(utils.member([true_A1, true_A2], A) is not None)

    def test_pdag_to_dag_4(self):
        # This PDAG does not admit a consistent extension, i.e. it
        # either creates a non-existing v-structure or it induces a
        # cycle
        P = np.array([[0, 0, 1, 1],
                      [0, 0, 1, 0],
                      [1, 0, 0, 0],
                      [0, 0, 1, 0]])
        try:
            ges.utils.pdag_to_dag(P, debug=False)
            self.fail("Exception should have been thrown")
        except ValueError as e:
            print("OK:", e)

    def test_pdag_to_dag_5(self):
        # Fully directed PDAG should return itself
        A = np.array([[0, 1, 1, 0],
                      [0, 0, 1, 0],
                      [0, 0, 0, 0],
                      [0, 0, 1, 0]])
        extension = ges.utils.pdag_to_dag(A, debug=False)
        self.assertTrue(utils.is_consistent_extension(extension, A))
        self.assertTrue((extension == A).all())

    def test_pdag_to_dag_6(self):
        # Check that the resulting extensions are indeed a consistent
        # extensions
        G = 500
        p = 20
        for i in range(G):
            A = sempler.generators.dag_avg_deg(p, 3, 1, 1)
            cpdag = utils.dag_to_cpdag(A)
            self.assertTrue(utils.is_consistent_extension(A, cpdag))
            extension = ges.utils.pdag_to_dag(cpdag, debug=False)
            is_consistent_extension = utils.is_consistent_extension(extension, cpdag)
            if not is_consistent_extension:
                print("DAG\n", A)
                print("CPDAG\n", cpdag)
                print("Extension\n", extension)
                utils.is_consistent_extension(extension, cpdag, debug=True)
                # Rerun with outputs
                assert (extension == ges.utils.pdag_to_dag(cpdag, debug=True)).all()
                self.assertTrue(is_consistent_extension)
        print("\nChecked PDAG to DAG conversion for %d PDAGs" % (i + 1))

    def test_order_edges_preconditions(self):
        A = np.array([[0, 0, 1, 1, 1], [0, 0, 1, 1, 1], [
                     0, 0, 0, 1, 1], [0, 0, 0, 0, 1], [0, 0, 0, 0, 0]])
        # Check that exception is thrown with pdag
        pdag = A.copy()
        pdag[4, 2] = 1
        try:
            ges.utils.order_edges(pdag)
            self.fail()
        except ValueError as e:
            print("OK:", e)
        # Check that exception is thrown with cyclic graph
        cyclic = pdag.copy()
        cyclic[2, 4] = 0
        try:
            ges.utils.order_edges(cyclic)
            self.fail()
        except ValueError as e:
            print("OK:", e)

    def test_order_edges_1(self):
        A = np.array([[0, 0, 1, 1, 1], [0, 0, 1, 1, 1], [
                     0, 0, 0, 1, 1], [0, 0, 0, 0, 1], [0, 0, 0, 0, 0]])
        ordered = ges.utils.order_edges(A)
        # print(ordered)
        # Ground truth derived by hand for the order [1,0,2,3,4]
        truth = np.array([[0, 0, 9, 6, 2],
                          [0, 0, 8, 5, 1],
                          [0, 0, 0, 7, 3],
                          [0, 0, 0, 0, 4],
                          [0, 0, 0, 0, 0]])
        self.assertTrue((ordered >= 0).all())
        self.assertTrue((ordered == truth).all())

    def test_order_edges_2(self):
        G = 500
        p = 20
        for i in range(G):
            A = sempler.generators.dag_avg_deg(p, 3, 1, 1)
            ordered = ges.utils.order_edges(A)
            no_edges = (A != 0).sum()
            self.assertEqual(sorted(ordered[ordered != 0]), list(range(1, no_edges + 1)))
        print("\nChecked valid ordering for %d DAGs" % (i + 1))

    def test_label_edges_preconditions(self):
        A = np.array([[0, 0, 1, 1, 1], [0, 0, 1, 1, 1], [
                     0, 0, 0, 1, 1], [0, 0, 0, 0, 1], [0, 0, 0, 0, 0]])
        # Check that exception is thrown with pdag
        pdag = A.copy()
        pdag[4, 2] = 1
        try:
            ges.utils.order_edges(pdag)
            self.fail()
        except ValueError as e:
            print("OK:", e)
        # Check that exception is thrown with cyclic graph
        cyclic = pdag.copy()
        cyclic[2, 4] = 0
        try:
            ges.utils.order_edges(cyclic)
            self.fail()
        except ValueError as e:
            print("OK:", e)
        # Check that if ordering is invalid an exception is thrown
        try:
            ges.utils.label_edges(A)
            self.fail()
        except ValueError as e:
            print("OK:", e)
        # Same same, but different :)
        ordered = ges.utils.order_edges(A)
        ordered[0, 4] = 1
        try:
            ges.utils.label_edges(ordered)
            self.fail()
        except ValueError as e:
            print("OK:", e)

    def test_label_edges_1(self):
        # For a hand-picked example
        A = np.array([[0, 0, 1, 0, 0],
                      [0, 0, 1, 0, 0],
                      [0, 0, 0, 1, 1],
                      [0, 0, 0, 0, 1],
                      [0, 0, 0, 0, 0]])
        ordered = ges.utils.order_edges(A)
        truth = np.array([[0, 0, 1, 0, 0],
                          [0, 0, 1, 0, 0],
                          [0, 0, 0, 1, 1],
                          [0, 0, 0, 0, -1],
                          [0, 0, 0, 0, 0]])
        labelled = ges.utils.label_edges(ordered)
        self.assertTrue((labelled == truth).all())

    def test_label_edges_2(self):
        # With randomly generated DAGs
        # np.random.seed(42)
        G = 500
        p = 20
        for i in range(G):
            A = sempler.generators.dag_avg_deg(p, 3, 1, 1)
            # Construct expected output
            cpdag = utils.dag_to_cpdag(A)
            undirected = np.logical_and(cpdag, cpdag.T)
            truth = A.copy()
            truth[np.logical_and(truth, undirected)] = -1
            # Run and assert
            ordered = ges.utils.order_edges(A)
            labelled = ges.utils.label_edges(ordered)
            self.assertTrue((labelled == truth).all())
        print("\nChecked edge labelling for %d DAGs" % (i + 1))

    def test_dag_to_cpdag(self):
        # Test by checking that applying the whole pipeline to a CPDAG
        # returns the same CPDAG
        G = 500
        p = 25
        for i in range(G):
            A = sempler.generators.dag_avg_deg(p, 4, 1, 1)
            truth = utils.dag_to_cpdag(A)
            # Run and assert
            cpdag = ges.utils.dag_to_cpdag(A)
            self.assertTrue((truth == cpdag).all())
        print("\nChecked DAG to CPDAG conversion for %d DAGs" % (i + 1))

    def test_cpdag_to_cpdag(self):
        # Test by checking that applying the whole pipeline to a CPDAG
        # returns the same CPDAG
        G = 500
        p = 30
        for i in range(G):
            A = sempler.generators.dag_avg_deg(p, 3, 1, 1)
            cpdag = utils.dag_to_cpdag(A)
            # Run and assert
            output = ges.utils.pdag_to_cpdag(cpdag)
            self.assertTrue((output == cpdag).all())
        print("\nChecked CPDAG to CPDAG conversion for %d CPDAGs" % (i + 1))

    def test_pdag_to_cpdag(self):
        # Now construct PDAGs whose extensions belong to the true MEC,
        # and test that the true CPDAG is recovered
        G = 500
        p = 32
        for g in range(G):
            A = sempler.generators.dag_avg_deg(p, 3, 1, 1)
            # Construct PDAG by undirecting random edges which do not
            # belong to a v-structure.
            # NOTE: I'm proceeding in this awkward way to avoid
            # using functions from the pipeline I'm testing,
            # i.e. ges.utils.order_edges and ges.utils.label_edges
            pdag = A.copy()
            mask_vstructs = np.zeros_like(A)
            for (i, c, j) in utils.vstructures(A):
                mask_vstructs[i, c] = 1
                mask_vstructs[j, c] = 1
            flippable = np.logical_and(A, np.logical_not(mask_vstructs))
            fros, tos = np.where(flippable)
            for (x, y) in zip(fros, tos):
                # Undirect approximately 2/3 of the possible edges
                if np.random.binomial(1, 2 / 3):
                    pdag[y, x] = 1
            # Run and assert
            self.assertTrue(utils.is_consistent_extension(A, pdag))
            truth = utils.dag_to_cpdag(A)
            output = ges.utils.pdag_to_cpdag(pdag)
            self.assertTrue((output == truth).all())
        print("\nChecked PDAG to CPDAG conversion for %d PDAGs" % (g + 1))
