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
Test the DecomposableScore class.
"""

import unittest
import numpy as np

from ges.scores.decomposable_score import DecomposableScore

# Class to to test cache behaviour


class TestScore(DecomposableScore):
    def _compute_local_score(self, x, pa):
        return np.random.uniform()

# Tests for the decomposable score class


class DecomposableScoreTests(unittest.TestCase):

    def test_initialization(self):
        p, n = 10, 1000
        data = np.random.uniform(size=(n, p))

        # Test 1
        score = DecomposableScore(data, False, False)
        self.assertIsNone(score._cache)
        self.assertEqual(score._debug, False)
        self.assertFalse(data is score._data)
        self.assertTrue((data == score._data).all())

        # Test 2
        score = DecomposableScore(data, False, True)
        self.assertIsNone(score._cache)
        self.assertEqual(score._debug, True)
        self.assertFalse(data is score._data)
        self.assertTrue((data == score._data).all())

        # Test 3
        score = DecomposableScore(data, True, False)
        self.assertIsNotNone(score._cache)
        self.assertEqual(score._debug, False)
        self.assertFalse(data is score._data)
        self.assertTrue((data == score._data).all())

        # Test 4
        score = DecomposableScore(data, True, True)
        self.assertIsNotNone(score._cache)
        self.assertEqual(score._debug, True)
        self.assertFalse(data is score._data)
        self.assertTrue((data == score._data).all())

        # Double-check memory leaks
        data[0, 0] = 99
        self.assertFalse((data == score._data).all())

    def test_api(self):
        score = DecomposableScore(None, True, True)
        score.local_score(0, {})
        score.p

    def test_cache_on(self):
        data = None
        score = TestScore(data, cache=True, debug=0)
        # Make some calls to local_score
        items = []
        for i in range(5):
            choice = np.random.choice(range(10), replace=False, size=i + 1)
            x = choice[0]
            pa = choice[1:]
            items.append((x, pa, score.local_score(x, pa)))
        # With the cache turned on, the call to local_score should
        # always return the same value (see
        # TestScore._compute_local_score)
        for item in items:
            x, pa, previous_score = item
            # print(previous_score, score.local_score(x,pa))
            self.assertEqual(previous_score, score.local_score(x, pa))
        self.assertEqual(5, len(score._cache))

    def test_cache_off(self):
        data = None
        score = TestScore(data, cache=False, debug=0)
        # Make some calls to local_score
        items = []
        for i in range(5):
            choice = np.random.choice(range(10), replace=False, size=i + 1)
            x = choice[0]
            pa = choice[1:]
            items.append((x, pa, score.local_score(x, pa)))
        self.assertIsNone(score._cache)
        # With the cache turned on, the call to local_score should
        # always return the a different value (see
        # TestScore._compute_local_score)
        for item in items:
            x, pa, previous_score = item
            # print(previous_score, score.local_score(x,pa))
            self.assertNotEqual(previous_score, score.local_score(x, pa))
