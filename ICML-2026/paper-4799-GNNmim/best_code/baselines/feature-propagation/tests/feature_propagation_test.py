"""
Copyright 2020 Twitter, Inc.
SPDX-License-Identifier: Apache-2.0
"""
import unittest
import math

import torch

from feature_propagation import FeaturePropagation


class TestFeaturePropagation(unittest.TestCase):
    def test_feature_propagation(self):
        X = torch.Tensor([1 / 2, 0, 1 / 3, 0]).reshape(-1, 1)
        node_mask = torch.BoolTensor([True, False, True, False])
        edge_index = torch.LongTensor(
            [[0, 2], [2, 0], [0, 3], [3, 0], [1, 2], [2, 1], [1, 3], [3, 1], [2, 3], [3, 2]]
        ).T

        expected_X_propagated = torch.Tensor(
            [[1 / 2], [(1 / math.sqrt(6)) * (1 / 3)], [1 / 3], [(1 / 3) * (1 / 3) + (1 / math.sqrt(6)) * (1 / 2)]]
        )

        fp = FeaturePropagation(num_iterations=1)
        X_propagated = fp.propagate(x=X, edge_index=edge_index, mask=node_mask)
        self.assertTrue(torch.allclose(expected_X_propagated, X_propagated))


if __name__ == "__main__":
    unittest.main()
