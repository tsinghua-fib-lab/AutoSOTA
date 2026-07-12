"""
Copyright 2020 Twitter, Inc.
SPDX-License-Identifier: Apache-2.0
"""
import unittest

import torch

from filling_strategies import zero_filling, mean_filling, neighborhood_mean_filling, compute_mean


class TestFillingStrategies(unittest.TestCase):
    def test_compute_mean(self):
        X = torch.Tensor([[1, 5, 9], [2, 6, 10], [3, 7, 11], [4, 8, 12],])
        feature_mask = torch.BoolTensor(
            [[False, True, True], [False, False, True], [False, True, True], [False, True, True]]
        )
        expected_mean = torch.Tensor([0, 20 / 3, 42 / 4])
        mean = compute_mean(X=X, feature_mask=feature_mask)

        self.assertTrue(torch.equal(expected_mean, mean))

    def test_zero_filling(self):
        n_nodes = 5
        X = torch.ones((n_nodes, 2))
        expected_X_filled = torch.Tensor([[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]])

        X_filled = zero_filling(X=X)
        self.assertTrue(torch.equal(expected_X_filled, X_filled))

    def test_mean_filling(self):
        X = torch.Tensor([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
        feature_mask = torch.BoolTensor([[True, True], [False, True], [False, False], [True, True], [False, False]])
        mean = [4, 14 / 3]  # Mean of nodes with features only
        expected_X_filled = torch.Tensor([mean, mean, mean, mean, mean])

        X_filled = mean_filling(X=X, feature_mask=feature_mask)
        self.assertTrue(torch.equal(expected_X_filled, X_filled))

    def test_neighborhood_mean_filling(self):
        X = torch.Tensor([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
        edge_index = torch.LongTensor([[2, 1], [0, 2], [3, 2], [0, 4], [1, 4]]).T
        feature_mask = torch.BoolTensor([[True, False], [False, False], [True, False], [True, True], [False, False]])
        expected_X_filled = torch.Tensor([[0, 0], [5, 0], [4, 8], [0, 0], [1, 0]])

        X_filled = neighborhood_mean_filling(edge_index=edge_index, X=X, feature_mask=feature_mask)
        self.assertTrue(torch.equal(expected_X_filled, X_filled))


if __name__ == "__main__":
    unittest.main()
