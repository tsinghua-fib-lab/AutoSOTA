"""
Copyright 2020 Twitter, Inc.
SPDX-License-Identifier: Apache-2.0
"""
import unittest
import torch
from utils import get_symmetrically_normalized_adjacency


class TestUtils(unittest.TestCase):
    def __init__(self, methodName: str = ...) -> None:
        super().__init__(methodName)
        self.edge_index = torch.LongTensor(
            [[0, 2], [2, 0], [0, 3], [3, 0], [1, 2], [2, 1], [1, 3], [3, 1], [2, 3], [3, 2]]
        ).T
        self.edge_weight = torch.FloatTensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    def test_get_symmetrically_normalized_adjacency(self):
        expected_adj_col_sum = torch.FloatTensor([0.40824829046] * 8 + [1 / 3] * 2)
        edge_index, edge_weight = get_symmetrically_normalized_adjacency(edge_index=self.edge_index, n_nodes=4)
        self.assertTrue(torch.all(torch.eq(self.edge_index, edge_index)))
        self.assertTrue(torch.allclose(expected_adj_col_sum, edge_weight))


if __name__ == "__main__":
    unittest.main()
