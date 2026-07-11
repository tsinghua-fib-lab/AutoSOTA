import unittest

import torch
from torch_geometric.data import Batch, Data

from scgfm.data.graph_features import precompute_graph_statistics
from scgfm.encoders import SCGFMEncoder
from scgfm.models.geometric_bases import GeometricBasesModel


def toy_graph(num_nodes=5, feat_dim=3):
    edge_index = torch.tensor(
        [[0, 1, 2, 3, 0, 2], [1, 2, 3, 4, 2, 0]],
        dtype=torch.long,
    )
    x = torch.randn(num_nodes, feat_dim)
    return Data(x=x, edge_index=edge_index, y=torch.tensor([0]), num_nodes=num_nodes)


class ModelTest(unittest.TestCase):
    def test_bases_are_adjacency_like(self):
        model = GeometricBasesModel(K=3, M=5, feature_dim=10)
        bases = model.get_normalized_bases()
        self.assertEqual(bases.shape, (3, 5, 5))
        self.assertTrue(torch.allclose(bases, bases.transpose(1, 2), atol=1e-6))
        self.assertTrue(torch.allclose(torch.diagonal(bases, dim1=1, dim2=2), torch.zeros(3, 5), atol=1e-6))
        self.assertTrue(torch.all((bases >= 0) & (bases <= 1)))

    def test_forward_has_finite_loss(self):
        graphs = precompute_graph_statistics([toy_graph(), toy_graph()], feature_dim=10, show_progress=False)
        batch = Batch.from_data_list(graphs)
        model = GeometricBasesModel(K=3, M=5, feature_dim=10, num_projections=4)
        loss, logs = model(batch)
        self.assertTrue(torch.isfinite(loss))
        self.assertIn("loss_gw", logs)

    def test_encoder_output_is_stable(self):
        model = GeometricBasesModel(K=3, M=5, feature_dim=10, num_projections=4)
        encoder = SCGFMEncoder(model, device="cpu", max_dim=10, num_projections=4, top_k=2)
        z = encoder.encode_single(toy_graph())
        self.assertEqual(z.ndim, 1)
        self.assertTrue(torch.isfinite(z).all())


if __name__ == "__main__":
    unittest.main()
