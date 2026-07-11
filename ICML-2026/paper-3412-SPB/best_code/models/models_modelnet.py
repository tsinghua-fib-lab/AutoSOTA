import torch
import torch.nn as nn
import torch.nn.functional as F

from e3nn import o3
from e3nn.o3 import Irreps
from e3nn.nn import FullyConnectedNet
from e3nn.math import soft_one_hot_linspace

from torch_cluster import radius_graph

# Baseline Model
class PointNetBaseline(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.mlp1 = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
        )

        self.mlp2 = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        # x: (B, N, 3)

        x = self.mlp1(x)           # (B, N, 256)
        x = x.max(dim=1).values

        x = self.mlp2(x)
        return x


class VNLinear(nn.Module):
    """
    Vector Neuron linear layer.

    Input:
        (B, C_in, N, 3)

    Output:
        (B, C_out, N, 3)
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels)
        )

    def forward(self, x):
        # x: (B, C_in, N, 3)

        x = torch.einsum('oi,binc->bonc', self.weight, x)

        return x


class VNLeakyReLU(nn.Module):
    """
    Equivariant vector nonlinearity.
    """

    def __init__(self, channels):
        super().__init__()

        self.direction = nn.Parameter(
            torch.randn(1, channels, 1, 3)
        )

    def forward(self, x):
        # projection onto learned directions
        dot = (x * self.direction).sum(-1, keepdim=True)

        mask = (dot >= 0).float()

        x_out = mask * x + (1 - mask) * (
            x - dot * self.direction
        )

        return x_out

class VNBatchNorm(nn.Module):
    """
    Equivariant batch normalization for vector neurons.

    Normalizes vector norms while preserving directions.
    """

    def __init__(self, num_features):
        super().__init__()

        self.bn = nn.BatchNorm1d(num_features)

    def forward(self, x):
        # x: (B, C, N, 3)

        norm = torch.norm(x, dim=-1) + 1e-6  # (B, C, N)

        norm_bn = self.bn(norm)

        x = x / norm.unsqueeze(-1) * norm_bn.unsqueeze(-1)

        return x


class VNLinearLeakyReLU(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.linear = VNLinear(in_channels, out_channels)
        self.relu = VNLeakyReLU(out_channels)

    def forward(self, x):
        return self.relu(self.linear(x))


class VNLinearBNLeakyReLU(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.linear = VNLinear(in_channels, out_channels)
        self.bn = VNBatchNorm(out_channels)
        self.relu = VNLeakyReLU(out_channels)

    def forward(self, x):
        x = self.linear(x)
        x = self.bn(x)
        x = self.relu(x)

        return x

class VNMaxPool(nn.Module):
    """
    Pool over points dimension.
    """

    def forward(self, x):
        # x: (B, C, N, 3)

        norms = torch.norm(x, dim=-1)

        idx = norms.max(dim=2).indices

        idx = idx.unsqueeze(-1).unsqueeze(-1).expand(
            -1, -1, 1, 3
        )

        pooled = torch.gather(x, 2, idx).squeeze(2)

        return pooled


# SO(3) rotation invariant model
class SO3InvariantModel(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.conv1 = VNLinearLeakyReLU(1, 64)
        self.conv2 = VNLinearLeakyReLU(64, 128)
        self.conv3 = VNLinearLeakyReLU(128, 256)

        self.pool = VNMaxPool()

        self.fc1 = nn.Linear(256 * 3, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        # x: (B, N, 3)

        x = x.unsqueeze(1)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x = self.pool(x)

        x = x.reshape(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x


# Scale Invariant Model
class ScaleInvariantModel(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
        )

        self.head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        # x: (B, N, 3)

        scale = x.norm(dim=-1).max(dim=1, keepdim=True).values.unsqueeze(-1)
        x = x / (scale + 1e-8)   # remove scaling

        x = self.mlp(x)
        x = x.max(dim=1).values

        return self.head(x)




######## NEW EXPERIMENT #######
class BaselinePointNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),

            nn.Linear(64, 128),
            nn.ReLU(),

            nn.Linear(128, 256),
            nn.ReLU(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),

            nn.Linear(128, num_classes)
        )

    def forward(self, pos):
        # pos: [B, N, 3]

        x = self.mlp(pos)

        # permutation invariant pooling
        x = x.mean(dim=1)

        return self.classifier(x)



class EquivariantModelNet(nn.Module):
    """
    Small SO(3)-equivariant network for ModelNet10.

    Input:
        positions: [B, N, 3]

    Output:
        logits: [B, num_classes]

    Features:
    - rotational equivariance via e3nn irreps
    - invariant global pooling
    - compact architecture suitable for PAC-Bayes experiments
    """

    def __init__(
        self,
        num_classes=10,
        num_neighbors=16,
        max_radius=0.25,
    ):
        super().__init__()

        self.num_neighbors = num_neighbors
        self.max_radius = max_radius

        # ------------------------------------------------------------------
        # Irreducible representations
        # ------------------------------------------------------------------

        # scalar input feature per point
        self.input_irreps = Irreps("1x0e")

        # hidden equivariant features
        self.hidden_irreps = Irreps(
            "32x0e + 32x1o + 16x2e"
        )

        # output scalar features
        self.output_irreps = Irreps("64x0e")

        # ------------------------------------------------------------------
        # Spherical harmonics
        # ------------------------------------------------------------------

        self.sh_irreps = o3.Irreps.spherical_harmonics(lmax=2)

        # ------------------------------------------------------------------
        # Tensor product layer
        # ------------------------------------------------------------------

        self.tp = o3.FullyConnectedTensorProduct(
            self.input_irreps,
            self.sh_irreps,
            self.hidden_irreps,
            shared_weights=False,
            internal_weights=False
        )

        # ------------------------------------------------------------------
        # Radial embedding network
        # ------------------------------------------------------------------

        self.radial_net = FullyConnectedNet(
            [10, 32, 32, self.tp.weight_numel],
            act=torch.relu
        )

        # ------------------------------------------------------------------
        # Equivariant linear layer
        # ------------------------------------------------------------------

        self.linear = o3.Linear(
            self.hidden_irreps,
            self.output_irreps
        )

        # ------------------------------------------------------------------
        # Invariant classifier
        # ------------------------------------------------------------------

        self.classifier = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, pos):
        """
        pos: [B, N, 3]
        """

        B, N, _ = pos.shape

        # --------------------------------------------------------------
        # flatten batch
        # --------------------------------------------------------------

        pos_flat = pos.reshape(B * N, 3)

        batch = torch.arange(B, device=pos.device)
        batch = batch.repeat_interleave(N)

        # --------------------------------------------------------------
        # scalar input features
        # --------------------------------------------------------------

        x = torch.ones(
            (B * N, 1),
            device=pos.device
        )

        # --------------------------------------------------------------
        # graph construction
        # --------------------------------------------------------------

        edge_src, edge_dst = radius_graph(
            pos_flat,
            r=self.max_radius,
            batch=batch,
            max_num_neighbors=self.num_neighbors
        )

        # relative vectors
        edge_vec = pos_flat[edge_dst] - pos_flat[edge_src]

        edge_length = edge_vec.norm(dim=1)

        # --------------------------------------------------------------
        # spherical harmonics
        # --------------------------------------------------------------

        sh = o3.spherical_harmonics(
            self.sh_irreps,
            edge_vec,
            normalize=True,
            normalization='component'
        )

        # --------------------------------------------------------------
        # radial embedding
        # --------------------------------------------------------------

        edge_length_embedded = soft_one_hot_linspace(
            edge_length,
            start=0.0,
            end=self.max_radius,
            number=10,
            basis='smooth_finite',
            cutoff=True
        )

        edge_length_embedded *= (10 ** 0.5)

        radial_weights = self.radial_net(edge_length_embedded)

        # --------------------------------------------------------------
        # equivariant tensor product
        # --------------------------------------------------------------

        messages = self.tp(
            x[edge_src],
            sh,
            radial_weights
        )

        # aggregate messages
        out = torch.zeros(
            (B * N, self.hidden_irreps.dim),
            device=pos.device
        )

        out.index_add_(0, edge_dst, messages)

        # --------------------------------------------------------------
        # equivariant linear projection
        # --------------------------------------------------------------

        out = self.linear(out)

        # --------------------------------------------------------------
        # reshape back to batch
        # --------------------------------------------------------------

        out = out.reshape(B, N, -1)

        # --------------------------------------------------------------
        # invariant global pooling
        # --------------------------------------------------------------

        out = out.mean(dim=1)

        # scalar invariant features only
        logits = self.classifier(out)

        return logits