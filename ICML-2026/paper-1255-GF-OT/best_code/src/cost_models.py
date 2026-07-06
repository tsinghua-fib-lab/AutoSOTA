import torch
import numpy as np


class MLPCostModel(torch.nn.Module):
    def __init__(
        self,
        d_in: int = None,
        d_hidden: int = None,
        d_out: int = None,
        n_layers: int = 3,
    ):
        """A MLP architecture for cost modeling."""
        super(MLPCostModel, self).__init__()

        self.fc1_x = torch.nn.Linear(d_in, d_hidden, bias=False)
        self.hidden_layers_x = torch.nn.ModuleList(
            [
                torch.nn.Linear(d_hidden, d_hidden, bias=False)
                for _ in range(n_layers)
            ]
        )
        self.fc3_x = torch.nn.Linear(d_hidden, d_out, bias=False)

        self.fc1_y = torch.nn.Linear(d_in, d_hidden, bias=False)
        self.hidden_layers_y = torch.nn.ModuleList(
            [
                torch.nn.Linear(d_hidden, d_hidden, bias=False)
                for _ in range(n_layers)
            ]
        )
        self.fc3_y = torch.nn.Linear(d_hidden, d_out, bias=False)

        self.activation = torch.nn.ReLU()

    def forward(
        self, x: torch.Tensor, y: torch.Tensor, return_embeddings: bool = False
    ) -> torch.Tensor:
        x = torch.relu(self.fc1_x(x))
        for layer in self.hidden_layers_x:
            x = torch.relu(layer(x))
        x = self.fc3_x(x)

        y = torch.relu(self.fc1_y(y))
        for layer in self.hidden_layers_y:
            y = torch.relu(layer(y))
        y = self.fc3_y(y)

        if return_embeddings:
            return x, y

        diff = (x[:, None, :] - y[None, :, :]) ** 2
        result = diff.sum(dim=2)
        return result.reshape(diff.shape[:2])

    def project_data(self, x):
        x = self.activation(self.fc1_x(x))
        for layer in self.hidden_layers_x:
            x = self.activation(layer(x))
        x = self.fc3_x(x)
        return x


class ResNetCostModel(torch.nn.Module):
    def __init__(
        self,
        d_in: int = None,
        d_hidden: int = None,
        d_out: int = None,
        n_layers: int = 3,
        **kwargs
    ):
        """A ResNet architecture for cost modeling."""
        super(ResNetCostModel, self).__init__()

        self.fc1 = torch.nn.Linear(d_in, d_hidden)
        self.hidden_layers = torch.nn.ModuleList(
            [
                torch.nn.Linear(d_hidden, d_hidden, bias=False)
                for _ in range(n_layers)
            ]
        )
        self.fc3 = torch.nn.Linear(d_hidden, d_out, bias=False)

        self.activation = torch.nn.ReLU()
        self.normalization_factor = torch.sqrt(
            torch.tensor(d_hidden, dtype=torch.float32) * n_layers
        )

    def forward(self, x, y):
        x = self.activation(self.fc1(x))
        for layer in self.hidden_layers:
            x = x + 1 / self.normalization_factor * self.activation(layer(x))
        x = self.fc3(x)

        y = self.activation(self.fc1(y))
        for layer in self.hidden_layers:
            y = y + 1 / self.normalization_factor * self.activation(layer(y))
        y = self.fc3(y)

        diff = (x[:, None, :] - y[None, :, :]) ** 2
        result = diff.sum(dim=2)
        return result.reshape(diff.shape[:2])

    def project_data(self, x):
        x = self.activation(self.fc1(x))
        for layer in self.hidden_layers:
            x = x + 1 / self.normalization_factor * self.activation(layer(x))
        x = self.fc3(x)
        return x


class MahalanobisCostModel(torch.nn.Module):
    """A PyTorch module that implements a Mahalanobis cost model.

    Parameters:
    ----------
    d : int
        The dimensionality of the input data.
    M_init : np.ndarray, optional
        Initial value for the linear transformation matrix M. If None, M will
        be initialized randomly.
    Attributes:
    ----------
    M : torch.nn.Linear
        A linear transformation matrix that maps the difference between the
        encodings of X and Y to a cost matrix.

    Methods:
    -------
    forward(x, y) -> torch.Tensor:
        Computes the cost matrix based on the input data x and y. If encoders
        are present, it first encodes the input data using the encoders, then
        computes the cost matrix as M(x - y) @ (M(x - y).T). The output is a
        square matrix representing the cost between all pairs of points in x
        and y.
    """

    def __init__(self, d_in: int, M_init: str = "identity", **kwargs):
        """
        Initializes the CostModel with the specified parameters.
        """
        super(MahalanobisCostModel, self).__init__()

        # Initialize the linear transformation M
        self.M = torch.nn.Linear(d_in, d_in, bias=False)
        if M_init == "identity":
            self.M.weight.data = torch.eye(d_in, dtype=torch.float32)
        elif M_init is not None:
            self.M.weight.data = M_init

        self.M.weight.data = self.M.weight.data.T @ self.M.weight.data

    def forward(
        self, x: torch.Tensor, y: torch.Tensor, return_embeddings=False
    ) -> torch.Tensor:
        if return_embeddings:
            return self.M(x), self.M(y)
        else:
            diff = x[:, None, :] - y[None, :, :]
            output = self.M(diff.reshape(-1, diff.shape[-1]))
            result = (output * output).sum(dim=1)
            return result.reshape(diff.shape[:2])

    def project(self) -> None:
        """Projects the matrix M to ensure it is symmetric positive
        semi-definite."""
        with torch.no_grad():
            self.M.weight.data = 0.5 * (
                self.M.weight.data + self.M.weight.data.T
            )
            _matrix_for_pen = self.M.weight.data
            D, Q = torch.linalg.eigh(_matrix_for_pen)
            S = torch.clamp(D, min=0)
            self.M.weight.data = (Q * S) @ Q.T

    def project_data(self, x):
        x = self.M(x)
        return x


class WeightedDistanceFunction(torch.nn.Module):
    def __init__(
        self,
        n_features=9,
        table_distance=None,
        pre_computed_matrix=None,
        match_matrix=None,
    ):
        super(WeightedDistanceFunction, self).__init__()
        self.cost_function = table_distance
        weights = torch.ones((1, 1, n_features))
        self.weights = torch.nn.Parameter(weights)
        self.pre_computed_matrix = pre_computed_matrix
        self.match_matrix = match_matrix
        if torch.cuda.is_available:
            self.pre_computed_matrix = self.pre_computed_matrix.cuda()

    def forward(self, x, y, id_x, id_y):
        if self.pre_computed_matrix is not None:
            rank_x = np.where(id_x == id_x)
            rank_y = np.where(id_y == id_y)
            distances = self.pre_computed_matrix[rank_x, :][:, rank_y, :]
        else:
            distances = self.cost_function(
                x, y, id_x, id_y, self.match_matrix, return_sum=False
            )
            distances = torch.tensor(distances, dtype=torch.float32)
        distances = distances.reshape(len(x), len(y), -1)
        return torch.sum(self.weights * distances, dim=-1)

    def project(self) -> None:
        with torch.no_grad():
            self.weights.data = torch.clamp(self.weights.data, min=0)
