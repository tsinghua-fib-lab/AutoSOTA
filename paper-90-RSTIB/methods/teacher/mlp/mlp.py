import torch
from torch import nn

class MultiLayerPerceptron(nn.Module):
    """Two fully connected layer."""

    def __init__(self, history_seq_len: int, prediction_seq_len: int, hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(history_seq_len, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, prediction_seq_len)
        self.act = nn.ReLU()

    def forward(self, history_data: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            history_data (torch.Tensor): inputs with shape [B, L, N, C].

        Returns:
            torch.Tensor: outputs with shape [B, L, N, C]
        """

        history_data = history_data[..., :].permute(0, 2, 3, 1)     # B, N, C, L
        out_emb = self.act(self.fc1(history_data))
        # print(out_emb.shape)
        prediction = self.fc2(out_emb).permute(0, 3, 1, 2)     # B, L, N
        return prediction, out_emb         # B, L, N, C
