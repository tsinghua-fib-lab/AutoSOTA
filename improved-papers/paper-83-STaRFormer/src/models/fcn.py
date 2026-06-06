import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['FCN']

class FCN(nn.Module):
    """A 1D Fully Convolutional Network (FCN) for feature extraction.

    This module implements a stack of 1D convolutional and batch normalization layers
    designed for sequence or temporal data. It extracts feature representations
    from the input using multiple Conv1d layers with ReLU activations.

    Attributes:
        conv1 (nn.Conv1d): First 1D convolutional layer.
        bn1 (nn.BatchNorm1d): Batch normalization after the first convolution.
        conv2 (nn.Conv1d): Second 1D convolutional layer.
        bn2 (nn.BatchNorm1d): Batch normalization after the second convolution.
        conv3 (nn.Conv1d): Third 1D convolutional layer.
        bn3 (nn.BatchNorm1d): Batch normalization after the third convolution.
    """
    def __init__(self, in_channels, *args, **kwargs):
        super().__init__(*args, **kwargs)
        """Initializes the FCN module.

        Args:
            in_channels (int): Number of input channels (feature dimensions) for the first convolutional layer.
            *args: Additional positional arguments for the base nn.Module.
            **kwargs: Additional keyword arguments for the base nn.Module.

        """
        # Define the layers
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=128, kernel_size=8, stride=1, padding='same')
        self.bn1 = nn.BatchNorm1d(128)
        
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=5, stride=1, padding='same')
        self.bn2 = nn.BatchNorm1d(256)
        
        self.conv3 = nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding='same')
        self.bn3 = nn.BatchNorm1d(128)
        
        #self.global_avg_pool = nn.AdaptiveAvgPool1d(1)  # Global Average Pooling
        #self.fc = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor):
        """Forward pass through the model.

        Args:
            x (Tensor): Input tensor of shape (N, B, D), where
                N is sequence length,
                B is batch size,
                D is input feature dimension (must match in_channels).

        Returns:
            dict: A dictionary containing:
                - 'features' (Tensor): The extracted feature tensor of shape (N, B, 128).

        Notes:
            - The input is permuted to (B, D, N) for Conv1d processing, then
              permuted back to (N, B, 128) before returning.
            - All convolutional layers use 'same' padding to preserve sequence length.
            - Each convolutional layer is followed by batch normalization and a ReLU activation.
        """

        # [N, B, D] --> [B, D, N]
        x = x.permute(1,2,0)
        # Apply layers
        x = F.relu(self.bn1(self.conv1(x))) # [B, 128, N]
        x = F.relu(self.bn2(self.conv2(x))) # [B, 256, N]
        x = F.relu(self.bn3(self.conv3(x))) # [B, 128, N]
        # [B, 128, N] --> [N, B, 128]
        x = x.permute(2,0,1)
        return {'features': x}