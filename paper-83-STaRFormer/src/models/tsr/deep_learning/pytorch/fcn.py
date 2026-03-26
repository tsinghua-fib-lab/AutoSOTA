import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

from .deep_learning_models import DLRegressorPyTorch

__all__ = ['FCNRegressorPyTorch']

class FCNRegressorPyTorch(DLRegressorPyTorch):
    """
    This is a class implementing the FCN model for time series regression.
    The code is adapted from https://github.com/hfawaz/dl-4-tsc designed for time series classification.
    """

    def __init__(
            self,
            output_directory,
            input_shape,
            verbose=False,
            epochs=500,
            batch_size=16,
            loss="mean_squared_error",
            metrics=None,
            config: DictConfig=None
    ):
        """
        Initialise the FCN model

        Inputs:
            output_directory: path to store results/models
            input_shape: input shape for the models
            verbose: verbosity for the models
            epochs: number of epochs to train the models
            batch_size: batch size to train the models
            loss: loss function for the models
            metrics: metrics for the models
        """

        self.name = "FCN"
        super().__init__(
            output_directory=output_directory,
            input_shape=input_shape,
            verbose=verbose,
            epochs=epochs,
            batch_size=batch_size,
            loss=loss,
            metrics=metrics,
            config=config,
        )
    
    def build_model(self, input_shape):
        """
        Build the FCN model

        Inputs:
            input_shape: input shape for the model
        """
        return FCN(in_channels=input_shape[-1])


class FCN(nn.Module):
    def __init__(self, in_channels, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Define the layers
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=128, kernel_size=8, stride=1, padding='same')
        self.bn1 = nn.BatchNorm1d(128)
        
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=5, stride=1, padding='same')
        self.bn2 = nn.BatchNorm1d(256)
        
        self.conv3 = nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding='same')
        self.bn3 = nn.BatchNorm1d(128)
        
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)  # Global Average Pooling
        self.fc = nn.Linear(128, 1)

    def forward(self, x):
        # [B, N, D] --> [B, D, N]
        x = x.permute(0,2,1)

        # Apply layers
        x = F.relu(self.bn1(self.conv1(x))) # [B, 128, N]
        x = F.relu(self.bn2(self.conv2(x))) # [B, 256, N]
        x = F.relu(self.bn3(self.conv3(x))) # [B, 128, N]
        # [B, 128, N] --> [N, B, 128]
        #x = x.permute(2,0,1)
        #print(x.shape)

        # golabl pooling
        x = self.global_avg_pool(x).squeeze(-1)
        x = self.fc(x)        
        return x