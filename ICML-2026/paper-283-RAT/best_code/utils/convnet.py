import gym
import torch
from torch import nn

class NatureCNN(nn.Module):
    """
    CNN from DQN Nature paper:
        Mnih, Volodymyr, et al.
        "Human-level control through deep reinforcement learning."
        Nature 518.7540 (2015): 529-533.

    :param observation_space: The observation space of the environment
    :param features_dim: Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(
        self,
        img_size: int,
        features_dim: int = 512,
        with_bn=False,
        p_dropout=0.0,
    ) -> None:
        super().__init__()
        assert features_dim > 0
        self._img_size = img_size[1]
        self._features_dim = features_dim
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper

        n_input_channels = img_size[0]
        # self.cnn = nn.Sequential(
        #     nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
        #     nn.BatchNorm2d(32) if with_bn else nn.Identity(),
        #     nn.ReLU(),
        #     DropBlock2d(p=p_dropout, block_size=8) if p_dropout > 0 else nn.Identity(),
        #     nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
        #     nn.BatchNorm2d(64) if with_bn else nn.Identity(),
        #     nn.ReLU(),
        #     DropBlock2d(p=p_dropout, block_size=4) if p_dropout > 0 else nn.Identity(),
        #     nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
        #     nn.BatchNorm2d(64) if with_bn else nn.Identity(),
        #     nn.ReLU(),
        #     DropBlock2d(p=p_dropout, block_size=3) if p_dropout > 0 else nn.Identity(),
        #     nn.Flatten(),
        # )
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.cnn(torch.ones((1, n_input_channels, self._img_size, self._img_size)).float()).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        if observations.dim() > 4:
            # N, L, C, H, W -> (N * L), C, H, W
            N, L, C, H, W = observations.shape
            observations = observations.reshape(-1, C, H, W)
            embeds = self.linear(self.cnn(observations))
            embeds = embeds.reshape(N, L, -1)
        else:
            embeds = self.linear(self.cnn(observations))
            
        return embeds
