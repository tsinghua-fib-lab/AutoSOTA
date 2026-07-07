import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, embed_dim, with_bn=False):
        super(ResBlock, self).__init__()
        # self.conv1 = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=1, padding='same')
        self.conv1 = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(embed_dim) if with_bn else nn.Identity()
        self.relu1 = nn.ReLU()
        # self.conv2 = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=1, padding='same')
        self.conv2 = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(embed_dim) if with_bn else nn.Identity()
        self.relu2 = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.bn1(out) # bn after relu
        out = self.conv2(out)
        out = self.relu2(x + out)
        out = self.bn2(out)
        return out

class ResNetImpala(nn.Module):
    """
    Model used in the paper "IMPALA: Scalable Distributed Deep-RL with
    Importance Weighted Actor-Learner Architectures" https://arxiv.org/abs/1802.01561
    see https://github.com/openai/baselines/blob/ea25b9e8b234e6ee1bca43083f8f3cf974143998/baselines/common/models.py#L28
    """

    def __init__(self, img_size, embed_size=256, depths=[16, 32, 32], with_bn=False):
        super(ResNetImpala, self).__init__()
        self.img_size = img_size
        self.conv_layers = self._make_layer(depths, with_bn)
        self.relu = nn.ReLU()

        # Compute shape by doing one forward pass
        with torch.no_grad():
            flat_size = torch.flatten(self.conv_layers(torch.ones((1, 3, img_size, img_size)))).shape[0]

        self.linear = nn.Linear(flat_size, embed_size)
    
    def _make_layer(self, depths, with_bn):
        layers = []
        input_sizes = [3] + depths[:-1]
        for i, depth in enumerate(depths):
            layers.append(nn.Conv2d(input_sizes[i], depth, kernel_size=3, stride=1, padding=1))
            layers.append(nn.MaxPool2d(kernel_size=3, stride=2))
            layers.append(ResBlock(depth, with_bn=with_bn))
            layers.append(ResBlock(depth, with_bn=with_bn))
        layers.append(nn.Flatten())
        return nn.Sequential(*layers)

    def forward(self, x):
        if x.dim() > 4:
            # N, L, C, H, W -> (N * L), C, H, W
            N, L, C, H, W = x.shape
            reshaped_x = x.reshape(-1, C, H, W)
            out = self.linear(self.conv_layers(reshaped_x))
            out = out.reshape(N, L, -1)
        else:
            out = self.linear(self.conv_layers(x))

        out = self.relu(out)

        return out