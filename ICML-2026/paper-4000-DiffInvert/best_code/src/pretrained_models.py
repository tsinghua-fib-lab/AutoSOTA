# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
from typing import Union, Tuple, List, Optional
import math
import numpy as np

import torch
from torch import nn, Tensor
from torchvision.models import resnet18, ResNet18_Weights, ResNet


class ResNet18Classifier(nn.Module):
    """A wrapper to handle grayscale images for pretrained resnet18"""
    def __init__(self, classifier: ResNet):
        super().__init__()
        self.classifier = classifier

    def forward(self, x: Tensor) -> Tensor:
        assert x.ndim == 4
        if x.shape[1] == 1:
            # grayscale to RGB
            x = x.repeat(1, 3, 1, 1)
        logits = self.classifier(x)
        return logits


def get_resnet18_mnist_classifier_its(device: Union[str, torch.device]) -> ResNet18Classifier:
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(512, 10)
    )
    # load pretrained MNIST classifier checkpoint
    state = torch.load(
        "pretrained_checkpoints/resnet18_mnist_classifier.pth",
        map_location=device
    )
    # change the position of fc weight
    state["fc.1.weight"] = state.pop("fc.weight")
    state["fc.1.bias"]   = state.pop("fc.bias")
    # load weight
    model.load_state_dict(state)
    # move device
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return ResNet18Classifier(model)


def get_resnet18_mnist_classifier(device: Union[str, torch.device]) -> ResNet18Classifier:
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(512, 10)
    model.to(device)
    model.load_state_dict(torch.load("pretrained_checkpoints/resnet18_mnist_classifier.pth", map_location=device))
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return ResNet18Classifier(model)


class LieLACClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(20736, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x: Tensor) -> Tensor:
        assert x.ndim == 4
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = nn.functional.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = nn.functional.log_softmax(x, dim=1)
        return output


def get_lielac_mnist_classifier(device: Union[str, torch.device]) -> LieLACClassifier:
    model = LieLACClassifier()
    model.to(device)
    model.load_state_dict(torch.load("pretrained_checkpoints/lielac_mnist_classifier.pth", map_location=device))
    model.eval()
    return model


class LieLACVAE(nn.Module):
    def __init__(self, x_dim: int, h_dim1: int, h_dim2: int, z_dim: int):
        super().__init__()
        self.dim = int(math.sqrt(x_dim))
        # encoder
        self.fc1 = nn.Linear(x_dim, h_dim1)
        self.fc2 = nn.Linear(h_dim1, h_dim2)
        self.fc31 = nn.Linear(h_dim2, z_dim)
        self.fc32 = nn.Linear(h_dim2, z_dim)
        # decoder
        self.fc4 = nn.Linear(z_dim, h_dim2)
        self.fc5 = nn.Linear(h_dim2, h_dim1)
        self.fc6 = nn.Linear(h_dim1, x_dim)

    def encoder(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        h = nn.functional.relu(self.fc1(x))
        h = nn.functional.relu(self.fc2(h))
        mu = self.fc31(h)
        log_var = self.fc32(h)
        return mu, log_var

    def sampling(self, mu: Tensor, log_var: Tensor) -> Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = eps.mul(std).add_(mu)
        return z

    def decoder(self, z: Tensor) -> Tensor:
        h = nn.functional.relu(self.fc4(z))
        h = nn.functional.relu(self.fc5(h))
        return nn.functional.sigmoid(self.fc6(h)) 

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        bsize = x.shape[0]
        mu, log_var = self.encoder(x.view(bsize, self.dim ** 2))
        z = self.sampling(mu, log_var)
        x_recon = self.decoder(z)
        return x_recon, mu, log_var
    

class BaseModel(nn.Module):
    """
    Base class for all models
    """

    def forward(self, *inputs):
        """
        Forward pass logic

        :return: Model output
        """
        raise NotImplementedError

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + f'\nTrainable parameters: {params}'


class LieLACVanillaVAE(BaseModel):
    def __init__(
        self,
        in_channels: int,
        latent_dims: int,
        hidden_dims: Optional[List[int]] = None,
        flow_check=False,
        **kwargs
    ) -> None:
        """Instantiates the VAE model
        Params:
            in_channels (int): Number of input channels
            latent_dims (int): Size of latent dimensions
            hidden_dims (List[int]): List of hidden dimensions
        """
        super().__init__()
        self.latent_dim = latent_dims
        self.flow_check = flow_check

        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

        # Build Encoder
        for idx, h_dim in enumerate(hidden_dims):
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size=3, stride=2, padding=1),
                    nn.LayerNorm([h_dim, 128//2**(idx+1), 128//2**(idx+1)]),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1]*4, latent_dims)
        self.fc_var = nn.Linear(hidden_dims[-1]*4, latent_dims)

        # Build Decoder
        modules = []
        self.decoder_input = nn.Linear(latent_dims, hidden_dims[-1] * 4)

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride=2,
                                       padding=1,
                                       output_padding=1),
                    nn.LayerNorm([
                        hidden_dims[i + 1],
                        128//2**(len(hidden_dims)-i-1),
                        128//2**(len(hidden_dims)-i-1)
                    ]),
                    nn.LeakyReLU())
            )

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(hidden_dims[-1],
                               hidden_dims[-1],
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1),
            nn.LayerNorm([hidden_dims[-1], 128, 128]),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dims[-1], out_channels=1,
                      kernel_size=3, padding=1)
            )

    def encode(self, inputs: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the convolutional network
        and outputs the latent variables.

        Params:
            input (Tensor): Input tensor [N x C x H x W]

        Returns:
            mu (Tensor) and log_var (Tensor) of latent variables
        """

        result = self.encoder(inputs)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        assert isinstance(mu, Tensor)
        assert isinstance(log_var, Tensor)

        if self.flow_check:
            z, log_det = self.reparameterize(mu, log_var)
            return [mu, log_var, z, log_det]

        z = self.reparameterize(mu, log_var)
        return [mu, log_var, z]

    def decode(self, z: Tensor) -> Tensor:
        """
        Maps the given latent variables
        onto the image space.

        Params:
            z (Tensor): Latent variable [B x D]

        Returns:
            result (Tensor) [B x C x H x W]
        """

        result = self.decoder_input(z)
        result = result.view(result.shape[0], -1, 2, 2)
        result = self.decoder(result)
        result = self.final_layer(result)

        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1)

        Params:
            mu (Tensor): Mean of Gaussian latent variables [B x D]
            logvar (Tensor): log-Variance of Gaussian latent variables [B x D]

        Returns: 
            z (Tensor) [B x D]
        """

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = eps.mul(std).add_(mu)

        if self.flow_check:
            return self.flow(z)

        return z

    def forward(self, inputs: Tensor) -> List[Tensor]:

        if self.flow_check:
            mu, log_var, z, log_det = self.encode(inputs)

            return [self.decode(z), mu, log_var, log_det]

        else:
             mu, log_var, z = self.encode(inputs)

             return [self.decode(z), mu, log_var]

    def sample(self, num_samples: int, current_device: int) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        Params:
            num_samples (Int): Number of samples
            current_device (Int): Device to run the model

        Returns:
            samples (Tensor)
        """
        z = torch.randn(num_samples, self.latent_dim)
        z = z.to(current_device)
        samples = self.decode(z)

        return samples

    def generate(self, x: Tensor) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        Params:
            x (Tensor): input image Tensor [B x C x H x W]

        Returns:
            (Tensor) [B x C x H x W]
        """
        return self.forward(x)[0]


def get_lielac_vae(device: Union[str, torch.device], name: str) -> Union[LieLACVAE, LieLACVanillaVAE]:
    assert name in ("affine", "homography")
    model = LieLACVAE(x_dim=40**2, h_dim1=512, h_dim2=256, z_dim=10)
    model.to(device)
    model.load_state_dict(torch.load("pretrained_checkpoints/lielac_mnist_vae.pth", map_location=device))
    model.eval()
    return model


class LieLACAR(nn.Module):
    """ConvexNet for adversarial regularization"""
    def __init__(self, n_channels: int = 16, kernel_size: int = 5, n_layers: int = 5, n_chan: int = 1):
        super().__init__()
        self.convex = False
        self.n_layers = n_layers
        self.leaky_relu = nn.ReLU()
        # these layers can have arbitrary weights
        self.wxs = nn.ModuleList([
            nn.Conv2d(n_chan, n_channels, kernel_size=kernel_size, stride=1, padding=2, bias=True)
            for _ in range(self.n_layers + 1)
        ])
        # these layers should have non-negative weights
        self.wzs = nn.ModuleList([
            nn.Conv2d(n_channels, n_channels, kernel_size=kernel_size, stride=1, padding=2, bias=False)
            for _ in range(self.n_layers)
        ])
        self.final_conv2d = nn.Conv2d(n_channels, 1, kernel_size=kernel_size, stride=1, padding=2, bias=False)
        self.initialize_weights()

    def initialize_weights(self, min_val: int = 0, max_val: float = 1e-3) -> None:
        for layer in range(self.n_layers):
            self.wzs[layer].weight.data = min_val + (max_val - min_val) * torch.rand_like(self.wzs[layer].weight.data)
        self.final_conv2d.weight.data = min_val + (max_val - min_val) * torch.rand_like(self.final_conv2d.weight.data)

    def clamp_weights(self) -> None:
        for i in range(self.n_layers):
            self.wzs[i].weight.data.clamp_(0)
        self.final_conv2d.weight.data.clamp_(0)

    def forward(self, x: Tensor) -> Tensor:
        if self.convex:
            self.clamp_weights()
        z = self.leaky_relu(self.wxs[0](x))
        for layer_idx in range(self.n_layers):
            z = self.leaky_relu(self.wzs[layer_idx](z) + self.wxs[layer_idx+1](x))
        z = self.final_conv2d(z)
        net_output = z.view(z.shape[0], -1).mean(dim=1, keepdim=True)
        assert net_output.shape[0] == x.shape[0], f"{net_output.shape}, {x.shape[0]}"
        return net_output


def get_lielac_ar(device: Union[str, torch.device], name: str) -> LieLACAR:
    model = LieLACAR()
    model.to(device)
    assert name in ("affine", "homography")
    model.load_state_dict(torch.load(f"pretrained_checkpoints/lielac_ar_{name}.pth", map_location=device))
    model.eval()
    return model
