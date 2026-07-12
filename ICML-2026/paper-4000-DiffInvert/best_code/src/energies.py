# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring,invalid-name
from typing import Optional, Union, Mapping, Tuple

import torch
from torch import nn, Tensor
from torch.nn.functional import cross_entropy, mse_loss

from .groups import Group, ConnectedMatrixLieGroup, Heat1D, Burgers1D
from .pretrained_models import LieLACVAE, LieLACVanillaVAE, LieLACAR


class Energy(nn.Module):
    def __init__(self, group: Union[Group, ConnectedMatrixLieGroup], require_label: bool):
        super().__init__()
        self.group = group
        self.require_label = require_label

    def _check_label(self, y: Optional[Tensor]) -> None:
        if self.require_label and y is None:
            raise ValueError("Labels are required but not provided")
        if not self.require_label and y is not None:
            # warnings.warn("Labels are not required but provided. Ignoring labels.", UserWarning)
            pass

    def forward(self, g_inv: Tensor, x: Tensor, y: Optional[Tensor]) -> Tensor:
        """
        Evaluate energy
        Return:
            energy: [bsize,]
        """
        self._check_label(y)
        x_transformed = self.group.act(g_inv, x)
        energy = self.inner_forward(x_transformed, y)
        return energy

    def inner_forward(self, x: Tensor, y: Optional[Tensor]) -> Tensor:
        raise NotImplementedError


class ImageClassifierLoss(Energy):
    def __init__(self, classifier: nn.Module, group: ConnectedMatrixLieGroup):
        super().__init__(group, require_label=True)
        self.classifier = classifier

    def inner_forward(self, x: Tensor, y: Optional[Tensor]) -> Tensor:
        """
        Compute the energy for the image classification task.
        Arguments:
            x: [bsize, c, h, w]
            y: [bsize,]
        Return:
            energy: [bsize,]
        """
        assert x.ndim == 4
        assert y is not None
        if x.shape[1] == 1:
            # grayscale to RGB
            x = x.repeat(1, 3, 1, 1)
        logits = self.classifier(x)
        energy = cross_entropy(logits.clone(), y.clone(), reduction="none")
        return energy


class ImageClassifierEnergy(Energy):
    def __init__(self, classifier: nn.Module, group: ConnectedMatrixLieGroup):
        super().__init__(group, require_label=False)
        self.classifier = classifier

    def inner_forward(self, x: Tensor, y: Optional[Tensor]) -> Tensor:
        """
        Compute the energy for the image classification task.
        https://arxiv.org/abs/1912.03263
        Arguments:
            x: [bsize, c, h, w]
            y: [bsize,]
        Return:
            energy: [bsize,]
        """
        assert x.ndim == 4
        assert y is not None
        if x.shape[1] == 1:
            # grayscale to RGB
            x = x.repeat(1, 3, 1, 1)
        logits = self.classifier(x)
        energy = -torch.logsumexp(logits, dim=1)
        return energy


class LieLACImageVAEEnergy(Energy):
    def __init__(self, vae: Union[LieLACVAE, LieLACVanillaVAE], ar: LieLACAR, group: ConnectedMatrixLieGroup):
        super().__init__(group, require_label=False)
        self.vae = vae
        self.ar = ar

    def inner_forward(self, x: Tensor, y: Optional[Tensor]) -> Tensor:
        """
        Compute the energy for the image VAE task.
        Arguments:
            x: [bsize, c, h, w]
        Return:
            energy: [bsize,]
        """
        assert x.ndim == 4
        bsize, _, h, w = x.shape
        x = x.mean(dim=1, keepdim=True)

        recon_x, mu, logvar = self.vae(x)
        recon_x = recon_x.reshape(bsize, 1, h, w)
        mu = mu.reshape(bsize, 1, mu.shape[-1])
        logvar = logvar.reshape(bsize, 1, logvar.shape[-1])

        mse = mse_loss(recon_x, x, reduction='none').sum(dim=[1, 2, 3])
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=[1, 2])
        energy = (0.1 * (mse + kld)).unsqueeze(1)

        energy = energy + self.ar(x)

        return energy.view(bsize)


class ImageBoundaryEnergy(Energy):
    def __init__(self, base_energy: Energy):
        super().__init__(base_energy.group, require_label=base_energy.require_label)
        assert hasattr(self.group, 'act_warp')
        self.base_energy = base_energy

    def forward(self, g_inv: Tensor, x: Tensor, y: Optional[Tensor]) -> Tensor:
        """
        Evaluate energy
        Return:
            energy: [bsize,]
        """
        self._check_label(y)
        x_transformed, warp_grid = self.group.act_warp(g_inv, x)
        energy = self.inner_forward(x_transformed, y)

        above = nn.functional.relu(warp_grid - 1)
        below = nn.functional.relu(-1 - warp_grid)
        boundary_l2 = (above.pow(2) + below.pow(2)).mean(dim=[1, 2, 3])

        return energy + boundary_l2

    def inner_forward(self, x: Tensor, y: Optional[Tensor]) -> Tensor:
        return self.base_energy.inner_forward(x, y)


class LieLACL2Target(Energy):
    def __init__(self, group: Union[Heat1D, Burgers1D], target: Mapping[str, Union[Tensor, float]]):
        super().__init__(group, require_label=False)
        self.target = target

    def forward(self, g_inv: Tensor, x: Tuple[Tensor, Tensor], y: Optional[Tensor]) -> Tensor:  # type: ignore
        """
        Evaluate energy
        Return:
            energy: [bsize,]
        """
        self._check_label(y)

        jet_0, X_f = x
        jet_0_transformed = self.group.act(g_inv, jet_0)
        X_f_transformed = self.group.act(g_inv, X_f)
        x_transformed = (jet_0_transformed, X_f_transformed)

        energy = self.inner_forward(x_transformed, y)
        return energy

    def inner_forward(self, x: Tuple[Tensor, Tensor], y: Optional[Tensor]) -> Tensor:  # type: ignore
        jet_0, X_f = x

        x_0, t_0, u_0 = jet_0.to(torch.float64).unbind(1)
        x_f, t_f = X_f.to(torch.float64).unbind(1)

        bsize = jet_0.shape[0]

        energy = torch.zeros(bsize, dtype=torch.float64, device=jet_0.device)

        # initial condition energy
        energy += (x_0.amin(-1) - self.target['x_min']) ** 2
        energy += (x_0.amax(-1) - self.target['x_max']) ** 2
        energy += (t_0.amin(-1) - self.target['t_min']) ** 2
        energy += (t_0.amax(-1) - self.target['t_min']) ** 2  # this has to be t_min
        energy += (u_0.amin(-1) - self.target['u_min']) ** 2
        energy += (u_0.amax(-1) - self.target['u_max']) ** 2

        # prediction domain energy
        energy += (x_f.amin(-1) - self.target['x_min']) ** 2
        energy += (x_f.amax(-1) - self.target['x_max']) ** 2
        energy += (t_f.amin(-1) - self.target['t_min']) ** 2
        energy += (t_f.amax(-1) - self.target['t_max']) ** 2

        return energy
