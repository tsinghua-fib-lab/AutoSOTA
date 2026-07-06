from typing import Optional
import torch

from kernels.rbf_kernel import MyRBFKernel
from models.building_blocks.neural_nets import FCNetMLP, ResidualNetMLP, Identity
from models.dkl_lvmogp_base import dkl_lvmogp_base
from models.building_blocks.gp_modules import (
    Prior_H,
    Variational_H,
    Variational_inducing_dist,
    Natural_Variational_inducing_dist,
    TrilNatural_Variational_inducing_dist,
    Inducing_points,
)


class dkl_lvmogp_sarcos(dkl_lvmogp_base):
    """
    DKL-LVMOGP model (with Gaussian qH) on SARCOS dataset.
    """
    def __init__(
            self, D_H, M, qH_mean_field: bool = False, whitening: bool = True, tighter_elbo: bool = True,
            qU_type: str = "standard",  # standard, natural, tril-natural
            neural_network_type: str = "FCNet", out_dim: Optional[int] = 5, hidden_dims: Optional[list] = None,
            num_blocks: Optional[int] = 3, spectral_norm: Optional[bool] = True, sn_ub: Optional[float] = 1., jitter: float = 1e-6,
            # redundant params for both ResNet and FCNet
            use_cache_for_svgp: bool = False,
            sigma_init: float = 0.01, freeze_lik_before_epoch: int = 500,  # parameters to tune for better performance
    ):
        # Hardcoded parameters for the sarcos dataset
        P = 7  # output dimension
        D_X, batch_shape = 21, ()
        # sigma_init = 0.01
        self.freeze_lik_before_epoch = freeze_lik_before_epoch
        in_dim = D_X + D_H
        # init_lengthscale = 0.05

        # pH
        mean_pH_shape = batch_shape + (P, D_H,)
        mean_pH = torch.zeros(mean_pH_shape, dtype=torch.get_default_dtype())
        diag_cov_pH = torch.ones(mean_pH_shape, dtype=torch.get_default_dtype())
        pH = Prior_H(mean_pH, diag_cov_pH)

        # qH
        qH = Variational_H(P, D_H, batch_shape=batch_shape, mean_field=qH_mean_field)

        # qU
        if qU_type == "standard":
            qU = Variational_inducing_dist(M, batch_shape=batch_shape, jitter=jitter)
        elif qU_type == "natural":
            qU = Natural_Variational_inducing_dist(M, batch_shape=batch_shape, jitter=jitter)
        elif qU_type == "tril-natural":
            qU = TrilNatural_Variational_inducing_dist(M, batch_shape=batch_shape, jitter=jitter)

        # Z, inducing points
        if neural_network_type == "FCNet":
            D_T = out_dim
        elif neural_network_type == "ResNet":
            D_T = in_dim
        elif neural_network_type == "Identity":
            D_T = in_dim
        else:
            raise NotImplementedError

        IP_init_shape = batch_shape + (M, D_T,)
        IP_init = torch.randn(IP_init_shape, dtype=torch.get_default_dtype())
        Z = Inducing_points(M, D_T, IP_init, IP_name="Z", IP_joint=True)

        # neural networks
        if neural_network_type == "FCNet":
            neural_net = FCNetMLP(
                in_dim=in_dim, out_dim=D_T, hidden_dims=hidden_dims, spectral_norm=spectral_norm,
                sn_ub_per_layer=sn_ub
            )
        elif neural_network_type == "ResNet":
            neural_net = ResidualNetMLP(
                feature_dim=D_T, num_blocks=num_blocks, spectral_norm=spectral_norm, sn_ub=sn_ub
            )
        elif neural_network_type == "Identity":
            neural_net = Identity()
        else:
            raise NotImplementedError

        # kernel
        MyKernel = MyRBFKernel(
            multi_output=False, has_outputscale=True, ard_num_dims=D_T, batch_shape=torch.Size(batch_shape)
        )

        # MyKernel.lengthscale = init_lengthscale
        # MyKernel.raw_lengthscale.requires_grad = False  # freeze lengthscale

        super(dkl_lvmogp_sarcos, self).__init__(
            kernel=MyKernel, fnet=neural_net, pH=pH, qH=qH, qU=qU, Z=Z,
            lik_model={"type": "Gaussian", "sigma_joint": True, "sigma_init": sigma_init},
            whitening=whitening, jitter=jitter, tighter_elbo=tighter_elbo, use_cache_for_svgp=use_cache_for_svgp,
        )

    # override
    def _epoch_start_hook(self, epoch: int):
        if epoch < self.freeze_lik_before_epoch:
            self.lik_model.raw_sigma.requires_grad = False
        else:
            self.lik_model.raw_sigma.requires_grad = True

    # override
    def train_lvmogp(self, *args, **kwargs):
        coeff_exp_log_lik = 171266

        return super(dkl_lvmogp_sarcos, self).train_lvmogp(*args, coeff_exp_log_lik=coeff_exp_log_lik, **kwargs)