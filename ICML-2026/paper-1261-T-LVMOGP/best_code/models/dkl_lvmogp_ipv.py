from typing import Optional
import torch
from torch import Tensor

from kernels.rbf_kernel import MyRBFKernel
from models.building_blocks.neural_nets import FCNetMLP, ResidualNetMLP, Identity
from models.dkl_lvmogp_base import dkl_lvmogp_base, det_dkl_lvmogp_base
from models.building_blocks.gp_modules import (
    Delta_H,
    Prior_H,
    Variational_H,
    Variational_inducing_dist,
    Natural_Variational_inducing_dist,
    TrilNatural_Variational_inducing_dist,
    Inducing_points,
)


class dkl_lvmogp_ipv(dkl_lvmogp_base):
    """
    DKL-LVMOGP model (with Gaussian qH) on invasive prostate visium dataset
    """
    def __init__(
            self, D_H, M, qH_mean_field: bool = False, whitening: bool = True, tighter_elbo: bool = True, qU_type: str = "standard",  # standard, natural, tril-natural
            neural_network_type: str = "FCNet", out_dim: Optional[int] = 5, hidden_dims: Optional[list] = None, num_blocks: Optional[int] = 3,
            spectral_norm: Optional[bool] = True, sn_ub: Optional[float] = 1., jitter: float = 1e-6, # redundant params for both ResNet and FCNet
            use_cache_for_svgp: bool = False, k_m: float = 0.1, scale_factor: float = 1.0,
    ):
        # Hardcoded parameters for the ipv dataset
        P, D_X, batch_shape = 5000, 2, ()
        alpha_joint, alpha_init = True, 0.01
        lik_model = {"type": "NegativeBinomial", "k_m": k_m, "scale_factor": scale_factor, "alpha_joint": alpha_joint, "alpha_init": alpha_init}
        # sigma_joint, sigma_init = False, 0.01
        # lik_model = {"type": "Gaussian", "sigma_joint": sigma_joint, "sigma_init": sigma_init}
        in_dim = D_X + D_H

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
        # squeeze inducing points to [-1, 1] in the embedding space
        # _max, _min = IP_init.amax(dim=tuple(range(IP_init.ndim - 1))), IP_init.amin(dim=tuple(range(IP_init.ndim - 1)))
        # IP_init = ((IP_init - _min) / (_max - _min)) * 2 - 1
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
            multi_output=False, has_outputscale=False, ard_num_dims=D_T, batch_shape=torch.Size(batch_shape)
        )

        # MyKernel.raw_lengthscale.requires_grad = False  # fix lengthscale

        super(dkl_lvmogp_ipv, self).__init__(
            kernel=MyKernel, fnet=neural_net, pH=pH, qH=qH, qU=qU, Z=Z, lik_model=lik_model,
            whitening=whitening, jitter=jitter, tighter_elbo=tighter_elbo, use_cache_for_svgp=use_cache_for_svgp,
        )

    # override
    def train_lvmogp(
            self, *args, coeff_exp_log_lik = 19582941, **kwargs
    ):
        # coeff_exp_log_lik is the number of total training data points, hardcoded as 19582941 for 10% missing.
        return super(dkl_lvmogp_ipv, self).train_lvmogp(*args, coeff_exp_log_lik=coeff_exp_log_lik, **kwargs)

    def set_pH_means(self, new_pH_means: Tensor):
        # set the prior means of H to new_pH_means, this should be called before training
        assert new_pH_means.shape == self.pH.mean_pH.shape
        assert new_pH_means.requires_grad == False
        self.pH.mean_pH = new_pH_means

    def _epoch_start_hook(self, epoch: int):
        # hook function called at the start of each epoch during training
        if epoch < 500:
            self.lik_model.raw_alpha.requires_grad = False
        else:
            self.lik_model.raw_alpha.requires_grad = True


class det_dkl_lvmogp_ipv(det_dkl_lvmogp_base):
    """
    DKL-LVMOGP model (with Delta qH) on invasive prostate visium dataset
    """
    def __init__(
            self, D_H, M, whitening: bool = True, tighter_elbo: bool = True, qU_type: str = "standard",  # standard, natural, tril-natural
            neural_network_type: str = "FCNet", out_dim: Optional[int] = 5, hidden_dims: Optional[list] = None,
            num_blocks: Optional[int] = 3,
            spectral_norm: Optional[bool] = True, sn_ub: Optional[float] = 1., jitter: float = 1e-6, use_cache_for_svgp: bool = True,
            # redundant params for both ResNet and FCNet
    ):
        # Hardcoded parameters for the ipv dataset
        P, D_X, batch_shape = 5000, 2, ()
        k_m, scale_factor = 0.5, 1.
        alpha_joint, alpha_init = True, 1.
        in_dim = D_X + D_H
        H_trainable, init_as_index = True, False

        # qH
        qH = Delta_H(P=P, D_H=D_H, batch_shape=batch_shape, trainable=H_trainable, init_as_index=init_as_index)

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

        super(det_dkl_lvmogp_ipv, self).__init__(
            kernel=MyKernel, fnet=neural_net, qH=qH, qU=qU, Z=Z,
            lik_model={"type": "NegativeBinomial", "k_m": k_m, "scale_factor": scale_factor, "alpha_joint": alpha_joint, "alpha_init": alpha_init},
            whitening=whitening, jitter=jitter, tighter_elbo=tighter_elbo, use_cache_for_svgp=use_cache_for_svgp,
        )

    # override
    def train_lvmogp(
            self, *args, coeff_exp_log_lik = 10880927, **kwargs
    ):
        # coeff_exp_log_lik is the number of total training data points, hardcoded as 10880927.
        return super(det_dkl_lvmogp_ipv, self).train_lvmogp(*args, coeff_exp_log_lik=coeff_exp_log_lik, **kwargs)

