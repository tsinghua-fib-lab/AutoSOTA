"""
Adapted from DGBFGP (Balik et al., 2025): https://github.com/YigitBalik/DGBFGP
"""

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from model.layer import LinearNet
from model.tools import decompose_cs

class BayesianLinear(nn.Module):
    def __init__(self, in_dim, out_dim, pretrained_model = None, device = "cpu"):
        super(BayesianLinear, self).__init__()
        
        self.input_dim = in_dim
        self.output_dim = out_dim
        self.device = device
        
        scale = 1. * np.sqrt(6. / (in_dim + out_dim))        
        
        self.mu_weights = nn.Parameter(torch.zeros(self.input_dim, self.output_dim))
        self.rho_weights = nn.Parameter(torch.Tensor(self.input_dim, self.output_dim).uniform_(-4, -2))

        if pretrained_model is not None:
            self._transfer_pretrained_weights(pretrained_model)

    def _transfer_pretrained_weights(self, pretrained_model):
        old_input_dim = pretrained_model.input_dim
        self.mu_weights.data[:old_input_dim, :] = pretrained_model.mu_weights.data
        self.rho_weights.data[:old_input_dim, :] = pretrained_model.rho_weights.data
        
    def forward(self, x, stochastic_flag = True, k = 1):
        if stochastic_flag:
            sigma_weights = torch.log(1 + torch.exp(self.rho_weights))
            epsilon_weights = torch.randn(k, self.input_dim, self.output_dim, device=self.device)
            
            weights = self.mu_weights.unsqueeze(0) + epsilon_weights * sigma_weights.unsqueeze(0)

            output = torch.einsum("bkm, kml -> bkl", x, weights)
            
        else:
            weights = self.mu_weights.unsqueeze(0)
            output = torch.matmul(x, weights)

        return output, weights
    
    def loss(self, sigma):
        eps = 1e-6
        sigma_weights= torch.log(1 + torch.exp(self.rho_weights))
        sigma = sigma.unsqueeze(-1)
        if sigma.shape[0] > 1:
            sigma = sigma.view(self.input_dim, self.output_dim)
        a = torch.log((sigma + eps) / (sigma_weights + eps))
        b = (sigma_weights**2 + self.mu_weights**2) / (2 * sigma**2 + eps)
        # KL_weights = torch.sum(- 0.5 + a + b) / (self.input_dim * self.output_dim)
        # KL_weights = torch.sum(- 0.5 + a + b) / (self.input_dim)
        KL_weights = torch.sum(- 0.5 + a + b)
        KL = KL_weights
        return KL
    
    def log_prior(self, A_sample, sigma):
        # mu = 0, sigma = sigma
        sigma = sigma.unsqueeze(-1)
        log_p_A = - torch.log(sigma) - 0.5 * np.log(2 * np.pi) - 0.5 * A_sample**2 / sigma**2
        return torch.sum(log_p_A, dim = [1, 2]) / (self.input_dim * self.output_dim)
    
    def log_posterior(self, A_sample):
        # mu = mu, sigma = sigma
        sigma_weights = torch.log(1 + torch.exp(self.rho_weights))
        mu_weights = self.mu_weights
        log_q_A = - torch.log(sigma_weights) - 0.5 * np.log(2 * np.pi) - 0.5 * (A_sample - mu_weights)**2 / sigma_weights**2
        return torch.sum(log_q_A, dim = [1, 2]) / (self.input_dim * self.output_dim)

    def get_output_distribution(self, x):
        z_mean = torch.matmul(x, self.mu_weights)
        z_std = torch.sqrt(torch.matmul(x**2, torch.log(1 + torch.exp(self.rho_weights))**2))
        return z_mean, z_std

    
class OneHotEncoder(nn.Module):
    def __init__(self, dim):
        super(OneHotEncoder, self).__init__()
        self.dim = dim
    
    def forward(self, x):
        x = x.long()
        x = F.one_hot(x, self.dim).float()
        return x

class BasisFunction(nn.Module):
    def __init__(self, M, basis_func : str = "regff", type = "SE", 
                 scale = 1.0, alpha = 1.0, 
                 alpha_fixed : bool = False,  scale_fixed : bool = False, C = 0, 
                 dim = 1, **kwargs):
        super(BasisFunction, self).__init__()
        self.M = M
        self.type = type
        self.device = kwargs.get("device", "cpu")
        self.basis_func = basis_func

        se_scales = float(scale)
        se_alphas = float(alpha)
        self.dim = dim

        if type == "SE" or type == "PROD":
            
            self.scale_prior_mean = torch.tensor([0], device = self.device)
            self.scale_prior_std = torch.tensor([1.], device = self.device)
            self.scale_posterior_mean = nn.Parameter(torch.tensor([se_scales] * dim, device = self.device))
            self.scale_posterior_log_std = nn.Parameter(torch.tensor([np.log(1)] * dim, device = self.device))
            print(f"{type} Initial E[ell] {torch.exp(self.scale_posterior_mean + torch.exp(self.scale_posterior_log_std)**2 / 2)}")

            self.alpha_prior_mean = torch.tensor([0], device = self.device)
            self.alpha_prior_std = torch.tensor([1.], device = self.device)
            self.alpha_posterior_mean = nn.Parameter(torch.tensor([se_alphas] * dim, device = self.device))
            self.alpha_posterior_log_std = nn.Parameter(torch.tensor([np.log(1)] * dim, device = self.device))

            if alpha_fixed:
                self.alpha_posterior_log_std.requires_grad_(False)
                self.alpha_posterior_mean.requires_grad_(False)
            
            if scale_fixed:
                self.scale_posterior_mean.requires_grad_(False)
                self.scale_posterior_log_std.requires_grad_(False)
                
            if type == "PROD":
                self.C = C
                # self.cat_eigval, self.cat_eigvec = decompose_cs(C, - 1 / (C - 1)) # zero sum kernel
                self.cat_eigval, self.cat_eigvec = decompose_cs(C, 0) # diagonal kernel 
                self.cat_eigval = torch.tensor(self.cat_eigval, device = self.device)
                self.cat_eigvec = torch.tensor(self.cat_eigvec, device = self.device)

        if basis_func == "regff" or basis_func == "hs":
            self.omega = torch.arange(1, self.M + 1, device = self.device).unsqueeze(0)
            # self.omega = torch.linspace(1, M, steps=M/10, device = self.device).unsqueeze(0)
            self.J = float(5.12)
        else:
            # TODO: Implement random Fourier features
            pass

    def set_J(self, J):
        if self.basis_func == "hs":
            self.J = float(J)
        else:
            raise ValueError("J can only be set for Hilbert space embeddings")
    def forward(self, x, v = None, stochastic_flag = True):
        densities = torch.ones((self.dim, self.M), device = self.device)
        if self.type == "SE" or self.type == "PROD":
            scale_posterior_std = torch.exp(self.scale_posterior_log_std)
            alpha_posterior_std = torch.exp(self.alpha_posterior_log_std)
            if stochastic_flag:
                if self.scale_posterior_mean.requires_grad:
                    scales = torch.randn(self.dim, device = self.device) * scale_posterior_std + self.scale_posterior_mean
                    scales = torch.exp(scales)
                else:
                    scales = torch.exp(self.scale_posterior_mean + scale_posterior_std**2 / 2)
                if self.alpha_posterior_log_std.requires_grad:
                    alphas = torch.randn(self.dim, device = self.device) * alpha_posterior_std + self.alpha_posterior_mean
                    alphas = torch.exp(alphas)
                else:
                    alphas = torch.exp(self.alpha_posterior_mean + alpha_posterior_std**2 / 2)
            else:
                scales = torch.exp(self.scale_posterior_mean + scale_posterior_std**2 / 2)
                alphas = torch.exp(self.alpha_posterior_mean + alpha_posterior_std**2 / 2)
                # print(f"{type} Prediction E[ell] {scales}")
            omega = self.omega.to(self.device).repeat(self.dim, 1)
            alphas = alphas.view(-1, 1)
            scales = scales.view(-1, 1)
            if self.basis_func == "regff":
                densities = (alphas**2 * scales * np.sqrt(2 * torch.pi) * torch.exp(- scales**2 * omega**2 / 2))
            elif self.basis_func == "hs":
                sqrt_lambdas = torch.pi * omega / (2 * self.J)
                densities = (alphas**2 * scales * np.sqrt(2 * torch.pi) * torch.exp(- scales**2 * sqrt_lambdas**2 / 2))
                # spectral density of matern 3/2 kernel
                # densities = alphas**2 * 12 * np.sqrt(3) * scales / (3 + scales**2 * sqrt_lambdas**2)**2
        elif self.type == "BIN":
            # one-hot encoding for binary variables
            phi_x = torch.zeros(x.shape[0], self.M, device = self.device)
            phi_x[:, 0] = 1 - x
            phi_x[:, 1] = x 
            # convert to -1 and 1
            phi_x = 2 * phi_x - 1
            densities = torch.ones((1, self.M), device = self.device)
            return phi_x, densities
        elif self.type == "ID":
            phi_x = x
            densities = torch.ones((self.dim, self.M), device = self.device)# * 0.1
            return phi_x, densities
        elif self.type == "POLY":
            p = torch.arange(1, self.M + 1, device = self.device)
            phi_x = x**p.unsqueeze(0)
            phi_x = phi_x.unsqueeze(1)
            densities = torch.ones((self.dim, self.M), device = self.device)
            return phi_x, densities
        
        if self.basis_func == "regff":
            wx = x.unsqueeze(-1) * self.omega
            phi_x  = torch.concat([torch.cos(wx), torch.sin(wx)], dim = -1)
            densities = torch.cat([densities, densities], dim = -1)
        elif self.basis_func == "hs":
            phi_x = 1 / np.sqrt(self.J) * torch.sin(torch.pi * self.omega * (x.unsqueeze(-1) + self.J) / (2 * self.J))

        if self.type == "PROD":
            vec = self.cat_eigvec[v.long()]
            phi_x = (phi_x.unsqueeze(-1) * vec.unsqueeze(-2)).view(*phi_x.shape[:-1],  densities.shape[-1] * self.C)
            densities = (densities.unsqueeze(-1) * self.cat_eigval.view(1, 1, self.C)).view(self.dim, densities.shape[-1] * self.C)
        
        densities = densities + 1e-5
        return phi_x, densities
    def loss(self):
        if self.type == "SE" or self.type == "PROD":
            scale_posterior_std = torch.exp(self.scale_posterior_log_std)
            alpha_posterior_std = torch.exp(self.alpha_posterior_log_std)
            prior_mean = self.scale_prior_mean
            prior_std = self.scale_prior_std
            posterior_mean = self.scale_posterior_mean
            posterior_std = scale_posterior_std
            KL_l = torch.sum(((posterior_mean - prior_mean)**2 + posterior_std**2 - prior_std**2) / (2 * prior_std**2) 
                             + torch.log(prior_std / posterior_std))
            prior_mean = self.alpha_prior_mean
            prior_std = self.alpha_prior_std
            posterior_mean = self.alpha_posterior_mean
            posterior_std = alpha_posterior_std
            KL_a = torch.sum(((posterior_mean - prior_mean)**2 + posterior_std**2 - prior_std**2) / (2 * prior_std**2) 
                             + torch.log(prior_std / posterior_std))
            return KL_l + KL_a
        else:
            return 0.0
        
class CovariateModule(nn.Module):
    def __init__(self, covar_info : dict):
        super(CovariateModule, self).__init__()
        self.covar_type = covar_info["type"]
        self.basis_func = covar_info["basis"]
        self.A = covar_info["A"]

        if self.covar_type in ["SE", "CA", "BIN", "ID_embed", "ID", "POLY"]:
            self.index = covar_info['index']
        elif self.covar_type == "PROD":
            self.cts_covar = covar_info["cts_covar"]
            self.cat_covar = covar_info["cat_covar"]
            self.interaction_index = covar_info["interaction_index"]
    
    def forward(self, x, x_id = None, y = None, stochastic_flag = True, k = 1):

        if self.covar_type in ["SE", "CA", "BIN", "ID_embed", "POLY"]:
            input_x = x[..., self.index]
            phi_x, densities_c = self.basis_func(input_x, stochastic_flag = stochastic_flag)
            gp_sample, A_sample = self.A(phi_x, stochastic_flag, k)
        elif self.covar_type == "ID":
            phi_x, densities_c = self.basis_func(x_id)
            gp_sample, A_sample = self.A(phi_x, stochastic_flag, k)
        elif self.covar_type == "PROD":
            cts_covar = self.cts_covar
            cat_covar = self.cat_covar
            x_cts = x[..., cts_covar]
            x_cat = x[..., cat_covar]
            phi_x, densities_c = self.basis_func(x_cts, x_cat, stochastic_flag = stochastic_flag)
            gp_sample, A_sample = self.A(phi_x, stochastic_flag, k)
        
        return gp_sample, densities_c, A_sample
    
    def loss(self, sigma):
        KL = self.A.loss(sigma)
        KL += self.basis_func.loss()
        return KL
    
    def log_prior(self, A_sample, sigma):
        log_p_A = self.A.log_prior(A_sample, sigma)
        return log_p_A
    
    def log_posterior(self, A_sample):
        log_q_A = self.A.log_posterior(A_sample)
        return log_q_A

class DGBFGP(nn.Module):
    def __init__(self, y_num_dim, x_num_dim, latent_dim, P, id_embed_dim, id_handler,
                 M, C, id_covariate, se_idx, ca_idx, bin_idx, poly_idx, interactions, basis_func : str = "hs",  
                 scale : float = 1.0, alpha : float = 1.0, alpha_fixed : bool = False, scale_fixed : bool = False,
                 vy_init=1, vy_fixed=False, p_drop = 0.2, dec_latent_list = [64, 32], 
                 k = 1, **kwargs):
        super(DGBFGP, self).__init__()
        assert basis_func in ["regff", "hs"] # only support regular Fourier features and Hilbert space embeddings
        self.y_num_dim = y_num_dim
        self.latent_dim = latent_dim
        self.id_covariate = id_covariate
        self.device = kwargs.get("device", "cpu")

        max_idx = max(se_idx + ca_idx + bin_idx)
        id_idx = []
        if id_handler == "onehot":
            id_embed_dim = P
            self.embed_model = OneHotEncoder(P)
            self.x_num_dim = x_num_dim
            id_idx = [id_covariate]
        else:
            self.id_covariate = np.inf
            self.x_num_dim = x_num_dim
        
        self.id_idx = id_idx
        self.id_handler = id_handler
        self.se_idx = se_idx
        self.ca_idx = ca_idx
        self.bin_idx = bin_idx
        self.interactions = interactions

        if isinstance(M, int):
            M = np.array([M] * self.x_num_dim)
        
        M[bin_idx] = 2
        # if id_handler == "onehot":
        #     M[id_idx] = 1
        # TODO: add support for different M for each covariate

        self.decoder = LinearNet(input_dim=latent_dim, latent_size_list=dec_latent_list, output_dim=y_num_dim, act_name="relu").to(self.device)
        LM = BayesianLinear

        self.covariate_modules = nn.ModuleList()
        for i in range(self.x_num_dim):
            index = i if i <= self.id_covariate else i - 1
            covar_info = {"index": index}
            if i in se_idx:
                covar_info.update({
                    "type": "SE",
                    "basis": BasisFunction(M[i], basis_func, "SE", 
                                                scale, alpha, 
                                                alpha_fixed, scale_fixed, dim = latent_dim, **kwargs),
                    "A": BayesianLinear(M[i] if basis_func == "hs" else 2 * M[i], latent_dim, device=self.device)
                })
                self.covariate_modules.append(CovariateModule(covar_info))
            if i in poly_idx:
                covar_info.update({
                    "type": "POLY",
                    "basis": BasisFunction(1, basis_func, "POLY", 
                                                scale, alpha, 
                                                alpha_fixed, scale_fixed, dim = latent_dim, **kwargs),
                    "A": BayesianLinear(1, latent_dim, device=self.device)
                })
                self.covariate_modules.append(CovariateModule(covar_info))
            if i in ca_idx:
                covar_info.update({
                    "type": "CA",
                    "basis": BasisFunction(M[i], basis_func, "CA", 
                                                scale, alpha, 
                                                alpha_fixed, scale_fixed, **kwargs),
                    "A": BayesianLinear(M[i] if basis_func == "hs" else 2 * M[i], latent_dim, device=self.device)
                })
                self.covariate_modules.append(CovariateModule(covar_info))
            if i in bin_idx:
                covar_info.update({
                    "type": "BIN",
                    "basis": BasisFunction(M[i], basis_func, "BIN", 
                                                scale, alpha, 
                                                alpha_fixed, scale_fixed, **kwargs),
                    "A": BayesianLinear(M[i], latent_dim, device=self.device)
                })
                self.covariate_modules.append(CovariateModule(covar_info))
            if i in id_idx:
                covar_info.update({
                    "type": "ID",
                    "basis": BasisFunction(id_embed_dim, basis_func, "ID", 
                                                scale, alpha, 
                                                alpha_fixed, scale_fixed, **kwargs),
                    "A": LM(id_embed_dim, latent_dim, device=self.device)
                })
                self.covariate_modules.append(CovariateModule(covar_info))

        for i, interaction in enumerate(interactions):
            cts_covar = interaction[0]
            cat_covar = interaction[1]
            if cts_covar > self.id_covariate:
                cts_covar -= 1
            if cat_covar > self.id_covariate:
                cat_covar -= 1
            interaction_info = {
                "type": "PROD",
                "basis": BasisFunction(M[cts_covar], basis_func, "PROD", 
                                            scale, alpha, 
                                            alpha_fixed, scale_fixed, C[i], dim = latent_dim,**kwargs),
                "A": BayesianLinear(M[cts_covar] * C[i] if basis_func == "hs" else 2 * M[cts_covar] * C[i], latent_dim, device=self.device),
                'cts_covar': cts_covar,
                'cat_covar': cat_covar,
                'interaction_index': i
            }
            self.covariate_modules.append(CovariateModule(interaction_info))
        
        self.p_drop = p_drop

        min_log_vy = torch.Tensor([-8.0])

        log_vy_init = torch.log(vy_init - torch.exp(min_log_vy))
        # log variance
        if isinstance(vy_init, float):
            self._log_vy = nn.Parameter(torch.Tensor(y_num_dim * [log_vy_init]))
        else:
            self._log_vy = nn.Parameter(torch.Tensor(log_vy_init))

        if vy_fixed:
            self._log_vy.requires_grad_(False)
        
        self.register_buffer('min_log_vy', min_log_vy * torch.ones(1))

        self.k = self.k_init = k # number of samples for importance sampling
        
    @property
    def vy(self):
        log_vy = self.min_log_vy + F.softplus(self._log_vy - self.min_log_vy)
        return torch.exp(log_vy)

    @vy.setter
    def vy(self, vy):
        assert torch.min(torch.tensor(vy)) >= 0.0005, "Smallest allowed value for vy is 0.0005"
        with torch.no_grad():
            self._log_vy.copy_(torch.log(vy - torch.exp(self.min_log_vy)))
    
    def encode(self, x, y = None, stochastic_flag = True, train = True):
        x_id = None
        if self.id_handler != "none":
            x_id = x[..., self.id_covariate].long()
            x = torch.cat([x[...,:self.id_covariate], x[...,self.id_covariate + 1:]], dim = -1)

        if self.id_handler == "onehot":
            x_id = self.embed_model(x_id)


        z = []
        densities = []
        A_samples = []
        for covar_module in self.covariate_modules:
            z_c, densities_c, A_c = covar_module(x, x_id, y, stochastic_flag, self.k)
            z.append(z_c)
            densities.append(densities_c)
            A_samples.append(A_c)
        if train:
            z = sum(z)
        return z, densities, A_samples
    
    def decode(self, z):
        y = self.decoder(z)
        return y

    def forward(self, x, y = None, stochastic_flag = True):
        z, densities, A_samples = self.encode(x, y, stochastic_flag)
        y = self.decode(z)
        return y, z, densities, A_samples
    
    def KL_loss(self, densities):
        kls = []
        for c,m in enumerate(self.covariate_modules):
            sigma_bar = torch.sqrt(densities[c])
            kls.append(m.loss(sigma_bar))
        KL = sum(kls)
        return KL
    
    def loss_function(self, recon_y, y, mask, densities):
        recon_y = recon_y.squeeze()
        loss = nn.MSELoss(reduction='none')
        se = torch.mul(loss(recon_y.view(-1, self.y_num_dim), y.view(-1, self.y_num_dim)), mask.view(-1, self.y_num_dim))
        mask_sum = torch.sum(mask.view(-1, self.y_num_dim), dim=1)
        mask_sum[mask_sum == 0] = 1
        mse = torch.sum(se, dim=1) / mask_sum

        nll = se / (2 * torch.exp(self._log_vy))
        nll += 0.5 * (np.log(2 * np.pi) + self._log_vy)

        kls = []
        i = 0
        for c,m in enumerate(self.covariate_modules):
            sigma_bar = torch.sqrt(densities[c])
            kls.append(m.loss(sigma_bar))
        KL = sum(kls)

        if self.id_handler == "bayesian_embedding":
            KL += self.embed_model.loss()
        return mse, torch.sum(nll, dim=1), KL
    
    def iwae_loss_function(self, recon_y, y, mask, densities, A_samples):
        recon_y = recon_y.view(-1, self.k, self.y_num_dim)
        y = y.view(-1, self.y_num_dim).unsqueeze(1).repeat(1, self.k, 1)
        mask = mask.view(-1, self.y_num_dim).unsqueeze(1).repeat(1, self.k, 1)

        loss = nn.MSELoss(reduction='none')
        se = torch.mul(loss(recon_y, y), mask)
        mask_sum = torch.sum(mask, dim=-1)
        mask_sum[mask_sum == 0] = 1
        mse = torch.sum(se, dim=-1) / mask_sum

        nll = se / (2 * torch.exp(self._log_vy))
        nll += 0.5 * (np.log(2 * np.pi) + self._log_vy)

        log_p_A = 0
        log_q_A = 0
        for c, m in enumerate(self.covariate_modules):
            sigma_bar = torch.sqrt(densities[c]).repeat(self.k, 1)
            log_p_A = log_p_A + m.log_prior(A_samples[c], sigma_bar)
            log_q_A = log_q_A + m.log_posterior(A_samples[c])
        
        log_diff = log_p_A - log_q_A

        return mse, torch.sum(nll, dim=-1), log_diff

    def train(self, mode = True):
        super().train(mode)
        if mode:
            self.k = self.k_init
        else:
            self.k = 1



