# %%
from itertools import combinations

import lightning as L
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import wandb
from entropy_estimators import mi
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from pycomets.gcm import GCM
from pycomets.pcm import PCM
from pycomets.regression import LM
from simu_data import generate_latent_mvn, generate_obs_data
from sklearn.linear_model import LinearRegression
from torch import Tensor, nn, optim, utils
from torch.utils.data import DataLoader, Dataset
from torch_losses import (
    compute_median_heuristic,
    distance_correlation,
    gaussian_entropy,
    hsic_poly,
    hsic_rbf,
    knn_entropy,
    mmd_poly,
    mmd_rbf,
    orthogonality,
    wilks_lambda_test_torch,
)

# %%
# define dataloader


class SimDataset(Dataset):
    def __init__(
        self,
        seed,
        num_obs,
        Sig_v,
        Sig_w,
        mean_hs,
        Sig_hs,
        etas,
        betas,
        alpha1s,
        alpha2s,
        thetas,
        dim_z,
        Beta=None,
        intercept=False,
        mixing_fn=None,
        discretize=False,
        num_draws=None,
    ):
        # 1. Use a local generator to avoid affecting global torch.randn
        self.master_seed = seed
        self.num_obs = num_obs
        self.num_pop = len(num_obs)
        self.num_draws = num_draws if num_draws is not None else max(num_obs)

        # Store parameters
        self.Sig_v, self.Sig_w = Sig_v, Sig_w
        self.mean_hs, self.Sig_hs = mean_hs, Sig_hs
        self.etas, self.betas = etas, betas
        self.alpha1s, self.alpha2s, self.thetas = alpha1s, alpha2s, thetas
        self.dim_z = dim_z
        self.Beta, self.intercept = Beta, intercept
        self.mixing_fn, self.discretize = mixing_fn, discretize

        # Storage
        self.Z, self.VW, self.H, self.D, self.Y = [], [], [], [], []

        # 2. Generation Loop
        for pp in range(self.num_pop):

            # Base seed for this population
            base_pop_seed = self.master_seed + (pp * 100)  # Spaced out seeds
            num_samples = self.num_obs[pp]

            # 1. Independent generator for H
            gen_h = torch.Generator().manual_seed(base_pop_seed + 1)
            H = (
                generate_latent_mvn(
                    num_samples=num_samples,
                    Sig=self.Sig_hs[pp],
                    generator=gen_h,
                )
                + self.mean_hs[pp]
            )

            # 2. Independent generator for V
            gen_v = torch.Generator().manual_seed(base_pop_seed + 2)
            V = generate_latent_mvn(
                num_samples=num_samples, Sig=self.Sig_v, generator=gen_v
            )

            # 3. Independent generator for W
            gen_w = torch.Generator().manual_seed(base_pop_seed + 3)
            W = generate_latent_mvn(
                num_samples=num_samples, Sig=self.Sig_w, generator=gen_w
            )

            # 4. Use a fourth generator for the noise/logic inside generate_obs_data
            gen_obs = torch.Generator().manual_seed(base_pop_seed + 4)
            Z_pop, VW_pop, D_pop, Y_pop, _ = generate_obs_data(
                V=V,
                W=W,
                H=H,
                eta=self.etas[pp],
                beta=self.betas[pp],
                alpha1=self.alpha1s[pp],
                alpha2=self.alpha2s[pp],
                theta=self.thetas[pp],
                Beta=self.Beta,
                intercept=self.intercept,
                dim_z=self.dim_z,
                mixing_fn=self.mixing_fn,
                discretize=self.discretize,
                generator=gen_obs,
            )

            self.Z.append(Z_pop)
            self.VW.append(VW_pop)
            self.H.append(H)
            self.D.append(D_pop)
            self.Y.append(Y_pop)

    def __len__(self):
        return self.num_draws

    def __getitem__(self, idx):
        # 3. Deterministic index sampling
        # We seed the generator with (master_seed + idx) so that
        # getitem(5) ALWAYS returns the same data regardless of order.
        idx_gen = torch.Generator().manual_seed(self.master_seed + 100000 + idx)

        z_list, vw_list, h_list, d_list, y_list = [], [], [], [], []

        for pp in range(self.num_pop):
            # Pick a random observation from the pre-generated population bank
            obs_idx = torch.randint(
                low=0, high=self.num_obs[pp], size=(1,), generator=idx_gen
            ).item()

            z_list.append(self.Z[pp][obs_idx])
            vw_list.append(self.VW[pp][obs_idx])
            h_list.append(self.H[pp][obs_idx])
            d_list.append(self.D[pp][obs_idx])
            y_list.append(self.Y[pp][obs_idx])

        return {
            "Z": torch.stack(z_list, dim=1),
            "VW": torch.stack(vw_list, dim=1),
            "H": torch.stack(h_list, dim=1),
            "D": torch.stack(d_list, dim=1),
            "Y": torch.stack(y_list, dim=1),
        }


# %%
# define model


class AdaptiveLossBalancer(nn.Module):
    """
    Balances multiple loss terms by adapting lambda weights so that
    all terms live on comparable scales.

    Loss = rec
         + lam1 * reg1
         + lam2 * reg2
         (+ lam3 * reg3)
    """

    def __init__(
        self,
        ema_decay=0.99,
        lambda_lr=0.05,
        lambda_update_every=100,
        clip_ratio=2.0,
        init_lam1=1.0,
        init_lam2=1.0,
        lam_min=1.0,
        lam_max=100.0,
        lam3=0.0,
        reg1_warmup=100,
        reg2_warmup=100,
    ):
        super().__init__()

        # Lambdas are buffers, not parameters
        self.register_buffer("lam1", torch.tensor(init_lam1))
        self.register_buffer("lam2", torch.tensor(init_lam2))
        self.register_buffer("lam3", torch.tensor(lam3))

        # EMA of loss magnitudes
        self.register_buffer("ema_rec", torch.tensor(0.0))
        self.register_buffer("ema_reg1", torch.tensor(0.0))
        self.register_buffer("ema_reg2", torch.tensor(0.0))

        self.ema_decay = ema_decay
        self.lambda_lr = lambda_lr
        self.lambda_update_every = lambda_update_every
        self.clip_ratio = clip_ratio
        self.reg1_warmup = reg1_warmup
        self.reg2_warmup = reg2_warmup
        self.lam_min = lam_min
        self.lam_max = lam_max

        self.register_buffer("step", torch.tensor(0))

    @torch.no_grad()
    def _update_ema(self, name, value):
        buf = getattr(self, name)
        buf.mul_(self.ema_decay).add_(value * (1.0 - self.ema_decay))

    @torch.no_grad()
    def _update_lambda(self, lam, target, current):
        """
        Update lambda so that lam * current ~= target
        """
        ratio = target / (current + 1e-8)
        ratio = torch.clamp(ratio, 1.0 / self.clip_ratio, self.clip_ratio)
        lam.mul_((1.0 - self.lambda_lr) + self.lambda_lr * ratio)
        lam.clamp_(self.lam_min, self.lam_max)

    def forward(self, rec_loss, reg1_loss, reg2_loss, reg3_loss=None):
        """
        Args:
            rec_loss  : reconstruction loss (scalar)
            reg1_loss : auxiliary loss 1
            reg2_loss : auxiliary loss 2
            reg3_loss : optional
        """

        self.step += 1

        # ---- Update EMAs ----
        self._update_ema("ema_rec", rec_loss.detach())
        self._update_ema("ema_reg1", reg1_loss.detach())
        self._update_ema("ema_reg2", reg2_loss.detach())

        # ---- Periodic lambda updates ----
        if self.step % self.lambda_update_every == 0:
            self._update_lambda(self.lam1, self.ema_rec, self.ema_reg1)
            self._update_lambda(self.lam2, self.ema_rec, self.ema_reg2)

        # ---- Total loss ----

        # total_loss = rec_loss + self.lam1 * reg1_loss + self.lam2 * reg2_loss
        if self.step < self.reg1_warmup:
            total_loss = rec_loss
        elif self.step < self.reg2_warmup:
            total_loss = rec_loss + self.lam1 * reg1_loss
        else:
            total_loss = (
                rec_loss
                + torch.clamp(self.lam1, self.lam_min, self.lam_max) * reg1_loss
                + torch.clamp(self.lam2, self.lam_min, self.lam_max) * reg2_loss
            )

        if reg3_loss is not None:
            total_loss = total_loss + self.lam3 * reg3_loss

        # ---- Logging dictionary (keep this!) ----
        log_dict = {
            "loss/rec": rec_loss.detach(),
            "loss/reg1": reg1_loss.detach(),
            "loss/reg2": reg2_loss.detach(),
            "loss/total": total_loss.detach(),
            "lambda/lam1": self.lam1.detach(),
            "lambda/lam2": self.lam2.detach(),
            "ema/rec": self.ema_rec.detach(),
            "ema/reg1": self.ema_reg1.detach(),
            "ema/reg2": self.ema_reg2.detach(),
        }

        return total_loss, log_dict


class LitAutoEncoder(L.LightningModule):
    def __init__(
        self,
        encoder,
        decoder,
        predictor,
        dim_z,
        dim_v,
        dim_w,
        lam1=1.0,
        lam2=1.0,
        lam3=0.0,
        lam4=0.01,
        inv_loss_type="poly",
        inv_ker_poly_degree=2,
        inv_ker_rbf_sigma=None,
        inv_ker_rbf_scales=None,
        ind_loss_type="poly",
        ind_ker_poly_degree=2,
        ind_ker_rbf_sigma=None,
        ind_ker_rbf_scales=None,
        rbf_global_sigma=False,
        dim_v_true=None,
        dim_w_true=None,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.encoder = encoder
        self.decoder = decoder
        self.predictor = predictor

        self.dim_z = dim_z
        self.dim_v = dim_v
        self.dim_w = dim_w
        self.dim_v_true = dim_v_true
        self.dim_w_true = dim_w_true

        self.inv_loss_type = inv_loss_type
        self.ind_loss_type = ind_loss_type
        self.inv_ker_rbf_sigma = inv_ker_rbf_sigma
        self.inv_ker_rbf_scales = inv_ker_rbf_scales if inv_ker_rbf_scales is not None else [1.0]
        self.ind_ker_rbf_sigma = ind_ker_rbf_sigma
        self.ind_ker_rbf_scales = ind_ker_rbf_scales if ind_ker_rbf_scales is not None else [1.0]
        self.rbf_global_sigma = rbf_global_sigma
        self.inv_ker_poly_degree = inv_ker_poly_degree
        self.ind_ker_poly_degree = ind_ker_poly_degree

        self.lam1 = lam1
        self.lam2 = lam2
        self.lam3 = lam3
        self.lam4 = lam4
        self.loss_balancer = AdaptiveLossBalancer(
            ema_decay=0.99,
            lambda_lr=0.05,
            lambda_update_every=100,
            reg1_warmup=0,
            reg2_warmup=0,
        )
        self.adapt_lam = False
        self.nobs = 0
        self.npop = 0

    # --------------------------------------------------
    # Loss components
    # --------------------------------------------------
    def inv_loss(self, w1, w2):
        if self.inv_loss_type == "rbf":
            loss, sigma = mmd_rbf(w1, w2, sigma=self.inv_ker_rbf_sigma, scales=self.inv_ker_rbf_scales)
            self.log("mmd sigma", sigma, on_step=False, on_epoch=True)
        elif self.inv_loss_type == "poly":
            loss = mmd_poly(w1, w2, degree=self.inv_ker_poly_degree, c=1.0)
        elif self.inv_loss_type == "meanvar":
            mean_diff = (w1.mean(0) - w2.mean(0)).pow(2).mean()
            cov_diff = (torch.cov(w1.T) - torch.cov(w2.T)).pow(2).mean()
            loss = mean_diff + cov_diff
        else:
            raise ValueError("Undefined inv_loss_type")
        loss = torch.clamp(loss, min=0.0)
        return loss

    def ind_loss(self, w, v):
        if self.ind_loss_type == "rbf":
            sigma_x = getattr(self, 'ind_ker_rbf_sigma_v', self.ind_ker_rbf_sigma)
            sigma_y = getattr(self, 'ind_ker_rbf_sigma_w', self.ind_ker_rbf_sigma)
            loss, sigma_x, sigma_y = hsic_rbf(
                v,
                w,
                sigma_x=sigma_x,
                sigma_y=sigma_y,
                scales=self.ind_ker_rbf_scales,
            )
            self.log("hsic sigma_x", sigma_x, on_step=False, on_epoch=True)
            self.log("hsic sigma_y", sigma_y, on_step=False, on_epoch=True)
        elif self.ind_loss_type == "poly":
            loss = hsic_poly(
                v, w, degree_x=self.ind_ker_poly_degree, degree_y=2, c=1.0
            )
        elif self.ind_loss_type == "wilks":
            loss, _ = wilks_lambda_test_torch(v, w)
        elif self.ind_loss_type == "orth":
            loss = orthogonality(v, w)
        elif self.ind_loss_type == "dcor":
            loss = distance_correlation(v, w)
        else:
            raise ValueError("Undefined ind_loss_type")
        loss = torch.clamp(loss, min=0.0)
        return loss

    def logdet_cov_penalty(self, w, eps=1e-6):
        """
        Computes -log det(Cov(w)) with numerical stabilization.
        """
        # w: [N, dim_w]
        cov = torch.cov(w.T)

        # Handle dim_w == 1
        if cov.ndim == 0:
            cov = cov.unsqueeze(0).unsqueeze(0)

        cov = cov + eps * torch.eye(
            cov.shape[0], device=cov.device, dtype=cov.dtype
        )

        return -torch.logdet(cov)

    def logdet_corr_penalty(self, w, eps=1e-6):
        """
        Computes -log det(Corr(w)).
        Penalizes degeneracy without penalizing scale.
        """
        # w: [N, dim_w]
        cov = torch.cov(w.T)

        # Handle dim_w == 1
        if cov.ndim == 0:
            return torch.tensor(0.0, device=w.device)

        var = torch.diag(cov)
        std = torch.sqrt(var + eps)

        corr = cov / (std[:, None] * std[None, :])
        corr = corr + eps * torch.eye(
            corr.shape[0], device=w.device, dtype=w.dtype
        )

        return -torch.logdet(corr)

    def logdet_loss(self, w):
        w_lst = w.view(self.npop, self.nobs, -1)
        loss = 0.0
        for pp in range(self.npop):
            loss = loss + self.logdet_cov_penalty(w_lst[pp])
            # loss = loss + self.logdet_corr_penalty(w_lst[pp])
        return loss / self.npop

    # --------------------------------------------------
    def compute_losses(self, x_hat, x, v, w, d):
        w_lst = w.view(self.npop, self.nobs, -1).unbind(0)
        v_lst = v.view(self.npop, self.nobs, -1).unbind(0)

        rec_loss = nn.functional.mse_loss(x_hat, x)

        inv_loss = 0.0
        pop_pairs = list(combinations(range(self.npop), 2))
        for i, j in pop_pairs:
            inv_loss += self.inv_loss(w_lst[i], w_lst[j])
        inv_loss /= len(pop_pairs)

        ind_loss = self.ind_loss(w, v)
        for i in range(self.npop):
            ind_loss += self.ind_loss(w_lst[i], v_lst[i])
        ind_loss += self.ind_loss(w, v)

        d_pred = self.predictor(w).squeeze(1)
        d_true = d.permute(2, 0, 1).flatten(0, 1).squeeze(1)
        rel_loss = nn.functional.mse_loss(d_pred, d_true)

        return rec_loss, inv_loss, ind_loss, rel_loss

    # --------------------------------------------------
    # Training / validation
    # --------------------------------------------------

    def on_train_epoch_start(self):
        if not self.rbf_global_sigma:
            return
        if self.inv_loss_type != "rbf" and self.ind_loss_type != "rbf":
            return

        # Encode all training data to compute global median heuristic
        all_hw, all_hv = [], []
        with torch.no_grad():
            for batch in self.trainer.train_dataloader:
                x = batch["Z"].permute(2, 0, 1).flatten(0, 1).to(self.device)
                hvhw = self.encoder(x)
                all_hv.append(hvhw[:, :self.dim_v])
                all_hw.append(hvhw[:, self.dim_v:self.dim_v + self.dim_w])
        all_hw = torch.cat(all_hw, dim=0)
        all_hv = torch.cat(all_hv, dim=0)

        if self.inv_loss_type == "rbf":
            self.inv_ker_rbf_sigma = compute_median_heuristic(all_hw, None)
            self.log("global_mmd_sigma", self.inv_ker_rbf_sigma, on_step=False, on_epoch=True)
        if self.ind_loss_type == "rbf":
            self.ind_ker_rbf_sigma_v = compute_median_heuristic(all_hv, None)
            self.ind_ker_rbf_sigma_w = compute_median_heuristic(all_hw, None)
            self.log("global_hsic_sigma_v", self.ind_ker_rbf_sigma_v, on_step=False, on_epoch=True)
            self.log("global_hsic_sigma_w", self.ind_ker_rbf_sigma_w, on_step=False, on_epoch=True)

    def training_step(self, batch, batch_idx):
        self.nobs = batch["Z"].shape[0]
        self.npop = batch["Z"].shape[2]

        x = batch["Z"].permute(2, 0, 1).flatten(0, 1)
        vw = batch["VW"].permute(2, 0, 1).flatten(0, 1)

        hvhw = self.encoder(x)
        hv = hvhw[:, : self.dim_v]
        hw = hvhw[:, self.dim_v : (self.dim_v + self.dim_w)]
        x_hat = self.decoder(hvhw)

        rec, inv, ind, rel = self.compute_losses(x_hat, x, hv, hw, batch["D"])

        if self.adapt_lam:
            loss, log_dict = self.loss_balancer(
                rec_loss=rec,
                reg1_loss=inv,
                reg2_loss=ind,
            )
            loss += self.lam3 * rel

            logdet = self.logdet_loss(hw)
            loss += self.lam4 * logdet
            self.lam1 = self.loss_balancer.lam1
            self.lam2 = self.loss_balancer.lam2
        else:
            loss = rec + self.lam1 * inv + self.lam2 * ind + self.lam3 * rel
            # logdet = self.logdet_loss(hw) + self.logdet_loss(hv)
            logdet = self.logdet_loss(hvhw)
            loss += self.lam4 * logdet

        self.logging(
            "train",
            rec,
            inv,
            ind,
            rel,
            loss,
            hvhw,
            vw,
            self.lam1,
            self.lam2,
        )
        self.log("lam1", self.lam1, on_step=False, on_epoch=True)
        self.log("lam2", self.lam2, on_step=False, on_epoch=True)
        opt = self.trainer.optimizers[0]
        self.log("lr", opt.param_groups[0]["lr"], on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        self.nobs = batch["Z"].shape[0]
        self.npop = batch["Z"].shape[2]

        x = batch["Z"].permute(2, 0, 1).flatten(0, 1)
        vw = batch["VW"].permute(2, 0, 1).flatten(0, 1)

        hvhw = self.encoder(x)
        hv = hvhw[:, : self.dim_v]
        hw = hvhw[:, self.dim_v : self.dim_v + self.dim_w]
        x_hat = self.decoder(hvhw)

        rec, inv, ind, rel = self.compute_losses(x_hat, x, hv, hw, batch["D"])
        loss = rec + self.lam1 * inv + self.lam2 * ind + self.lam3 * rel

        self.logging(
            "val",
            rec,
            inv,
            ind,
            rel,
            loss,
            hvhw,
            vw,
            self.lam1,
            self.lam2,
        )

    # --------------------------------------------------
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=1e-3, weight_decay=1e-4
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=20
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/tot_loss",
                "frequency": 1,
            },
        }

    def track_r2(self, hv_hw_pop, v_w_pop):
        mod = LinearRegression()
        X = v_w_pop.cpu().numpy()
        Y = hv_hw_pop.cpu().numpy()
        mod.fit(X=X, y=Y)
        r2 = mod.score(X=X, y=Y)
        return r2

    def track_cond_indep(self, hvhw_pop, vw_pop, given="v"):
        # print(f"hvhw_pop shape: {hvhw_pop.shape}, vw_pop shape: {vw_pop.shape}")
        gcm = GCM()
        if given == "v":
            gcm.test(
                X=hvhw_pop[:, : self.dim_v].cpu().numpy(),
                Y=vw_pop[
                    :, self.dim_v_true : (self.dim_v_true + self.dim_w_true)
                ]
                .cpu()
                .numpy(),
                Z=vw_pop[:, : self.dim_v_true].cpu().numpy(),
                reg_yz=LM(),
                reg_xz=LM(),
                test_type="max",
                B=4999,
                show_summary=False,
            )
        elif given == "w":
            gcm.test(
                X=hvhw_pop[:, self.dim_v : (self.dim_v + self.dim_w)]
                .cpu()
                .numpy(),
                Y=vw_pop[:, : self.dim_v_true].cpu().numpy(),
                Z=vw_pop[
                    :, self.dim_v_true : (self.dim_v_true + self.dim_w_true)
                ]
                .cpu()
                .numpy(),
                reg_yz=LM(),
                reg_xz=LM(),
                test_type="max",
                B=4999,
                show_summary=False,
            )
        return gcm.stat, gcm.pval

    def logging(
        self,
        step_name,
        rec_loss,
        inv_loss,
        ind_loss,
        rel_loss,
        loss,
        hvhw,
        vw,
        lam1,
        lam2,
    ):
        # 1. Log the main scalar losses
        self.log_dict(
            {
                f"{step_name}/rec_loss": rec_loss,
                f"{step_name}/inv_loss": inv_loss,
                f"{step_name}/ind_loss": ind_loss,
                f"{step_name}/rel_loss": rel_loss,
                f"{step_name}/tot_loss": loss,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        # 2. Log scaled losses and current lambdas
        self.log(
            f"{step_name}/scaled_inv",
            lam1 * inv_loss,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            f"{step_name}/scaled_ind",
            lam2 * ind_loss,
            on_step=False,
            on_epoch=True,
        )

        # 3. Population-specific metrics
        with torch.no_grad():
            for pp in range(self.npop):
                start_idx = pp * self.nobs
                end_idx = (pp + 1) * self.nobs

                hvhw_p = hvhw[start_idx:end_idx, :]
                hw_p = hvhw_p[:, self.dim_v : (self.dim_v + self.dim_w)]
                hv_p = hvhw_p[:, : self.dim_v]
                vw_p = vw[start_idx:end_idx, :]

                # --- Degeneracy Check (Eigenvalues of Cov(hW)) ---
                cov_hw = torch.cov(hw_p.T)

                # Handle scalar case if dim_w == 1
                if cov_hw.ndim == 0:
                    cov_hw = cov_hw.unsqueeze(0).unsqueeze(0)

                eigs = torch.linalg.eigvals(cov_hw).real
                self.log(
                    f"{step_name}/min_eig_pop{pp}",
                    eigs.min(),
                    on_step=False,
                    on_epoch=True,
                )
                self.log(
                    f"{step_name}/max_eig_pop{pp}",
                    eigs.max(),
                    on_step=False,
                    on_epoch=True,
                )

                # --- R2 Metrics ---
                if self.dim_v_true is not None and self.dim_w_true is not None:
                    r2_hww = self.track_r2(
                        hw_p,
                        vw_p[
                            :,
                            self.dim_v_true : (
                                self.dim_v_true + self.dim_w_true
                            ),
                        ],
                    )
                    self.log(
                        f"{step_name}/r2_hww_pop{pp}",
                        r2_hww,
                        on_step=False,
                        on_epoch=True,
                    )

                    r2_hvv = self.track_r2(
                        hv_p,
                        vw_p[:, : self.dim_v_true],
                    )
                    self.log(
                        f"{step_name}/r2_hvv_pop{pp}",
                        r2_hvv,
                        on_step=False,
                        on_epoch=True,
                    )

                    r2_hwv = self.track_r2(
                        hw_p,
                        vw_p[:, : self.dim_v_true],
                    )
                    self.log(
                        f"{step_name}/r2_hwv_pop{pp}",
                        r2_hwv,
                        on_step=False,
                        on_epoch=True,
                    )

                # --- Conditional Independence (Every 10 Epochs) ---
                if (
                    self.dim_v_true is not None
                    and self.dim_w_true is not None
                    and (self.current_epoch % 10 == 0)
                ):
                    # Logic only runs once every 10 epochs to save compute
                    _, pval_v = self.track_cond_indep(hvhw_p, vw_p, given="v")
                    _, pval_w = self.track_cond_indep(hvhw_p, vw_p, given="w")

                    self.log(
                        f"{step_name}/gcm_v_pop{pp}",
                        pval_v,
                        on_step=False,
                        on_epoch=True,
                    )
                    self.log(
                        f"{step_name}/gcm_w_pop{pp}",
                        pval_w,
                        on_step=False,
                        on_epoch=True,
                    )

    @torch.no_grad()
    def encode_dataset(self, dataloader):
        """
        Runs inference on a dataloader and returns a list of DataFrames (one per population).
        Dynamically handles missing ground truth columns (e.g., V, W, H) for real-world data.
        """
        self.eval()
        device = self.device

        # Buffer to collect batches
        data_buffer = {}

        # 1. Iterate over batches
        for batch in dataloader:
            # A. Process Input Z (Always Required)
            x = batch["Z"].to(device)

            # In case the dataloader lacks the 'Population' dim
            # (Batch, Dim) -> (Batch, Dim, 1)
            if x.ndim == 2:
                x = x.unsqueeze(2)

            B_size, _, N_pop = x.shape

            # Prepare for Encoder (Flatten populations)
            x_flat = x.permute(2, 0, 1).reshape(-1, self.dim_z)

            # Forward Pass
            hvw_flat = self.encoder(x_flat)
            hz_flat = self.decoder(hvw_flat)

            # Un-flatten: (N_pop*B, Dim) -> (N_pop, B, Dim) -> (B, Dim, N_pop)
            hvw = hvw_flat.view(N_pop, B_size, -1).permute(1, 2, 0)
            hz = hz_flat.view(N_pop, B_size, -1).permute(1, 2, 0)

            # B. Initialize Buffer on First Batch
            # We only track keys that actually exist in this specific dataloader
            if not data_buffer:
                keys_to_track = ["Z", "hVW", "hZ"]  # Always present

                # Check optional ground truth keys
                optional_keys = ["VW", "H", "D", "Y"]
                for k in optional_keys:
                    if k in batch:
                        keys_to_track.append(k)

                for k in keys_to_track:
                    data_buffer[k] = []

            # C. Store Data (Move to CPU)
            data_buffer["Z"].append(batch["Z"].cpu())
            data_buffer["hVW"].append(hvw.cpu())
            data_buffer["hZ"].append(hz.cpu())

            # Store optional keys if they exist
            for k in data_buffer:
                if k in ["Z", "hVW", "hZ"]:
                    continue
                data_buffer[k].append(batch[k].cpu())

        # 2. Concatenate All Batches
        if not data_buffer:
            return []

        # Stack lists of tensors into big tensors
        data_full = {
            k: torch.cat(v, dim=0).float() for k, v in data_buffer.items()
        }

        num_pop = data_full["Z"].shape[2]
        dfs = []

        # 3. Build DataFrame for each Population
        for pp in range(num_pop):

            # Helper to safely extract the slice for population 'pp'
            def get_slice(key):
                tensor = data_full[key]
                return tensor[:, :, pp].numpy()

            # Dictionary to construct DataFrame
            df_dict = {}

            # --- A. Inputs (Z) ---
            z_data = get_slice("Z")
            for i in range(z_data.shape[1]):
                df_dict[f"Z_{i}"] = z_data[:, i]

            # --- B. Learned Latents (hV, hW) ---
            hvw_data = get_slice("hVW")
            # Split based on model dimensions
            for i in range(self.dim_v):
                df_dict[f"hV_{i}"] = hvw_data[:, i]
            for i in range(self.dim_w):
                df_dict[f"hW_{i}"] = hvw_data[:, self.dim_v + i]

            # --- C. Reconstructions (hZ) ---
            hz_data = get_slice("hZ")
            for i in range(hz_data.shape[1]):
                df_dict[f"hZ_{i}"] = hz_data[:, i]

            # --- D. Optional Ground Truths (If present) ---

            # 1. True V and W
            if "VW" in data_full:
                vw_data = get_slice("VW")
                # Try to name them V and W if dimensions match config
                if vw_data.shape[1] == self.dim_v_true + self.dim_w_true:
                    for i in range(self.dim_v_true):
                        df_dict[f"V_{i}"] = vw_data[:, i]
                    for i in range(self.dim_w_true):
                        df_dict[f"W_{i}"] = vw_data[:, self.dim_v_true + i]
                else:
                    # Fallback if dimensions mismatch
                    for i in range(vw_data.shape[1]):
                        df_dict[f"VW_{i}"] = vw_data[:, i]

            # 2. True H
            if "H" in data_full:
                h_data = get_slice("H")
                for i in range(h_data.shape[1]):
                    df_dict[f"H_{i}"] = h_data[:, i]

            # 3. Treatment D
            if "D" in data_full:
                d_data = get_slice("D")
                if d_data.shape[1] == 1:
                    df_dict["D"] = d_data.flatten()
                else:
                    for i in range(d_data.shape[1]):
                        df_dict[f"D_{i}"] = d_data[:, i]

            # 4. Outcome Y
            if "Y" in data_full:
                y_data = get_slice("Y")
                if y_data.shape[1] == 1:
                    df_dict["Y"] = y_data.flatten()
                else:
                    for i in range(y_data.shape[1]):
                        df_dict[f"Y_{i}"] = y_data[:, i]

            # Create DataFrame
            dfs.append(pd.DataFrame(df_dict))

        return dfs
