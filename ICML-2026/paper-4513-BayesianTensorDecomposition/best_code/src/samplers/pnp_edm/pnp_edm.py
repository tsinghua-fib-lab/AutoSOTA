import torch, os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict
from .denoiser_edm import Denoiser_EDM
from ..decomposition.cusp_cp import CUSP_CP

class PnPEDM:
    def __init__(self, config, model, operator, noiser, device):
        self.config = config
        self.model = model
        self.operator = operator
        self.noiser = noiser
        self.device = device
        if config.mode == 'vp':
            self.edm = Denoiser_EDM(model, device, **config.common_kwargs, **config.vp_kwargs, mode='pfode')
        elif config.mode == 've':
            self.edm = Denoiser_EDM(model, device, **config.common_kwargs, **config.ve_kwargs, mode='pfode')
        elif config.mode == 'iddpm':
            self.edm = Denoiser_EDM(model, device, **config.common_kwargs, **config.iddpm_kwargs, mode='pfode')
        elif config.mode == 'edm':
            self.edm = Denoiser_EDM(model, device, **config.common_kwargs, **config.edm_kwargs, mode='pfode')
        elif config.mode == 'vp_sde':
            self.edm = Denoiser_EDM(model, device, **config.common_kwargs, **config.vp_kwargs, mode='sde')
        elif config.mode == 've_sde':
            self.edm = Denoiser_EDM(model, device, **config.common_kwargs, **config.ve_kwargs, mode='sde')
        elif config.mode == 'iddpm_sde':
            self.edm = Denoiser_EDM(model, device, **config.common_kwargs, **config.iddpm_kwargs, mode='sde')
        elif config.mode == 'edm_sde':
            self.edm = Denoiser_EDM(model, device, **config.common_kwargs, **config.edm_kwargs, mode='sde')
        else:
            raise NotImplementedError(f"Mode {self.config.mode} is not implemented.")
        
        if config.decomposition.use:
            self.decomposition = CUSP_CP(**config.decomposition).to(device)
        else:
            self.decomposition = None

    @property
    def display_name(self):
        if self.config.decomposition.use and self.config.use_tau_to_anneal:
            return f'rho0={self.config.rho}-autodecay={self.config.anneal_const}-rhomin={self.config.rho_min}-decomp={self.config.decomposition.use}'
        else:
            return f'rho0={self.config.rho}-decay={self.config.rho_decay_rate}-rhomin={self.config.rho_min}-decomp={self.config.decomposition.use}'

    def __call__(self, gt, y_n, record=False, fname=None, save_root=None, inv_transform=None, metrics={}):
        log = defaultdict(list)
        cmap = 'gray' if gt.shape[1] == 1 else None
        x = self.operator.initialize(gt, y_n)

        # logging
        x_save = inv_transform(x)
        z_save = torch.zeros_like(x_save)
        for name, metric in metrics.items():
            log[name].append(metric(x_save, inv_transform(gt)).item())
        
        xs_save = torch.cat((inv_transform(gt), x_save), dim=-1)
        try:
            zs_save = torch.cat((inv_transform(y_n.reshape(*gt.shape)), z_save), dim=-1)
        except:
            try:
                zs_save = torch.cat((inv_transform(self.operator.A_pinv(y_n).reshape(*gt.shape)), z_save), dim=-1)
            except:
                zs_save = torch.cat((z_save, z_save), dim=-1)

        if record:
            log["gt"] = inv_transform(gt).permute(0, 2, 3, 1).squeeze().cpu().numpy()
            log["x"].append(x_save.permute(0, 2, 3, 1).squeeze().cpu().numpy())

        x_samples = []
        z_samples = []
        iters_count_as_sample = np.linspace(
            self.config.num_burn_in_iters, 
            self.config.num_iters-1, 
            self.config.num_samples_per_run+1, 
            dtype=int
        )[1:]
        assert self.config.num_iters-1 in iters_count_as_sample, "num_iters-1 should be included in iters_count_as_sample"
        sub_pbar = tqdm(range(self.config.num_iters))
        tau_list = []

        for i in sub_pbar:
            # ALGO-01: Support decoupled annealing schedule independent of tau
            if self.config.get("use_decoupled_anneal", False):
                # Decoupled geometric schedule: rho starts high, decays independently
                rho_init = self.config.get("rho_init", 200.0)
                decay_rate = self.config.get("decoupled_decay_rate", 0.95)
                rho_iter = rho_init * (decay_rate ** i)
                rho_iter = max(rho_iter, self.config.rho_min)
            elif self.config.decomposition.use and self.config.use_tau_to_anneal:
                if i == 0:
                    rho_iter = self.config.anneal_const
                else:
                    rho_iter = np.sqrt(self.config.anneal_const / self.decomposition.tau.item())
                rho_iter = np.clip(rho_iter, self.config.rho_min, self.config.rho)
            else:
                rho_iter = self.config.rho * (self.config.rho_decay_rate**i)
                rho_iter = max(rho_iter, self.config.rho_min)

            # likelihood step
            if self.config.decomposition.use:
                self.decomposition.gibbs(
                    data=self.operator.transpose(y_n)[0],
                    z_ten=x[0],
                    mask=self.operator.mask,
                    rho=rho_iter,
                    n_iter=self.config.decomposition.num_gibbs_iters,
                    burnin=50,
                    jump=1
                )
                z = self.decomposition().unsqueeze(0)
                tau_list.append(self.decomposition.tau.item())
            else:
                z = self.operator.proximal_generator(x, y_n, self.noiser.sigma, rho_iter)

            # NOTE: There is one subtle but important difference here:
            # In PnP-EDM, we evaluate x.
            # But in the decomposition approach, we evaluate z instead.

            # prior step
            if z[0].shape != torch.Size([3, 256, 256]):
                z_ = torch.nn.functional.interpolate(
                    z, size=(256, 256), mode='bicubic', align_corners=False)
                _reshape = True
            else:
                z_ = z
                _reshape = False
            x = self.edm(z_, rho_iter)
            if _reshape:
                x = torch.nn.functional.interpolate(
                    x, size=(2048, 2048), mode='bicubic', align_corners=False)

            if i in iters_count_as_sample:
                x_samples.append(x.cpu())
                z_samples.append(z.cpu())

            # logging
            x_save = inv_transform(x)
            z_save = inv_transform(z)
            for name, metric in metrics.items():
                log[f'{name}_z'].append(metric(z_save, inv_transform(gt)).item())
                log[f'{name}_x'].append(metric(x_save, inv_transform(gt)).item())
            sub_pbar.set_description(
                f'(xrange=[{x.min().item():.2f}, {x.max().item():.2f}], zrange=[{z.min().item():.2f}, {z.max().item():.2f}]) | psnr: z-{log["psnr_z"][-1]:.4f}, x-{log["psnr_x"][-1]:.4f}')
            
            if i % (self.config.num_iters//10) == 0:
                xs_save = torch.cat((xs_save, x_save), dim=-1)
                zs_save = torch.cat((zs_save, z_save), dim=-1)
            
            if record:
                log["x"].append(x_save.permute(0, 2, 3, 1).squeeze().cpu().numpy())
                log["z"].append(z_save.permute(0, 2, 3, 1).squeeze().cpu().numpy())

        plt.figure(figsize=(20, 5))
        plt.subplot(1, 3, 1)
        plt.plot(log["psnr_x"], label='x')
        plt.plot(log["psnr_z"], label='z')
        plt.legend()
        plt.title(f'psnrx (max): {np.amax(log["psnr_x"]):.4f}, (last): {log["psnr_x"][-1]:.4f} \n psnrz (max): {np.amax(log["psnr_z"]):.4f}, (last): {log["psnr_z"][-1]:.4f}')
        plt.subplot(1, 3, 2)
        plt.plot(log["ssim_x"], label='x')
        plt.plot(log["ssim_z"], label='z')
        plt.legend()
        plt.title(f'ssimx (max): {np.amax(log["ssim_x"]):.4f}, (last): {log["ssim_x"][-1]:.4f} \n ssimz (max): {np.amax(log["ssim_z"]):.4f}, (last): {log["ssim_z"][-1]:.4f}')
        plt.subplot(1, 3, 3)
        plt.plot(log["lpips_x"], label='x')
        plt.plot(log["lpips_z"], label='z')
        plt.legend()
        if not _reshape:
            plt.title(f'lpipsx (min): {np.amin(log["lpips_x"]):.4f}, (last): {log["lpips_x"][-1]:.4f} \n lpipsz (min): {np.amin(log["lpips_z"]):.4f}, (last): {log["lpips_z"][-1]:.4f}')
        plt.savefig(os.path.join(save_root, 'progress', fname+"_metrics.png"))
        plt.close()

        if self.config.decomposition.use and self.noiser.sigma > 0:
            plt.figure()
            plt.plot(tau_list)
            plt.axhline(y=1./self.noiser.sigma, color='r', linestyle='dashed', label='true tau')
            plt.title(f'tau (last): {tau_list[-1]:.4f}')
            plt.savefig(os.path.join(save_root, 'progress', fname+"_tau.png"))
            plt.close()

        # logging
        xz_save = torch.cat((xs_save, zs_save), dim=-2).permute(0, 2, 3, 1).squeeze().cpu().numpy()
        # plt.imsave(os.path.join(save_root, 'progress', fname+"_x_and_z.png"), xz_save, cmap=cmap)
        np.save(os.path.join(save_root, 'progress', fname+"_log.npy"), log)

        return torch.concat(x_samples, dim=0), torch.concat(z_samples, dim=0)
