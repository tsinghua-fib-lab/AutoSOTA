from inverse_problems.base import BaseOperator
import ehtim.statistics.dataframes as ehdf
import pandas as pd
import torch
import numpy as np
import ehtim as eh
from eval import Evaluator
import copy
from piq import psnr
import torch.nn.functional as F
from typing import Dict


class BlackHoleImaging(BaseOperator):
    """
        PyTorch Version of Black Hole Imaging Forward Operator based on eht-imaging library.
            https://github.com/achael/eht-imaging

        This class utilize a reference observation for observation setup (e.g. telescope u,v map)
    """

    def __init__(self, root='dataset/blackhole', observation_time_ratio=1.0, noise_type='vis_thermal', ttype='nfft', imsize=64, w1=0,
                 w2=1, w3=1, w4=0.5, sigma_noise=0.0, unnorm_shift=1.0, unnorm_scale=0.5, ref_flux=None, device='cuda'):
        super().__init__(sigma_noise, unnorm_shift, unnorm_scale, device)
        # load observations
        A_vis, A_cp, A_camp, obs, im, multiplier, sigma = self.process_obs(root, imsize, observation_time_ratio)

        if ref_flux is not None:
          # rescale reference image to specified `ref_flux`
          im.ivec *= ref_flux / np.sum(im.ivec)

        self.ref_im = im
        self.ref_flux = np.sum(self.ref_im.ivec)
        self.ref_obs = obs
        self.ref_multiplier = multiplier
        self.observation_time_ratio = observation_time_ratio
        self.noise_type = noise_type
        self.ttype = ttype  # 'fast' | 'nfft' | 'direct'
        self.device = device

        # Get index  matrix for closure phases and closure amplitudes
        self.get_index_matrix(obs)

        # sigmas
        self.sigma = torch.tensor(sigma).to(device)

        # forward matrix
        self.A_vis = torch.from_numpy(A_vis).unsqueeze(0).unsqueeze(0).cfloat().to(device)  # [1,1,m,n]
        self.A_cp = torch.from_numpy(A_cp).unsqueeze(1).unsqueeze(1).cfloat().to(device)  # [3,1,1,m,n]
        self.A_camp = torch.from_numpy(A_camp).unsqueeze(1).unsqueeze(1).cfloat().to(device)  # [4,1,1,m,n]

        # dimension
        self.amp_dim = self.A_vis.shape[-2]
        self.cphase_dim = self.A_cp.shape[-2]
        self.logcamp_dim = self.A_camp.shape[-2]
        self.flux_dim = 1

        # params
        self.C = 1
        self.H = imsize
        self.W = imsize
        self.weight_amp = w1 * self.amp_dim
        self.weight_cp = w2 * self.cphase_dim
        self.weight_camp = w3 * self.logcamp_dim
        self.weight_flux = w4

    # 0. set up forward function
    @staticmethod
    def process_obs(
            root,
            imsize,
            observation_time_ratio=1.0
    ):
        obsfile = root + '/' + 'obs.uvfits'
        gtfile = root + '/' + 'gt.fits'
        # load observations
        obs = eh.obsdata.load_uvfits(obsfile)
        # subsample the observation
        pd_data = pd.DataFrame(obs.data)
        time_list = np.array(sorted(list(set(pd_data['time']))))
        time_list = time_list[:int(len(time_list) * observation_time_ratio)]
        pd_data = pd_data[pd_data['time'].isin(time_list)]
        obsdata = pd_data.to_records(index=False).view(np.ndarray).astype(eh.DTPOL_STOKES)
        obs.data = obsdata
        # get the reference ground truth image
        im = eh.image.load_fits(gtfile)
        im = im.regrid_image(im.fovx(), imsize)
        # rescale image
        multiplier = im.ivec.max()
        sigma = obs.data['sigma'] / multiplier
        # get forward model for complex visibilities.
        _, _, A_vis = eh.imaging.imager_utils.chisqdata_vis(obs, im, mask=[])
        # get forward model for closure phases.
        _, _, A_cp = eh.imaging.imager_utils.chisqdata_cphase(obs, im, mask=[])
        # get forward model for closure amplitudes.
        _, _, A_camp = eh.imaging.imager_utils.chisqdata_logcamp(obs, im, mask=[])
        return A_vis, np.stack(A_cp, axis=0), np.stack(A_camp, axis=0), obs, im, multiplier, sigma

    @staticmethod
    def estimate_flux(obs):
        # estimate the total flux from the observation
        data = obs.unpack_bl('ALMA', 'APEX', 'amp')
        amp_list = []
        for pair in data:
            amp = pair[0][1]
            amp_list.append(amp)
        flux = np.median(amp_list)
        return flux

    def get_index_matrix(self, obs):
        obs_data_df = pd.DataFrame(obs.data)
        map_fn, conjugate_fn = {}, {}
        for i, (time, t1, t2) in enumerate(zip(obs_data_df['time'], obs_data_df['t1'], obs_data_df['t2'])):
            map_fn[(time, t1, t2)] = i
            conjugate_fn[(time, t1, t2)] = 0
            map_fn[(time, t2, t1)] = i
            conjugate_fn[(time, t2, t1)] = 1

        # closure phase index
        bispec_df = pd.DataFrame(obs.bispectra(count='min'))
        cp_index, cp_conjugate = [], []
        for time, t1, t2, t3 in zip(bispec_df['time'], bispec_df['t1'], bispec_df['t2'], bispec_df['t3']):
            idx = [map_fn[(time, t1, t2)], map_fn[(time, t2, t3)], map_fn[(time, t3, t1)]]
            conj = [conjugate_fn[(time, t1, t2)], conjugate_fn[(time, t2, t3)], conjugate_fn[(time, t3, t1)]]
            cp_index.append(idx)
            cp_conjugate.append(conj)
        self.cp_index = torch.tensor(cp_index).long().to(self.device)
        self.cp_conjugate = torch.tensor(cp_conjugate).long().to(self.device)

        # log closure amplitude index
        camp_df = pd.DataFrame(obs.c_amplitudes(count='min'))
        camp_index, camp_conjugate = [], []
        for time, t1, t2, t3, t4 in zip(camp_df['time'], camp_df['t1'], camp_df['t2'], camp_df['t3'], camp_df['t4']):
            idx = [map_fn[(time, t1, t2)], map_fn[(time, t3, t4)], map_fn[(time, t1, t4)], map_fn[(time, t2, t3)]]
            conj = [conjugate_fn[(time, t1, t2)], conjugate_fn[(time, t3, t4)], conjugate_fn[(time, t1, t4)],
                    conjugate_fn[(time, t2, t3)]]
            camp_index.append(idx)
            camp_conjugate.append(conj)
        self.camp_index = torch.tensor(camp_index).long().to(self.device)
        self.camp_conjugate = torch.tensor(camp_conjugate).long().to(self.device)

    # 1. visibility and flux from image x in range [0,1]
    def forward_vis(self, x):
        x = x.to(self.A_vis)

        xvec = x.reshape(-1, self.C, 1, self.H * self.W)
        vis = (self.A_vis * xvec).sum(-1, keepdims=True)
        return vis

    def forward_amp(self, x):
        amp = self.forward_vis(x).abs()
        sigmaamp = self.sigma[None, None, :, None] + 0 * amp
        return amp, sigmaamp

    def forward_flux(self, x):
        return x.flatten(1).sum(-1)[:, None, None, None]

    # 2. forward from image x in range [0, 1]
    def forward_bisepectra_from_image(self, x):
        x = x.to(self.A_cp)

        xvec = x.reshape(-1, self.C, 1, self.H * self.W)
        i1 = (self.A_cp[0] * xvec).sum(-1, keepdims=True)
        i2 = (self.A_cp[1] * xvec).sum(-1, keepdims=True)
        i3 = (self.A_cp[2] * xvec).sum(-1, keepdims=True)
        return i1, i2, i3

    def forward_cp_from_image(self, x):
        i1, i2, i3 = self.forward_bisepectra_from_image(x)
        cphase = torch.angle(i1 * i2 * i3)

        v1 = self.sigma[self.cp_index[:, 0]][None, None, :, None]
        v2 = self.sigma[self.cp_index[:, 1]][None, None, :, None]
        v3 = self.sigma[self.cp_index[:, 2]][None, None, :, None]
        sigmacp = (v1 ** 2 / i1.abs() ** 2 + v2 ** 2 / i2.abs() ** 2 + v3 ** 2 / i3.abs() ** 2).sqrt()
        return cphase, sigmacp

    def forward_logcamp_bispectra_from_image(self, x):
        x = x.to(self.A_camp)

        x_vec = x.reshape(-1, self.C, 1, self.H * self.W)
        i1 = (self.A_camp[0] * x_vec).sum(-1, keepdims=True).abs()
        i2 = (self.A_camp[1] * x_vec).sum(-1, keepdims=True).abs()
        i3 = (self.A_camp[2] * x_vec).sum(-1, keepdims=True).abs()
        i4 = (self.A_camp[3] * x_vec).sum(-1, keepdims=True).abs()
        return i1, i2, i3, i4

    def forward_logcamp_from_image(self, x):
        i1, i2, i3, i4 = self.forward_logcamp_bispectra_from_image(x)
        camp = i1.log() + i2.log() - i3.log() - i4.log()

        v1 = self.sigma[self.camp_index[:, 0]][None, None, :, None]
        v2 = self.sigma[self.camp_index[:, 1]][None, None, :, None]
        v3 = self.sigma[self.camp_index[:, 2]][None, None, :, None]
        v4 = self.sigma[self.camp_index[:, 3]][None, None, :, None]
        sigmaca = (v1 ** 2 / i1 ** 2 + v2 ** 2 / i2 ** 2 + v3 ** 2 / i3 ** 2 + v4 ** 2 / i4 ** 2).sqrt()
        return camp, sigmaca

    def forward_from_image(self, x):
        amp, sigmaamp = self.forward_amp(x)
        cphase, sigmacp = self.forward_cp_from_image(x)
        logcamp, sigmacamp = self.forward_logcamp_from_image(x)
        flux = self.forward_flux(x)

        return self.compress(amp, sigmaamp, cphase, sigmacp, logcamp, sigmacamp, flux).float()

    # 3. forward from visibilities
    def correct_vis_direction(self, vis, conj):
        vis = vis * (1 - conj) + vis.conj() * conj
        return vis

    def forward_amp_from_vis(self, vis):
        amp = vis.abs()
        sigmaamp = self.sigma[None, None, :, None] + 0 * amp
        return amp, sigmaamp

    def forward_bisepectra_from_vis(self, vis):
        v1 = vis[:, :, self.cp_index[:, 0], :]
        v2 = vis[:, :, self.cp_index[:, 1], :]
        v3 = vis[:, :, self.cp_index[:, 2], :]

        cj1 = self.cp_conjugate[None, None, :, 0, None]
        cj2 = self.cp_conjugate[None, None, :, 1, None]
        cj3 = self.cp_conjugate[None, None, :, 2, None]

        i1 = self.correct_vis_direction(v1, cj1)
        i2 = self.correct_vis_direction(v2, cj2)
        i3 = self.correct_vis_direction(v3, cj3)
        return i1, i2, i3

    def forward_cp_from_vis(self, vis):
        i1, i2, i3 = self.forward_bisepectra_from_vis(vis)
        cphase = torch.angle(i1 * i2 * i3)

        v1 = self.sigma[self.cp_index[:, 0]][None, None, :, None]
        v2 = self.sigma[self.cp_index[:, 1]][None, None, :, None]
        v3 = self.sigma[self.cp_index[:, 2]][None, None, :, None]
        sigmacp = (v1 ** 2 / i1.abs() ** 2 + v2 ** 2 / i2.abs() ** 2 + v3 ** 2 / i3.abs() ** 2).sqrt()
        return cphase, sigmacp

    def forward_logcamp_bispectra_from_vis(self, vis):
        v1 = vis[:, :, self.camp_index[:, 0], :].abs()
        v2 = vis[:, :, self.camp_index[:, 1], :].abs()
        v3 = vis[:, :, self.camp_index[:, 2], :].abs()
        v4 = vis[:, :, self.camp_index[:, 3], :].abs()

        cj1 = self.camp_conjugate[None, None, :, 0, None]
        cj2 = self.camp_conjugate[None, None, :, 1, None]
        cj3 = self.camp_conjugate[None, None, :, 2, None]
        cj4 = self.camp_conjugate[None, None, :, 3, None]

        i1 = self.correct_vis_direction(v1, cj1)
        i2 = self.correct_vis_direction(v2, cj2)
        i3 = self.correct_vis_direction(v3, cj3)
        i4 = self.correct_vis_direction(v4, cj4)
        return i1, i2, i3, i4

    def forward_logcamp_from_vis(self, vis):
        i1, i2, i3, i4 = self.forward_logcamp_bispectra_from_vis(vis)
        logcamp = i1.log() + i2.log() - i3.log() - i4.log()

        v1 = self.sigma[self.camp_index[:, 0]][None, None, :, None]
        v2 = self.sigma[self.camp_index[:, 1]][None, None, :, None]
        v3 = self.sigma[self.camp_index[:, 2]][None, None, :, None]
        v4 = self.sigma[self.camp_index[:, 3]][None, None, :, None]
        sigmaca = (v1 ** 2 / i1 ** 2 + v2 ** 2 / i2 ** 2 + v3 ** 2 / i3 ** 2 + v4 ** 2 / i4 ** 2).sqrt()
        return logcamp, sigmaca

    def forward_from_vis(self, x):
        vis = self.forward_vis(x)

        amp, sigmaamp = self.forward_amp_from_vis(vis)
        cphase, sigmacp = self.forward_cp_from_vis(vis)
        logcamp, sigmacamp = self.forward_logcamp_from_vis(vis)
        flux = self.forward_flux(x)

        return self.compress(amp, sigmaamp, cphase, sigmacp, logcamp, sigmacamp, flux).float()

    # 4. forward from EHT library
    @staticmethod
    def pt2ehtim(pt_image, res, ref_im):
        im = copy.deepcopy(ref_im)
        im.ivec = pt_image.clip(0, 1).detach().cpu().numpy().reshape(res, res).flatten()
        return im

    @staticmethod
    def pt2ehtim_batch(pt_images, res, ref_im):
        eht_images = []
        for pt_image in pt_images:
            eh_image = copy.deepcopy(ref_im)
            eh_image.ivec = pt_image.clip(0, 1).detach().cpu().numpy().reshape(res, res).flatten()
            eht_images.append(eh_image)
        return eht_images

    def forward_from_eht(self, x):
        ref_im = self.ref_im
        multiplier = self.ref_multiplier
        ref_obs = self.ref_obs
        res = self.H
        pt_obs = []
        for pt_image in x:
            eh_image = self.pt2ehtim(pt_image, res, ref_im)
            eh_image.ivec = eh_image.ivec * multiplier

            # observe the image
            obs = eh_image.observe_same_nonoise(ref_obs, ttype=self.ttype, verbose=False)

            # visibilities amplitude
            adf = ehdf.make_amp(obs, debias=False)
            amp = torch.from_numpy(adf['amp'].to_numpy())[None, None, :, None].float().to(x.device) / multiplier
            sigmaamp = torch.from_numpy(adf['sigma'].to_numpy())[None, None, :, None].float().to(
                x.device) / multiplier

            # closure phase
            cdf = ehdf.make_cphase_df(obs, count='min')
            cp = torch.from_numpy(cdf['cphase'].to_numpy())[None, None, :, None].float().to(x.device) * eh.DEGREE
            sigmacp = torch.from_numpy(cdf['sigmacp'].to_numpy())[None, None, :, None].float().to(
                x.device) * eh.DEGREE

            # log closure amplitude
            ldf = ehdf.make_camp_df(obs, count='min')
            camp = torch.from_numpy(ldf['camp'].to_numpy())[None, None, :, None].float().to(x.device)
            sigmaca = torch.from_numpy(ldf['sigmaca'].to_numpy())[None, None, :, None].float().to(x.device)

            # flux
            flux = torch.tensor([self.estimate_flux(obs)])[None, None, :, None].float().to(x.device) / multiplier

            y = torch.cat([amp, sigmaamp, cp, sigmacp, camp, sigmaca, flux], dim=2)
            pt_obs.append(y)
        pt_obs = torch.cat(pt_obs, dim=0).to(x.device)
        return pt_obs

    # 5. chi-square evalutation
    def chi2_amp(self, x, y_amp, y_amp_sigma):
        amp_pred, _ = self.forward_amp(x)
        return self.chi2_amp_from_meas(amp_pred, y_amp, y_amp_sigma)

    @staticmethod
    def chi2_amp_from_meas(y_amp_meas, y_amp, y_amp_sigma):
        residual = y_amp_meas - y_amp
        return torch.mean(torch.square(residual / y_amp_sigma), dim=(1, 2, 3))

    def chi2_cphase(self, x, y_cphase, y_cphase_sigma):
        cphase_pred, _ = self.forward_cp_from_image(x)
        return self.chi2_cphase_from_meas(cphase_pred, y_cphase, y_cphase_sigma)

    @staticmethod
    def chi2_cphase_from_meas(y_cphase_meas, y_cphase, y_cphase_sigma):
        angle_residual = y_cphase - y_cphase_meas
        return 2. * torch.mean((1 - torch.cos(angle_residual)) / torch.square(y_cphase_sigma), dim=(1, 2, 3))

    def chi2_logcamp(self, x, y_camp, y_logcamp_sigma):
        y_camp_pred, _ = self.forward_logcamp_from_image(x)
        return self.chi2_logcamp_from_meas(y_camp_pred, y_camp, y_logcamp_sigma)

    @staticmethod
    def chi2_logcamp_from_meas(y_logcamp_meas, y_logcamp, y_logcamp_sigma):
        return torch.mean(torch.abs((y_logcamp_meas - y_logcamp) / y_logcamp_sigma) ** 2, dim=(1, 2, 3))

    def chi2_flux(self, x, y_flux):
        flux_pred = self.forward_flux(x)
        return self.chi2_flux_from_meas(flux_pred, y_flux)

    @staticmethod
    def chi2_flux_from_meas(y_flux_meas, y_flux):
        return torch.mean(torch.square((y_flux_meas - y_flux) / 2), dim=(1, 2, 3))

    # 6. noisy measurement
    def measure_guassian(self, x):
        vis = self.forward_vis(x)

        amp, sigmaamp = self.forward_amp_from_vis(vis)
        cphase, sigmacp = self.forward_cp_from_vis(vis)
        logcamp, sigmacamp = self.forward_logcamp_from_vis(vis)
        flux = self.forward_flux(x)

        # add isotropic Gaussian noise
        amp = amp + torch.randn_like(amp) * sigmaamp
        cphase = cphase + torch.randn_like(cphase) * sigmacp
        logcamp = logcamp + torch.randn_like(logcamp) * sigmacamp

        return self.compress(amp, sigmaamp, cphase, sigmacp, logcamp, sigmacamp, flux).float()

    def measure_vis_error(self, x):
        vis = self.forward_vis(x)

        # add noise
        sigma = self.sigma[None, None, :, None].repeat(x.shape[0], 1, 1, 1)
        vis = vis + (torch.randn_like(vis) + 1j * torch.randn_like(vis)) * sigma

        # noiseless measurements
        amp, sigmaamp = self.forward_amp_from_vis(vis)
        cphase, sigmacp = self.forward_cp_from_vis(vis)
        logcamp, sigmacamp = self.forward_logcamp_from_vis(vis)
        flux = self.forward_flux(x)

        return self.compress(amp, sigmaamp, cphase, sigmacp, logcamp, sigmacamp, flux).float()

    def measure_eht(self, x):
        ref_im = self.ref_im
        multiplier = self.ref_multiplier
        ref_obs = self.ref_obs
        res = self.H
        pt_obs = []
        # eht_obs = []
        for pt_image in x:
            eh_image = self.pt2ehtim(pt_image, res, ref_im)
            eh_image.ivec = eh_image.ivec * multiplier

            # observe the image
            obs = eh_image.observe_same(ref_obs, phasecal=False, ampcal=False, ttype=self.ttype, verbose=False)
            # eht_obs.append(obs)

            # visibilities amplitude
            adf = ehdf.make_amp(obs, debias=False)
            amp = torch.from_numpy(adf['amp'].to_numpy())[None, None, :, None].float().to(x.device) / multiplier
            sigmaamp = torch.from_numpy(adf['sigma'].to_numpy())[None, None, :, None].float().to(
                x.device) / multiplier

            # closure phase
            cdf = ehdf.make_cphase_df(obs, count='min')
            cp = torch.from_numpy(cdf['cphase'].to_numpy())[None, None, :, None].float().to(x.device) * eh.DEGREE
            sigmacp = torch.from_numpy(cdf['sigmacp'].to_numpy())[None, None, :, None].float().to(
                x.device) * eh.DEGREE

            # log closure amplitude
            ldf = ehdf.make_camp_df(obs, count='min')
            camp = torch.from_numpy(ldf['camp'].to_numpy())[None, None, :, None].float().to(x.device)
            sigmaca = torch.from_numpy(ldf['sigmaca'].to_numpy())[None, None, :, None].float().to(x.device)

            # flux
            flux = torch.tensor([self.estimate_flux(obs)])[None, None, :, None].float().to(x.device) / multiplier

            y = torch.cat([amp, sigmaamp, cp, sigmacp, camp, sigmaca, flux], dim=2)
            pt_obs.append(y)
        pt_obs = torch.cat(pt_obs, dim=0).to(x.device)
        return pt_obs

    # 7. util functions
    def compress(self, amp, sigmaamp, cphase, sigmacp, logcamp, sigmacamp, flux):
        return torch.cat([amp, sigmaamp, cphase, sigmacp, logcamp, sigmacamp, flux], dim=2)

    def decompress(self, y):
        cur = 0
        # visibillity amplitude
        amp = y[:, :, cur:cur + self.amp_dim]
        cur += self.amp_dim
        sigmaamp = y[:, :, cur:cur + self.amp_dim]
        cur += self.amp_dim

        # closure phase
        cphase = y[:, :, cur:cur + self.cphase_dim]
        cur += self.cphase_dim
        sigmacp = y[:, :, cur:cur + self.cphase_dim]
        cur += self.cphase_dim

        # log closure amplitude
        logcamp = y[:, :, cur:cur + self.logcamp_dim]
        cur += self.logcamp_dim
        sigmacamp = y[:, :, cur:cur + self.logcamp_dim]
        cur += self.logcamp_dim

        # flux
        flux = y[:, :, cur:cur + self.flux_dim]
        cur += self.flux_dim
        assert cur == y.shape[2]

        return amp, sigmaamp, cphase, sigmacp, logcamp, sigmacamp, flux

    # 8. sanity check
    def cosine_similarity(self, a1, a2):
        a1 = a1.flatten(1)
        a2 = a2.flatten(1)
        a1_norm = torch.norm(a1, dim=1)
        a2_norm = torch.norm(a2, dim=1)
        similarity = (a1 * a2).sum(1) / (a1_norm * a2_norm)
        return similarity.min().item()

    def compare(self, y1, y2, verbose=False):
        amp1, sigmaamp1, cphase1, sigmacp1, logcamp1, sigmacamp1, flux1 = self.decompress(y1)
        amp2, sigmaamp2, cphase2, sigmacp2, logcamp2, sigmacamp2, flux2 = self.decompress(y2)

        amp_similarity = self.cosine_similarity(amp1, amp2)
        cphase_similarity = self.cosine_similarity(cphase1, cphase2)
        logcamp_similarity = self.cosine_similarity(logcamp1, logcamp2)
        flux_similarity = self.cosine_similarity(flux1, flux2)

        sigmaamp_similarity = self.cosine_similarity(sigmaamp1, sigmaamp2)
        sigmacp_similarity = self.cosine_similarity(sigmacp1, sigmacp2)
        sigmacamp_similarity = self.cosine_similarity(sigmacamp1, sigmacamp2)

        if verbose:
            print("amp similarity: {:.3f} %".format(amp_similarity * 100))
            print("cphase similarity: {:.3f} %".format(cphase_similarity * 100))
            print("logcamp similarity: {:.3f} %".format(logcamp_similarity * 100))
            print("flux similarity: {:.3f} %".format(flux_similarity * 100))
            print("sigmaamp similarity: {:.3f} %".format(sigmaamp_similarity * 100))
            print("sigmacp similarity: {:.3f} %".format(sigmacp_similarity * 100))
            print("sigmacamp similarity: {:.3f} %".format(sigmacamp_similarity * 100))
        similarity = np.max(
            [amp_similarity, cphase_similarity, logcamp_similarity, flux_similarity, sigmaamp_similarity,
             sigmacp_similarity, sigmacamp_similarity])
        return similarity

    def sanity_check(self, x):
        x = self.unnormalize(x)
        # from image
        print('forward by image...')
        y_image = self.forward_from_image(x)

        # from vis
        print('forward by visibility...')
        y_vis = self.forward_from_vis(x)

        # from EHT
        print('forward by EHT...')
        y_eht = self.forward_from_eht(x)

        # compare
        print('compare image and vis (cosine similarity): {:.3f} %'.format(self.compare(y_image, y_vis) * 100))
        print('compare image and EHT (cosine similarity): {:.3f} %'.format(self.compare(y_image, y_eht) * 100))
        print('compare vis and EHT (cosine similarity): {:.3f} %'.format(self.compare(y_vis, y_eht) * 100))

    # 9. evaluating chi-square
    @staticmethod
    def normalize_chisq(chisq):
        overfit = chisq < 1.0
        e_chisq = chisq * (~overfit) + 1 / chisq * overfit
        return e_chisq

    def evaluate_chisq(self, x, y, normalize=True):
        _, _, y_cp, y_cphase_sigma, y_camp, y_logcamp_sigma, y_flux = self.decompress(y)
        # align flux
        x_flux = self.forward_flux(x)
        x_aligned = x * (y_flux / x_flux)

        cp_loss = self.chi2_cphase(x_aligned, y_cp, y_cphase_sigma)
        camp_loss = self.chi2_logcamp(x_aligned, y_camp, y_logcamp_sigma)
        if normalize:
            cp_loss = self.normalize_chisq(cp_loss)
            camp_loss = self.normalize_chisq(camp_loss)
        return cp_loss, camp_loss

    # 10. evaluating blury PSNR
    @staticmethod
    def aligned_images(image1, image2, search_range=(-0.5, 0.5), steps=30):
        # shift search grid
        batch_size, shape = image1.shape[0], image1.shape[1:]
        tx_values = torch.linspace(search_range[0], search_range[1], steps)
        ty_values = torch.linspace(search_range[0], search_range[1], steps)
        tx, ty = torch.meshgrid(tx_values, ty_values)
        tx, ty = tx.flatten(), ty.flatten()

        first_row = torch.stack([torch.ones_like(tx), torch.zeros_like(tx), tx], dim=-1)
        second_row = torch.stack([torch.zeros_like(ty), torch.ones_like(ty), ty], dim=-1)
        theta = torch.stack([first_row, second_row], dim=1)
        grid = F.affine_grid(theta, (tx.shape[0], *shape), align_corners=True)
        grid = grid.unsqueeze(0).repeat(batch_size, 1, 1, 1, 1)

        # shift image2
        N, S = grid.shape[:2]
        flatten_image1 = image1.unsqueeze(1).repeat(1, S, 1, 1, 1).flatten(0, 1).clip(0, 1)
        flatten_image2 = image2.unsqueeze(1).repeat(1, S, 1, 1, 1).flatten(0, 1).clip(0, 1)
        flatten_grid = grid.flatten(0, 1)
        trans_image2 = F.grid_sample(flatten_image2.cpu(), flatten_grid.cpu(), align_corners=True).to(image1.device).clip(0, 1)
        eval_psnr = psnr(flatten_image1, trans_image2, data_range=1.0, reduction='none')
        argmax = eval_psnr.view(N, S).max(dim=1)[1]
        aligned_image2 = trans_image2.view(N, S, *image2.shape[1:])[torch.arange(N), argmax]
        return aligned_image2

    def blur_images(self, samples, factor=15):
        eht_images = self.pt2ehtim_batch(samples, 64, self.ref_im)
        blur_samples = []
        for eht_image in eht_images:
            blur_eht_image = eht_image.blur_circ(factor * eh.RADPERUAS)
            pt_image = torch.from_numpy(blur_eht_image.ivec.reshape(1, 1, 64, 64).astype(np.float32))
            blur_samples.append(pt_image)
        blur_samples = torch.cat(blur_samples).to(samples.device)
        return blur_samples

    @staticmethod
    def aligned_psnr(image1, aligned_image2):
        return psnr(image1.clip(0, 1), aligned_image2.clip(0, 1), data_range=1.0, reduction='none')

    def blur_aligned_psnr(self, image1, aligned_image2, factor=15):
        blur_image1 = self.blur_images(image1, factor).clip(0, 1)
        blur_aligned_image2 = self.blur_images(aligned_image2, factor).clip(0, 1)
        return psnr(blur_image1, blur_aligned_image2, data_range=1.0, reduction='none')

    def evaluate_psnr(self, image1, image2, blur_factors=(0, 10, 15, 20)):
        aligned_image2 = self.aligned_images(image1, image2)
        eval_psnr = self.aligned_psnr(image1, aligned_image2)[:, None]
        for f in blur_factors:
            f_psnr = self.blur_aligned_psnr(image1, aligned_image2, factor=f)[:, None]
            eval_psnr = torch.cat([eval_psnr, f_psnr], dim=1)
        return eval_psnr

    # 11. public interface
    def unnormalize(self, inputs):
        return (inputs + self.unnorm_shift) * self.unnorm_scale

    def normalize(self, inputs):
        return inputs / self.unnorm_scale - self.unnorm_shift

    def forward(self, x, **kwargs):
        x = self.unnormalize(x)
        return self.forward_from_vis(x)

    def __call__(self, 
                 data, 
                 **kwargs):
        x = data['target']
        x = self.unnormalize(x)
        if self.noise_type == 'gaussian':
            return self.measure_guassian(x)
        elif self.noise_type == 'vis_thermal':
            return self.measure_vis_error(x)
        elif self.noise_type == 'eht':
            return self.measure_eht(x)
        else:
            raise ValueError('Unknown noise type')

    def loss(self, x, y):
        x = self.unnormalize(x)
        y_amp, y_amp_sigma, y_cp, y_cphase_sigma, y_camp, y_logcamp_sigma, y_flux = self.decompress(y)
        amp_loss = self.chi2_amp(x, y_amp, y_amp_sigma)
        cp_loss = self.chi2_cphase(x, y_cp, y_cphase_sigma)
        camp_loss = self.chi2_logcamp(x, y_camp, y_logcamp_sigma)
        flux_loss = self.chi2_flux(x, y_flux)

        data_fit = self.weight_amp * amp_loss + self.weight_cp * cp_loss + self.weight_camp * camp_loss + self.weight_flux * flux_loss
        return data_fit * 2

    def loss_m(self, yx, y):
        yx_amp, yx_amp_sigma, yx_cp, yx_cphase_sigma, yx_camp, yx_logcamp_sigma, yx_flux = self.decompress(yx)
        y_amp, y_amp_sigma, y_cp, y_cphase_sigma, y_camp, y_logcamp_sigma, y_flux = self.decompress(y)
        amp_loss = self.chi2_amp_from_meas(yx_amp, y_amp, y_amp_sigma)
        cp_loss = self.chi2_cphase_from_meas(yx_cp, y_cp, y_cphase_sigma)
        camp_loss = self.chi2_logcamp_from_meas(yx_camp, y_camp, y_logcamp_sigma)
        flux_loss = self.chi2_flux_from_meas(yx_flux, y_flux)

        data_fit = self.weight_amp * amp_loss + self.weight_cp * cp_loss + self.weight_camp * camp_loss + self.weight_flux * flux_loss
        return data_fit * 2



