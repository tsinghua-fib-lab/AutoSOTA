# Copyright (c) 2025 Harbin Institute of Technology, Huawei Noah's Ark Lab
# SPDX-License-Identifier: CC-BY-NC-SA-4.0
import math
import torch
from ..arguments import OptimizationParams, PipelineParams, DensifyParams


class DashGaussianScheduler():
    """
    DashGaussian training scheduler of resolution and primitive number.
    """
    def __init__(self, pipe: PipelineParams, densify: DensifyParams, init_n_gaussian: int, original_images: list, iterations_per_epoch: int) -> None:

        self.init_n_gaussian = init_n_gaussian

        self.densify_from_iter = densify.densify_from * iterations_per_epoch
        self.densify_until_iter = densify.densify_until * iterations_per_epoch
        self.densification_interval = densify.densification_interval * iterations_per_epoch

        self.start_significance_factor = 4
        self.max_reso_scale = 8
        self.reso_sample_num = 32 # Must be no less than 2
        self.max_densify_rate_per_step = 0.33
        print(f'max_densify_rate_per_step: {self.max_densify_rate_per_step}')
        self.reso_scales = None
        self.reso_level_significance = None
        self.reso_level_begin = None
        self.next_i = 2

        self.max_n_gaussian = pipe.max_n_gaussian

        # Calculate allowed render scales
        self.allowed_render_scales = self._calculate_allowed_render_scales(pipe.tile_size, original_images[0].shape, pipe.max_gaussians_per_tile_at_beginning)

        # Generate schedulers.
        self.init_reso_scheduler(original_images)

    def get_densify_rate(self, iteration, cur_n_gaussian, cur_scale):
        if self.densification_interval + iteration < self.densify_until_iter:
            next_n_gaussian = int((self.max_n_gaussian - self.init_n_gaussian) / cur_scale**(2 - iteration / self.densify_until_iter)) + self.init_n_gaussian
        else:
            next_n_gaussian = self.max_n_gaussian
        return min(max((next_n_gaussian - cur_n_gaussian) / cur_n_gaussian, 0.), self.max_densify_rate_per_step)

    def _calculate_allowed_render_scales(self, tile_size, image_shape, max_gaussians_per_tile_at_beginning):
        """Calculate allowed render scales based on image dimensions."""

        def _calculate_max_downsample_scale(gaussian_count, tile_count, max_gaussians_per_tile_at_beginning):
            """Calculate maximum downsample scale for GT images based on gaussian count, tile count, and max gaussians per tile at beginning.

            At the very beginning the gaussian scale is large. If gt image is too small, the gaussian list length per tile will be too long.
            So we prevent from downsampling gt image too much.

            Mathematical derivation:
            We want: G / (T/x^2) <= max_gaussians_per_tile_at_beginning
            Where:
            - G = gaussian count at the beginning
            - T = original tile count  
            - x = downsample scale per side
            - T/x^2 = tile count after downsampling by x per side
            - G / (T/x^2) = gaussians per tile after downsampling
            
            Solving for x:
            G / (T/x^2) <= max_gaussians_per_tile_at_beginning
            G * x^2 / T <= max_gaussians_per_tile_at_beginning
            x^2 <= max_gaussians_per_tile_at_beginning * T / G
            x <= sqrt(max_gaussians_per_tile_at_beginning * T / G)
            
            Args:
                gaussian_count (int): Number of gaussians (G)
                tile_count (int): Number of tiles from original GT image (T)  
                max_gaussians_per_tile_at_beginning (int): Maximum allowed gaussians per tile at the beginning
                
            Returns:
                float: sqrt(max_gaussians_per_tile_at_beginning * T / G) - maximum scale that GT can be downsampled
            """
            return math.sqrt(max_gaussians_per_tile_at_beginning * tile_count / gaussian_count)


        # Calculate initial dropout scale using image dimensions
        tiles_x = math.ceil(image_shape[2] / float(tile_size))
        tiles_y = math.ceil(image_shape[1] / float(tile_size))
        initial_tile_count = tiles_x * tiles_y
        max_render_scale = _calculate_max_downsample_scale(self.init_n_gaussian, initial_tile_count, max_gaussians_per_tile_at_beginning)
        print(f'max_render_scale: {max_render_scale:.2f}')
        
        # Set allowed_render_scales based on max_render_scale
        # Meaning: [sqrt(32), 4, sqrt(8), 2, sqrt(2), 1] -> [5.7, 4, 2.8, 2, 1.4, 1]
        if max_render_scale >= 4:
            return [4, 2.8, 2, 1.4, 1]
        elif max_render_scale >= 2.8:
            return [2.8, 2, 1.4, 1]
        elif max_render_scale >= 2:
            return [2, 1.4, 1]
        else:
            return [1.4, 1]

        # if max_render_scale >= 4:
        #     return [4, 3, 2, 1]
        # elif max_render_scale >= 3:
        #     return [3, 2, 1]
        # else:
        #     return [2, 1]

    def _get_res_scale(self, iteration):
        if iteration >= self.densify_until_iter:
            return 1
        if iteration < self.reso_level_begin[1]:
            return self.reso_scales[0]
        while iteration >= self.reso_level_begin[self.next_i]:
            # If the index is out of the range of 'reso_level_begin', there must be something wrong with the scheduler.
            self.next_i += 1
        i = self.next_i - 1
        i_now, i_nxt = self.reso_level_begin[i: i + 2]
        s_lst, s_now = self.reso_scales[i - 1: i + 1]
        scale = (1 / ((iteration - i_now) / (i_nxt - i_now) * (1/s_now**2 - 1/s_lst**2) + 1/s_lst**2))**0.5
        return scale

    def _massage_render_scale1(self, scale):
        """Find the closest allowed render scale to the given scale."""
        return min(self.allowed_render_scales, key=lambda x: abs(x - scale))

    def _massage_render_scale2(self, scale):
        """Find the largest allowed render scale that's smaller than the given scale."""
        smaller_scales = [x for x in self.allowed_render_scales if x <= scale]
        return max(smaller_scales) if smaller_scales else min(self.allowed_render_scales)

    def get_res_scale(self, iteration):
        """Get the massaged render scale for the given iteration."""
        raw_scale = self._get_res_scale(iteration)
        return self._massage_render_scale2(raw_scale)

    def init_reso_scheduler(self, original_images):

        def compute_win_significance(significance_map: torch.Tensor, scale: float):
            h, w = significance_map.shape[-2:]
            c = ((h + 1) // 2, (w + 1) // 2)
            win_size = (int(h / scale), int(w / scale))
            win_significance = significance_map[..., c[0]-win_size[0]//2: c[0]+win_size[0]//2, c[1]-win_size[1]//2: c[1]+win_size[1]//2].sum().item()
            return win_significance

        def scale_solver(significance_map: torch.Tensor, target_significance: float):
            L, R, T = 0., 1., 64
            for _ in range(T):
                mid = (L + R) / 2
                win_significance = compute_win_significance(significance_map, 1 / mid)
                if win_significance < target_significance:
                    L = mid
                else:
                    R = mid
            return 1 / mid

        # print("[ INFO ] Initializing resolution scheduler...")

        scene_freq_image = None

        for img in original_images:
            img_fft_centered = torch.fft.fftshift(torch.fft.fft2(img), dim=(-2, -1))
            img_fft_centered_mod = (img_fft_centered.real.square() + img_fft_centered.imag.square()).sqrt()
            scene_freq_image = img_fft_centered_mod if scene_freq_image is None else scene_freq_image + img_fft_centered_mod

            e_total = img_fft_centered_mod.sum().item()
            e_min = e_total / self.start_significance_factor
            self.max_reso_scale = min(self.max_reso_scale, scale_solver(img_fft_centered_mod, e_min))

        modulation_func = math.log

        self.reso_scales = []
        self.reso_level_significance = []
        self.reso_level_begin = []
        scene_freq_image /= len(original_images)
        E_total = scene_freq_image.sum().item()
        E_min = compute_win_significance(scene_freq_image, self.max_reso_scale)
        self.reso_level_significance.append(E_min)
        self.reso_scales.append(self.max_reso_scale)
        self.reso_level_begin.append(0)
        for i in range(1, self.reso_sample_num - 1):
            self.reso_level_significance.append((E_total - E_min) * (i - 0) / (self.reso_sample_num-1 - 0) + E_min)
            self.reso_scales.append(scale_solver(scene_freq_image, self.reso_level_significance[-1]))
            self.reso_level_significance[-2] = modulation_func(self.reso_level_significance[-2] / E_min)
            self.reso_level_begin.append(int(self.densify_until_iter * self.reso_level_significance[-2] / modulation_func(E_total / E_min)))
        self.reso_level_significance.append(modulation_func(E_total / E_min))
        self.reso_scales.append(1.)
        self.reso_level_significance[-2] = modulation_func(self.reso_level_significance[-2] / E_min)
        self.reso_level_begin.append(int(self.densify_until_iter * self.reso_level_significance[-2] / modulation_func(E_total / E_min)))
        self.reso_level_begin.append(self.densify_until_iter)

        # print("================== Resolution Scheduler ==================")
        # for idx, (e, s, i) in enumerate(zip(self.reso_level_significance, self.reso_scales, self.reso_level_begin)):
        #     print(" - idx: {:02d}; scale: {:.2f}; significance: {:.2f}; begin: {}".format(idx, s, e, i))
        # print("==========================================================")
