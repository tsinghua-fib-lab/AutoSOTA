import random

import cv2
import numpy as np
from albumentations import brightness_contrast_adjust, random_utils
from albumentations.augmentations import _maybe_process_in_chunks

from scipy import special

class DegradationGeneration:
    def __init__(self, noise_params, blur_params, lc_params):
        self.noise_params = noise_params
        self.blur_params = blur_params
        self.lc_params = lc_params

    @staticmethod
    def stripe_noise(image, **params):
        sg = params['sg']
        sb = params['sb']

        if params['dic'] == 1:
            g = np.random.randn(1, image.shape[1]) * sg
            b = np.random.randn(1, image.shape[1]) * sb
        else:
            g = np.random.randn(image.shape[0], 1) * sg
            b = np.random.randn(image.shape[0], 1) * sb
        if len(image.shape) == 3:
            g = np.expand_dims(g, -1)
            b = np.expand_dims(b, -1)
        noise = image * g + b
        image = np.clip(image.astype("float32") + noise.astype("float32"), 0, 255).astype("uint8")
        return image

    @staticmethod
    def nonuniformity_optical(image, **params):
        h, w = image.shape
        odelta = params['odelta']
        noise = np.ones((h, w)).astype("float32")
        idx_h = np.expand_dims(np.arange(1, h + 1), 1)
        idx_w = np.expand_dims(np.arange(1, w + 1), 0)
        delta = odelta#np.random.randint(15, 75 + 1)
        ch = np.random.randint(h)
        cw = np.random.randint(w)

        p = (np.abs(idx_h - ch) ** 2 + np.abs(idx_w - cw) ** 2) ** 0.5
        p /= np.max(p)
        noise *= p
        noise = np.cos(noise * np.pi / 2) ** 4
        # noise = noise / np.max(noise)
        # noise = np.square(noise)
        if len(image.shape) == 3:
            noise = np.expand_dims(noise, -1)
        if random.random() < 0.5:
            image = np.clip(image.astype("float32") + noise.astype("float32") * delta, 0, 255).astype("uint8")
        else:
            image = np.clip(image.astype("float32") + (1 - noise.astype("float32")) * delta, 0, 255).astype("uint8")
        return image

    @staticmethod
    def nonuniformity_optical_prime(image, **params):
        intensity = params['odelta']
        h, w = image.shape[:2]
        W_pos = params['W_pos']#np.random.randint(w // 4, 3 * w // 4)  # w // 2
        H_pos = params['H_pos']#np.random.randint(h // 4, 3 * h // 4)  # h // 2
        cy_out = np.array([H_pos, -3 * H_pos, 5 * H_pos, H_pos])
        cx_out = np.array([-3 * W_pos, W_pos, W_pos, 5 * W_pos])
        idx_h = np.expand_dims(np.arange(1, h + 1), 1)
        idx_w = np.expand_dims(np.arange(1, w + 1), 0)
        dist1 = np.maximum(((idx_h - H_pos) ** 2 + (idx_w - W_pos) ** 2) ** 0.5, 10)
        dist2 = np.min(((idx_h[:, :, None] - cy_out[np.newaxis, np.newaxis, :]) ** 2 + (
                    idx_w[:, :, None] - cx_out[np.newaxis, np.newaxis, :]) ** 2) ** 0.5, axis=-1)
        noise = -np.log(dist2 / dist1)
        noise = ((noise - noise.min()) / (noise.max() - noise.min())) ** 4
        if params['o_cond'] == 1:
            noise = 1 - noise
        noise = noise * intensity - intensity / 2
        image = np.clip(image.astype("float32") + noise.astype("float32"), 0, 255).astype("uint8")
        return image

    @staticmethod
    def generate_nonuniformity_optical_noise(image, **params):
        h, w = image.shape[:2]
        W_pos = params['W_pos']  # np.random.randint(w // 4, 3 * w // 4)  # w // 2
        H_pos = params['H_pos']  # np.random.randint(h // 4, 3 * h // 4)  # h // 2
        cy_out = np.array([H_pos, -3 * H_pos, 5 * H_pos, H_pos])
        cx_out = np.array([-3 * W_pos, W_pos, W_pos, 5 * W_pos])
        idx_h = np.expand_dims(np.arange(1, h + 1), 1)
        idx_w = np.expand_dims(np.arange(1, w + 1), 0)
        dist1 = np.maximum(((idx_h - H_pos) ** 2 + (idx_w - W_pos) ** 2) ** 0.5, 10)
        dist2 = np.min(((idx_h[:, :, None] - cy_out[np.newaxis, np.newaxis, :]) ** 2 + (
                idx_w[:, :, None] - cx_out[np.newaxis, np.newaxis, :]) ** 2) ** 0.5, axis=-1)
        noise = -np.log(dist2 / dist1)
        noise = ((noise - noise.min()) / (noise.max() - noise.min())) ** 4
        return noise

    def nonuniformity_optical_noise_prime(self, image, **params):
        intensity = params['odelta']
        noise = params['o_noise']
        if params['o_cond'] == 1:
            noise = 1 - noise
        noise = noise * intensity - intensity / 2
        w, h = image.shape[:2]
        nw, nh = noise.shape[:2]
        if w != nw or h !=nh:
            noise = cv2.resize(noise, (w, h), interpolation=cv2.INTER_LINEAR)
        image = np.clip(image.astype("float32") + noise.astype("float32"), 0, 255).astype("uint8")
        return image

    @staticmethod
    def gaussian_noise(image, **params):
        var = params["var"]
        sigma = var ** 0.5

        if len(image.shape) == 3:
            noise = random_utils.normal(0, sigma, image.shape[:2])
            noise = np.expand_dims(noise, -1)
        else:
            noise = random_utils.normal(0, sigma, image.shape)
        return noise

    @staticmethod
    def poisson_noise(img, **params):
        scale = params["scale"]
        img = img / 255.
        vals = len(np.unique(img))
        vals = 2 ** np.ceil(np.log2(vals))
        out = np.float32(np.random.poisson(img * vals) / float(vals))
        noise = out - img
        return noise * 255. * scale

    @staticmethod
    def random_noise(image, **params):
        var = params["var"]
        sigma = var ** 0.5

        if len(image.shape) == 3:
            gauss = random_utils.normal(0, sigma, image.shape[:2])
            gauss = np.expand_dims(gauss, -1)
        else:
            gauss = random_utils.normal(0, sigma, image.shape)
        image = np.clip(image.astype("float32") + gauss, 0, 255).astype("uint8")
        return image


    def random_noise_prime(self, image, **params):
        noise_cond = params["noise_cond"]
        if noise_cond == 0:
            noise = self.gaussian_noise(image, **params)
        else:
            noise = self.poisson_noise(image, **params)
        image = np.clip(image.astype("float32") + noise, 0, 255).astype("uint8")
        return image

    @staticmethod
    def gaussian_blur(image, **params):
        ksize = params["ksize"]
        sigma = params["sigma"]
        image = _maybe_process_in_chunks(cv2.GaussianBlur, ksize=(ksize, ksize), sigmaX=sigma)(image)
        return image

    @staticmethod
    def gaussian_blur_prime(image, **params):

        ksize = params["ksize"]
        sigma_x = params["sigma_x"]
        sigma_y = params["sigma_y"]
        angle = np.deg2rad(params["rotate"])

        # Generate mesh grid centered at zero.
        ax = np.arange(-ksize // 2 + 1.0, ksize // 2 + 1.0)
        # > Shape (ksize, ksize, 2)
        grid = np.stack(np.meshgrid(ax, ax), axis=-1)

        # Calculate rotated sigma matrix
        d_matrix = np.array([[sigma_x ** 2, 0], [0, sigma_y ** 2]])
        u_matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        sigma_matrix = np.dot(u_matrix, np.dot(d_matrix, u_matrix.T))

        inverse_sigma = np.linalg.inv(sigma_matrix)
        kernel = np.exp(-0.5 * np.sum(np.dot(grid, inverse_sigma) * grid, 2))

        # Normalize kernel
        kernel = kernel.astype(np.float32) / np.sum(kernel)
        conv_fn = _maybe_process_in_chunks(cv2.filter2D, ddepth=-1, kernel=kernel)
        return conv_fn(image)

    @ staticmethod
    def low_contrast(image, **params):
        alpha = params["alpha"]
        beta = params["beta"]
        brightness_by_max = params["brightness_by_max"]
        image = brightness_contrast_adjust(image, alpha, beta, brightness_by_max)
        return image

    def Noise(self, image):
        sg_high = self.noise_params['sg_high']
        sg_low = self.noise_params['sg_low']

        sb_high = self.noise_params['sb_high']
        sb_low = self.noise_params['sb_low']

        o_high = self.noise_params['o_high']
        o_low = self.noise_params['o_low']

        var_high = self.noise_params['var_high']
        var_low = self.noise_params['var_low']
        sg = np.random.rand() * (sg_high - sg_low) + sg_low
        sb = np.random.rand() * (sb_high - sb_low) + sb_low
        o = np.random.randint(o_low, o_high + 1)
        var = random.uniform(var_low, var_high)
        params = {'sg': sg, 'sb': sb, 'odelta': o, 'var': var}
        # print(params)
        image = self.nonuniformity_optical(image, **params)
        image = self.stripe_noise(image, **params)
        image = self.random_noise(image, **params)
        return image


    def Noise_prime(self, image, params=None):
        if params is None:
            sg_high = self.noise_params['sg_high']
            sg_low = self.noise_params['sg_low']

            sb_high = self.noise_params['sb_high']
            sb_low = self.noise_params['sb_low']

            dic = 1 if random.random() < 0.5 else 0

            o_high = self.noise_params['o_high']
            o_low = self.noise_params['o_low']

            sg = np.random.rand() * (sg_high - sg_low) + sg_low
            sb = np.random.rand() * (sb_high - sb_low) + sb_low
            o = np.random.randint(o_low, o_high + 1)
            h, w = image.shape[:2]
            H_pos = np.random.randint(h // 4, 3 * h // 4)
            W_pos = np.random.randint(w // 4, 3 * w // 4)

            o_noise = self.generate_nonuniformity_optical_noise(image, **{'H_pos': H_pos, 'W_pos': W_pos})

            o_cond = 1 if random.random() < 0.5 else 0

            noise_cond = 0 if random.random() < self.noise_params['noise_prob'] else 1
            if noise_cond == 0:
                var_high = self.noise_params['var_high']
                var_low = self.noise_params['var_low']
                var = random.uniform(var_low, var_high)
                scale = None
            else:
                scale_high = self.noise_params['scale_high']
                scale_low = self.noise_params['scale_low']
                scale = random.uniform(scale_low, scale_high)
                var = None
            params = {'sg': sg, 'sb': sb, 'dic': dic, 'odelta': o, 'o_noise': o_noise, 'o_cond': o_cond, 'var': var, 'scale': scale, 'noise_cond': noise_cond}
        # print(params)
        image = self.nonuniformity_optical_noise_prime(image, **params)
        image = self.stripe_noise(image, **params)
        image = self.random_noise_prime(image, **params)
        return image
    
    def Noise_prime_params(self, image):
        sg_high = self.noise_params['sg_high']
        sg_low = self.noise_params['sg_low']

        sb_high = self.noise_params['sb_high']
        sb_low = self.noise_params['sb_low']

        dic = 1 if random.random() < 0.5 else 0

        o_high = self.noise_params['o_high']
        o_low = self.noise_params['o_low']

        sg = np.random.rand() * (sg_high - sg_low) + sg_low
        sb = np.random.rand() * (sb_high - sb_low) + sb_low
        o = np.random.randint(o_low, o_high + 1)
        h, w = image.shape[:2]
        H_pos = np.random.randint(h // 4, 3 * h // 4)
        W_pos = np.random.randint(w // 4, 3 * w // 4)

        o_noise = self.generate_nonuniformity_optical_noise(image, **{'H_pos': H_pos, 'W_pos': W_pos})

        o_cond = 1 if random.random() < 0.5 else 0

        noise_cond = 0 if random.random() < self.noise_params['noise_prob'] else 1
        if noise_cond == 0:
            var_high = self.noise_params['var_high']
            var_low = self.noise_params['var_low']
            var = random.uniform(var_low, var_high)
            scale = None
        else:
            scale_high = self.noise_params['scale_high']
            scale_low = self.noise_params['scale_low']
            scale = random.uniform(scale_low, scale_high)
            var = None
        params = {'sg': sg, 'sb': sb, 'dic': dic, 'odelta': o, 'o_noise': o_noise, 'o_cond': o_cond, 'var': var, 'scale': scale, 'noise_cond': noise_cond}
        return params

    def Blur(self, image):
        ksize_high = self.blur_params['ksize_high']
        ksize_low = self.blur_params['ksize_low']

        sigma_high = self.blur_params['sigma_high']
        sigma_low = self.blur_params['sigma_low']

        ksize = random.randrange(ksize_low, ksize_high + 1, 2)
        sigma = random.uniform(sigma_low, sigma_high)
        params = {"ksize": ksize, "sigma": sigma}
        # print(params)
        return self.gaussian_blur(image, **params)


    def Blur_prime(self, image):
        ksize_high = self.blur_params['ksize_high']
        ksize_low = self.blur_params['ksize_low']

        sigma_high = self.blur_params['sigma_high']
        sigma_low = self.blur_params['sigma_low']

        rotate_limit = [-90, 90]

        ksize = random.randrange(ksize_low, ksize_high + 1, 2)
        sigma_x = random.uniform(sigma_low, sigma_high)
        sigma_y = random.uniform(sigma_low, sigma_high)
        rotate = random.uniform(*rotate_limit)
        params = {"ksize": ksize, "sigma_x": sigma_x, "sigma_y": sigma_y, "rotate": rotate}
        return self.gaussian_blur_prime(image, **params)
    
    @staticmethod
    def downscale(img, **params):
        scale = params["scale"]
        interpolation = params["interpolation"]
        img = (img / 255.).astype(np.float32)
        if scale != 1:
            img = cv2.resize(img, None, fx=1/scale, fy=1/scale, interpolation=interpolation)
        return (np.clip(img, 0, 1) * 255).astype("uint8")
    
    def DownSample(self, image):
        scale = np.random.choice(self.blur_params['scale_choice'], 1)[0]
        interpolation = np.random.choice(self.blur_params['interpolation_choice'], 1)[0]

        params = {'scale': scale, 'interpolation': interpolation}

        self.downscale(image, **params)
        return image
        
    def Blur_DS_prime(self, image, params=None):
        if params is None:
            ksize_high = self.blur_params['ksize_high']
            ksize_low = self.blur_params['ksize_low']

            sigma_high = self.blur_params['sigma_high']
            sigma_low = self.blur_params['sigma_low']

            scale = np.random.choice(self.blur_params['scale_choice'], 1)[0]
            interpolation = np.random.choice(self.blur_params['interpolation_choice'], 1)[0]

            rotate_limit = [-90, 90]

            ksize = random.randrange(ksize_low, ksize_high + 1, 2)
            sigma_x = random.uniform(sigma_low, sigma_high)
            sigma_y = random.uniform(sigma_low, sigma_high)
            rotate = random.uniform(*rotate_limit)
            params = {"ksize": ksize, "sigma_x": sigma_x, "sigma_y": sigma_y, "rotate": rotate, 'scale': scale, 'interpolation': interpolation}
        image = self.gaussian_blur_prime(image, **params)
        image = self.downscale(image, **params)
        return image
    
    def Blur_DS_prime_params(self, image):
        ksize_high = self.blur_params['ksize_high']
        ksize_low = self.blur_params['ksize_low']

        sigma_high = self.blur_params['sigma_high']
        sigma_low = self.blur_params['sigma_low']

        scale = np.random.choice(self.blur_params['scale_choice'], 1)[0]
        interpolation = np.random.choice(self.blur_params['interpolation_choice'], 1)[0]

        rotate_limit = [-90, 90]

        ksize = random.randrange(ksize_low, ksize_high + 1, 2)
        sigma_x = random.uniform(sigma_low, sigma_high)
        sigma_y = random.uniform(sigma_low, sigma_high)
        rotate = random.uniform(*rotate_limit)
        params = {"ksize": ksize, "sigma_x": sigma_x, "sigma_y": sigma_y, "rotate": rotate, 'scale': scale, 'interpolation': interpolation}
        return params

    def LC(self, image, params=None):
        if params is None:
            alpha_high = self.lc_params['alpha_high']
            alpha_low = self.lc_params['alpha_low']

            beta_high = self.lc_params['beta_high']
            beta_low = self.lc_params['beta_low']

            brightness_by_max = self.lc_params['brightness_by_max']

            alpha = random.uniform(alpha_low, alpha_high)
            beta = random.uniform(beta_low, beta_high)
            params = {"alpha": alpha, "beta": beta, "brightness_by_max": brightness_by_max}
        # print(params)
        return self.low_contrast(image, **params)
    
    def LC_params(self, image):
        alpha_high = self.lc_params['alpha_high']
        alpha_low = self.lc_params['alpha_low']

        beta_high = self.lc_params['beta_high']
        beta_low = self.lc_params['beta_low']

        brightness_by_max = self.lc_params['brightness_by_max']

        alpha = random.uniform(alpha_low, alpha_high)
        beta = random.uniform(beta_low, beta_high)
        params = {"alpha": alpha, "beta": beta, "brightness_by_max": brightness_by_max}
        # print(params)
        return params
    
    
    def comp1_all(self, image):
        height, width = image.shape[:2]
        
        if random.random() < 0.7:
            image = self.LC(image)
        if random.random() < 0.7:
            image = self.Blur_DS_prime(image)
        if random.random() < 0.7:
            image = self.Noise_prime(image)
        
        # image = self.Noise_prime(self.Blur_DS_prime(self.LC(image)))
        nh, hw = image.shape[:2]
        if nh != height or hw != width:
            image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
        return image
    
    
    def get_comp1_all_params(self, image):
        return {'noise_params': self.Noise_prime_params(image), 'blur_params': self.Blur_DS_prime_params(image), 'lc_params': self.LC_params(image), 'act_noise_prob': random.random(),'act_blur_prob': random.random(),'act_lc_prob': random.random()}
    
    
    def comp1_all_params(self, image, **params):
        lc_params = params['lc_params']
        blur_params = params['blur_params']
        noise_params = params['noise_params']

        lc_pros = params['act_lc_prob']
        blur_pros = params['act_blur_prob']
        noise_pros = params['act_noise_prob']
        
        height, width = image.shape[:2]
        if lc_pros < 0.7:
            image = self.LC(image, lc_params)
        if blur_pros < 0.7:
            image = self.Blur_DS_prime(image, blur_params)
        if noise_pros < 0.7:
            image = self.Noise_prime(image, noise_params)
            
        # image = self.Noise_prime(self.Blur_DS_prime(self.LC(image, lc_params), blur_params), noise_params)
        nh, hw = image.shape[:2]
        if nh != height or hw != width:
            image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
        return image
    

