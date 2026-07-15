from pathlib import Path

import torch
from models.dsd.dsd import DSD
from models.dsd.stylegan2 import Generator
from models.losses.dsd_loss import LossBuilderStyleGAN2


class DSDStyleGAN2(DSD):
    def __init__(self, opt, cache_dir):
        super(DSDStyleGAN2, self).__init__(opt, cache_dir)

    def load_synthesis_network(self):
        self.synthesis = Generator(size=256, style_dim=512, n_mlp=8).cuda()
        self.synthesis.load_state_dict(torch.load("experiments/pretrained/stylegan2.pt")["g_ema"], strict=False)
        for v in self.synthesis.parameters():
            v.requires_grad = False

    def initialize_mapping_network(self):
        if Path("experiments/pretrained/gaussian_fit_stylegan2.pt").exists():
            self.gaussian_fit = torch.load("experiments/pretrained/gaussian_fit_stylegan2.pt")
        else:
            if self.verbose:
                print("\tRunning Mapping Network")
            with torch.no_grad():
                torch.manual_seed(0)
                latent = torch.randn((1000000, 512), dtype=torch.float32, device="cuda")
                latent_out = torch.nn.LeakyReLU(5)(self.synthesis.get_latent(latent))
                self.gaussian_fit = {"mean": latent_out.mean(0), "std": latent_out.std(0)}
                torch.save(self.gaussian_fit, "experiments/pretrained/gaussian_fit_stylegan2.pt")
                if self.verbose:
                    print('\tSaved "gaussian_fit_stylegan2.pt"')

    def initialize_latent_space(self):
        batch_size = self.opt["batch_size"]

        # Generate latent tensor
        if self.opt["tile_latent"]:
            self.latent = torch.randn((batch_size, 1, 512), dtype=torch.float, requires_grad=True, device="cuda")
        else:
            self.latent = torch.randn((batch_size, 14, 512), dtype=torch.float, requires_grad=True, device="cuda")

        # Generate list of noise tensors
        noise = []  # stores all of the noise tensors
        noise_vars = []  # stores the noise tensors that we want to optimize on

        for i in range(14):
            res = (i + 5) // 2
            res = [1, 1, 2 ** res, 2 ** res]

            noise_type = self.opt["noise_type"]
            bad_noise_layers = self.opt["bad_noise_layers"]
            if noise_type == "zero" or i in [int(layer) for layer in bad_noise_layers.split(".")]:
                new_noise = torch.zeros(res, dtype=torch.float, device="cuda")
                new_noise.requires_grad = False
            elif noise_type == "fixed":
                new_noise = torch.randn(res, dtype=torch.float, device="cuda")
                new_noise.requires_grad = False
            elif noise_type == "trainable":
                new_noise = torch.randn(res, dtype=torch.float, device="cuda")
                if i < self.opt["num_trainable_noise_layers"]:
                    new_noise.requires_grad = True
                    noise_vars.append(new_noise)
                else:
                    new_noise.requires_grad = False
            else:
                raise Exception("unknown noise type")

            noise.append(new_noise)

        self.latent_x_var_list = [self.latent] + noise_vars
        self.noise = noise

    def initialize_loss(self, ref_im):
        self.loss_builder = LossBuilderStyleGAN2(ref_im, self.opt).cuda()

    def get_gen_im(self, latent_in):
        return (self.synthesis([latent_in], input_is_latent=True, noise=self.noise)[0] + 1) / 2
