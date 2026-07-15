from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import utils.util as util
from models.dips import KernelDIP
from models.dsd.spherical_optimizer import SphericalOptimizer
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm


class DSD(torch.nn.Module):
    def __init__(self, opt, cache_dir):
        super(DSD, self).__init__()

        self.opt = opt

        self.verbose = opt["verbose"]
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Initialize synthesis network
        if self.verbose:
            print("Loading Synthesis Network")
        self.load_synthesis_network()
        if self.verbose:
            print("Synthesis Network loaded!")

        self.lrelu = torch.nn.LeakyReLU(negative_slope=0.2)

        self.initialize_mapping_network()

    def initialize_dip(self):
        self.dip_zk = util.get_noise(64, "noise", (64, 64)).cuda().detach()
        self.k_dip = KernelDIP(self.opt["KernelDIP"]).cuda()

    def initialize_latent_space(self):
        pass

    def initialize_optimizers(self):
        # Optimizer for k
        self.optimizer_k = torch.optim.Adam(self.k_dip.parameters(), lr=self.opt["k_lr"])
        self.scheduler_k = StepLR(
            self.optimizer_k, step_size=self.opt["num_epochs"] * self.opt["num_k_iters"] // 5, gamma=0.7
        )

        # Optimizer for x
        optimizer_dict = {
            "sgd": torch.optim.SGD,
            "adam": torch.optim.Adam,
            "sgdm": partial(torch.optim.SGD, momentum=0.9),
            "adamax": torch.optim.Adamax,
        }
        optimizer_func = optimizer_dict[self.opt["optimizer_name"]]
        self.optimizer_x = SphericalOptimizer(optimizer_func, self.latent_x_var_list, lr=self.opt["x_lr"])

        steps = self.opt["num_epochs"] * self.opt["num_x_iters"]
        schedule_dict = {
            "fixed": lambda x: 1,
            "linear1cycle": lambda x: (9 * (1 - np.abs(x / steps - 1 / 2) * 2) + 1) / 10,
            "linear1cycledrop": lambda x: (9 * (1 - np.abs(x / (0.9 * steps) - 1 / 2) * 2) + 1) / 10
            if x < 0.9 * steps
            else 1 / 10 + (x - 0.9 * steps) / (0.1 * steps) * (1 / 1000 - 1 / 10),
        }
        schedule_func = schedule_dict[self.opt["lr_schedule"]]
        self.scheduler_x = torch.optim.lr_scheduler.LambdaLR(self.optimizer_x.opt, schedule_func)

    def warmup_dip(self):
        self.reg_noise_std = self.opt["reg_noise_std"]
        warmup_k = torch.load("experiments/pretrained/kernel.pth")

        mse = nn.MSELoss().cuda()

        print("Warming up k DIP")
        for step in tqdm(range(self.opt["num_warmup_iters"])):
            self.optimizer_k.zero_grad()
            dip_zk_rand = self.dip_zk + self.reg_noise_std * torch.randn_like(self.dip_zk).cuda()
            k = self.k_dip(dip_zk_rand)

            loss = mse(k, warmup_k)
            loss.backward()
            self.optimizer_k.step()

    def optimize_k_step(self, epoch):
        # Optimize k
        tq_k = tqdm(range(self.opt["num_k_iters"]))
        for j in tq_k:
            for p in self.k_dip.parameters():
                p.requires_grad = True
            for p in self.latent_x_var_list:
                p.requires_grad = False

            self.optimizer_k.zero_grad()

            # Duplicate latent in case tile_latent = True
            if self.opt["tile_latent"]:
                latent_in = self.latent.expand(-1, 14, -1)
            else:
                latent_in = self.latent

            dip_zk_rand = self.dip_zk + self.reg_noise_std * torch.randn_like(self.dip_zk).cuda()
            # Apply learned linear mapping to match latent distribution to that of the mapping network
            latent_in = self.lrelu(latent_in * self.gaussian_fit["std"] + self.gaussian_fit["mean"])

            # Normalize image to [0,1] instead of [-1,1]
            self.gen_im = self.get_gen_im(latent_in)
            self.gen_ker = self.k_dip(dip_zk_rand)

            # Calculate Losses
            loss, loss_dict = self.loss_builder(latent_in, self.gen_im, self.gen_ker, epoch)
            self.cur_loss = loss.cpu().detach().numpy()

            loss.backward()
            self.optimizer_k.step()
            self.scheduler_k.step()

            msg = " | ".join("{}: {:.4f}".format(k, v) for k, v in loss_dict.items())
            tq_k.set_postfix(loss=msg)

    def optimize_x_step(self, epoch):
        tq_x = tqdm(range(self.opt["num_x_iters"]))
        for j in tq_x:
            for p in self.k_dip.parameters():
                p.requires_grad = False
            for p in self.latent_x_var_list:
                p.requires_grad = True

            self.optimizer_x.opt.zero_grad()

            # Duplicate latent in case tile_latent = True
            if self.opt["tile_latent"]:
                latent_in = self.latent.expand(-1, 14, -1)
            else:
                latent_in = self.latent

            dip_zk_rand = self.dip_zk + self.reg_noise_std * torch.randn_like(self.dip_zk).cuda()
            # Apply learned linear mapping to match latent distribution to that of the mapping network
            latent_in = self.lrelu(latent_in * self.gaussian_fit["std"] + self.gaussian_fit["mean"])

            # Normalize image to [0,1] instead of [-1,1]
            self.gen_im = self.get_gen_im(latent_in)
            self.gen_ker = self.k_dip(dip_zk_rand)

            # Calculate Losses
            loss, loss_dict = self.loss_builder(latent_in, self.gen_im, self.gen_ker, epoch)
            self.cur_loss = loss.cpu().detach().numpy()

            loss.backward()
            self.optimizer_x.step()
            self.scheduler_x.step()

            msg = " | ".join("{}: {:.4f}".format(k, v) for k, v in loss_dict.items())
            tq_x.set_postfix(loss=msg)

    def log(self):
        if self.cur_loss < self.min_loss:
            self.min_loss = self.cur_loss
            self.best_im = self.gen_im.clone()
            self.best_ker = self.gen_ker.clone()

    def forward(self, ref_im):
        if self.opt["seed"]:
            seed = self.opt["seed"]
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.backends.cudnn.deterministic = True

        self.initialize_dip()
        self.initialize_latent_space()
        self.initialize_optimizers()
        self.warmup_dip()

        self.min_loss = np.inf
        self.gen_im = None
        self.initialize_loss(ref_im)

        if self.verbose:
            print("Optimizing")

        for epoch in range(self.opt["num_epochs"]):
            print("Step: {}".format(epoch + 1))

            self.optimize_x_step(epoch)
            self.log()
            self.optimize_k_step(epoch)
            self.log()

            if self.opt["save_intermediate"]:
                yield (
                    self.best_im.cpu().detach().clamp(0, 1),
                    self.loss_builder.get_blur_img(self.best_im, self.best_ker),
                )
