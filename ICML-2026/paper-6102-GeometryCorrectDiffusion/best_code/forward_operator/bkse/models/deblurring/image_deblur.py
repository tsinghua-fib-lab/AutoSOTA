import torch
import torch.nn as nn
import utils.util as util
from models.dips import ImageDIP, KernelDIP
from models.kernel_encoding.kernel_wizard import KernelWizard
from models.losses.hyper_laplacian_penalty import HyperLaplacianPenalty
from models.losses.perceptual_loss import PerceptualLoss
from models.losses.ssim_loss import SSIM
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm


class ImageDeblur:
    def __init__(self, opt):
        self.opt = opt

        # losses
        self.ssim_loss = SSIM().cuda()
        self.mse = nn.MSELoss().cuda()
        self.perceptual_loss = PerceptualLoss().cuda()
        self.laplace_penalty = HyperLaplacianPenalty(3, 0.66).cuda()

        self.kernel_wizard = KernelWizard(opt["KernelWizard"]).cuda()
        self.kernel_wizard.load_state_dict(torch.load(opt["KernelWizard"]["pretrained"]))

        for k, v in self.kernel_wizard.named_parameters():
            v.requires_grad = False

    def reset_optimizers(self):
        self.x_optimizer = torch.optim.Adam(self.x_dip.parameters(), lr=self.opt["x_lr"])
        self.k_optimizer = torch.optim.Adam(self.k_dip.parameters(), lr=self.opt["k_lr"])

        self.x_scheduler = StepLR(self.x_optimizer, step_size=self.opt["num_iters"] // 5, gamma=0.7)

        self.k_scheduler = StepLR(self.k_optimizer, step_size=self.opt["num_iters"] // 5, gamma=0.7)

    def prepare_DIPs(self):
        # x is stand for the sharp image, k is stand for the kernel
        self.x_dip = ImageDIP(self.opt["ImageDIP"]).cuda()
        self.k_dip = KernelDIP(self.opt["KernelDIP"]).cuda()

        # fixed input vectors of DIPs
        # zk and zx are the length of the corresponding vectors
        self.dip_zk = util.get_noise(64, "noise", (64, 64)).cuda()
        self.dip_zx = util.get_noise(8, "noise", self.opt["img_size"]).cuda()

    def warmup(self, warmup_x, warmup_k):
        # Input vector of DIPs is sampled from N(z, I)
        reg_noise_std = self.opt["reg_noise_std"]

        for step in tqdm(range(self.opt["num_warmup_iters"])):
            self.x_optimizer.zero_grad()
            dip_zx_rand = self.dip_zx + reg_noise_std * torch.randn_like(self.dip_zx).cuda()
            x = self.x_dip(dip_zx_rand)

            loss = self.mse(x, warmup_x)
            loss.backward()
            self.x_optimizer.step()

        print("Warming up k DIP")
        for step in tqdm(range(self.opt["num_warmup_iters"])):
            self.k_optimizer.zero_grad()
            dip_zk_rand = self.dip_zk + reg_noise_std * torch.randn_like(self.dip_zk).cuda()
            k = self.k_dip(dip_zk_rand)

            loss = self.mse(k, warmup_k)
            loss.backward()
            self.k_optimizer.step()

    def deblur(self, img):
        pass
