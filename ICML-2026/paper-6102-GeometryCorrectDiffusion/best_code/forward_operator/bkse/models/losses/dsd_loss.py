import torch
from models.dsd.bicubic import BicubicDownSample
from models.kernel_encoding.kernel_wizard import KernelWizard
from models.losses.ssim_loss import SSIM


class LossBuilder(torch.nn.Module):
    def __init__(self, ref_im, opt):
        super(LossBuilder, self).__init__()
        assert ref_im.shape[2] == ref_im.shape[3]
        self.ref_im = ref_im
        loss_str = opt["loss_str"]
        self.parsed_loss = [loss_term.split("*") for loss_term in loss_str.split("+")]
        self.eps = opt["eps"]

        self.ssim = SSIM().cuda()

        self.D = KernelWizard(opt["KernelWizard"]).cuda()
        self.D.load_state_dict(torch.load(opt["KernelWizard"]["pretrained"]))
        for v in self.D.parameters():
            v.requires_grad = False

    # Takes a list of tensors, flattens them, and concatenates them into a vector
    # Used to calculate euclidian distance between lists of tensors
    def flatcat(self, l):
        l = l if (isinstance(l, list)) else [l]
        return torch.cat([x.flatten() for x in l], dim=0)

    def _loss_l2(self, gen_im_lr, ref_im, **kwargs):
        return (gen_im_lr - ref_im).pow(2).mean((1, 2, 3)).clamp(min=self.eps).sum()

    def _loss_l1(self, gen_im_lr, ref_im, **kwargs):
        return 10 * ((gen_im_lr - ref_im).abs().mean((1, 2, 3)).clamp(min=self.eps).sum())

    # Uses geodesic distance on sphere to sum pairwise distances of the 18 vectors
    def _loss_geocross(self, latent, **kwargs):
        pass


class LossBuilderStyleGAN(LossBuilder):
    def __init__(self, ref_im, opt):
        super(LossBuilderStyleGAN, self).__init__(ref_im, opt)
        im_size = ref_im.shape[2]
        factor = opt["output_size"] // im_size
        assert im_size * factor == opt["output_size"]
        self.bicub = BicubicDownSample(factor=factor)

    # Uses geodesic distance on sphere to sum pairwise distances of the 18 vectors
    def _loss_geocross(self, latent, **kwargs):
        if latent.shape[1] == 1:
            return 0
        else:
            X = latent.view(-1, 1, 18, 512)
            Y = latent.view(-1, 18, 1, 512)
            A = ((X - Y).pow(2).sum(-1) + 1e-9).sqrt()
            B = ((X + Y).pow(2).sum(-1) + 1e-9).sqrt()
            D = 2 * torch.atan2(A, B)
            D = ((D.pow(2) * 512).mean((1, 2)) / 8.0).sum()
            return D

    def forward(self, latent, gen_im, kernel, step):
        var_dict = {
            "latent": latent,
            "gen_im_lr": self.D.adaptKernel(self.bicub(gen_im), kernel),
            "ref_im": self.ref_im,
        }
        loss = 0
        loss_fun_dict = {
            "L2": self._loss_l2,
            "L1": self._loss_l1,
            "GEOCROSS": self._loss_geocross,
        }
        losses = {}

        for weight, loss_type in self.parsed_loss:
            tmp_loss = loss_fun_dict[loss_type](**var_dict)
            losses[loss_type] = tmp_loss
            loss += float(weight) * tmp_loss
        loss += 5e-5 * torch.norm(kernel)
        losses["Norm"] = torch.norm(kernel)

        return loss, losses

    def get_blur_img(self, sharp_img, kernel):
        return self.D.adaptKernel(self.bicub(sharp_img), kernel).cpu().detach().clamp(0, 1)


class LossBuilderStyleGAN2(LossBuilder):
    def __init__(self, ref_im, opt):
        super(LossBuilderStyleGAN2, self).__init__(ref_im, opt)

    # Uses geodesic distance on sphere to sum pairwise distances of the 18 vectors
    def _loss_geocross(self, latent, **kwargs):
        if latent.shape[1] == 1:
            return 0
        else:
            X = latent.view(-1, 1, 14, 512)
            Y = latent.view(-1, 14, 1, 512)
            A = ((X - Y).pow(2).sum(-1) + 1e-9).sqrt()
            B = ((X + Y).pow(2).sum(-1) + 1e-9).sqrt()
            D = 2 * torch.atan2(A, B)
            D = ((D.pow(2) * 512).mean((1, 2)) / 6.0).sum()
            return D

    def forward(self, latent, gen_im, kernel, step):
        var_dict = {
            "latent": latent,
            "gen_im_lr": self.D.adaptKernel(gen_im, kernel),
            "ref_im": self.ref_im,
        }
        loss = 0
        loss_fun_dict = {
            "L2": self._loss_l2,
            "L1": self._loss_l1,
            "GEOCROSS": self._loss_geocross,
        }
        losses = {}

        for weight, loss_type in self.parsed_loss:
            tmp_loss = loss_fun_dict[loss_type](**var_dict)
            losses[loss_type] = tmp_loss
            loss += float(weight) * tmp_loss
        loss += 1e-4 * torch.norm(kernel)
        losses["Norm"] = torch.norm(kernel)

        return loss, losses

    def get_blur_img(self, sharp_img, kernel):
        return self.D.adaptKernel(sharp_img, kernel).cpu().detach().clamp(0, 1)
