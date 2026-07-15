import logging
from collections import OrderedDict

import models.lr_scheduler as lr_scheduler
import torch
import torch.nn as nn
from models.kernel_encoding.base_model import BaseModel
from models.kernel_encoding.kernel_wizard import KernelWizard
from models.losses.charbonnier_loss import CharbonnierLoss
from torch.nn.parallel import DataParallel, DistributedDataParallel


logger = logging.getLogger("base")


class ImageBaseModel(BaseModel):
    def __init__(self, opt):
        super(ImageBaseModel, self).__init__(opt)

        if opt["dist"]:
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = -1  # non dist training
        train_opt = opt["train"]

        # define network and load pretrained models
        self.netG = KernelWizard(opt["KernelWizard"]).to(self.device)
        self.use_vae = opt["KernelWizard"]["use_vae"]
        if opt["dist"]:
            self.netG = DistributedDataParallel(self.netG, device_ids=[torch.cuda.current_device()])
        else:
            self.netG = DataParallel(self.netG)
        # print network
        self.print_network()
        self.load()

        if self.is_train:
            self.netG.train()

            # loss
            loss_type = train_opt["pixel_criterion"]
            if loss_type == "l1":
                self.cri_pix = nn.L1Loss(reduction="sum").to(self.device)
            elif loss_type == "l2":
                self.cri_pix = nn.MSELoss(reduction="sum").to(self.device)
            elif loss_type == "cb":
                self.cri_pix = CharbonnierLoss().to(self.device)
            else:
                raise NotImplementedError(
                    "Loss type [{:s}] is not\
                                          recognized.".format(
                        loss_type
                    )
                )
            self.l_pix_w = train_opt["pixel_weight"]
            self.l_kl_w = train_opt["kl_weight"]

            # optimizers
            wd_G = train_opt["weight_decay_G"] if train_opt["weight_decay_G"] else 0
            params = []
            for k, v in self.netG.named_parameters():
                if v.requires_grad:
                    params.append(v)
                else:
                    if self.rank <= 0:
                        logger.warning(
                            "Params [{:s}] will not\
                                       optimize.".format(
                                k
                            )
                        )
            optim_params = [
                {"params": params, "lr": train_opt["lr_G"]},
            ]

            self.optimizer_G = torch.optim.Adam(
                optim_params, lr=train_opt["lr_G"], weight_decay=wd_G, betas=(train_opt["beta1"], train_opt["beta2"])
            )
            self.optimizers.append(self.optimizer_G)

            # schedulers
            if train_opt["lr_scheme"] == "MultiStepLR":
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.MultiStepLR_Restart(
                            optimizer,
                            train_opt["lr_steps"],
                            restarts=train_opt["restarts"],
                            weights=train_opt["restart_weights"],
                            gamma=train_opt["lr_gamma"],
                            clear_state=train_opt["clear_state"],
                        )
                    )
            elif train_opt["lr_scheme"] == "CosineAnnealingLR_Restart":
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.CosineAnnealingLR_Restart(
                            optimizer,
                            train_opt["T_period"],
                            eta_min=train_opt["eta_min"],
                            restarts=train_opt["restarts"],
                            weights=train_opt["restart_weights"],
                        )
                    )
            else:
                raise NotImplementedError()

            self.log_dict = OrderedDict()

    def feed_data(self, data, need_GT=True):
        self.LQ = data["LQ"].to(self.device)
        self.HQ = data["HQ"].to(self.device)

    def set_params_lr_zero(self, groups):
        # fix normal module
        for group in groups:
            self.optimizers[0].param_groups[group]["lr"] = 0

    def optimize_parameters(self, step):
        batchsz, _, _, _ = self.LQ.shape

        self.optimizer_G.zero_grad()
        kernel_mean, kernel_sigma = self.netG(self.HQ, self.LQ)

        kernel = kernel_mean + kernel_sigma * torch.randn_like(kernel_mean)
        self.fake_LQ = self.netG.module.adaptKernel(self.HQ, kernel)

        l_pix = self.l_pix_w * self.cri_pix(self.fake_LQ, self.LQ)
        l_total = l_pix

        if self.use_vae:
            KL_divergence = (
                self.l_kl_w
                * torch.sum(
                    torch.pow(kernel_mean, 2)
                    + torch.pow(kernel_sigma, 2)
                    - torch.log(1e-8 + torch.pow(kernel_sigma, 2))
                    - 1
                ).sum()
            )
            l_total += KL_divergence
            self.log_dict["l_KL"] = KL_divergence.item() / batchsz

        l_total.backward()
        self.optimizer_G.step()

        # set log
        self.log_dict["l_pix"] = l_pix.item() / batchsz
        self.log_dict["l_total"] = l_total.item() / batchsz

    def test(self):
        self.netG.eval()
        with torch.no_grad():
            self.fake_H = self.netG(self.var_L)
        self.netG.train()

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_GT=True):
        out_dict = OrderedDict()
        out_dict["LQ"] = self.LQ.detach()[0].float().cpu()
        out_dict["rlt"] = self.fake_LQ.detach()[0].float().cpu()
        return out_dict

    def print_network(self):
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel):
            net_struc_str = "{} - {}".format(self.netG.__class__.__name__, self.netG.module.__class__.__name__)
        else:
            net_struc_str = "{}".format(self.netG.__class__.__name__)
        if self.rank <= 0:
            logger.info(
                "Network G structure: {}, \
                        with parameters: {:,d}".format(
                    net_struc_str, n
                )
            )
            logger.info(s)

    def load(self):
        if self.opt["path"]["pretrain_model_G"]:
            load_path_G = self.opt["path"]["pretrain_model_G"]
            if load_path_G is not None:
                logger.info(
                    "Loading model for G [{:s}]\
                            ...".format(
                        load_path_G
                    )
                )
                self.load_network(load_path_G, self.netG, self.opt["path"]["strict_load"])

    def save(self, iter_label):
        self.save_network(self.netG, "G", iter_label)
