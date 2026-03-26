import copy
import os

import torch
from torch.autograd import Variable

from .submodules import *
from .cla_func import *
from .loss_func import *
from engine.utils import one_hot, mmd
from engine.configs import Algorithms
# import torch.nn.functional as F
from engine.utils import AverageMeter
from .networks import Masker
from .model_func import LinearFeatExtractor

import matplotlib.pyplot as plt
import random

class Autoencoder(nn.Module):
    def __init__(self, model_func, cla_func, hparams):
        super().__init__()
        self.model_func = model_func
        self.cla_func = cla_func
        self.hparams = hparams
        self.zc_dim = hparams['zc_dim']
        self.zd_dim = hparams['zd_dim']
        self.num_classes = hparams['num_classes']
        self.seen_domains = hparams['source_domains']
        self.data_size = hparams['data_size']

        # loss weight
        self.lambda_evolve = hparams['lambda_evolve']
        self.lambda_mi = hparams['lambda_mi']
        self.lambda_causal = hparams['lambda_causal']

        self.recon_criterion = nn.MSELoss(reduction='sum')
        self.criterion = nn.CrossEntropyLoss()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                if m.weight is not None and m.bias is not None:
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 1)
            elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)

    def _get_decoder_func(self):
        if len(self.data_size) > 2:
            if self.data_size[-1] == 28:
                decoder_class = CovDecoder28x28
            elif self.data_size[-1] == 84:
                decoder_class = CovDecoder84x84
            elif self.data_size[-1] == 32:
                decoder_class = CovDecoder32x32
            elif self.data_size[-1] == 64:
                decoder_class = CovDecoder64x64
            elif self.data_size[-1] == 224:
                decoder_class = CovDecoder224x224
            else:
                raise ValueError('Don\'t support shape:{}'.format(self.hparams['data_size']))
        else:
            decoder_class = LinearDecoder
        return decoder_class

    @abstractmethod
    def _build(self):
        pass

    @abstractmethod
    def update(self, minibatches, iteration, writer, *args):
        """
        Perform one update step, given a list of (x, y) tuples for all
        environments.
        Admits an optional list of unlabeled minibatches from the test domains,
        when task is domain_adaptation.
        """
        pass

    @abstractmethod
    def predict(self, x, y, *args, **kwargs):
        pass

    def calc_recon_loss(self, recon_x, x):
        recon_loss = self.recon_criterion(recon_x, x)
        recon_loss = recon_loss.sum()
        return recon_loss

    def update_scheduler(self):
        pass


@Algorithms.register('sync')
class SYNC(Autoencoder):
    """
    Implementation of SYNC.
    """
    def __init__(self, model_func, cla_func, hparams):
        super(SYNC, self).__init__(model_func, cla_func, hparams)
        self.stochastic = True
        self.factorised = False
        self.zd_dim = self.num_classes
        self.env_dim = self.seen_domains

        self.register_buffer('latent_c_priors', torch.zeros([2 * self.zc_dim]))
        self.register_buffer('latent_d_priors', torch.zeros([self.zd_dim]))

        self._build()
        self._init()

    def _build(self):
        # prior
        self.static_prior = GaussianModule(self.zc_dim)
        self.dynamic_prior = ProbabilisticSingleLayerLSTM(input_dim=self.zc_dim,
                                                          hidden_dim=2 * self.zc_dim,
                                                          stochastic=self.stochastic)

        self.dynamic_d_prior = ProbabilisticCatSingleLayer(input_dim=self.zd_dim,
                                                               hidden_dim=2 * self.zd_dim,
                                                               stochastic=self.stochastic)

        # posterior
        self.static_encoder = StaticProbabilisticEncoder(self.model_func, self.zc_dim,
                                                           factorised=True,
                                                           stochastic=self.stochastic)
        self.dynamic_encoder = DynamicProbabilisticEncoder(copy.deepcopy(self.model_func),
                                                           latent_dim=self.zc_dim,
                                                           factorised=self.factorised,
                                                           stochastic=self.stochastic)


        self.dynamic_d_encoder = DynamicCatEncoder(input_dim=self.zd_dim,
                                                       factorised=self.factorised,
                                                       stochastic=self.stochastic)


        # reconstruction
        self.decoder = self._get_decoder_func()(2 * self.zc_dim, self.data_size)

        # predictor
        self.classifier = SingleLayerClassifier(2 * self.zc_dim + self.zd_dim, self.num_classes)

        # Masker
        self.masker_stc = Masker(in_dim=self.zc_dim, num_classes=self.zc_dim, middle=6 * self.zc_dim,
                                 k=int(self.zc_dim * 0.6))
        self.masker_dyc = Masker(in_dim=self.zc_dim, num_classes=self.zc_dim, middle=6 * self.zc_dim,
                                 k=int(self.zc_dim * 0.6))

        self.opt = torch.optim.Adam(
            [{'params': self.dynamic_prior.parameters()},
             {'params': self.dynamic_d_prior.parameters()},
             {'params': self.static_encoder.parameters()},
             {'params': self.dynamic_encoder.parameters()},
             {'params': self.dynamic_d_encoder.parameters()},
             {'params': self.decoder.parameters(), 'lr': 1 * self.hparams["lr"]},
             {'params': self.classifier.parameters(), 'lr': 1 * self.hparams["lr"]},
             {'params': self.masker_stc.parameters()},
             {'params': self.masker_dyc.parameters()}],
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )

    @staticmethod
    def gen_dynamic_prior(dynamic_prior_net, latent_priors, domains, batch_size=1):
        z_out, z_out_value = None, None
        hx = dynamic_prior_net.h0.detach().clone()
        cx = dynamic_prior_net.c0.detach().clone()
        z_t = Variable(latent_priors.detach().clone(), requires_grad=True).unsqueeze(0)

        for _ in range(domains):
            z_t, hx, cx = dynamic_prior_net(z_t, Variable(hx.detach().clone(), requires_grad=True),
                                            Variable(cx.detach().clone(), requires_grad=True))

            if z_out is None:
                # If z_out is none it means z_t is z_1, hence store it in the format [batch_size, 1, z_dim]
                z_out = z_t.unsqueeze(1)
                z_out_value = dynamic_prior_net.sampling(batch_size)
            else:
                z_out = torch.cat((z_out, z_t.unsqueeze(1)), dim=1)
                z_out_value = torch.cat((z_out_value, dynamic_prior_net.sampling(batch_size)), dim=1)
        return z_out, z_out_value


    def update(self, minibatches, iteration, writer, *args):
        """
        :param minibatches:
        :param iteration:
        :return:
        """
        cur_epoch, total_it = args

        all_x = torch.stack([x for x, y in minibatches])
        all_y = torch.stack([y for x, y in minibatches])
        domains, batch_size = all_x.shape[:2]

        all_x = torch.transpose(all_x, 0, 1)
        all_y = torch.transpose(all_y, 0, 1)

        # ------------------------------ Covariant shift  -------------------------------
        # prior
        dynamic_p_latent_variables, _ = self.gen_dynamic_prior(self.dynamic_prior, self.latent_c_priors, domains)

        # posterior
        static_qc_latent_variables = self.static_encoder(all_x)
        dynamic_q_latent_variables = self.dynamic_encoder(all_x, None)

        zst = self.static_encoder.sampling()
        zdy = self.dynamic_encoder.sampling()

        recon_x = self.decoder(torch.cat([zst, zdy], dim=1))
        all_x = all_x.contiguous().view(batch_size * domains, *all_x.shape[2:])
        CE_x = self.calc_recon_loss(recon_x, all_x) / batch_size

        # Distribution loss
        # kld on static features
        static_qc_mu, static_qc_log_sigma = \
            static_qc_latent_variables[:, :self.zc_dim], static_qc_latent_variables[:, self.zc_dim:]
        static_qc_sigma = torch.exp(static_qc_log_sigma)
        static_c_kld = 0.5 * torch.sum(
            torch.pow(static_qc_mu, 2) + static_qc_sigma - static_qc_log_sigma - 1
        ) / batch_size

        # kld on dynamic features
        dynamic_q_mu, dynamic_q_log_sigma = \
            dynamic_q_latent_variables[:, :, :self.zc_dim], dynamic_q_latent_variables[:, :, self.zc_dim:]
        dynamic_p_mu, dynamic_p_log_sigma = \
            dynamic_p_latent_variables[:, :, :self.zc_dim], dynamic_p_latent_variables[:, :, self.zc_dim:]
        dynamic_q_sigma = torch.exp(dynamic_q_log_sigma)
        dynamic_p_sigma = torch.exp(dynamic_p_log_sigma)
        dynamic_kld = 0.5 * torch.sum(
            dynamic_p_log_sigma - dynamic_q_log_sigma +
            ((dynamic_q_sigma + torch.pow(dynamic_q_mu - dynamic_p_mu, 2)) / dynamic_p_sigma) - 1
        ) / batch_size

        # ------------------------------ Concept shift  -------------------------------
        all_y = all_y.contiguous().view(-1)
        one_hot_y = one_hot(all_y, self.num_classes, all_y.device).view(batch_size, domains, -1)
        dynamic_qd_latent_variables = self.dynamic_d_encoder(one_hot_y, None)
        dynamic_pd_latent_variables, _ = self.gen_dynamic_prior(self.dynamic_d_prior, self.latent_d_priors, domains)

        # recon y
        zd = self.dynamic_d_encoder.sampling()

        mask_stc = self.masker_stc(zst.detach())
        mask_dyc = self.masker_dyc(zdy.detach())
        zstc = mask_stc * zst
        zdyc = mask_dyc * zdy
        recon_y = self.classifier(torch.cat([zstc, zdyc, zd], dim=1))
        CE_y = self.criterion(recon_y, all_y) * domains

        # kld on drift variables
        dynamic_d_kld = 0.5 * torch.sum(
            torch.softmax(dynamic_qd_latent_variables, dim=-1) *
            (torch.log_softmax(dynamic_qd_latent_variables, dim=-1) - torch.log_softmax(dynamic_pd_latent_variables, dim=-1))
        ) / batch_size

        # l_evolve
        loss_evolve = CE_x + static_c_kld + dynamic_kld + dynamic_d_kld

        # ------------------------------ learning causal representations  -------------------------------
        zst = self.static_encoder.latent_space.base_dist.loc
        zdy = self.dynamic_encoder.latent_space.base_dist.loc

        loss_MI = calculate_mi_loss([dynamic_q_mu, dynamic_q_sigma], [static_qc_mu, static_qc_sigma])

        ######
        mask_stc = self.masker_stc(zst.detach())
        mask_dyc = self.masker_dyc(zdy.detach())
        zstc = mask_stc * zst.detach()
        zdyc = mask_dyc * zdy.detach()

        loss_stc = cross_domain_contrastive_loss(zstc, all_y, domains, batch_size)
        loss_dyc = inter_domain_contrastive_loss(zdyc, zstc.detach(), all_y, domains, batch_size)

        # ------------------------------ optimization objective summary  -------------------------------
        loss_total = CE_y + self.lambda_evolve * loss_evolve + self.lambda_mi * loss_MI + \
                     self.lambda_causal * (loss_stc + loss_dyc)


        self.opt.zero_grad()
        loss_total.backward()
        self.opt.step()

        if iteration % 50 == 0:
            print(
                'loss_recon_x: {:.3f}, CE_y: {:.3f}, '
                'loss_evolve: {:.3f}, loss_MI: {:.3f}, loss_stc: {:.3f},  loss_dyc: {:.3f},'
                'loss_total: {:.3f}'
                .format(CE_x, CE_y, loss_evolve, loss_MI, loss_stc, loss_dyc, loss_total)
            )

        return CE_y, loss_total, recon_y, all_y

    def predict(self, x, domain_idx, *args, **kwargs):
        _ = self.static_encoder(x.unsqueeze(1))
        _ = self.dynamic_encoder(x.unsqueeze(1), use_cached_hidden=True, need_cache=kwargs["need_cache"])

        zst = self.static_encoder.latent_space.base_dist.loc
        zdy = self.dynamic_encoder.latent_space.base_dist.loc
        _, zd_prob = self.gen_dynamic_prior(self.dynamic_d_prior, self.latent_d_priors, domain_idx + 1, x.size(0))
        zd = zd_prob[:, -1, :]

        zstc = self.masker_stc(zst) * zst
        zdyc = self.masker_dyc(zdy) * zdy
        y_logit = self.classifier(torch.cat([zstc, zdyc, zd], dim=1))

        return y_logit