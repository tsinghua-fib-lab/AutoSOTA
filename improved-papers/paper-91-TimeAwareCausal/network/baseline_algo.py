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


class Algorithm(nn.Module):
    """
    A subclass of Algorithm implements a domain generalization algorithm.
    Subclasses should implement the following:
    - update()
    - predict()
    """

    def __init__(self, model_func, cla_func, hparams):
        super(Algorithm, self).__init__()
        self.model_func = model_func
        self.cla_func = cla_func
        self.hparams = hparams
        self.criterion = nn.CrossEntropyLoss()

    @abstractmethod
    def _build(self):
        pass

    @abstractmethod
    def update(self, minibatches, unlabeled=None):
        """
        Perform one update step, given a list of (x, y) tuples for all
        environments.

        Admits an optional list of unlabeled minibatches from the test domains,
        when task is domain_adaptation.
        """
        # raise NotImplementedError
        pass

    @abstractmethod
    def predict(self, x, *args, **kwargs):
        # raise NotImplementedError
        pass

    def update_scheduler(self):
        pass


class AbstractAutoencoder(nn.Module):
    def __init__(self, model_func, cla_func, hparams):
        super().__init__()
        self.model_func = model_func
        self.cla_func = cla_func
        self.hparams = hparams
        # self.feature_dim = model_func.n_outputs
        self.zc_dim = hparams['zc_dim']
        self.zw_dim = hparams['zw_dim']  # for LSSAE
        self.num_classes = hparams['num_classes']
        self.seen_domains = hparams['source_domains']
        self.data_size = hparams['data_size']

        self.recon_criterion = nn.MSELoss(reduction='sum')
        self.criterion = nn.CrossEntropyLoss()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
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
    def update(self, minibatches, iteration, *args):
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


@Algorithms.register('erm')
class ERM(Algorithm):
    """
    Empirical Risk Minimization (ERM)
    """

    def __init__(self, model_func, cla_func, hparams):
        super(ERM, self).__init__(model_func, cla_func, hparams)
        self.network = nn.Sequential(self.model_func, self.cla_func)
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )

    def update(self, minibatches, *args, **kwargs):
        all_x = torch.stack([x for x, y in minibatches])  # [source_domains, batch_size, input_shape]
        all_y = torch.stack([y for x, y in minibatches])  # [source_domains, batch_size]
        if len(self.hparams['data_size']) == 3:
            all_x = torch.reshape(all_x, (all_x.shape[0]*all_x.shape[1], all_x.shape[2], all_x.shape[3], all_x.shape[4]))

        T = all_x.shape[0]
        n = all_x.shape[2]

        # all_z = self.featurizer(all_x)
        # all_logits = self.classifier(all_z.view(-1, all_z.shape[-1]))
        # all_logits = self.network(all_x.view(-1, all_x.shape[-1]))
        all_logits = self.network(all_x.view(-1, all_x.shape[-1]))
        all_y = all_y.view(-1)

        loss = self.criterion(all_logits, all_y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss, all_logits, all_y

    def predict(self, x, *args, **kwargs):
        n = x.shape[1]
        # z = self.featurizer(x)
        # logits = self.classifier(z.view(-1, z.shape[1]))
        # logits = self.network(x.view(-1, x.shape[-1]))
        logits = self.network(x)

        return logits


@Algorithms.register('vae')
class VAE(AbstractAutoencoder):
    """
    Implementation of Vanilla VAE.
    """
    def __init__(self, model_func, cla_func, hparams):
        super(VAE, self).__init__(model_func, cla_func, hparams)
        self.model_func = model_func
        self.cla_func = cla_func
        self._build()

    def _build(self):
        # Static env components
        self.gaussian = GaussianModule(self.hparams['zc_dim'])
        self.encoder = ProbabilisticEncoder(self.model_func, self.hparams['zc_dim'], self.hparams['stochastic'])
        self.decoder = self._get_decoder_func()(self.hparams['zc_dim'] + self.hparams['env_zc_dim'], self.data_size)

        self.domain_cla_func = SingleLayerClassifier(self.hparams['zc_dim'], self.seen_domains)

        self.opt = torch.optim.Adam([{'params': self.encoder.parameters()},
                                     {'params': self.decoder.parameters()}],
                                    lr=self.hparams["lr"],
                                    weight_decay=self.hparams['weight_decay'])

        self.opt_cla = torch.optim.Adam(params=self.cla_func.parameters(), lr=self.hparams["lr"],
                                        weight_decay=self.hparams['weight_decay'])

        self.opt_domain_cla = torch.optim.Adam(params=self.domain_cla_func.parameters(), lr=self.hparams["lr"],
                                               weight_decay=self.hparams['weight_decay'])

    def update(self, minibatches, unlabeled=None):
        cla_losses, all_y_pred, all_y = [], [], []

        for domain_idx, (x, y) in enumerate(minibatches):
            # Step1: Optimize VAE
            _ = self.encoder(x)
            zx_q = self.encoder.sampling()
            recon_x = self.decoder(zx_q)
            recon_loss = self.calc_recon_loss(recon_x, x)
            recon_kl_loss = kl_divergence(self.encoder.latent_space, self.gaussian.latent_space)
            vae_loss = recon_loss + recon_kl_loss
            self.opt.zero_grad()
            vae_loss.backward()
            self.opt.step()

            # Classification
            zx_mu = Variable(self.encoder.latent_space.base_dist.loc, requires_grad=True)
            pred_logit = self.cla_func(zx_mu)
            cla_loss = self.criterion(pred_logit, y)
            self.opt_cla.zero_grad()
            cla_loss.backward()
            self.opt_cla.step()

            domain_y = torch.ones_like(y) * domain_idx
            zx_mu = Variable(self.encoder.latent_space.base_dist.loc, requires_grad=True)
            domain_logit = self.domain_cla_func(zx_mu)
            domain_cla_loss = self.criterion(domain_logit, domain_y)
            self.opt_domain_cla.zero_grad()
            domain_cla_loss.backward()
            self.opt_domain_cla.step()

            # Append status for each domain
            cla_losses.append(cla_loss)
            all_y_pred.append(pred_logit)
            all_y.append(y)

        # Record training procedure
        cla_losses = torch.mean(torch.stack(cla_losses, dim=0))
        all_y_pred = torch.cat(all_y_pred, dim=0)
        all_y = torch.cat(all_y, dim=0)

        return cla_losses, all_y_pred, all_y

    def predict(self, x, *args, **kwargs):
        training = False
        _ = self.encoder(x)
        zx_q = self.encoder.latent_space.base_dist.loc
        output = self.cla_func(zx_q)
        return output


@Algorithms.register('diva')
class DIVA(AbstractAutoencoder):
    """
    Implementation of DIVA.
    """
    def __init__(self, model_func, cla_func, hparams):
        super(DIVA, self).__init__(model_func, cla_func, hparams)
        self.model_func = model_func
        self.cla_func = cla_func
        self.aux_loss_multiplier_d = 3500
        self.aux_loss_multiplier_y = 2000
        self._build()

    def _build(self):
        # Static env components
        self.gaussian = GaussianModule(self.hparams['zc_dim'])
        self.qzx = ProbabilisticEncoder(self.model_func, self.hparams['zc_dim'], self.hparams['stochastic'])
        self.qzy = ProbabilisticEncoder(copy.deepcopy(self.model_func), self.hparams['zdy_dim'],
                                        self.hparams['stochastic'])
        self.qzd = ProbabilisticEncoder(copy.deepcopy(self.model_func), self.hparams['zdy_dim'],
                                        self.hparams['stochastic'])

        self.px = self._get_decoder_func()(self.hparams['zc_dim'] + 2 * self.hparams['zdy_dim'], self.data_size)

        self.pzd = BranchDecoder(self.seen_domains, self.hparams['zdy_dim'], self.hparams['stochastic'])
        self.pzy = BranchDecoder(self.num_classes, self.hparams['zdy_dim'], self.hparams['stochastic'])

        # Auxiliary branch
        self.qd = SingleLayerClassifier(self.hparams['zdy_dim'], self.seen_domains)
        self.qy = self.cla_func

        self.opt = torch.optim.Adam([{'params': self.qzx.parameters()},  #
                                     {'params': self.qzy.parameters()},
                                     {'params': self.qzd.parameters()},
                                     {'params': self.px.parameters()},
                                     {'params': self.pzd.parameters()},
                                     {'params': self.pzy.parameters()},
                                     {'params': self.qd.parameters()},
                                     {'params': self.qy.parameters()}],
                                    lr=self.hparams["lr"],
                                    weight_decay=self.hparams['weight_decay'])

    def update(self, minibatches, unlabeled=None):
        cla_losses, all_y_pred, all_y = [], [], []

        for domain_idx, (x, y) in enumerate(minibatches):
            domain_y = torch.ones_like(y) * domain_idx

            _ = self.qzx(x)
            _ = self.qzy(x)
            _ = self.qzd(x)

            zx_q = self.qzx.sampling()
            zy_q = self.qzy.sampling()
            zd_q = self.qzd.sampling()

            recon_x = self.px(torch.cat([zd_q, zx_q, zy_q], dim=1))

            one_hot_y = one_hot(y, self.num_classes, x.device)
            one_hot_d = one_hot(domain_y, self.seen_domains, x.device)

            _ = self.pzy(one_hot_y)
            _ = self.pzd(one_hot_d)

            d_hat = self.qd(zd_q)
            y_hat = self.qy(zy_q)

            CE_x = self.calc_recon_loss(recon_x, x)
            zd_p_minus_zd_q = torch.sum(self.pzd.latent_space.log_prob(zd_q) - self.qzd.latent_space.log_prob(zd_q))
            KL_zx = torch.sum(self.gaussian.latent_space.log_prob(zx_q) - self.qzx.latent_space.log_prob(zx_q))
            zy_p_minus_zy_q = torch.sum(self.pzy.latent_space.log_prob(zy_q) - self.qzy.latent_space.log_prob(zy_q))

            # Semantic classification
            CE_y = self.criterion(y_hat, y)
            # Domain classification
            CE_d = self.criterion(d_hat, domain_y)

            loss = CE_x - zd_p_minus_zd_q - KL_zx - zy_p_minus_zy_q + self.aux_loss_multiplier_d * CE_d + \
                   self.aux_loss_multiplier_y * CE_y
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

            # Append status for each domain
            cla_losses.append(loss)
            all_y_pred.append(y_hat)
            all_y.append(y)

        # Record training procedure
        cla_losses = torch.mean(torch.stack(cla_losses, dim=0))
        all_y_pred = torch.cat(all_y_pred, dim=0)
        all_y = torch.cat(all_y, dim=0)

        return cla_losses, all_y_pred, all_y

    def predict(self, x, *args, **kwargs):
        _ = self.qzy(x)
        zx_q = self.qzy.latent_space.base_dist.loc
        output = self.qy(zx_q)
        return output


@Algorithms.register('lssae')
class LSSAE(AbstractAutoencoder):
    """
    Implementation of LSSAE.
    """
    def __init__(self, model_func, cla_func, hparams):
        super(LSSAE, self).__init__(model_func, cla_func, hparams)
        self.stochastic = True
        self.factorised = True
        self.zv_dim = self.num_classes

        self.register_buffer('latent_w_priors', torch.zeros([2 * self.zw_dim]))
        if not self.regression_task:
            self.register_buffer('latent_v_priors', torch.zeros([self.zv_dim]))
        else:
            self.register_buffer('latent_v_priors', torch.zeros([2 * self.zv_dim]))

        self.aux_loss_multiplier_y = 200
        self.ts_multiplier = 20.
        # self.aux_loss_multiplier_y = 1.
        # self.ts_multiplier = 0.1
        self._build()
        self._init()

    def _build(self):
        # Static env components
        self.static_prior = GaussianModule(self.zc_dim)
        self.dynamic_w_prior = ProbabilisticSingleLayerLSTM(input_dim=self.zw_dim,
                                                            hidden_dim=2 * self.zw_dim,
                                                            stochastic=self.stochastic)
        self.static_encoder = StaticProbabilisticEncoder(self.model_func, self.zc_dim,
                                                         factorised=self.factorised,
                                                         stochastic=self.stochastic)
        self.dynamic_w_encoder = DynamicProbabilisticEncoder(copy.deepcopy(self.model_func),
                                                             self.zw_dim, self.zc_dim,
                                                             factorised=self.factorised,
                                                             stochastic=self.stochastic)
        if not self.regression_task:
            self.dynamic_v_prior = ProbabilisticCatSingleLayer(input_dim=self.zv_dim,
                                                               hidden_dim=2 * self.zv_dim,
                                                               stochastic=self.stochastic)
            self.dynamic_v_encoder = DynamicCatEncoder(self.zv_dim, self.zc_dim,
                                                       factorised=self.factorised,
                                                       stochastic=self.stochastic)
        else:
            self.dynamic_v_prior = ProbabilisticSingleLayerLSTM(input_dim=self.zv_dim,
                                                                hidden_dim=2 * self.zv_dim,
                                                                stochastic=self.stochastic)
            self.dynamic_v_encoder = DynamicProbabilisticEncoder(copy.deepcopy(self.model_func),
                                                                 self.zv_dim,
                                                                 env_latent_dim=None,
                                                                 factorised=self.factorised,
                                                                 stochastic=self.stochastic,
                                                                 regression=True)

        self.decoder = self._get_decoder_func()(self.zc_dim + self.zw_dim, self.data_size)

        self.category_cla_func = SingleLayerClassifier(self.zc_dim + self.zv_dim, self.num_classes)

        self.opt = torch.optim.Adam([{'params': self.static_encoder.parameters()},
                                     {'params': self.dynamic_w_encoder.parameters(), 'lr': 0.1 * self.hparams["lr"]},
                                     {'params': self.dynamic_v_encoder.parameters(), 'lr': 0.1 * self.hparams["lr"]},
                                     {'params': self.dynamic_w_prior.parameters(), 'lr': 0.1 * self.hparams["lr"]},
                                     {'params': self.dynamic_v_prior.parameters(), 'lr': 0.1 * self.hparams["lr"]},
                                     {'params': self.category_cla_func.parameters()},
                                     {'params': self.decoder.parameters()}],
                                    lr=self.hparams["lr"],
                                    weight_decay=self.hparams['weight_decay'])

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
                # print('latent_space:', dynamic_prior_net.latent_space)
                z_out_value = dynamic_prior_net.sampling(batch_size)
            else:
                z_out = torch.cat((z_out, z_t.unsqueeze(1)), dim=1)
                z_out_value = torch.cat((z_out_value, dynamic_prior_net.sampling(batch_size)), dim=1)
        return z_out, z_out_value

    def update(self, minibatches, iteration, writer, *args):
        """
        :param minibatches:
        :param unlabeled:
        :return:
        """
        all_x = torch.stack([x for x, y in minibatches])
        all_y = torch.stack([y for x, y in minibatches])

        domains, batch_size = all_x.shape[:2]

        all_x = torch.transpose(all_x, 0, 1)
        all_y = torch.transpose(all_y, 0, 1)

        # ------------------------------ Covariant shift  -------------------------------
        static_qx_latent_variables = self.static_encoder(all_x)
        dynamic_qw_latent_variables = self.dynamic_w_encoder(all_x, None)
        dynamic_pw_latent_variables, _ = self.gen_dynamic_prior(self.dynamic_w_prior, self.latent_w_priors, domains)

        zx = self.static_encoder.sampling()
        zw = self.dynamic_w_encoder.sampling()
        recon_x = self.decoder(torch.cat([zx, zw], dim=1))
        all_x = all_x.contiguous().view(batch_size * domains, *all_x.shape[2:])
        CE_x = self.calc_recon_loss(recon_x, all_x)

        # Distribution loss
        # kld on zc
        static_kld = - 2.0 * torch.sum(1 + static_qx_latent_variables[:, self.zc_dim:] -
                                      torch.pow(static_qx_latent_variables[:, :self.zc_dim], 2) -
                                      torch.exp(static_qx_latent_variables[:, self.zc_dim:]))
        # kld on zw
        dynamic_qw_mu, dynamic_qw_log_sigma = dynamic_qw_latent_variables[:, :, :self.zw_dim], \
                                              dynamic_qw_latent_variables[:, :, self.zw_dim:]
        dynamic_pw_mu, dynamic_pw_log_sigma = dynamic_pw_latent_variables[:, :, :self.zw_dim], \
                                              dynamic_pw_latent_variables[:, :, self.zw_dim:]
        dynamic_qw_sigma = torch.exp(dynamic_qw_log_sigma)
        dynamic_pw_sigma = torch.exp(dynamic_pw_log_sigma)

        dynamic_w_kld = 1.0 * torch.sum(dynamic_pw_log_sigma - dynamic_qw_log_sigma + (
                    (dynamic_qw_sigma + torch.pow(dynamic_qw_mu - dynamic_pw_mu, 2)) / dynamic_pw_sigma) - 1)

        # ------------------------------ Concept shift  -------------------------------
        dynamic_pv_latent_variables, _ = self.gen_dynamic_prior(self.dynamic_v_prior, self.latent_v_priors, domains)
        if not self.regression_task:
            all_y = all_y.contiguous().view(-1)
            one_hot_y = one_hot(all_y, self.num_classes, all_y.device)
            one_hot_y = one_hot_y.view(batch_size, domains, -1)
            dynamic_qv_latent_variables = self.dynamic_v_encoder(one_hot_y, None)
        else:
            dynamic_qv_latent_variables = self.dynamic_v_encoder(all_y.contiguous().unsqueeze(2).float(), None)


        # recon y
        zv = self.dynamic_v_encoder.sampling()
        zv.view(batch_size, domains, -1)
        recon_y = self.category_cla_func(torch.cat([zv, zx], dim=1))

        CE_y = self.aux_loss_multiplier_y * self.criterion(recon_y, all_y)

        # # kld on zv
        dynamic_v_kld = 1.0 * torch.sum(torch.softmax(dynamic_qv_latent_variables, dim=-1) *
                                  (torch.log_softmax(dynamic_qv_latent_variables, dim=-1) -
                                   torch.log_softmax(dynamic_pv_latent_variables, dim=-1)))

        # temporal smooth constrain on prior_dynamic_latent_variables
        ts_w_loss = self.ts_multiplier * temporal_smooth_loss(dynamic_qw_latent_variables)
        ts_v_loss = self.ts_multiplier * temporal_smooth_loss(dynamic_qv_latent_variables)

        total_loss = (CE_x + 1 * static_kld + 2 * dynamic_w_kld + dynamic_v_kld) / batch_size + CE_y + ts_w_loss + ts_v_loss
        # total_loss = \
        #     0.005*(CE_x + 1 * static_kld + dynamic_w_kld + dynamic_v_kld) / batch_size + \
        #     self.criterion(recon_y, all_y) + 0.005*(ts_w_loss + ts_v_loss)

        self.opt.zero_grad()
        total_loss.backward()
        self.opt.step()

        recon_x_loss = CE_x / batch_size
        static_loss = static_kld / batch_size
        dynamic_w_kld = dynamic_w_kld / batch_size
        dynamic_v_kld = dynamic_v_kld / batch_size

        # if iteration % 50 == 0:
            # print('Total loss:{:.3f}, recon_x_loss:{:.3f}, recon_y_loss:{:.3f}, static_loss:{:.3f}, dynamic_w_loss:{:.3f}, '
            #       'dynamic_v_loss:{:.3f}, TS_W_loss:{:.3f}, TS_V_loss:{:.3f}'.
            #       format(total_loss, recon_x_loss, CE_y, static_loss, dynamic_w_kld, dynamic_v_kld, ts_w_loss, ts_v_loss))
        # print('CE_y:', self.criterion(recon_y, all_y))

        return self.criterion(recon_y, all_y), total_loss, recon_y, all_y

    def predict(self, x, domain_idx, *args, **kwargs):
        _ = self.static_encoder(x.unsqueeze(1))
        zx = self.static_encoder.latent_space.base_dist.loc
        _, zv_prob = self.gen_dynamic_prior(self.dynamic_v_prior, self.latent_v_priors, domain_idx + 1, x.size(0))
        zv = zv_prob[:, -1, :]
        y_logit = self.category_cla_func(torch.cat([zv, zx], dim=1))

        return y_logit

@Algorithms.register('mmd_lsae')
class MMD_LSAE(AbstractAutoencoder):
    """
    Implementation for MMD-LSAE.
    """
    def __init__(self, model_func, cla_func, hparams):
        super(MMD_LSAE, self).__init__(model_func, cla_func, hparams)
        self.stochastic = False
        self.factorised = False
        self.zv_dim = self.num_classes

        self.register_buffer('latent_w_priors', torch.zeros([2 * self.zw_dim]))
        self.register_buffer('latent_v_priors', torch.zeros([self.zv_dim]))

        self.zc_kernel_type = self.zw_kernel_type = self.zv_kernel_type = 'gaussian'
        self.sigma_list = self.gen_sigma_list()

        self.aux_loss_multiplier_y = 100
        self.lambda_zc_mmd, self.lambda_zw_mmd, self.lambda_zv_mmd = 5.0, 1.0, 1.0
        self.noise_mean, self.noise_var = 0., 0.5
        self._build()
        self._init()

    def _build(self):
        # Static env components
        self.static_prior = GaussianModule(self.zc_dim)
        self.dynamic_w_prior = ProbabilisticSingleLayerLSTM(input_dim=self.zw_dim,
                                                            hidden_dim=2 * self.zw_dim,
                                                            stochastic=True)
        self.dynamic_v_prior = ProbabilisticCatSingleLayer(input_dim=self.zv_dim,
                                                           hidden_dim=2 * self.zv_dim,
                                                           stochastic=True)
        self.static_encoder = StaticProbabilisticEncoder(self.model_func, self.zc_dim,
                                                         factorised=self.factorised,
                                                         stochastic=self.stochastic)
        self.dynamic_w_encoder = DynamicProbabilisticEncoder(copy.deepcopy(self.model_func),
                                                             self.zw_dim, self.zc_dim,
                                                             factorised=self.factorised,
                                                             stochastic=self.stochastic)
        self.dynamic_v_encoder = DynamicCatEncoder(self.zv_dim, self.zc_dim,
                                                   factorised=self.factorised,
                                                   stochastic=self.stochastic)

        self.decoder = self._get_decoder_func()(self.zc_dim + self.zw_dim, self.data_size)

        self.category_cla_func = SingleLayerClassifier(self.zc_dim + self.zv_dim, self.num_classes)

        self.opt = torch.optim.Adam([{'params': self.static_encoder.parameters()},
                                     {'params': self.dynamic_w_encoder.parameters(), 'lr': 0.1 * self.hparams["lr"]},  # 1.0->0.1
                                     {'params': self.dynamic_v_encoder.parameters(), 'lr': 0.1 * self.hparams["lr"]},
                                     {'params': self.dynamic_w_prior.parameters(), 'lr': 0.1 * self.hparams["lr"]},
                                     {'params': self.dynamic_v_prior.parameters(), 'lr': 0.1 * self.hparams["lr"]},
                                     {'params': self.category_cla_func.parameters()},
                                     {'params': self.decoder.parameters()}],
                                    lr=self.hparams["lr"],
                                    weight_decay=self.hparams['weight_decay'])

    @staticmethod
    def gen_sigma_list():
        # sigma for MMD
        base = 1.0
        sigma_list = [1, 2, 4, 8, 16]
        sigma_list = [sigma / base for sigma in sigma_list]
        return sigma_list

    @staticmethod
    def gen_dynamic_prior(domains, dynamic_prior_net, latent_priors, batch_size=1):
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
        :param unlabeled:
        :return:
        """
        all_x = torch.stack([x for x, y in minibatches])  # [source_domains, batch_size, 2]
        all_y = torch.stack([y for x, y in minibatches])  # [source_domains, batch_size]

        domains, batch_size = all_x.shape[:2]

        all_x = torch.transpose(all_x, 0, 1)  # [batch_size, source_domains, 2]
        all_y = torch.transpose(all_y, 0, 1)  # [batch_size, source_domains]

        # ------------------------------ Covariant shift  -------------------------------
        static_qzc_samples = self.static_encoder(all_x)
        dynamic_qzw_samples = self.dynamic_w_encoder(all_x, None)

        static_pzc_samples = self.static_prior.sampling(batch_size*domains).squeeze(1)
        dynamic_pzw_sigma_mu, dynamic_pzw_samples = self.gen_dynamic_prior(domains, self.dynamic_w_prior, self.latent_w_priors, batch_size)

        # flatten
        dynamic_qzw_samples = dynamic_qzw_samples.view(-1, dynamic_qzw_samples.size(-1))
        dynamic_pzw_samples = dynamic_pzw_samples.view(-1, dynamic_pzw_samples.size(-1))

        recon_x = self.decoder(torch.cat([static_qzc_samples, dynamic_qzw_samples], dim=1))
        all_x = all_x.contiguous().view(batch_size * domains, *all_x.shape[2:])
        CE_x = 1.0 * self.calc_recon_loss(recon_x, all_x)

        # Distribution loss
        static_zc_mmd = self.lambda_zc_mmd * mmd(static_qzc_samples, static_pzc_samples, self.zc_kernel_type, self.sigma_list)
        dynamic_zw_mmd = self.lambda_zw_mmd * mmd(dynamic_qzw_samples, dynamic_pzw_samples, self.zw_kernel_type, self.sigma_list)

        # ------------------------------ Concept shift  -------------------------------
        all_y = all_y.contiguous().view(-1)
        one_hot_y = one_hot(all_y, self.num_classes, all_y.device)

        # Add noise for universal approximator posterior [AAE]
        one_hot_y += torch.normal(mean=self.noise_mean, std=self.noise_var, size=one_hot_y.shape).to(one_hot_y.device)

        one_hot_y = one_hot_y.view(batch_size, domains, -1)
        dynamic_qzv_samples = self.dynamic_v_encoder(one_hot_y, None)
        dynamic_pzv_sigma_mu, dynamic_pzv_samples = self.gen_dynamic_prior(domains, self.dynamic_v_prior, self.latent_v_priors, batch_size)

        # flatten
        dynamic_qzv_samples = dynamic_qzv_samples.view(batch_size * domains, -1)
        dynamic_pzv_samples = dynamic_pzv_samples.view(batch_size * domains, -1)

        # recon y
        recon_y = self.category_cla_func(torch.cat([static_qzc_samples, dynamic_qzv_samples], dim=1))
        CE_y = self.aux_loss_multiplier_y * self.criterion(recon_y, all_y)

        # Distribution loss
        dynamic_zv_mmd = self.lambda_zv_mmd * mmd(dynamic_qzv_samples, dynamic_pzv_samples, self.zv_kernel_type)


        # total_loss = (CE_x / domains + static_zc_mmd + dynamic_zw_mmd + dynamic_zv_mmd) / batch_size + CE_y
        total_loss =  CE_x / domains + (static_zc_mmd + dynamic_zw_mmd + dynamic_zv_mmd) / batch_size + CE_y
        self.opt.zero_grad()
        total_loss.backward()
        self.opt.step()

        recon_x_loss = CE_x / (domains) # batch_size *
        static_zc_mmd = static_zc_mmd / batch_size
        dynamic_zw_mmd = dynamic_zw_mmd / batch_size
        dynamic_zv_mmd = dynamic_zv_mmd / batch_size

        # print('Total loss:{:.3f}, recon_x_loss:{:.3f}, recon_y_loss:{:.3f}, static_loss:{:.3f}, dynamic_w_loss:{:.3f}, '
        #       'dynamic_v_loss:{:.3f}'.format(total_loss, recon_x_loss, CE_y, static_zc_mmd, dynamic_zw_mmd, dynamic_zv_mmd))

        return CE_y, recon_y, all_y

    def predict(self, x, domain_idx, *args, **kwargs):
        zc = self.static_encoder(x.unsqueeze(1))
        _, zv_prob = self.gen_dynamic_prior(domain_idx + 1, self.dynamic_v_prior, self.latent_v_priors, x.size(0))

        zv = zv_prob[:, -1, :]
        y_logit = self.category_cla_func(torch.cat([zc, zv], dim=1))

        return y_logit