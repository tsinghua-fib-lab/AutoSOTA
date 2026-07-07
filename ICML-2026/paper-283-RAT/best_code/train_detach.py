import os
import yaml
import time
import math
import types
import torch
import argparse
import numpy as np
from tqdm import trange
from datetime import datetime
from collections import deque
import utils.logger as logger

from torch.nn import functional as F
from torch.func import vmap, grad, functional_call

# pytorch distributed training
import torch.multiprocessing as mp

from utils.runners import Runner
from utils.cg import conjugate_gradient
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter

from utils.utils import build_mlp
from utils.utils import ActorCritic, count_vars, safemean, set_grads_from_flat, set_seed
from vec_env import VecNormalize

def learn(world_size, algo, actor_critic, writer, venv, device,
          total_timesteps, nsteps, algo_config, log_config, log_dir=None):

    gamma = .999
    lam = .95

    per_epoch_timesteps = nsteps * venv.num_envs
    epochs = total_timesteps // per_epoch_timesteps + 1

    pi_minibatch_size = per_epoch_timesteps // algo_config.pi_minibatches
    v_minibatch_size = per_epoch_timesteps // algo_config.v_minibatches

    # Instantiate the runner object
    runner = Runner(env=venv, model=actor_critic, nsteps=nsteps, gamma=gamma, lam=lam, adv_type=algo_config.adv_type, device=device)
    epinfobuf = deque(maxlen=100)

    params_pi = list(actor_critic.pi_net.parameters())
    # Functional copies of the policy state are used for per-sample gradients.
    # They should be treated as read-only snapshots inside the update rules.
    dict_params = {k: v.detach() for k, v in actor_critic.pi_net.named_parameters() if v.requires_grad}
    dict_buffers = {k: v.detach() for k, v in actor_critic.pi_net.named_buffers() if v.requires_grad}

    if algo_config.optimizer == 'adam':
        pi_optimizer = Adam(params_pi, lr=algo_config.lr_pi, weight_decay=algo_config.weight_decay)
    elif algo_config.optimizer == 'sgd': 
        # momentum is enabled to facilitate the implementation of adv-normalized SGD
        # gradient update does not use momentum
        pi_optimizer = SGD(params_pi, lr=algo_config.lr_pi, momentum=1e-6)
    elif algo_config.optimizer == 'kfac':
        from kfac.kfac import KFACOptimizer
        pi_optimizer = KFACOptimizer(actor_critic.pi_net, lr=algo_config.lr_pi,
                                     weight_decay=algo_config.weight_decay)
    elif algo_config.optimizer == 'ekfac':
        from kfac.ekfac import EKFACOptimizer
        pi_optimizer = EKFACOptimizer(actor_critic.pi_net, lr=algo_config.lr_pi,
                                     weight_decay=algo_config.weight_decay)
    else:
        raise NotImplementedError
    
    if hasattr(algo_config, 'lr_decay') and algo_config.lr_decay == 'cosine':
        pi_scheduler = CosineAnnealingLR(pi_optimizer, T_max=epochs*algo_config.pi_epochs*algo_config.pi_minibatches, eta_min=0.01)
    else:
        pi_scheduler = None

    v_optimizer = Adam(actor_critic.v_net.parameters(), lr=algo_config.lr_v)

    # for trust region
    make_flat = lambda x:  torch.cat([grad.contiguous().view(-1) for grad in x if grad is not None])
    get_flat_grad = lambda params:  torch.cat([p.grad.contiguous().view(-1) for p in params if p.grad is not None])

    # Start total timer
    tfirststart = time.perf_counter()

    def TrustRegion_ActorUpdate(_obs, _act, _adv, _outputs_old):
        # Build a Fisher-vector product and solve for the natural-gradient step.
        _outputs = actor_critic.forward_pi(_obs)

        if actor_critic.is_discrete:
            _logp_full = F.log_softmax(_outputs, dim=-1)
            _logp_full_old = F.log_softmax(_outputs_old, dim=-1)
            _llr = torch.gather(_logp_full - _logp_full_old, dim=-1, index=_act.unsqueeze(-1)).squeeze(1)
            _ratio = torch.exp(_llr)
            _p_log_p = torch.exp(_logp_full) * _logp_full
            _entropy = - _p_log_p.sum(-1).mean()
            _outputs_ref = _outputs.detach()
            _logp_full_ref = F.log_softmax(_outputs_ref, dim=-1)
            full_llr = _logp_full_ref - _logp_full
            _ent_kl = (torch.exp(_logp_full_ref) * full_llr).sum(dim=-1).mean()
            _real_kl = (torch.exp(_logp_full_old) * (_logp_full_old - _logp_full)).sum(dim=-1).mean()

        else:
            _mu, _logstd = _outputs.chunk(2, dim=-1)
            _dist = torch.distributions.Normal(_mu, torch.exp(_logstd))
            _logp = _dist.log_prob(_act).sum(dim=-1) 

            _mu_old, _logstd_old = _outputs_old.chunk(2, dim=-1)
            _dist_old = torch.distributions.Normal(_mu_old, torch.exp(_logstd_old))
            _logp_old = _dist_old.log_prob(_act).sum(dim=-1)

            _llr = _logp - _logp_old
            _ratio = torch.exp(_llr)

            _entropy = _dist.entropy().sum(dim=-1).mean()
            _outputs_ref = _outputs.detach()

            _mu_ref, _logstd_ref = _outputs_ref.chunk(2, dim=-1)
            _ent_kl = (_logstd - _logstd_ref + 0.5 * ( torch.exp(_logstd_ref).pow(2) + (_mu_ref - _mu).pow(2) ) / torch.exp(_logstd).pow(2) - 0.5).sum(dim=-1).mean()
            _real_kl = (_logstd - _logstd_old + 0.5 * ( torch.exp(_logstd_old).pow(2) + (_mu_old - _mu).pow(2) ) / torch.exp(_logstd).pow(2) - 0.5).sum(dim=-1).mean()

        # zero mean of advantage
        _adv = _adv - _adv.mean() 
        
        # clamp the ratio
        if algo_config.clamp_ratio:
            _ratio = torch.clamp(_ratio, algo_config.min_ratio, algo_config.max_ratio)
        _loss_pi = (- _ratio * _adv).mean() 

        if algo_config.norm_obj == 'adv':
            _rms_sqrt = torch.sqrt( _adv.pow(2).mean() ).detach()
        elif algo_config.norm_obj == 'obj':
            _rms_sqrt = torch.sqrt( (_ratio * _adv).pow(2).mean() ).detach() # might related to variance reduction in importance sampling
        elif algo_config.norm_obj == 'ratio':
            _rms_sqrt = _ratio.mean().detach() * torch.sqrt( _adv.pow(2).mean() ).detach()
        else: 
            raise NotImplementedError

        # normalize the loss to stabilize the training
        _loss_pi = _loss_pi / (_rms_sqrt + 1e-8)

        # The entropy-based KL surrogate gives the curvature term for CG.
        kl_grad = torch.autograd.grad(_ent_kl, params_pi, create_graph=True)
        kl_grad_flat = make_flat(kl_grad)
        def fisher_vector_product(x):
            dot_prod = torch.dot(kl_grad_flat, x)
            fvp = torch.autograd.grad(dot_prod, params_pi, retain_graph=True)
            fvp_flat = make_flat(fvp)
            return fvp_flat + algo_config.cg_damping * x

        # Write the solved step into gradients so the optimizer can apply it.
        pi_optimizer.zero_grad()
        _loss = _loss_pi - algo_config.ent_coef * _entropy
        _loss.backward(retain_graph=True)

        if algo_config.grad == 'pg':
            step_dir = get_flat_grad(params_pi).detach()
        elif algo_config.grad == 'npg':
            loss_grad_pi_flat = get_flat_grad(params_pi).detach()
            step_dir = conjugate_gradient(fisher_vector_product, loss_grad_pi_flat, nsteps=algo_config.cg_steps)
        else:
            raise NotImplementedError

        ## gradient clipping
        max_grad_norm = algo_config.max_grad_norm
        if algo_config.post_grad == 'fisher_clip':
            grad_norm = torch.dot(step_dir, fisher_vector_product(step_dir)) # Fisher norm
            assert grad_norm.item() >= 0.0
            step_dir = step_dir * torch.clamp(max_grad_norm / grad_norm, max=1.0)
        elif algo_config.post_grad == 'l2_clip':
            grad_norm = torch.dot(step_dir, step_dir) # L2 norm
            assert grad_norm.item() >= 0.0
            step_dir = step_dir * torch.clamp(max_grad_norm / grad_norm, max=1.0)
        elif algo_config.post_grad == 'norm':
            grad_norm = torch.dot(step_dir, fisher_vector_product(step_dir)) # Fisher norm
            assert grad_norm.item() >= 0.0
            step_dir = step_dir / grad_norm.sqrt()
        else:
            grad_norm = torch.tensor(max_grad_norm) # no clipping

        set_grads_from_flat(params_pi, step_dir)

        pi_optimizer.step()

        # Useful extra info
        with torch.no_grad():
            clipfrac = 0.0
            pi_info = dict(kl=_real_kl.item(), curr_lr=pi_optimizer.param_groups[0]['lr'], ent=_entropy.item(), cf=clipfrac, ent_kl=_ent_kl.item(),
                           kl_grad_norm=kl_grad_flat.norm().item(), grad_norm=grad_norm.item(),
                           ratio_max=_ratio.max().item(), ratio_min=_ratio.min().item())

        return _loss, _loss_pi, pi_info

    def RAT_ActorUpdate(_obs, _act, _adv, _outputs_old):
        # RAT solves a sample-space system before mapping the step back to parameters.
        _outputs = actor_critic.forward_pi(_obs)

        if actor_critic.is_discrete:
            _logp_full = F.log_softmax(_outputs, dim=-1)
            _logp_full_old = F.log_softmax(_outputs_old, dim=-1)
            _llr = torch.gather(_logp_full - _logp_full_old, dim=-1, index=_act.unsqueeze(-1)).squeeze(1)
            _ratio = torch.exp(_llr)
            _p_log_p = torch.exp(_logp_full) * _logp_full
            _entropy = - _p_log_p.sum(-1).mean()
            _logp = torch.gather(_logp_full, dim=-1, index=_act.unsqueeze(-1)).squeeze(1)
            _real_kl = (torch.exp(_logp_full_old) * (_logp_full_old - _logp_full)).sum(dim=-1).mean()

            def compute_logp(params, buffers, batch_obs, batch_act):
                batch_obs, batch_act = batch_obs.unsqueeze(0), batch_act.unsqueeze(0)
                batch_outs = functional_call(actor_critic.pi_net, (params, buffers), (batch_obs,) )
                batch_logp_full = F.log_softmax(batch_outs, dim=-1)
                batch_logp = torch.gather(batch_logp_full, dim=-1, index=batch_act.unsqueeze(-1)).squeeze(1)
                return batch_logp.squeeze(0)

        else:
            _mu, _logstd = _outputs.chunk(2, dim=-1)
            _dist = torch.distributions.Normal(_mu, torch.exp(_logstd))
            _logp = _dist.log_prob(_act).sum(dim=-1) 

            _mu_old, _logstd_old = _outputs_old.chunk(2, dim=-1)
            _dist_old = torch.distributions.Normal(_mu_old, torch.exp(_logstd_old))
            _logp_old = _dist_old.log_prob(_act).sum(dim=-1)

            _llr = _logp - _logp_old
            _ratio = torch.exp(_llr)
            _entropy = _dist.entropy().sum(dim=-1).mean()
            _real_kl = (_logstd - _logstd_old + 0.5 * ( torch.exp(_logstd_old).pow(2) + (_mu_old - _mu).pow(2) ) / torch.exp(_logstd).pow(2) - 0.5).sum(dim=-1).mean()

            def compute_logp(params, buffers, batch_obs, batch_act):
                batch_obs, batch_act = batch_obs.unsqueeze(0), batch_act.unsqueeze(0)
                batch_outs = functional_call(actor_critic.pi_net, (params, buffers), (batch_obs,) )
                batch_mu, batch_logstd = batch_outs.chunk(2, dim=-1)

                var = torch.exp(batch_logstd)**2
                batch_logp = (
                    -((batch_act - batch_mu) ** 2) / (2 * var)
                    - batch_logstd
                    - math.log(math.sqrt(2 * math.pi))
                )

                return batch_logp.sum(dim=-1).squeeze(0)

        # zero mean of advantage
        _adv = _adv - _adv.mean() 
        
        # clamp the ratio
        if algo_config.clamp_ratio:
            _ratio = torch.clamp(_ratio, algo_config.min_ratio, algo_config.max_ratio)

        if algo_config.norm_obj == 'adv':
            _rms_sqrt = torch.sqrt( _adv.pow(2).mean() ).detach()
        elif algo_config.norm_obj == 'obj':
            _rms_sqrt = torch.sqrt( (_ratio * _adv).pow(2).mean() ).detach() # might related to variance reduction in importance sampling
        elif algo_config.norm_obj == 'ratio':
            _rms_sqrt = _ratio.mean().detach() * torch.sqrt( _adv.pow(2).mean() ).detach()
        else: 
            raise NotImplementedError
        _adv = _adv / (_rms_sqrt + 1e-8)

        pi_optimizer.zero_grad()
        # Per-sample policy gradients are required to form the batch-space linear system.
        ft_compute_sample_grad = vmap(grad(compute_logp), in_dims=(None, None, 0, 0))
        ft_per_sample_grads = ft_compute_sample_grad(dict_params, dict_buffers, _obs, _act) # num_samples x param_shape

        with torch.no_grad():
            num_sa = _obs.shape[0]
            H = torch.cat([v.contiguous().view(num_sa, -1) for v in ft_per_sample_grads.values()], dim=-1)  # num_samples x num_params
            # HHT is the sample-space curvature matrix used to correct the advantage.
            HHT = H @ H.t() @ torch.diag(_ratio) / num_sa # num_samples x num_samples

            gk_list = [ v['momentum_buffer'].contiguous().flatten() for v in pi_optimizer.state.values() if v['momentum_buffer'] is not None ]
            if algo_config.is_karzmarz and len(gk_list) > 0:
                g_k = torch.cat(gk_list, dim=0)
                _adv = _adv - torch.mv(H, g_k)

            _png_adv = torch.linalg.solve( HHT + algo_config.cg_damping * torch.eye(num_sa, device=device), _adv)

        # udpate actor
        _loss_pi = (- _ratio.detach() * _logp * _png_adv).mean() 
        pi_optimizer.zero_grad()
        _loss = _loss_pi - algo_config.ent_coef * _entropy
        _loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(params_pi, algo_config.max_grad_norm) 
        pi_optimizer.step()
        if pi_scheduler is not None:
            pi_scheduler.step()

        # Useful extra info
        with torch.no_grad():
            clipfrac = 0.0
            pi_info = dict(kl=_real_kl.item(), curr_lr=pi_optimizer.param_groups[0]['lr'], ent=_entropy.item(), cf=clipfrac, 
                           grad_norm=grad_norm.item(), ratio_max=_ratio.max().item(), ratio_min=_ratio.min().item())

        return _loss, _loss_pi, pi_info

    def diag_ActorUpdate(_obs, _act, _adv, _outputs_old):
        # This variant keeps only the diagonal of the Fisher approximation.
        _outputs = actor_critic.forward_pi(_obs)

        if actor_critic.is_discrete:
            _logp_full = F.log_softmax(_outputs, dim=-1)
            _logp_full_old = F.log_softmax(_outputs_old, dim=-1)
            _llr = torch.gather(_logp_full - _logp_full_old, dim=-1, index=_act.unsqueeze(-1)).squeeze(1)
            _ratio = torch.exp(_llr)
            _p_log_p = torch.exp(_logp_full) * _logp_full
            _entropy = - _p_log_p.sum(-1).mean()
            _logp = torch.gather(_logp_full, dim=-1, index=_act.unsqueeze(-1)).squeeze(1)
            _real_kl = (torch.exp(_logp_full_old) * (_logp_full_old - _logp_full)).sum(dim=-1).mean()

            def compute_logp(params, buffers, batch_obs, batch_act):
                batch_obs, batch_act = batch_obs.unsqueeze(0), batch_act.unsqueeze(0)
                batch_outs = functional_call(actor_critic.pi_net, (params, buffers), (batch_obs,) )
                batch_logp_full = F.log_softmax(batch_outs, dim=-1)
                batch_logp = torch.gather(batch_logp_full, dim=-1, index=batch_act.unsqueeze(-1)).squeeze(1)
                return batch_logp.squeeze(0)

        else:
            _mu, _logstd = _outputs.chunk(2, dim=-1)
            _dist = torch.distributions.Normal(_mu, torch.exp(_logstd))
            _logp = _dist.log_prob(_act).sum(dim=-1) 

            _mu_old, _logstd_old = _outputs_old.chunk(2, dim=-1)
            _dist_old = torch.distributions.Normal(_mu_old, torch.exp(_logstd_old))
            _logp_old = _dist_old.log_prob(_act).sum(dim=-1)

            _llr = _logp - _logp_old
            _ratio = torch.exp(_llr)
            _entropy = _dist.entropy().sum(dim=-1).mean()
            _real_kl = (_logstd - _logstd_old + 0.5 * ( torch.exp(_logstd_old).pow(2) + (_mu_old - _mu).pow(2) ) / torch.exp(_logstd).pow(2) - 0.5).sum(dim=-1).mean()

            def compute_logp(params, buffers, batch_obs, batch_act):
                batch_obs, batch_act = batch_obs.unsqueeze(0), batch_act.unsqueeze(0)
                batch_outs = functional_call(actor_critic.pi_net, (params, buffers), (batch_obs,) )
                batch_mu, batch_logstd = batch_outs.chunk(2, dim=-1)

                var = torch.exp(batch_logstd)**2
                batch_logp = (
                    -((batch_act - batch_mu) ** 2) / (2 * var)
                    - batch_logstd
                    - math.log(math.sqrt(2 * math.pi))
                )

                return batch_logp.sum(dim=-1).squeeze(0)

        # zero mean of advantage
        _adv = _adv - _adv.mean() 
        
        # clamp the ratio
        if algo_config.clamp_ratio:
            _ratio = torch.clamp(_ratio, algo_config.min_ratio, algo_config.max_ratio)

        if algo_config.norm_obj == 'adv':
            _rms_sqrt = torch.sqrt( _adv.pow(2).mean() ).detach()
        elif algo_config.norm_obj == 'obj':
            _rms_sqrt = torch.sqrt( (_ratio * _adv).pow(2).mean() ).detach() # might related to variance reduction in importance sampling
        elif algo_config.norm_obj == 'ratio':
            _rms_sqrt = _ratio.mean().detach() * torch.sqrt( _adv.pow(2).mean() ).detach()
        else: 
            raise NotImplementedError
        _adv = _adv / (_rms_sqrt + 1e-8)

        pi_optimizer.zero_grad()
        # Reuse the same per-sample gradients, then collapse them to a diagonal preconditioner.
        ft_compute_sample_grad = vmap(grad(compute_logp), in_dims=(None, None, 0, 0))
        ft_per_sample_grads = ft_compute_sample_grad(dict_params, dict_buffers, _obs, _act) # num_samples x param_shape

        with torch.no_grad():
            num_sa = _obs.shape[0]
            H = torch.cat([v.contiguous().view(num_sa, -1) for v in ft_per_sample_grads.values()], dim=-1)  # num_samples x num_params
            diag_fisher = torch.linalg.norm(H, dim=0).pow(2) / num_sa  # num_params

        # udpate actor
        _loss_pi = (- _ratio.detach() * _logp * _adv).mean() 
        pi_optimizer.zero_grad()
        _loss = _loss_pi - algo_config.ent_coef * _entropy
        _loss.backward()

        loss_grad_pi_flat = get_flat_grad(params_pi).detach()
        step_dir = loss_grad_pi_flat / (diag_fisher + algo_config.cg_damping)
        set_grads_from_flat(params_pi, step_dir)

        grad_norm = torch.nn.utils.clip_grad_norm_(params_pi, algo_config.max_grad_norm) 
        pi_optimizer.step()
        if pi_scheduler is not None:
            pi_scheduler.step()

        # Useful extra info
        with torch.no_grad():
            clipfrac = 0.0
            pi_info = dict(kl=_real_kl.item(), curr_lr=pi_optimizer.param_groups[0]['lr'], ent=_entropy.item(), cf=clipfrac, 
                           grad_norm=grad_norm.item(), ratio_max=_ratio.max().item(), ratio_min=_ratio.min().item())

        return _loss, _loss_pi, pi_info

    def KFAC_ActorUpdate(_obs, _act, _adv, _outputs_old):
        # KFAC alternates between collecting Fisher statistics and applying the policy step.
        pi_optimizer.zero_grad()
        _outputs = actor_critic.forward_pi(_obs)

        if actor_critic.is_discrete:
            _logp_full = F.log_softmax(_outputs, dim=-1)
            _logp_full_old = F.log_softmax(_outputs_old, dim=-1)
            _llr = torch.gather(_logp_full - _logp_full_old, dim=-1, index=_act.unsqueeze(-1)).squeeze(1)
            _ratio = torch.exp(_llr)
            _p_log_p = torch.exp(_logp_full) * _logp_full
            _entropy = - _p_log_p.sum(-1).mean()
            _kl = (torch.exp(_logp_full_old) * (_logp_full_old - _logp_full)).sum(dim=-1).mean()

        else:
            _mu, _logstd = _outputs.chunk(2, dim=-1)
            _dist = torch.distributions.Normal(_mu, torch.exp(_logstd))
            _logp = _dist.log_prob(_act).sum(dim=-1) 

            _mu_old, _logstd_old = _outputs_old.chunk(2, dim=-1)
            _dist_old = torch.distributions.Normal(_mu_old, torch.exp(_logstd_old))
            _logp_old = _dist_old.log_prob(_act).sum(dim=-1)

            _llr = _logp - _logp_old
            _ratio = torch.exp(_llr)

            _entropy = _dist.entropy().sum(dim=-1).mean()
            _kl = (_logstd - _logstd_old + 0.5 * ( torch.exp(_logstd_old).pow(2) + (_mu_old - _mu).pow(2) ) / torch.exp(_logstd).pow(2) - 0.5).sum(dim=-1).mean()

        # if pi_optimizer.steps % pi_optimizer.TCov == 0:
        if pi_optimizer.steps % pi_optimizer.TInv == 0:
            # Compute fisher, see Martens 2014
            actor_critic.pi_net.zero_grad()
            pg_fisher_loss = - _logp.mean()
            pi_optimizer.acc_stats = True
            pg_fisher_loss.backward(retain_graph=True)
            pi_optimizer.acc_stats = False

        # zero mean of advantage
        _adv = _adv - _adv.mean() 
        
        # clamp the ratio
        if algo_config.clamp_ratio:
            _ratio = torch.clamp(_ratio, algo_config.min_ratio, algo_config.max_ratio)
        _loss_pi = (- _ratio * _adv).mean() 

        if algo_config.norm_obj == 'adv':
            _rms_sqrt = torch.sqrt( _adv.pow(2).mean() ).detach()
        elif algo_config.norm_obj == 'obj':
            _rms_sqrt = torch.sqrt( (_ratio * _adv).pow(2).mean() ).detach() # might related to variance reduction in importance sampling
        elif algo_config.norm_obj == 'ratio':
            _rms_sqrt = _ratio.mean().detach() * torch.sqrt( _adv.pow(2).mean() ).detach()
        else: 
            raise NotImplementedError

        # normalize the loss to stabilize the training
        _loss_pi = _loss_pi / (_rms_sqrt + 1e-8)
        _loss = _loss_pi - algo_config.ent_coef * _entropy

        _loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(params_pi, algo_config.max_grad_norm)
        pi_optimizer.step()

        # Useful extra info
        with torch.no_grad():
            clipfrac = 0.0
            approx_kl = _kl.item()
            ent = _entropy.item()
            pi_info = dict(kl=approx_kl, ent=ent, curr_lr=pi_optimizer.param_groups[0]['lr'], cf=clipfrac,
                           grad_norm=grad_norm.item(), ratio_max=_ratio.max().item(), ratio_min=_ratio.min().item())

        return _loss, _loss_pi, pi_info

    # Select the actor update rule for the configured algorithm variant.
    if algo in {'fvp'}: 
        update_actor = TrustRegion_ActorUpdate
    elif algo in {'rat'}:
        update_actor = RAT_ActorUpdate
    elif algo in {'kfac', 'ekfac'}:
        update_actor = KFAC_ActorUpdate
    elif algo in {'diag'}:
        update_actor = diag_ActorUpdate
    else: 
        raise NotImplementedError

    tepochs = trange(epochs+1, desc='Epoch starts', leave=True)

    # Main loop: collect experience in env and update/log each epoch
    inds = np.arange(per_epoch_timesteps)
    compute_time = []

    for epoch in tepochs:
        tstart = time.perf_counter()

        tepochs.set_description('Stepping environment...')

        actor_critic.eval() # set to eval mode
        obs, ret, act, adv, outputs_old, epinfos = runner.run() #pylint: disable=E0632

        epinfobuf.extend(epinfos)
        tepochs.set_description('Minibatch training...')

        # pop art
        if actor_critic.with_popart:
            actor_critic.last_v_layer.update(ret) # update the mean/var
            ret = actor_critic.last_v_layer.normalize(ret)
            adv = actor_critic.last_v_layer.normalize(adv)

        if actor_critic.obs_rms is not None:
            actor_critic.obs_rms.training = True
            obs = actor_critic.obs_rms(obs) # norm obs for training
            actor_critic.obs_rms.training = False
            # Re-evaluate the old policy on normalized observations so the ratios
            # are computed in the same input space used for training.
            with torch.no_grad():
                outputs_old = actor_critic.forward_pi(obs)

        actor_critic.train()  # set to train mode
        actor_tstart = time.perf_counter()
        for _ in range(algo_config.pi_epochs):
            # Randomize the indexes
            np.random.shuffle(inds)
            # 0 to batch_size with batch_train_size step
            for start in range(0, per_epoch_timesteps, pi_minibatch_size):
                end = start + pi_minibatch_size
                mbinds = inds[start:end]
                mb_obs, mb_act, mb_adv, mb_outputs_old = obs[mbinds], act[mbinds], adv[mbinds], outputs_old[mbinds]
                mb_loss, mb_loss_pi, pi_info = update_actor(mb_obs, mb_act, mb_adv, mb_outputs_old)
        actor_tnow = time.perf_counter()
        actor_time_elapsed = actor_tnow - actor_tstart
        compute_time.append(actor_time_elapsed)

        # kl adaptive lr adjustment
        if algo_config.use_kl_adaptive_lr:
            curr_kl = pi_info['kl']
            if curr_kl > 0.008 * 2:
                pi_optimizer.param_groups[0]['lr'] = max(pi_optimizer.param_groups[0]['lr'] / 1.5, 1e-4)
            elif curr_kl < 0.008 / 2:
                pi_optimizer.param_groups[0]['lr'] = min(pi_optimizer.param_groups[0]['lr'] * 1.5, 5e-2)

        for _ in range(algo_config.v_epochs):
            # Randomize the indexes
            np.random.shuffle(inds)
            # 0 to batch_size with batch_train_size step
            for start in range(0, per_epoch_timesteps, v_minibatch_size):
                end = start + v_minibatch_size
                mbinds = inds[start:end]
                _obs, _ret = obs[mbinds], ret[mbinds]
                _vals = actor_critic.forward_v(_obs) # get the value estimate

                # value loss
                mb_loss_v = F.mse_loss(_vals, _ret)

                v_optimizer.zero_grad()
                mb_loss_v.backward()
                torch.nn.utils.clip_grad_norm_(actor_critic.v_net.parameters(), 5.0)
                v_optimizer.step()

        tepochs.set_postfix(loss_pi=mb_loss_pi.item(), loss_v=mb_loss_v.item(), entropy=pi_info['ent'], kl=pi_info['kl'], cf=pi_info['cf'], lr=pi_info['curr_lr'])

        # clean GPU cache
        torch.cuda.empty_cache()

        tnow = time.perf_counter()
        # Calculate the fps (frame per second)
        fps = int(per_epoch_timesteps / (tnow - tstart))

        if logger.get_dir() is not None and (epoch+1) % log_config.log_interval == 0:
            # Calculates if value function is a good predicator of the returns (ev > 1)
            # or if it's just worse than predicting nothing (ev =< 0)
            logger.logkv("misc/serial_timesteps", (epoch+1)*per_epoch_timesteps)
            logger.logkv("misc/nupdates", epoch)
            logger.logkv("misc/total_timesteps", (epoch+1)*per_epoch_timesteps*world_size)
            logger.logkv("fps", fps)
            logger.logkv("loss_pi", mb_loss_pi.item())
            logger.logkv("loss_v", mb_loss_v.item())
            logger.logkv("ret_max", ret.max().item())
            logger.logkv("ret_min", ret.min().item())
            logger.logkv("ret_avg", ret.mean().item())
            logger.logkv("ret_med", ret.median().item())
            logger.logkv("ret_var", ret.var().item())
            logger.logkv("action_max", act.max().item())
            logger.logkv("action_min", act.min().item())
            logger.logkv("adv_max", adv.max().item())
            logger.logkv("adv_min", adv.min().item())
            logger.logkv("adv_avg", adv.mean().item())
            logger.logkv("adv_med", adv.median().item())
            logger.logkv("adv_var", adv.var().item())
            logger.logkv("entropy", pi_info['ent'])
            logger.logkv("lr_pi", pi_info['curr_lr'])
            logger.logkv("kl", pi_info['kl'])
            if algo in {'fvp'}:
                logger.logkv("ent_kl", pi_info['ent_kl'])
                logger.logkv("kl_grad_norm", pi_info['kl_grad_norm'])
            logger.logkv("grad_norm", pi_info['grad_norm'])
            logger.logkv("lr_v", v_optimizer.param_groups[0]['lr'])
            logger.logkv("clipfrac", pi_info['cf'])
            logger.logkv("ratio_max", pi_info['ratio_max'])
            logger.logkv("ratio_min", pi_info['ratio_min'])
            logger.logkv('eprewmean', safemean([epinfo['r'] for epinfo in epinfobuf]))
            logger.logkv('eplenmean', safemean([epinfo['l'] for epinfo in epinfobuf]))
            logger.logkv('misc/time_elapsed', tnow - tfirststart)

            logger.dumpkvs()

        # Log changes from update
        # writer.add_scalar('train/rewards', rew.sum(), epoch)
        if writer is not None:
            writer.add_scalar('train/kl', pi_info['kl'], epoch)
            if algo in {'fvp'}:
                writer.add_scalar("ent_kl", pi_info['ent_kl'], epoch)
                writer.add_scalar("kl_grad_norm", pi_info['kl_grad_norm'], epoch)
            writer.add_scalar("grad_norm", pi_info['grad_norm'], epoch)
            writer.add_scalar('train/clipfrac', pi_info['cf'], epoch)
            writer.add_scalar('train/entropy', pi_info['ent'], epoch)
            writer.add_scalar('train/lr_pi', pi_info['curr_lr'], epoch)
            writer.add_scalar('train/ratio_max', pi_info['ratio_max'], epoch)
            writer.add_scalar('train/ratio_min', pi_info['ratio_min'], epoch)
            writer.add_scalar('train/loss_pi', mb_loss_pi, epoch)
            writer.add_scalar('train/loss_v', mb_loss_v, epoch)
            writer.add_scalar('train/lr_v', v_optimizer.param_groups[0]['lr'], epoch)
            writer.add_scalar("train/ret_max", ret.max().item(), epoch)
            writer.add_scalar("train/ret_min", ret.min().item(), epoch)
            writer.add_scalar("train/ret_avg", ret.mean().item(), epoch)
            writer.add_scalar("train/ret_med", ret.median().item(), epoch)
            writer.add_scalar("train/ret_var", ret.var().item(), epoch)
            writer.add_scalar("train/act_max", act.max().item(), epoch)
            writer.add_scalar("train/act_min", act.min().item(), epoch)
            writer.add_scalar("train/adv_max", adv.max().item(), epoch)
            writer.add_scalar("train/adv_min", adv.min().item(), epoch)
            writer.add_scalar("train/adv_avg", adv.mean().item(), epoch)
            writer.add_scalar("train/adv_med", adv.median().item(), epoch)
            writer.add_scalar("train/adv_var", adv.var().item(), epoch)
            writer.add_scalar('train/eprewmean', safemean([epinfo['r'] for epinfo in epinfobuf]), epoch)
            writer.add_scalar('train/eplenmean', safemean([epinfo['l'] for epinfo in epinfobuf]), epoch)
            writer.add_scalar('misc/time_elapsed', tnow - tfirststart, epoch)
            writer.add_scalar("misc/serial_timesteps", (epoch+1)*per_epoch_timesteps, epoch)
            writer.add_scalar("misc/nupdates", epoch)
            writer.add_scalar("misc/total_timesteps", (epoch+1)*per_epoch_timesteps*world_size, epoch)

    if log_dir is not None:
        # save checkpoints
        torch.save({'model_state_dict': actor_critic.state_dict(), }, f'{log_dir}/model.ckpt')
        import json
        with open(f'{log_dir}/time.json', 'w') as f:
            json.dump({'compute_time_array': compute_time, 
                       'average': np.mean(compute_time), 
                       'time_per_update': np.mean(compute_time)/(per_epoch_timesteps / pi_minibatch_size * algo_config.pi_epochs), 
                       'stderr': np.std(compute_time)/np.sqrt(len(compute_time)), 
                       'updates': len(compute_time)}, f)

def train_fn(rank, world_size, algo, seed, algo_config, env_config, nets_config, log_config, device=-1):
    # Serialize data into file:
    time_now = datetime.now().strftime('%Y%m%d-%H%M%S')

    # Random seed
    if seed is None:
        seed = np.random.randint(1e6) + 10000 * rank # different seeds for each process
    set_seed(seed, torch_deterministic=True)

    env_name = env_config.env_name
    num_envs = env_config.num_envs

    if env_name in ['cartpole', 'acrobot', 'mountaincar', 'lunarlander', 'carracing', 'hopper', 'invertedpendulum', 'inverteddoublependulum',
                    'halfcheetah', 'walker2d', 'humanoid', 'humanoidstandup', 'reacher', 'swimmer', 'ant']:
        timesteps_per_proc = env_config.timesteps_per_proc

    elif 'atari' not in env_name:
        env_name, distribution_mode, start_level, num_levels = env_name.split('-')
        start_level, num_levels = int(start_level), int(num_levels)

        if distribution_mode == 'easy':
            timesteps_per_proc = env_config.timesteps_per_proc_easy
        elif distribution_mode == 'hard':
            timesteps_per_proc = env_config.timesteps_per_proc_hard

    if rank==0:
        if env_name in {'cartpole', 'acrobot', 'mountaincar', 'lunarlander', 'carracing', 'hopper', 'invertedpendulum', 'inverteddoublependulum',
                        'halfcheetah', 'walker2d', 'humanoid', 'humanoidstandup', 'reacher', 'swimmer', 'ant'}:
            log_dir = f"logs/{algo}.karzmarz_{algo_config.is_karzmarz}.{nets_config.type}.a{nets_config.a_hidden_size}x{nets_config.a_num_layers}x{nets_config.a_dropout}e{algo_config.pi_epochs}x{algo_config.pi_minibatches}.c{nets_config.c_hidden_size}x{nets_config.c_num_layers}x{nets_config.c_dropout}e{algo_config.v_epochs}x{algo_config.v_minibatches}.{algo_config.grad}_{algo_config.post_grad}_{algo_config.max_grad_norm}.{algo_config.sigma_type}.damping_{algo_config.cg_damping}.lr_pi_{algo_config.lr_pi}/{env_name}.{time_now}_{seed}"
        else:
            log_dir = f"logs/{algo}.karzmarz_{algo_config.is_karzmarz}.{nets_config.type}{'_bn' if nets_config.with_bn else ''}_{algo_config.pi_epochs}epoch.damping_{algo_config.cg_damping}.lr_pi_{algo_config.lr_pi}/{env_config.env_name}.{time_now}_{seed}"

        format_strs = ['csv', 'stdout'] 
        logger.configure(dir=log_dir, format_strs=format_strs)
        writer = SummaryWriter(log_dir=log_dir)
    else:
        log_dir = None
        writer = None
    
    if rank==0:
        logger.info("creating environment")

    if env_name in ['cartpole', 'acrobot', 'mountaincar', 'lunarlander', 'carracing', 'invertedpendulum', 'inverteddoublependulum',
                      'hopper', 'halfcheetah', 'walker2d', 'humanoid', 'humanoidstandup', 'reacher', 'swimmer', 'ant']:
        from stable_baselines3.common.env_util import make_vec_env
        tag_name = {'cartpole': 'CartPole-v1', 'acrobot': 'Acrobot-v1', 'mountaincar': 'MountainCar-v0', 
                    'lunarlander': 'LunarLander-v2', 'carracing': 'CarRacing-v2', 'invertedpendulum': 'InvertedPendulum-v4',
                    'inverteddoublependulum': 'InvertedDoublePendulum-v4',
                    'hopper': 'Hopper-v4', 'halfcheetah': 'HalfCheetah-v4', 'walker2d': 'Walker2d-v4', 
                    'humanoid': 'Humanoid-v4', 'humanoidstandup': 'HumanoidStandup-v4', 'reacher': 'Reacher-v4', 
                    'swimmer': 'Swimmer-v3', 'ant': 'Ant-v4'}
        
        venv = make_vec_env(tag_name[env_name], n_envs=num_envs, env_kwargs={'continuous': False} if env_name == 'carracing' else {})

    else:
        raise NotImplementedError

    if device == -1:
        if torch.cuda.is_available(): # i.e. for NVIDIA GPUs
            device_type = "cuda"
        else:
            device_type = "cpu"
        
        device = torch.device(device_type) # Select best available device
    else:
        assert device >= 0
        device = f"cuda:{device}"

    obs_space = venv.observation_space

    # Create actor-critic module
    if nets_config.type == 'mlp':
        kwargs = {'device': device}
        fn_neural_nets, preprocess = build_mlp(obs_space, **kwargs)
        obs_shape = obs_space.shape

    else: 
        raise NotImplementedError

    act_num, act_dim = None, None
    try:
        act_num = venv.action_space.n
    except AttributeError:
        act_dim = venv.action_space.shape[0]

    actor_critic = ActorCritic(fn_neural_nets, obs_shape, nets_config=nets_config, n_actions=act_num, 
                            dim_actions=act_dim, with_popart=algo_config.with_popart, 
                            sigma_type=algo_config.sigma_type, device=device).to(device)

    venv = VecNormalize(venv=venv, norm_ret=env_config.norm_ret, obs_preprocess=preprocess) # img transform and reward normalization

    if rank==0:
        logger.info(f'Running on device: {device}')
        logger.info(f"training...")

        # Count variables
        var_counts = count_vars(actor_critic)
        logger.log(f'\nNumber of parameters: {var_counts}\n')

        # yaml.dump(args, open( f"{log_dir}/args.yaml", 'w' ))
        config = {'algo_config': algo_config.__dict__, 
                'env_config': env_config.__dict__, 
                'nets_config': nets_config.__dict__, 
                'log_config': log_config.__dict__}

        yaml.dump(config, open( f"{log_dir}/config.yaml", 'w' ))

    learn(world_size, algo, actor_critic, writer, venv, device,
          total_timesteps=timesteps_per_proc, nsteps=env_config.nsteps, 
          algo_config=algo_config, log_config=log_config, log_dir=log_dir)

def main():
    parser = argparse.ArgumentParser(description='Process procgen training arguments.')
    parser.add_argument('--config', type=str, default='true_mlp.yaml')
    parser.add_argument('--device', type=int, default=-1) # -1: use any available device
    parser.add_argument('--env_name', type=str, default=None) # -1: use any available device
    parser.add_argument('--n_proc', type=int, default=1) # distributed training: number of processes
    parser.add_argument('--port_num', type=int, default=29500) # distributed training: number of processes
    parser.add_argument('--a_dropout', type=float, default=None) # distributed training: number of processes
    parser.add_argument('--a_hidden_size', type=int, default=None) # distributed training: number of processes
    parser.add_argument('--a_num_layers', type=int, default=None) # distributed training: number of processes
    parser.add_argument('--c_dropout', type=float, default=None) # distributed training: number of processes
    parser.add_argument('--c_hidden_size', type=int, default=None) # distributed training: number of processes
    parser.add_argument('--c_num_layers', type=int, default=None) # distributed training: number of processes
    parser.add_argument('--norm_obj', type=str, default=None) # distributed training: number of processes
    parser.add_argument('--optimizer', type=str, default=None) # distributed training: number of processes
    parser.add_argument('--sigma_type', type=str, default=None, choices=['vector', 'mu_shared', 'separate', 'linear']) 
    parser.add_argument('--cg_damping', type=float, default=None) # distributed training: number of processes
    parser.add_argument('--pi_epochs', type=int, default=None) # distributed training: number of processes
    parser.add_argument('--timesteps_per_proc', type=int, default=None) # distributed training: number of processes
    parser.add_argument('--lr_pi', type=float, default=None) # distributed training: number of processes
    parser.add_argument('--grad', type=str, default=None) # distributed training: number of processes
    parser.add_argument('--post_grad', type=str, default=None) # distributed training: number of processes
    parser.add_argument('--seed', type=int, default=None) 

    args = parser.parse_args()

    with open(f'configs/{args.config}') as fin:
        config = yaml.safe_load(fin)

    algo = config['algo']
    algo_config = types.SimpleNamespace(**config['algo_config'])
    env_config = types.SimpleNamespace(**config['env_config'])
    nets_config = types.SimpleNamespace(**config['nets_config'])
    log_config = types.SimpleNamespace(**config['log_config'])

    if args.env_name is not None:
        env_config.env_name = args.env_name

    if args.a_hidden_size is not None:
        nets_config.a_hidden_size = args.a_hidden_size
    if args.a_num_layers is not None:
        nets_config.a_num_layers = args.a_num_layers
    if args.a_dropout is not None:
        nets_config.a_dropout = args.a_dropout
    if args.c_hidden_size is not None:
        nets_config.c_hidden_size = args.c_hidden_size
    if args.c_num_layers is not None:
        nets_config.c_num_layers = args.c_num_layers
    if args.c_dropout is not None:
        nets_config.c_dropout = args.c_dropout

    if args.optimizer is not None:
        algo_config.optimizer = args.optimizer

    if args.sigma_type is not None:
        algo_config.sigma_type = args.sigma_type

    if args.norm_obj is not None:
        algo_config.norm_obj = args.norm_obj

    if args.cg_damping is not None:
        algo_config.cg_damping = args.cg_damping

    if args.pi_epochs is not None:
        algo_config.pi_epochs = args.pi_epochs

    if args.lr_pi is not None:
        algo_config.lr_pi = args.lr_pi

    if args.grad is not None:
        algo_config.grad = args.grad

    if args.post_grad is not None:
        algo_config.post_grad = args.post_grad

    if args.timesteps_per_proc is not None:
        env_config.timesteps_per_proc = args.timesteps_per_proc

    if args.n_proc > 1:
        # multiple nodes
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(args.port_num)

        mp.spawn(train_fn, args=(args.n_proc, algo, args.seed, algo_config, env_config, nets_config, log_config, args.device),
                        nprocs=args.n_proc, # INFO: for TPU, either 1 or the maximum number of TPU chips
                        join=True)

    else:
        train_fn(0, args.n_proc, algo, args.seed, algo_config, env_config, nets_config, log_config, args.device)

if __name__ == '__main__':
    main()
