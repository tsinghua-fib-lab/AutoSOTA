import os
import numpy as np
import torch
import torch.nn as nn
from noise import Noise, get_noise
from sampling import batched_sample_tau_leaping
from model import EMA, ControllerWrapper
from utils import Logger, compute_ess
from energy import get_log_discrete_score, eval_samples, get_log_prob
import wandb
from omegaconf import OmegaConf


def bregman_div(a: torch.Tensor, b: torch.Tensor, func: str) -> torch.Tensor:
    r'''
    Computing the elementwise Bregman divergence
    D_phi(a||b) = phi(a) - phi(b) - (a - b) * phi'(b).

    Args:
        a: shape (...,)
        b: shape (...,)
        func: The convex function to use. Options are:
            - 'tlogt': phi(t) = t * log(t), D_phi(a||b) = a * log(a/b) - a + b,
            - '-logt': phi(t) = -log(t), D_phi(a||b) = a/b - log(a/b) - 1,
            - 't2': phi(t) = t^2, D_phi(a||b) = (a - b)^2.

    Returns:
        shape (...,)
    '''
    if func == 'tlogt':
        return a * (a / b).clamp(min=1e-10).log() - a + b
    elif func == '-logt':
        return a / b - (a / b).clamp(min=1e-10).log() - 1
    elif func == 't2':
        return (a - b) ** 2
    else:
        raise ValueError(f'Unknown function {func} for Bregman divergence.')


def sample_ref_bridge(t: torch.Tensor, x0: torch.Tensor, x1: torch.Tensor, vocab_size: int, noise: Noise) -> torch.Tensor:
    r'''
    For each pair of (t, x0, x1), draw one sample from p^r_{t|0,1}(x|x0,x1).
    See paper for formula.

    Args:
        t: time, shape (..., 1), 0 <= t <= 1
        x0: start state, shape (..., D), values in range(N)
        x1: end state, shape (..., D), values in range(N)
        vocab_size: size of the vocabulary (N)
    Returns:
        Samples, shape (..., D), values in range(N)
    '''
    zeros, ones = torch.zeros_like(t), torch.ones_like(t)

    def A(u, v):
        exp_minus_gamma = (-noise.integral_noise(u, v)).exp()
        return (1 - exp_minus_gamma) / vocab_size # (..., 1)
    def B(u, v):
        exp_minus_gamma = (-noise.integral_noise(u, v)).exp()
        return (1 + (vocab_size - 1) * exp_minus_gamma) / vocab_size
    
    A0t = A(zeros, t); B0t = B(zeros, t)  # (..., 1)
    At1 = A(t, ones); Bt1 = B(t, ones)
    A01 = A(zeros, ones); B01 = B(zeros, ones)
    eq_mask = (x0 == x1)[..., None]  # (..., D, 1)

    n_values = torch.arange(vocab_size, device=t.device)

    n_eq_x0 = x0[..., None] == n_values  # (..., D, N)
    n_eq_x1 = x1[..., None] == n_values  # (..., D, N)
    
    probs = (~eq_mask) * (
        (A0t * At1 / A01)[..., None] * ((~n_eq_x0) & (~n_eq_x1)) +
        (B0t * At1 / A01)[..., None] * n_eq_x0 +
        (A0t * Bt1 / A01)[..., None] * n_eq_x1
    ) + eq_mask * (
        (A0t * At1 / B01)[..., None] * (~n_eq_x0) +
        (B0t * Bt1 / B01)[..., None] * n_eq_x0
    ) # (..., D, N)
    # print(torch.allclose(probs.sum(dim=-1), torch.ones_like(x0, dtype=probs.dtype)))
    return torch.distributions.Categorical(probs=probs).sample()


def ref_trans_log_p_diff(s: torch.Tensor, t: torch.Tensor, x: torch.Tensor, y: torch.Tensor,
                         noise: Noise, vocab_size: int, forward: bool) -> torch.Tensor:
    r'''
    Compute all reference transition log probabilities needed for loss computation.

    Args:
        s: start time, shape (..., 1)
        t: end time, shape (..., 1)
        x: start state, shape (..., D), values in range(N)
        y: end state, shape (..., D), values in range(N)
        noise: Noise object
        vocab_size: size of the discrete state space N
        forward: specifies whether to compute forward or backward transition log probabilities, see below.
    Returns:
        A matrix of transition log probabilities, shape (..., D, N),

        If forward (for computing loss in corrector):
            the [..., d, n]-th entry corresponds to
                log p^r_{t|s}(y^{d<-n}|x) / p^r_{t|s}(y|x)
                = (1_{n != x^d} - 1_{y^d != x^d}) * log C(s, t)

        Else (for computing loss in controller):
            the [..., d, n]-th entry corresponds to
                log p^r_{t|s}(y|x^{d<-n}) / p^r_{t|s}(y|x)
                = (1_{n != y^d} - 1_{x^d != y^d}) * log C(s, t)
    '''
    exp_minus_gamma = (-noise.integral_noise(s, t)).exp()  # (..., 1)
    c = (1 - exp_minus_gamma) / (1 + (vocab_size - 1) * exp_minus_gamma)  # (..., 1)
    log_c = c.log()[..., None]  # (..., 1, 1)
    indicator1 = (x if forward else y)[..., None] != torch.arange(vocab_size, device=x.device)  # (..., D, N)
    indicator2 = (x != y)[..., None]  # (..., D, 1)
    return (indicator1.float() - indicator2.float()) * log_c


def get_t_weights(losses: torch.Tensor, func: str, t: torch.Tensor = None,
                  noise: Noise = None, vocab_size: int = None) -> torch.Tensor:
    r'''
    Compute time weights for the loss, assuming losses are computed at uniformly random t's.

    Args:
        losses: shape (bsz,)
        func: method to weight the loss
        t: shape (bsz, 1), optional time values corresponding to the losses, required if func is 'gamma'
        noise: Noise object, required if func is 'gamma'
        vocab_size: size of the discrete state space N, required if func is 'gamma'
    
    Returns:
        t_weights: shape (bsz,) or scalar 1
    '''
    if func == 'unif':
        t_weights = 1
    elif func == 'gamma':
        t_weights = noise.rate_noise(t)[..., 0] / vocab_size
    elif func == 'ada':
        # when all losses are identical, this gives uniform weights
        t_weights = 1 / (losses.detach().abs() + 1e-10)
        t_weights = len(losses) * t_weights / t_weights.sum()
    else:
        raise NotImplementedError(f'Unknown func {func}.')
    return t_weights


def get_path_weights(log_rnd: torch.Tensor, num_repl: int, apply_softmax: bool = True) -> torch.Tensor:
    r'''
    Compute path weights from log-RND values.

    Args:
        log_rnd: shape (B,) if provided, otherwise return uniform weights
        num_repl: number of t samples per (x0, x1) pair
        apply_softmax: whether to apply softmax to normalize the log_rnd

    Returns:
        path_weights: shape (B * num_repl,) or scalar 1
    '''
    if log_rnd is None:
        return 1
    if apply_softmax:
        # when all log_rnd are identical, this gives uniform weights
        return log_rnd.shape[0] * log_rnd.softmax(dim=0).repeat(num_repl)  # (B * num_repl,)
    else:
        return log_rnd.exp().repeat(num_repl)  # (B * num_repl,)


def loss_controller_dm(x0: torch.Tensor, x1: torch.Tensor, controller: nn.Module, noise: Noise,
                       vocab_size: int, num_repl: int = 1, cond: torch.Tensor = None,
                       breg_func: str = 'tlogt', t_weight_func: str = 'unif',
                       eps: float = 0.01,
                       log_rnd: torch.Tensor = None, log_rnd_apply_softmax: bool = True
                       ) -> torch.Tensor:
    r'''
    Compute the denoising matching loss for the controller varPhi (ctrl-DM).
    
    Args:
        x0: start state, shape (B, D), values in range(N)
        x1: end state, shape (B, D), values in range(N)
        controller: model for log varPhi_t(x), takes inputs (x, t, cond) and outputs (B, D, N)
        noise: Noise object
        vocab_size: size of the discrete state space N
        num_repl: number of t's to sample for each (x0, x1) pair
        cond: optional conditioning information for the model
        breg_func: Bregman divergence function to use
        t_weight_func: function to weight the loss by t
        eps: sampling time uniformly from [0, 1 - eps]
        log_rnd: optional path log-RND log dp*/dp_u (x_{[0, 1]}), shape (B,)
        log_rnd_apply_softmax: whether to apply softmax to normalize the log_rnd

    Returns:
        loss: scalar tensor
    '''
    x0, x1 = x0.repeat(num_repl, 1), x1.repeat(num_repl, 1)  # (B * num_repl, D)
    t = torch.rand(x0.shape[0], 1, device=x0.device) * (1 - eps)  # (B * num_repl, 1)
    xt = sample_ref_bridge(t, x0, x1, vocab_size, noise)  # (B * num_repl, D)
    log_p_diff = ref_trans_log_p_diff(t, torch.ones_like(t), xt, x1, noise, vocab_size, forward=False)  # (B * num_repl, D, N)
    log_varPhi = controller(xt, t, cond=cond) # (B * num_repl, D, N)
    divg = bregman_div(log_p_diff.exp(), log_varPhi.exp(), func=breg_func)  # (B * num_repl, D, N)

    mask = xt[..., None] != torch.arange(vocab_size).to(x0.device)  # (B * num_repl, D, N)
    divg = (divg * mask).sum(dim=(1, 2))  # (B * num_repl,)
    t_weights = get_t_weights(divg, func=t_weight_func, t=t, noise=noise, vocab_size=vocab_size)  # (B * num_repl,) or scalar 1
    path_weights = get_path_weights(log_rnd, num_repl, apply_softmax=log_rnd_apply_softmax)  # (B * num_repl,) or scalar 1
    return (divg * t_weights * path_weights).mean() / num_repl / x0.shape[-1] / vocab_size


def loss_controller_pin(x1: torch.Tensor, controller: nn.Module, corrector: nn.Module, log_discrete_score: callable,
                        cond: torch.Tensor = None, breg_func: str = 't2',
                        ) -> torch.Tensor:
    r'''
    Compute the pinning loss for the controller varPhi given the corrector varPhi and the target log discrete score.

    Args:
        x1: end state, shape (B, D), values in range(N)
        controller: model for log varPhi_t(x), takes inputs (x, t, cond) and outputs (B, D, N)
        corrector: model for log hat_varPhi(x), takes inputs (x, cond) and outputs (B, D, N)
        log_discrete_score: function to compute the target log discrete score
        cond: optional conditioning information for the model
        breg_func: Bregman divergence function to use
    
    Returns:
        loss: scalar tensor
    '''
    ones = torch.ones((x1.shape[0], 1), device=x1.device)  # (B, 1)
    log_varPhi = controller(x1, ones, cond=cond)  # (B, D, N)
    log_target_discrete_score = log_discrete_score(x1, cond=cond)  # (B, D, N)
    with torch.no_grad():
        log_hat_varPhi = corrector(x1, cond=cond)  # (B, D, N)
    divg = bregman_div((log_target_discrete_score - log_hat_varPhi).exp(), log_varPhi.exp(), func=breg_func)  # (B, D, N)
    vocab_size = divg.shape[-1]
    mask = x1[..., None] != torch.arange(vocab_size).to(x1.device)  # (B, D, N)
    divg = (divg * mask).sum(dim=(1, 2))  # (B,)
    return divg.mean() / x1.shape[-1] / vocab_size


def loss_controller_tm(x0: torch.Tensor, x1: torch.Tensor, controller: nn.Module, corrector: nn.Module,
                       log_discrete_score: callable, noise: Noise,
                       vocab_size: int, num_repl: int = 1, cond: torch.Tensor = None,
                       breg_func: str = 'tlogt', t_weight_func: str = 'unif',
                       log_rnd: torch.Tensor = None, log_rnd_apply_softmax: bool = True,
                       anneal_strength: float = None,
                       anneal_func: callable = lambda t: t ** 0.5,
                       ) -> torch.Tensor:
    r'''
    Compute the target matching loss for the controller varPhi (ctrl-TM).

    Args:
        x0: start state, shape (B, D), values in range(N)
        x1: end state, shape (B, D), values in range(N)
        controller: model for log varPhi_t(x), takes inputs (x, t, cond) and outputs (B, D, N)
        corrector: model for log hat_varPhi(x), takes inputs (x, cond) and outputs (B, D, N)
        log_discrete_score: function to compute the target log discrete score
        noise: Noise object
        vocab_size: size of the discrete state space N
        num_repl: number of t's to sample for each (x0, x1) pair
        cond: optional conditioning information for the model
        breg_func: Bregman divergence function to use
        t_weight_func: function to weight the loss by t
        log_rnd: optional path log-RND log dp*/dp_u (x_{[0, 1]}), shape (B,)
        log_rnd_apply_softmax: whether to apply softmax to normalize the log_rnd
        anneal_strength: optional annealing strength in [0, 1] to temper the target log discrete score
        anneal_func: function to compute the annealing multiplier from anneal_strength

    Returns:
        loss: scalar tensor
    '''
    x0, x1 = x0.repeat(num_repl, 1), x1.repeat(num_repl, 1)  # (B * num_repl, D)
    t = torch.rand(x0.shape[0], 1, device=x0.device)  # (B * num_repl, 1)
    xt = sample_ref_bridge(t, x0, x1, vocab_size, noise)  # (B * num_repl, D)

    log_varPhi = controller(xt, t, cond=cond) # (B * num_repl, D, N)
    log_target_discrete_score = log_discrete_score(x1, cond=cond)  # (B * num_repl, D, N)
    if anneal_strength is not None and anneal_func is not None:
        log_target_discrete_score = anneal_func(anneal_strength) * log_target_discrete_score
    with torch.no_grad():
        log_hat_varPhi = corrector(x1, cond=cond)  # (B * num_repl, D, N)
    log_varPhi1_tgt = log_target_discrete_score - log_hat_varPhi # (B * num_repl, D, N)

    shifted_states = (x1 - xt)[..., None] + torch.arange(vocab_size, device=x0.device)  # (B * num_repl, D, N)
    shifted_states = shifted_states % vocab_size  # Z -> range(N)
    shifted_log_varPhi1_tgt = torch.gather(input=log_varPhi1_tgt, dim=2, index=shifted_states)  # (B * num_repl, D, N)

    divg = bregman_div(shifted_log_varPhi1_tgt.exp(), log_varPhi.exp(), func=breg_func)  # (B * num_repl, D, N)
    
    mask = xt[..., None] != torch.arange(vocab_size).to(x0.device)  # (B * num_repl, D, N)
    divg = (divg * mask).sum(dim=(1, 2))  # (B * num_repl,)
    t_weights = get_t_weights(divg, func=t_weight_func, t=t, noise=noise, vocab_size=vocab_size)  # (B * num_repl,) or scalar 1
    path_weights = get_path_weights(log_rnd, num_repl, apply_softmax=log_rnd_apply_softmax)  # (B * num_repl,) or scalar 1
    return (divg * t_weights * path_weights).mean() / num_repl / x0.shape[-1] / vocab_size


def loss_corrector_dm(x0: torch.Tensor, x1: torch.Tensor, corrector: nn.Module, noise: Noise,
                      vocab_size: int, num_repl: int = 1, cond: torch.Tensor = None,
                      breg_func: str = 'tlogt', t_weight_func: str = 'unif',
                      eps: float = 0.01,
                      log_rnd: torch.Tensor = None, log_rnd_apply_softmax: bool = True
                      ) -> torch.Tensor:
    r'''
    Compute the denoising matching loss for the corrector hat varPhi (corr-DM).
    
    Args:
        x0: start state, shape (B, D), values in range(N)
        x1: end state, shape (B, D), values in range(N)
        corrector: model for log hat_varPhi(x), takes inputs (x, cond) and outputs (B, D, N)
        noise: Noise object
        vocab_size: size of the discrete state space N
        num_repl: number of t's to sample for each (x0, x1) pair
        cond: optional conditioning information for the model
        breg_func: Bregman divergence function to use
        t_weight_func: function to weight the loss by t
        eps: sampling time uniformly from [0, 1 - eps]
        log_rnd: optional path log-RND log dp*/dp_u (x_{[0, 1]}), shape (B,)
        log_rnd_apply_softmax: whether to apply softmax to normalize the log_rnd
    
    Returns:
        loss: scalar tensor
    '''
    if eps == 1.0: num_repl = 1  # t is always zero, no need to replicate
    x0, x1 = x0.repeat(num_repl, 1), x1.repeat(num_repl, 1)  # (B * num_repl, D)
    t = torch.rand(x0.shape[0], 1, device=x0.device) * (1 - eps)  # (B * num_repl, 1)
    xt = sample_ref_bridge(t, x0, x1, vocab_size, noise)  # (B * num_repl, D)
    log_p_diff = ref_trans_log_p_diff(t, torch.ones_like(t), xt, x1, noise, vocab_size, forward=True)  # (B * num_repl, D, N)
    log_hat_varPhi = corrector(x1, cond=cond) # (B * num_repl, D, N)
    divg = bregman_div(log_p_diff.exp(), log_hat_varPhi.exp(), func=breg_func) # (B * num_repl, D, N)
    mask = x1[..., None] != torch.arange(vocab_size).to(x0.device)  # (B * num_repl, D, N)
    divg = (divg * mask).sum(dim=(1, 2))  # (B * num_repl,)
    t_weights = get_t_weights(divg, func=t_weight_func, t=t, noise=noise, vocab_size=vocab_size)  # (B * num_repl,) or scalar 1
    path_weights = get_path_weights(log_rnd, num_repl, apply_softmax=log_rnd_apply_softmax)  # (B * num_repl,) or scalar 1
    return (divg * t_weights * path_weights).mean() / num_repl / x0.shape[-1] / vocab_size


def loss_corrector_tm(x0: torch.Tensor, x1: torch.Tensor, controller: nn.Module, corrector: nn.Module,
                      vocab_size: int, cond: torch.Tensor = None,
                      init_log_discrete_score: callable = None,
                      breg_func: str = 'tlogt',
                      log_rnd: torch.Tensor = None, log_rnd_apply_softmax: bool = True
                      ) -> torch.Tensor:
    r'''
    Compute the target matching loss for the corrector hat varPhi (corr-TM).
    
    Args:
        x0: start state, shape (B, D), values in range(N)
        x1: end state, shape (B, D), values in range(N)
        controller: model for log varPhi_t(x), takes inputs (x, t, cond) and outputs (B, D, N)
        corrector: model for log hat_varPhi(x), takes inputs (x, cond) and outputs (B, D, N)
        vocab_size: size of the discrete state space N
        cond: optional conditioning information for the model
        init_log_discrete_score: function to compute the target log discrete score for initial distribution
        breg_func: Bregman divergence function to use
        log_rnd: optional path log-RND log dp*/dp_u (x_{[0, 1]}), shape (B,)
        log_rnd_apply_softmax: whether to apply softmax to normalize the log_rnd

    Returns:
        loss: scalar tensor
    '''
    zeros = torch.zeros(x0.shape[0], 1).to(x0.device)  # (B, 1)
    
    log_hat_varPhi = corrector(x1, cond=cond) # (B, D, N)

    shifted_states = (x0 - x1)[..., None] + torch.arange(vocab_size, device=x0.device)  # (B, D, N)
    shifted_states = shifted_states % vocab_size  # Z -> range(N)
    with torch.no_grad():
        log_varPhi = controller(x0, zeros, cond=cond)  # (B, D, N)
    log_init_discrete_score = 0 if init_log_discrete_score is None else init_log_discrete_score(x0, cond=cond)  # (B, D, N)
    log_hat_varPhi_tgt = log_init_discrete_score - log_varPhi  # (B, D, N)
    shifted_log_hat_varPhi_tgt = torch.gather(log_hat_varPhi_tgt, dim=2, index=shifted_states)  # (B, D, N) 
    
    divg = bregman_div(shifted_log_hat_varPhi_tgt.exp(), log_hat_varPhi.exp(), func=breg_func) # (B, D, N)
    
    mask = x1[..., None] != torch.arange(vocab_size).to(x0.device)  # (B, D, N)
    divg = (divg * mask).sum(dim=(1, 2))  # (B,)
    path_weights = get_path_weights(log_rnd, num_repl=1, apply_softmax=log_rnd_apply_softmax)  # (B,) or scalar 1
    return (divg * path_weights).mean() / x0.shape[-1] / vocab_size


def zero_corrector(x: torch.Tensor, cond: torch.Tensor = None) -> torch.Tensor:
    r'''
    A handler function that always returns zero corrector output.

    Args:
        x: input state, shape (B, D), values in range(N)
        cond: optional conditioning information for the model
    
    Returns:
        out: zero tensor of shape (B, D, N)
    '''
    return torch.zeros((x.shape[0], x.shape[1], 1), device=x.device)


def train(controller: nn.Module, corrector: nn.Module,
          optim_controller: torch.optim.Optimizer, optim_corrector: torch.optim.Optimizer,
          ema_controller: EMA, ema_corrector: EMA,
          args, logger: Logger):
    noise = get_noise(args)
    log_discrete_score = get_log_discrete_score(args)
    log_prob = get_log_prob(args)
    gt_sample_dataset = None
    if args.get('gt_sample_path', None) is not None:
        assert os.path.exists(args.gt_sample_path)
        gt_sample_dataset = torch.from_numpy(np.load(args.gt_sample_path)).to(
            dtype=torch.long, device=args.device).view(-1, args.seq_len)
        logger.info(f'Loaded {gt_sample_dataset.shape[0]} initial samples from {args.gt_sample_path}.')

    if args.controller_reparam: # the controller depends on corrector and log discrete score, deprecated
        controller_wrapper = ControllerWrapper(log_discrete_score, controller, corrector)
    else:
        controller_wrapper = controller
    if args.memoryless:  # \hat\varphi_1 \propto p^r_1 = unif, no need to train corrector
        args.num_steps.corrector = 0
        assert args.noise.type == 'log_linear' and args.noise.alpha <= 0.01, \
            'Memoryless only works with log-linear noise with near zero alpha'
        if args.loss.use_log_rnd and not args.loss.log_rnd_apply_softmax:
            args.loss.log_rnd_apply_softmax = True
            logger.info('Setting loss.log_rnd_apply_softmax to True for memoryless training.')
    if args.buffer_size < args.batch_size:
        args.buffer_size = args.batch_size
        logger.info(f'Setting buffer_size to batch_size {args.batch_size}.')

    init_log_discrete_score = init_sample_dataset = None
    if args.beta_init in [float('inf'), 'inf']:
        assert args.loss.corrector == 'dm', 'Corrector TM loss is not supported for beta_init = inf.'
    else:
        assert isinstance(args.beta_init, (float, int)) and args.beta_init >= 0, \
            'beta_init must be a non-negative number or inf.'
        _args = OmegaConf.merge(args); _args.beta = args.beta_init # create an independent copy
        init_log_discrete_score = get_log_discrete_score(_args) # for TM loss
        if args.beta_init > 0:
            assert args.init_sample_path is not None and os.path.exists(args.init_sample_path), \
                'init_sample_path must be provided and exist for 0 < beta_init < inf.'
            init_sample_dataset = torch.from_numpy(np.load(args.init_sample_path)).to(
                dtype=torch.long, device=args.device).view(-1, args.seq_len)
            if args.dist == 'ising':
                init_sample_dataset = (init_sample_dataset + 1) // 2  # -1/1 to 0/1
            logger.info(f'Loaded {init_sample_dataset.shape[0]} initial samples from {args.init_sample_path}.')
    
    if args.logging.use_wandb and args.device == 'cuda:0':
        wandb.define_metric("step_controller")
        wandb.define_metric("step_corrector")
        wandb.define_metric("controller/*", step_metric="step_controller")
        wandb.define_metric("corrector/*", step_metric="step_corrector")

    for stage in range(args.start_stage, args.num_stages):

        ##### 1. train controller #####
        controller.train(); corrector.eval()

        if args.optim.reset:
            optim_controller = torch.optim.AdamW(controller.parameters(), lr=args.optim.lr, weight_decay=args.optim.wd)
        if args.ema.reset:
            ema_controller = EMA(controller.parameters(), decay=args.ema.decay)

        logger.info(f'Stage {stage + 1}: Training controller...')

        if not args.controller_first and stage == args.start_stage:
            # skip controller training at the first stage if specified
            num_steps_controller = 0
            logger.info(f'Skipping controller training at stage {stage + 1}.')
        else:
            num_steps_controller = args.num_steps.controller

        for step in range(num_steps_controller):
            args.step_count_controller += 1; info = {'step_controller': args.step_count_controller}
            if step % args.resample_freq == 0:
                with torch.no_grad():
                    controller.eval(); corrector.eval()
                    ema_controller.store(controller.parameters())
                    ema_controller.copy_to(controller.parameters())
                    [x0, x1], sampling_info = batched_sample_tau_leaping(
                        rounds=1 if step > 0 else max(args.init_buffer_size // args.batch_size, 1),
                        batch_size=args.batch_size, steps=args.sampling_steps, args=args,
                        controller=controller_wrapper, noise=noise, cond=None, log_prob=log_prob,
                        init_sample_dataset=init_sample_dataset,
                        ) # (B, D)
                    log_rnd = sampling_info['log_dp_star_dp_u']  # (B,)
                    ema_controller.restore(controller.parameters())
                    controller.train(); corrector.eval()
                logger.info(f'Stage {stage + 1}, controller step {step + 1}: obtained {args.batch_size} pairs of (x0, x1).')

                info.update(eval_samples(x1, args, gt_sample_dataset))
                info['controller/num_jumps_per_dim'] = sampling_info['num_jumps_per_dim'].mean().item()
                info['controller/log_rnd'] = log_rnd.mean().item()
                info['controller/log_rnd_std'] = log_rnd.std().item()
                info['controller/ess'] = compute_ess(log_rnd)

                if step == 0:
                    x0_buffer, x1_buffer, log_rnd_buffer = x0, x1, log_rnd
                else:
                    x0_buffer = torch.cat([x0_buffer, x0], dim=0)[-args.buffer_size:]
                    x1_buffer = torch.cat([x1_buffer, x1], dim=0)[-args.buffer_size:]
                    log_rnd_buffer = torch.cat([log_rnd_buffer, log_rnd], dim=0)[-args.buffer_size:]

            idx = np.random.choice(x0_buffer.shape[0], args.batch_size, replace=False)
            x0, x1, log_rnd = x0_buffer[idx], x1_buffer[idx], log_rnd_buffer[idx]

            if args.loss.controller == 'dm':
                loss = loss_controller_dm(
                    x0, x1, controller_wrapper, noise, vocab_size=args.vocab_size,
                    num_repl=args.num_repl, cond=None,
                    breg_func=args.breg_func.controller,
                    t_weight_func=args.t_weight_func.controller,
                    eps=args.dm_eps,
                    log_rnd=log_rnd if args.loss.use_log_rnd else None,
                    log_rnd_apply_softmax=args.loss.log_rnd_apply_softmax,
                )
            elif args.loss.controller == 'tm':
                if args.memoryless or (args.zero_init_corrector and stage == args.start_stage):
                    corrector_to_use = zero_corrector
                else:
                    corrector_to_use = corrector
                loss = loss_controller_tm(
                    x0, x1, controller_wrapper, corrector_to_use, log_discrete_score, noise,
                    vocab_size=args.vocab_size, num_repl=args.num_repl, cond=None,
                    breg_func=args.breg_func.controller,
                    t_weight_func=args.t_weight_func.controller,
                    log_rnd=log_rnd if args.loss.use_log_rnd else None,
                    log_rnd_apply_softmax=args.loss.log_rnd_apply_softmax,
                    anneal_strength=args.step_count_controller / (args.num_stages * args.num_steps.controller),
                    anneal_func=args.anneal_func if args.anneal else None,
                )
            else:
                raise ValueError(f'Unknown controller loss {args.loss.controller}.')

            info['controller/loss'] = loss.item()
            loss.backward()
            if args.optim.clip_grad_norm > 0:
                grad_norm = nn.utils.clip_grad_norm_(controller.parameters(), max_norm=args.optim.clip_grad_norm)
                info['controller/grad_norm'] = grad_norm.item()
            elif args.optim.log_grad_norm_freq > 0 and (step + 1) % args.optim.log_grad_norm_freq == 0:
                grad_norm = nn.utils.clip_grad_norm_(controller.parameters(), max_norm=1e10)
                info['controller/grad_norm'] = grad_norm.item()
            optim_controller.step(); ema_controller.update(controller.parameters()); optim_controller.zero_grad()
            logger.log(info, step=args.step_count_controller + args.step_count_corrector) # total steps

        ema_controller.copy_to(controller.parameters())

        ##### 2. train corrector #####
        controller.eval(); corrector.train()

        if args.optim.reset:
            optim_corrector = torch.optim.AdamW(corrector.parameters(), lr=args.optim.lr, weight_decay=args.optim.wd)
        if args.ema.reset:
            ema_corrector = EMA(corrector.parameters(), decay=args.ema.decay)

        if controller_wrapper is not controller:
            ema_corrector.store(corrector.parameters(), idx=1)
            # store the parameters of corrector before training, as they will be used in the controller_wrapper

        for step in range(args.num_steps.corrector):
            args.step_count_corrector += 1; info = {'step_corrector': args.step_count_corrector}
            if step % args.resample_freq == 0:
                with torch.no_grad():
                    controller.eval(); corrector.eval()
                    if controller_wrapper is not controller:
                        ema_corrector.store(corrector.parameters())
                        ema_corrector.restore(corrector.parameters(), idx=1) # use the parameters before training
                    [x0, x1], sampling_info = batched_sample_tau_leaping(
                        rounds=1 if step > 0 else max(args.init_buffer_size // args.batch_size, 1),
                        batch_size=args.batch_size, steps=args.sampling_steps, args=args,
                        controller=controller_wrapper, noise=noise, cond=None, log_prob=log_prob,
                        init_sample_dataset=init_sample_dataset,
                        )  # (B, D)
                    log_rnd = sampling_info['log_dp_star_dp_u']  # (B,)
                    if controller_wrapper is not controller:
                        ema_corrector.restore(corrector.parameters())
                    controller.eval(); corrector.train()
                logger.info(f'Stage {stage + 1}, corrector step {step + 1}: obtained {args.batch_size} pairs of (x0, x1).')
                # we do not log sample evaluation here as the controller is not updated

                if step == 0:
                    x0_buffer, x1_buffer, log_rnd_buffer = x0, x1, log_rnd
                else:
                    x0_buffer = torch.cat([x0_buffer, x0], dim=0)[-args.buffer_size:]
                    x1_buffer = torch.cat([x1_buffer, x1], dim=0)[-args.buffer_size:]
                    log_rnd_buffer = torch.cat([log_rnd_buffer, log_rnd], dim=0)[-args.buffer_size:]

            idx = np.random.choice(x0_buffer.shape[0], args.batch_size, replace=False)
            x0, x1, log_rnd = x0_buffer[idx], x1_buffer[idx], log_rnd_buffer[idx]
            if args.loss.corrector == 'dm':
                loss = loss_corrector_dm(
                    x0, x1, corrector, noise, args.vocab_size, num_repl=args.num_repl, cond=None,
                    breg_func=args.breg_func.corrector,
                    t_weight_func=args.t_weight_func.corrector,
                    eps=args.dm_eps,
                    log_rnd=log_rnd if args.loss.use_log_rnd else None,
                    log_rnd_apply_softmax=args.loss.log_rnd_apply_softmax,
                    )
            elif args.loss.corrector == 'tm':
                loss = loss_corrector_tm(
                    x0, x1, controller_wrapper, corrector, vocab_size=args.vocab_size, cond=None,
                    init_log_discrete_score=init_log_discrete_score,
                    breg_func=args.breg_func.corrector,
                    log_rnd=log_rnd if args.loss.use_log_rnd else None,
                    log_rnd_apply_softmax=args.loss.log_rnd_apply_softmax,
                    )
            else:
                raise ValueError(f'Unknown corrector loss {args.loss.corrector}.')

            info['corrector/loss'] = loss.item()
            loss.backward()
            if args.optim.clip_grad_norm > 0:
                grad_norm = nn.utils.clip_grad_norm_(corrector.parameters(), max_norm=args.optim.clip_grad_norm)
                info['corrector/grad_norm'] = grad_norm.item()
            elif args.optim.log_grad_norm_freq > 0 and (step + 1) % args.optim.log_grad_norm_freq == 0:
                grad_norm = nn.utils.clip_grad_norm_(corrector.parameters(), max_norm=1e10)
                info['corrector/grad_norm'] = grad_norm.item()
            optim_corrector.step(); ema_corrector.update(corrector.parameters()); optim_corrector.zero_grad()
            logger.log(info, step=args.step_count_controller + args.step_count_corrector) # total steps

        ema_corrector.copy_to(corrector.parameters())

        if args.device == 'cuda:0' and ((stage + 1) % args.save_stage_freq == 0 or stage + 1 == args.num_stages):
            save_dir = os.path.join(args.logging.dir, f'ckpt{stage+1}.pth')
            _controller = controller.module if hasattr(controller, 'module') else controller
            _corrector = corrector.module if hasattr(corrector, 'module') else corrector
            torch.save({
                'controller_state_dict': _controller.state_dict(),
                'corrector_state_dict': _corrector.state_dict(),
                'optim_controller_state_dict': optim_controller.state_dict(),
                'optim_corrector_state_dict': optim_corrector.state_dict(),
                'ema_controller_state_dict': ema_controller.state_dict(),
                'ema_corrector_state_dict': ema_corrector.state_dict(),
                'args': args,
            }, save_dir)
            logger.info(f'Stage {stage + 1}: Saved model checkpoint to {save_dir}.')
