from diffusion_SDE.schedule import marginal_prob_std
from diffusion_SDE.model import ScoreNet
from diffusion_SDE.dpm_solver_pytorch import expand_dims
from search.configs import Arguments
import d4rl
import gym
import torch
import functools
import os

def build_model(device="cuda", args=None):
    if args is None:
        raise ValueError("Arguments must be provided to build the model.")
    args.device = device
    # args.env = "halfcheetah-medium-expert-v2"
    # args.actor_load_path = "models_rl/halfcheetah-medium-expert-v20large_actor/behavior_ckpt600.pth"
    # args.q_load_path = "models_rl/halfcheetah-medium-expert-v20large_actor/critic_ckpt100.pth"
    env = gym.make(args.env)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    # args.writer = writer
    # args.s = [0.0, 1.0, 2.0, 3.0, 5.0, 8.0, 10.0]
    marginal_prob_std_fn = functools.partial(marginal_prob_std, device=args.device)
    args.marginal_prob_std_fn = marginal_prob_std_fn
    score_model= ScoreNet(input_dim=state_dim+action_dim, output_dim=action_dim, marginal_prob_std=marginal_prob_std_fn, args=args).to(args.device)
    score_model.q[0].to(args.device)
    actor_ckpt = torch.load(args.actor_load_path, map_location=args.device)
    score_model.load_state_dict(actor_ckpt)
    q_ckpt = torch.load(args.q_load_path, map_location=args.device)
    score_model.q[0].load_state_dict(q_ckpt, strict=False)
    return score_model


class DDIMSampler:
    def __init__(self, args:Arguments=None, model: ScoreNet=None):
        if args is None:
            args = Arguments()
        self.args = args
        self.device = torch.device(args.device)
        if model is None:
            self.model = build_model(device=self.device)
        else:
            self.model = model
        self.model.q[0].guidance_scale = args.guidance_strength  ## disable time-dependent guidance
        # self.args.inference_steps = 15
        self.build_diffusion()
    

    def build_diffusion(self):
        self.dpm_solver = self.model.dpm_solver
        self.noise_schedule = self.model.noise_schedule

        t_0 = 1. / self.dpm_solver.noise_schedule.total_N 
        t_T = self.dpm_solver.noise_schedule.T 
        device = self.args.device
        steps = self.args.inference_steps
        order = 1
        skip_type = "time_uniform"
        orders = self.dpm_solver.get_orders_for_singlestep_solver(steps=steps, order=order)
        self.timesteps = self.dpm_solver.get_time_steps(skip_type=skip_type, t_T=t_T, t_0=t_0, N=steps, device=device)
        self.alpha_prod_ts = [self.noise_schedule.marginal_alpha(t) for t in self.timesteps[:-1]]
        self.alpha_prod_t_prevs = [self.noise_schedule.marginal_alpha(t) for t in self.timesteps[1:]]
        self.alpha_prod_ts = torch.tensor(self.alpha_prod_ts, device=device)
        self.alpha_prod_t_prevs = torch.tensor(self.alpha_prod_t_prevs, device=device)


    def solver_update(self, x, vec_s, vec_t, solver_type='dpm_solver', order=1):
        return self.dpm_solver.singlestep_dpm_solver_update(x, vec_s, vec_t, solver_type=solver_type, order=order, return_intermediate=True)
    
    def _predict_xt(self, xs, s, t):
        assert torch.all(s < t), "All elements of s must be smaller than t"
        alpha_s = self.noise_schedule.marginal_alpha(s)
        alpha_t = self.noise_schedule.marginal_alpha(t)
        sigma_s = self.noise_schedule.marginal_std(s)
        sigma_t = self.noise_schedule.marginal_std(t)
        dims = xs.dim()
        mean_coef = (alpha_s / alpha_t) 
        std = (sigma_t ** 2 - alpha_t ** 2  * (sigma_s ** 2 / alpha_s ** 2)) ** 0.5
        return expand_dims(mean_coef, dims) * xs + expand_dims(std, dims) * torch.randn_like(xs)
    
    def guide_step(self, x, i):
        # self.args.recur_steps = 4
        for recur_step in range(self.args.recur_steps):
            vec_s, vec_t = self.timesteps[i].expand(x.shape[0]), self.timesteps[i + 1].expand(x.shape[0])
            x_t, x0 = self.solver_update(x, vec_s, vec_t)
            x = self._predict_xt(x_t, vec_t, vec_s)
        
        return x_t

    @torch.enable_grad()
    def get_guidance(self, x_need_grad, func=lambda x: x, post_process=lambda x: x, return_logp=True, check_grad=False):
        if check_grad:
            assert x_need_grad.requires_grad, "x_need_grad must require grad"
        x = func(x_need_grad)
        x = post_process(x)
        q0 = self.model.calculateQ(s=None, a=x)
        if return_logp:
            return q0
        grad = torch.autograd.grad(q0.sum(), x_need_grad)[0]
        return grad.detach()
        
    @torch.no_grad()
    def get_reward(
        self,
        actions: torch.Tensor,
        states: torch.Tensor | None = None,
        repeat_factor: int | None = None,
    ) -> torch.Tensor:
        """
        Return Q(s,a) as reward.

        actions: (..., A) -> flatten to (B, A)
        states:  optional (..., S) -> flatten to (K, S)
                if None, use self.model.condition
        repeat_factor: if provided, expand states from (K,S) to (repeat_factor*K,S)
                    to match actions batch (e.g., repeat_factor=M when actions is (M*K,A))

        return: (B,)
        """
        actions = torch.as_tensor(actions, device=self.device, dtype=torch.float32)
        if actions.dim() > 2:
            actions = actions.reshape(-1, actions.shape[-1])
        B = actions.shape[0]

        if states is None:
            states = getattr(self.model, "condition", None)

        if states is not None:
            states = torch.as_tensor(states, device=self.device, dtype=torch.float32)
            if states.dim() > 2:
                states = states.reshape(-1, states.shape[-1])

            # ---- expand states to match actions batch ----
            K = states.shape[0]

            if repeat_factor is not None:
                # expect B == repeat_factor * K
                if K * repeat_factor != B:
                    raise ValueError(f"repeat_factor mismatch: states K={K}, repeat_factor={repeat_factor}, actions B={B}")
                states = states.repeat(repeat_factor, 1)
            else:
                # auto-expand if possible
                if K == 1 and B > 1:
                    states = states.repeat(B, 1)
                elif B % K == 0 and B != K:
                    states = states.repeat(B // K, 1)

        q = self.model.calculateQ(states, actions)
        return q.view(-1)


    @torch.no_grad()
    def calculate_reward(
        self,
        actions: torch.Tensor,
        states: torch.Tensor | None = None,
        repeat_factor: int | None = None,
    ) -> torch.Tensor:
        return self.get_reward(actions, states=states, repeat_factor=repeat_factor)


    def __call__(self, states, **kwargs):
        self.model.eval()
        multiple_input = True
        states = torch.FloatTensor(states).to(self.device)
        if states.dim == 1:
            states = states.unsqueeze(0)
            multiple_input = False
        states = states.repeat_interleave(self.args.per_sample_batch_size, dim=0)
        num_states = states.shape[0]
        self.model.condition = states
        x = torch.randn(states.shape[0], self.model.output_dim, device=self.device)
        for i in range(self.args.inference_steps):
            x = self.guide_step(x, i)
        return self.tensor_to_obj(x, num_states, multiple_input)
        
    def tensor_to_obj(self, x, num_states, multiple_input):
        '''
        Perform Best-of-N sampling before final output, number of particles is equal to per_sample_batch_size
        '''
        logprobs = self.get_guidance(x, return_logp=True)
        num_states = x.shape[0] // self.args.per_sample_batch_size
        chunked_x = x.chunk(num_states, dim=0)
        chunked_logprobs = logprobs.chunk(num_states, dim=0)
        selected = []
        for chunk, logprob in zip(chunked_x, chunked_logprobs):
            selected.append(chunk[torch.argmax(logprob)].unsqueeze(0))
        x = torch.cat(selected, dim=0)
        results = x.detach().cpu().numpy()
        actions = results.reshape(num_states, self.model.output_dim).copy()
        out_actions = [actions[i] for i in range(actions.shape[0])] if multiple_input else actions[0]
        return out_actions
        

        



