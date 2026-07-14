from search.configs import Arguments
from search.ddim import DDIMSampler, build_model
from utils import get_args
import d4rl
import gym
import torch
import functools
import numpy as np
from utils import pallaral_eval_policy
from search.doob_search import DoobHGuidanceSampler
from typing import Optional, Type
def get_env_args(args:Arguments):
    env_args = get_args(['--env=' + args.dataset, '--device=' + args.device])
    dataset = args.dataset
    ckpt_dir = f"models_rl/{dataset}0large_actor"
    env_args.env = dataset
    env_args.actor_load_path = f"{ckpt_dir}/behavior_ckpt600.pth"
    env_args.q_load_path = f"{ckpt_dir}/critic_ckpt100.pth"
    env_args.device = args.device
    env_args.diffusion_steps = args.inference_steps
    return env_args


class Pipeline:
    def __init__(
        self,
        args: Arguments = None,
        model=None,
        sampler_cls: Optional[Type[DDIMSampler]] = None,
    ):
        if args is None:
            args = Arguments()
        self.args = args
        self.device = args.device
        self.env_args = get_env_args(args)
        self.env_args.seed = args.seed
        if not model:
            self.model = build_model(device=self.device, args=self.env_args)
        else:
            self.model = model
        if sampler_cls is None:
            sampler_cls = DoobHGuidanceSampler
        self.sampler = sampler_cls(args=self.args, model=self.model)

        self.setup_env()
    
    def setup_env(self):
        args = self.env_args
        env = gym.make(args.env)
        env.seed(args.seed)
        env.action_space.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        args.eval_func = functools.partial(pallaral_eval_policy, env_name=args.env, seed=args.seed, eval_episodes=args.seed_per_evaluation, diffusion_steps=args.diffusion_steps)

    def eval(self):
        envs = self.env_args.eval_func(self.sampler)
        mean = np.mean([envs[i].buffer_return for i in range(self.env_args.seed_per_evaluation)])
        std = np.std([envs[i].buffer_return for i in range(self.env_args.seed_per_evaluation)])
        return mean, std



if __name__ == "__main__":
    args = Arguments()
    args.dataset = "walker2d-medium-replay-v2"
    args.device = "cuda"
    for seed in range(0, 5):
        args.seed = seed
        args.per_sample_batch_size = 4
        args.inference_steps = 15
        args.guidance_strength = 0.0
        pipeline = Pipeline(args)
        mean, std = pipeline.eval()
        print(f"Mean: {mean}, Std: {std}")