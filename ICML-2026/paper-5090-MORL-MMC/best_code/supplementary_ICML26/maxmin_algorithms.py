#!/usr/bin/env python

from stable_baselines3.dqn.dqn import MaxminMFQ
import torch as th
import random
import argparse
import numpy as np
# import wandb
from envs.building import env_building
from envs.building.utils_building import ParameterGenerator


if __name__ == "__main__":
    prs = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                  description="""Maxmin value-based MORL algorithm""")
    
    ### 0. GPU ID 
    prs.add_argument("--gpu_id", dest="gpu_id", type=int, default=0, help="GPU ID to use")

    ### 1. Environment parameters
    prs.add_argument("--rd", dest="reward_dim", type=int, default=3, help="Reward dimension\n")
    prs.add_argument("--rdp", dest="r_dim_policy", type=int, default=1, help="equals 1\n")
    prs.add_argument("--ct", dest="cost_type", type=int, default=1, help="Cost type")
    prs.add_argument("--tr", dest="task_rate", type=int, default=10, help="Task rate")
    prs.add_argument("--ei", dest="eval_interval", type=int, default=960, help="Evaluation interval")
    prs.add_argument("--ep", dest="max_episode_steps", type=int, default=480, help="Number of max episodes")
    prs.add_argument("--td", dest="time_delta", type=int, default=1, help="Time delta")
    prs.add_argument("--ub", dest="use_beta", type=bool, default=True, help="Use beta")
    prs.add_argument("--er", dest="empty_reward", type=bool, default=True, help="Empty reward")
    prs.add_argument("--tt", dest="total_timesteps", type=int, default=800000, help="Total Timesteps")
    prs.add_argument("--bf", dest="buffer_size", type=int, default=800000, help="Buffer size\n")
    prs.add_argument("--se", dest="seed", type=int, default=0, help="Random seed\n")

    ### 2. Algorithm parameters
    ## Main Q learning rate
    prs.add_argument("--mlr", dest="main_learning_rate", type=float, default=0.00075, help="Learning Rate of Main Q function\n")

    ## soft target update to incorporate updated w information
    prs.add_argument("--tgit", dest="target_update_interval", type=int, default=1, help="Target_update_interval\n")
    prs.add_argument("--tau", dest="tau", type=float, default=0.001, help="Soft Target update ratio\n")

    ## Exploration strategy for soft Q-update: alpha scheduling
    prs.add_argument("--alin", dest="ent_alpha_act_init", type=float, default=1.0,
                     help="Entropy coefficient for initial action selection. Less than ent_alpha.\n")
    prs.add_argument("--al", dest="ent_alpha", type=float, default=0.05, help="Entropy coefficient for training and final action selection\n")
    prs.add_argument("--alann", dest="annealing_step", type=int, default=10000, help="Length of Linear Entropy schedule. Less than total timesteps.\n")
    ## Perturbation parameters
    prs.add_argument("--init", dest="init_frac", type=float, default=0.0005,
                     help="Initialization ratio for Soft-Q Update\n")
    prs.add_argument("--pwlr", dest="perturb_w_learning_rate", type=float, default=0.000001, help="Learning Rate of w\n")

    prs.add_argument("--perw", dest="period_cal_w_grad", type=int, default=1, help="Period of calculating w gradient\n")
    prs.add_argument("--pqlr", dest="perturb_q_learning_rate", type=float, default=0.073,
                     help="Learning Rate of each q_w\n")
    prs.add_argument("--pgst", dest="perturb_grad_step", type=int, default=1,
                     help="Number of gradient steps for perturbation\n")
    prs.add_argument("--pqnum", dest="perturb_q_copy_num", type=int, default=10,
                     help="Number of copied q networks for perturbation\n")
    prs.add_argument("--pstd", dest="perturb_std_dev", type=float, default=0.01,
                     help="Standard deviation for perturbed noise\n")

    ## Main Q grad step
    prs.add_argument("--mqst", dest="q_grad_st_after_init", type=int, default=3,
                     help="Number of gradient steps for main Q function after init state\n")
    
    def parse_input(arg):
        if ',' in arg:
            return [float(item) for item in arg.split(',')]
        elif isinstance(arg, str): # ['uniform', 'dirichlet']
            return arg
        else:
            raise NotImplementedError

    # weight initialize
    prs.add_argument("--winit", dest="weight_initialize", type=parse_input, nargs='?', default='uniform', help='Initialize Weight w')
    ## Option for w scheduling. For now, we set this as 'sqrt_inverse'
    prs.add_argument("--wsch", dest="w_schedule_option", type=str, choices=['sqrt_inverse', 'inverse', 'linear'],
                     default='sqrt_inverse', help="Option for w scheduling\n")
    prs.add_argument("--abl", dest="weight_ablation", type=str, choices=['main', 'uniform'], default='main', 
                     help="w update ablation: 'main' (learnable) or 'uniform' (static)")
    
    ## Not used - epsilon greedy parameter
    prs.add_argument("--epinit", dest="exploration_initial_eps", type=float, default=0,
                     help="exploration_initial_eps\n")
    prs.add_argument("--epfin", dest="exploration_final_eps", type=float, default=0,
                     help="exploration_final_eps\n")
    prs.add_argument("--epfr", dest="exploration_fraction", type=float, default=0,
                     help="exploration_fraction\n")

    # Others: 
    prs.add_argument("--wd", dest="weight_decay", type=float, default=0,
                     help="Weight for L2 regularization in Adam optimizer\n")
    prs.add_argument("--avwin", dest="stats_window_size", type=int, default=32, help="The number of episodes to average\n")
    prs.add_argument("--nac", dest="N_action_samples_critic", type=int, default=1, help="Number of action samples for critic")
    prs.add_argument("--naa", dest="N_action_samples_actor", type=int, default=1, help="Number of action samples for actor")    

    ### gs. Constraints dim
    prs.add_argument("--cd", dest="constraint_dim", type=int, default=1, help="Constraint dimension\n")
    ### We fix for "maximize"
    prs.add_argument("--ctype", dest="constraint_type", type=str, choices=['minimize', 'maximize'], default='maximize', # J geq C
                     help="Whether we want J less than cth (minimize) or larger than cth (maximize)")
    ### Then, ceps should be positive. 
    prs.add_argument("--ceps", dest="constraint_epsilon", type=float, default= 1.0, help="Initial value of weight for constraints (i.e., u)\n")
    ### Here, cth should be negative because reward is -cost, negative. 
    prs.add_argument("--cth", dest="constraint_threshold", type=float, default=-180, help="We want power consumption lower than certain threshold. \n")
    prs.add_argument("--cwlr", dest="constraint_w_learning_rate", type=float, default=0.000001, help="Learning Rate of u\n")
    prs.add_argument("--csch", dest="c_schedule_option", type=str, choices=["sqrt_inverse", "inverse", "linear"], 
                     default="sqrt_inverse", help="Schedule option for constraint weight u")

    ### GradNet Hyperparams
    ## gradient update params
    prs.add_argument("--glr", dest="gradient_est_learning_rate", type=float, default=5e-05, help="Learning Rate for gradient estimation\n")
    prs.add_argument("--gst", dest="gradient_est_step", type=int, default=1, help="Number of steps for gradient estimation\n")
    prs.add_argument("--gtau", dest="g_tau", type=float, default=0.001, help="Target gradient network update ratio\n")

    ## Algo selection
    prs.add_argument("--galg", dest="grad_algorithm", type=str, choices=['gaussian','gradnet'], default='gradnet')


    args = prs.parse_args()
    
    r_dim = args.reward_dim
    assert r_dim > 1
    
    # # wandb
    # wandb.login(key="")
    # wandb.init(
    #     project="",
    #     job_type="eval",
    #     sync_tensorboard=False,
    #     settings=wandb.Settings(
    #         _disable_stats=True
    #     )
    # )
    #
    # if args.grad_algorithm == 'gradnet':
    #     wandb.run.name = f"[ICML26] gradnet"


    # random seed ## Already in set_random_seed in utils.py, but set randomness fixed in env
    random.seed(args.seed)
    np.random.seed(args.seed)

    params = ParameterGenerator(
        Building='OfficeLarge',
        Weather='Warm_Marine',
        Location='ElPaso'
    )

    env = env_building.BuildingEnv_9d(
        Parameter=params
    )

    # wandb.config.update({
    #     "seed": args.seed,
    #     "ablation" : args.weight_ablation,
    #     "task rate": args.task_rate,
    #     "V": args.cost_type,
    #     "time_delta": args.time_delta,
    #     "use_beta": args.use_beta,
    #     "empty_reward": args.empty_reward,
    #     "eval_interval": args.eval_interval,
    #     "max_episode_steps": args.max_episode_steps,
    #     "total_timesteps": args.total_timesteps,
    #     "mlr": args.main_learning_rate,
    #     "target_update_interval": args.target_update_interval,
    #     "tau": args.tau,
    #     "alin": args.ent_alpha_act_init,
    #     "al": args.ent_alpha,
    #     "alann": args.annealing_step,
    #     "init_frac": args.init_frac,
    #     "pwlr": args.perturb_w_learning_rate,
    #     "cwlr": args.constraint_w_learning_rate,
    #     "period_cal_w_grad": args.period_cal_w_grad,
    #     "pqlr": args.perturb_q_learning_rate,
    #     "perturb_grad_step": args.perturb_grad_step,
    #     "perturb_q_copy_num": args.perturb_q_copy_num,
    #     "perturb_std_dev": args.perturb_std_dev,
    #     "mqst": args.q_grad_st_after_init,
    #     "N_action_samples_critic": args.N_action_samples_critic,
    #     "N_action_samples_actor": args.N_action_samples_actor
    # })

    device = th.device(f"cuda:{args.gpu_id}" if th.cuda.is_available() else "cpu")

    model = MaxminMFQ(
            env=env,
            policy="SQLPolicy",
            learning_rate=args.main_learning_rate,
            N_action_samples_critic=args.N_action_samples_critic,
            N_action_samples_actor=args.N_action_samples_actor,
            learning_starts=0,
            train_freq=1,
            target_update_interval=args.target_update_interval,
            tau=args.tau,
            exploration_initial_eps=args.exploration_initial_eps,
            exploration_final_eps=args.exploration_final_eps,
            exploration_fraction=args.exploration_fraction,
            verbose=1,
            seed=args.seed,
            r_dim=r_dim,
            c_dim=args.constraint_dim,
            c_eps=args.constraint_epsilon,
            c_th=args.constraint_threshold,
            c_type=args.constraint_type,
            r_dim_policy=1,
            buffer_size=args.buffer_size,
            ent_alpha=args.ent_alpha,
            weight_decay=args.weight_decay,
            eval_interval=args.eval_interval, 
            max_episode_steps=args.max_episode_steps,
            ####### perturbation parameters
            soft_q_init_fraction=args.init_frac,
            perturb_w_learning_rate=args.perturb_w_learning_rate,
            constraint_w_learning_rate = args.constraint_w_learning_rate,
            period_cal_w_grad=args.period_cal_w_grad,
            perturb_q_copy_num=args.perturb_q_copy_num,
            perturb_std_dev=args.perturb_std_dev,
            perturb_q_learning_rate=args.perturb_q_learning_rate,
            perturb_grad_step=args.perturb_grad_step,
            q_grad_st_after_init=args.q_grad_st_after_init,
            ###
            gradient_est_learning_rate=args.gradient_est_learning_rate,
            gradient_est_step=args.gradient_est_step,
            g_tau = args.g_tau,
            grad_algorithm = args.grad_algorithm,
            ###
            weight_initialize=args.weight_initialize,
            w_schedule_option=args.w_schedule_option,
            c_schedule_option=args.c_schedule_option,
            weight_ablation= args.weight_ablation,
            ##
            stats_window_size=args.stats_window_size,
            ## alpha scheduling for SQL variants
            ent_alpha_act_init=args.ent_alpha_act_init,
            annealing_step=args.annealing_step,
            device = device 
        )

    model.learn(total_timesteps=args.total_timesteps,
                tb_log_name="MaxminMFQ")

    # wandb.finish()