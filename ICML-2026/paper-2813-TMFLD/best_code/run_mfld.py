from utils.configs import CFG
from utils.problems import *
from mfld import MFLD_nn, MFLD_vlm, MFLD_mmd_flow
from utils.datasets import load_student_teacher, load_covertype
import os
# os.environ["JAX_PLATFORM_NAME"] = "cpu"
# os.environ["CUDA_VISIBLE_DEVICES"] = ""
import jax.numpy as jnp
import jax
import time
import argparse
import pickle
import time
from utils.lotka_volterra import lotka_volterra_ws, lotka_volterra_ms
from utils.evaluate import eval_nn_classification, eval_nn_regression, eval_vlm, eval_mmd_flow


def get_config():
    parser = argparse.ArgumentParser(description='thinned_mfld')

    # Args settings
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--kernel', type=str, default='sobolev')
    parser.add_argument('--step_size', type=float, default=0.1)
    parser.add_argument('--dataset', type=str, default='boston')
    parser.add_argument('--g', type=int, default=0)
    parser.add_argument('--noise_scale', type=float, default=0.1)
    parser.add_argument('--sigma_start', type=float, default=None)
    parser.add_argument('--sigma_end', type=float, default=None)
    parser.add_argument('--noise_schedule', type=str, default='fixed')
    parser.add_argument('--bandwidth', type=float, default=1.0)
    parser.add_argument('--step_num', type=int, default=100)
    parser.add_argument('--bs', type=int, default=1024)
    parser.add_argument('--particle_num', type=int, default=100)
    parser.add_argument('--save_path', type=str, default='./results/')
    parser.add_argument('--thinning', type=str, default='kt')
    parser.add_argument('--zeta', type=float, default=1.0)
    parser.add_argument('--d', type=int, default=20)
    parser.add_argument('--teacher_num', type=int, default=100)
    parser.add_argument('--kt_function', type=str, default='compresspp_kt')
    parser.add_argument("--skip_swap", action="store_true")
    parser.add_argument("--momentum", type=float, default=0.0)
    parser.add_argument("--nesterov", action="store_true")
    args = parser.parse_args()  
    return args

def create_dir(args):
    if args.seed is None:
        args.seed = int(time.time())
    args.save_path += f"neural_network_{args.dataset}/thinning_{args.thinning}/"
    args.save_path += f"kernel_{args.kernel}__step_size_{args.step_size}__bandwidth_{args.bandwidth}__step_num_{args.step_num}"
    args.save_path += f"__g_{args.g}__particle_num_{args.particle_num}__noise_scale_{args.noise_scale}__zeta_{args.zeta}"
    args.save_path += f"__d_{args.d}__teacher_num_{args.teacher_num}__seed_{args.seed}__kt_function_{args.kt_function}"
    args.save_path += f"__skip_swap_{args.skip_swap}"
    os.makedirs(args.save_path, exist_ok=True)
    with open(f'{args.save_path}/configs', 'wb') as handle:
        pickle.dump(vars(args), handle, protocol=pickle.HIGHEST_PROTOCOL)
    return args

def main(args):
    rng_key = jax.random.PRNGKey(args.seed)
    if args.dataset == 'student_teacher':
        def R1_prime(hat_y, y):  # R1(s)=0.5*s^2
            return hat_y - y

        def q1_nn(z, x):
            d_hidden = z.shape[-1]
            W1, b1, W2 = x[:d_hidden], x[d_hidden+1], x[d_hidden+1:]
            h = jax.nn.relu(z @ W1 + b1)
            y = jnp.dot(W2, h)
            return jnp.clip(y, -1e3, 1e3)

        data = load_student_teacher(batch_size=args.bs, train_size=args.particle_num * args.bs, 
                                    q1_nn_apply=q1_nn, d=args.d, M=args.teacher_num,
                                    standardize_Z=True, standardize_y=False)
        # data['Z'] = data['Z'].reshape((args.bs, args.bs, args.particle_num, args.d))
        # data['y'] = data['y'].reshape((args.bs, args.bs, args.particle_num, -1))

        @jax.jit
        def loss(Z, y, params):
            """Compute MSE for a given parameter vector `params`."""
            preds_all = jax.vmap(                       # over particles
                    jax.vmap(q1_nn, in_axes=(0, None)),     # over batch
                    in_axes=(None, 0)                          # Z[p], params[p]
                )(Z, params)
            preds = preds_all.mean(axis=0)
            return jnp.mean((preds - y) ** 2)
        
        output_d = data["y"].shape[-1] if len(data["y"].shape) > 2 else 1
        input_d = data["Z"].shape[-1]
        problem_nn = Problem_nn(
            particle_d=data["Z"].shape[-1] + 1 + output_d,  # NN params dimension
            input_d=input_d,
            output_d=output_d,
            R1_prime=R1_prime,
            q1=q1_nn,
            q2=None,
            data=data
        )
    elif args.dataset == 'vlm':
        from utils.kernel import gaussian_kernel
        kernel = gaussian_kernel(sigma=1.0)
        init = jnp.array([10.0, 15.0])
        # init = jnp.array([10.0, 10.0])
        x_ground_truth = jnp.array([-1., -1.5413]) # True parameters from Clementine's code
        # x_ground_truth = jnp.array([-2.0, -1.733]) # True parameters from Zheyang's paper
        rng_key = jax.random.PRNGKey(14) # Fix random seed for data generation
        data = lotka_volterra_ms(init, x_ground_truth, rng_key)
        def q2(x, x_prime, rng_key):
            rng_key, _ = jax.random.split(rng_key)
            traj_1 = lotka_volterra_ws(init, x, rng_key)
            rng_key, _ = jax.random.split(rng_key)
            traj_2 = lotka_volterra_ws(init, x_prime, rng_key)
            kernel_vmap = jax.vmap(kernel, in_axes=(0, 0))
            part1 = kernel_vmap(traj_1, traj_2)
            part2 = kernel_vmap(traj_1, data)
            return part1.sum() - 2 * part2.sum()
        
        problem_vlm = Problem_vlm(
            particle_d=2,
            q2=q2,
            data=data
        )
    elif args.dataset == 'mmd_flow':
        from utils.kernel import gaussian_kernel
        kernel = gaussian_kernel(sigma=args.bandwidth)
        def q2(x, x_prime, rng_key):
            return kernel(x, x_prime)
        
        problem_mmd_flow = Problem_mmd_flow(
            particle_d=2,
            q2=q2,
            distribution=Distribution(kernel)
        )
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    
    if args.dataset in ['student_teacher']:
        # This is mean-field neural network
        cfg = CFG(N=args.particle_num, steps=args.step_num, step_size=args.step_size, sigma=args.noise_scale, kernel=args.kernel,
              sigma_start=args.sigma_start if args.sigma_start is not None else args.noise_scale*10, sigma_end=args.sigma_end if args.sigma_end is not None else args.noise_scale, noise_schedule=args.noise_schedule,
              zeta=args.zeta, g=args.g, seed=args.seed, bandwidth=args.bandwidth, return_path=True, kt_function=args.kt_function,
              skip_swap=args.skip_swap, alpha=args.momentum, nesterov=args.nesterov)
        sim = MFLD_nn(problem=problem_nn, save_freq=data["num_batches_tr"], thinning=args.thinning, cfg=cfg, args=args)
        rng_key, sub = jax.random.split(rng_key)
        X0 = 0.05 * jax.random.normal(sub, (cfg.N, problem_nn.particle_d))
    elif args.dataset == 'vlm':
        # This is post-Bayesian inference
        cfg = CFG(N=args.particle_num, steps=args.step_num, step_size=args.step_size, sigma=args.noise_scale, kernel=args.kernel,
              sigma_start=args.sigma_start if args.sigma_start is not None else args.noise_scale*10, sigma_end=args.sigma_end if args.sigma_end is not None else args.noise_scale, noise_schedule=args.noise_schedule,
              zeta=args.zeta, g=args.g, seed=args.seed, bandwidth=args.bandwidth, return_path=True, kt_function=args.kt_function,
              skip_swap=args.skip_swap, alpha=args.momentum, nesterov=args.nesterov)
        sim = MFLD_vlm(problem=problem_vlm, save_freq=1, thinning=args.thinning, cfg=cfg, args=args)
        X0 = jnp.stack([x_ground_truth] * args.particle_num, 0)
        rng_key, _ = jax.random.split(rng_key)
        X0 += 0.1 * jax.random.normal(rng_key, X0.shape)
    elif args.dataset == 'mmd_flow':
        # This is MMD flow
        cfg = CFG(N=args.particle_num, steps=args.step_num, step_size=args.step_size, sigma=args.noise_scale, kernel=args.kernel,
              sigma_start=args.sigma_start if args.sigma_start is not None else args.noise_scale*10, sigma_end=args.sigma_end if args.sigma_end is not None else args.noise_scale, noise_schedule=args.noise_schedule,
              zeta=args.zeta, g=args.g, seed=args.seed, bandwidth=args.bandwidth, return_path=True, kt_function=args.kt_function,
              skip_swap=args.skip_swap, alpha=args.momentum, nesterov=args.nesterov)
        sim = MFLD_mmd_flow(problem=problem_mmd_flow, save_freq=1, thinning=args.thinning, cfg=cfg, args=args)
        rng_key, _ = jax.random.split(rng_key)
        X0 = 2.0 * jax.random.normal(rng_key, (args.particle_num, problem_mmd_flow.particle_d))
    xT, mmd_path, thin_original_mse_path, time_path, x_ema = sim.simulate(x0=X0)
    jnp.save(f'{args.save_path}/ema_particles.npy', x_ema)

    if args.dataset in ['covertype']:
        eval_nn_classification(args, sim, xT, data, loss, mmd_path, thin_original_mse_path, time_path)
    elif args.dataset in ['student_teacher']:
        eval_nn_regression(args, sim, X0, xT, data, loss, mmd_path, thin_original_mse_path, time_path)
        # Compute EMA test loss
        ema_test_loss = 0.0
        for z_te, y_te in zip(data["Z_test"], data["y_test"]):
            ema_test_loss += loss(z_te, y_te, x_ema)
        ema_test_loss = float(ema_test_loss / data["num_batches_te"])
        jnp.save(f'{args.save_path}/test_loss_ema.npy', jnp.array([ema_test_loss]))
        print(f'EMA Test Loss: {ema_test_loss:.10f}')
    elif args.dataset == 'vlm':
        eval_vlm(args, sim, xT, data, init, x_ground_truth, 
                 lotka_volterra_ws, lotka_volterra_ms, 
                 mmd_path, thin_original_mse_path, time_path)
    elif args.dataset == 'mmd_flow':
        eval_mmd_flow(args, sim, xT, None, mmd_path, thin_original_mse_path, time_path)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    
    jnp.save(f'{args.save_path}/time_path.npy', time_path)
    return


if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)  # optional for stability
    args = get_config()
    args = create_dir(args)
    new_save_path = args.save_path + '__complete'
    # Early exit if job already completed
    if os.path.exists(new_save_path):
        print(f"Job already completed. Folder exists: {new_save_path}")
        sys.exit(0)

    print('Program started!')
    print(vars(args))
    main(args)
    print('Program finished!')
    import shutil
    if os.path.exists(new_save_path):
        shutil.rmtree(new_save_path)  # Deletes existing folder
    os.rename(args.save_path, new_save_path)
    print(f'Job completed. Renamed folder to: {new_save_path}')
