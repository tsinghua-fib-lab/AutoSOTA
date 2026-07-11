import jax
import jax.numpy as jnp
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils.kernel import *

def eval_nn_regression(args, sim, X0, xT, data, loss, mmd_path, thin_original_mse_path, time_path):
    train_losses = []
    test_losses = []
    for p in tqdm(jnp.vstack([X0[None, :], xT])):
        tr_, te_ = 0.0, 0.0
        # for z_tr, y_tr in zip(data["Z"], data["y"]):
        #     tr_ += loss(z_tr, y_tr, p)
        for z_te, y_te in zip(data["Z_test"], data["y_test"]):
            te_ += loss(z_te, y_te, p)
        train_losses.append(float(tr_) / data["num_batches_tr"])
        test_losses.append(float(te_) / data["num_batches_te"])
    
    train_losses = jnp.array(train_losses)
    test_losses = jnp.array(test_losses)

    jnp.save(f'{args.save_path}/trajectory.npy', jnp.vstack([X0[None, :], xT]))
    jnp.save(f'{args.save_path}/mmd_path.npy', mmd_path)
    jnp.save(f'{args.save_path}/thin_original_mse_path.npy', thin_original_mse_path)
    jnp.save(f'{args.save_path}/train_losses.npy', train_losses)
    jnp.save(f'{args.save_path}/test_losses.npy', test_losses)
    jnp.save(f'{args.save_path}/time_path.npy', time_path)

    axs = plt.subplots(1, 1, figsize=(8, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Training Step')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.legend()
    plt.savefig(f'{args.save_path}/nn_regression_loss.png')
    plt.close()
    return

def eval_nn_classification(args, sim, xT, data, loss, mmd_path, thin_original_mse_path,
                   time_path):
    train_losses, train_accs = [], []
    test_losses, test_accs = [], []
    for p in tqdm(xT):
        tr_loss, tr_acc = 0.0, 0.0
        te_loss, te_acc = 0.0, 0.0
        for z_tr, y_tr in zip(data["Z"], data["y"]):
            l, a = loss(z_tr, y_tr, p)
            tr_loss += l
            tr_acc += a
        for z_te, y_te in zip(data["Z_test"], data["y_test"]):
            l, a = loss(z_te, y_te, p)
            te_loss += l
            te_acc += a
        train_losses.append(float(tr_loss) / data["num_batches_tr"])
        train_accs.append(float(tr_acc) / data["num_batches_tr"])
        test_losses.append(float(te_loss) / data["num_batches_te"])
        test_accs.append(float(te_acc) / data["num_batches_te"])
    train_losses = jnp.array(train_losses)
    train_accs = jnp.array(train_accs)
    test_losses = jnp.array(test_losses)
    test_accs = jnp.array(test_accs)

    # print("Final Train pred:", sim._vm_q1(data["Z"][0, ...], xT[-1]).mean(axis=0)[:5].squeeze())
    # print("Final Train label:", data["y"][0, ...][:5].squeeze())
    # print("Final Test pred:", sim._vm_q1(data["Z_test"][0, ...], xT[-1]).mean(axis=0)[:5].squeeze())
    # print("Final Test label:", data["y_test"][0, ...][:5].squeeze())
    # print("Final Train Loss:", train_losses[-1])
    # print("Final Test Loss:", test_losses[-1])
    # print("Final Train Acc:", train_accs[-1])
    # print("Final Test Acc:", test_accs[-1])

    jnp.save(f'{args.save_path}/trajectory.npy', xT)
    jnp.save(f'{args.save_path}/mmd_path.npy', mmd_path)
    jnp.save(f'{args.save_path}/thin_original_mse_path.npy', thin_original_mse_path)
    jnp.save(f'{args.save_path}/train_losses.npy', train_losses)
    jnp.save(f'{args.save_path}/test_losses.npy', test_losses)
    jnp.save(f'{args.save_path}/time_path.npy', time_path)
    return


def eval_vlm(args, sim, xT, data, init, x_ground_truth, 
             lotka_volterra_ws, lotka_volterra_ms, 
             mmd_path, thin_original_mse_path, time_path):
    rng_key = jax.random.PRNGKey(14)
    data_longer = lotka_volterra_ms(init, x_ground_truth, rng_key, end=100, noise_scale=0.)
    loss = jnp.zeros(xT.shape[0])
    kgd_values = jnp.zeros(xT.shape[0])

    def S_PQ(X):
        N, d = X.shape
        bs = int(jnp.sqrt(N))
        G = jnp.zeros((bs, bs, d), dtype=X.dtype)
        X_batched = X.reshape((-1, bs, d))
        for i in tqdm(range(X_batched.shape[0])):
            X_batch = X_batched[i, :]
            key_batch = jax.random.split(jax.random.fold_in(rng_key, i), num=bs)  # (bs, 2)
            keys = jax.vmap(lambda k: jax.random.split(k, num=N))(key_batch)
            gi = sim._vm_grad_q2(X_batch, X, keys).mean(axis=1)
            G = G.at[i, :].set(gi)
        G = G.reshape((N, d))
        return -args.zeta * X + G

    for t, particles in enumerate(tqdm(xT)):
        rng_key, _ = jax.random.split(rng_key)
        sampled_trajectories_all = jax.vmap(lambda p: lotka_volterra_ws(init, p, rng_key, 100))(particles)
        sampled_trajectories = sampled_trajectories_all.mean(axis=0)
        sampled_trajectories_std = sampled_trajectories_all.std(axis=0)

        loss = loss.at[t].set(jnp.mean((sampled_trajectories - data_longer) ** 2, axis=(0,)).sum())

        l = jax.jit(lambda x, y: k_imq(x, y, 1, 0.5,0.1)) #matern_kernel(x,y,5.0))
        alpha = 2.0
        beta = 1.0
        k = lambda x,y : recommended_kernel(x,y,l,alpha,beta,1.0)
        k_pq = GradientKernel(S_PQ, k)
        kgd = KernelGradientDiscrepancy(k_pq)
        kgd_value = kgd.evaluate(particles)
        kgd_values = kgd_values.at[t].set(kgd_value)

        if t % 10 == 0:
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            axes[1].plot(sampled_trajectories[:, 0], color='red', label='Prey')
            axes[1].plot(sampled_trajectories[:, 1], color='blue', label='Predator')
            axes[1].fill_between(
                jnp.arange(sampled_trajectories.shape[0]),
                sampled_trajectories[:, 0] - 2 * sampled_trajectories_std[:, 0],
                sampled_trajectories[:, 0] + 2 * sampled_trajectories_std[:, 0],
                color='red',
                alpha=0.3,
            )
            axes[1].fill_between(
                jnp.arange(sampled_trajectories.shape[0]),
                sampled_trajectories[:, 1] - 2 * sampled_trajectories_std[:, 1],
                sampled_trajectories[:, 1] + 2 * sampled_trajectories_std[:, 1],
                color='blue',
                alpha=0.3,
            )
            axes[1].plot(data_longer[:, 0], color='grey', linestyle='dashed', label='Ground Truth')
            axes[1].plot(data_longer[:, 1], color='grey', linestyle='dashed')
            axes[1].scatter(jnp.arange(data.shape[0]), data[:, 0], color='black', s=10, label='Prey Data')
            axes[1].scatter(jnp.arange(data.shape[0]), data[:, 1], color='black', s=10, label='Predator Data')
            axes[1].legend()
            axes[1].grid(True)
            plt.savefig(f'{args.save_path}/vlm_distribution_step_{t}.png')
            plt.close(fig)

    jnp.save(f'{args.save_path}/vlm_trajectory.npy', xT)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left subplot
    axes[0].plot(loss, label='MSE Loss')
    axes[0].set_yscale('log')
    axes[0].set_xlabel('Training Step')
    axes[0].legend()

    # Right subplot (example — duplicate or plot another metric)
    axes[1].plot(kgd_values, label='KGD')
    axes[1].set_yscale('log')
    axes[1].set_xlabel('Training Step')
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(f'{args.save_path}/vlm_loss.png')
    plt.close()


    jnp.save(f'{args.save_path}/vlm_loss.npy', loss)
    jnp.save(f'{args.save_path}/vlm_kgd.npy', kgd_values)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # --- (2) MMD^2 path ---
    axes[0].plot(mmd_path, color="C2", label="MMD$^2$")
    axes[0].set_xlabel("Training Step")
    axes[0].set_ylabel("MMD$^2$")
    axes[0].legend()
    axes[0].set_yscale("log")
    axes[0].grid(True, linestyle="--", alpha=0.5)

    # --- (3) Thinned vs Original MSE path ---
    axes[1].plot(thin_original_mse_path, color="C3", label="Thin-Original MSE")
    axes[1].set_xlabel("Training Step")
    axes[1].set_ylabel("MSE")
    axes[1].set_title("Thinned vs Original Output MSE")
    axes[1].legend()
    axes[1].set_yscale("log")
    axes[1].grid(True, linestyle="--", alpha=0.5)
    # --- Layout and save ---
    plt.tight_layout()
    plt.savefig(f"{args.save_path}/mfld_debug_vector_field.png", dpi=300)
    plt.show()

    jnp.save(f'{args.save_path}/mmd_path.npy', mmd_path)
    jnp.save(f'{args.save_path}/thin_original_mse_path.npy', thin_original_mse_path)
    jnp.save(f'{args.save_path}/time_path.npy', time_path)
    return


def eval_mmd_flow(args, sim, xT, data, mmd_path, thin_original_mse_path, time_path):
    @jax.jit
    def mmd_func(Y):
        K_XX = sim.problem.distribution.mean_mean_embedding()
        K_YY = sim.kernel.make_distance_matrix(Y, Y)
        K_XY = sim.problem.distribution.mean_embedding(Y)
        return jnp.sqrt(K_XX + K_YY.mean() - 2 * K_XY.mean())

    mmd_values = jnp.zeros(xT.shape[0])
    for t, particles in enumerate(tqdm(xT)):
        mmd_value = mmd_func(particles)
        mmd_values = mmd_values.at[t].set(mmd_value)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].plot(mmd_values)
    axes[0].set_yscale('log')
    axes[0].set_xlabel('Training Step')
    axes[0].set_ylabel('MMD Value')
    axes[1].plot(mmd_path)
    axes[1].set_yscale('log')
    axes[1].set_xlabel('Training Step')
    axes[1].set_ylabel('MMD between thinned set and original set')
    plt.savefig(f'{args.save_path}/mmd_values_trajectory.png')
    plt.close(fig)

    freq = 10
    jnp.save(f'{args.save_path}/mmd_flow_trajectory.npy', xT[::freq, :, :])
    jnp.save(f'{args.save_path}/mmd_path.npy', mmd_path)
    jnp.save(f'{args.save_path}/thin_original_mse_path.npy', thin_original_mse_path)
    jnp.save(f'{args.save_path}/mmd_values_trajectory.npy', mmd_values)
    jnp.save(f'{args.save_path}/time_path.npy', time_path)
    # save_animation_2d(xT, sim.problem.distribution, save_path=args.save_path)
    return 


def eval_mfg(args, ts, X, kernel, x_history, p_history, time_history):
    M, N = X.shape[0] - 1, X.shape[1]
    energy_history = []
    interaction_cost_history = []
    terminal_cost_history = []
    total_cost_history = []

    for x, p in tqdm(zip(x_history, p_history)):
        energy = jnp.sum(jnp.square(p)) / args.particle_num
        interaction = 0.0
        for t in range(M + 1):
            temp = kernel(x, x).mean()
            interaction += temp
        interaction /= (M + 1)
        terminal = 10 * jnp.sum(jnp.square(x[-1]), axis=1).mean()
        energy_history.append(energy)
        interaction_cost_history.append(interaction)
        terminal_cost_history.append(terminal)
        total_cost_history.append(energy + interaction + terminal)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes[0, 0].plot(energy_history)
    axes[0, 0].set_yscale('log')
    axes[0, 0].set_title('Energy Cost')
    axes[0, 1].plot(interaction_cost_history)
    axes[0, 1].set_yscale('log')
    axes[0, 1].set_title('Interaction Cost')
    axes[1, 0].plot(terminal_cost_history)
    axes[1, 0].set_yscale('log')
    axes[1, 0].set_title('Terminal Cost')
    axes[1, 1].plot(total_cost_history)
    axes[1, 1].set_yscale('log')
    axes[1, 1].set_title('Total Cost')
    plt.savefig(f'{args.save_path}/mfg_costs.png')
    plt.close()

    jnp.save(f'{args.save_path}/mfg_energy_history.npy', jnp.array(energy_history))
    jnp.save(f'{args.save_path}/mfg_interaction_cost_history.npy', jnp.array(interaction_cost_history))
    jnp.save(f'{args.save_path}/mfg_terminal_cost_history.npy', jnp.array(terminal_cost_history))
    jnp.save(f'{args.save_path}/mfg_total_cost_history.npy', jnp.array(total_cost_history))
    jnp.save(f'{args.save_path}/mfg_time_history.npy', jnp.array(time_history))
    # save_animation_mfg(args, ts, X[:, :, :2], title=f"MFG particles", interval=35)
    return 


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def save_animation_2d(
    trajectory,
    distribution,
    save_path,
    duration_sec=6.0,   # fixed total video length
    fps=20,             # fixed fps
    interval_ms=50,     # display interval (doesn't affect saved mp4 if fps is set)
    resolution=100,
    xlim=(-5, 5),
    ylim=(-5, 5),
):
    """
    Save an animation with fixed duration, regardless of trajectory length.

    trajectory: array of shape (T, N, 2)
    duration_sec: desired video length in seconds
    fps: frames per second for the saved mp4
    """
    T = int(trajectory.shape[0])
    if T <= 0:
        raise ValueError("trajectory must have at least one timestep")

    # Fixed number of frames in the output video
    num_frames = max(int(round(duration_sec * fps)), 1)

    # Map frame indices -> trajectory indices (monotone, covers [0, T-1])
    if num_frames == 1:
        t_idx = np.array([0], dtype=int)
    else:
        t_idx = np.linspace(0, T - 1, num_frames)
        t_idx = np.round(t_idx).astype(int)
        t_idx = np.clip(t_idx, 0, T - 1)

    # ---- create initial plot ----
    fig, ax = plt.subplots()
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)

    # Background: logpdf on a grid
    x_vals = np.linspace(xlim[0], xlim[1], resolution)
    y_vals = np.linspace(ylim[0], ylim[1], resolution)
    X, Y = np.meshgrid(x_vals, y_vals)
    grid = np.stack([X.ravel(), Y.ravel()], axis=1)

    # If distribution.pdf expects jax arrays, swap np -> jnp here; otherwise keep numpy.
    logpdf = np.log(distribution.pdf(grid)).reshape(resolution, resolution)

    ax.imshow(logpdf, extent=(xlim[0], xlim[1], ylim[0], ylim[1]), origin="lower")

    # NOTE: you had (y,x) swapped in scatter; keep consistent with your original intent.
    scat = ax.scatter(trajectory[0, :, 1], trajectory[0, :, 0], label="source")

    def update(frame_j):
        t = t_idx[frame_j]
        # set_offsets expects (N, 2) with columns [x, y]
        pts = np.asarray(trajectory[t])
        scat.set_offsets(np.column_stack([pts[:, 1], pts[:, 0]]))
        return (scat,)

    ani = FuncAnimation(
        fig,
        update,
        frames=num_frames,
        blit=True,
        interval=interval_ms,
    )

    ani.save(f"{save_path}/animation.mp4", writer="ffmpeg", fps=fps)
    plt.close(fig)
    return


def save_animation_mfg(args, ts, X, title="MFG particle evolution", interval=40):
    """
    Animate particles in 2D.
    - X: (M+1, N, 2)
    - trail: how many past frames to show as faint points (0 disables)
    """
    Np1, N, _ = X.shape

    # Plot bounds
    xmin = float(np.min(X[:, :, 0])) - 0.5
    xmax = float(np.max(X[:, :, 0])) + 0.5
    ymin = float(np.min(X[:, :, 1])) - 0.5
    ymax = float(np.max(X[:, :, 1])) + 0.5

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_title(title)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")

    # Target at origin
    ax.scatter([0.0], [0.0], marker="x", s=80, color='red', label='Target')
    # Initial particles
    scat = ax.scatter(X[0, :, 0], X[0, :, 1], s=80, color='blue')
    time_text = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top")

    def update(frame):
        scat.set_offsets(X[frame])
        time_text.set_text(f"t = {ts[frame]:.3f}")
        return (scat, time_text)

    anim = FuncAnimation(fig, update, frames=range(Np1), interval=interval, blit=True)
    from matplotlib.animation import FFMpegWriter
    writer = FFMpegWriter(fps=30, metadata=dict(artist='MFG'), bitrate=1800)
    anim.save(f"{args.save_path}/mfg_trajectory.mp4", writer=writer)
    return anim
