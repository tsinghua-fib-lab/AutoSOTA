# compare_gradients_lunar_autograd.py
import argparse
import numpy as np
import torch as th
from torch.autograd.functional import jacobian

def _maybe_imports():
    global gym, TD3
    import gymnasium as gym
    from stable_baselines3 import TD3
    return gym, TD3

# -------------------- utils --------------------

def uniform_l2_ball(center: th.Tensor, radius: float, n: int) -> th.Tensor:
    """
    Sample n points uniformly from an L2 ball around `center` (per batch sample).
    center: (B, S) -> return (n, B, S)
    """
    B, S = center.shape
    x = th.randn(n, B, S, device=center.device, dtype=center.dtype)
    x = x / (th.norm(x, dim=-1, keepdim=True) + 1e-12)
    u = th.rand(n, B, 1, device=center.device, dtype=center.dtype)
    r = radius * (u ** (1.0 / S))
    return center.unsqueeze(0) + r * x

def dqda(critic, s: th.Tensor, a: th.Tensor, *, create_graph: bool = False, retain_graph: bool = False) -> th.Tensor:
    """
    Pure autograd dQ/da at (s,a). If critic has two heads, average grads.
    returns: (B, A)
    """
    with th.enable_grad():
        a_req = a.detach().to(s.device).clone().requires_grad_(True)
        out = critic(s, a_req)
        q1 = out[0] if isinstance(out, (tuple, list)) else out
        assert q1.requires_grad, "q1 has no grad; ensure training mode and no `no_grad` contexts"
        (g1,) = th.autograd.grad(q1.sum(), a_req, create_graph=create_graph, retain_graph=True, allow_unused=False)
        if isinstance(out, (tuple, list)) and len(out) >= 2 and out[1] is not None:
            (g2,) = th.autograd.grad(out[1].sum(), a_req, create_graph=create_graph, retain_graph=True, allow_unused=False)
            g = 0.5 * (g1 + g2)
        else:
            g = g1
        if not retain_graph and not create_graph:
            g = g.detach()
        return g

def grad_s_Q(critic, s: th.Tensor, a: th.Tensor, *, create_graph: bool = False, retain_graph: bool = False) -> th.Tensor:
    """
    Pure autograd ∇_s Q(s,a) using Q1 head. returns: (B, S)
    """
    with th.enable_grad():
        s_req = s.detach().to(a.device).clone().requires_grad_(True)
        out = critic(s_req, a.detach())
        q1 = out[0] if isinstance(out, (tuple, list)) else out
        assert q1.requires_grad, "q1 has no grad; ensure training mode"
        (gs,) = th.autograd.grad(q1.sum(), s_req, create_graph=create_graph, retain_graph=retain_graph, allow_unused=False)
        if not retain_graph and not create_graph:
            gs = gs.detach()
        return gs

def delta_adv_vjp(critic, s, a, eps_state=0.2, alpha=0.02, steps=5):
    """
    목적: max_{||δ||<=eps} || dQ/da(s+δ, a) - dQ/da(s, a) ||
    구현: F=0.5||g-g0||^2, ∇_{s'}F = J(s')^T (g-g0).
    - g0, (g-g0)는 detach 고정
    - 미분은 s' 경로로만 흐름
    - L_inf PGD: delta += alpha * sign(grad_s)
    """
    device = s.device
    a = a.detach().clone().requires_grad_(True)
    s = s.detach().clone().requires_grad_(True)

    delta = th.zeros_like(s, device=device).requires_grad_(True)
    q1_adv, q2_adv = critic(s, a)
    q_adv = (q1_adv+q2_adv)*0.5
    J = th.autograd.grad(q_adv.sum(), s)[0].detach()

    for _ in range(steps):
        s,a,delta=s.detach(),a.detach(),delta.detach()
        elta = th.zeros_like(s, device=device).requires_grad_(True)
        out = critic(s+delta,a)
        c_value = 0.5 * (out[0] + out[1]).detach()
        out_2 = critic((s+delta).detach()+elta,a)
        c_real_value = 0.5 * (out_2[0] + out_2[1])
        # loss=th.nn.MSELoss(c_value.detach()+J*elta-c_real_value)
        loss = ((c_real_value) ** 2).mean()
        gradient = th.autograd.grad(loss.sum(), elta, retain_graph=True, create_graph=True)[0]
        delta = delta + alpha * (gradient/(1e-24+gradient.norm(p=2)))

    return delta  # detach된 최종 perturbation

# -------------------- episode / eval --------------------

def run_episode(env, model):
    obs, _ = env.reset(seed=None)
    done, trunc = False, False
    obs_list, act_list, rew_list = [], [], []
    while not (done or trunc):
        with th.no_grad():
            action, _ = model.predict(obs, deterministic=True)
        obs_list.append(np.asarray(obs, dtype=np.float32))
        act_list.append(np.asarray(action, dtype=np.float32))
        obs, reward, done, trunc, _ = env.step(action)
        rew_list.append(float(reward))
    return np.array(obs_list), np.array(act_list), np.array(rew_list)

def select_split_states(obs_np: np.ndarray):
    T = len(obs_np)
    idxs = [max(0, T // 3 - 1), max(0, (2 * T) // 3 - 1), max(0, T - 2)]
    return sorted(set(idxs))

def compare_at_state(critic, s: th.Tensor, a: th.Tensor, eps_state: float,
                     n_uniform: int = 128, n_worst: int = 2048):
    with th.enable_grad():
        base_grad = dqda(critic, s, a)  # (1, A)

        # (1) uniform samples
        samples = uniform_l2_ball(s, eps_state, n_uniform)  # (n, 1, S)
        diffs = []
        for i in range(n_uniform):
            g_i = dqda(critic, samples[i], a)               # (1, A)
            diffs.append(th.norm(g_i - base_grad, dim=-1))  # (1,)
        diffs = th.stack(diffs, dim=0).squeeze(-1).squeeze(-1)  # (n,)

        # (2) adversarial (autograd + SVD on J)
        delta_adv = delta_adv_vjp(critic, s, a, eps_state, alpha=0.1, steps=10)
        g_adv     = dqda(critic, s + delta_adv, a)
        diff_adv  = th.norm(g_adv - base_grad, dim=-1).item()

        # (3) worst-of-many random
        samples_worst = uniform_l2_ball(s, eps_state, n_worst)
        worst = 0.0
        for i in range(n_worst):
            g_i = dqda(critic, samples_worst[i], a)
            d = th.norm(g_i - base_grad, dim=-1).item()
            if d > worst:
                worst = d

        return {
            "uniform_mean": diffs.mean().item(),
            "uniform_std": diffs.std(unbiased=False).item(),
            "uniform_max": diffs.max().item(),
            "adv_diff": diff_adv,
            "worst_of_many": worst,
            "n_uniform": int(n_uniform),
            "n_worst": int(n_worst),
        }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--eps_state", type=float, default=0.2)
    parser.add_argument("--n_uniform", type=int, default=128)
    parser.add_argument("--n_worst", type=int, default=1024)
    args = parser.parse_args()

    gym, TD3 = _maybe_imports()
    env = gym.make("LunarLanderContinuous-v3")

    model = TD3.load(args.model_path, device=args.device)
    model.policy.set_training_mode(True)
    model.policy.critic.train(True)
    critic = model.policy.critic.to(args.device)

    obs_np, act_np, _ = run_episode(env, model)
    T = len(obs_np)

    # 최대 100개 시점을 균등 간격으로 선택
    num_points = min(100, T)
    idxs = np.linspace(0, T - 1, num=num_points, dtype=int).tolist()

    print(f"Episode length: {T} steps; evaluating {len(idxs)} states")
    print(f"Comparing gradients in L2 ball radius eps_state={args.eps_state}")

    results = []
    for idx in idxs:
        s = th.as_tensor(obs_np[idx:idx+1], dtype=th.float32, device=args.device)  # (1,S)
        a = th.as_tensor(act_np[idx:idx+1], dtype=th.float32, device=args.device)  # (1,A)
        res = compare_at_state(
            critic, s, a,
            eps_state=args.eps_state,
            n_uniform=args.n_uniform,
            n_worst=args.n_worst
        )
        results.append(res)

    # 평균값 집계
    keys = ["uniform_mean", "uniform_std", "uniform_max", "adv_diff", "worst_of_many"]
    avg = {k: float(np.mean([r[k] for r in results])) for k in keys}

    print("\n=== Averages over selected states ===")
    for k in keys:
        print(f"{k:>16}: {avg[k]:.6f}")

if __name__ == "__main__":
    main()
