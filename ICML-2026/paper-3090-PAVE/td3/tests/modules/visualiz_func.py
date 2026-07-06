# sb3_visualizer.py
"""
Stable-Baselines3 시각화 & 메트릭 모듈 (ver. 2025-08-03)
========================================================
TD3·SAC·DDPG 등 Off-policy 모델 학습 시
정책-가치-곡률 시각화와 Lipschitz·Hessian·FFT 스칼라 지표를
TensorBoard/PNG 로 기록합니다.
"""
from __future__ import annotations

import io, os, warnings
from typing import Tuple, Optional, List

import numpy as np
import torch, gym
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 – needed for 3-D proj
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from stable_baselines3.common.callbacks import BaseCallback

__all__ = ["VisualizerCallback", "_generate_all"]

# ---------------------------------------------------------------------------
# Low-level utils
# ---------------------------------------------------------------------------

def _plot_to_image(fig: plt.Figure) -> np.ndarray:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    buf.seek(0)
    img = np.asarray(Image.open(buf))[:, :, :3]
    return np.transpose(img, (2, 0, 1))


def _fft_smoothness(actions: np.ndarray, fs: float = 1.0) -> float:
    """
    NumPy 로 구현한 smoothness 지표.
    actions: shape (T,) 또는 (T, d)  (T = timestep 수, d = 액션 차원)
    fs: 샘플링 주파수 (기본 1.0)
    """
    # 1차원일 때 (T,) -> (T,1)
    a = np.array(actions, dtype=float)
    if a.ndim == 1:
        a = a[:, None]

    n = a.shape[0]
    if n < 2:
        return 0.0

    # FFT
    # axis=0 방향으로 fft, 양의 주파수 절반만 취함
    yf = np.fft.fft(a, axis=0)
    yf = np.abs(yf[: n // 2, :])    # shape (n//2, d)

    # 주파수 벡터 생성 (n//2 길이)
    freqs = np.fft.fftfreq(n, d=1/fs)[: n // 2]  # shape (n//2,)
    freqs = freqs.reshape(-1, 1)                  # (n//2,1)

    # 식: Sm = 2/(n*fs) * sum_i (M_i * f_i)
    smooth_per_dim = (2.0 / (n * fs)) * np.sum(freqs * yf, axis=0)  # shape (d,)

    # 다차원 액션이면 차원별 평균, 1차원 액션이면 그냥 원소 반환
    return float(np.mean(smooth_per_dim))



def _avg_fft_plot(action_seqs: List[np.ndarray], fs: float = 1.0) -> plt.Figure:
    if not action_seqs:
        fig = plt.figure(); plt.text(0.5, 0.5, "No data", ha="center"); return fig
    min_T = min(s.shape[0] for s in action_seqs)
    data = np.stack([s[:min_T] for s in action_seqs], 0)
    fft_vals = np.fft.rfft(data, axis=1)
    freqs = np.fft.rfftfreq(min_T, d=1 / fs)
    mag = np.abs(fft_vals).mean(axis=(0, 2))
    fig = plt.figure(figsize=(8, 4))
    plt.plot(freqs, mag, label="Avg FFT"); plt.grid(True)
    plt.xlabel("Freq [Hz]"); plt.ylabel("Amplitude"); plt.legend(); plt.tight_layout()
    return fig

# ---------------------------------------------------------------------------
# Actor / Critic accessor helpers (SB3 최신 버전 대응)
# ---------------------------------------------------------------------------

def _get_actor(model):
    return model.policy.actor


def _get_q1_fn(model):
    critic = model.policy.critic
    return lambda s, a: critic(s, a)[0]

# ---------------------------------------------------------------------------
# Gradient / Hessian helpers
# ---------------------------------------------------------------------------

def _grad_Q(state, action, q1_fn):
    a_var = action.clone().detach().requires_grad_(True)
    q_val = q1_fn(state, a_var)
    grad, = torch.autograd.grad(q_val.sum(), a_var)
    return grad

# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _surface(ax, X, Y, Z, title, zlab):
    ax.plot_surface(X, Y, Z, cmap="viridis", rstride=1, cstride=1, linewidth=0)
    ax.set_title(title); ax.set_xlabel("x₁"); ax.set_ylabel("x₂"); ax.set_zlabel(zlab)


def _contour(ax, X, Y, Z, title: str, clab: str):
    """
    Filled-contour helper that draws a square plotting box so
    the x- and y-axes have identical on-screen lengths,
    regardless of the data range.
    """
    cf = ax.contourf(X, Y, Z, levels=20, cmap="viridis")
    ax.set_title(title)
    ax.set_xlabel("x₁")
    ax.set_ylabel("x₂")

    # --- make the axes box square ---
    if hasattr(ax, "set_box_aspect"):         # Matplotlib ≥ 3.4
        ax.set_box_aspect(1)                  # exact 1:1 box
    else:                                     # Matplotlib ≤ 3.3
        ax.set_aspect("equal", adjustable="box")

    return cf


# ---------------------------------------------------------------------------
# Heavy metric + plot core
# ---------------------------------------------------------------------------


def _generate_all(model,
                  env_func: function,
                  writer: SummaryWriter,
                  step: int,
                  *,
                  grid: int = 50,
                  boundary: Optional[Tuple[float, float]] = None,
                  eval_eps: int = 10,
                  max_steps: int = 1000,
                  png_dir: Optional[str] = None):
    """
    한 번 호출로 그리드 스캔 + 롤아웃 기반 지표 전체를 기록.
    - SB3 ↔ Gym 통신: NumPy
    - 미분·시각화 계산: Torch (GPU 사용)
    """
    device = model.device                       # ex) cuda:0
    actor  = _get_actor(model)                  # Torch actor (for grad/Hess)
    q1_fn  = _get_q1_fn(model)
    N_SAMPLES   = 32      # 각 격자점에서 δ 샘플 수
    DELTA_SCALE = 0.01   # δ 크기 스케일
    
    if actor is None:
        warnings.warn("Actor network not accessible – skip viz"); return

    # ------------------------------------------------------------------
    # 1. 상태 그리드 스캔 (행동은 model.predict 로 NumPy 반환 → Torch 변환)
    # ------------------------------------------------------------------
    env = env_func()
    low, high = env.observation_space.low, env.observation_space.high
    if boundary is not None:
        low[:], high[:] = boundary[0], boundary[1]

    x1 = np.linspace(low[0], high[0], grid)
    x2 = np.linspace(low[1], high[1], grid)
    X, Y   = np.meshgrid(x1, x2)
    Z_act  = np.zeros_like(X)    # env-scale action 값 저장
    Z_q    = np.zeros_like(X)
    Z_qg   = np.zeros_like(X)
    p_lip, qg_lip, hess_sv = [], [], []

    # SB3 helper
    scale   = getattr(model.policy, "scale_action",   lambda a: a)   # env→raw
    unscale = getattr(model.policy, "unscale_action", lambda a: a)   # raw→env

    for i in range(grid):
        for j in range(grid):
            obs_np = np.array([X[i, j], Y[i, j]], dtype=np.float32)        # (obs_dim,)
            act_env, _ = model.predict(obs_np, deterministic=True)         # NumPy, env-scale
            Z_act[i, j] = act_env[0]

            # ----- Torch 변환 (grad/Hessian 계산용) -----
            s_torch = torch.tensor(obs_np, dtype=torch.float32, device=device).unsqueeze(0)
            a_raw   = torch.tensor(scale(act_env), dtype=torch.float32, device=device).unsqueeze(0)

            with torch.no_grad():
                Z_q[i, j] = q1_fn(s_torch, a_raw).cpu().numpy()[0, 0]

            g = _grad_Q(s_torch, a_raw, q1_fn)
            Z_qg[i, j] = g.detach().cpu().numpy()[0, 0]

            # Lipschitz 근사
            local_p_max = 0.0
            local_qg_max = 0.0
            local_hess_max = 0.0

            for _ in range(N_SAMPLES):
                delta = ((torch.rand_like(s_torch) - 0.5) * DELTA_SCALE * torch.tensor(high - low, device=device))
                s2    = s_torch + delta
                d     = torch.norm(delta).item()

                # Policy Lipschitz: ‖π(s) - π(s+δ)‖ / ‖δ‖
                act2_env, _ = model.predict(s2.cpu().numpy().squeeze(), deterministic=True)
                a2_raw  = torch.tensor(scale(act2_env), dtype=torch.float32, device=device).unsqueeze(0)
                val_p = torch.norm(a_raw - a2_raw).item() / d
                if val_p > local_p_max:
                    local_p_max = val_p

                # Q-grad Lipschitz: ‖∇_a Q(s,a) - ∇_a Q(s+δ,a)‖ / ‖δ‖  (a 고정)
                g2 = _grad_Q(s2, a_raw, q1_fn)
                val_qg = torch.norm(g - g2).item() / d
                if val_qg > local_qg_max:
                    local_qg_max = val_qg


            # 이 격자점의 '로컬 최대'를 전역 리스트에 반영
            p_lip.append(local_p_max)
            qg_lip.append(local_qg_max)

    # ===== 스칼라 기록: '평균(mean)' 위주로 기록, 필요하면 max도 함께 =====
    writer.add_scalar("grid/policy_lip_mean", float(np.mean(p_lip)) if p_lip else 0.0, step)
    writer.add_scalar("grid/qg_lip_mean",    float(np.mean(qg_lip)) if qg_lip else 0.0, step)
    writer.add_scalar("grid/policy_lip_max", float(np.max(p_lip)) if p_lip else 0.0, step)
    writer.add_scalar("grid/qg_lip_max",    float(np.max(qg_lip)) if qg_lip else 0.0, step)


    # --- 3-D 및 등고선 플롯 기록 ---
    for tag, Z, title, zlab in [
        ("policy3d", Z_act, "Policy Surface", "Action (env-scale)"),
        ("qgrad3d",  Z_qg,  "Q-Grad Surface", "∂Q/∂a"),
        ("q3d",      Z_q,   "Q Surface",      "Q"),
    ]:
        fig = plt.figure(figsize=(8, 6))
        _surface(fig.add_subplot(111, projection="3d"), X, Y, Z, title, zlab)
        writer.add_image(f"{tag}", _plot_to_image(fig), step)
        if png_dir: fig.savefig(os.path.join(png_dir, f"{tag}_{step}.png")); plt.close(fig)

    for tag, Z, title, clab in [
        ("policy_contour", Z_act, "Policy Contour", "a (env)"),
        ("qgrad_contour",  Z_qg,  "∂Q/∂a Contour",  "∂Q/∂a"),
        ("q_contour",      Z_q,   "Q Contour",      "Q"),
    ]:
        fig, ax = plt.subplots(figsize=(6, 5))
        cf = _contour(ax, X, Y, Z, title, clab); plt.colorbar(cf, ax=ax)
        writer.add_image(f"{tag}", _plot_to_image(fig), step)
        if png_dir: fig.savefig(os.path.join(png_dir, f"{tag}_{step}.png")); plt.close(fig)

    # ----------------------------------
    # 2. Rollout 기반 지표
    # ----------------------------------
    fluct, smooth, seqs, traj = [], [], [], []
    for _ in range(eval_eps):
        obs = env.reset()                       # NumPy
        done = False
        last_a, _ = model.predict(obs, deterministic=True)
        ep_seq = []
        while not done and len(ep_seq) < max_steps:
            a, _ = model.predict(obs, deterministic=True)  # NumPy
            obs, _, done, _ = env.step(a)                  # NumPy ↔ Gym
            ep_seq.append(a); traj.append(obs[0])
            fluct.append(np.linalg.norm(a - last_a)); last_a = a
        if ep_seq:
            seqs.append(np.asarray(ep_seq)); smooth.append(_fft_smoothness(ep_seq))

    writer.add_scalar("ep/action_fluct", np.mean(fluct) if fluct else 0, step)
    writer.add_scalar("ep/smooth",       np.mean(smooth) if smooth else 0, step)

    # FFT & trajectory 시각화
    if seqs:
        fig_fft = _avg_fft_plot(seqs)
        writer.add_image("ep/fft_avg", _plot_to_image(fig_fft), step)
        if png_dir: fig_fft.savefig(os.path.join(png_dir, f"fft_{step}.png")); plt.close(fig_fft)
    if traj:
        fig = plt.figure(figsize=(8,3)); plt.plot(traj); plt.axhline(0, color="r", ls="--")
        plt.title("Particle x₁ trajectory"); plt.xlabel("t"); plt.ylabel("x₁")
        writer.add_image("particle_traj", _plot_to_image(fig), step)
        if png_dir: fig.savefig(os.path.join(png_dir, f"traj_{step}.png")); plt.close(fig)

    env.close()
