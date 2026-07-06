import sys
import os
import gymnasium as gym
import numpy as np
import csv
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from .envs import make_ant_env, make_hopper_env, make_humanoid_env, make_lunar_env, make_pendulum_env, make_reacher_env, make_walker_env

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from models.custom_sac import CustomSAC
from models.lips_sac import LipsSAC, LipsSACPolicy
import torch as th

max_loop = 10
seed_file_path = "./sac/tests/validation_seeds.txt"
DEVICE = "cuda" if th.cuda.is_available() else "cpu"

def load_seeds(filepath):
    with open(filepath, "r") as f:
        return [int(line.strip()) for line in f if line.strip()]

train_envs_dict = dict({
    "ant" : make_ant_env,
    "hopper" : make_hopper_env,
    "humanoid" : make_humanoid_env,
    "lunar" : make_lunar_env,
    "pendulum" : make_pendulum_env,
    "reacher" : make_reacher_env,
    "walker" : make_walker_env
})

def calculate_smoothness_np(actions: np.ndarray, fs: float = 1.0) -> float:
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


def calculate_oscillation_np(actions: np.ndarray) -> float:
    """
    NumPy 로 구현한 oscillation 지표.
    actions: shape (T,) 또는 (T, d)
    """
    a = np.array(actions, dtype=float)
    # 차원 상관없이 앞뒤 차이를 절대값으로 평균
    diffs = np.abs(a[1:] - a[:-1])  # shape (T-1,) or (T-1, d)
    return float(np.mean(diffs))

def find_matching_files(save_dir: str, al_name: str) -> list[str]:
    if not os.path.isdir(save_dir):
        return []
    return [
        fname
        for fname in os.listdir(save_dir)
        if al_name in fname and os.path.isfile(os.path.join(save_dir, fname))
    ]
    

def to_tensor(x: np.ndarray) -> th.Tensor:
    x = np.asarray(x, dtype=np.float32).reshape(1, -1)
    return th.as_tensor(x, dtype=th.float32, device=DEVICE)

def test_q(root_dir, al_name, env_name, deterministic=True, mode="rgb_array"):
    try:
        seeds = load_seeds(seed_file_path)
        counter = 0
        if env_name not in train_envs_dict:
            print(f"invalid env : {env_name}")
            return 0.0, 0.0, 0.0, 0.0
        save_dir = root_dir + env_name + "/"
        env = train_envs_dict[env_name](mode)()
        files = find_matching_files(save_dir, al_name)
        q_ss_list = []
        q_sa_list = []
        for filename in files:
            model = CustomSAC.load(f"{save_dir}{filename}", env=env)
            obs, info = env.reset(seed=seeds[counter])
            counter += 1
            for _ in range(max_loop):
                total_reward = 0.0
                while True:
                    action, _states = model.predict(obs, deterministic=deterministic)
                    ## q gradient
                    obs_tensor, _ = model.policy.obs_to_tensor(obs)
                    action_tensor = th.as_tensor(action, device=obs_tensor.device).unsqueeze(0)
                    obs_tensor.requires_grad_(True)
                    action_tensor.requires_grad_(True)

                    q1_pred, q2_pred = model.critic(obs_tensor, action_tensor)

                    q1 = q1_pred.sum()

                    grad_sa_sum = 0.0
                    grad_ss_sum = 0.0

                    # ∂Q / ∂a
                    grad_a = th.autograd.grad(
                        q1,
                        action_tensor,
                        create_graph=True,
                        retain_graph=True
                    )[0]   # shape: (1, action_dim)

                    # ∂²Q / ∂s∂a
                    grad_sa = th.autograd.grad(
                        grad_a.sum(),
                        obs_tensor,
                        create_graph=True
                    )[0]   # shape: (1, obs_dim)
                    grad_sa_sum += th.norm(grad_sa).item()

                    # nabla ss는 너무 커서 그냥 trace만 구할거임
                    grad_s = th.autograd.grad(
                        q1,
                        obs_tensor,
                        create_graph=True,
                        retain_graph=True
                    )[0]

                    trace = 0.0
                    for i in range(grad_s.shape[1]):
                        grad2 = th.autograd.grad(
                            grad_s[0, i],
                            obs_tensor,
                            retain_graph=True
                        )[0]
                        trace += grad2[0, i].item()

                    grad_ss_sum += abs(trace)

                    q_ss_list.append(grad_ss_sum)
                    q_sa_list.append(grad_sa_sum)

                    # print(f"grad_ss: {grad_ss_sum}, grad_sa: {grad_sa_sum}")

                    obs, reward, terminated, truncated, info = env.step(action)
                    total_reward += reward
                    if terminated or truncated:
                        obs, info = env.reset(seed=seeds[counter])
                        counter += 1
                        break
        q_ss_avg = np.mean(q_ss_list)
        q_ss_std = np.std(q_ss_list)
        q_sa_avg = np.mean(q_sa_list)
        q_sa_std = np.std(q_sa_list)
        return float(q_ss_avg), float(q_ss_std), float(q_sa_avg), float(q_sa_std)
    except Exception as e:
        print("ERROR:", e)
        raise
    return 0.0, 0.0, 0.0, 0.0


def test_some_path(root_dir, deterministic=True, add_al_names : list[str] = [], sub_dir=""):
    basic_al = []
    basic_envs = list(train_envs_dict.keys())
    # basic_envs = ["hopper"]

    all_al = basic_al + add_al_names

    nabla_ss_mean_rows = [["al_name"]+basic_envs]
    nabla_ss_std_rows = [["al_name"]+basic_envs]
    nabla_sa_mean_rows = [["al_name"]+basic_envs]
    nabla_sa_std_rows = [["al_name"]+basic_envs]

    combined_path = os.path.join(root_dir, sub_dir)
    os.makedirs(combined_path, exist_ok=True)

    print("Testing Q-function gradients...")
    
    for al_name in all_al:
        nabla_ss_mean_row = [al_name]
        nabla_ss_std_row = [al_name]
        nabla_sa_mean_row = [al_name]
        nabla_sa_std_row = [al_name]
        for env_name in basic_envs:
            smoothness_mean, smoothness_std, reward_mean, reward_std = test_q(root_dir, al_name, env_name, deterministic)

            nabla_ss_mean_row.append(smoothness_mean)
            nabla_ss_std_row.append(smoothness_std)
            nabla_sa_mean_row.append(reward_mean)
            nabla_sa_std_row.append(reward_std)
        nabla_ss_mean_rows.append(nabla_ss_mean_row)
        nabla_ss_std_rows.append(nabla_ss_std_row)
        nabla_sa_mean_rows.append(nabla_sa_mean_row)
        nabla_sa_std_rows.append(nabla_sa_std_row)

    
    with open(os.path.join(combined_path, "nabla_ss_mean.csv"), mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        # 3) 한 줄씩 쓰기
        for row in nabla_ss_mean_rows:
            writer.writerow(row)
    
    with open(os.path.join(combined_path, "nabla_ss_std.csv"), mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        # 3) 한 줄씩 쓰기
        for row in nabla_ss_std_rows:
            writer.writerow(row)
    
    with open(os.path.join(combined_path, "nabla_sa_mean.csv"), mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        # 3) 한 줄씩 쓰기
        for row in nabla_sa_mean_rows:
            writer.writerow(row)
    
    with open(os.path.join(combined_path, "nabla_sa_std.csv"), mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        # 3) 한 줄씩 쓰기
        for row in nabla_sa_std_rows:
            writer.writerow(row)

    

    
