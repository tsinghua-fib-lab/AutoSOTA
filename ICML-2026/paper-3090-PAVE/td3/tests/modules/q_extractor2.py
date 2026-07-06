import sys
import os
import gymnasium as gym
import numpy as np
import csv
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from .envs import make_ant_env, make_hopper_env, make_humanoid_env, make_lunar_env, make_pendulum_env, make_reacher_env, make_walker_env

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from stable_baselines3 import TD3
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
    matched_paths = []
    # save_dir 내의 모든 항목 탐색
    for fname in os.listdir(save_dir):
        full_dir_path = os.path.join(save_dir, fname)
        
        # 1. 파일 이름에 알고리즘 이름이 포함되어 있고
        if al_name in fname:
            # Case A: 폴더 구조인 경우 (Code B 방식)
            if os.path.isdir(full_dir_path):
                try:
                    subfiles = [f for f in os.listdir(full_dir_path) if f.endswith(".zip")]
                except OSError: continue
                
                target_zip = None
                if "final.zip" in subfiles: target_zip = "final.zip"
                elif "best_model.zip" in subfiles: target_zip = "best_model.zip"
                elif subfiles: target_zip = subfiles[0] # 아무 zip이나 선택
                
                if target_zip:
                    matched_paths.append(os.path.join(fname, target_zip)) # 상대 경로 유지를 위해 fname 결합
            
            # Case B: 그냥 파일인 경우 (Code A 방식 호환)
            elif os.path.isfile(full_dir_path) and fname.endswith(".zip"):
                matched_paths.append(fname)
                
    return matched_paths

    

def to_tensor(x: np.ndarray) -> th.Tensor:
    x = np.asarray(x, dtype=np.float32).reshape(1, -1)
    return th.as_tensor(x, dtype=th.float32, device=DEVICE)

def test_q(root_dir, al_name, env_name, deterministic=True, mode="rgb_array"):
    try:
        seeds = load_seeds(seed_file_path)
        counter = 0
        if env_name not in train_envs_dict:
            print(f"invalid env : {env_name}")
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 # Return 6 values
        save_dir = root_dir + env_name + "/"
        env = train_envs_dict[env_name](mode)()
        files = find_matching_files(save_dir, al_name)
        
        q_ss_list = []
        q_sa_list = []
        q_aa_list = [] # [Added] AA list

        for filename in files:
            model = TD3.load(f"{save_dir}{filename}", env=env)
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
                    grad_aa_sum = 0.0 # [Added]

                    # 1. ∂Q / ∂a
                    grad_a = th.autograd.grad(
                        q1,
                        action_tensor,
                        create_graph=True,
                        retain_graph=True
                    )[0]   # shape: (1, action_dim)

                    # [Added] 2. ∂²Q / ∂a∂a (Trace)
                    # grad_a를 다시 action_tensor로 미분하여 대각 성분(Curvature) 합 계산
                    for i in range(grad_a.shape[1]):
                        grad_aa_i = th.autograd.grad(
                            grad_a[0, i],
                            action_tensor,
                            create_graph=True,
                            retain_graph=True
                        )[0]
                        grad_aa_sum += grad_aa_i[0, i].item()

                    # 3. ∂²Q / ∂s∂a
                    grad_sa = th.autograd.grad(
                        grad_a.sum(),
                        obs_tensor,
                        create_graph=True,
                        retain_graph=True # retain_graph 추가 권장
                    )[0]   # shape: (1, obs_dim)
                    grad_sa_sum += th.norm(grad_sa).item()

                    # 4. ∂²Q / ∂s∂s (Trace)
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
                    q_aa_list.append(abs(grad_aa_sum)) # [Added] 절대값 크기 저장

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
        q_aa_avg = np.mean(q_aa_list) # [Added]
        q_aa_std = np.std(q_aa_list)  # [Added]
        
        return float(q_ss_avg), float(q_ss_std), float(q_sa_avg), float(q_sa_std), float(q_aa_avg), float(q_aa_std)

    except Exception as e:
        print("ERROR:", e)
        raise
    return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0


def test_some_path(root_dir, deterministic=True, add_al_names : list[str] = [], sub_dir=""):
    basic_al = []
    basic_envs = list(train_envs_dict.keys())

    all_al = basic_al + add_al_names

    nabla_ss_mean_rows = [["al_name"]+basic_envs]
    nabla_ss_std_rows = [["al_name"]+basic_envs]
    nabla_sa_mean_rows = [["al_name"]+basic_envs]
    nabla_sa_std_rows = [["al_name"]+basic_envs]
    # [Added] AA Rows
    nabla_aa_mean_rows = [["al_name"]+basic_envs]
    nabla_aa_std_rows = [["al_name"]+basic_envs]

    combined_path = os.path.join(root_dir, sub_dir)
    os.makedirs(combined_path, exist_ok=True)

    print("Testing Q-function gradients (SS, SA, AA)...")
    
    for al_name in all_al:
        nabla_ss_mean_row = [al_name]
        nabla_ss_std_row = [al_name]
        nabla_sa_mean_row = [al_name]
        nabla_sa_std_row = [al_name]
        nabla_aa_mean_row = [al_name] # [Added]
        nabla_aa_std_row = [al_name]  # [Added]
        
        for env_name in basic_envs:
            # [Updated] Unpack 6 values
            ss_m, ss_s, sa_m, sa_s, aa_m, aa_s = test_q(root_dir, al_name, env_name, deterministic)

            nabla_ss_mean_row.append(ss_m)
            nabla_ss_std_row.append(ss_s)
            nabla_sa_mean_row.append(sa_m)
            nabla_sa_std_row.append(sa_s)
            nabla_aa_mean_row.append(aa_m) # [Added]
            nabla_aa_std_row.append(aa_s)  # [Added]
            
        nabla_ss_mean_rows.append(nabla_ss_mean_row)
        nabla_ss_std_rows.append(nabla_ss_std_row)
        nabla_sa_mean_rows.append(nabla_sa_mean_row)
        nabla_sa_std_rows.append(nabla_sa_std_row)
        nabla_aa_mean_rows.append(nabla_aa_mean_row) # [Added]
        nabla_aa_std_rows.append(nabla_aa_std_row)   # [Added]

    # Write existing CSVs
    with open(os.path.join(combined_path, "nabla_ss_mean.csv"), mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f); writer.writerows(nabla_ss_mean_rows) # Simplified logic
    with open(os.path.join(combined_path, "nabla_ss_std.csv"), mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f); writer.writerows(nabla_ss_std_rows)
    with open(os.path.join(combined_path, "nabla_sa_mean.csv"), mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f); writer.writerows(nabla_sa_mean_rows)
    with open(os.path.join(combined_path, "nabla_sa_std.csv"), mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f); writer.writerows(nabla_sa_std_rows)

    # [Added] Write AA CSVs
    with open(os.path.join(combined_path, "nabla_aa_mean.csv"), mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f); writer.writerows(nabla_aa_mean_rows)
    with open(os.path.join(combined_path, "nabla_aa_std.csv"), mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f); writer.writerows(nabla_aa_std_rows)
    

    
