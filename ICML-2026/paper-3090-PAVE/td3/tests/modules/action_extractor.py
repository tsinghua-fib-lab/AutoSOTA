import sys
import os
import gymnasium as gym
import numpy as np
import csv
import traceback
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from .envs import make_ant_env, make_hopper_env, make_humanoid_env, make_lunar_env, make_pendulum_env, make_reacher_env, make_walker_env, make_cheetah_env

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from models.custom_td3 import CustomTD3
# from visualiz_func import _generate_all #gym이 없어서 임시로 죽임
from torch.utils.tensorboard import SummaryWriter

max_loop = 10
seed_file_path = "./sac/tests/validation_seeds.txt"

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
    "walker" : make_walker_env,
    "cheetah" : make_cheetah_env
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
        os.path.join(os.path.join(save_dir, fname), 'final.zip')
        for fname in os.listdir(save_dir)
        if al_name in fname and os.path.isfile(os.path.join(os.path.join(save_dir, fname), 'final.zip'))
    ]

def test_vanilla(root_dir, al_name, env_name, deterministic=True, mode="rgb_array"):
    try:
        seeds = load_seeds(seed_file_path)
        counter = 0
        if env_name not in train_envs_dict:
            print(f"invalid env : {env_name}")
            return 0.0
        save_dir = root_dir + env_name + "/"
        env = train_envs_dict[env_name](mode)()
        files = find_matching_files(save_dir, al_name)
        actions_list = []
        rewards_list = []
        for filename in files:
            model = CustomTD3.load(filename, env=env)
            obs, info = env.reset(seed=seeds[counter])
            counter += 1
            for _ in range(max_loop):
                actions = []
                total_reward = 0.0
                while True:
                    action, _states = model.predict(obs, deterministic=deterministic)
                    actions.append(action)
                    obs, reward, terminated, truncated, info = env.step(action)
                    total_reward += reward
                    if terminated or truncated:
                        obs, info = env.reset(seed=seeds[counter])
                        counter += 1
                        break
                actions_list.append(np.array(actions))
                rewards_list.append(total_reward)
        smoothness_list = []
        for actions in actions_list:
            smoothness_list.append(calculate_smoothness_np(actions))
        if len(smoothness_list) == 0:
            print(f"invalid algorithm : {al_name}")
            return 0.0, 0.0, 0.0, 0.0, np.array(actions_list)
        else:
            smoothness_avg = np.mean(smoothness_list)
            smoothness_std = np.std(smoothness_list)
            mean_reward = np.mean(rewards_list)
            std_reward = np.std(rewards_list)
            return float(smoothness_avg), float(smoothness_std), float(mean_reward), float(std_reward), np.array(actions_list, dtype=object)
    except :
        return 0.0, 0.0, 0.0, 0.0, np.array([])
    
def find_matching_folder_viz(save_dir: str, al_name: str) -> list[str]:
    if not os.path.isdir(save_dir):
        return []
    return [
        os.path.join(save_dir, fname)
        for fname in os.listdir(save_dir)
        if al_name in fname and os.path.isdir(os.path.join(save_dir, fname))
    ]
    
def visualize_steps(root_dir, al_name, env_name, combined_path):
    mode="rgb_array"
    base_dir = os.path.join(combined_path, 'visualize')
    try:
        if env_name not in train_envs_dict:
            print(f"invalid env : {env_name}")
            return 0.0
        save_dir = os.path.join(root_dir, env_name)
        env_func = train_envs_dict[env_name](mode)
        folders = find_matching_folder_viz(save_dir, al_name)
        if len(folders) == 0:
            return
        for algo_dir in folders:
            algo_path = algo_dir
            
            # TensorBoard 로그 저장 경로: base_dir/env_name/algo_dir
            log_dir = os.path.join(base_dir, env_name, os.path.basename(algo_dir))
            os.makedirs(log_dir, exist_ok=True)
            writer = SummaryWriter(log_dir)

            # 평가할 모델 체크포인트들
            checkpoints = sorted([f for f in os.listdir(algo_path) if f.endswith(".zip") and f.startswith("mid")])
            
            for ckpt_file in checkpoints:
                ckpt_path = os.path.join(algo_path, ckpt_file)
                # 스텝 정보 추출
                step_str = ckpt_file.split("_")[1]
                step = int(step_str)

                # 모델 로드
                model = CustomTD3.load(ckpt_path)

                # 평가 및 TensorBoard 저장
                # _generate_all(model, env_func=env_func, writer=writer, step=step)

            writer.close()
    except Exception as e:
        print(f"[Exception] An error occurred during visualization: {e}")
        traceback.print_exc()
        return
    return

def test_some_path(root_dir, deterministic=True, add_al_names : list[str] = [], sub_dir=""):
    basic_al = []
    basic_envs = list(train_envs_dict.keys())

    all_al = basic_al + add_al_names

    sm_mean_rows = [["al_name"]+basic_envs]
    sm_std_rows = [["al_name"]+basic_envs]
    re_mean_rows = [["al_name"]+basic_envs]
    re_std_rows = [["al_name"]+basic_envs]

    combined_path = os.path.join(root_dir, sub_dir)
    os.makedirs(combined_path, exist_ok=True)
    
    for al_name in all_al:
        sm_mean_row = [al_name]
        sm_std_row = [al_name]
        re_mean_row = [al_name]
        re_std_row = [al_name]
        for env_name in basic_envs:
            smoothness_mean, smoothness_std, reward_mean, reward_std, action_np = test_vanilla(root_dir, al_name, env_name, deterministic)

            # visualize_steps(root_dir, al_name, env_name, combined_path)
            
            if isinstance(action_np, np.ndarray) and action_np.size != 0:
                comb_env_path = os.path.join(combined_path, env_name)
                os.makedirs(comb_env_path, exist_ok=True)
                np.savez(
                    os.path.join(comb_env_path, f"{al_name}_action_data.npz"),
                    action_list=action_np
                )

            sm_mean_row.append(smoothness_mean)
            sm_std_row.append(smoothness_std)
            re_mean_row.append(reward_mean)
            re_std_row.append(reward_std)
        sm_mean_rows.append(sm_mean_row)
        sm_std_rows.append(sm_std_row)
        re_mean_rows.append(re_mean_row)
        re_std_rows.append(re_std_row)

    
    with open(os.path.join(combined_path, "sm_mean.csv"), mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        # 3) 한 줄씩 쓰기
        for row in sm_mean_rows:
            writer.writerow(row)
    
    with open(os.path.join(combined_path, "sm_std.csv"), mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        # 3) 한 줄씩 쓰기
        for row in sm_std_rows:
            writer.writerow(row)
    
    with open(os.path.join(combined_path, "re_mean.csv"), mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        # 3) 한 줄씩 쓰기
        for row in re_mean_rows:
            writer.writerow(row)
    
    with open(os.path.join(combined_path, "re_std.csv"), mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        # 3) 한 줄씩 쓰기
        for row in re_std_rows:
            writer.writerow(row)

def predict_smooth(root_dir, deterministic=True, add_al_names : list[str] = []):
    basic_al = ["vanilla_hopper_model"]
    basic_envs = list(train_envs_dict.keys())

    all_al = basic_al + add_al_names

    rows = [["al_name"]+basic_envs]
    
    for al_name in all_al:
        al_row = [al_name]
        for env_name in basic_envs:
            smoothness = test_vanilla(root_dir, al_name, env_name, deterministic)
            al_row.append(smoothness)
        rows.append(al_row)

    with open(root_dir+"smoothness.csv", mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        # 3) 한 줄씩 쓰기
        for row in rows:
            writer.writerow(row)

    

    
