import sys
import os
import gymnasium as gym
import numpy as np
import csv
import torch as th
from stable_baselines3 import TD3

# =========================================================
# 기본 설정
# =========================================================
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
try:
    from .envs import make_ant_env, make_hopper_env, make_humanoid_env, make_lunar_env, make_pendulum_env, make_reacher_env, make_walker_env
except ImportError:
    pass 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# [설정] Robustness 측정용
NUM_TEST_EPISODES = 20  # 통계적 신뢰도를 위해 20회 추천
SEED_FILE_PATH = "./sac/tests/validation_seeds.txt"
DEVICE = "cuda" if th.cuda.is_available() else "cpu"
ROBUSTNESS_SIGMAS = [0.01, 0.025, 0.05, 0.1]

# =========================================================
# Helper Functions
# =========================================================
def load_seeds(filepath):
    if not os.path.exists(filepath): return [0] * 100
    with open(filepath, "r") as f: return [int(line.strip()) for line in f if line.strip()]

train_envs_dict = dict({
    "ant" : make_ant_env, "hopper" : make_hopper_env, "humanoid" : make_humanoid_env,
    "lunar" : make_lunar_env, "pendulum" : make_pendulum_env, "reacher" : make_reacher_env, "walker" : make_walker_env
})

def estimate_obs_scale(model, env, num_episodes=10):
    """ 관측값의 스케일(표준편차) 측정 """
    all_obs = []
    for _ in range(num_episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            all_obs.append(obs)
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
    all_obs = np.array(all_obs)
    obs_std = np.std(all_obs, axis=0)
    obs_std = np.maximum(obs_std, 1e-6) 
    return obs_std

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


def test_robustness(model, env, noise_sigma, obs_scale=None, num_episodes=5):
    """ Scale-Aware Noise Injection Test """
    all_rewards = []
    all_sm_scores = []
    
    if hasattr(env, 'observation_space') and isinstance(env.observation_space, gym.spaces.Box):
        low, high = env.observation_space.low, env.observation_space.high
    else:
        low, high = -np.inf, np.inf

    for _ in range(num_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        actions_in_episode = []
        
        while True:
            # Scale-Aware Noise 생성
            base_noise = np.random.uniform(-noise_sigma, noise_sigma, size=obs.shape)
            if obs_scale is not None:
                scaled_noise = base_noise * obs_scale
            else:
                scaled_noise = base_noise
                
            noisy_obs = np.clip(obs + scaled_noise, low, high)
            action, _ = model.predict(noisy_obs, deterministic=True)
            
            # 환경은 원본 obs에서 전이
            obs, reward, terminated, truncated, _ = env.step(action)
            
            episode_reward += reward
            actions_in_episode.append(action)
            
            if terminated or truncated:
                all_rewards.append(episode_reward)
                all_sm_scores.append(calculate_smoothness_np(np.array(actions_in_episode)))
                break
                
    return float(np.mean(all_rewards)), float(np.mean(all_sm_scores))

def find_matching_files(save_dir: str, al_name: str) -> list[str]:
    if not os.path.isdir(save_dir): return []
    matched_paths = []
    for fname in os.listdir(save_dir):
        full_dir_path = os.path.join(save_dir, fname)
        if os.path.isdir(full_dir_path) and al_name in fname:
            try:
                subfiles = [f for f in os.listdir(full_dir_path) if f.endswith(".zip")]
            except OSError: continue
            
            target_zip = None
            if "final.zip" in subfiles: target_zip = "final.zip"
            elif "best_model.zip" in subfiles: target_zip = "best_model.zip"
            elif subfiles: target_zip = subfiles[0]
            
            if target_zip:
                matched_paths.append(os.path.join(full_dir_path, target_zip))
    return matched_paths

GLOBAL_BASE_SCALES = {} 

# =========================================================
# test_q (Scale Locking 적용됨)
# =========================================================
def test_q(root_dir, al_name, env_name, deterministic=True, mode="rgb_array", output_dir=""):
    global GLOBAL_BASE_SCALES  # 전역 변수 사용

    try:
        seeds = load_seeds(SEED_FILE_PATH)
        
        if env_name not in train_envs_dict: return
        save_dir = os.path.join(root_dir, env_name)
        files = find_matching_files(save_dir, al_name)
        if not files: return
        
        env = train_envs_dict[env_name](mode)()
        robust_agg = {sigma: {'re': [], 'sm': []} for sigma in ROBUSTNESS_SIGMAS}

        for i, model_path in enumerate(files):
            print(i,model_path)
            try:
                model = TD3.load(model_path, env=env)
                print(f"   [Processing] {al_name} ({env_name}) | Seed {i+1}/{len(files)}")
            except: continue

            # ============================================================
            # [수정] Scale Locking 로직 (최소 수정)
            # ============================================================
            is_base = "base" in al_name.lower() # 이름에 base가 들어가는지 확인
            
            # 1. Base 모델이거나, 아직 등록된 스케일이 없으면 측정
            if is_base or env_name not in GLOBAL_BASE_SCALES:
                current_obs_scale = estimate_obs_scale(model, env, num_episodes=20)
                # Base라면 이 값을 '표준'으로 등록 (덮어쓰기)
                if is_base: 
                    GLOBAL_BASE_SCALES[env_name] = current_obs_scale
            # 2. 그 외 모델은 Base가 등록해둔 스케일 강제 사용
            else:
                current_obs_scale = GLOBAL_BASE_SCALES[env_name]
            # ============================================================
            
            for sigma in ROBUSTNESS_SIGMAS:
                re_val, sm_val = test_robustness(model, env, sigma, 
                                                 obs_scale=current_obs_scale, 
                                                 num_episodes=NUM_TEST_EPISODES)
                robust_agg[sigma]['re'].append(re_val)
                robust_agg[sigma]['sm'].append(sm_val)

        # CSV 저장
        final_robust_rows = []
        for sigma in ROBUSTNESS_SIGMAS:
            rewards = robust_agg[sigma]['re']
            smoothness = robust_agg[sigma]['sm']
            final_robust_rows.append([
                al_name, env_name, sigma, 
                np.mean(rewards), np.std(rewards), 
                np.mean(smoothness), np.std(smoothness)
            ])

        if final_robust_rows and output_dir:
            os.makedirs(output_dir, exist_ok=True)
            rob_path = os.path.join(output_dir, f"robust_stats_{al_name}_{env_name}.csv")
            with open(rob_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["Algorithm", "Env", "Noise Sigma", "Mean Reward", "Std Reward", "Mean Smoothness", "Std Smoothness"])
                writer.writerows(final_robust_rows)
            print(f"      ✅ Stats Saved: {rob_path}")

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

# =========================================================
# test_some_path (실행 순서 정렬 추가)
# =========================================================
def test_some_path(root_dir, deterministic=True, add_al_names : list[str] = [], sub_dir="", target_envs=None):
    if target_envs: basic_envs = target_envs
    else: basic_envs = list(train_envs_dict.keys())
    
    combined_path = os.path.join(root_dir, sub_dir)
    robust_save_dir = os.path.join(combined_path, "robustness_data")
    os.makedirs(robust_save_dir, exist_ok=True)
    
    print(f"\n📂 Data Output Directory: {os.path.abspath(robust_save_dir)}\n")
    print("Starting Pure Robustness Test (Scale Locked to Base)...")

    # [수정] 'base'가 포함된 알고리즘을 무조건 리스트 맨 앞으로 보냄 (먼저 실행되게)
    add_al_names.sort(key=lambda x: 0 if "base" in x.lower() else 1)
    
    for al_name in add_al_names:
        for env_name in basic_envs:
            test_q(root_dir, al_name, env_name, deterministic, "rgb_array", output_dir=robust_save_dir)
            
    print("\n✅ All Robustness Tests Completed. Check 'robustness_data' folder.")