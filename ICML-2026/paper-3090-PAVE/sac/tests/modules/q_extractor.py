import sys
import os
import gymnasium as gym
import numpy as np
import csv
import matplotlib.pyplot as plt
import torch as th
from mpl_toolkits.mplot3d import Axes3D
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from .envs import make_ant_env, make_hopper_env, make_humanoid_env, make_lunar_env, make_pendulum_env, make_reacher_env, make_walker_env

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from models.custom_sac import CustomSAC
from models.lips_sac import LipsSAC, LipsSACPolicy

max_loop = 10
seed_file_path = "./sac/tests/validation_seeds.txt"
DEVICE = "cuda" if th.cuda.is_available() else "cpu"

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

def find_matching_files(save_dir: str, al_name: str) -> list[str]:
    if not os.path.isdir(save_dir): return []
    matched_paths = []
    
    # [수정] 디렉토리가 아닌 해당 경로의 파일들을 바로 순회
    for fname in os.listdir(save_dir):
        full_path = os.path.join(save_dir, fname)
        
        # 1. 파일이어야 함 (.isfile)
        # 2. .zip 확장자여야 함
        # 3. 알고리즘 이름(al_name)이 파일명에 포함되어야 함
        if os.path.isfile(full_path) and fname.endswith(".zip") and al_name in fname:
            matched_paths.append(full_path)
            
    return matched_paths

def find_dominant_axis(model, env, seed):
    print("      [Axis] 🔍 Scanning for Strongest Interaction...")
    obs, _ = env.reset(seed=seed)
    for _ in range(20):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, trunc, _ = env.step(action)
        if done or trunc: obs, _ = env.reset(seed=seed)

    obs_tensor = th.as_tensor(obs, device=DEVICE).float().unsqueeze(0).requires_grad_(True)
    with th.no_grad(): base_action = model.predict(obs, deterministic=True)[0]
    action_tensor = th.as_tensor(base_action, device=DEVICE).float().unsqueeze(0).requires_grad_(True)
    
    q1, _ = model.critic(obs_tensor, action_tensor)
    grad_a = th.autograd.grad(q1.sum(), action_tensor, create_graph=True)[0]
    
    max_val, best_pair = -1.0, (0, 0)
    for a_idx in range(base_action.shape[0]):
        grad_sa = th.autograd.grad(grad_a[0, a_idx], obs_tensor, retain_graph=True)[0]
        abs_grads = th.abs(grad_sa[0])
        local_max = th.max(abs_grads).item()
        if local_max > max_val:
            max_val, best_pair = local_max, (th.argmax(abs_grads).item(), a_idx)
            
    print(f"      [Axis] ✅ Selected: State[{best_pair[0]}] <-> Action[{best_pair[1]}] (Score: {max_val:.4f})")
    return best_pair

# =========================================================
# Visualization Functions
# =========================================================
# def visualize_hessian_heatmap(model, obs, env_name, al_name, save_path, s_dim=0, a_dim=0):
#     print(f"      [Vis] Generating 2D & 3D Hessian Plots...")
#     res = 50 
#     eps = 1e-3
    
#     with th.no_grad():
#         base_action = model.predict(obs, deterministic=True)[0]

#     s_grid = np.linspace(obs[s_dim] - 1.0, obs[s_dim] + 1.0, res)
#     a_grid = np.linspace(base_action[a_dim] - 1.5, base_action[a_dim] + 1.5, res)
#     S, A = np.meshgrid(s_grid, a_grid)
#     num_points = res * res
    
#     def get_grad_a(states_np, actions_np):
#         s_tensor = th.as_tensor(states_np, device=DEVICE).float().requires_grad_(True)
#         a_tensor = th.as_tensor(actions_np, device=DEVICE).float().requires_grad_(True)
#         q, _ = model.critic(s_tensor, a_tensor)
#         grad_a = th.autograd.grad(q.sum(), a_tensor, create_graph=False)[0]
#         return grad_a[:, a_dim].detach().cpu().numpy()

#     batch_obs = np.tile(obs, (num_points, 1))
#     batch_obs[:, s_dim] = S.flatten()
#     batch_act = np.tile(base_action, (num_points, 1))
#     batch_act[:, a_dim] = A.flatten()
    
#     grad_a_orig = get_grad_a(batch_obs, batch_act)
#     batch_obs_eps = batch_obs.copy()
#     batch_obs_eps[:, s_dim] += eps
#     grad_a_eps = get_grad_a(batch_obs_eps, batch_act)
    
#     # 2D Hessian Data
#     numerical_hessian = np.abs((grad_a_eps - grad_a_orig) / eps).reshape(res, res)
    
#     # PDF Path Handling
#     if save_path.endswith(".png"): pdf_path = save_path.replace(".png", ".pdf")
#     else: pdf_path = save_path + ".pdf"

#     # 1. 2D Heatmap
#     plt.figure(figsize=(9, 8))
#     im = plt.imshow(numerical_hessian, extent=[s_grid.min(), s_grid.max(), a_grid.min(), a_grid.max()],
#                origin='lower', cmap='magma', aspect='auto', interpolation='nearest',
#                vmin=0, vmax=300) 
#     plt.colorbar(im, label=r'Approx. $| \nabla_{s,a}^2 Q |$ (Instability)')
#     plt.xlabel(f"State Dim {s_dim}")
#     plt.ylabel(f"Action Dim {a_dim}")
#     # plt.title(f"2D Instability: {al_name}")
#     plt.tight_layout()
#     plt.savefig(pdf_path, format='pdf', dpi=300, bbox_inches='tight', pad_inches=0.5)
#     plt.close()
    
#     # 2. 3D Surface Plot
#     pdf_path_3d = pdf_path.replace(".pdf", "_3d.pdf")
#     hessian_clipped = np.clip(numerical_hessian, 0, 300) 
    
#     fig_3d = plt.figure(figsize=(10, 8))
#     ax = fig_3d.add_subplot(111, projection='3d')
#     surf = ax.plot_surface(S, A, hessian_clipped, cmap='magma', 
#                            edgecolor='none', alpha=0.9, antialiased=True,
#                            vmin=0, vmax=300)
#     ax.set_zlim(0, 300)
#     ax.view_init(elev=45, azim=225)
#     ax.set_xlabel(f"State Dim {s_dim}")
#     ax.set_ylabel(f"Action Dim {a_dim}")
#     ax.set_zlabel(r'$||\nabla_{s,a} Q||$')    
#     # ax.set_title(f"3D Surface: {al_name}")
#     fig_3d.colorbar(surf, ax=ax, shrink=0.5, aspect=10, pad=0.1)
    
#     plt.savefig(pdf_path_3d, format='pdf', dpi=300, bbox_inches='tight', pad_inches=0.5)
#     plt.close()

#     print(f"      ✅ Saved: {pdf_path} & {pdf_path_3d}")
#     return numerical_hessian 


def visualize_hessian_heatmap(model, obs, env_name, al_name, save_path, s_dim=0, a_dim=0):
    print(f"      [Vis] Generating 2D & 3D Hessian Plots...")
    res = 50 
    eps = 1e-3
    DEVICE = next(model.critic.parameters()).device # 모델 디바이스 참조
    
    # --- 폰트 사이즈 설정 ---
    LABEL_SIZE = 40  # 축 이름 (State Dim 등)
    TICK_SIZE = 0   # 축 숫자 (0.5, 1.0 등)
    # ---------------------
    
    with th.no_grad():
        base_action = model.predict(obs, deterministic=True)[0]

    s_grid = np.linspace(obs[s_dim] - 1.0, obs[s_dim] + 1.0, res)
    a_grid = np.linspace(base_action[a_dim] - 1.5, base_action[a_dim] + 1.5, res)
    S, A = np.meshgrid(s_grid, a_grid)
    num_points = res * res
    
    def get_grad_a(states_np, actions_np):
        s_tensor = th.as_tensor(states_np, device=DEVICE).float().requires_grad_(True)
        a_tensor = th.as_tensor(actions_np, device=DEVICE).float().requires_grad_(True)
        q, _ = model.critic(s_tensor, a_tensor)
        grad_a = th.autograd.grad(q.sum(), a_tensor, create_graph=False)[0]
        return grad_a[:, a_dim].detach().cpu().numpy()

    batch_obs = np.tile(obs, (num_points, 1))
    batch_obs[:, s_dim] = S.flatten()
    batch_act = np.tile(base_action, (num_points, 1))
    batch_act[:, a_dim] = A.flatten()
    
    grad_a_orig = get_grad_a(batch_obs, batch_act)
    batch_obs_eps = batch_obs.copy()
    batch_obs_eps[:, s_dim] += eps
    grad_a_eps = get_grad_a(batch_obs_eps, batch_act)
    
    # 2D Hessian Data
    numerical_hessian = np.abs((grad_a_eps - grad_a_orig) / eps).reshape(res, res)
    
    # PDF Path Handling
    if save_path.endswith(".png"): pdf_path = save_path.replace(".png", ".pdf")
    else: pdf_path = save_path + ".pdf"

    # 1. 2D Heatmap
    plt.figure(figsize=(9, 8))
    im = plt.imshow(numerical_hessian, extent=[s_grid.min(), s_grid.max(), a_grid.min(), a_grid.max()],
               origin='lower', cmap='magma', aspect='auto', interpolation='nearest',
               vmin=0, vmax=300) 
    
    # 컬러바 폰트 설정
    cb = plt.colorbar(im)
    cb.set_label(r'Approx. $| \nabla_{s,a}^2 Q |$ (Instability)', fontsize=LABEL_SIZE)
    # cb.ax.tick_params(labelsize=TICK_SIZE)

    # 축 레이블 및 숫자 크기 설정
    # plt.xlabel(f"State Dim {s_dim}", fontsize=LABEL_SIZE)
    # plt.ylabel(f"Action Dim {a_dim}", fontsize=LABEL_SIZE)
    # plt.xticks(fontsize=TICK_SIZE)
    # plt.yticks(fontsize=TICK_SIZE)
    
    # plt.tight_layout()
    # plt.savefig(pdf_path, format='pdf', dpi=300, bbox_inches='tight', pad_inches=0.5)
    # plt.close()
    
# 2. 3D Surface Plot
    pdf_path_3d = pdf_path.replace(".pdf", "_3d.pdf")
    hessian_clipped = np.clip(numerical_hessian, 0, 300) 
    
    fig_3d = plt.figure(figsize=(8, 7)) 
    ax = fig_3d.add_subplot(111, projection='3d')
    
    # 배경 판 투명하게 설정
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

    surf = ax.plot_surface(S, A, hessian_clipped, cmap='magma', 
                           edgecolor='none', alpha=0.9, antialiased=True,
                           vmin=0, vmax=300)
    
    # 축 이름 설정
    ax.set_xlabel(f"State", fontsize=LABEL_SIZE, labelpad=10)
    ax.set_ylabel(f"Action", fontsize=LABEL_SIZE, labelpad=10)
    
    # [수정] Z축 라벨이 잘리지 않도록 labelpad를 조절 (값이 작을수록 그래프에 가까워짐)
    # 만약 위쪽이 잘린다면 pad 값을 5 이하로 낮추거나 음수(예: -2)를 시도해보세요.
    ax.set_zlabel(r'$||\nabla_{s,a} Q||$', fontsize=LABEL_SIZE, labelpad=5) 
    
    ax.tick_params(axis='both', which='major', labelsize=TICK_SIZE)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.set_zlim(0, 300)
    ax.view_init(elev=45, azim=225)
    
    # [수정] Z축 잘림 방지를 위해 zoom을 0.9 정도로 미세하게 조정
    # 0.98은 너무 꽉 차서 상단이 잘릴 확률이 높습니다.
    ax.set_box_aspect(None, zoom=0.9) 
    
    # 서브플롯 여백 제거
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    
    # 저장 시 여백 최소화
    plt.savefig(pdf_path_3d, 
                format='pdf', 
                dpi=300, 
                bbox_inches='tight', 
                pad_inches=0.1) # 잘림 방지를 위해 0.1 정도로 약간의 여유 부여
    plt.close()

# =========================================================
# test_q (Pure Visualization Mode)
# =========================================================
def test_q(root_dir, al_name, env_name, deterministic=True, mode="rgb_array", 
           save_vis=False, vis_dir="", fixed_axis=None):
    try:
        seeds = load_seeds(seed_file_path)
        
        if env_name not in train_envs_dict: return None
        save_dir = os.path.join(root_dir, env_name)
        files = find_matching_files(save_dir, al_name)
        if not files: return None
        
        env = train_envs_dict[env_name](mode)()
        current_axis = fixed_axis

        # [수정] 모든 Pretrained 모델 시드를 순회
        for i, model_path in enumerate(files):
            print(model_path,i)
            try:
                model = CustomSAC.load(model_path, env=env)
                # print(f"   [Processing] {al_name} Seed {i+1}/{len(files)}")
            except: continue

            # 시각화를 위한 고정된 상태값 (Validation Seed 1개만 사용)
            # 모든 시드가 동일한 환경 상태(seeds[0])에서 평가되므로 공정합니다.
            eval_seed = seeds[0] 
            obs, info = env.reset(seed=eval_seed)
            
            # 1. 축 찾기 (첫 번째 모델에서 기준 축 결정)
            if current_axis is None:
                current_axis = find_dominant_axis(model, env, eval_seed)
            
            # 2. 시각화 (이미지 저장)
            if save_vis and vis_dir:
                try:
                    # 파일명에 idx{i}를 추가하여 모든 시드 결과 보존
                    heat_path = os.path.join(vis_dir, f"vis_Heat_{al_name}_{env_name}_seed{eval_seed}_idx{i}.png")
                    visualize_hessian_heatmap(model, obs, env_name, al_name, heat_path, *current_axis)
                except Exception as e:
                    print(f"      [Warning] Vis Error at seed idx {i}: {e}")

        # 모든 파일을 다 돌고 최종 결정된(또는 전달받은) 축 반환
        return current_axis

    except Exception as e:
        print(f"ERROR: {e}")
        return None

# =========================================================
# test_some_path
# =========================================================
def test_some_path(root_dir, deterministic=True, add_al_names : list[str] = [], sub_dir="", visualize=True, target_envs=None):
    if target_envs: basic_envs = target_envs
    else: basic_envs = list(train_envs_dict.keys())
    
    # [수정] 현재 대상 알고리즘들 출력
    print(f"\n[DEBUG] 🚀 Starting Visual Test for: {add_al_names}")
    
    combined_path = os.path.join(root_dir, sub_dir)
    vis_root_dir = os.path.join(combined_path, "visualizations")
    if visualize:
        os.makedirs(vis_root_dir, exist_ok=True)
        print(f"\n📂 Vis Output Directory: {os.path.abspath(vis_root_dir)}\n")

    env_axis_map = {} 

    for idx, al_name in enumerate(add_al_names):
        # [수정] 알고리즘 이름 명확하게 출력
        print(f"\n[DEBUG] ▶ Processing Algorithm [{idx+1}/{len(add_al_names)}]: '{al_name}'")
        
        for env_name in basic_envs:
            env_vis_dir = os.path.join(vis_root_dir, env_name)
            if visualize: os.makedirs(env_vis_dir, exist_ok=True)

            fixed_axis = env_axis_map.get(env_name, None)
            
            # test_q 호출 (이제 루프 내부에서 모든 파일 처리)
            found_axis = test_q(root_dir, al_name, env_name, deterministic, "rgb_array", 
                                save_vis=visualize, vis_dir=env_vis_dir, fixed_axis=fixed_axis)
            
            # Axis Locking: 리스트의 첫 번째 알고리즘(보통 Base)이 찾은 축으로 고정
            if fixed_axis is None and found_axis is not None:
                env_axis_map[env_name] = found_axis
                print(f"   [System] 📌 Locking Axis for {env_name}: {found_axis}")

    print("\n✅ All Visualization Tests Completed.")