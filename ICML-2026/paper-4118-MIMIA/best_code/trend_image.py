import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
import sys

# ====================================================================
# 1. 配置参数与环境
# ====================================================================
MODE = "mix"
# 路径请确保正确
FOLDER_PATH = r"./random_cremad_5client_mix"

# 设置中文字体和样式（同步原始脚本）
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
# 如果还需要显示中文，可以写成： ['Times New Roman', 'SimSun'] (新罗马+宋体)
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
sns.set_style("whitegrid", {"font.family": "serif", "font.serif": ["Times New Roman"]})

# ====================================================================
# 2. 读取和预处理数据
# ====================================================================
excel_file = os.path.join(FOLDER_PATH, f"Combined_Attack_Scores_Epoch_50.xlsx")
try:
    df = pd.read_excel(excel_file)
except:
    df = pd.read_csv(os.path.join(FOLDER_PATH, f"Combined_Attack_Scores_Epoch_70_{MODE}.csv"))


# 定义模态偏好分类逻辑
def categorize_preference(row):
    if row['LOSS_BASED_audio_Score'] > row['LOSS_BASED_visual_Score']:
        return 'A-Preference (Audio Dominant)'
    else:
        return 'V-Preference (Visual Dominant)'


df['Preference_Category'] = df.apply(categorize_preference, axis=1)
df_a_pref = df[df['Preference_Category'] == 'A-Preference (Audio Dominant)']
df_v_pref = df[df['Preference_Category'] == 'V-Preference (Visual Dominant)']

metrics = {
    'LOSS_audio_Score': 'Audio (Loss-LiRA)',
    'LOSS_visual_Score': 'Visual (Loss-LiRA)',
    'LOSS_BASED_audio_Score': 'Audio (Loss-Based)',
    'LOSS_BASED_visual_Score': 'Visual (Loss-Based)'
}

# ====================================================================
# 3. 核心绘图：成员vs非成员归一化密度对比图 (配色还原版)
# ====================================================================
fig, axes = plt.subplots(4, 3, figsize=(20, 8.5))
# fig.suptitle(f'MODE: {MODE.upper()} - Member vs Non-Member Normalized Density',
#              fontsize=16, fontweight='bold', y=0.995)

for idx, (metric, label) in enumerate(metrics.items()):
    if metric not in df.columns: continue

    # 提取各子组数据
    subsets = [
        (df, "All Samples"),
        (df_a_pref, "A-Preference"),
        (df_v_pref, "V-Preference")
    ]

    for col_idx, (data_scope, scope_label) in enumerate(subsets):
        ax = axes[idx, col_idx]

        m_data = data_scope[data_scope['Sample_Type'] == 'Member'][metric].dropna()
        nm_data = data_scope[data_scope['Sample_Type'] != 'Member'][metric].dropna()

        # --- 绘制直方图 (严格匹配原始配色) ---
        ax.hist(m_data, bins=50, alpha=0.6, color='orange', edgecolor='black',
                density=True, label=f'Member')
        ax.hist(nm_data, bins=50, alpha=0.6, color='blue', edgecolor='black',
                density=True, label=f'Non-Member')

        # --- 绘制 KDE 曲线 (严格匹配原始线型和颜色) ---
        if len(m_data) > 1:
            kde_m = stats.gaussian_kde(m_data)
            x_m = np.linspace(m_data.min(), m_data.max(), 300)
            ax.plot(x_m, kde_m(x_m), color='darkorange', linewidth=2.5,
                    linestyle='--', label='Member KDE')

        if len(nm_data) > 1:
            kde_nm = stats.gaussian_kde(nm_data)
            x_nm = np.linspace(nm_data.min(), nm_data.max(), 300)
            ax.plot(x_nm, kde_nm(x_nm), color='darkblue', linewidth=2.5,
                    linestyle='--', label='Non-Member KDE')

        # 标题与标签
        title_suffix = "" if col_idx == 0 else ""
        ax.set_title(f'{label} - {scope_label} {title_suffix}', fontsize=15, fontweight='bold')
        ax.set_xlabel('Score', fontsize=15,fontweight='bold')
        ax.set_ylabel('Density ', fontsize=15,fontweight='bold')
        ax.legend(fontsize=11, loc='best')
        ax.grid(True, alpha=0.3)


plt.tight_layout()
output_file = os.path.join(FOLDER_PATH, f'{MODE}_metric_member_vs_nonmember_density_normalized.png')
plt.savefig(output_file, dpi=300, bbox_inches='tight')

print(f"✓ 图表已按原始配色重新生成并保存至: {output_file}")