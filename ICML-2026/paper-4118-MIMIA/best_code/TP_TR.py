import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
import os
import numpy as np

# ====================================================================
# 配置
# ====================================================================
try:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    USE_CHINESE = True
except:
    USE_CHINESE = False
FOLDER_PATH = r"E:\machang\code\FedMIA-main\FedMIA-main\saved_mia_models\cremad_K6_N1100_AVClassifer_defnone_iid$1_$1.0_$sgd_local3_s42_mod$multimodal_random\train_excel"
# FOLDER_PATH = r"E:\machang\code\FedMIA-main\FedMIA-main\saved_mia_models\balance_K20_N1100_AVClassifer_defnone_iid$1_$1.0_$sgd_local3_s42_mod$multimodal_random\train_excel"

if not os.path.exists(FOLDER_PATH):
    os.makedirs(FOLDER_PATH)
FOLDER_PATH = r"E:\machang\code\FedMIA-main\FedMIA-main\saved_mia_models\cremad_K6_N1100_AVClassifer_defnone_iid$1_$1.0_$sgd_local3_s42_mod$multimodal_random\train_excel"
# ====================================================================
# 读取数据
# ====================================================================
csv_file = os.path.join(FOLDER_PATH, "Combined_Attack_Scores_Epoch_80.csv")
xlsx_file = os.path.join(FOLDER_PATH, "training_construct_0_[1, 2, 3, 4]_60_mix_lira_opti_one_attack_client_test.xlsx")
# xlsx_file = os.path.join(FOLDER_PATH, "training_construct_0_[1, 2]_100_mix_lira_opti.xlsx")


try:
    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file)
    elif os.path.exists(xlsx_file):
        df = pd.read_excel(xlsx_file)
    else:
        raise FileNotFoundError("未找到数据文件")
except Exception as e:
    print(f"❌ 错误: {e}")
    raise

df['is_member'] = (df['Sample_Type'] == 'Member').astype(int)
target = df['is_member'].values

metrics_to_plot = [
    'LOSS_full_Score', 'LOSS_audio_Score', 'LOSS_visual_Score',
    'COS_full_Score', 'COS_audio_Score', 'COS_visual_Score',
    'COSINE_ATTACK_full_Score', 'COSINE_ATTACK_audio_Score', 'COSINE_ATTACK_visual_Score',
    'GRAD_DIFF_full_Score', 'GRAD_DIFF_audio_Score', 'GRAD_DIFF_visual_Score',
    'LOSS_BASED_full_Score', 'LOSS_BASED_audio_Score', 'LOSS_BASED_visual_Score',
    'GRAD_NORM_full_Score', 'GRAD_NORM_audio_Score', 'GRAD_NORM_visual_Score',
    'gap'
]
available_metrics = [m for m in metrics_to_plot if m in df.columns]

# ====================================================================
# 核心计算循环
# ====================================================================
low_fpr_tpr_results = {}

plt.figure(figsize=(10, 8))
plt.plot([0, 1], [0, 1], 'k--', label='Random (AUC=0.5)')

for metric in available_metrics:
    try:
        raw_scores = pd.to_numeric(df[metric], errors='coerce').values
        if np.isnan(raw_scores).any() or len(raw_scores) == 0: continue

        # 1. 预计算 AUC 以确定指标方向
        # 如果 raw_auc < 0.5，说明指标越小越像 Member (如 Loss)
        raw_auc = roc_auc_score(target, raw_scores)

        if raw_auc < 0.5:
            # 反转分数：变成越大越像 Member
            scores_aligned = -raw_scores
            is_inverted = True
            final_auc = 1 - raw_auc
        else:
            scores_aligned = raw_scores
            is_inverted = False
            final_auc = raw_auc

        # 2. 基于对齐后的分数计算 ROC
        # ★★★ 关键：这一步生成的 fpr, tpr, thresholds 是严格对齐的 ★★★
        # thresholds[i] 对应 tpr[i] 和 fpr[i]
        fpr, tpr, thresholds = roc_curve(target, scores_aligned)

        # 3. 寻找最优阈值 (Youden Index)
        # J = Sensitivity + Specificity - 1 = TPR + (1-FPR) - 1 = TPR - FPR
        youden_index = tpr - fpr

        # 找到最大 J 值对应的索引
        best_idx = np.argmax(youden_index)

        # 4. ★★★ 严格使用同一个索引提取数据 ★★★
        optimal_tpr = tpr[best_idx]
        optimal_fpr = fpr[best_idx]
        optimal_tnr = 1 - optimal_fpr

        # 提取计算用的阈值
        thresh_calc = thresholds[best_idx]

        # 5. 还原阈值显示的符号
        # 如果之前取反了分数，这里把阈值符号反转回来，方便人类阅读原始数值
        if is_inverted:
            real_threshold_display = -thresh_calc
        else:
            real_threshold_display = thresh_calc

        # 6. 计算均衡精度
        balanced_acc = (optimal_tpr + optimal_tnr) / 2

        # 7. 计算低 FPR 指标 (用于画图和表格)
        # 需确保 FPR 排序正确 (roc_curve 返回的 fpr 默认是升序，所以可以直接用 searchsorted)
        tpr_at_low_fpr = {}
        for thresh_fpr in [0.001, 0.01]:
            idx = np.searchsorted(fpr, thresh_fpr)
            # idx 指向第一个 > thresh_fpr 的位置，所以取 idx-1
            val = tpr[idx - 1] if idx > 0 else 0.0
            tpr_at_low_fpr[f'TPR@{thresh_fpr * 100}%FPR'] = val

        # 8. 存储结果
        low_fpr_tpr_results[metric] = {
            'AUC_max': final_auc,
            'Balanced_Acc': balanced_acc,
            'Optimal_Threshold': real_threshold_display,
            'Optimal_TPR': optimal_tpr,
            'Optimal_FPR': optimal_fpr,
            **tpr_at_low_fpr
        }

        plt.plot(fpr, tpr, label=f'{metric} (AUC={final_auc:.4f})')
        print(
            f"✓ {metric}: B-Acc={balanced_acc:.4f}, Thresh={real_threshold_display:.4f}, TPR={optimal_tpr:.4f}, FPR={optimal_fpr:.4f}")

    except Exception as e:
        print(f"❌ {metric} Error: {e}")

# ====================================================================
# 保存与输出
# ====================================================================
if USE_CHINESE:
    plt.xlabel('假阳性率 (FPR)')
    plt.ylabel('真阳性率 (TPR)')
    plt.title('ROC 曲线')
else:
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('ROC Curves')

plt.legend(loc='lower right', fontsize='x-small')
plt.grid(True)
plt.savefig(os.path.join(FOLDER_PATH, 'ROC_Balanced.png'), dpi=300, bbox_inches='tight')
plt.close()
# 打印表格
print("\n" + "=" * 100)
if USE_CHINESE:
    print("              🎉 性能总结 (按 AUC 排序) 🎉")
else:
    print("              🎉 Performance Summary (Sorted by AUC) 🎉")
print("=" * 100)

summary_df = pd.DataFrame.from_dict(low_fpr_tpr_results, orient='index')
summary_df = summary_df.reset_index().rename(columns={'index': 'Metric'})
summary_df = summary_df.sort_values(by='AUC_max', ascending=False)

cols = ['Metric', 'AUC_max', 'Balanced_Acc', 'Optimal_Threshold', 'Optimal_TPR', 'Optimal_FPR', 'TPR@0.1%FPR',
        'TPR@1.0%FPR']
summary_df = summary_df[cols]

pd.set_option('display.max_rows', None)
pd.set_option('display.width', 1000)
pd.set_option('display.float_format', '{:.4f}'.format)

print(summary_df.to_string(index=False))

# 保存 CSV
csv_out = os.path.join(FOLDER_PATH, 'Results_Balanced.csv')
summary_df.to_csv(csv_out, index=False, encoding='utf_8_sig')

print("\n" + "=" * 100)
# 修复了这里的 if-else 缩进错误
if USE_CHINESE:
    print("说明:")
    print("  • Balanced_Acc: 均衡精度 = (Optimal_TPR + (1 - Optimal_FPR)) / 2")
    print("  • Optimal_Threshold: 使得均衡精度达到最大的那个截断值")
    print("  • Optimal_TPR/FPR: 严格对应于该最优阈值下的真阳性率和假阳性率")
    print("  • AUC_max: 自动校正方向后的 AUC (>=0.5)")
else:
    print("Notes:")
    print("  • Balanced_Acc: (TPR + TNR) / 2 at optimal threshold")
    print("  • Optimal_TPR/FPR: Strictly corresponding to the single optimal threshold")
    print("  • AUC_max: Auto-corrected AUC")
print("=" * 100)