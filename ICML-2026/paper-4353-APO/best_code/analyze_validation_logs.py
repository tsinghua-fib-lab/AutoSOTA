import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import argparse
import numpy as np

def load_logs(log_dir: str) -> pd.DataFrame:
    """
    加载指定目录下所有的 seed_*.jsonl 文件到一个 DataFrame 中。
    
   
    """
    # 查找所有 seed 日志文件
    search_path = os.path.join(log_dir, "seed_*.jsonl")
    log_files = glob.glob(search_path)
    
    if not log_files:
        print(f"错误: 在 '{log_dir}' 中未找到 'seed_*.jsonl' 文件。")
        print("请确保您提供的 --log_dir 路径正确，")
        print(f"它应该是类似: results/validation_logs/SemiSynthNews/TheoreticalRMidpoint/")
        return pd.DataFrame()

    print(f"找到了 {len(log_files)} 个日志文件...")
    
    all_dfs = []
    for f_path in log_files:
        try:
            # 从文件名中提取 seed
            seed_str = os.path.basename(f_path).replace("seed_", "").replace(".jsonl", "")
            seed = int(seed_str)
            
            #
            df_seed = pd.read_json(f_path, lines=True) 
            df_seed['seed'] = seed
            all_dfs.append(df_seed)
        except Exception as e:
            print(f"警告: 无法加载或解析 {f_path}: {e}")

    if not all_dfs:
        print("错误: 无法从找到的文件中加载任何数据。")
        return pd.DataFrame()

    # 合并所有
    full_df = pd.concat(all_dfs, ignore_index=True)
    print(f"加载完成。总共 {len(full_df)} 行日志数据。")
    return full_df

def plot_metric_with_ci(ax, data_mean, data_sem, x_index, title, y_label, y_lim=None):
    """
    一个辅助函数，用于绘制均值和置信区间（均值 +/- 1 SEM）。
    """
    ax.plot(x_index, data_mean, marker='.')
    ax.fill_between(x_index, 
                    data_mean - data_sem, 
                    data_mean + data_sem, 
                    alpha=0.2, label="Mean +/- 1 SEM")
    ax.set_title(title, fontsize=10)
    ax.set_ylabel(y_label)
    ax.set_xlabel("Round (Active Learning Step)")
    ax.grid(True, linestyle=':')
    if y_lim:
        ax.set_ylim(y_lim)
    ax.legend()

def plot_bounds_comparison(ax, mean_df, sem_df, x_index):
    """
    一个专门的辅助函数，用于在同一张图上绘制三个界限。
   
    """
    metrics_to_plot = {
        'Actual_sup_sigma_plus': "Actual sup(σ+)",
        'Bound_Thm1_Prime': "Thm 1' Bound (Data-Dep.)",
        'Bound_Thm1_DoublePrime': "Thm 1'' Bound (Closed-Form)"
    }
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(metrics_to_plot)))
    
    for (metric, label), color in zip(metrics_to_plot.items(), colors):
        if metric in mean_df:
            ax.plot(x_index, mean_df[metric], label=label, color=color, marker='.')
            ax.fill_between(x_index, 
                            mean_df[metric] - sem_df[metric], 
                            mean_df[metric] + sem_df[metric], 
                            alpha=0.15, color=color)

    ax.set_title("Theorem 1'' Bounds vs. Actual Reduction", fontsize=10)
    ax.set_ylabel("sup(σ)")
    ax.set_xlabel("Round (Active Learning Step)")
    ax.grid(True, linestyle=':')
    ax.legend()


def plot_analysis(df: pd.DataFrame, output_dir: str):
    """
    对加载的数据进行分组、聚合，并绘制所有关键验证指标的图像。
   
    """
    if df.empty:
        print("DataFrame 为空，跳过绘图。")
        return

    # 按 'round' 分组，计算均值和标准误 (SEM)
    grouped = df.groupby('round')
    mean_df = grouped.mean(numeric_only=True)
    sem_df = grouped.sem(numeric_only=True) # 标准误 = std / sqrt(n_seeds)
    x_index = mean_df.index

    # --- 创建一个 3x2 的图表网格 ---
    fig, axes = plt.subplots(3, 2, figsize=(18, 22))
    fig.suptitle(f"Validation Analysis (Aggregated over {df['seed'].nunique()} seeds)", fontsize=16, y=1.02)
    
    # 1. A2 覆盖率
    plot_metric_with_ci(axes[0, 0], mean_df['A2_coverage_freq'], sem_df['A2_coverage_freq'], x_index,
                        "Hypothesis A2: Coverage Frequency", "Freq. |f-μ| <= βσ", y_lim=[0, 1.05])
    
    # 2. A1μ 强凹频率
    plot_metric_with_ci(axes[0, 1], mean_df['A1mu_strong_concave_freq'], sem_df['A1mu_strong_concave_freq'], x_index,
                        "Hypothesis A1μ: Strong Concavity Freq.", "Freq. m_I > 0", y_lim=[0, 1.05])

    # 3. A4 梯度比率
    plot_metric_with_ci(axes[1, 0], mean_df['A4_ratio_mean'], sem_df['A4_ratio_mean'], x_index,
                        "Hypothesis A4: Sigma Gradient Ratio", "Mean |σ'| / sup(σ)")

    # 4. κ_I (准平坦性)
    plot_metric_with_ci(axes[1, 1], mean_df['Thm1_kappa_I'], sem_df['Thm1_kappa_I'], x_index,
                        "Thm 1'' Condition: Quasi-Flatness (κ_I)", "Mean s_I / S_I", y_lim=[0, 1.05])

    # 5. 上界对比 (核心图表)
    plot_bounds_comparison(axes[2, 0], mean_df, sem_df, x_index)

    # 6. 上界间隙 (保守性)
    plot_metric_with_ci(axes[2, 1], mean_df['Gap_Actual_vs_Bound1PP'], sem_df['Gap_Actual_vs_Bound1PP'], x_index,
                        "Thm 1'' Conservatism (Gap)", "Gap = Bound_ClosedForm - Actual")

    # --- 保存图像 ---
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plot_path = os.path.join(output_dir, "validation_analysis_summary.png")
    fig.savefig(plot_path, dpi=150)
    print(f"\n分析图表已保存至: {plot_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Analyze validation .jsonl logs.")
    parser.add_argument("--log_dir", type=str, required=True,
                        help="Path to the directory containing 'seed_*.jsonl' files. "
                             "E.g., results/validation_logs/SemiSynthNews/TheoreticalRMidpoint/")
    
    args = parser.parse_args()

    # 定义输出目录
    output_dir = os.path.join(args.log_dir, "_analysis_plots")
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 加载数据
    df = load_logs(args.log_dir)
    
    # 2. 绘制分析图
    plot_analysis(df, output_dir)

if __name__ == "__main__":
    main()