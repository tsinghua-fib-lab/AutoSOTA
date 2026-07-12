import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LinearRegression
import warnings
import os

# 尝试导入 pypdf 用于合并
try:
    from pypdf import PdfWriter
    HAS_PYPDF = True
except ImportError:
    HAS_PYPDF = False

warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False


class GapAnalyzer:
    """
    模态差异(Gap)对成员推断攻击指标影响的综合分析工具
    """

    def __init__(self, excel_path, gap_col='gap'):
        self.df = pd.read_excel(excel_path)
        self.gap_col = gap_col

        # 自动识别所有分数列
        self.score_cols = [col for col in self.df.columns if '_Score' in col]

        # 全局强制转换为数值，失败变NaN
        for col in self.score_cols:
            self.df[col] = pd.to_numeric(self.df[col], errors='coerce')

        # 过滤掉全是NaN的列
        valid_before = len(self.score_cols)
        self.score_cols = [col for col in self.score_cols if self.df[col].notna().any()]
        skipped = valid_before - len(self.score_cols)
        if skipped > 0:
            print(f"   ⚠️ 初始化时跳过 {skipped} 个无法转换的列")

        print(f"📊 数据加载成功！")
        print(f"   - 总样本数: {len(self.df)}")
        print(f"   - 成员样本: {len(self.df[self.df['Sample_Type'] == 'Member'])}")
        print(f"   - 非成员样本: {len(self.df[self.df['Sample_Type'] == 'Non_Member'])}")
        print(f"   - 有效攻击指标: {len(self.score_cols)} 个")

    def compute_gap(self, audio_col='LOSS_BASED_audio_Score', visual_col='LOSS_BASED_visual_Score'):
        """
        计算模态差异 gap = audio - visual
        """
        if audio_col in self.df.columns and visual_col in self.df.columns:
            self.df[self.gap_col] = self.df[audio_col] - self.df[visual_col]
            print(f"✅ Gap 计算完成: {self.gap_col} = {audio_col} - {visual_col}")
            print(f"   Gap 范围: [{self.df[self.gap_col].min():.4f}, {self.df[self.gap_col].max():.4f}]")
        else:
            print(f"❌ 错误: 找不到列 {audio_col} 或 {visual_col}")

    def plot_correlation_heatmap(self, save_path='gap_correlation_heatmap.pdf'):
        print("\n📊 生成相关性热力图...")
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        for idx, sample_type in enumerate(['Member', 'Non_Member']):
            subset = self.df[self.df['Sample_Type'] == sample_type].copy()

            # 强制转换，过滤无效列
            for col in self.score_cols:
                subset[col] = pd.to_numeric(subset[col], errors='coerce')
            valid_cols = [col for col in self.score_cols if subset[col].notna().any()]
            skipped = set(self.score_cols) - set(valid_cols)
            if skipped:
                print(f"   ⚠️ [{sample_type}] 跳过无法转换的列: {skipped}")

            if not valid_cols:
                print(f"   ❌ [{sample_type}] 没有有效列，跳过热力图")
                continue

            corr_data = subset[[self.gap_col] + valid_cols].corr()
            gap_corr = corr_data[self.gap_col].drop(self.gap_col).to_frame()
            gap_corr.columns = ['Correlation']
            gap_corr.index = [i.split('_')[0] + '_' + i.split('_')[1] for i in gap_corr.index]

            sns.heatmap(gap_corr, annot=True, fmt='.3f', cmap='RdYlGn', center=0,
                        cbar_kws={'label': 'Correlation'}, ax=axes[idx],
                        vmin=-1, vmax=1, linewidths=0.5, cbar=True)

            axes[idx].set_title(f'{sample_type} Samples: Gap Correlation', fontsize=14, fontweight='bold')
            axes[idx].set_xlabel('')
            axes[idx].set_ylabel('Attack Methods', fontsize=11)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"   ✅ 保存至: {save_path}")
        plt.close()

    def plot_scatter_with_trend(self, save_path='gap_impact_scatter.pdf'):
        print("\n📊 生成散点图（带趋势线）...")
        n_scores = len(self.score_cols)
        if n_scores == 0:
            print("   ❌ 没有有效的分数列，跳过散点图")
            return

        n_cols = 3
        n_rows = (n_scores + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6 * n_rows))
        axes = axes.flatten() if n_scores > 1 else [axes]

        for idx, score_col in enumerate(self.score_cols):
            ax = axes[idx]
            members = self.df[self.df['Sample_Type'] == 'Member'].dropna(subset=[self.gap_col, score_col])
            non_members = self.df[self.df['Sample_Type'] == 'Non_Member'].dropna(subset=[self.gap_col, score_col])

            ax.scatter(members[self.gap_col], members[score_col],
                       alpha=0.4, s=25, c='#E74C3C', label='Member', edgecolors='none')
            ax.scatter(non_members[self.gap_col], non_members[score_col],
                       alpha=0.4, s=25, c='#3498DB', label='Non-Member', edgecolors='none')

            for data, color, linestyle in [(members, '#C0392B', '-'), (non_members, '#2874A6', '--')]:
                if len(data) > 1:
                    try:
                        z = np.polyfit(data[self.gap_col], data[score_col], 1)
                        p = np.poly1d(z)
                        x_trend = np.linspace(data[self.gap_col].min(), data[self.gap_col].max(), 100)
                        ax.plot(x_trend, p(x_trend), linestyle, color=color, linewidth=2.5, alpha=0.8)
                    except:
                        pass

            attack_name = score_col.split('_')[0]
            modality = score_col.split('_')[1] if len(score_col.split('_')) > 1 else 'full'
            ax.set_xlabel('Modality Gap (Audio - Visual)', fontsize=11)
            ax.set_ylabel('Attack Score', fontsize=11)
            ax.legend(fontsize=9, loc='best')
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.set_title(f'{attack_name} ({modality})', fontsize=12, fontweight='bold')

        for idx in range(n_scores, len(axes)):
            fig.delaxes(axes[idx])

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"   ✅ 保存至: {save_path}")
        plt.close()

    def plot_binned_violin(self, score_cols=None, n_bins=5, save_dir='./'):
        """所有指标放在一张图里，每个指标一个子图"""
        print("\n📊 生成分箱小提琴图...")
        if score_cols is None:
            score_cols = self.score_cols
        if not score_cols:
            print("   ❌ 没有有效的分数列，跳过小提琴图")
            return

        self.df['gap_bin'] = pd.qcut(self.df[self.gap_col], q=n_bins,
                                     labels=[f'Q{i + 1}' for i in range(n_bins)],
                                     duplicates='drop')

        n_cols = 3
        n_rows = (len(score_cols) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows))
        axes = axes.flatten() if len(score_cols) > 1 else [axes]

        for idx, score_col in enumerate(score_cols):
            ax = axes[idx]
            attack_name = score_col.split('_')[0]
            modality = score_col.split('_')[1] if len(score_col.split('_')) > 1 else 'full'

            try:
                sns.violinplot(data=self.df, x='gap_bin', y=score_col, hue='Sample_Type',
                               split=True, palette={'Member': '#E74C3C', 'Non_Member': '#3498DB'},
                               ax=ax, inner='quartile', linewidth=1.5)
            except Exception as e:
                print(f"   ⚠️ {score_col} 小提琴图生成失败: {e}，跳过")
                continue

            ax.set_xlabel('Modality Gap Level', fontsize=11)
            ax.set_ylabel('Score', fontsize=11)
            ax.set_title(f'{attack_name} ({modality})', fontsize=12, fontweight='bold')
            ax.legend(fontsize=9)

        # 隐藏多余子图
        for idx in range(len(score_cols), len(axes)):
            fig.delaxes(axes[idx])

        plt.tight_layout()
        save_path = f'{save_dir}/3_gap_violin_all.pdf'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"   ✅ 保存至: {save_path}")
        plt.close()
        self.df.drop(columns=['gap_bin'], inplace=True)

    def plot_separation_comparison(self, n_bins=4, save_path='gap_separation_comparison.pdf'):
        print("\n📊 生成攻击分离度对比图...")
        if not self.score_cols:
            print("   ❌ 没有有效的分数列，跳过分离度对比图")
            return

        self.df['gap_bin'] = pd.qcut(self.df[self.gap_col], q=n_bins,
                                     labels=['Low', 'Med-Low', 'Med-High', 'High'][:n_bins], duplicates='drop')

        results = []
        for bin_name in self.df['gap_bin'].unique():
            if pd.isna(bin_name): continue
            bin_data = self.df[self.df['gap_bin'] == bin_name]
            for score_col in self.score_cols:
                member_mean = bin_data[bin_data['Sample_Type'] == 'Member'][score_col].mean()
                non_member_mean = bin_data[bin_data['Sample_Type'] == 'Non_Member'][score_col].mean()
                results.append({'Gap_Bin': bin_name, 'Attack': score_col.split('_')[0],
                                 'Separation': member_mean - non_member_mean})

        results_df = pd.DataFrame(results)
        fig, ax = plt.subplots(figsize=(16, 7))
        gap_bins, attacks = results_df['Gap_Bin'].unique(), results_df['Attack'].unique()
        x, width = np.arange(len(gap_bins)), 0.8 / len(attacks)
        colors = plt.cm.Set3(np.linspace(0, 1, len(attacks)))

        for i, attack in enumerate(attacks):
            attack_data = results_df[results_df['Attack'] == attack]
            values = [attack_data[attack_data['Gap_Bin'] == gb]['Separation'].values[0] if len(
                attack_data[attack_data['Gap_Bin'] == gb]) > 0 else 0 for gb in gap_bins]
            ax.bar(x + i * width, values, width, label=attack, alpha=0.85, color=colors[i])

        ax.set_xlabel('Modality Gap Level', fontsize=13)
        ax.set_ylabel('Score Separation', fontsize=13)
        ax.set_title('Attack Effectiveness vs Modality Gap', fontsize=15, fontweight='bold')
        ax.set_xticks(x + width * (len(attacks) - 1) / 2)
        ax.set_xticklabels(gap_bins)
        ax.legend(fontsize=10, ncol=3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"   ✅ 保存至: {save_path}")
        plt.close()
        self.df.drop(columns=['gap_bin'], inplace=True)
        return results_df

    def compute_correlation_stats(self):
        print("\n📊 计算相关系数...")
        for sample_type in ['Member', 'Non_Member']:
            subset = self.df[self.df['Sample_Type'] == sample_type]
            print(f"\n  [{sample_type}]")
            for score_col in self.score_cols:
                data = subset[[self.gap_col, score_col]].dropna()
                if len(data) < 3:
                    print(f"    {score_col}: 数据不足，跳过")
                    continue
                try:
                    pearson_r, pearson_p = pearsonr(data[self.gap_col], data[score_col])
                    spearman_r, spearman_p = spearmanr(data[self.gap_col], data[score_col])
                    print(f"    {score_col}: Pearson={pearson_r:.4f}(p={pearson_p:.4f}), "
                          f"Spearman={spearman_r:.4f}(p={spearman_p:.4f})")
                except Exception as e:
                    print(f"    {score_col}: 计算失败 ({e})，跳过")

    def compute_auc_by_gap_bins(self, score_col=None, n_bins=5):
        if score_col is None:
            if not self.score_cols:
                return
            score_col = self.score_cols[0]
        print(f"\n📊 按Gap分箱计算AUC ({score_col})...")
        self.df['gap_bin'] = pd.qcut(self.df[self.gap_col], q=n_bins,
                                     labels=[f'Q{i + 1}' for i in range(n_bins)],
                                     duplicates='drop')
        for bin_name in self.df['gap_bin'].unique():
            if pd.isna(bin_name): continue
            bin_data = self.df[self.df['gap_bin'] == bin_name].dropna(subset=[score_col])
            labels = (bin_data['Sample_Type'] == 'Member').astype(int)
            if labels.nunique() < 2:
                print(f"  {bin_name}: 标签单一，跳过")
                continue
            try:
                auc = roc_auc_score(labels, bin_data[score_col])
                print(f"  {bin_name}: AUC = {auc:.4f}")
            except Exception as e:
                print(f"  {bin_name}: AUC计算失败 ({e})")
        self.df.drop(columns=['gap_bin'], inplace=True)

    def compute_regression_analysis(self, score_col=None):
        if score_col is None:
            if not self.score_cols:
                return
            score_col = self.score_cols[0]
        print(f"\n📊 回归分析 ({score_col})...")
        data = self.df[[self.gap_col, score_col]].dropna()
        if len(data) < 3:
            print("   数据不足，跳过回归分析")
            return
        try:
            X = data[[self.gap_col]].values
            y = data[score_col].values
            reg = LinearRegression().fit(X, y)
            print(f"   斜率: {reg.coef_[0]:.4f}, 截距: {reg.intercept_:.4f}, R²: {reg.score(X, y):.4f}")
        except Exception as e:
            print(f"   回归分析失败: {e}")

    def merge_all_pdfs(self, output_dir, final_filename="Final_Gap_Analysis_Report.pdf"):
        if not HAS_PYPDF:
            print("\n❌ 无法合并PDF: 未安装 pypdf。请运行 'pip install pypdf'")
            return

        print(f"\n🔄 开始合并 PDF 文件到 {final_filename}...")

        pdf_files = []
        for file in os.listdir(output_dir):
            if file.endswith(".pdf") and file != final_filename:
                pdf_files.append(os.path.join(output_dir, file))

        pdf_files.sort()

        if not pdf_files:
            print("⚠️ 未找到可合并的 PDF 文件。")
            return

        try:
            writer = PdfWriter()
            for pdf in pdf_files:
                print(f"   + 添加页面: {os.path.basename(pdf)}")
                writer.append(pdf)

            output_path = os.path.join(output_dir, final_filename)
            with open(output_path, 'wb') as f:
                writer.write(f)
            print(f"✅ 🎉 成功！所有图表已合并至: {output_path}")
        except Exception as e:
            print(f"❌ 合并失败: {str(e)}")

    def run_full_analysis(self, output_dir='./gap_analysis_results'):
        os.makedirs(output_dir, exist_ok=True)

        # 清理旧的PDF文件，防止重复合并
        for f in os.listdir(output_dir):
            if f.endswith(".pdf"):
                os.remove(os.path.join(output_dir, f))

        print("\n" + "=" * 80)
        print("🚀 开始完整的 Gap 影响分析")
        print("=" * 80)

        # 1. 相关性热力图
        self.plot_correlation_heatmap(save_path=f'{output_dir}/1_correlation_heatmap.pdf')

        # 2. 散点图
        self.plot_scatter_with_trend(save_path=f'{output_dir}/2_scatter_trends.pdf')

        # 3. 小提琴图（所有指标放一张图）
        self.plot_binned_violin(score_cols=self.score_cols, save_dir=output_dir)

        # 4. 分离度对比图
        self.plot_separation_comparison(save_path=f'{output_dir}/4_separation_comparison.pdf')

        # 5. 相关系数统计
        self.compute_correlation_stats()

        # 6. AUC分析
        if self.score_cols:
            self.compute_auc_by_gap_bins(score_col=self.score_cols[0])

        # 7. 回归分析
        if self.score_cols:
            self.compute_regression_analysis(score_col=self.score_cols[0])

        # 合并PDF
        self.merge_all_pdfs(output_dir)

        print("\n" + "=" * 80)
        print(f"✅ 分析全部完成！结果保存至: {output_dir}")
        print("=" * 80)


# ==================== 主函数 ====================
def main():
    folder_path = r"E:\machang\code\FedMIA-main\FedMIA-main\saved_mia_models\cremad_K5_N1100_AVClassifer_defnone_iid$1_$1.0_$sgd_local3_s42_mod$multimodal_random"
    excel_path = 'Combined_Attack_Scores_Epoch_40_mix.xlsx'
    output_dir = './gap_analysis_results'
    full_path = os.path.join(folder_path, excel_path)

    analyzer = GapAnalyzer(full_path)

    analyzer.compute_gap(
        audio_col='LOSS_BASED_audio_Score',
        visual_col='LOSS_BASED_visual_Score'
    )

    analyzer.run_full_analysis(output_dir=output_dir)


if __name__ == "__main__":
    main()