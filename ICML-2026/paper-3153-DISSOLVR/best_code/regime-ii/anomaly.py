"""
Threshold Recommendation Tool for Solubility Model
===================================================

Analyzes prediction error distribution to recommend optimal thresholds for:
- Good predictions (accurate): error < good_threshold
- Bad predictions (inaccurate): error > bad_threshold

This tool helps you determine the best thresholds to use in enhanced_explain_pipeline.py
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import json
import joblib
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, confusion_matrix, roc_curve
)
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns


class ThresholdRecommender:
    """Analyzes prediction errors and recommends optimal thresholds."""
    
    def __init__(self, 
                 test_file="data/test.csv",
                 model_dir="model",
                 store_dir="feature_store",
                 transformer_path="transformer.pth",
                 num_good_samples=15,
                 num_bad_samples=15):
        """
        Args:
            test_file: Path to test data
            model_dir: Directory with trained model
            store_dir: Feature store directory
            transformer_path: Path to transformer model
            num_good_samples: Target number of good samples needed
            num_bad_samples: Target number of bad samples needed
        """
        self.test_file = test_file
        self.model_dir = model_dir
        self.store_dir = store_dir
        self.transformer_path = transformer_path
        self.num_good_samples = num_good_samples
        self.num_bad_samples = num_bad_samples
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def load_data_and_models(self):
        """Load all required data and models."""
        print("Loading data and models...")
        
        # Load test data
        self.df_test = pd.read_csv(self.test_file)
        
        # Load feature stores
        self.sol_raw = pd.read_parquet(
            os.path.join(self.store_dir, "solute_raw.parquet")
        ).set_index("SMILES_KEY")
        self.solv_raw = pd.read_parquet(
            os.path.join(self.store_dir, "solvent_raw.parquet")
        ).set_index("SMILES_KEY")
        self.sol_council = pd.read_parquet(
            os.path.join(self.store_dir, "solute_council.parquet")
        ).set_index("SMILES_KEY")
        self.solv_council = pd.read_parquet(
            os.path.join(self.store_dir, "solvent_council.parquet")
        ).set_index("SMILES_KEY")
        
        # Load transformer
        from train_transformer import InteractionTransformer
        self.transformer = InteractionTransformer().to(self.device)
        self.transformer.load_state_dict(
            torch.load(self.transformer_path, map_location=self.device)
        )
        self.transformer.eval()
        
        # Load CatBoost model and selector
        self.catboost_model = joblib.load(
            os.path.join(self.model_dir, "model.joblib")
        )
        self.selector = joblib.load(
            os.path.join(self.model_dir, "selector.joblib")
        )
        
        print(f"  ✓ Loaded {len(self.df_test)} test samples")
        
    def generate_features(self, df):
        """Generate features for samples."""
        X_sol = self.sol_council.loc[df["Solute"]].values.astype(np.float32)
        X_solv = self.solv_council.loc[df["Solvent"]].values.astype(np.float32)
        
        embeds = []
        with torch.no_grad():
            for i in range(len(df)):
                sol = torch.tensor(X_sol[i:i+1]).to(self.device)
                solv = torch.tensor(X_solv[i:i+1]).to(self.device)
                _, feats, _ = self.transformer(sol, solv)
                embeds.append(feats.cpu().numpy())
        
        X_embed = np.vstack(embeds)
        
        T = df["Temperature"].values.reshape(-1, 1)
        T_inv = (1000 / df["Temperature"]).values.reshape(-1, 1)
        Tm = self.sol_raw.loc[df["Solute"], "pred_Tm"].values.reshape(-1, 1)
        T_red = T / Tm
        
        X_reshaped = X_embed.reshape(-1, 24, 32)
        X_mod = np.linalg.norm(X_reshaped, axis=2)
        X_sign = np.sign(X_reshaped.mean(axis=2))
        X_interact = (X_sign * X_mod) * T_inv
        
        X_raw = np.hstack([
            self.sol_raw.loc[df["Solute"]].values,
            self.solv_raw.loc[df["Solvent"]].values
        ])
        
        X_full = np.hstack([X_raw, X_interact, Tm, T_red, T, T_inv])
        return X_full
    
    def compute_predictions_and_errors(self):
        """Generate predictions and compute errors."""
        print("\nGenerating predictions...")
        
        X_full = self.generate_features(self.df_test)
        X_pruned = self.selector.transform(X_full)
        predictions = self.catboost_model.predict(X_pruned)
        
        # Calculate errors
        y_true = self.df_test["LogS"].values
        errors = np.abs(y_true - predictions)
        
        print(f"  ✓ Computed {len(errors)} predictions")
        
        return {
            'y_true': y_true,
            'y_pred': predictions,
            'errors': errors,
            'X_pruned': X_pruned
        }
    
    
    def compute_feature_thresholds(self, num_std=2.0):
        """
        Compute per-feature thresholds using mean ± num_std * std from ENTIRE database.
        
        Args:
            num_std: Number of standard deviations for threshold (default: 2.0)
            
        Returns:
            Dictionary mapping feature names to threshold statistics
        """
        print(f"\n" + "="*70)
        print(f"COMPUTING PER-FEATURE THRESHOLDS (±{num_std} std)")
        print("="*70)
        print("\nUsing ENTIRE database (train + test) for robust statistics...")
        
        # Load both train and test datasets
        df_train = pd.read_csv(self.test_file.replace("test.csv", "train.csv"))
        df_test = pd.read_csv(self.test_file)
        
        # Combine into full dataset
        df_full = pd.concat([df_train, df_test], ignore_index=True)
        print(f"  ✓ Loaded {len(df_train)} train + {len(df_test)} test = {len(df_full)} total samples")
        
        # Generate features for the full dataset
        print(f"  → Generating features for {len(df_full)} samples...")
        X_full = self.generate_features(df_full)
        X_pruned_full = self.selector.transform(X_full)
        print(f"  ✓ Generated {X_pruned_full.shape[1]} features")
        
        # Get feature names from the selector
        feature_map = pd.read_csv(
            os.path.join(self.model_dir.replace("model", "cgboost_explanations"), 
                        "trained_feature_map.csv")
        )
        feature_names = feature_map["Feature"].tolist()
        
        # Compute statistics for each feature across FULL dataset
        feature_thresholds = {}
        
        print(f"  → Computing statistics across all {len(df_full)} samples...")
        for i, feat_name in enumerate(feature_names):
            feat_values = X_pruned_full[:, i]
            
            # Compute mean and std
            mean_val = np.mean(feat_values)
            std_val = np.std(feat_values)
            
            # Compute thresholds
            low_threshold = mean_val - num_std * std_val
            high_threshold = mean_val + num_std * std_val
            
            feature_thresholds[feat_name] = {
                'mean': float(mean_val),
                'std': float(std_val),
                'low': float(low_threshold),
                'high': float(high_threshold),
                'min': float(np.min(feat_values)),
                'max': float(np.max(feat_values)),
                'n_samples': int(len(feat_values))
            }
        
        print(f"\n  ✓ Computed thresholds for {len(feature_thresholds)} features")
        print(f"  ✓ Using ±{num_std} standard deviations from mean")
        print(f"  ✓ Based on {len(df_full)} samples (full database)")
        
        # Show examples for key features
        print(f"\n  Example thresholds (first 10 features):")
        for i, (feat, stats) in enumerate(list(feature_thresholds.items())[:10]):
            print(f"    {feat}:")
            print(f"      Range: [{stats['low']:.4f}, {stats['high']:.4f}]")
            print(f"      Mean: {stats['mean']:.4f}, Std: {stats['std']:.4f}")
        
        return feature_thresholds

    
    def analyze_error_distribution(self, errors):
        """Analyze error distribution and compute statistics."""
        print("\n" + "="*70)
        print("ERROR DISTRIBUTION ANALYSIS")
        print("="*70)
        
        # Basic statistics
        print(f"\n📊 Basic Statistics:")
        print(f"  Total samples: {len(errors)}")
        print(f"  Min error: {errors.min():.4f}")
        print(f"  Max error: {errors.max():.4f}")
        print(f"  Mean error: {errors.mean():.4f}")
        print(f"  Median error: {errors.median():.4f}")
        print(f"  Std error: {errors.std():.4f}")
        
        # Percentiles
        print(f"\n📈 Percentile Distribution:")
        percentiles = [5, 10, 25, 50, 75, 90, 95, 99]
        for p in percentiles:
            val = np.percentile(errors, p)
            count = (errors <= val).sum()
            print(f"  {p:2d}th percentile: {val:.4f} ({count:4d} samples, {count/len(errors)*100:5.1f}%)")
        
        # Error ranges
        print(f"\n🎯 Error Range Distribution:")
        ranges = [
            (0.0, 0.1, "Excellent"),
            (0.1, 0.2, "Very Good"),
            (0.2, 0.3, "Good"),
            (0.3, 0.5, "Fair"),
            (0.5, 0.75, "Moderate"),
            (0.75, 1.0, "Poor"),
            (1.0, 1.5, "Very Poor"),
            (1.5, float('inf'), "Extremely Poor")
        ]
        
        for low, high, label in ranges:
            if high == float('inf'):
                mask = errors >= low
                range_str = f"≥ {low:.2f}"
            else:
                mask = (errors >= low) & (errors < high)
                range_str = f"{low:.2f}-{high:.2f}"
            count = mask.sum()
            pct = count / len(errors) * 100
            print(f"  {range_str:12s} ({label:14s}): {count:4d} samples ({pct:5.1f}%)")
    
    def recommend_thresholds(self, errors):
        """Recommend optimal thresholds based on different criteria."""
        print("\n" + "="*70)
        print("THRESHOLD RECOMMENDATIONS")
        print("="*70)
        
        recommendations = []
        
        # Strategy 1: Percentile-based (fixed percentiles)
        print("\n📌 Strategy 1: Percentile-Based Thresholds")
        print("  Description: Use fixed percentiles of the error distribution")
        
        percentile_configs = [
            (25, 75, "Conservative: Q1 for good, Q3 for bad"),
            (33, 90, "Balanced: 33rd for good, 90th for bad"),
            (40, 85, "Moderate: 40th for good, 85th for bad"),
        ]
        
        for good_pct, bad_pct, desc in percentile_configs:
            good_thresh = np.percentile(errors, good_pct)
            bad_thresh = np.percentile(errors, bad_pct)
            
            n_good = (errors <= good_thresh).sum()
            n_bad = (errors >= bad_thresh).sum()
            n_middle = ((errors > good_thresh) & (errors < bad_thresh)).sum()
            
            recommendations.append({
                'strategy': 'Percentile',
                'description': desc,
                'good_threshold': good_thresh,
                'bad_threshold': bad_thresh,
                'n_good': n_good,
                'n_bad': n_bad,
                'n_middle': n_middle
            })
            
            print(f"  {desc}")
            print(f"    good_threshold: {good_thresh:.4f} ({good_pct}th percentile)")
            print(f"    bad_threshold:  {bad_thresh:.4f} ({bad_pct}th percentile)")
            print(f"    → {n_good:4d} good samples, {n_bad:4d} bad samples, {n_middle:4d} middle\n")
        
        # Strategy 2: Target sample count (ensure you get enough samples)
        print("\n📌 Strategy 2: Target Sample Count")
        print(f"  Description: Ensure at least {self.num_good_samples} good and {self.num_bad_samples} bad samples")
        
        # Find thresholds that give exactly the target counts
        sorted_errors = np.sort(errors)
        
        # Good threshold: error at position (num_good_samples - 1)
        good_idx = min(self.num_good_samples - 1, len(sorted_errors) - 1)
        good_thresh_target = sorted_errors[good_idx]
        
        # Bad threshold: error at position (len - num_bad_samples)
        bad_idx = max(len(sorted_errors) - self.num_bad_samples, 0)
        bad_thresh_target = sorted_errors[bad_idx]
        
        # Count samples with some buffer
        n_good_target = (errors <= good_thresh_target).sum()
        n_bad_target = (errors >= bad_thresh_target).sum()
        n_middle_target = ((errors > good_thresh_target) & (errors < bad_thresh_target)).sum()
        
        recommendations.append({
            'strategy': 'Target Count',
            'description': f'Target {self.num_good_samples} good, {self.num_bad_samples} bad',
            'good_threshold': good_thresh_target,
            'bad_threshold': bad_thresh_target,
            'n_good': n_good_target,
            'n_bad': n_bad_target,
            'n_middle': n_middle_target
        })
        
        print(f"  Target {self.num_good_samples} good, {self.num_bad_samples} bad")
        print(f"    good_threshold: {good_thresh_target:.4f}")
        print(f"    bad_threshold:  {bad_thresh_target:.4f}")
        print(f"    → {n_good_target:4d} good samples, {n_bad_target:4d} bad samples, {n_middle_target:4d} middle\n")
        
        # Strategy 3: Natural breakpoints (using standard deviations)
        print("\n📌 Strategy 3: Statistical Breakpoints")
        print("  Description: Use mean ± standard deviation")
        
        mean_err = errors.mean()
        std_err = errors.std()
        
        stat_configs = [
            (mean_err - 0.5*std_err, mean_err + 1.0*std_err, "Mean ± 0.5/1.0 std"),
            (mean_err - std_err, mean_err + std_err, "Mean ± 1.0 std"),
        ]
        
        for good_thresh, bad_thresh, desc in stat_configs:
            # Ensure thresholds are valid
            good_thresh = max(0.0, good_thresh)
            bad_thresh = max(good_thresh + 0.1, bad_thresh)
            
            n_good = (errors <= good_thresh).sum()
            n_bad = (errors >= bad_thresh).sum()
            n_middle = ((errors > good_thresh) & (errors < bad_thresh)).sum()
            
            recommendations.append({
                'strategy': 'Statistical',
                'description': desc,
                'good_threshold': good_thresh,
                'bad_threshold': bad_thresh,
                'n_good': n_good,
                'n_bad': n_bad,
                'n_middle': n_middle
            })
            
            print(f"  {desc}")
            print(f"    good_threshold: {good_thresh:.4f}")
            print(f"    bad_threshold:  {bad_thresh:.4f}")
            print(f"    → {n_good:4d} good samples, {n_bad:4d} bad samples, {n_middle:4d} middle\n")
        
        # Strategy 4: Current enhanced_explain_pipeline.py defaults
        print("\n📌 Strategy 4: Current Pipeline Defaults")
        print("  Description: Currently used in enhanced_explain_pipeline.py")
        
        current_good = 0.30
        current_bad = 1.00
        n_good_current = (errors <= current_good).sum()
        n_bad_current = (errors >= current_bad).sum()
        n_middle_current = ((errors > current_good) & (errors < current_bad)).sum()
        
        recommendations.append({
            'strategy': 'Current',
            'description': 'enhanced_explain_pipeline.py defaults',
            'good_threshold': current_good,
            'bad_threshold': current_bad,
            'n_good': n_good_current,
            'n_bad': n_bad_current,
            'n_middle': n_middle_current
        })
        
        print(f"  Current defaults: good=0.30, bad=1.00")
        print(f"    good_threshold: {current_good:.4f}")
        print(f"    bad_threshold:  {current_bad:.4f}")
        print(f"    → {n_good_current:4d} good samples, {n_bad_current:4d} bad samples, {n_middle_current:4d} middle\n")
        
        return recommendations
    
    def find_optimal_threshold(self, errors, recommendations):
        """Identify the best recommended threshold."""
        print("\n" + "="*70)
        print("OPTIMAL RECOMMENDATION")
        print("="*70)
        
        # Score each recommendation
        # Criteria:
        # 1. Both good and bad should have enough samples (>= target)
        # 2. Balance between good and bad counts
        # 3. Reasonable separation (bad_thresh - good_thresh)
        
        scores = []
        for rec in recommendations:
            score = 0
            
            # Criterion 1: Sufficient samples
            if rec['n_good'] >= self.num_good_samples:
                score += 3
            elif rec['n_good'] >= self.num_good_samples * 0.8:
                score += 2
            elif rec['n_good'] >= self.num_good_samples * 0.5:
                score += 1
            
            if rec['n_bad'] >= self.num_bad_samples:
                score += 3
            elif rec['n_bad'] >= self.num_bad_samples * 0.8:
                score += 2
            elif rec['n_bad'] >= self.num_bad_samples * 0.5:
                score += 1
            
            # Criterion 2: Balance (prefer similar counts)
            balance_ratio = min(rec['n_good'], rec['n_bad']) / max(rec['n_good'], rec['n_bad'])
            score += balance_ratio * 2
            
            # Criterion 3: Separation (prefer thresholds that are not too close)
            separation = rec['bad_threshold'] - rec['good_threshold']
            if separation >= 0.5:
                score += 2
            elif separation >= 0.3:
                score += 1
            
            # Criterion 4: Prefer not too extreme thresholds
            if 0.2 <= rec['good_threshold'] <= 0.5:
                score += 1
            if 0.8 <= rec['bad_threshold'] <= 1.5:
                score += 1
            
            scores.append(score)
        
        best_idx = np.argmax(scores)
        best_rec = recommendations[best_idx]
        
        print(f"\n✨ RECOMMENDED THRESHOLDS (Strategy: {best_rec['strategy']}):")
        print(f"  {best_rec['description']}")
        print(f"\n  good_prediction_threshold = {best_rec['good_threshold']:.4f}")
        print(f"  bad_prediction_threshold  = {best_rec['bad_threshold']:.4f}")
        print(f"\n  Expected sample counts:")
        print(f"    Good predictions  (error < {best_rec['good_threshold']:.4f}): {best_rec['n_good']:4d} samples")
        print(f"    Bad predictions   (error > {best_rec['bad_threshold']:.4f}): {best_rec['n_bad']:4d} samples")
        print(f"    Middle range: {best_rec['n_middle']:4d} samples")
        print(f"\n  To use in enhanced_explain_pipeline.py:")
        print(f"    python enhanced_explain_pipeline.py \\")
        print(f"        --good-threshold {best_rec['good_threshold']:.4f} \\")
        print(f"        --bad-threshold {best_rec['bad_threshold']:.4f}")
        
        return best_rec, scores
    
    def plot_analysis(self, errors, recommendations, best_rec, output_dir="threshold_analysis"):
        """Generate visualization plots."""
        print(f"\n\nGenerating plots in {output_dir}/...")
        os.makedirs(output_dir, exist_ok=True)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Error distribution histogram
        axes[0, 0].hist(errors, bins=100, alpha=0.7, edgecolor='black')
        axes[0, 0].axvline(best_rec['good_threshold'], color='green', linestyle='--', 
                          linewidth=2, label=f'Good threshold: {best_rec["good_threshold"]:.3f}')
        axes[0, 0].axvline(best_rec['bad_threshold'], color='red', linestyle='--', 
                          linewidth=2, label=f'Bad threshold: {best_rec["bad_threshold"]:.3f}')
        axes[0, 0].set_xlabel('Absolute Error')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Error Distribution with Recommended Thresholds')
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)
        
        # 2. Cumulative distribution
        sorted_errors = np.sort(errors)
        cumulative = np.arange(1, len(errors) + 1) / len(errors) * 100
        axes[0, 1].plot(sorted_errors, cumulative, linewidth=2)
        axes[0, 1].axvline(best_rec['good_threshold'], color='green', linestyle='--', 
                          linewidth=2, alpha=0.7)
        axes[0, 1].axvline(best_rec['bad_threshold'], color='red', linestyle='--', 
                          linewidth=2, alpha=0.7)
        axes[0, 1].axhline(best_rec['n_good']/len(errors)*100, color='green', 
                          linestyle=':', alpha=0.5)
        axes[0, 1].axhline(100 - best_rec['n_bad']/len(errors)*100, color='red', 
                          linestyle=':', alpha=0.5)
        axes[0, 1].set_xlabel('Absolute Error')
        axes[0, 1].set_ylabel('Cumulative Percentage')
        axes[0, 1].set_title('Cumulative Error Distribution')
        axes[0, 1].grid(alpha=0.3)
        
        # 3. Sample counts for different threshold combinations
        good_thresholds = [rec['good_threshold'] for rec in recommendations]
        bad_thresholds = [rec['bad_threshold'] for rec in recommendations]
        n_goods = [rec['n_good'] for rec in recommendations]
        n_bads = [rec['n_bad'] for rec in recommendations]
        labels = [f"{rec['strategy']}\n{rec['description'][:30]}" for rec in recommendations]
        
        x = np.arange(len(recommendations))
        width = 0.35
        
        bars1 = axes[1, 0].bar(x - width/2, n_goods, width, label='Good samples', color='green', alpha=0.7)
        bars2 = axes[1, 0].bar(x + width/2, n_bads, width, label='Bad samples', color='red', alpha=0.7)
        
        axes[1, 0].axhline(self.num_good_samples, color='green', linestyle='--', 
                          alpha=0.5, label=f'Target good: {self.num_good_samples}')
        axes[1, 0].axhline(self.num_bad_samples, color='red', linestyle='--', 
                          alpha=0.5, label=f'Target bad: {self.num_bad_samples}')
        
        axes[1, 0].set_xlabel('Strategy')
        axes[1, 0].set_ylabel('Sample Count')
        axes[1, 0].set_title('Sample Counts by Threshold Strategy')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels([rec['strategy'] for rec in recommendations], rotation=45, ha='right')
        axes[1, 0].legend()
        axes[1, 0].grid(alpha=0.3, axis='y')
        
        # 4. Threshold values comparison
        axes[1, 1].scatter(good_thresholds, bad_thresholds, s=200, alpha=0.6, c=range(len(recommendations)), 
                          cmap='viridis')
        
        for i, (gt, bt, rec) in enumerate(zip(good_thresholds, bad_thresholds, recommendations)):
            axes[1, 1].annotate(rec['strategy'], (gt, bt), fontsize=9, ha='center')
        
        # Highlight best recommendation
        best_idx = recommendations.index(best_rec)
        axes[1, 1].scatter([best_rec['good_threshold']], [best_rec['bad_threshold']], 
                          s=400, facecolors='none', edgecolors='red', linewidths=3, 
                          label='Recommended')
        
        axes[1, 1].set_xlabel('Good Threshold')
        axes[1, 1].set_ylabel('Bad Threshold')
        axes[1, 1].set_title('Threshold Combinations')
        axes[1, 1].legend()
        axes[1, 1].grid(alpha=0.3)
        
        plt.tight_layout()
        plot_path = os.path.join(output_dir, 'threshold_analysis.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {plot_path}")
        plt.close()
        
    def generate_report(self, errors, recommendations, best_rec, feature_thresholds=None, output_dir="threshold_analysis"):
        """Generate comprehensive report."""
        os.makedirs(output_dir, exist_ok=True)
        
        # JSON report
        report = {
            'summary': {
                'total_samples': int(len(errors)),
                'min_error': float(errors.min()),
                'max_error': float(errors.max()),
                'mean_error': float(errors.mean()),
                'median_error': float(errors.median()),
                'std_error': float(errors.std())
            },
            'recommended_thresholds': {
                'good_threshold': float(best_rec['good_threshold']),
                'bad_threshold': float(best_rec['bad_threshold']),
                'strategy': best_rec['strategy'],
                'description': best_rec['description'],
                'expected_good_samples': int(best_rec['n_good']),
                'expected_bad_samples': int(best_rec['n_bad']),
                'expected_middle_samples': int(best_rec['n_middle'])
            },
            'all_recommendations': [
                {
                    'strategy': rec['strategy'],
                    'description': rec['description'],
                    'good_threshold': float(rec['good_threshold']),
                    'bad_threshold': float(rec['bad_threshold']),
                    'n_good': int(rec['n_good']),
                    'n_bad': int(rec['n_bad']),
                    'n_middle': int(rec['n_middle'])
                }
                for rec in recommendations
            ]
        }
        
        report_path = os.path.join(output_dir, 'threshold_recommendations.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\n✓ Saved JSON report: {report_path}")
        
        # Save feature thresholds separately (can be very large)
        if feature_thresholds is not None:
            feature_thresh_path = os.path.join(output_dir, 'feature_thresholds.json')
            with open(feature_thresh_path, 'w') as f:
                json.dump(feature_thresholds, f, indent=2)
            print(f"✓ Saved feature thresholds: {feature_thresh_path}")
            print(f"  → {len(feature_thresholds)} features with individual thresholds")
        
        # Text report
        text_path = os.path.join(output_dir, 'threshold_recommendations.txt')
        with open(text_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("THRESHOLD RECOMMENDATIONS FOR ENHANCED_EXPLAIN_PIPELINE.PY\n")
            f.write("="*70 + "\n\n")
            
            f.write("RECOMMENDED THRESHOLDS\n")
            f.write("-"*70 + "\n")
            f.write(f"Strategy: {best_rec['strategy']} - {best_rec['description']}\n\n")
            f.write(f"good_prediction_threshold = {best_rec['good_threshold']:.4f}\n")
            f.write(f"bad_prediction_threshold  = {best_rec['bad_threshold']:.4f}\n\n")
            f.write(f"Expected sample counts:\n")
            f.write(f"  Good predictions  (error < {best_rec['good_threshold']:.4f}): {best_rec['n_good']:4d} samples\n")
            f.write(f"  Bad predictions   (error > {best_rec['bad_threshold']:.4f}): {best_rec['n_bad']:4d} samples\n")
            f.write(f"  Middle range: {best_rec['n_middle']:4d} samples\n\n")
            
            f.write("USAGE IN ENHANCED_EXPLAIN_PIPELINE.PY\n")
            f.write("-"*70 + "\n")
            f.write(f"python enhanced_explain_pipeline.py \\\n")
            f.write(f"    --good-threshold {best_rec['good_threshold']:.4f} \\\n")
            f.write(f"    --bad-threshold {best_rec['bad_threshold']:.4f}\n\n")
            
            if feature_thresholds is not None:
                f.write("FEATURE THRESHOLDS\n")
                f.write("-"*70 + "\n")
                f.write(f"Computed {len(feature_thresholds)} per-feature thresholds using ±2 std from mean\n")
                f.write(f"Saved to: feature_thresholds.json\n")
                f.write(f"Use these thresholds in identify_unusual_features() function\n\n")
            
            f.write("\nALL RECOMMENDATIONS\n")
            f.write("-"*70 + "\n")
            for i, rec in enumerate(recommendations, 1):
                f.write(f"\n{i}. {rec['strategy']}: {rec['description']}\n")
                f.write(f"   good={rec['good_threshold']:.4f}, bad={rec['bad_threshold']:.4f}\n")
                f.write(f"   → {rec['n_good']:4d} good, {rec['n_bad']:4d} bad, {rec['n_middle']:4d} middle\n")
        
        print(f"✓ Saved text report: {text_path}")
        
        return report

    
    def run(self, output_dir="threshold_analysis"):
        """Run full threshold recommendation pipeline."""
        print("="*70)
        print("THRESHOLD RECOMMENDATION FOR ENHANCED_EXPLAIN_PIPELINE.PY")
        print("="*70)
        
        # Load data
        self.load_data_and_models()
        
        # Compute predictions and errors
        data = self.compute_predictions_and_errors()
        errors = pd.Series(data['errors'])
        
        # Compute per-feature thresholds from ENTIRE database (train + test)
        feature_thresholds = self.compute_feature_thresholds(num_std=2.0)

        
        # Analyze error distribution
        self.analyze_error_distribution(errors)
        
        # Recommend thresholds
        recommendations = self.recommend_thresholds(errors)
        
        # Find optimal
        best_rec, scores = self.find_optimal_threshold(errors, recommendations)
        
        # Generate plots
        self.plot_analysis(errors, recommendations, best_rec, output_dir)
        
        # Generate report (including feature thresholds)
        report = self.generate_report(errors, recommendations, best_rec, 
                                      feature_thresholds=feature_thresholds, 
                                      output_dir=output_dir)
        
        print("\n" + "="*70)
        print("ANALYSIS COMPLETE")
        print("="*70)
        print(f"\nOutput directory: {output_dir}")
        print(f"\nNext steps:")
        print(f"  1. Review the visualizations in {output_dir}/threshold_analysis.png")
        print(f"  2. Check the detailed report in {output_dir}/threshold_recommendations.txt")
        print(f"  3. Use feature_thresholds.json in identify_unusual_features()")
        print(f"  4. Use the recommended thresholds in enhanced_explain_pipeline.py")
        
        return report



# ============================================================================
# CLI ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Recommend optimal thresholds for enhanced_explain_pipeline.py"
    )
    parser.add_argument("--test-file", default="data/test.csv",
                       help="Path to test data")
    parser.add_argument("--model-dir", default="model",
                       help="Model directory")
    parser.add_argument("--store-dir", default="feature_store",
                       help="Feature store directory")
    parser.add_argument("--transformer-path", default="transformer.pth",
                       help="Path to transformer model")
    parser.add_argument("--num-good-samples", type=int, default=15,
                       help="Target number of good samples needed")
    parser.add_argument("--num-bad-samples", type=int, default=15,
                       help="Target number of bad samples needed")
    parser.add_argument("--output-dir", default="threshold_analysis",
                       help="Output directory for analysis results")
    
    args = parser.parse_args()
    
    recommender = ThresholdRecommender(
        test_file=args.test_file,
        model_dir=args.model_dir,
        store_dir=args.store_dir,
        transformer_path=args.transformer_path,
        num_good_samples=args.num_good_samples,
        num_bad_samples=args.num_bad_samples
    )
    
    recommender.run(output_dir=args.output_dir)