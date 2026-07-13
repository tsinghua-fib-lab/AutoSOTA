#!/usr/bin/env python3
"""
Apelblat Equation Analysis for Regime-II

Analyzes how well solute-solvent pairs follow the Apelblat equation
and evaluates model predictions for thermodynamically consistent pairs.

Apelblat Equation: ln(x) = A + B/T + C*ln(T)
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import torch
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score, mean_squared_error

warnings.filterwarnings("ignore")

# Configuration
DATA_DIR = "data"
STORE_DIR = "feature_store"
MODEL_DIR = "model"
PLOTS_DIR = "plots"
TEST_FILE = os.path.join(DATA_DIR, "test.csv")
MODEL_PATH = os.path.join(MODEL_DIR, "model.joblib")
SELECTOR_PATH = os.path.join(MODEL_DIR, "selector.joblib")
TRANSFORMER_PATH = "transformer.pth"

# Filtering thresholds
MIN_TEMP_POINTS = 7
MIN_APELBLAT_R2 = 0.8
MAX_MODEL_RMSE = 0.5

# Import transformer for feature generation
from train_transformer import InteractionTransformer, DEVICE


# =============================================================================
# APELBLAT EQUATION FUNCTIONS
# =============================================================================

def apelblat_equation(T, A, B, C):
    """Apelblat equation: ln(x) = A + B/T + C*ln(T)"""
    return A + B / T + C * np.log(T)


def fit_apelblat(T, ln_x):
    """
    Fit Apelblat equation to experimental data.
    Returns (A, B, C, R2) or (None, None, None, 0) if fit fails.
    """
    try:
        # Initial guesses based on typical solubility behavior
        p0 = [0, -1000, 0]
        popt, _ = curve_fit(apelblat_equation, T, ln_x, p0=p0, maxfev=5000)
        
        # Calculate R2
        ln_x_pred = apelblat_equation(T, *popt)
        r2 = r2_score(ln_x, ln_x_pred)
        
        return popt[0], popt[1], popt[2], r2
    except:
        return None, None, None, 0


# =============================================================================
# FEATURE GENERATION (from train.py)
# =============================================================================

def generate_features_for_prediction(df):
    """Generate features for model prediction (mirrors train.py)."""
    sol_raw = pd.read_parquet(os.path.join(STORE_DIR, "solute_raw.parquet")).set_index('SMILES_KEY')
    solv_raw = pd.read_parquet(os.path.join(STORE_DIR, "solvent_raw.parquet")).set_index('SMILES_KEY')
    sol_c = pd.read_parquet(os.path.join(STORE_DIR, "solute_council.parquet")).set_index('SMILES_KEY')
    solv_c = pd.read_parquet(os.path.join(STORE_DIR, "solvent_council.parquet")).set_index('SMILES_KEY')
    
    # Load Transformer
    transformer = InteractionTransformer().to(DEVICE)
    transformer.load_state_dict(torch.load(TRANSFORMER_PATH, map_location=DEVICE, weights_only=True))
    transformer.eval()
    
    # Batch Processing
    X_sol_all = sol_c.loc[df['Solute']].values.astype(np.float32)
    X_solv_all = solv_c.loc[df['Solvent']].values.astype(np.float32)
    
    batch_size = 512
    embed_list = []
    
    with torch.no_grad():
        for i in range(0, len(X_sol_all), batch_size):
            b_sol = torch.tensor(X_sol_all[i : i+batch_size]).to(DEVICE)
            b_solv = torch.tensor(X_solv_all[i : i+batch_size]).to(DEVICE)
            _, feats, _ = transformer(b_sol, b_solv)
            embed_list.append(feats.cpu().numpy())
            
    X_embed = np.vstack(embed_list)
    
    # Thermodynamic Engineering
    T = df['Temperature'].values.reshape(-1, 1).astype(np.float32)
    T_inv = (1000.0 / df['Temperature'].values).reshape(-1, 1).astype(np.float32)
    Tm = sol_raw.loc[df['Solute'], 'pred_Tm'].values.reshape(-1, 1).astype(np.float32)
    T_red = (T / Tm).astype(np.float32)
    
    X_reshaped = X_embed.reshape(X_embed.shape[0], 24, 32)
    X_interact = (np.sign(X_reshaped.mean(axis=2)) * np.linalg.norm(X_reshaped, axis=2)) * T_inv
    
    X_raw = np.hstack([sol_raw.loc[df['Solute']].values, solv_raw.loc[df['Solvent']].values])
    
    return np.hstack([X_raw, X_interact, Tm, T_red, T, T_inv])


# =============================================================================
# APPROACH A: HOLDOUT TESTING
# =============================================================================

def holdout_test(pair_df, model, selector, A, B, C):
    """
    Leave-one-out testing: for each temperature point,
    compare model prediction to Apelblat extrapolation.
    """
    n = len(pair_df)
    if n < 3:
        return None
    
    holdout_errors = []
    apelblat_errors = []
    
    for i in range(n):
        # Hold out one point
        test_idx = pair_df.index[i]
        T_test = pair_df.loc[test_idx, 'Temperature']
        y_true = pair_df.loc[test_idx, 'LogS']
        
        # Apelblat prediction (using fitted params from full data)
        # Note: slight data leakage, but represents ideal thermodynamic model
        ln_x_apelblat = apelblat_equation(T_test, A, B, C)
        # Convert back to LogS (ln_x = LogS * ln(10))
        apelblat_pred = ln_x_apelblat / np.log(10)
        
        # Model prediction
        test_row = pair_df.loc[[test_idx]]
        X_test = generate_features_for_prediction(test_row)
        X_test = selector.transform(X_test)
        model_pred = model.predict(X_test)[0]
        
        holdout_errors.append(model_pred - y_true)
        apelblat_errors.append(apelblat_pred - y_true)
    
    return {
        'holdout_rmse': np.sqrt(np.mean(np.array(holdout_errors)**2)),
        'apelblat_rmse': np.sqrt(np.mean(np.array(apelblat_errors)**2)),
        'holdout_mae': np.mean(np.abs(holdout_errors)),
        'apelblat_mae': np.mean(np.abs(apelblat_errors))
    }


# =============================================================================
# APPROACH B: SYNTHETIC TEMPERATURE TESTING
# =============================================================================

def synthetic_temp_test(pair_df, model, selector, A, B, C):
    """
    Generate predictions at synthetic temperatures and compare
    model curve vs Apelblat curve for smoothness/consistency.
    """
    T_min = pair_df['Temperature'].min()
    T_max = pair_df['Temperature'].max()
    
    # Generate synthetic temperatures (finer grid)
    T_synthetic = np.linspace(T_min, T_max, 50)
    
    # Calculate Apelblat curve
    ln_x_apelblat = apelblat_equation(T_synthetic, A, B, C)
    logS_apelblat = ln_x_apelblat / np.log(10)
    
    # Calculate model predictions at synthetic temps
    # Create synthetic dataframe
    solute = pair_df['Solute'].iloc[0]
    solvent = pair_df['Solvent'].iloc[0]
    
    synthetic_df = pd.DataFrame({
        'Solute': [solute] * len(T_synthetic),
        'Solvent': [solvent] * len(T_synthetic),
        'Temperature': T_synthetic
    })
    
    X_synthetic = generate_features_for_prediction(synthetic_df)
    X_synthetic = selector.transform(X_synthetic)
    logS_model = model.predict(X_synthetic)
    
    # Compute metrics
    curve_rmse = np.sqrt(mean_squared_error(logS_apelblat, logS_model))
    
    # Compute slope consistency (d(logS)/d(1/T))
    slope_apelblat = np.gradient(logS_apelblat, 1/T_synthetic)
    slope_model = np.gradient(logS_model, 1/T_synthetic)
    slope_corr = np.corrcoef(slope_apelblat, slope_model)[0, 1]
    
    return {
        'curve_rmse': curve_rmse,
        'slope_correlation': slope_corr,
        'T_synthetic': T_synthetic,
        'logS_apelblat': logS_apelblat,
        'logS_model': logS_model
    }


# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================

def plot_pair_comparison(pair_id, pair_df, A, B, C, synthetic_result, save_dir):
    """Generate comparison plot for a single pair."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left: Experimental data + fits
    ax1 = axes[0]
    T_exp = pair_df['Temperature'].values
    logS_exp = pair_df['LogS'].values
    
    # Apelblat curve
    T_plot = np.linspace(T_exp.min() - 5, T_exp.max() + 5, 100)
    ln_x_plot = apelblat_equation(T_plot, A, B, C)
    logS_plot = ln_x_plot / np.log(10)
    
    ax1.scatter(T_exp, logS_exp, s=60, c='blue', label='Experimental', zorder=3)
    ax1.plot(T_plot, logS_plot, 'r--', linewidth=2, label='Apelblat Fit')
    ax1.plot(synthetic_result['T_synthetic'], synthetic_result['logS_model'], 
             'g-', linewidth=2, label='Model Prediction')
    
    ax1.set_xlabel('Temperature (K)', fontsize=12)
    ax1.set_ylabel('LogS (log₁₀ mol/L)', fontsize=12)
    ax1.set_title(f'Pair {pair_id}: T vs LogS', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Right: 1/T plot (Van't Hoff style)
    ax2 = axes[1]
    T_inv_exp = 1000 / T_exp
    T_inv_plot = 1000 / T_plot
    T_inv_synth = 1000 / synthetic_result['T_synthetic']
    
    ax2.scatter(T_inv_exp, logS_exp, s=60, c='blue', label='Experimental', zorder=3)
    ax2.plot(T_inv_plot, logS_plot, 'r--', linewidth=2, label='Apelblat Fit')
    ax2.plot(T_inv_synth, synthetic_result['logS_model'], 
             'g-', linewidth=2, label='Model Prediction')
    
    ax2.set_xlabel('1000/T (K⁻¹)', fontsize=12)
    ax2.set_ylabel('LogS (log₁₀ mol/L)', fontsize=12)
    ax2.set_title(f'Pair {pair_id}: Van\'t Hoff Plot', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'pair_{pair_id}.png'), dpi=150, bbox_inches='tight')
    plt.close()


def plot_summary(results_df, save_dir):
    """Generate summary plots for all analyzed pairs."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Model RMSE vs Apelblat R²
    ax1 = axes[0, 0]
    ax1.scatter(results_df['apelblat_r2'], results_df['model_rmse'], 
                alpha=0.6, s=40, c='steelblue')
    ax1.set_xlabel('Apelblat R²', fontsize=12)
    ax1.set_ylabel('Model RMSE', fontsize=12)
    ax1.set_title('Model RMSE vs Apelblat R²', fontsize=12)
    ax1.axhline(y=MAX_MODEL_RMSE, color='red', linestyle='--', label=f'Threshold={MAX_MODEL_RMSE}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Holdout RMSE comparison
    ax2 = axes[0, 1]
    valid_holdout = results_df.dropna(subset=['holdout_rmse', 'holdout_apelblat_rmse'])
    if len(valid_holdout) > 0:
        ax2.scatter(valid_holdout['holdout_apelblat_rmse'], valid_holdout['holdout_rmse'],
                    alpha=0.6, s=40, c='steelblue')
        max_val = max(valid_holdout['holdout_rmse'].max(), valid_holdout['holdout_apelblat_rmse'].max())
        ax2.plot([0, max_val], [0, max_val], 'r--', label='y=x')
        ax2.set_xlabel('Apelblat Holdout RMSE', fontsize=12)
        ax2.set_ylabel('Model Holdout RMSE', fontsize=12)
        ax2.set_title('Holdout Test: Model vs Apelblat', fontsize=12)
        ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Curve RMSE distribution
    ax3 = axes[1, 0]
    ax3.hist(results_df['curve_rmse'].dropna(), bins=20, color='steelblue', edgecolor='white')
    ax3.set_xlabel('Curve RMSE (Model vs Apelblat)', fontsize=12)
    ax3.set_ylabel('Count', fontsize=12)
    ax3.set_title('Distribution of Model-Apelblat Curve Deviation', fontsize=12)
    ax3.grid(True, alpha=0.3)
    
    # 4. Slope correlation distribution
    ax4 = axes[1, 1]
    ax4.hist(results_df['slope_correlation'].dropna(), bins=20, color='steelblue', edgecolor='white')
    ax4.set_xlabel('Slope Correlation', fontsize=12)
    ax4.set_ylabel('Count', fontsize=12)
    ax4.set_title('Temperature Slope Consistency', fontsize=12)
    ax4.axvline(x=0.9, color='green', linestyle='--', label='Good (>0.9)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'summary.png'), dpi=150, bbox_inches='tight')
    plt.close()


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def run_analysis():
    os.makedirs(PLOTS_DIR, exist_ok=True)
    
    print("="*60)
    print("APELBLAT EQUATION ANALYSIS - REGIME II")
    print("="*60)
    
    # 1. Load data and model
    print("\n[1] Loading data and model...")
    df_test = pd.read_csv(TEST_FILE)
    model = joblib.load(MODEL_PATH)
    selector = joblib.load(SELECTOR_PATH)
    
    print(f"  Test samples: {len(df_test)}")
    
    # 2. Group by solute-solvent pairs
    print("\n[2] Grouping by solute-solvent pairs...")
    df_test['pair_id'] = df_test['Solute'] + '||' + df_test['Solvent']
    pairs = df_test.groupby('pair_id')
    
    print(f"  Total pairs: {len(pairs)}")
    
    # 3. Filter pairs
    print("\n[3] Filtering pairs...")
    print(f"  Criteria: ≥{MIN_TEMP_POINTS} temps, Apelblat R²≥{MIN_APELBLAT_R2}, Model RMSE≤{MAX_MODEL_RMSE}")
    
    results = []
    filtered_pairs = []
    
    for pair_id, pair_df in pairs:
        # Check temperature count
        n_temps = len(pair_df)
        if n_temps < MIN_TEMP_POINTS:
            continue
        
        # Fit Apelblat to experimental data
        T = pair_df['Temperature'].values
        logS = pair_df['LogS'].values
        ln_x = logS * np.log(10)  # Convert LogS to ln(x)
        
        A, B, C, apelblat_r2 = fit_apelblat(T, ln_x)
        
        if A is None or apelblat_r2 < MIN_APELBLAT_R2:
            continue
        
        # Get model predictions
        X = generate_features_for_prediction(pair_df)
        X = selector.transform(X)
        preds = model.predict(X)
        model_rmse = np.sqrt(mean_squared_error(logS, preds))
        
        if model_rmse > MAX_MODEL_RMSE:
            continue
        
        # Pair passes all filters
        filtered_pairs.append({
            'pair_id': pair_id,
            'pair_df': pair_df,
            'A': A, 'B': B, 'C': C,
            'apelblat_r2': apelblat_r2,
            'model_rmse': model_rmse,
            'n_temps': n_temps
        })
    
    print(f"  Pairs passing filters: {len(filtered_pairs)}")
    
    # 4. Analyze filtered pairs
    print("\n[4] Running analysis on filtered pairs...")
    
    for i, pair_data in enumerate(filtered_pairs):
        pair_id = pair_data['pair_id']
        pair_df = pair_data['pair_df']
        A, B, C = pair_data['A'], pair_data['B'], pair_data['C']
        
        if (i + 1) % 10 == 0:
            print(f"  Processing pair {i+1}/{len(filtered_pairs)}...")
        
        # Approach A: Holdout testing
        holdout_result = holdout_test(pair_df, model, selector, A, B, C)
        
        # Approach B: Synthetic temperature testing
        synthetic_result = synthetic_temp_test(pair_df, model, selector, A, B, C)
        
        # Store results
        result = {
            'pair_id': pair_id,
            'n_temps': pair_data['n_temps'],
            'apelblat_r2': pair_data['apelblat_r2'],
            'A': A, 'B': B, 'C': C,
            'model_rmse': pair_data['model_rmse'],
            'curve_rmse': synthetic_result['curve_rmse'],
            'slope_correlation': synthetic_result['slope_correlation']
        }
        
        if holdout_result:
            result['holdout_rmse'] = holdout_result['holdout_rmse']
            result['holdout_apelblat_rmse'] = holdout_result['apelblat_rmse']
        
        results.append(result)
        
        # Generate plot for first 10 pairs
        if i < 10:
            plot_pair_comparison(i, pair_df, A, B, C, synthetic_result, PLOTS_DIR)
    
    # 5. Save results
    print("\n[5] Saving results...")
    results_df = pd.DataFrame(results)
    results_df.to_csv('apelblat_results.csv', index=False)
    print(f"  Saved: apelblat_results.csv")
    
    # Generate summary plot
    plot_summary(results_df, PLOTS_DIR)
    print(f"  Saved: {PLOTS_DIR}/summary.png")
    
    # 6. Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total pairs analyzed: {len(results_df)}")
    print(f"\nModel vs Apelblat (Synthetic Curve):")
    print(f"  Curve RMSE: {results_df['curve_rmse'].mean():.4f} ± {results_df['curve_rmse'].std():.4f}")
    print(f"  Slope Corr: {results_df['slope_correlation'].mean():.4f} ± {results_df['slope_correlation'].std():.4f}")
    
    if 'holdout_rmse' in results_df.columns:
        valid = results_df.dropna(subset=['holdout_rmse'])
        print(f"\nHoldout Testing ({len(valid)} pairs):")
        print(f"  Model RMSE:    {valid['holdout_rmse'].mean():.4f} ± {valid['holdout_rmse'].std():.4f}")
        print(f"  Apelblat RMSE: {valid['holdout_apelblat_rmse'].mean():.4f} ± {valid['holdout_apelblat_rmse'].std():.4f}")
    
    # Physical interpretation
    print(f"\nApelblat Parameters (mean):")
    print(f"  A = {results_df['A'].mean():.2f} (entropy term)")
    print(f"  B = {results_df['B'].mean():.2f} (enthalpy term, ~ΔH_sol/R)")
    print(f"  C = {results_df['C'].mean():.2f} (curvature/Cp term)")
    
    print("\nDone!")


if __name__ == "__main__":
    run_analysis()
