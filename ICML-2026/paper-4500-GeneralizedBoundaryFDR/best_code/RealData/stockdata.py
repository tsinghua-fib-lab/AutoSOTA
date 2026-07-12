import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats
from utils import *
from collections import defaultdict
# Here is data pre-processing to get the p-values.
df = pd.read_csv('yourstockdata.csv')
df['Date'] = pd.to_datetime(df['Date'])  
results_df_2024 = pd.read_csv('computedinformation.csv')
p_values_oneside = results_df_2024['P-value (One-sided > 0)'].values
p_values_twoside = results_df_2024['P-value (Two-sided)'].values
stock_pool = results_df_2024['Ticker'].tolist()
df_2025 = df[df['Date'].dt.year == 2025].copy().reset_index(drop=True)
df_2025['SPY'] = df_2025.iloc[:, 1:].mean(axis=1)

# The main experiment loop
alphas = [0.1,  0.2]
df = df_2025.copy()
spy_prices = df['SPY']
stock_pool_prices = df[stock_pool]
n_random_stocks = 20  
n_iterations = 100    

random_results = defaultdict(lambda: defaultdict(list))

for alpha in alphas:
    
    np.random.seed(42)    
    valid_iterations = 0  
    
    for iteration in range(n_iterations):

        total_stocks = len(stock_pool)
        if total_stocks < n_random_stocks:
            selected_random_indices = np.arange(total_stocks)
        else:
            selected_random_indices = np.random.choice(total_stocks, n_random_stocks, replace=False)

        p_values_random = p_values_oneside[selected_random_indices]

        rejections_domino, boundary_index, _ = harmonicpbfdr(p_values_random, alpha)
        rejections_bh = BH(p_values_random, alpha)
        rejections_sl, _, _ = sl_procedure(p_values_random, alpha)
        rejections_2domino, _ = bfdr_k_domino(p_values_random, k=2, alpha=alpha)
        rejections_3domino,_ = bfdr_k_domino(p_values_random, k=3, alpha=alpha)

        selected_indices_domino = np.where(rejections_domino)[0] if isinstance(rejections_domino[0], (bool, np.bool_)) else np.array(rejections_domino)
        selected_indices_bh = np.where(rejections_bh)[0] if isinstance(rejections_bh[0], (bool, np.bool_)) else np.array(rejections_bh)
        selected_indices_sl = np.where(rejections_sl)[0] if isinstance(rejections_sl[0], (bool, np.bool_)) else np.array(rejections_sl)
        selected_indices_2domino = np.where(rejections_2domino)[0] if isinstance(rejections_2domino[0], (bool, np.bool_)) else np.array(rejections_2domino)
        selected_indices_3domino = np.where(rejections_3domino)[0] if isinstance(rejections_3domino[0], (bool, np.bool_)) else np.array(rejections_3domino)
  
        if len(selected_indices_domino) == 0 and len(selected_indices_bh) == 0 and len(selected_indices_sl) == 0:

            continue
        
        valid_iterations += 1
        

        original_indices_domino = selected_random_indices[selected_indices_domino]
        original_indices_bh = selected_random_indices[selected_indices_bh]
        original_indices_sl = selected_random_indices[selected_indices_sl]
        original_indices_2domino = selected_random_indices[selected_indices_2domino]
        original_indices_3domino = selected_random_indices[selected_indices_3domino]
        start_idx = 0
        end_idx = -1
        
        # ========== Domino Method ==========
        if len(original_indices_domino) > 0:
            initial_capital_domino = 0
            final_strategy_value_domino = 0
            
            for i in original_indices_domino:
                start_p = stock_pool_prices.iloc[start_idx, i]
                end_p = stock_pool_prices.iloc[end_idx, i]
                shares = 1000
                cost = shares * start_p
                val = shares * end_p
                initial_capital_domino += cost
                final_strategy_value_domino += val
            
            strategy_return_domino = (final_strategy_value_domino - initial_capital_domino) / initial_capital_domino
            random_results[alpha]['domino_returns'].append(strategy_return_domino * 100)
            random_results[alpha]['domino_stocks'].append(len(original_indices_domino))
        else:

            spy_start = spy_prices.iloc[start_idx]
            spy_end = spy_prices.iloc[end_idx]
            spy_return = (spy_end - spy_start) / spy_start
            random_results[alpha]['domino_returns'].append(spy_return * 100)
            random_results[alpha]['domino_stocks'].append(0)
            
        # ========== Domino2 Method ==========
        if len(original_indices_2domino) > 0:
            initial_capital_2domino = 0
            final_strategy_value_2domino = 0
            
            for i in original_indices_2domino:
                start_p = stock_pool_prices.iloc[start_idx, i]
                end_p = stock_pool_prices.iloc[end_idx, i]
                shares = 1000
                cost = shares * start_p
                val = shares * end_p
                initial_capital_2domino += cost
                final_strategy_value_2domino += val
            
            strategy_return_2domino = (final_strategy_value_2domino - initial_capital_2domino) / initial_capital_2domino
            random_results[alpha]['domino2_returns'].append(strategy_return_2domino * 100)
            random_results[alpha]['domino2_stocks'].append(len(original_indices_2domino))
        else:

            spy_start = spy_prices.iloc[start_idx]
            spy_end = spy_prices.iloc[end_idx]
            spy_return = (spy_end - spy_start) / spy_start
            random_results[alpha]['domino2_returns'].append(spy_return * 100)
            random_results[alpha]['domino2_stocks'].append(0)
            
        
        # ========== Domino3 Method ==========
        if len(original_indices_3domino) > 0:
            initial_capital_3domino = 0
            final_strategy_value_3domino = 0
            
            for i in original_indices_3domino:
                start_p = stock_pool_prices.iloc[start_idx, i]
                end_p = stock_pool_prices.iloc[end_idx, i]
                shares = 1000
                cost = shares * start_p
                val = shares * end_p
                initial_capital_3domino += cost
                final_strategy_value_3domino += val
            
            strategy_return_3domino = (final_strategy_value_3domino - initial_capital_3domino) / initial_capital_3domino
            random_results[alpha]['domino3_returns'].append(strategy_return_3domino * 100)
            random_results[alpha]['domino3_stocks'].append(len(original_indices_3domino))
        else:

            spy_start = spy_prices.iloc[start_idx]
            spy_end = spy_prices.iloc[end_idx]
            spy_return = (spy_end - spy_start) / spy_start
            random_results[alpha]['domino3_returns'].append(spy_return * 100)
            random_results[alpha]['domino3_stocks'].append(0)

        # ========== BH Method ==========
        if len(original_indices_bh) > 0:
            initial_capital_bh = 0
            final_strategy_value_bh = 0
            
            for i in original_indices_bh:
                start_p = stock_pool_prices.iloc[start_idx, i]
                end_p = stock_pool_prices.iloc[end_idx, i]
                shares = 1000
                cost = shares * start_p
                val = shares * end_p
                initial_capital_bh += cost
                final_strategy_value_bh += val
            
            strategy_return_bh = (final_strategy_value_bh - initial_capital_bh) / initial_capital_bh
            random_results[alpha]['bh_returns'].append(strategy_return_bh * 100)
            random_results[alpha]['bh_stocks'].append(len(original_indices_bh))
        else:
     
            spy_start = spy_prices.iloc[start_idx]
            spy_end = spy_prices.iloc[end_idx]
            spy_return = (spy_end - spy_start) / spy_start
            random_results[alpha]['bh_returns'].append(spy_return * 100)
            random_results[alpha]['bh_stocks'].append(0)

        # ========== SL Method ==========
        if len(original_indices_sl) > 0:
            initial_capital_sl = 0
            final_strategy_value_sl = 0
            
            for i in original_indices_sl:
                start_p = stock_pool_prices.iloc[start_idx, i]
                end_p = stock_pool_prices.iloc[end_idx, i]
                shares = 1000
                cost = shares * start_p
                val = shares * end_p
                initial_capital_sl += cost
                final_strategy_value_sl += val
            
            strategy_return_sl = (final_strategy_value_sl - initial_capital_sl) / initial_capital_sl
            random_results[alpha]['sl_returns'].append(strategy_return_sl * 100)
            random_results[alpha]['sl_stocks'].append(len(original_indices_sl))
        else:
         
            spy_start = spy_prices.iloc[start_idx]
            spy_end = spy_prices.iloc[end_idx]
            spy_return = (spy_end - spy_start) / spy_start
            random_results[alpha]['sl_returns'].append(spy_return * 100)
            random_results[alpha]['sl_stocks'].append(0)

    
  




random_summary = []
for alpha in alphas:
    domino_returns = random_results[alpha]['domino_returns']
    bh_returns = random_results[alpha]['bh_returns']
    sl_returns = random_results[alpha]['sl_returns']
    domino2_returns = random_results[alpha]['domino2_returns']
    domino_stocks = random_results[alpha]['domino_stocks']
    bh_stocks = random_results[alpha]['bh_stocks']
    sl_stocks = random_results[alpha]['sl_stocks']
    domino2_stocks = random_results[alpha]['domino2_stocks']
    domino3_returns = random_results[alpha]['domino3_returns']
    domino3_stocks = random_results[alpha]['domino3_stocks']
    random_summary.append({
        'Alpha': alpha,
        'Valid Iterations': len(domino_returns),
        'Domino Avg Stocks': np.mean(domino_stocks),
        'Domino Avg Return (%)': np.mean(domino_returns),
        'Domino Std Return (%)': np.std(domino_returns),
        'BH Avg Stocks': np.mean(bh_stocks),
        'BH Avg Return (%)': np.mean(bh_returns),
        'BH Std Return (%)': np.std(bh_returns),
        'SL Avg Stocks': np.mean(sl_stocks),
        'SL Avg Return (%)': np.mean(sl_returns),
        'SL Std Return (%)': np.std(sl_returns),
        'Domino2 Avg Stocks': np.mean(domino2_stocks),
        'Domino2 Avg Return (%)': np.mean(domino2_returns),
        'Domino2 Std Return (%)': np.std(domino2_returns),
        'Domino3 Avg Stocks': np.mean(domino3_stocks),
        'Domino3 Avg Return (%)': np.mean(domino3_returns),
        'Domino3 Std Return (%)': np.std(domino3_returns),
    })

random_summary_df = pd.DataFrame(random_summary)


