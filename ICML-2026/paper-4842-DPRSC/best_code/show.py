import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.patheffects as pe
from matplotlib.patches import Patch

def epsilon_show(result_file_name, picture_file_name, pattern_graph):
    if (pattern_graph == '2star'):
        pattern_graph = '2-star'
    df = pd.read_csv(result_file_name)
    epsilon_test = df['epsilon'].values
    
    pure_DP_error = df['pure_DP_error'].values
    pure_DP_std = df['pure_DP_std'].values
    
    approx_DP_error = df['approx_DP_error'].values
    approx_DP_std = df['approx_DP_std'].values
    
    base_comp_error = df['base_comp_error'].values
    base_comp_std = df['base_comp_std'].values
    
    base_comp_ADP_error = df['base_comp_ADP_error'].values
    base_comp_ADP_std = df['base_comp_ADP_std'].values
    
    y_min = min(np.min(pure_DP_error - pure_DP_std), np.min(approx_DP_error - approx_DP_std), 
                np.min(base_comp_error), np.min(base_comp_ADP_error))
    y_max = max(np.max(pure_DP_error + pure_DP_std), np.max(approx_DP_error + approx_DP_std), 
                np.max(base_comp_error + base_comp_std), np.max(base_comp_ADP_error + base_comp_ADP_std))

    if (y_min <= 0):
        y_min = min(np.min(base_comp_error), np.min(base_comp_ADP_error), np.min(pure_DP_error), np.min(approx_DP_error)) / 10  
    y_min_log = np.floor(np.log10(y_min))
    y_max_log = np.ceil(np.log10(y_max))
    
    yticks = [10**i for i in range(int(y_min_log), int(y_max_log) + 1)]

    fig, ax = plt.subplots()
    
    ax.plot(epsilon_test, base_comp_error, label='PDP_Comp', marker='x', linestyle='-', color='black')
    ax.fill_between(epsilon_test, base_comp_error-base_comp_std, base_comp_error+base_comp_std, color='black', alpha=0.05)
    
    ax.plot(epsilon_test, base_comp_ADP_error, label='ADP_Comp', marker='^', linestyle='-', color='g')
    ax.fill_between(epsilon_test, base_comp_ADP_error-base_comp_ADP_std, base_comp_ADP_error+base_comp_ADP_std, color='g', alpha=0.1)
    
    line, = ax.plot(epsilon_test, pure_DP_error, label='PDP_RSC', marker='o', linestyle='-', color='b')
    line.set_path_effects([pe.Stroke(linewidth=1, foreground='white'), pe.Normal()])
    ax.fill_between(epsilon_test, pure_DP_error-pure_DP_std, pure_DP_error+pure_DP_std, color='b', alpha=0.3)
    
    line, = ax.plot(epsilon_test, approx_DP_error, label='ADP_RSC', marker='s', linestyle='-', color='r')
    line.set_path_effects([pe.Stroke(linewidth=1, foreground='white'), pe.Normal()])
    ax.fill_between(epsilon_test, approx_DP_error-approx_DP_std, approx_DP_error+approx_DP_std, color='r', alpha=0.3)
    
    ax.set_yscale('log')  
    ax.set_ylim(10**y_min_log, 10**y_max_log)
    ax.set_yticks(yticks)
    ax.tick_params(axis='both', labelsize=13)

    ax.legend(loc='upper right', fontsize=16, framealpha=0.8)
    ax.grid(True, which='major', linestyle='--', linewidth=0.5)

    ax.set_title(pattern_graph, fontsize=30)
    ax.set_xlabel(r'$\epsilon$', fontsize=30) 
    ax.set_ylabel('Relative Error', fontsize=30)
    plt.tight_layout()
    plt.savefig(picture_file_name)
    plt.close()

def Q_show(result_file_name, picture_file_name, pattern_graph, lim_Q):
    if (pattern_graph == '2star'):
        pattern_graph = '2-star'
    df = pd.read_csv(result_file_name)
    Q_num_test = df['Q_num'].values / (10**lim_Q)
    
    pure_DP_error = df['pure_DP_error'].values
    pure_DP_std = df['pure_DP_std'].values
    
    approx_DP_error = df['approx_DP_error'].values
    approx_DP_std = df['approx_DP_std'].values
    
    base_comp_error = df['base_comp_error'].values
    base_comp_std = df['base_comp_std'].values
    
    base_comp_ADP_error = df['base_comp_ADP_error'].values
    base_comp_ADP_std = df['base_comp_ADP_std'].values

    y_min = min(np.min(pure_DP_error - pure_DP_std), np.min(approx_DP_error - approx_DP_std))
    y_max = max(np.max(pure_DP_error + pure_DP_std), np.max(approx_DP_error + approx_DP_std), 
                np.max(base_comp_error + base_comp_std), np.max(base_comp_ADP_error + base_comp_ADP_std))

    if (y_min <= 0):
        y_min = min(np.min(pure_DP_error), np.min(approx_DP_error)) / 10  
    y_min_log = np.floor(np.log10(y_min))
    y_max_log = np.ceil(np.log10(y_max))
    yticks = [10**i for i in range(int(y_min_log), int(y_max_log) + 1)]

    fig, ax = plt.subplots()
    
    ax.plot(Q_num_test, base_comp_error, label='PDP_Comp', marker='x', linestyle='-', color='black')
    ax.fill_between(Q_num_test, base_comp_error-base_comp_std, base_comp_error+base_comp_std, color='black', alpha=0.05)
    
    ax.plot(Q_num_test, base_comp_ADP_error, label='ADP_Comp', marker='^', linestyle='-', color='g')
    ax.fill_between(Q_num_test, base_comp_ADP_error-base_comp_ADP_std, base_comp_ADP_error+base_comp_ADP_std, color='g', alpha=0.1)
    
    line, = ax.plot(Q_num_test, pure_DP_error, label='PDP_RSC', marker='o', linestyle='-', color='b')
    line.set_path_effects([pe.Stroke(linewidth=1, foreground='white'), pe.Normal()])
    ax.fill_between(Q_num_test, pure_DP_error-pure_DP_std, pure_DP_error+pure_DP_std, color='b', alpha=0.3)
    
    line, = ax.plot(Q_num_test, approx_DP_error, label='ADP_RSC', marker='s', linestyle='-', color='r')
    line.set_path_effects([pe.Stroke(linewidth=1, foreground='white'), pe.Normal()])
    ax.fill_between(Q_num_test, approx_DP_error-approx_DP_std, approx_DP_error+approx_DP_std, color='r', alpha=0.3)
    
    ax.set_yscale('log')  
    ax.set_ylim(10**y_min_log, 10**y_max_log)
    ax.set_yticks(yticks)
    ax.tick_params(axis='both', labelsize=13)

    ax.legend(loc='upper left', fontsize=16, framealpha=0.8)
    ax.grid(True, which='major', linestyle='--', linewidth=0.5)

    ax.set_title(pattern_graph, fontsize=30)
    ax.set_xlabel(fr'$|Q|(\times 10^{lim_Q})$', fontsize=30)
    ax.set_ylabel('Relative Error', fontsize=30)
    plt.tight_layout()
    plt.savefig(picture_file_name)  
    plt.close()

def qtime_and_prtime(prtime_file_name, qtime_file_name, picture_file_name, pattern_graph):
    if pattern_graph == '2star':
        pattern_graph = '2-star'
    df_q = pd.read_csv(qtime_file_name)
    df_pr = pd.read_csv(prtime_file_name)

    labels = ['PDP_RSC', 'ADP_RSC', 'PDP_Comp', 'ADP_Comp']
    colors = ['blue', 'red', 'black', 'green']

    q_values = df_q.iloc[0].tolist()      
    pr_values = df_pr.iloc[0].tolist()    

    q_values_s = [v * 1e-6 for v in q_values]
    pr_values_s = [v * 60 for v in pr_values]
    
    x = np.arange(len(labels))
    width = 0.45

    all_values = q_values_s + pr_values_s
    non_zero_values = [v for v in all_values if v > 0]
    ymin = min(non_zero_values)
    ymax = max(all_values)
    plt.ylim(ymin / 5.0, ymax * 5.0)
    plt.bar(x - width/2, q_values_s, width, color=colors, edgecolor=colors, label='Query Time', linewidth=1.5)
    plt.bar(x + width/2, pr_values_s, width, color='none', edgecolor=colors, hatch='//', label='Preprocessing Time', linewidth=1.5)
    for i, v in enumerate(q_values_s):
        if v > 0:
            if v <= 0.001:
                plt.text(x[i] - width/2, v, f'{q_values[i]:.0f}μs', ha='center', va='bottom', fontsize=12)
            else:
                plt.text(x[i] - width/2, v, f'{q_values[i]/1000:.0f}ms', ha='center', va='bottom', fontsize=12)
    for i, v in enumerate(pr_values_s):
        if v > 0:
            if pr_values[i] >= 1:
                plt.text(x[i] + width/2, v, f'{pr_values[i]:.1f}min', ha='center', va='bottom', fontsize=12)
            elif pr_values[i] >= 1/60:
                plt.text(x[i] + width/2, v, f'{pr_values[i]*60:.0f}s', ha='center', va='bottom', fontsize=12)
            else:
                plt.text(x[i] + width/2, v, f'{pr_values[i]*6e4:.0f}ms', ha='center', va='bottom', fontsize=12)
                
    plt.yscale('log')

    plt.xticks(x, labels, fontsize=15)
    plt.xlabel('Algorithms', fontsize=30)
    plt.ylabel('Time (s)', fontsize=30)
    plt.title(pattern_graph, fontsize=30)

    legend_elements = [
        Patch(facecolor='grey', edgecolor='grey', label='Query Time', linewidth=1.5),
        Patch(facecolor='none', edgecolor='grey', hatch='//', label='Preprocessing Time', linewidth=1.5)
    ]
    plt.legend(handles=legend_elements, loc='lower right', fontsize=16, framealpha=0.8)
    plt.tight_layout()
    plt.savefig(picture_file_name)
    plt.close()
    
def total_time(prtime_file_name, qtime_file_name, picture_file_name, pattern_graph, n):
    if (pattern_graph == '2star'):
        pattern_graph = '2-star'
    df_prtime = pd.read_csv(prtime_file_name)  
    df_qtime = pd.read_csv(qtime_file_name)  
    prtime_values = df_prtime.iloc[0].tolist() 
    qtime_values = df_qtime.iloc[0].tolist()
    
    total_Qnum = (n * (n - 1) // 2) + n
    Qnum_test = [int(round(total_Qnum**(i/8.0))) for i in range(9)]
    time = []
    for i in range(4):
        time.append([])
        for j in range(9):
            time[i].append(prtime_values[i] * 60.0 + qtime_values[i] * Qnum_test[j] / 1e6)
    
    plt.plot(Qnum_test, time[3], label='ADP_Comp', marker='^', linestyle='-', color='g')
    plt.plot(Qnum_test, time[2], label='PDP_Comp', marker='x', linestyle='--', color='black')
    plt.plot(Qnum_test, time[1], label='ADP_RSC', marker='s', linestyle='-', color='r')
    plt.plot(Qnum_test, time[0], label='PDP_RSC', marker='o', linestyle='--', color='b')
    
    plt.title(pattern_graph, fontsize=30)
    plt.xscale('log')
    plt.yscale('log')
    plt.legend(loc='upper left', fontsize=16, framealpha=0.8)
    plt.grid(True, which='major', linestyle='--', linewidth=0.5)

    plt.title(pattern_graph, fontsize=30)
    plt.xlabel(fr'$|Q|$', fontsize=30)
    plt.ylabel('Total Time (s)', fontsize=30)
    plt.tight_layout()
    plt.savefig(picture_file_name)  
    plt.close()

if __name__ == "__main__":
    d = 1
    n_list = [379, 5201, 16347]
    dataset_names = ["ca-netscience", "musae-squirrel", "bio-WormNet-v3"]
    for i in range(1, 3):
        dataset_name = dataset_names[i]
        patterns = ['triangle', 'edge', '2star']
        for j in range(0, 3):
            pattern = patterns[j]
            
            prtime_file_name = f'{dataset_name}/{dataset_name}_{pattern}_prtime_result.csv'
            qtime_file_name = f'{dataset_name}/{dataset_name}_{pattern}_qtime_result.csv'    
            qprtime_picture_file_name = f'{dataset_name}/{dataset_name}_{pattern}_qprtime.png'
            qtime_and_prtime(prtime_file_name, qtime_file_name, qprtime_picture_file_name, pattern.capitalize())
            
            ttime_picture_file_name = f'{dataset_name}/{dataset_name}_{pattern}_ttime.png'
            total_time(prtime_file_name, qtime_file_name, ttime_picture_file_name, pattern.capitalize(), n_list[i])