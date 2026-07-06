"""
Generate learning curve PDFs for TD3 and SAC (SiLU-unified, 5 seeds).
Processes one env at a time to avoid OOM.
"""
import os, gc, sys
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

TRAIN_SEEDS = ['178132','410580','922852','787576','660993']

def extract_raw_data(log_dir, tag='rollout/ep_rew_mean', interval=10000):
    try:
        ea = EventAccumulator(log_dir)
        ea.Reload()
        if tag not in ea.Tags()['scalars']:
            return None
        sl = ea.Scalars(tag)
        df = pd.DataFrame([(s.step, s.value) for s in sl], columns=['Step','Value'])
        df['Step'] = (df['Step']//interval)*interval
        del ea
        gc.collect()
        return df
    except:
        return None

def plot_env(root_dir, env, ax):
    method_order = ['BASE','CAPS','GRAD','ASAP','PAVE']
    palette = {'BASE':'#ff7f0e','CAPS':'#2ca02c','GRAD':'#d62728','ASAP':'#1f77b4','PAVE':'#9467bd'}
    env_path = os.path.join(root_dir, env)
    if not os.path.isdir(env_path):
        return

    for m_key in method_order:
        dfs = []
        for d in sorted(os.listdir(env_path)):
            path = os.path.join(env_path, d)
            if not os.path.isdir(path):
                continue
            raw = d.split('_')[0]
            if raw == 'Vanilla':
                mname = 'BASE'
            elif raw in method_order:
                mname = raw
            else:
                continue
            if mname != m_key:
                continue
            if not any(s in d for s in TRAIN_SEEDS):
                continue
            df = extract_raw_data(path)
            if df is not None:
                df['Value'] = df['Value'].rolling(window=5, min_periods=1).mean()
                dfs.append(df)
            if len(dfs) >= 5:
                break

        if len(dfs) < 2:
            continue
        merged = dfs[0][['Step','Value']].rename(columns={'Value':'V0'})
        for j, df2 in enumerate(dfs[1:], 1):
            merged = pd.merge(merged, df2[['Step','Value']].rename(columns={'Value':f'V{j}'}), on='Step', how='inner')
        vcols = [c for c in merged.columns if c.startswith('V')]
        merged['mean'] = merged[vcols].mean(axis=1)
        merged['std'] = merged[vcols].std(axis=1)
        ax.plot(merged['Step'], merged['mean'], color=palette[m_key], linewidth=2.5, label=m_key, alpha=0.9)
        ax.fill_between(merged['Step'], merged['mean']-merged['std'], merged['mean']+merged['std'], color=palette[m_key], alpha=0.15)
        del dfs, merged
        gc.collect()

def main():
    env_names = {'lunar':'Lunar','pendulum':'Pendulum','reacher':'Reacher','ant':'Ant','hopper':'Hopper','walker':'Walker'}
    env_list = list(env_names.keys())
    out_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'viz')
    os.makedirs(out_dir, exist_ok=True)

    for algo, root in [('td3', './Full/td3/tensorboard_logs/'), ('sac', './Full/sac/tensorboard_logs/')]:
        print(f'\n=== {algo.upper()} ===')
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()

        for i, env in enumerate(env_list):
            print(f'  {env}...', end=' ', flush=True)
            plot_env(root, env, axes[i])
            axes[i].set_title(env_names[env], fontsize=25)
            axes[i].set_xlabel('Step', fontsize=22)
            axes[i].set_ylabel('Episode Reward', fontsize=22)
            axes[i].tick_params(labelsize=15)
            axes[i].ticklabel_format(style='sci', axis='x', scilimits=(0,0))
            axes[i].xaxis.get_offset_text().set_fontsize(15)
            axes[i].legend(loc='lower right', fontsize=14, frameon=True, edgecolor='black')
            gc.collect()
            print('done')

        plt.tight_layout()
        save_path = os.path.join(out_dir, f'learning_curves_{algo}_silu.pdf')
        plt.savefig(save_path, format='pdf', bbox_inches='tight', dpi=200)
        plt.close()
        gc.collect()
        print(f'Saved: {save_path}')

    print('\nAll done!')

if __name__ == '__main__':
    main()
