"""Export actual score history and held-out per-phone error comparisons."""
from pathlib import Path
import os
ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

if __name__=='__main__':
    d=pd.read_csv(ROOT/'research/submission_ledger.csv').sort_values('date').reset_index(drop=True)
    out=ROOT/'artifacts/diagnostics';out.mkdir(exist_ok=True)
    fig,axes=plt.subplots(1,2,figsize=(12,4.4),layout='constrained')
    for ax,col,winner,late in zip(axes,['privateScore','publicScore'],[1.61938,2.75339],[1.39533,2.59287]):
        ax.plot(np.arange(1,len(d)+1),d[col],'-o',markersize=4,label='Actual Kaggle score')
        ax.axhline(winner,color='#c44e52',ls='--',label='Historical leaderboard leader')
        ax.axhline(late,color='#55a868',ls=':',label='Explicit third-place late report')
        ix=d[col].idxmin();ax.scatter(ix+1,d.loc[ix,col],s=90,marker='*',color='black',zorder=5)
        ax.set(title=col.replace('Score','')+' (lower is better)',xlabel='Submission order',ylabel='Competition error (m)')
        ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('2021 GSDC: independently scored submissions; historical rank remains frozen')
    fig.savefig(out/'score_history.png',dpi=180);fig.savefig(out/'score_history.svg');plt.close(fig)
    root=ROOT/'artifacts/validation'
    selected=pd.read_csv(root/'pseudorange_raw_selected_phones.csv').set_index('phone')
    baseline=pd.read_csv(root/'pseudorange_raw_raw_ppk_phones.csv').set_index('phone').loc[selected.index]
    ix=np.argsort(baseline.score_m.to_numpy());y=np.arange(len(ix))
    fig,ax=plt.subplots(figsize=(11,max(5,len(ix)*.25)),layout='constrained')
    ax.barh(y-.18,baseline.score_m.iloc[ix],height=.35,label='Raw PPK')
    ax.barh(y+.18,selected.score_m.iloc[ix],height=.35,label='Selected raw graph + bias')
    ax.set_yticks(y,selected.index[ix],fontsize=7);ax.set_xlabel('Per-phone mean of P50/P95 (m)')
    ax.set_title('Reserved dates: repeatedly inspected development holdout')
    ax.legend();ax.grid(axis='x',alpha=.2)
    fig.savefig(out/'holdout_phone_errors.png',dpi=180);plt.close(fig)
