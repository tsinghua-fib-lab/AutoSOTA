import numpy as np
from numpy.polynomial.legendre import Legendre
from scipy import stats

def pearsonr(df, col_x, col_y, n):
    if col_y == "SE_S":
        df_new = df.query("R==0")[[col_x, col_y]].dropna()
    if col_y == "SE_A":
        df_new = df.query("R==0 & S==1")[[col_x, col_y]].dropna()
    if col_y == "SE_Y0":
        df_new = df.query("R==0 & S==1 & A==0")[[col_x, col_y]].dropna()
    if col_y == "SE_Y1":
        df_new = df.query("R==0 & S==1 & A==1")[[col_x, col_y]].dropna()
    if col_y == "SE_R": 
        df_new = df[[col_x, col_y]].dropna()

    df_new = df_new.iloc[:n, :]

    r, p = stats.pearsonr(df_new[col_x], df_new[col_y])

    return r, p

def spearmanr(df, col_x, col_y):    
    if col_y != "SE_R":
        df_new = df.query("R==0")[[col_x, col_y]].dropna()
    else:
        df_new = df[[col_x, col_y]].dropna()

    if col_y == "SE_A":
        df_new = df_new.query("S==1")
    if col_y == "SE_Y0":
        df_new = df_new.query("S==1 & A==0")
    if col_y == "SE_Y1":
        df_new = df_new.query("S==1 & A==1")

    res = stats.spearmanr(df_new[col_x], df_new[col_y])

    r = res.statistic
    p = res.pvalue

    return r, p

def mmr_test(df, Kxx, B=100):
    
    n = len(df)
    np.fill_diagonal(Kxx, 0)
    psi = np.array(df['psi'])

    mmr_stat = psi @ Kxx @ psi / (n - 1)

    h0_sample = np.zeros(B)
    for k in range(B):
        wpsi = psi * (np.random.multinomial(n, [1 / n] * n) - 1) 
        wprod = wpsi @ Kxx @ wpsi / n
        h0_sample[k] = wprod

    pval = (np.sum(mmr_stat < h0_sample) + 1) / (len(h0_sample) + 1)
    return int(pval < 0.05), pval
