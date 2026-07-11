import numpy as np
np.random.seed(99)
from utils_data_gen import *
from utils_estimate import *
from scipy import stats

d = 6
d_meas = d - 1
predictors = [f"Xp{i+1}" for i in range(2**d_meas)]
p_positivity = 0.1
p_bound = np.random.uniform(0.2, 0.5)
pl_range = (p_positivity, p_bound)
ph_range = (1 - p_bound, 1 - p_positivity)

bias_values = {"Y0": False, "Y1": True, "A": True, "S": False}
obs_probs = sample_all_probs(d, pl_range, ph_range, bias_values)
rct_probs = {"Y0": obs_probs["Y0"], "Y1": obs_probs["Y1"], "A": (2**d)*[0.5], "S": (2**d)*[0.95]}
pX = {"R=0": [0.4, 0.6], "R=1": [0.6, 0.4]}

df_rct = init_df(50000, d, d_meas, 1, rct_probs, pX, False, [], "disc")
df_obs = init_df(50000, d, d_meas, 0, obs_probs, pX, False, [], "disc")

rct_models = fit_models(df_rct, predictors)
obs_models = fit_models(df_obs, predictors)

df_rct_val = init_df(2000 * 20, d, d_meas, 1, rct_probs, pX, False, [], "disc")
df_obs_val = init_df(2000 * 20, d, d_meas, 0, obs_probs, pX, False, [], "disc")

make_preds(df_rct_val, predictors, rct_models)
make_preds(df_obs_val, predictors, obs_models)

df_val = merge_df_val(df_rct_val, df_obs_val, predictors, rct_models, obs_models)

b1 = df_val["b1(X)"]
abs_b1 = df_val["abs(b1(X))"]
print(f"p_bound={p_bound:.4f}")
print(f"b1(X): mean={b1.mean():.6f}, std={b1.std():.6f}")
print(f"abs(b1): mean={abs_b1.mean():.6f}")
print()

for key in ["SE_S", "SE_A", "SE_Y0", "SE_Y1"]:
    if key == "SE_S":
        flt = "R==0"
    elif key == "SE_A":
        flt = "R==0 & S==1"
    elif key == "SE_Y0":
        flt = "R==0 & S==1 & A==0"
    else:
        flt = "R==0 & S==1 & A==1"
    
    df_sub = df_val.query(flt)[["abs(b1(X))", key]].dropna()
    n_val = 2000
    df_sub = df_sub.iloc[:n_val]
    r, p = stats.pearsonr(df_sub["abs(b1(X))"], df_sub[key])
    print(f"{key}: r={r:.4f}, p={p:.6f}, n={len(df_sub)}")
