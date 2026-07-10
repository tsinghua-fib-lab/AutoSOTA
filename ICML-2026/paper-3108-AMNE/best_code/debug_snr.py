import numpy as np
from sklearn.datasets import make_friedman1
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score

# Generate data with paper params
for seed in [1, 42, 100]:
    X, y = make_friedman1(n_samples=512, n_features=20, noise=2.0, random_state=seed)
    y_norm = StandardScaler().fit_transform(y.reshape(-1,1)).ravel()
    print(f'Seed {seed}: y var={np.var(y_norm):.4f}, X means={X.mean(axis=0)[:5].round(3)}')

    # Fit single MLP
    cv = KFold(n_splits=5, shuffle=True, random_state=seed)
    r2s = []
    for train, test in cv.split(X):
        mlp_c = MLPRegressor(hidden_layer_sizes=(64,32,8), max_iter=500, early_stopping=True, random_state=seed)
        mlp_c.fit(X[train], y_norm[train])
        r2s.append(r2_score(y_norm[test], mlp_c.predict(X[test])))
    print(f'  Single MLP R2: mean={np.mean(r2s):.4f}, median={np.median(r2s):.4f}, folds={[round(r,3) for r in r2s]}')

    # Fit bagging ensemble
    r2s_bag = []
    for train, test in cv.split(X):
        bag_c = BaggingRegressor(
            estimator=MLPRegressor(hidden_layer_sizes=(64,32,8), max_iter=500, early_stopping=True, random_state=seed),
            n_estimators=10, random_state=seed, n_jobs=1)
        bag_c.fit(X[train], y_norm[train])
        r2s_bag.append(r2_score(y_norm[test], bag_c.predict(X[test])))
    print(f'  Bagging R2: mean={np.mean(r2s_bag):.4f}, median={np.median(r2s_bag):.4f}, folds={[round(r,3) for r in r2s_bag]}')

# Compute effective SNR
X0, y0 = make_friedman1(n_samples=100000, n_features=20, noise=0.0, random_state=0)
Xn, yn = make_friedman1(n_samples=100000, n_features=20, noise=2.0, random_state=0)
signal_var = np.var(y0)
noise_var = np.var(yn - y0)
print(f'Effective SNR: {signal_var/noise_var:.2f} (signal var={signal_var:.2f}, noise var={noise_var:.2f})')
print(f'Noise std used: 2.0, Signal std: {np.sqrt(signal_var):.2f}')
