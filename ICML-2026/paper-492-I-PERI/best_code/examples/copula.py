import numpy as np
import pandas as pd
from pyciphod.utils.stat_tests.dependency_measures import Copula, compute_copula_fit
from pyciphod.utils.stat_tests.independence_tests import CopulaTest

n = 1000


z = np.random.normal(size=n)
z2 = np.random.normal(size=n) + np.random.normal(size=n)

df = pd.DataFrame({
    "z": z,
    "z2": z2,
    "x_cont": 0.5*z + 0.7*z2 + 0.3*np.random.normal(size=n),
    "y_bin": (z+ z2+ 0.5*np.random.normal(size=n) > 0).astype(int),
})

copula_fit = compute_copula_fit(
    df,
    cols=df.columns,
    n_iter=1000,
    burn_in=50,
    thin=5,
    random_state=0,
)

C = copula_fit["copula_matrix"]
n_eff = copula_fit["effective_n"]
print(C, n_eff)

m = Copula("x_cont", "y_bin", ["z, z2"], copula_fit=copula_fit)
res = m.get_dependence(df)

print("Test 1 result:", res)
print("Finite:", np.isfinite(res))
print("In [-1,1]:", -1 <= res <= 1)

t = CopulaTest("x_cont", "y_bin", ["z, z2"],  copula_fit=copula_fit)
res = t.get_pvalue(df)
print("Pval:", res)

m = Copula("x_cont", "y_bin", ["z"], copula_fit=copula_fit)
res = m.get_dependence(df)

print("Test 1 result:", res)
print("Finite:", np.isfinite(res))
print("In [-1,1]:", -1 <= res <= 1)

t = CopulaTest("x_cont", "y_bin", ["z"],  copula_fit=copula_fit)
res = t.get_pvalue(df)
print("Pval:", res)