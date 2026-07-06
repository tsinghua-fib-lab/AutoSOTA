from abc import ABC, abstractmethod
from sklearn.linear_model import LinearRegression as lr
import numpy as np
import pandas as pd
from scipy.special import digamma
from scipy import linalg
from scipy.stats import invwishart, norm, truncnorm
from typing import List, Optional, Sequence, Union
from typing import Hashable


class DependenceMeasures(ABC):
    def __init__(self, x, y, cond_list=None, drop_na=False):
        self.x = x
        self.y = y
        if cond_list is None:
            self.cond_list = []
        else:
            self.cond_list = cond_list
        self.drop_na = drop_na

    def _prepare_data(self, df):
        if self.drop_na:
            return df.dropna(subset=[self.x, self.y] + self.cond_list)
        return df

    @abstractmethod
    def get_dependence(self, df):
        pass


class PartialCorrelation(DependenceMeasures):
    """
    Compute the partial correlation between X and Y given Z.
    Returns a correlation coefficient r in [-1,1] or np.nan when undefined.
    """
    def __init__(self, x, y, cond_list=None, drop_na=False):
        super().__init__(x, y, cond_list, drop_na)

    def get_dependence(self, df):
        df = self._prepare_data(df)
        list_nodes = [self.x, self.y] + self.cond_list
        # If any of the requested columns are missing, let pandas raise KeyError
        df = df[list_nodes]

        # Need at least two observations to compute correlation; with conditioning
        # variables we typically need more than (2 + len(cond_list)) samples
        if df.shape[0] < 3:
            return np.nan

        a = df.values.T

        if len(self.cond_list) > 0:
            cond_list_int = [i + 2 for i in range(len(self.cond_list))]
        else:
            cond_list_int = []

        # Compute correlation matrix of variables (rows of `a`)
        correlation_matrix = np.corrcoef(a)

        var = list((0, 1) + tuple(cond_list_int))
        sub_corr_matrix = correlation_matrix[np.ix_(var, var)]

        # If the submatrix contains NaNs (constant columns) or is nearly singular,
        # partial correlation is undefined -> return np.nan
        if np.isnan(sub_corr_matrix).any():
            return np.nan

        # Use condition number to detect singularity / ill-conditioning robustly
        try:
            cond_number = np.linalg.cond(sub_corr_matrix)
        except np.linalg.LinAlgError:
            return np.nan

        if not np.isfinite(cond_number) or cond_number > 1e12:
            return np.nan

        try:
            inv = np.linalg.inv(sub_corr_matrix)
        except np.linalg.LinAlgError:
            return np.nan

        r = -inv[0, 1] / np.sqrt(inv[0, 0] * inv[1, 1])
        # Numerical issues might push slightly out of [-1,1]
        r = max(-1.0, min(1.0, float(r)))
        return r


class LinearRegressionCoefficient(DependenceMeasures):
    """
    Compute the coefficient of X in a linear regression of Y on X and Z.
    Returns a coefficient that can be any real number or np.nan when undefined.
    """
    def __init__(self, x, y, cond_list=None, drop_na=False):
            super().__init__(x, y, cond_list, drop_na)

    def get_dependence(self, df):
        df = self._prepare_data(df)

        # select columns; if some are missing, allow pandas to raise KeyError
        cols = [self.x] + self.cond_list
        if df.shape[0] == 0:
            return np.nan

        X_data = df[cols].values
        Y_data = df[self.y].values

        # Need at least two samples to fit, and at least as many samples as coef count
        if X_data.shape[0] < 2 or X_data.shape[0] <= X_data.shape[1]:
            # Not enough data to fit a stable regression
            return np.nan

        try:
            reg = lr().fit(X_data, Y_data)
        except Exception:
            return np.nan

        # Return coefficient corresponding to the first column (self.x)
        try:
            coef = getattr(reg, "coef_", None)
            if coef is None:
                return np.nan
            coef_arr = np.asarray(coef).ravel()
            if coef_arr.size == 0:
                return np.nan
            res = float(coef_arr[0])
        except Exception:
            # In case coef_ is not as expected
            return np.nan
        return res


class Gsq(DependenceMeasures):
    """
    Estimate the G-squared statistic for testing independence of X and Y given Z.
    Suitable for discrete variables. See gsq() function for details.
    """

    def __init__(self, x: Union[str, Sequence[str]], y: Union[str, Sequence[str]], cond_list: Optional[Union[str, Sequence[str]]] = None, drop_na=False):
        super().__init__(x, y, cond_list, drop_na)

    def _contingency_table(self, df):
        """
        Compute the joint frequency table for X, Y given Z.
        Returns a dictionary mapping conditioning states to 2D contingency arrays.
        """
        if not self.cond_list:
            # No conditioning: simple 2D table
            table = pd.crosstab(df[self.x], df[self.y])
            return {(): table.values}
        else:
            # Group by conditioning variables
            grouped = df.groupby(self.cond_list)
            tables = {}
            for cond_values, sub_df in grouped:
                table = pd.crosstab(sub_df[self.x], sub_df[self.y])
                tables[cond_values] = table.values
            return tables

    def get_dependence(self, df):
        """
        Returns the G² statistic safely without runtime warnings.
        """
        df = self._prepare_data(df)
        tables = self._contingency_table(df)
        g2_stat = 0

        for table in tables.values():
            if table.size == 0:
                continue
            n = np.sum(table)
            row_sums = np.sum(table, axis=1)
            col_sums = np.sum(table, axis=0)
            expected = np.outer(row_sums, col_sums) / n

            # Only consider nonzero expected counts
            mask = expected > 0
            with np.errstate(divide='ignore', invalid='ignore'):
                contrib = 2 * table[mask] * np.log(table[mask] / expected[mask])
                contrib = np.nan_to_num(contrib)  # convert nan/inf to 0
            g2_stat += np.sum(contrib)

        return g2_stat


class KernelPartialCorrelation(DependenceMeasures):
    """
    Estimate partial correlation using a kernel-based approach.
    Suitable for continuous variables. See kpc() function for details.
    """

    def __init__(self, x, y, cond_list=None, drop_na=False):
        super().__init__(x, y, cond_list, drop_na)

    def get_dependence(self, df):
        """
        Practical kernel partial-correlation estimator.

        Approach implemented here (simple, robust):
        - Regress X on Z using kernel ridge regression (RBF kernel) to obtain
          predicted values X_hat and residuals r_x = X - X_hat.
        - Regress Y on Z similarly to obtain r_y.
        - Return the Pearson correlation between r_x and r_y.

        Notes / choices:
        - Bandwidth (sigma) is selected by the median heuristic on pairwise
          squared distances of Z. If Z is empty, kernel regression reduces to
          predicting the mean (so residuals are centered values and the
          estimator reduces to ordinary Pearson correlation).
        - Regularization lambda fixed to 1e-3 for numeric stability.
        - Returns np.nan when the estimate is undefined (too few samples,
          constant residuals, numerical issues).
        """
        df = self._prepare_data(df)

        # Columns
        x_col = self.x
        y_col = self.y
        z_cols = list(self.cond_list) if self.cond_list is not None else []

        # Basic checks
        if x_col not in df.columns or y_col not in df.columns:
            # Let pandas KeyError behavior be visible to caller; but return nan
            return np.nan

        # Extract data
        try:
            X = df[x_col].to_numpy(dtype=float).ravel()
            Y = df[y_col].to_numpy(dtype=float).ravel()
        except Exception:
            return np.nan

        n = X.shape[0]
        if n < 3:
            return np.nan

        # Helper: compute RBF kernel matrix with median heuristic for sigma
        def _rbf_kernel_matrix(Z, sigma=None):
            # Z: (n_samples, n_features)
            Z = np.asarray(Z, dtype=float)
            if Z.size == 0:
                return None
            # pairwise squared distances
            sq_sum = np.sum(Z * Z, axis=1)
            d2 = sq_sum[:, None] + sq_sum[None, :] - 2.0 * (Z @ Z.T)
            # numerical cleanup
            d2 = np.maximum(d2, 0.0)

            if sigma is None:
                # median heuristic (on squared distances -> use sqrt later)
                # use upper-triangle entries (i<j)
                iu = np.triu_indices(n, k=1)
                if iu[0].size == 0:
                    sigma = 1.0
                else:
                    med = np.median(d2[iu])
                    if med <= 0 or not np.isfinite(med):
                        sigma = 1.0
                    else:
                        # convert median squared-distance to sigma using typical RBF
                        sigma = np.sqrt(med)
            # avoid sigma being zero
            if sigma <= 0 or not np.isfinite(sigma):
                sigma = 1.0

            K = np.exp(-d2 / (2.0 * (sigma ** 2)))
            return K

        # Kernel ridge regression returning residuals on training set
        def _krr_residuals(Z_cols, target, lambda_reg=1e-3):
            # If no conditioning variables, residuals are centered target
            if len(Z_cols) == 0:
                t = target.astype(float)
                return t - np.mean(t)

            Z = df[Z_cols].to_numpy(dtype=float)
            if Z.shape[0] != target.shape[0]:
                return np.nan

            K = _rbf_kernel_matrix(Z, sigma=None)
            if K is None:
                return np.nan

            # Regularize and solve for dual coefficients
            try:
                # Add small jitter to the diagonal for numeric stability
                reg = lambda_reg
                A = K + reg * np.eye(n)
                # Solve A alpha = y
                alpha = np.linalg.solve(A, target.astype(float))
                y_hat = K.dot(alpha)
                resid = target.astype(float) - y_hat
                return resid
            except np.linalg.LinAlgError:
                # fallback: try larger regularization
                try:
                    reg = 1e-2
                    A = K + reg * np.eye(n)
                    alpha = np.linalg.solve(A, target.astype(float))
                    y_hat = K.dot(alpha)
                    resid = target.astype(float) - y_hat
                    return resid
                except Exception:
                    return np.nan
            except Exception:
                return np.nan

        rx = _krr_residuals(z_cols, X)
        ry = _krr_residuals(z_cols, Y)

        # If either residual is nan (estimation failed), return nan
        if rx is None or ry is None:
            return np.nan
        if isinstance(rx, float) and np.isnan(rx):
            return np.nan
        if isinstance(ry, float) and np.isnan(ry):
            return np.nan

        rx = np.asarray(rx, dtype=float).ravel()
        ry = np.asarray(ry, dtype=float).ravel()

        if rx.shape[0] != ry.shape[0] or rx.shape[0] != n:
            return np.nan

        # If residuals are (nearly) constant, correlation undefined
        if np.allclose(np.std(rx), 0.0) or np.allclose(np.std(ry), 0.0):
            return np.nan

        # Compute Pearson correlation between residuals
        try:
            r = np.corrcoef(rx, ry)[0, 1]
            if not np.isfinite(r):
                return np.nan
            # Clip to [-1, 1]
            r = float(max(-1.0, min(1.0, r)))
            return r
        except Exception:
            return np.nan


def _copula_spd(A):
    A = np.asarray(A, dtype=float)
    A = (A + A.T) / 2.0
    I = np.eye(A.shape[0])
    for _ in range(6):
        try:
            np.linalg.cholesky(A)
            return A
        except np.linalg.LinAlgError:
            m = np.min(np.linalg.eigvalsh(A))
            A = A + ((1e-8 - m) if m < 1e-8 else 1e-8) * I
    return A


def _copula_corr(V):
    d = np.sqrt(np.clip(np.diag(V), 1e-12, None))
    C = V / np.outer(d, d)
    np.fill_diagonal(C, 1.0)
    return (C + C.T) / 2.0


def _copula_partial_corr(C):
    try:
        P = linalg.inv(C)
    except Exception:
        P = np.linalg.pinv(C)
    d = np.sqrt(np.clip(np.diag(P), 1e-12, None))
    PC = -P / np.outer(d, d)
    np.fill_diagonal(PC, 1.0)
    return (PC + PC.T) / 2.0


def _encode_copula_df(df):
    """
    Encode a dataframe for Hoff-style Gaussian copula estimation.

    Supported:
    - numeric / bool
    - ordered pandas categoricals
    - unordered binary categoricals

    Returns
    -------
    X : pd.DataFrame
        Encoded numeric dataframe
    R : np.ndarray
        Integer rank-level matrix, -1 for missing
    """
    X = pd.DataFrame(index=df.index)
    R = np.full(df.shape, -1, dtype=int)

    for j, col in enumerate(df.columns):
        s = df[col]

        if pd.api.types.is_bool_dtype(s) or pd.api.types.is_numeric_dtype(s):
            v = pd.to_numeric(s, errors="coerce").astype(float)

        elif isinstance(s.dtype, pd.CategoricalDtype) and s.cat.ordered:
            v = s.cat.codes.astype(float)
            v[v < 0] = np.nan
            v = pd.Series(v, index=s.index)

        else:
            u = pd.unique(s.dropna())
            if len(u) == 2:
                order = sorted(u, key=lambda z: str(z))
                v = s.map({order[0]: 0.0, order[1]: 1.0}).astype(float)
            else:
                raise ValueError(f"Unsupported column '{col}' for copula encoding")

        obs = v.dropna().to_numpy()
        if len(obs) < 2 or np.unique(obs).size < 2:
            raise ValueError(f"Degenerate column '{col}'")

        uniq = np.unique(obs)
        mask = v.notna().to_numpy()
        R[mask, j] = np.searchsorted(uniq, v.to_numpy()[mask], side="left")
        X[col] = v

    return X, R


def estimate_effective_n_from_samples(C_samples, raw_n=None, min_var=1e-10):
    """
    Estimate effective sample size from Gibbs samples of correlation matrices.

    Parameters
    ----------
    C_samples : np.ndarray
        Shape (m, p, p), retained Gibbs samples of latent correlation matrices.
    raw_n : int or float, optional
        Optional upper cap for the estimate.
    min_var : float
        Variance floor.

    Returns
    -------
    float
        Effective sample size estimate or np.nan.
    """
    C_samples = np.asarray(C_samples, dtype=float)

    if C_samples.ndim != 3:
        return np.nan

    m, p, p2 = C_samples.shape
    if p != p2 or m < 2 or p < 2:
        return np.nan

    vals = []
    for i in range(p):
        for j in range(i + 1, p):
            x = C_samples[:, i, j]
            mu = float(np.mean(x))
            var = float(np.var(x, ddof=1))

            if not np.isfinite(mu) or not np.isfinite(var) or var <= min_var:
                continue

            nu_ij = ((1.0 - mu * mu) ** 2) / var
            if np.isfinite(nu_ij) and nu_ij > 0:
                vals.append(nu_ij)

    if len(vals) == 0:
        return np.nan

    n_eff = float(np.mean(vals))

    if raw_n is not None:
        try:
            raw_n = float(raw_n)
            if np.isfinite(raw_n):
                n_eff = float(np.clip(n_eff, 4.0, raw_n))
        except Exception:
            pass

    return n_eff


def compute_copula_fit(
    df,
    cols=None,
    n_iter=600,
    burn_in=200,
    thin=2,
    random_state=0,
    return_samples=False,
):
    """
    Compute Hoff-style posterior mean latent correlation matrix and
    an approximate effective sample size.

    Returns
    -------
    dict
        Keys:
        - 'copula_matrix' : pd.DataFrame
        - 'effective_n'   : float
        - 'columns'       : list[str]
        - 'samples'       : np.ndarray, optional
    """
    if cols is None:
        cols = list(df.columns)
    else:
        cols = list(cols)

    if len(cols) < 2:
        raise ValueError("Need at least 2 columns to compute a copula matrix")

    X, R = _encode_copula_df(df[cols].copy())
    n, p = X.shape
    if n < 3:
        raise ValueError("Need at least 3 rows to compute a copula matrix")

    rng = np.random.default_rng(random_state)

    # Initialize latent Z from Gaussian mid-rank scores
    Z = np.zeros((n, p), dtype=float)
    for j in range(p):
        rj = R[:, j]
        obs = rj >= 0

        lev, cnt = np.unique(rj[obs], return_counts=True)
        cum = np.cumsum(cnt)
        mids = (np.concatenate([[0], cum[:-1]]) + 0.5 * cnt) / cnt.sum()
        mids = np.clip(mids, 1e-6, 1 - 1e-6)
        mapping = {int(a): float(norm.ppf(b)) for a, b in zip(lev, mids)}

        for a in lev:
            Z[rj == a, j] = mapping[int(a)]

        if np.any(~obs):
            Z[~obs, j] = rng.standard_normal(np.sum(~obs))

    Z -= Z.mean(axis=0, keepdims=True)
    sd = np.std(Z, axis=0, ddof=1, keepdims=True)
    sd = np.where(sd < 1e-8, 1.0, sd)
    Z /= sd

    V = _copula_corr(_copula_spd(np.cov(Z, rowvar=False)))
    obs_mask = (R >= 0)
    nu0 = p + 2
    kept = []

    for it in range(n_iter):
        for j in rng.permutation(p):
            o = np.arange(p) != j
            Voo = _copula_spd(V[np.ix_(o, o)])
            Voj = V[o, j]

            beta = linalg.solve(Voo, Voj, assume_a="sym")
            mu = Z[:, o] @ beta
            var = max(float(V[j, j] - V[j, o] @ beta), 1e-10)
            sdj = np.sqrt(var)

            rj = R[:, j]
            oj = obs_mask[:, j]

            if np.any(oj):
                for r in np.unique(rj[oj]):
                    idx = np.where(rj == r)[0]
                    lo_pool = Z[oj & (rj < r), j]
                    hi_pool = Z[oj & (rj > r), j]
                    lo = -np.inf if lo_pool.size == 0 else np.max(lo_pool)
                    hi = np.inf if hi_pool.size == 0 else np.min(hi_pool)
                    a = (lo - mu[idx]) / sdj
                    b = (hi - mu[idx]) / sdj
                    Z[idx, j] = truncnorm.rvs(
                        a, b, loc=mu[idx], scale=sdj, random_state=rng
                    )

            miss = np.where(~oj)[0]
            if miss.size:
                Z[miss, j] = rng.normal(mu[miss], sdj, size=miss.size)

        S = _copula_spd(nu0 * np.eye(p) + Z.T @ Z)
        V = _copula_spd(
            np.asarray(invwishart.rvs(df=nu0 + n, scale=S, random_state=rng), dtype=float)
        )

        if it >= burn_in and (it - burn_in) % thin == 0:
            kept.append(_copula_corr(V))

    if len(kept) == 0:
        raise ValueError("No posterior draws kept; check MCMC settings")

    C_samples = np.stack(kept, axis=0)
    C_mean = _copula_corr(np.mean(C_samples, axis=0))
    n_eff = estimate_effective_n_from_samples(C_samples, raw_n=n)

    out = {
        "copula_matrix": pd.DataFrame(C_mean, index=cols, columns=cols),
        "effective_n": float(n_eff) if np.isfinite(n_eff) else np.nan,
        "columns": cols,
    }

    if return_samples:
        out["samples"] = C_samples

    return out


def compute_copula_matrix(
    df,
    cols=None,
    n_iter=600,
    burn_in=200,
    thin=2,
    random_state=0,
):
    """
    Backward-compatible wrapper returning only the posterior mean
    latent correlation matrix.
    """
    return compute_copula_fit(
        df=df,
        cols=cols,
        n_iter=n_iter,
        burn_in=burn_in,
        thin=thin,
        random_state=random_state,
        return_samples=False,
    )["copula_matrix"]


class Copula(DependenceMeasures):
    """
    Hoff-style semiparametric Gaussian copula dependence measure.

    Behavior
    --------
    - if cond_list is empty: returns latent correlation Corr(X, Y)
    - if cond_list exists: returns latent partial correlation Corr(X, Y | Z)

    Reuse
    -----
    You can pass either:
    - copula_fit: dict returned by compute_copula_fit(...)
    - copula_matrix: pd.DataFrame or np.ndarray

    If neither is given, the matrix is computed locally from the columns
    [x, y] + cond_list.
    """

    def __init__(
        self,
        x,
        y,
        cond_list=None,
        drop_na=False,
        copula_fit=None,
        copula_matrix=None,
        matrix_columns=None,
        n_iter=600,
        burn_in=200,
        thin=2,
        random_state=0,
    ):
        super().__init__(x, y, cond_list, drop_na)
        self.copula_fit = copula_fit
        self.copula_matrix = copula_matrix
        self.matrix_columns = matrix_columns
        self.n_iter = n_iter
        self.burn_in = burn_in
        self.thin = thin
        self.random_state = random_state

    def _cond_cols(self):
        if self.cond_list is None:
            return []
        if isinstance(self.cond_list, str):
            return [self.cond_list]
        return list(self.cond_list)

    def _matrix_to_df(self, M, cols, matrix_columns=None):
        if isinstance(M, pd.DataFrame):
            return M

        M = np.asarray(M, dtype=float)

        if matrix_columns is None:
            if M.shape[0] != len(cols) or M.shape[1] != len(cols):
                raise ValueError(
                    "matrix_columns must be provided when copula_matrix is a NumPy array "
                    "unless its shape matches [x, y] + cond_list exactly"
                )
            return pd.DataFrame(M, index=cols, columns=cols)

        matrix_columns = list(matrix_columns)
        if M.shape[0] != len(matrix_columns) or M.shape[1] != len(matrix_columns):
            raise ValueError("copula_matrix shape does not match matrix_columns")

        return pd.DataFrame(M, index=matrix_columns, columns=matrix_columns)

    def _get_matrix_df(self, df, cols):
        if self.copula_fit is not None:
            if isinstance(self.copula_fit, dict) and "copula_matrix" in self.copula_fit:
                return self._matrix_to_df(
                    self.copula_fit["copula_matrix"],
                    cols=cols,
                    matrix_columns=self.copula_fit.get("columns", None),
                )
            raise ValueError("copula_fit must be a dict returned by compute_copula_fit")

        if self.copula_matrix is not None:
            return self._matrix_to_df(
                self.copula_matrix,
                cols=cols,
                matrix_columns=self.matrix_columns,
            )

        # Fallback: compute only from needed columns
        local_df = self._prepare_data(df)
        return compute_copula_fit(
            df=local_df,
            cols=cols,
            n_iter=self.n_iter,
            burn_in=self.burn_in,
            thin=self.thin,
            random_state=self.random_state,
            return_samples=False,
        )["copula_matrix"]

    def get_dependence(self, df):
        cond = self._cond_cols()
        cols = [self.x, self.y] + cond

        try:
            Cdf = self._get_matrix_df(df, cols)
            Cdf = Cdf.loc[cols, cols]
            C = Cdf.to_numpy(dtype=float)
        except Exception:
            return np.nan

        try:
            if C.shape[0] < 2:
                return np.nan

            if len(cond) == 0:
                val = float(C[0, 1])
            else:
                PC = _copula_partial_corr(C)
                val = float(PC[0, 1])

            if not np.isfinite(val):
                return np.nan

            return float(np.clip(val, -1.0, 1.0))
        except Exception:
            return np.nan



class CMIh(DependenceMeasures):
    """
    Estimate conditional mutual information I(X;Y|Z) for mixed discrete and continuous variables
    using a hybrid approach. See cmih() function for details.

    """

    def __init__(self, x: Union[str, Sequence[str]], y: Union[str, Sequence[str]], cond_list: Optional[Union[str, Sequence[str]]] = None, drop_na=False, discrete_vars: Optional[Union[str, Sequence[str]]] = None, k: Optional[int] = None, discrete_alpha: float = 0.0):
        super().__init__(x, y, cond_list, drop_na)

        # discrete_vars: names of columns to treat as discrete for plugin terms
        self.discrete_vars = discrete_vars
        # k for kNN continuous entropy estimation
        self.k = k
        # Laplace pseudocount for discrete plugin entropy (default 0 = no smoothing)
        self.discrete_alpha = float(discrete_alpha)

    def _as_list(self, v):
        if v is None:
            return []
        if isinstance(v, Hashable):
            return [v]
        return list(v)


    def _make_discrete_keys(self, df: pd.DataFrame, cols: List[str]) -> np.ndarray:
        """
        Turn one or more discrete columns into hashable row-wise keys.
        """
        if len(cols) == 0:
            return np.array([("__NO_DISC__",)] * len(df), dtype=object)
        return np.array([tuple(row) for row in df[cols].itertuples(index=False, name=None)], dtype=object)


    def _plugin_entropy_discrete(self, df: pd.DataFrame, cols: List[str]) -> float:
        """
        Plug-in entropy estimator for purely discrete variables.
        Returns 0 when cols is empty.
        """
        if len(cols) == 0:
            return 0.0

        keys = self._make_discrete_keys(df, cols)
        # counts of each observed discrete key
        _, counts = np.unique(keys, return_counts=True)
        m = counts.size

        # Apply Laplace smoothing (add discrete_alpha to each observed bin)
        alpha = getattr(self, "discrete_alpha", 0.0)
        if alpha is None:
            alpha = 0.0
        if m == 0:
            return 0.0

        if alpha > 0.0:
            counts_sm = counts.astype(float) + float(alpha)
            n = counts_sm.sum()
        else:
            counts_sm = counts.astype(float)
            n = counts_sm.sum()

        if n == 0:
            return 0.0

        # plugin entropy (with smoothing if alpha>0)
        p = counts_sm / n
        with np.errstate(divide='ignore', invalid='ignore'):
            H_plugin = -np.sum(p * np.log(p))

        # Miller-Madow bias correction for discrete entropy: + (m - 1) / (2n)
        # where m = number of non-zero distinct bins observed (before smoothing)
        H_mm = H_plugin + float(max(0, m - 1)) / (2.0 * n)
        return float(H_mm)


    def _chebyshev_knn_entropy(self, X: np.ndarray, k: Optional[int] = None) -> float:
        """
        Kozachenko-Leonenko-style differential entropy estimator using the
        Chebyshev (L_infinity) norm and epsilon_i = 2 * distance to the k-th neighbor.

        For the max norm and this epsilon convention, the unit-ball volume term drops out.
        """
        X = np.asarray(X, dtype=float)
        # Ensure X is 2D: single-column inputs can come as shape (n,) from numpy
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        n, d = X.shape

        if d == 0:
            return 0.0
        if n < 2:
            return np.nan

        if k is None:
            k = max(1, int(np.sqrt(n)))
        k = min(max(1, k), n - 1)

        # Pairwise Chebyshev distances
        # dist[i, j] = max_l |X[i,l] - X[j,l]|
        diff = np.abs(X[:, None, :] - X[None, :, :])
        dist = diff.max(axis=2)

        # Ignore self-distance
        np.fill_diagonal(dist, np.inf)

        # k-th nearest neighbor distance
        kth = np.partition(dist, kth=k - 1, axis=1)[:, k - 1]

        # epsilon_i = 2 * distance_i
        eps = 2.0 * kth

        # Degenerate case: repeated points can give eps = 0
        if np.any(eps <= 0):
            return np.nan

        h = digamma(n) - digamma(k) + d * np.mean(np.log(eps))
        return float(h)

    def _conditional_entropy_cont_given_disc(self,
        df: pd.DataFrame,
        cont_cols: List[str],
        disc_cols: List[str],
        k: Optional[int] = None,
    ) -> float:
        """
        Estimate h(U_c | U_d) by averaging KL entropy estimates within each
        discrete bin of U_d.

        Returns:
        - 0 if cont_cols is empty
        - unconditional continuous entropy if disc_cols is empty
        """
        if len(cont_cols) == 0:
            return 0.0

        X = df[cont_cols].to_numpy(dtype=float)
        # Ensure X is 2D (n_samples, n_features)
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if len(disc_cols) == 0:
            return self._chebyshev_knn_entropy(X, k=k)

        keys = self._make_discrete_keys(df, disc_cols)
        # Force keys to be a 1D array of hashable keys (rows)
        keys = np.asarray(keys).ravel()
        total_n = len(df)

        out = 0.0
        grouped = df.groupby(disc_cols)
        total_n = len(df)
        out = 0.0
        for _, sub_df in grouped:
            sub = sub_df[cont_cols].to_numpy(dtype=float)
            if sub.ndim == 1:
                sub = sub.reshape(-1, 1)

            p_key = sub.shape[0] / total_n
            if sub.shape[0] < 2:
                # not enough samples in this bin to estimate entropy robustly
                return np.nan

            h_sub = self._chebyshev_knn_entropy(sub, k=k)
            if np.isnan(h_sub):
                return np.nan

            out += p_key * h_sub

        return float(out)

    def get_dependence(self, df: pd.DataFrame,) -> float:
        """
        Hybrid decomposition:
            I(X;Y|Z)
          = H(X_d,Z_d) + H(Y_d,Z_d) - H(X_d,Y_d,Z_d) - H(Z_d)
            + h(X_c,Z_c | X_d,Z_d)
            + h(Y_c,Z_c | Y_d,Z_d)
            - h(X_c,Y_c,Z_c | X_d,Y_d,Z_d)
            - h(Z_c | Z_d)

        where:
        - discrete entropy terms H(.) are plug-in histogram estimates
        - continuous conditional entropy terms h(. | .) are KL-style kNN estimates
          within discrete bins, using the Chebyshev norm

        Parameters
        ----------
        df : pd.DataFrame
        x, y, z : column name(s)
        discrete_vars : list of column names treated as discrete
        k : int, optional
            k for the kNN entropy estimator. If None, uses floor(sqrt(n_bin)) heuristically.
        drop_na : bool
            Whether to drop missing rows in used columns.

        Returns
        -------
        float
            Estimated conditional mutual information.
        """
        df = self._prepare_data(df)
        x_cols = self._as_list(self.x)
        y_cols = self._as_list(self.y)
        z_cols = self._as_list(self.cond_list)
        discrete_vars = set([] if self.discrete_vars is None else self.discrete_vars)

        cols = x_cols + y_cols + z_cols
        df_sub = df[cols].copy()

        if len(df_sub) == 0:
            return np.nan

        # Split each block into discrete and continuous parts
        x_d = [c for c in x_cols if c in discrete_vars]
        x_c = [c for c in x_cols if c not in discrete_vars]

        y_d = [c for c in y_cols if c in discrete_vars]
        y_c = [c for c in y_cols if c not in discrete_vars]

        z_d = [c for c in z_cols if c in discrete_vars]
        z_c = [c for c in z_cols if c not in discrete_vars]

        # If ALL variables are discrete (no continuous parts), compute CMI by
        # averaging conditional entropies per stratum using the plugin estimator
        # with Miller-Madow correction inside each stratum. This preserves the
        # plug-in/hybrid philosophy while reducing bias that arises from global
        # histogram estimation across many cells.
        if len(x_c) == 0 and len(y_c) == 0 and len(z_c) == 0:
            # No conditioning variables: simple plugin formula with Miller-Madow
            if len(z_cols) == 0:
                H_x = self._plugin_entropy_discrete(df_sub, x_d)
                H_y = self._plugin_entropy_discrete(df_sub, y_d)
                H_xy = self._plugin_entropy_discrete(df_sub, x_d + y_d)
                return float(H_x + H_y - H_xy)

            # Conditioning variables present: compute conditional entropies by
            # averaging per-stratum plugin entropies (each corrected by Miller-Madow)
            total_n = len(df_sub)
            if total_n == 0:
                return np.nan

            cmi = 0.0
            grouped = df_sub.groupby(z_cols, dropna=False)
            for _, sub_df in grouped:
                n_z = len(sub_df)
                if n_z == 0:
                    continue
                H_x_z = self._plugin_entropy_discrete(sub_df, x_d)
                H_y_z = self._plugin_entropy_discrete(sub_df, y_d)
                H_xy_z = self._plugin_entropy_discrete(sub_df, x_d + y_d)
                cmi += (n_z / total_n) * (H_x_z + H_y_z - H_xy_z)

            return float(cmi)

        # Discrete plug-in entropy terms
        H_xd_zd = self._plugin_entropy_discrete(df_sub, x_d + z_d)
        H_yd_zd = self._plugin_entropy_discrete(df_sub, y_d + z_d)
        H_xd_yd_zd = self._plugin_entropy_discrete(df_sub, x_d + y_d + z_d)
        H_zd = self._plugin_entropy_discrete(df_sub, z_d)

        # Continuous conditional entropy terms
        h_xc_zc_given_xd_zd = self._conditional_entropy_cont_given_disc(
            df_sub, cont_cols=x_c + z_c, disc_cols=x_d + z_d, k=self.k
        )
        h_yc_zc_given_yd_zd = self._conditional_entropy_cont_given_disc(
            df_sub, cont_cols=y_c + z_c, disc_cols=y_d + z_d, k=self.k
        )
        h_xc_yc_zc_given_xd_yd_zd = self._conditional_entropy_cont_given_disc(
            df_sub, cont_cols=x_c + y_c + z_c, disc_cols=x_d + y_d + z_d, k=self.k
        )
        h_zc_given_zd = self._conditional_entropy_cont_given_disc(
            df_sub, cont_cols=z_c, disc_cols=z_d, k=self.k
        )

        vals = [
            h_xc_zc_given_xd_zd,
            h_yc_zc_given_yd_zd,
            h_xc_yc_zc_given_xd_yd_zd,
            h_zc_given_zd,
        ]
        if any(np.isnan(v) for v in vals):
            return np.nan

        cmi = (
            H_xd_zd
            + H_yd_zd
            - H_xd_yd_zd
            - H_zd
            + h_xc_zc_given_xd_zd
            + h_yc_zc_given_yd_zd
            - h_xc_yc_zc_given_xd_yd_zd
            - h_zc_given_zd
        )
        return float(cmi)

    def get_cmi_components(self, df: pd.DataFrame) -> dict:
        """Compute and return the components used to form the hybrid CMI estimate.

        Returns a dict with keys:
          H_xd_zd, H_yd_zd, H_xd_yd_zd, H_zd,
          h_xc_zc_given_xd_zd, h_yc_zc_given_yd_zd, h_xc_yc_zc_given_xd_yd_zd, h_zc_given_zd,
          cmi

        This is only a diagnostic helper; it does not modify estimator behavior.
        """
        df = self._prepare_data(df)

        x_cols = self._as_list(self.x)
        y_cols = self._as_list(self.y)
        z_cols = self._as_list(self.cond_list)
        discrete_vars = set([] if self.discrete_vars is None else self.discrete_vars)

        cols = x_cols + y_cols + z_cols
        df_sub = df[cols].copy()

        if len(df_sub) == 0:
            return {"error": "empty_dataframe"}

        # Split each block into discrete and continuous parts
        x_d = [c for c in x_cols if c in discrete_vars]
        x_c = [c for c in x_cols if c not in discrete_vars]

        y_d = [c for c in y_cols if c in discrete_vars]
        y_c = [c for c in y_cols if c not in discrete_vars]

        z_d = [c for c in z_cols if c in discrete_vars]
        z_c = [c for c in z_cols if c not in discrete_vars]

        components = {}

        # Discrete plug-in entropy terms (with Miller-Madow correction)
        components["H_xd_zd"] = self._plugin_entropy_discrete(df_sub, x_d + z_d)
        components["H_yd_zd"] = self._plugin_entropy_discrete(df_sub, y_d + z_d)
        components["H_xd_yd_zd"] = self._plugin_entropy_discrete(df_sub, x_d + y_d + z_d)
        components["H_zd"] = self._plugin_entropy_discrete(df_sub, z_d)

        # Continuous conditional entropy terms
        components["h_xc_zc_given_xd_zd"] = self._conditional_entropy_cont_given_disc(
            df_sub, cont_cols=x_c + z_c, disc_cols=x_d + z_d, k=self.k
        )
        components["h_yc_zc_given_yd_zd"] = self._conditional_entropy_cont_given_disc(
            df_sub, cont_cols=y_c + z_c, disc_cols=y_d + z_d, k=self.k
        )
        components["h_xc_yc_zc_given_xd_yd_zd"] = self._conditional_entropy_cont_given_disc(
            df_sub, cont_cols=x_c + y_c + z_c, disc_cols=x_d + y_d + z_d, k=self.k
        )
        components["h_zc_given_zd"] = self._conditional_entropy_cont_given_disc(
            df_sub, cont_cols=z_c, disc_cols=z_d, k=self.k
        )

        # If any continuous-subterms are NaN, keep them as-is
        vals = [
            components["h_xc_zc_given_xd_zd"],
            components["h_yc_zc_given_yd_zd"],
            components["h_xc_yc_zc_given_xd_yd_zd"],
            components["h_zc_given_zd"],
        ]
        if any(np.isnan(v) for v in vals):
            components["cmi"] = np.nan
            return components

        cmi = (
            components["H_xd_zd"]
            + components["H_yd_zd"]
            - components["H_xd_yd_zd"]
            - components["H_zd"]
            + components["h_xc_zc_given_xd_zd"]
            + components["h_yc_zc_given_yd_zd"]
            - components["h_xc_yc_zc_given_xd_yd_zd"]
            - components["h_zc_given_zd"]
        )
        components["cmi"] = float(cmi)
        return components

