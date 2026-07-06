from abc import ABC, abstractmethod
import warnings
import pandas as pd
import numpy as np
from scipy.stats import norm, chi2

from sklearn.feature_selection import f_regression as fr
from sklearn.neighbors import NearestNeighbors
from pandas.api.types import (
    is_bool_dtype,
    is_integer_dtype,
    is_object_dtype,
)


from .dependency_measures import DependenceMeasures, PartialCorrelation, LinearRegressionCoefficient, Gsq, CMIh, KernelPartialCorrelation, Copula


class CiTests(DependenceMeasures, ABC):
    def __init__(self, x, y, cond_list=None, drop_na=False):
        # Ensure DependenceMeasures is initialized with the same args
        super().__init__(x, y, cond_list, drop_na)
        self.x = x
        self.y = y
        if cond_list is None:
            self.cond_list = []
        else:
            self.cond_list = cond_list
        self.drop_na = drop_na


    @abstractmethod
    def get_pvalue(self, df):
        pass

    def _is_discrete_series(self, s: pd.Series) -> bool:
        return (
                is_bool_dtype(s)
                or isinstance(s.dtype, pd.CategoricalDtype)
                or is_object_dtype(s)
                or is_integer_dtype(s)
        )

    def _split_conditioning_vars(self, df: pd.DataFrame):
        discrete = []
        continuous = []
        for c in self.cond_list:
            if self._is_discrete_series(df[c]):
                discrete.append(c)
            else:
                continuous.append(c)
        return discrete, continuous

    def _local_swap_x_in_subset(
            self,
            df_subset: pd.DataFrame,
            x_col: str,
            continuous_cols,
            rng: np.random.Generator,
            k_neighbors: int = None,
            n_swaps_multiplier: int = 5,
    ) -> pd.Series:
        """
        Approximate local permutation of X in a subset using only continuous Z.
        Preserves the multiset of X values in the subset exactly by swapping X values.
        """
        n = len(df_subset)
        x_vals = df_subset[x_col].to_numpy().copy()

        if n <= 1:
            return pd.Series(x_vals, index=df_subset.index)

        if not continuous_cols:
            rng.shuffle(x_vals)
            return pd.Series(x_vals, index=df_subset.index)

        Z = df_subset[continuous_cols].copy().astype(float)

        if Z.isna().values.any():
            raise ValueError(
                "Continuous conditioning variables contain NaN. "
                "Either set drop_na=True on the test instance or pre-clean your data."
            )

        if k_neighbors is None:
            k_neighbors = min(max(5, int(np.sqrt(n))), n)

        # include self among neighbors; later we exclude it when possible
        nbrs = NearestNeighbors(n_neighbors=k_neighbors)
        nbrs.fit(Z.to_numpy())
        neigh_ind = nbrs.kneighbors(return_distance=False)

        n_swaps = n_swaps_multiplier * n

        for _ in range(n_swaps):
            i = rng.integers(0, n)
            candidates = neigh_ind[i]
            if len(candidates) <= 1:
                continue

            candidate_pool = candidates[candidates != i]
            if len(candidate_pool) == 0:
                continue

            j = rng.choice(candidate_pool)

            # swap X values locally
            x_vals[i], x_vals[j] = x_vals[j], x_vals[i]

        return pd.Series(x_vals, index=df_subset.index)

    def _permute_x_mixed_z(
            self,
            df_pre: pd.DataFrame,
            x_col: str,
            discrete_vars,
            continuous_vars,
            rng: np.random.Generator,
            k_neighbors: int = None,
    ) -> pd.Series:
        """
        Mixed-Z permutation:
        - stratify by discrete vars
        - do local continuous permutation of X within each discrete stratum
        """
        new_x = df_pre[x_col].copy()

        grouped = df_pre.groupby(discrete_vars, dropna=False).indices
        for _, idx in grouped.items():
            idx = np.asarray(idx)
            if len(idx) <= 1:
                continue

            # groupby(...).indices already gives positional indices
            subset = df_pre.iloc[idx].copy()
            permuted_subset_x = self._local_swap_x_in_subset(
                df_subset=subset,
                x_col=x_col,
                continuous_cols=continuous_vars,
                rng=rng,
                k_neighbors=k_neighbors,
            )
            new_x.iloc[idx] = permuted_subset_x.to_numpy()

        return new_x

    def get_pvalue_by_permutation(
        self,
        df,
        n_permutations: int = 1000,
        seed: int = None,
        k_neighbors: int = None,
    ):
        """
        Permutation p-value for independence .

        Null:
            dependence(df) == 0

        Permutation logic:
        - no conditioning variables: global label permutation
        - only discrete Z: exact stratified permutation
        - only continuous Z: local nearest-neighbor label permutation
        - mixed Z: stratify on discrete Z, local nearest-neighbor permutation
          on continuous Z within each stratum

        Notes:
        - rows (x, y, z) stay intact
        - for continuous or mixed Z, this is approximate rather than exact. The conditional permutation scheme for mixed Z corresponds to the local-permutation test of Zan et al. (2022),
        """
        df_pre = self._prepare_data(df).copy()

        n = len(df_pre)

        if n <= 1:
            return np.nan

        try:
            obs = self.get_dependence(df_pre)
        except Exception as e:
            warnings.warn(f"Error computing observed dependence: {e}")
            return np.nan

        if obs is None or pd.isna(obs):
            return np.nan

        obs_val = abs(float(obs))

        rng = np.random.default_rng(seed)

        discrete_vars, continuous_vars = self._split_conditioning_vars(df_pre)

        count = 0
        valid = 0

        x_orig = df_pre[self.x].to_numpy()

        for it in range(n_permutations):
            try:
                # Case 1: no conditioning variables
                if not self.cond_list:
                    x_perm = x_orig[rng.permutation(n)]

                # Case 2: only discrete conditioning variables
                elif len(discrete_vars) > 0 and len(continuous_vars) == 0:
                    x_perm = x_orig.copy()
                    groups = df_pre.groupby(discrete_vars, dropna=False).indices

                    changed = False
                    for _, idx in groups.items():
                        idx = np.asarray(idx)
                        if len(idx) <= 1:
                            continue
                        perm = rng.permutation(len(idx))
                        x_perm[idx] = x_orig[idx][perm]
                        if not np.all(perm == np.arange(len(idx))):
                            changed = True

                    if not changed:
                        continue

                # Case 3: only continuous conditioning variables
                elif len(discrete_vars) == 0 and len(continuous_vars) > 0:
                    x_perm = self._local_swap_x_in_subset(
                        df_subset=df_pre,
                        x_col=self.x,
                        continuous_cols=continuous_vars,
                        rng=rng,
                        k_neighbors=k_neighbors,
                    ).to_numpy()

                # Case 4: mixed discrete + continuous conditioning variables
                else:
                    x_perm = self._permute_x_mixed_z(
                        df_pre=df_pre,
                        x_col=self.x,
                        discrete_vars=discrete_vars,
                        continuous_vars=continuous_vars,
                        rng=rng,
                        k_neighbors=k_neighbors,
                    ).to_numpy()

                df_perm = df_pre.copy()
                df_perm[self.x] = x_perm

                # total counts should remain unchanged
                if len(df_perm) != n:
                    continue

                stat = self.get_dependence(df_perm)
                if stat is None or pd.isna(stat):
                    continue

                valid += 1

                if abs(stat) >= obs_val:
                    count += 1

            except Exception as e:
                warnings.warn(f"Permutation iteration {it} failed: {e}")
                continue

        if valid == 0:
            return np.nan
        return float((count + 1) / (valid + 1))

    def old_get_pvalue_by_permutation(self, df, n_permutations: int = 1000, seed: int = None):
        """
        Generic permutation-based p-value estimator for conditional independence tests.

        Procedure (Monte-Carlo):
        - Prepare the data via _prepare_data.
        - Compute the observed dependence statistic via get_dependence.
        - For each permutation: permute the values of X (stratified within conditioning set if present),
          recompute the dependence statistic on the permuted dataset.
        - Compute the p-value as (count_perm_ge_obs + 1) / (n_permutations + 1) (to avoid zero p-values).

        Args:
            df (pd.DataFrame): data frame with the variables.
            n_permutations (int): number of permutations to perform (default 1000).
            seed (int): optional random seed for reproducibility.

        Returns:
            float: estimated p-value (or np.nan if test statistic cannot be computed).
        """
        df_pre = self._prepare_data(df)

        # Need sufficient samples
        if df_pre.shape[0] <= 1:
            return np.nan

        # observed statistic
        try:
            obs = self.get_dependence(df_pre)
        except Exception:
            return np.nan

        if obs is None or (isinstance(obs, float) and np.isnan(obs)):
            return np.nan

        obs_val = abs(obs)

        rng = np.random.default_rng(seed)
        X_orig = df_pre[self.x].values
        n = len(X_orig)

        count = 0
        # Precompute group indices if conditioning set present
        if self.cond_list:
            try:
                groups = df_pre.groupby(self.cond_list).indices
            except Exception:
                # fallback to no stratification
                groups = None
        else:
            groups = None

        for _ in range(n_permutations):
            permuted_x = np.empty_like(X_orig)
            if groups and len(groups) > 0:
                # Permute within each group
                permuted_x[:] = X_orig
                for _, idx in groups.items():
                    if len(idx) <= 1:
                        continue
                    permuted_x[idx] = X_orig[np.array(idx)[rng.permutation(len(idx))]]
            else:
                permuted_x = X_orig[rng.permutation(n)]

            df_perm = df_pre.copy()
            df_perm[self.x] = permuted_x

            try:
                stat = self.get_dependence(df_perm)
            except Exception:
                # ignore permutations that fail
                continue

            if stat is None or (isinstance(stat, float) and np.isnan(stat)):
                continue

            if abs(stat) >= obs_val:
                count += 1

        # p-value with +1 correction
        pval = (count + 1) / (n_permutations + 1)
        return float(pval)


class LinearRegressionCoefficientTTest(CiTests, LinearRegressionCoefficient):
    def __init__(self, x, y, cond_list=None, drop_na=False):
        super().__init__(x, y, cond_list, drop_na)

    # def get_dependence(self, df):
    #     # df = self._prepare_data(df)
    #     # X_data = df[[self.x] + self.cond_list].values
    #     # Y_data = df[self.y].values
    #     # reg = lr().fit(X_data, Y_data)
    #     dep = LinearRegressionCoefficient(self.x, self.y, self.cond_list, self.drop_na)
    #     res = dep.get_dependence(df)
    #     return res

    def get_pvalue(self, df):
        df = self._prepare_data(df)
        X_data = df[[self.x] + self.cond_list].values
        Y_data = df[self.y].values
        pval = fr(X_data, Y_data)[1][0]
        return pval


class FisherZTest(CiTests, PartialCorrelation):
    """
    Continuous conditional independence test using the Fisher Z-transform.
    Suitable for Gaussian data. Computes partial correlations and corresponding p-values.
    Reference: Spirtes et al., "Causation, Prediction, and Search".
    """

    def __init__(self, x, y, cond_list=None, drop_na=False):
        """
        Initialize the test with variables X, Y and optional conditioning set Z.
        :param x: str, name of first variable
        :param y: str, name of second variable
        :param cond_list: list of str, conditioning variables
        """
        super().__init__(x, y, cond_list, drop_na)

    # def get_dependence(self, df):
    #     """
    #     Compute the partial correlation between X and Y given Z.
    #     Returns a correlation coefficient r in [-1,1].
    #     """
    #     df = self._prepare_data(df)
    #     dep = PartialCorrelation(self.x, self.y, self.cond_list, self.drop_na)
    #     r = dep.get_dependence(df)
    #
    #     # list_nodes = [self.x, self.y] + self.cond_list
    #     # df = df[list_nodes]
    #     # a = df.values.T
    #     #
    #     # if len(self.cond_list) > 0:
    #     #     cond_list_int = [i + 2 for i in range(len(self.cond_list))]
    #     # else:
    #     #     cond_list_int = []
    #     #
    #     # correlation_matrix = np.corrcoef(a)
    #     # var = list((0, 1) + tuple(cond_list_int))
    #     # sub_corr_matrix = correlation_matrix[np.ix_(var, var)]
    #     #
    #     # if np.linalg.det(sub_corr_matrix) == 0:
    #     #     r = 1
    #     # else:
    #     #     inv = np.linalg.inv(sub_corr_matrix)
    #     #     r = -inv[0, 1] / np.sqrt(inv[0, 0] * inv[1, 1])
    #     return r

    def get_pvalue(self, df):
        """
        Compute the p-value for the partial correlation.
        Uses the Fisher Z-transform:
            z = 0.5 * ln((1+r)/(1-r))
            test statistic = sqrt(n - |Z| - 3) * |z|
        Returns a two-sided p-value.
        """
        df = self._prepare_data(df)
        r = self.get_dependence(df)
        # validate correlation
        if r is None or not np.isfinite(r):
            return np.nan

        # Effective sample size for Fisher Z
        n_eff = df.shape[0] - len(self.cond_list) - 3
        if n_eff <= 0:
            return np.nan

        # Clip r away from ±1 to avoid numerical blowups
        eps = 1e-12
        r = float(np.clip(r, -1 + eps, 1 - eps))

        z = 0.5 * np.log((1 + r) / (1 - r))
        test_stat = np.sqrt(n_eff) * abs(z)
        pval = 2 * (1 - norm.cdf(abs(test_stat)))
        return float(pval)


class GsqTest(CiTests, Gsq):
    """
    Discrete conditional independence test using the G² likelihood-ratio test.
    Reference: Spirtes et al., "Causation, Prediction, and Search".
    """

    def __init__(self, x, y, cond_list=None, drop_na=False):
        super().__init__(x, y, cond_list, drop_na)

    def get_pvalue(self, df):
        """
        Returns the p-value of the G² test.
        Degrees of freedom = (|X|-1)*(|Y|-1)*prod(|Z_i|) for each conditioning state.
        """
        df = self._prepare_data(df)
        tables = self._contingency_table(df)
        g2_stat = self.get_dependence(df)
        # Compute degrees of freedom
        df_total = 0
        for table in tables.values():
            r, c = table.shape
            df_total += (r - 1) * (c - 1)

        if df_total <= 0 or not np.isfinite(g2_stat):
            return np.nan

        pval = 1 - chi2.cdf(g2_stat, df_total)
        return float(pval)


class KernelPartialCorrelationTest(CiTests, KernelPartialCorrelation):
    """
    Kernel-based conditional independence test using the Hilbert-Schmidt Independence Criterion (HSIC).
    Suitable for nonlinear relationships and mixed data types. Uses permutation testing for p-value estimation.
    Reference: Zhang et al., "Kernel-based Conditional Independence Test and Application in Causal Discovery".
    """

    def __init__(self, x, y, cond_list=None, drop_na=False):
        super().__init__(x, y, cond_list, drop_na)

    def get_pvalue(self, df, n_permutations: int = 1000, seed: int = None):
        """Estimate a permutation p-value for the CMIh statistic using the
        generic permutation method implemented in `get_pvalue_by_permutation`.

        Note: this can be expensive; choose n_permutations accordingly.
        """
        # Delegate to the generic permutation-based p-value estimator
        return self.get_pvalue_by_permutation(df, n_permutations=n_permutations, seed=seed)
        # raise NotImplementedError("KernelPartialCorrelationTest relies on get_pvalue_by_permutation for p-value estimation. Call that method directly.")


class CopulaTest(CiTests, Copula):
    """
    Approximate conditional independence test for mixed data using the
    Hoff-style Gaussian copula model.

    Test:
    - latent correlation if cond_list is empty
    - latent partial correlation if cond_list is not empty
    - p-value from Fisher-z transform

    Reuse
    -----
    You can pass either:
    - copula_fit: dict returned by compute_copula_fit(...)
    - copula_matrix + effective_n
    - nothing, in which case the code computes a local fit
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
        effective_n=None,
        n_iter=600,
        burn_in=200,
        thin=2,
        random_state=0,
    ):
        super().__init__(
            x=x,
            y=y,
            cond_list=cond_list,
            drop_na=drop_na,
        )
        self.copula_fit = copula_fit
        self.copula_matrix = copula_matrix
        self.matrix_columns = matrix_columns
        self.n_iter = n_iter
        self.burn_in = burn_in
        self.thin = thin
        self.random_state = random_state
        self.effective_n = effective_n

    def _get_effective_n(self, df):
        if self.effective_n is not None:
            try:
                n_eff = float(self.effective_n)
                return n_eff if np.isfinite(n_eff) else np.nan
            except Exception:
                return np.nan

        if isinstance(self.copula_fit, dict) and "effective_n" in self.copula_fit:
            try:
                n_eff = float(self.copula_fit["effective_n"])
                return n_eff if np.isfinite(n_eff) else np.nan
            except Exception:
                return np.nan

        # Fallback if no global effective_n was supplied
        try:
            local_df = self._prepare_data(df)
            return float(local_df.shape[0])
        except Exception:
            return np.nan

    def get_pvalue(self, df):
        cond = self._cond_cols()
        k = len(cond)

        n_eff = self._get_effective_n(df)
        if not np.isfinite(n_eff) or n_eff <= k + 3:
            return np.nan

        try:
            r = Copula.get_dependence(self, df)
        except Exception:
            return np.nan

        if not np.isfinite(r):
            return np.nan

        r = float(np.clip(r, -0.999999, 0.999999))

        try:
            fisher_z = 0.5 * np.log((1.0 + r) / (1.0 - r))
            stat = np.sqrt(n_eff - k - 3.0) * abs(fisher_z)
            pval = 2.0 * (1.0 - norm.cdf(stat))
            return float(pval)
        except Exception:
            return np.nan


class CIMhTest(CiTests, CMIh):
    """
    Conditional independence test based on the Maximal Information Coefficient (MIC).
    Suitable for continuous data. Uses permutation testing for p-value estimation.
    Reference: Reshef et al., "Detecting Novel Associations in Large Data Sets".
    """

    def __init__(self, x, y, cond_list=None, drop_na=False):
        super().__init__(x, y, cond_list, drop_na)

    # def get_dependence(self, df):
    #     dep = CMIh(self.x, self.y, self.cond_list, self.drop_na)
    #     res = dep.get_dependence(df)
    #     return res

    def get_pvalue(self, df, n_permutations: int = 10, seed: int = None):
        """Estimate a permutation p-value for the CMIh statistic using the
        generic permutation method implemented in `get_pvalue_by_permutation`.

        Note: this can be expensive; choose n_permutations accordingly.
        """
        # Delegate to the generic permutation-based p-value estimator
        pval = self.get_pvalue_by_permutation(df, n_permutations=n_permutations, seed=seed)
        return pval
        # raise NotImplementedError("CMIhTest relies on get_pvalue_by_permutation for p-value estimation. Call that method directly.")
