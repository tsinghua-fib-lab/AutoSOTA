from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
import warnings

from sklearn.neighbors import NearestNeighbors
from scipy import stats

from pandas.api.types import (
    is_bool_dtype,
    is_integer_dtype,
    is_object_dtype,
)

from pyciphod.utils.stat_tests.dependency_measures import DependenceMeasures, PartialCorrelation, LinearRegressionCoefficient, Gsq, CMIh
from pyciphod.causal_estimation.meta_learners import SLearner, TLearner, XLearner
from pyciphod.causal_estimation.outcome_regression import GComputation


class CeTests(DependenceMeasures, ABC):
    def __init__(self, x, y, cond_list=None, drop_na=False):
        # Pass x and y to the next __init__ in MRO so mixin classes that
        # expect (x, y) in their constructors (e.g. PartialCorrelation)
        # are correctly initialized and we avoid the TypeError seen earlier.
        super().__init__(x, y, cond_list, drop_na)
        self.x = x
        self.y = y
        self.cond_list = [] if cond_list is None else cond_list
        self.drop_na = drop_na

    @abstractmethod
    def get_pvalue(self, *args, **kwargs):
        """Abstract p-value interface. Subclasses may accept different
        argument signatures (e.g., equality tests take df1, df2, ...).
        """
        raise NotImplementedError

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

    def _permute_group_labels_within_strata(
        self,
        df_pre: pd.DataFrame,
        group_col: str,
        strata_cols,
        rng: np.random.Generator,
    ) -> pd.Series:
        """
        Exact stratified label permutation for discrete conditioning variables.
        Preserves label counts within each stratum.
        """
        new_group = df_pre[group_col].copy()
        grouped = df_pre.groupby(strata_cols, dropna=False).indices

        for _, idx in grouped.items():
            idx = np.asarray(idx)
            if len(idx) <= 1:
                continue

            # Convert groupby indices (which may be index labels) to positional indices
            pos_idx = df_pre.index.get_indexer(idx)
            # filter out any missing (-1) just in case
            pos_idx = pos_idx[pos_idx != -1]
            if len(pos_idx) <= 1:
                continue

            labels = df_pre.iloc[pos_idx][group_col].to_numpy().copy()
            rng.shuffle(labels)
            new_group.iloc[pos_idx] = labels

        return new_group

    def _local_swap_labels_in_subset(self,
        df_subset: pd.DataFrame,
        group_col: str,
        continuous_cols,
        rng: np.random.Generator,
        k_neighbors: int = None,
        n_swaps_multiplier: int = 5,
    ) -> pd.Series:
        """
        Approximate local label permutation in a subset using only continuous Z.
        Preserves the number of labels in the subset exactly by swapping labels.
        """
        n = len(df_subset)
        labels = df_subset[group_col].to_numpy().copy()

        if n <= 1:
            return pd.Series(labels, index=df_subset.index)

        if not continuous_cols:
            # If no continuous variables here, fall back to exact shuffle
            rng.shuffle(labels)
            return pd.Series(labels, index=df_subset.index)

        Z = df_subset[continuous_cols].copy().astype(float)

        # NearestNeighbors cannot handle NaN values; fail early with informative error
        if Z.isna().values.any():
            raise ValueError(
                "Continuous conditioning variables contain NaN. "
                "Either set drop_na=True on the test instance or pre-clean your data."
            )

        if k_neighbors is None:
            k_neighbors = min(max(5, int(np.sqrt(n))), n)

        nbrs = NearestNeighbors(n_neighbors=k_neighbors)
        nbrs.fit(Z.to_numpy())
        neigh_ind = nbrs.kneighbors(return_distance=False)

        n_swaps = n_swaps_multiplier * n

        for _ in range(n_swaps):
            i = rng.integers(0, n)
            candidates = neigh_ind[i]
            if len(candidates) <= 1:
                continue

            # avoid trivial self-swap when possible
            candidate_pool = candidates[candidates != i]
            if len(candidate_pool) == 0:
                continue

            j = rng.choice(candidate_pool)

            # swap only if labels differ, to preserve counts
            if labels[i] != labels[j]:
                labels[i], labels[j] = labels[j], labels[i]

        return pd.Series(labels, index=df_subset.index)

    def _permute_group_labels_mixed_z(
        self,
        df_pre: pd.DataFrame,
        group_col: str,
        discrete_vars,
        continuous_vars,
        rng: np.random.Generator,
        k_neighbors: int = None,
    ) -> pd.Series:
        """
        Mixed-Z permutation:
        - stratify by discrete vars
        - do local continuous permutation within each discrete stratum
        """
        new_group = df_pre[group_col].copy()

        grouped = df_pre.groupby(discrete_vars, dropna=False).indices
        for _, idx in grouped.items():
            idx = np.asarray(idx)
            if len(idx) <= 1:
                continue

            # convert to positional indices for iloc-based operations
            pos_idx = df_pre.index.get_indexer(idx)
            pos_idx = pos_idx[pos_idx != -1]
            if len(pos_idx) <= 1:
                continue

            subset = df_pre.iloc[pos_idx].copy()
            permuted_subset_group = self._local_swap_labels_in_subset(
                df_subset=subset,
                group_col=group_col,
                continuous_cols=continuous_vars,
                rng=rng,
                k_neighbors=k_neighbors,
            )
            # assign back using positional indices
            new_group.iloc[pos_idx] = permuted_subset_group.to_numpy()

        return new_group

    def get_pvalue_by_permutation(
        self,
        df1,
        df2,
        n_permutations: int = 1000,
        seed: int = None,
        k_neighbors: int = None,
    ):
        """
        Permutation p-value for equality of dependence across two populations.

        Null:
            dependence(df1) == dependence(df2)

        Permutation logic:
        - no conditioning variables: global label permutation
        - only discrete Z: exact stratified permutation
        - only continuous Z: local nearest-neighbor label permutation
        - mixed Z: stratify on discrete Z, local nearest-neighbor permutation
          on continuous Z within each stratum

        Notes:
        - rows (x, y, z) stay intact
        - for continuous or mixed Z, this is approximate rather than exact. The conditional permutation scheme for mixed Z is inspired by the local-permutation idea of Zan et al. (2022),
        """
        df1_pre = self._prepare_data(df1).copy()
        df2_pre = self._prepare_data(df2).copy()

        n1 = len(df1_pre)
        n2 = len(df2_pre)

        if n1 <= 1 or n2 <= 1:
            return np.nan

        try:
            obs1 = self.get_dependence(df1_pre)
            obs2 = self.get_dependence(df2_pre)
        except Exception as e:
            warnings.warn(f"Error computing observed dependence: {e}")
            return np.nan

        if obs1 is None or obs2 is None or pd.isna(obs1) or pd.isna(obs2):
            return np.nan

        obs_val = abs(obs1 - obs2)

        df1_pre["__group__"] = 0
        df2_pre["__group__"] = 1
        df_pre = pd.concat([df1_pre, df2_pre], ignore_index=True)

        rng = np.random.default_rng(seed)

        discrete_vars, continuous_vars = self._split_conditioning_vars(df_pre)

        count = 0
        valid = 0

        for it in range(n_permutations):
            try:
                # Case 1: no conditioning variables
                if not self.cond_list:
                    labels = np.array([0] * n1 + [1] * n2)
                    rng.shuffle(labels)
                    perm_group = pd.Series(labels, index=df_pre.index)

                # Case 2: only discrete conditioning variables
                elif len(discrete_vars) > 0 and len(continuous_vars) == 0:
                    perm_group = self._permute_group_labels_within_strata(
                        df_pre=df_pre,
                        group_col="__group__",
                        strata_cols=discrete_vars,
                        rng=rng,
                    )

                # Case 3: only continuous conditioning variables
                elif len(discrete_vars) == 0 and len(continuous_vars) > 0:
                    perm_group = self._local_swap_labels_in_subset(
                        df_subset=df_pre,
                        group_col="__group__",
                        continuous_cols=continuous_vars,
                        rng=rng,
                        k_neighbors=k_neighbors,
                    )

                # Case 4: mixed discrete + continuous conditioning variables
                else:
                    perm_group = self._permute_group_labels_mixed_z(
                        df_pre=df_pre,
                        group_col="__group__",
                        discrete_vars=discrete_vars,
                        continuous_vars=continuous_vars,
                        rng=rng,
                        k_neighbors=k_neighbors,
                    )

                df_perm = df_pre.copy()
                df_perm["__perm_group__"] = perm_group

                df1_perm = (
                    df_perm.loc[df_perm["__perm_group__"] == 0]
                    .drop(columns=["__group__", "__perm_group__"])
                    .reset_index(drop=True)
                )
                df2_perm = (
                    df_perm.loc[df_perm["__perm_group__"] == 1]
                    .drop(columns=["__group__", "__perm_group__"])
                    .reset_index(drop=True)
                )

                # total counts should remain unchanged
                if len(df1_perm) != n1 or len(df2_perm) != n2:
                    continue

                stat1 = self.get_dependence(df1_perm)
                stat2 = self.get_dependence(df2_perm)

                if stat1 is None or stat2 is None or pd.isna(stat1) or pd.isna(stat2):
                    continue

                stat = stat1 - stat2
                valid += 1

                if abs(stat) >= obs_val:
                    count += 1

            except Exception as e:
                warnings.warn(f"Permutation iteration {it} failed: {e}")
                continue

        if valid == 0:
            return np.nan

        return float((count + 1) / (valid + 1))


class PartialCorrelationEqualityTest(CeTests, PartialCorrelation):
    def __init__(self, x, y, cond_list=None, drop_na=False):
        super().__init__(x, y, cond_list, drop_na)
    def get_dependence(self, df):
        # Use the PartialCorrelation mixin's implementation
        return PartialCorrelation.get_dependence(self, df)

    def get_pvalue(self, df1, df2):
        raise NotImplementedError("Use get_pvalue_by_permutation for this test, as the distribution of the difference in partial correlations is not analytically tractable.")


class LinearRegressionCoefficientEqualityTest(CeTests, LinearRegressionCoefficient):
    def __init__(self, x, y, cond_list=None, drop_na=False):
        super().__init__(x, y, cond_list, drop_na)
    def get_dependence(self, df):
        # Use the LinearRegressionCoefficient mixin's implementation
        return LinearRegressionCoefficient.get_dependence(self, df)

    def get_pvalue(self, df1, df2):
        cond_list = [] if self.cond_list is None else list(self.cond_list)
        predictors = [self.x] + cond_list
        cols = [self.y] + predictors

        d1 = df1[cols].dropna()
        d2 = df2[cols].dropna()

        def fit_ols(d):
            y = d[self.y].to_numpy(dtype=float)
            X = d[predictors].to_numpy(dtype=float)

            # add intercept
            X = np.column_stack([np.ones(len(X)), X])

            n, p = X.shape
            if n <= p:
                return np.nan, np.nan

            XtX_inv = np.linalg.pinv(X.T @ X)
            beta = XtX_inv @ X.T @ y
            resid = y - X @ beta
            sigma2 = np.sum(resid ** 2) / (n - p)

            # coefficient and SE for self.x
            # index 1 because index 0 is intercept
            beta_x = beta[1]
            se_x = np.sqrt(sigma2 * XtX_inv[1, 1])

            return beta_x, se_x

        beta1, se1 = fit_ols(d1)
        beta2, se2 = fit_ols(d2)

        if np.isnan(beta1) or np.isnan(beta2):
            return np.nan

        z_stat = (beta1 - beta2) / np.sqrt(se1 ** 2 + se2 ** 2)
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

        return p_value
    # def get_pvalue(self, df1, df2):
    #     cond_list = [] if self.cond_list is None else list(self.cond_list)
    #     cols = [self.y, self.x] + cond_list
    #
    #     d1 = df1[cols].dropna()
    #     d2 = df2[cols].dropna()
    #
    #     X1 = sm.add_constant(d1[[self.x] + cond_list], has_constant="add")
    #     y1 = d1[self.y]
    #
    #     X2 = sm.add_constant(d2[[self.x] + cond_list], has_constant="add")
    #     y2 = d2[self.y]
    #
    #     m1 = sm.OLS(y1, X1).fit()
    #     m2 = sm.OLS(y2, X2).fit()
    #
    #     beta1 = m1.params[self.x]
    #     beta2 = m2.params[self.x]
    #
    #     se1 = m1.bse[self.x]
    #     se2 = m2.bse[self.x]
    #
    #     z_stat = (beta1 - beta2) / np.sqrt(se1 ** 2 + se2 ** 2)
    #     p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
    #
    #     return p_value
    # def get_pvalue(self, df1, df2):
    #     """
    #     Test equality of the regression coefficient of x in the regression
    #     y ~ x + cond_list across two datasets df1 and df2.
    #
    #     H0: beta_x^{(1)} = beta_x^{(2)}
    #
    #     Returns
    #     -------
    #     p_value : float
    #     """
    #     cond_list = [] if self.cond_list is None else list(self.cond_list)
    #
    #     cols = cond_list + [self.x]
    #     target = self.y
    #
    #     data1 = df1[cols + [target]].dropna()
    #     data2 = df2[cols + [target]].dropna()
    #
    #     X1 = data1[cols].to_numpy()
    #     y1 = data1[target].to_numpy()
    #
    #     X2 = data2[cols].to_numpy()
    #     y2 = data2[target].to_numpy()
    #
    #     n1, p = X1.shape
    #     n2, _ = X2.shape
    #
    #     if n1 <= p or n2 <= p:
    #         return np.nan
    #
    #     S1 = np.cov(data1[cols + [target]].to_numpy(), rowvar=False)
    #     S2 = np.cov(data2[cols + [target]].to_numpy(), rowvar=False)
    #
    #     idx_pred = np.arange(p)
    #     idx_y = p
    #
    #     K1 = np.linalg.pinv(S1[np.ix_(idx_pred, idx_pred)])
    #     K2 = np.linalg.pinv(S2[np.ix_(idx_pred, idx_pred)])
    #
    #     b1 = K1 @ S1[idx_y, idx_pred].T
    #     b2 = K2 @ S2[idx_y, idx_pred].T
    #
    #     residuals1 = y1 - X1 @ b1
    #     residuals2 = y2 - X2 @ b2
    #
    #     ssr1 = np.sum(residuals1 ** 2)
    #     ssr2 = np.sum(residuals2 ** 2)
    #
    #     var1 = ssr1 / (n1 - p)
    #     var2 = ssr2 / (n2 - p)
    #
    #     beta1_x = b1[-1]
    #     beta2_x = b2[-1]
    #
    #     se2 = var1 * K1[-1, -1] + var2 * K2[-1, -1]
    #
    #     if se2 <= 0:
    #         return np.nan
    #
    #     test_stat = (beta1_x - beta2_x) ** 2 / se2
    #
    #     df_denom = n1 + n2 - 2 * p
    #     p_value = 1.0 - stats.f.cdf(test_stat, 1, df_denom)
    #
    #     return p_value


class GsqEqualityTest(CeTests, Gsq):
    def __init__(self, x, y, cond_list=None, drop_na=False):
        super().__init__(x, y, cond_list, drop_na)

    def get_pvalue(self, *args, **kwargs):
        raise NotImplementedError("Use get_pvalue_by_permutation for this test, as the distribution of the difference in Gsq is not analytically tractable.")


class KernelPartialCorrelationEqualityTest(CeTests, PartialCorrelation):
    def __init__(self, x, y, cond_list=None, drop_na=False):
        super().__init__(x, y, cond_list, drop_na)

    def get_dependence(self, df):
        # Use the PartialCorrelation mixin's implementation
        return PartialCorrelation.get_dependence(self, df)

    def get_pvalue(self, *args, **kwargs):
        raise NotImplementedError("Use get_pvalue_by_permutation for this test, as the distribution of the difference in kernel partial correlations is not analytically tractable.")


class CMIhEqualityTest(CeTests, CMIh):
    def __init__(self, x, y, cond_list=None, drop_na=False):
        super().__init__(x, y, cond_list, drop_na)

    def get_pvalue(self, *args, **kwargs):
        raise NotImplementedError("Use get_pvalue_by_permutation for this test, as the distribution of the difference in CMIh is not analytically tractable.")


class SLearnerEqualityTest(CeTests):
    def __init__(self, x, y, cond_list=None, drop_na=False, model=None, seed=None):
        super().__init__(x, y, cond_list, drop_na)
        self.seed = seed

    def get_dependence(self, df):
        # Use the SLearner mixin's implementation
        ce = SLearner(self.x, self.y, z=self.cond_list, w=None, model=None, seed=self.seed)
        result = ce.run(df)
        if isinstance(result, dict) and 'cate' in result:
            return result['cate']
        else:
            raise ValueError("Unexpected result format from SLearner get_dependence")

    def get_pvalue(self, *args, **kwargs):
        raise NotImplementedError("Use get_pvalue_by_permutation for this test, as the distribution of the difference in CATE estimates is not analytically tractable.")

from sklearn.ensemble import RandomForestRegressor

class GComputationEqualityTest(CeTests):
    def __init__(self, x, y, cond_list=None, drop_na=False, seed=None):
        super().__init__(x, y, cond_list, drop_na)
        self.seed = seed

    def get_dependence(self, df):
        # Use the GComputation mixin's implementation
        # rf = RandomForestRegressor(n_estimators=100, random_state=self.seed)
        ce = GComputation(self.x, self.y, z=self.cond_list, w=None, seed=self.seed)
        result = ce.run(df)
        if isinstance(result, dict) and 'cate' in result:
            return result['cate']
        else:
            raise ValueError("Unexpected result format from GComputation get_dependence")

    def get_pvalue(self, *args, **kwargs):
        raise NotImplementedError("Use get_pvalue_by_permutation for this test, as the distribution of the difference in CATE estimates is not analytically tractable.")