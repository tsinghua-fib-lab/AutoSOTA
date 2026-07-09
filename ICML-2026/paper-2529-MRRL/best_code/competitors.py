import numpy as np
from sklearn.linear_model import LinearRegression


class MREgger:
    """
    MR-Egger Regression for individual-level data.

    1. Computes summary statistics (association of Z with Exposure D and Outcome Y)
       for each instrument Z_j, adjusting for controls C if provided.
    2. Orients instruments so that Z-Exposure associations are positive.
    3. Performs weighted linear regression of Z-Outcome coefs on Z-Exposure coefs.
       - Slope = Causal Effect estimate
       - Intercept = Average Pleiotropic Effect
    """

    def __init__(self):
        self.coef_ = None  # The causal effect estimate (beta)
        self.intercept_ = None  # The pleiotropy estimate
        self.n_ivs_ = 0

    def fit(self, Z, X, y, C=None):
        """
        Z: Instruments (n_samples, n_instruments)
        X: Exposure/Treatment (n_samples, 1)
        y: Outcome (n_samples, 1)
        C: Covariates/Controls (n_samples, n_controls)
        """
        Z = np.asarray(Z)
        X = np.asarray(X)
        y = np.asarray(y)

        self.n_ivs_ = Z.shape[1]

        gamma_hats = []  # Coefficients of Z -> X
        Gamma_hats = []  # Coefficients of Z -> Y
        Gamma_ses = []  # Standard Errors of Z -> Y

        # Step 1: Generate Summary Statistics for each IV
        for j in range(self.n_ivs_):
            z_j = Z[:, j : j + 1]

            # Construct features: [Z_j, Controls]
            if C is not None:
                features = np.hstack([z_j, C])
            else:
                features = z_j

            # A. Exposure Association (gamma): X ~ Z_j + C
            # We use sklearn here for speed/stability, only need the coef
            reg_x = LinearRegression(fit_intercept=True).fit(features, X)
            gamma = reg_x.coef_.flatten()[0]  # Coef for Z_j is at index 0

            # B. Outcome Association (Gamma): Y ~ Z_j + C
            # We need the SE of the coefficient for weights.
            # Calculating OLS manually to get SE efficiently without statsmodels dependency.
            G, G_se = self._ols_coef_and_se(features, y)

            gamma_hats.append(gamma)
            Gamma_hats.append(G)
            Gamma_ses.append(G_se)

        gamma_hats = np.array(gamma_hats)
        Gamma_hats = np.array(Gamma_hats)
        Gamma_ses = np.array(Gamma_ses)

        # Step 2: Orient Instruments
        # MR-Egger assumes all instruments have a positive effect on the exposure
        mask_flip = gamma_hats < 0
        gamma_hats[mask_flip] *= -1
        Gamma_hats[mask_flip] *= -1

        # Step 3: Weighted Egger Regression
        # Gamma_j = beta_0 + beta_1 * gamma_j
        # Weights = 1 / SE(Gamma_j)^2 (Inverse Variance)

        # Safety check: avoid div by zero if SE is 0 (unlikely in real data)
        weights = 1.0 / (Gamma_ses**2 + 1e-10)

        egger_model = LinearRegression(fit_intercept=True)
        egger_model.fit(
            gamma_hats.reshape(-1, 1), Gamma_hats, sample_weight=weights
        )

        # Store results compatible with KClass style (coef_ as array)
        self.coef_ = egger_model.coef_
        self.intercept_ = egger_model.intercept_

        return self

    def _ols_coef_and_se(self, X, y):
        """Helper to get Z_j coef and its SE from OLS of y ~ X"""
        # X shape (N, K), we assume first col is Z_j
        N = X.shape[0]

        # Add intercept column for matrix math
        X_mat = np.hstack([np.ones((N, 1)), X])  # Cols: [Intercept, Z_j, C...]

        # Solve (X'X)b = X'y
        try:
            XTX_inv = np.linalg.inv(X_mat.T @ X_mat)
            beta = XTX_inv @ (X_mat.T @ y)
        except np.linalg.LinAlgError:
            # Fallback for singular matrix
            return 0.0, 1.0

        # Residuals
        y_pred = X_mat @ beta
        residuals = y - y_pred
        rss = np.sum(residuals**2)

        # MSE = RSS / (N - p)
        p = X_mat.shape[1]
        mse = rss / (N - p)

        # SE(beta) = sqrt(diagonal of MSE * (X'X)^-1)
        # Beta index for Z_j is 1 (0 is intercept)
        se_beta = np.sqrt(mse * XTX_inv[1, 1])

        # Z_j coef is at index 1
        return beta[1].item(), se_beta
