from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.kernel_ridge import KernelRidge
import numpy as np


class FairKernelLearningKRR:  # Translated the Matlab code available at http://isp.uv.es/ by Adrian Perez-Suay
    def __init__(self, alpha, gamma):
        self.alpha = alpha
        self.gamma = gamma
        self.KRR = KernelRidge(kernel="precomputed", alpha=alpha)

    def train(self, X, y, p, mu_reg):
        """
        Parameters
        -------

        X : ndarray
            Training data. Shape is assumed: (n_samples, n_features).

        y : ndarray
            Training labels. Shape is assumed: (n_samples, n_features).

        p : ndarray
            Protected attribute. Shape is assumed: (n_samples, n_features).

        Returns
        -------
        None

        """
        X = np.array(X, dtype=np.float64)
        y = np.array(y, dtype=np.float64)
        p = np.array(p, dtype=np.float64)


        n_train = np.shape(X)[0]

        K = pairwise_kernels(X, X, metric="rbf", gamma=self.gamma)
        KS = pairwise_kernels(
            p.reshape(n_train, -1),
            p.reshape(n_train, -1),
            metric="rbf",
            gamma=self.gamma,
        )

        H = np.eye(n_train) - 1 / n_train * np.ones((n_train, n_train))
        HKSH = H @ KS @ H

        KERNEL = K + mu_reg * HKSH @ K

        self.KRR.fit(KERNEL, y)

        self.training_data = X

    def predict(self, X_test):
        """
        Parameters
        -------

        X_test : ndarray
            Testing data. Shape is assumed: (n_samples, n_features).

        Returns
        -------
        ndarray
            Prediction given the testing data.

        """

        X_test = np.array(X_test, dtype=np.float64)


        K_test = pairwise_kernels(
            X_test, self.training_data, metric="rbf", gamma=self.gamma
        )

        return np.array(self.KRR.predict(K_test), dtype=np.float32)
