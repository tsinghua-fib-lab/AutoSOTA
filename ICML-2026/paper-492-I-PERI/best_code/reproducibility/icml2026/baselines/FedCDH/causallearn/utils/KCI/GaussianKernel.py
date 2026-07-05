from __future__ import annotations

from numpy import exp, median, shape, sqrt, ndarray, zeros
from numpy.random import permutation
from scipy.spatial.distance import cdist, pdist, squareform

from causallearn.utils.KCI.Kernel import Kernel


class GaussianKernel(Kernel):
    def __init__(self, width=1.0):
        Kernel.__init__(self)
        self.width: float = 1.0 / width ** 2

    def kernel(self, X: ndarray, Y: ndarray | None = None):
        """
        Original Gaussian Kernel Method:
            Computes the Gaussian kernel k(x,y)=exp(-0.5* ||x-y||**2 / sigma**2)=exp(-0.5* ||x-y||**2 *self.width)
        """
        flag = 0

        if flag==0: 
            # ---------------- Method 0: original kernel-based KCI.-------------------------
            if Y is None:
                sq_dists = squareform(pdist(X, 'sqeuclidean'))
            else:
                assert (shape(X)[1] == shape(Y)[1])
                sq_dists = cdist(X, Y, 'sqeuclidean')
            gamma = self.width 
            K = exp(-0.5 * sq_dists * gamma)
        
        elif flag==1:
            # ---------------- Method 1: random features. --------------------------------------
            from sklearn.kernel_approximation import RBFSampler
            # a, b = X.shape
            # best_n = 1*int(sqrt(a))
            rbf_feature = RBFSampler(gamma=1.0, n_components=100, random_state=1)
            X_features = rbf_feature.fit_transform(X) # print(X_features.shape, X_features.T.shape)
            K = X_features @ X_features.T # print(K.shape)
            # -----------------------------------------------------------------------------------

        elif flag==2:
            # ---------------- Method 2: Nystroem. --------------------------------------
            from sklearn.kernel_approximation import Nystroem
            # a, b = X.shape
            # best_n = 1*int(sqrt(a))
            feature_map_nystroem = Nystroem(gamma=0.2, n_components=100, random_state=1)   # gamma=0.2
            X_features = feature_map_nystroem.fit_transform(X) # print(X_features.shape, X_features.T.shape)
            K = X_features @ X_features.T # print(K.shape)
            # -----------------------------------------------------------------------------------

        else:
            # ---------------- Failed Method: fill out off-diagonal with 0s.-------------------------
            a, b = X.shape
            K = 12       # number of client.
            w = int(a/K) # each clients have w samples.
            mask = zeros((a,a))
            for k in range(K):
                start = k*w
                end = (k+1)*w
                mask[start:end,start:end] = 1
            sq_dists = sq_dists * mask
            K = exp(-0.5 * sq_dists * self.width)
            # ----------------------------------------------------------------------------------

        return K


    # kernel width using median trick
    def set_width_median(self, X: ndarray):
        n = shape(X)[0]
        if n > 1000:
            X = X[permutation(n)[:1000], :]
        dists = squareform(pdist(X, 'euclidean'))
        median_dist = median(dists[dists > 0])
        width = sqrt(2.) * median_dist
        theta = 1.0 / (width ** 2)
        self.width = theta

    # use empirical kernel width instead of the median
    def set_width_empirical_kci(self, X: ndarray):
        n = shape(X)[0]
        if n < 200:
            width = 1.2
        elif n < 1200:
            width = 0.7
        else:
            width = 0.4
        theta = 1.0 / (width ** 2)
        self.width = theta / X.shape[1]

    def set_width_empirical_hsic(self, X: ndarray):
        n = shape(X)[0]
        if n < 200:
            width = 0.8
        elif n < 1200:
            width = 0.5
        else:
            width = 0.3
        theta = 1.0 / (width ** 2)
        self.width = theta * X.shape[1]
        # self.width = theta
