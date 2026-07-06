import numpy as np
from numpy import sqrt
from numpy.linalg import eigh, eigvalsh
from scipy import stats
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process.kernels import ConstantKernel as C
from sklearn.gaussian_process.kernels import WhiteKernel

from causallearn.utils.KCI.GaussianKernel import GaussianKernel
from causallearn.utils.KCI.Kernel import Kernel
from causallearn.utils.KCI.LinearKernel import LinearKernel
from causallearn.utils.KCI.PolynomialKernel import PolynomialKernel

# from scipy import stats
import random  
import math 
import time 
from numpy.linalg import inv
# import warnings
# import arviz as az
# import matplotlib.pyplot as plt
# import theano.tensor as tt
# warnings.simplefilter(action="ignore", category=FutureWarning)
# MV2_USE_THREAD_WARNING=0

# try:
#     import rpy2
#     import rpy2.robjects
#     rpy2.robjects.r['options'](warn=-1)
#     from rpy2.robjects.packages import importr
#     import rpy2.robjects.numpy2ri
#     rpy2.robjects.numpy2ri.activate()
# except:
#     print("Could not import rpy package")

# try:
#     importr('RCIT')
# except:
#     print("Could not import r-package RCIT")
# import random 

from reproducibility.icml2026.baselines.rcit.rcit.independence_tests import RIT, RCoT

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

# Cannot find reference 'xxx' in '__init__.pyi | __init__.pyi | __init__.pxd' is a bug in pycharm, please ignore
class KCI_UInd(object):
    """
    Python implementation of Kernel-based Conditional Independence (KCI) test. Unconditional version.
    The original Matlab implementation can be found in http://people.tuebingen.mpg.de/kzhang/KCI-test.zip

    References
    ----------
    [1] K. Zhang, J. Peters, D. Janzing, and B. Schölkopf,
    "A kernel-based conditional independence test and application in causal discovery," In UAI 2011.
    [2] A. Gretton, K. Fukumizu, C.-H. Teo, L. Song, B. Schölkopf, and A. Smola, "A kernel
       Statistical test of independence." In NIPS 21, 2007.
    """
    def __init__(self, kernelX='Gaussian', kernelY='Gaussian', null_ss=1000, approx=True, est_width='empirical',
                 polyd=2, kwidthx=None, kwidthy=None):
        """
        Construct the KCI_UInd model.

        Parameters
        ----------
        kernelX: kernel function for input data x
            'Gaussian': Gaussian kernel
            'Polynomial': Polynomial kernel
            'Linear': Linear kernel
        kernelY: kernel function for input data y
        est_width: set kernel width for Gaussian kernels
            'empirical': set kernel width using empirical rules
            'median': set kernel width using the median trick
            'manual': set by users
        null_ss: sample size in simulating the null distribution
        approx: whether to use gamma approximation (default=True)
        polyd: polynomial kernel degrees (default=1)
        kwidthx: kernel width for data x (standard deviation sigma)
        kwidthy: kernel width for data y (standard deviation sigma)
        """

        self.kernelX = kernelX
        self.kernelY = kernelY
        self.est_width = est_width
        self.polyd = polyd
        self.kwidthx = kwidthx
        self.kwidthy = kwidthy
        self.nullss = null_ss
        self.thresh = 1e-6
        self.approx = approx

    def compute_pvalue(self, data_x=None, data_y=None):
        """
        Main function: compute the p value and return it together with the test statistic

        Parameters
        ----------
        data_x: input data for x (nxd1 array)
        data_y: input data for y (nxd2 array)

        Returns
        _________
        pvalue: p value (scalar)
        test_stat: test statistic (scalar)

        [Notes for speedup optimization]
            Kx, Ky are both symmetric with diagonals equal to 1 (no matter what the kernel is)
            Kxc, Kyc are both symmetric
        """

        # print(f"data shape: {data_x.shape}, {data_y.shape}.")

        # checkpoint()
        
        Kx, Ky = self.kernel_matrix(data_x, data_y)             ################ This is the key step for federated learning  Kx: n*n = z(X) z(X). 
        test_stat, Kxc, Kyc = self.HSIC_V_statistic(Kx, Ky)  # Kxc=H Kx H = H z(X) z(X) H = z'(X) z'(X) 

        # print("test 101: {},{},{}".format(np.trace(Kx), np.trace(Kx.T.dot(Kx)),np.trace(Kx)**2))
        # print(f"Test 102: kernel shape {Kx.shape},{Ky.shape}. {Kx.shape[0]}.  Kc shape: {Kxc.shape},{Kyc.shape}.")

        if self.approx:
            k_appr, theta_appr = self.get_kappa(Kxc, Kyc)
            pvalue = 1 - stats.gamma.cdf(test_stat, k_appr, 0, theta_appr)
        else:
            null_dstr = self.null_sample_spectral(Kxc, Kyc)
            pvalue = sum(null_dstr.squeeze() > test_stat) / float(self.nullss)
        return pvalue, test_stat

    def compute_pvalue_rf(self, data_x=None, data_y=None):
        # print(data_x.shape, data_y.shape)
        rit = RIT(data_x, data_y, approx="gamma", seed=42)
        sta = float(rit['Sta'])
        pval = float(rit['p'])
        return pval, sta


    def kernel_matrix(self, data_x, data_y):
        """
        Compute kernel matrix for data x and data y

        Parameters
        ----------
        data_x: input data for x (nxd1 array)
        data_y: input data for y (nxd2 array)

        Returns
        _________
        Kx: kernel matrix for data_x (nxn)
        Ky: kernel matrix for data_y (nxn)
        """
        if self.kernelX == 'Gaussian':
            if self.est_width == 'manual':
                if self.kwidthx is not None:
                    kernelX = GaussianKernel(self.kwidthx)
                else:
                    raise Exception('specify kwidthx')
            else:
                kernelX = GaussianKernel()
                if self.est_width == 'median':
                    kernelX.set_width_median(data_x)
                elif self.est_width == 'empirical':
                    # print("Test I am in Ucondi empiricalX.")
                    kernelX.set_width_empirical_hsic(data_x)
                else:
                    raise Exception('Undefined kernel width estimation method')
        elif self.kernelX == 'Polynomial':
            kernelX = PolynomialKernel(self.polyd)
        elif self.kernelX == 'Linear':
            kernelX = LinearKernel()
        else:
            raise Exception('Undefined kernel function')

        if self.kernelY == 'Gaussian':
            if self.est_width == 'manual':
                if self.kwidthy is not None:
                    kernelY = GaussianKernel(self.kwidthy)
                else:
                    raise Exception('specify kwidthy')
            else:
                kernelY = GaussianKernel()
                if self.est_width == 'median':
                    kernelY.set_width_median(data_y)
                elif self.est_width == 'empirical':
                    kernelY.set_width_empirical_hsic(data_y)
                else:
                    raise Exception('Undefined kernel width estimation method')
        elif self.kernelY == 'Polynomial':
            kernelY = PolynomialKernel(self.polyd)
        elif self.kernelY == 'Linear':
            kernelY = LinearKernel()
        else:
            raise Exception('Undefined kernel function')

        data_x = stats.zscore(data_x, ddof=1, axis=0)
        data_x[np.isnan(data_x)] = 0.   # in case some dim of data_x is constant
        data_y = stats.zscore(data_y, ddof=1, axis=0)
        data_y[np.isnan(data_y)] = 0.
        # We set 'ddof=1' to conform to the normalization way in the original Matlab implementation in
        # http://people.tuebingen.mpg.de/kzhang/KCI-test.zip

        # print(f"Shape X: {data_x.shape}; Shape Y: {data_y.shape}.")

        Kx = kernelX.kernel(data_x)
        Ky = kernelY.kernel(data_y)

        # Kx = kernelX.kernel_appro(data_x)
        # Ky = kernelY.kernel_appro(data_y)

        return Kx, Ky

    def HSIC_V_statistic(self, Kx, Ky):
        """
        Compute V test statistic from kernel matrices Kx and Ky
        Parameters
        ----------
        Kx: kernel matrix for data_x (nxn)
        Ky: kernel matrix for data_y (nxn)

        Returns
        _________
        Vstat: HSIC v statistics
        Kxc: centralized kernel matrix for data_x (nxn)
        Kyc: centralized kernel matrix for data_y (nxn)
        """
        Kxc = Kernel.center_kernel_matrix(Kx)
        Kyc = Kernel.center_kernel_matrix(Ky)
        V_stat = np.sum(Kxc * Kyc)  # V_stat = np.trace( Kxc @ kyc)    | * element-wise product; @ dot product.  
        return V_stat, Kxc, Kyc

    def null_sample_spectral(self, Kxc, Kyc):
        """
        Simulate data from null distribution

        Parameters
        ----------
        Kxc: centralized kernel matrix for data_x (nxn)
        Kyc: centralized kernel matrix for data_y (nxn)

        Returns
        _________
        null_dstr: samples from the null distribution

        """
        T = Kxc.shape[0]
        if T > 1000:
            num_eig = np.int(np.floor(T / 2))
        else:
            num_eig = T
        lambdax = eigvalsh(Kxc)
        lambday = eigvalsh(Kyc)
        lambdax = -np.sort(-lambdax)
        lambday = -np.sort(-lambday)
        lambdax = lambdax[0:num_eig]
        lambday = lambday[0:num_eig]
        lambda_prod = np.dot(lambdax.reshape(num_eig, 1), lambday.reshape(1, num_eig)).reshape(
            (num_eig ** 2, 1))
        lambda_prod = lambda_prod[lambda_prod > lambda_prod.max() * self.thresh]
        f_rand = np.random.chisquare(1, (lambda_prod.shape[0], self.nullss))
        null_dstr = lambda_prod.T.dot(f_rand) / T
        return null_dstr

    def get_kappa(self, Kx, Ky):
        """
        Get parameters for the approximated gamma distribution
        Parameters
        ----------
        Kx: kernel matrix for data_x (nxn)
        Ky: kernel matrix for data_y (nxn)

        Returns
        _________
        k_appr, theta_appr: approximated parameters of the gamma distribution

        [Updated @Haoyue 06/24/2022]
        equivalent to:
            var_appr = 2 * np.trace(Kx.dot(Kx)) * np.trace(Ky.dot(Ky)) / T / T
        based on the fact that:
            np.trace(K.dot(K)) == np.sum(K * K.T), where here K is symmetric
        we can save time on the dot product by only considering the diagonal entries of K.dot(K)
        time complexity is reduced from O(n^3) (matrix dot) to O(n^2) (traverse each element),
        where n is usually big (sample size).
        """
        T = Kx.shape[0]

        # print("test 101: {}".format(Kx[0,:] - Kx[:,0]))
        # print("test 101: {},{},{}".format(np.trace(Kx), np.trace(Kx.T.dot(Kx)),np.trace(Kx)**2))

        mean_appr = np.trace(Kx) * np.trace(Ky) / T
        var_appr = 2 * np.sum(Kx ** 2) * np.sum(Ky ** 2) / T / T  # same as np.sum(Kx * Kx) ..., here Kx is symmetric | * element-wise product; @ dot product.
        
        '''
        updated mean and var calculation
        '''
        # tmp = np.trace(Kx) * np.trace(Ky)
        # mean_appr = tmp / T
        # var_appr = 2 * (tmp**2) / T / T # same as np.sum(Kx * Kx.T) ..., here Kx is symmetric
        # print("test 101: {},{}".format(var_appr1, var_appr))

        k_appr = mean_appr ** 2 / var_appr
        theta_appr = var_appr / mean_appr
        return k_appr, theta_appr

class KCI_CInd(object):
    """
    Python implementation of Kernel-based Conditional Independence (KCI) test. Conditional version.
    The original Matlab implementation can be found in http://people.tuebingen.mpg.de/kzhang/KCI-test.zip

    References
    ----------
    [1] K. Zhang, J. Peters, D. Janzing, and B. Schölkopf, "A kernel-based conditional independence test and application in causal discovery," In UAI 2011.
    """

    def __init__(self, kernelX='Gaussian', kernelY='Gaussian', kernelZ='Gaussian', nullss=5000, est_width='empirical',
                 use_gp=False, approx=True, polyd=2, kwidthx=None, kwidthy=None, kwidthz=None):
        """
        Construct the KCI_CInd model.
        Parameters
        ----------
        kernelX: kernel function for input data x
            'Gaussian': Gaussian kernel
            'Polynomial': Polynomial kernel
            'Linear': Linear kernel
        kernelY: kernel function for input data y
        kernelZ: kernel function for input data z (conditional variable)
        est_width: set kernel width for Gaussian kernels
            'empirical': set kernel width using empirical rules
            'median': set kernel width using the median trick
            'manual': set by users
        null_ss: sample size in simulating the null distribution
        use_gp: whether use gaussian process to determine kernel width for z
        approx: whether to use gamma approximation (default=True)
        polyd: polynomial kernel degrees (default=1)
        kwidthx: kernel width for data x (standard deviation sigma, default None)
        kwidthy: kernel width for data y (standard deviation sigma)
        kwidthz: kernel width for data z (standard deviation sigma)
        """
        self.kernelX = kernelX
        self.kernelY = kernelY
        self.kernelZ = kernelZ
        self.est_width = est_width
        self.polyd = polyd
        self.kwidthx = kwidthx
        self.kwidthy = kwidthy
        self.kwidthz = kwidthz
        self.nullss = nullss
        self.epsilon_x = 1e-3   # To conform to the original Matlab implementation.
        self.epsilon_y = 1e-3
        self.use_gp = use_gp
        self.thresh = 1e-5
        self.approx = approx


    def compute_pvalue_rf(self, data_x=None, data_y=None, data_z=None):
        # print(data_x.shape, data_y.shape)
        rit = RCoT(data_x, data_y, data_z, num_f2=25, approx="gamma", seed=42)
        sta = float(rit['Sta'])
        pval = float(rit['p'])
        return pval, sta

    def compute_pvalue(self, data_x=None, data_y=None, data_z=None):
        """
        Main function: compute the p value and return it together with the test statistic
        Parameters
        ----------
        data_x: input data for x (nxd1 array)
        data_y: input data for y (nxd2 array)
        data_z: input data for z (nxd3 array)

        Returns
        _________
        pvalue: p value
        test_stat: test statistic
        """
        Kx, Ky, Kzx, Kzy = self.kernel_matrix(data_x, data_y, data_z)           ################ This is the key step for federated learning
        test_stat, KxR, KyR = self.KCI_V_statistic(Kx, Ky, Kzx, Kzy)
        uu_prod, size_u = self.get_uuprod(KxR, KyR)
        if self.approx:
            k_appr, theta_appr = self.get_kappa(uu_prod)
            pvalue = 1 - stats.gamma.cdf(test_stat, k_appr, 0, theta_appr)
        else:
            null_samples = self.null_sample_spectral(uu_prod, size_u, Kx.shape[0])
            pvalue = sum(null_samples > test_stat) / float(self.nullss)
        return pvalue, test_stat

    def kernel_matrix(self, data_x, data_y, data_z):
        """
        Compute kernel matrix for data x, data y, and data_z
        Parameters
        ----------
        data_x: input data for x (nxd1 array)
        data_y: input data for y (nxd2 array)
        data_z: input data for z (nxd3 array)

        Returns
        _________
        Kx: kernel matrix for data_x (nxn)
        Ky: kernel matrix for data_y (nxn)
        Kzx: centering kernel matrix for data_x (nxn)
        kzy: centering kernel matrix for data_y (nxn)
        """
        # normalize the data
        data_x = stats.zscore(data_x, ddof=1, axis=0)
        data_x[np.isnan(data_x)] = 0.
        
        data_y = stats.zscore(data_y, ddof=1, axis=0)
        data_y[np.isnan(data_y)] = 0.
        
        data_z = stats.zscore(data_z, ddof=1, axis=0)
        data_z[np.isnan(data_z)] = 0.
        # We set 'ddof=1' to conform to the normalization way in the original Matlab implementation in
        # http://people.tuebingen.mpg.de/kzhang/KCI-test.zip

        # concatenate x and z
        data_x = np.concatenate((data_x, 0.5 * data_z), axis=1)
        if self.kernelX == 'Gaussian':
            if self.est_width == 'manual':
                if self.kwidthx is not None:
                    kernelX = GaussianKernel(self.kwidthx)
                else:
                    raise Exception('specify kwidthx')
            else:
                kernelX = GaussianKernel()
                if self.est_width == 'median':
                    kernelX.set_width_median(data_x)
                elif self.est_width == 'empirical':
                    # kernelX's empirical width is determined by data_z's shape, please refer to the original code
                    # (http://people.tuebingen.mpg.de/kzhang/KCI-test.zip) in the file
                    # 'algorithms/CInd_test_new_withGP.m', Line 37 to 52.
                    kernelX.set_width_empirical_kci(data_z)
                else:
                    raise Exception('Undefined kernel width estimation method')
        elif self.kernelX == 'Polynomial':
            kernelX = PolynomialKernel(self.polyd)
        elif self.kernelX == 'Linear':
            kernelX = LinearKernel()
        else:
            raise Exception('Undefined kernel function')

        if self.kernelY == 'Gaussian':
            if self.est_width == 'manual':
                if self.kwidthy is not None:
                    kernelY = GaussianKernel(self.kwidthy)
                else:
                    raise Exception('specify kwidthy')
            else:
                kernelY = GaussianKernel()
                if self.est_width == 'median':
                    kernelY.set_width_median(data_y)
                elif self.est_width == 'empirical':
                    # kernelY's empirical width is determined by data_z's shape, please refer to the original code
                    # (http://people.tuebingen.mpg.de/kzhang/KCI-test.zip) in the file
                    # 'algorithms/CInd_test_new_withGP.m', Line 37 to 52.
                    kernelY.set_width_empirical_kci(data_z)
                else:
                    raise Exception('Undefined kernel width estimation method')
        elif self.kernelY == 'Polynomial':
            kernelY = PolynomialKernel(self.polyd)
        elif self.kernelY == 'Linear':
            kernelY = LinearKernel()
        else:
            raise Exception('Undefined kernel function')

        Kx = kernelX.kernel(data_x)
        Ky = kernelY.kernel(data_y)

        # centering kernel matrix
        Kx = Kernel.center_kernel_matrix(Kx)
        Ky = Kernel.center_kernel_matrix(Ky)

        if self.kernelZ == 'Gaussian':
            if not self.use_gp:
                if self.est_width == 'manual':
                    if self.kwidthz is not None:                       
                        kernelZ = GaussianKernel(self.kwidthz)
                    else:
                        raise Exception('specify kwidthz')
                else:
                    kernelZ = GaussianKernel()
                    if self.est_width == 'median':
                        kernelZ.set_width_median(data_z)
                    elif self.est_width == 'empirical':
                        kernelZ.set_width_empirical_kci(data_z)
                Kzx = kernelZ.kernel(data_z)
                Kzx = Kernel.center_kernel_matrix(Kzx)
                # centering kernel matrix to conform with the original Matlab implementation,
                # specifically, Line 100 in the file 'algorithms/CInd_test_new_withGP.m'
                Kzy = Kzx
            else:
                # learning the kernel width of Kz using Gaussian process
                n, Dz = data_z.shape
                if self.kernelX == 'Gaussian':
                    widthz = sqrt(1.0 / (kernelX.width * data_x.shape[1]))
                else:
                    widthz = 1.0
                # Instantiate a Gaussian Process model for x
                wx, vx = eigh(0.5 * (Kx + Kx.T))
                topkx = int(np.min((400, np.floor(n / 4))))
                idx = np.argsort(-wx)
                wx = wx[idx]
                vx = vx[:, idx]
                wx = wx[0:topkx]
                vx = vx[:, 0:topkx]
                vx = vx[:, wx > wx.max() * self.thresh]
                wx = wx[wx > wx.max() * self.thresh]
                vx = 2 * sqrt(n) * vx.dot(np.diag(np.sqrt(wx))) / sqrt(wx[0])
                kernelx = C(1.0, (1e-3, 1e3)) * RBF(widthz * np.ones(Dz), (1e-2, 1e2)) + WhiteKernel(0.1, (1e-10, 1e+1))
                gpx = GaussianProcessRegressor(kernel=kernelx)
                # fit Gaussian process, including hyperparameter optimization
                gpx.fit(data_z, vx)

                # construct Gaussian kernels according to learned hyperparameters
                Kzx = gpx.kernel_.k1(data_z, data_z)
                self.epsilon_x = np.exp(gpx.kernel_.theta[-1])

                # Instantiate a Gaussian Process model for y
                wy, vy = eigh(0.5 * (Ky + Ky.T))
                topky = int(np.min((400, np.floor(n / 4))))
                idy = np.argsort(-wy)
                wy = wy[idy]
                vy = vy[:, idy]
                wy = wy[0:topky]
                vy = vy[:, 0:topky]
                vy = vy[:, wy > wy.max() * self.thresh]
                wy = wy[wy > wy.max() * self.thresh]
                vy = 2 * sqrt(n) * vy.dot(np.diag(np.sqrt(wy))) / sqrt(wy[0])
                kernely = C(1.0, (1e-3, 1e3)) * RBF(widthz * np.ones(Dz), (1e-2, 1e2)) + WhiteKernel(0.1, (1e-10, 1e+1))
                gpy = GaussianProcessRegressor(kernel=kernely)
                # fit Gaussian process, including hyperparameter optimization
                gpy.fit(data_z, vy)

                # construct Gaussian kernels according to learned hyperparameters
                Kzy = gpy.kernel_.k1(data_z, data_z)
                self.epsilon_y = np.exp(gpy.kernel_.theta[-1])
        elif self.kernelZ == 'Polynomial':
            kernelZ = PolynomialKernel(self.polyd)
            Kzx = kernelZ.kernel(data_z)
            Kzx = Kernel.center_kernel_matrix(Kzx)
            Kzy = Kzx
        elif self.kernelZ == 'Linear':
            kernelZ = LinearKernel()
            Kzx = kernelZ.kernel(data_z)
            Kzx = Kernel.center_kernel_matrix(Kzx)
            Kzy = Kzx
        else:
            raise Exception('Undefined kernel function')
        return Kx, Ky, Kzx, Kzy

    def KCI_V_statistic(self, Kx, Ky, Kzx, Kzy):
        """
        Compute V test statistic from kernel matrices Kx and Ky
        Parameters
        ----------
        Kx: kernel matrix for data_x (nxn)
        Ky: kernel matrix for data_y (nxn)
        Kzx: centering kernel matrix for data_x (nxn)
        kzy: centering kernel matrix for data_y (nxn)

        Returns
        _________
        Vstat: KCI v statistics
        KxR: centralized kernel matrix for data_x (nxn)
        KyR: centralized kernel matrix for data_y (nxn)

        [Updated @Haoyue 06/24/2022]
        1. Kx, Ky, Kzx, Kzy are all symmetric matrices.
            - * Kx's diagonal elements are not the same, because the kernel Kx is centered.
              * Before centering, Kx's all diagonal elements are 1 (because of exp(-0.5 * sq_dists * self.width)).
              * The same applies to Ky.
            - * If (self.kernelZ == 'Gaussian' and self.use_gp), then Kzx has all the same diagonal elements (not necessarily 1).
              * The same applies to Kzy.
        2. If not (self.kernelZ == 'Gaussian' and self.use_gp): assert (Kzx == Kzy).all()
           With this we could save one repeated calculation of pinv(Kzy+\epsilonI), which consumes most time.
        """
        KxR, Rzx = Kernel.center_kernel_matrix_regression(Kx, Kzx, self.epsilon_x)
        if self.epsilon_x != self.epsilon_y or (self.kernelZ == 'Gaussian' and self.use_gp):
            KyR, _ = Kernel.center_kernel_matrix_regression(Ky, Kzy, self.epsilon_y)
        else:
            # assert np.all(Kzx == Kzy), 'Kzx and Kzy are the same'
            KyR = Rzx.dot(Ky.dot(Rzx))
        Vstat = np.sum(KxR * KyR)
        return Vstat, KxR, KyR

    def get_uuprod(self, Kx, Ky):
        """
        Compute eigenvalues for null distribution estimation

        Parameters
        ----------
        Kx: centralized kernel matrix for data_x (nxn)
        Ky: centralized kernel matrix for data_y (nxn)

        Returns
        _________
        uu_prod: product of the eigenvectors of Kx and Ky
        size_u: number of producted eigenvectors

        """
        wx, vx = eigh(0.5 * (Kx + Kx.T))
        wy, vy = eigh(0.5 * (Ky + Ky.T))
        idx = np.argsort(-wx)
        idy = np.argsort(-wy)
        wx = wx[idx]
        vx = vx[:, idx]
        wy = wy[idy]
        vy = vy[:, idy]
        vx = vx[:, wx > np.max(wx) * self.thresh]
        wx = wx[wx > np.max(wx) * self.thresh]
        vy = vy[:, wy > np.max(wy) * self.thresh]
        wy = wy[wy > np.max(wy) * self.thresh]
        vx = vx.dot(np.diag(np.sqrt(wx)))
        vy = vy.dot(np.diag(np.sqrt(wy)))

        # calculate their product
        T = Kx.shape[0]
        num_eigx = vx.shape[1]
        num_eigy = vy.shape[1]
        size_u = num_eigx * num_eigy
        uu = np.zeros((T, size_u))
        for i in range(0, num_eigx):
            for j in range(0, num_eigy):
                uu[:, i * num_eigy + j] = vx[:, i] * vy[:, j]

        if size_u > T:
            uu_prod = uu.dot(uu.T)
        else:
            uu_prod = uu.T.dot(uu)

        return uu_prod, size_u

    def null_sample_spectral(self, uu_prod, size_u, T):
        """
        Simulate data from null distribution

        Parameters
        ----------
        uu_prod: product of the eigenvectors of Kx and Ky
        size_u: number of producted eigenvectors
        T: sample size

        Returns
        _________
        null_dstr: samples from the null distribution

        """
        eig_uu = eigvalsh(uu_prod)
        eig_uu = -np.sort(-eig_uu)
        eig_uu = eig_uu[0:np.min((T, size_u))]
        eig_uu = eig_uu[eig_uu > np.max(eig_uu) * self.thresh]

        f_rand = np.random.chisquare(1, (eig_uu.shape[0], self.nullss))
        null_dstr = eig_uu.T.dot(f_rand)
        return null_dstr

    def get_kappa(self, uu_prod):
        """
        Get parameters for the approximated gamma distribution
        Parameters
        ----------
        uu_prod: product of the eigenvectors of Kx and Ky

        Returns
        ----------
        k_appr, theta_appr: approximated parameters of the gamma distribution

        """
        mean_appr = np.trace(uu_prod)
        var_appr = 2 * np.trace(uu_prod.dot(uu_prod))
        k_appr = mean_appr ** 2 / var_appr
        theta_appr = var_appr / mean_appr
        return k_appr, theta_appr



# There is no conditional set
class LGM_UInd(object):
    def __init__(self, kernelX='Gaussian', kernelY='Gaussian'):
        self.kernelX = kernelX
        self.kernelY = kernelY
    
    # data_x is the domain index; data_y is the observed variable.
    def compute_pval(self, data_x=None, data_y=None, K=0):
        # print(f"GMM UI is on!!!!!!! K={K}. x shape: {data_x.shape}. y shape: {data_y.shape}. {type(data_x)}")
        p_value = 1 # independent

        s1, s2 = data_y.shape 
        fed_data = data_y.reshape(K, int(s1/K), s2) 
        

        """
            Using Linear Gaussian Model to get an approximate distribution. 
        """
        res_dist = [] # shape: [K clients, 2 parameters]
        for k in range(K):
            mu, sigma = fed_data[k].mean(), fed_data[k].var()
            res_dist.append([mu, sigma])
        res_dist = np.array(res_dist)


        """
            Using Monte Carlo Simulation to recover the samples from Linear Gaussian Model. 
        """
        recovered_samples = [] # shape: [K clients, 2 parameters]
        for k in range(K):
            N = 200
            mu, sigma = res_dist[k]
            samples = np.random.normal(mu, sigma, N)
            # random.shuffle(samples)
            recovered_samples.append(samples)
        recovered_samples = np.array(recovered_samples, dtype=np.float64)
        # print("Recovered shape: ", recovered_samples.shape)


        # t_3 = time.time()
        # print(f"Time for getting recovered samples: {t_3-t_2}s.")
        # import matplotlib.pyplot as plt
        # import pandas as pd 
        # import seaborn as sns
        # plt.rcParams['figure.figsize'] = [20, 10]
        # ax = plt.figure()
        # for k in range(8):
        #     ax.add_subplot(2,4,k+1)
        #     sns.distplot(fed_data[k,:,:].flatten(), hist=False, kde=True, label="Original")
        #     sns.distplot(recovered_samples[k].flatten(), hist=False, kde=True, label="Recovered")
        # # plt.title('$X_0$')
        # plt.legend(fontsize=12, loc=1)
        # plt.savefig("result/expRes/x1.pdf")


        """
            Do Kolmogorov-Smirnov Test (KS Test) to determine whether two distributions are equal or not, according to their samples.
            Two key parameters: alpha=1e-4, ratio=0.4. 
        """
        rand_list = np.array(range(K))
        random.shuffle(rand_list)
        pval_list = []
        pval_rec_list = []
        for k in range(0,K,2):
            """
                Distribution Matching Test for all pairs.
            """
            i, j = rand_list[k], rand_list[k+1]
            res1 = stats.ks_2samp(fed_data[i].flatten(), fed_data[j].flatten())
            res2 = stats.ks_2samp(recovered_samples[i].flatten(), recovered_samples[j].flatten())
            print(f"GroundTrue: i-{i},j-{j}:", res1)
            print(f"Recovered: i-{i},j-{j}:", res2)
            pval_list.append(res1.pvalue)
            pval_rec_list.append(res2.pvalue)

        count = np.sum([i<=0.05 for i in pval_list])
        count_rec = np.sum([i<=1e-4 for i in pval_rec_list])
        print(f"Truth count: ", count)
        print(f"Recover count: ", count_rec)

        p_value = 0 if count/len(pval_list)>0.3 else 1
        print(f"P-val: {p_value}.")
        p_value_rec = 0 if count_rec/len(pval_rec_list)>0.4 else 1
        print(f"P-rec-val: {p_value_rec}.")


        # t_4 = time.time()
        # print(f"########### Time for KS test: {t_4-t_3}s.")

        return p_value_rec


# conditional set length =1
class LGM_CInd_1(object):
    def __init__(self, kernelX='Gaussian', kernelY='Gaussian'):
        self.kernelX = kernelX
        self.kernelY = kernelY
    
    def compute_pval(self, data_x=None, data_y=None, data_z=None, K=0):
        print("Linear Gaussian Model and Conditional Independent Test with |S|=1.")

        s1, s2 = data_y.shape
        fed_data_y = data_y.reshape(K, int(s1/K), s2)
        s1, s2 = data_z.shape
        fed_data_z = data_z.reshape(K, int(s1/K), s2)

        """
            Using Linear Gaussian Model to get approximate distributions. 
        """
        res_para = []
        for k in range(K):
            dy, dz = fed_data_y[k], fed_data_z[k]
            dy, dz = np.squeeze(dy), np.squeeze(dz) # [n,1] -> [n]
            cov_mat = np.cov(dy, dz, bias=True)
            inv_cov = inv(cov_mat)
            dy_mu, dz_mu = dy.mean(), dz.mean()

            # according to the formulations.
            coef = inv_cov[0][1] / inv_cov[1][1]
            mu = dy_mu - coef * dz_mu 
            sigma = inv_cov[0][0] - inv_cov[0][1] / inv_cov[1][1] * inv_cov[1][0]
            res_para.append([coef, mu, sigma])

        
        """
            Using Monte Carlo Simulation to recover the samples from Gaussian model.
            We skip the coef term as an approximation. 
        """
        recovered_samples = [] # shape: [K clients, N samples]
        for k in range(K):
            N = 200
            coef, mu, sigma = res_para[k]
            samples = np.random.normal(mu, sigma, N)
            recovered_samples.append(samples)
        recovered_samples = np.array(recovered_samples, dtype=np.float64)
        

        """
            Do Kolmogorov-Smirnov Test (KS Test) to determine whether two distributions are equal or not, according to their samples.
            Two key parameters: alpha=1e-4, ratio=0.4. 
        """
        pval_list = []
        rand_list = np.array(range(K))
        random.shuffle(rand_list)
        for k in range(0,K,2):
            """
                Distribution Matching Test for all pairs.
            """
            i, j = rand_list[k], rand_list[k+1]
            res = stats.ks_2samp(recovered_samples[i].flatten(), recovered_samples[j].flatten())
            pval_list.append(res.pvalue)

        count_rec = np.sum([i<=1e-4 for i in pval_list])
        print(f"Recovered count: ", count_rec)

        p_value_rec = 0 if count_rec/len(pval_list)>0.4 else 1
        print(f"P-val: {p_value_rec}.")

        return p_value_rec

# conditional set length >=2
class LGM_CInd_2(object):
    def __init__(self, kernelX='Gaussian', kernelY='Gaussian'):
        self.kernelX = kernelX
        self.kernelY = kernelY
    
    def compute_pval(self, data_x=None, data_y=None, data_z=None, K=0):
        print("Hi, Linear Gaussian Model. I am in Conditional Independent Test.")

        s1, s2 = data_y.shape
        fed_data_y = data_y.reshape(K, int(s1/K), s2)
        s1, s2 = data_z.shape
        fed_data_z = data_z.reshape(K, int(s1/K), s2)


        """
            Seperate y into y1,y2 according to the L2 norm values of z. 
            Using Gaussian Mixture Model to get approximate distributions for y1,y2. 
            We randomly choose 2 clients and make comparisons. 
        """
        res_dist_1 = []
        res_dist_2 = []
        rand_list = np.array(range(K))
        random.shuffle(rand_list)
        for k in range(0,K,2):
            i, j = rand_list[k], rand_list[k+1]

            dy1, dy2 = fed_data_y[k], fed_data_y[k+1]
            dz1, dz2 = fed_data_z[k], fed_data_z[k+1] 
            sep_data_1 = np.linalg.norm(dz1, axis=1, ord=2)
            sep_data_2 = np.linalg.norm(dz2, axis=1, ord=2)

            sep_num = math.floor((max(sep_data_1) + min(sep_data_1) + max(sep_data_2) + min(sep_data_2))/4)-0.5 # choose a seperation number 
            print(f"Sep_num: {sep_num}", sep_num)

            dy1 = np.squeeze(dy1) # [n,1] -> [n]
            dy2 = np.squeeze(dy2) # [n,1] -> [n]
            
            sep_idx_1_1 = np.where(sep_data_1 > sep_num)
            sep_idx_1_2 = np.where(sep_data_1 <= sep_num)
            dy_1_1 = dy1[sep_idx_1_1].reshape(-1,1)
            dy_1_2 = dy1[sep_idx_1_2].reshape(-1,1)
            print(f"dy1 shape: {dy_1_1.shape},{dy_1_2.shape}.")
            print(f"sep idx 1: {sep_idx_1_1}, sep idx 2: {sep_idx_1_2}.")
            print(f"length: {len(sep_idx_1_1)}, {len(sep_idx_1_2)}.")

            sep_idx_2_1 = np.where(sep_data_2 > sep_num)
            sep_idx_2_2 = np.where(sep_data_2 <= sep_num)
            dy_2_1 = dy2[sep_idx_2_1].reshape(-1,1)
            dy_2_2 = dy2[sep_idx_2_2].reshape(-1,1)
            print(f"dy2 shape: {dy_2_1.shape},{dy_2_2.shape}.")

            if dy_1_1.shape[0]<3:
                print("######## Zero happens.")
                dy_1_1 = dy1
            if dy_1_2.shape[0]<3:
                print("######## Zero happens.")
                dy_1_2 = dy1
            if dy_2_1.shape[0]<3:
                print("######## Zero happens.")
                dy_2_1 = dy2
            if dy_2_2.shape[0]<3:
                print("######## Zero happens.")
                dy_2_2 = dy2 

            mu, sigma, pi = EM_single(dy_1_1, 2, 1e-8)
            res_dist_1.append([mu, sigma, pi])
            mu, sigma, pi = EM_single(dy_2_1, 2, 1e-8)
            res_dist_1.append([mu, sigma, pi])

            mu, sigma, pi = EM_single(dy_1_2, 2, 1e-8)
            res_dist_2.append([mu, sigma, pi])
            mu, sigma, pi = EM_single(dy_2_2, 2, 1e-8)
            res_dist_2.append([mu, sigma, pi])

        
        """
            Using Monte Carlo Simulation to recover the samples from GMM. 
        """
        recovered_samples_1 = [] # shape: [K clients, 3 parameters, 2 clusters]
        recovered_samples_2 = [] 
        for k in range(K):
            N = 200
            mus, sigmas, ps = res_dist_1[k]
            samples = np.hstack([np.random.normal(mus[0], sigmas[0], int(ps[0]*N)), np.random.normal(mus[1], sigmas[1], max(int(ps[1]*N), N-int(ps[0]*N)) )])
            random.shuffle(samples)
            recovered_samples_1.append(samples)

            mus, sigmas, ps = res_dist_2[k]
            samples = np.hstack([np.random.normal(mus[0], sigmas[0], int(ps[0]*N)), np.random.normal(mus[1], sigmas[1], max(int(ps[1]*N), N-int(ps[0]*N)) )])
            random.shuffle(samples)
            recovered_samples_2.append(samples)
        recovered_samples_1 = np.array(recovered_samples_1, dtype=np.float64)
        recovered_samples_2 = np.array(recovered_samples_2, dtype=np.float64)
        

        """
            Do Kolmogorov-Smirnov Test (KS Test) to determine whether two distributions are equal or not, according to their samples.
            Two key parameters: alpha=1e-4, ratio=0.4. 
        """
        pval1_list = []
        pval2_list = []
        for k in range(0,K,2):
            """
                Distribution Matching Test for all pairs.
            """
            # i, j = rand_list[k], rand_list[k+1]
            i, j = k, k+1
            res1 = stats.ks_2samp(recovered_samples_1[i].flatten(), recovered_samples_1[j].flatten())
            res2 = stats.ks_2samp(recovered_samples_2[i].flatten(), recovered_samples_2[j].flatten())
            print(f"Recovered pvalue1 and pvalue2: i-{i},j-{j}:", res1.pvalue, res2.pvalue)
            pval1_list.append(res1.pvalue)
            pval2_list.append(res2.pvalue)

        count_rec_1 = np.sum([i<=1e-4 for i in pval1_list])
        count_rec_2 = np.sum([i<=1e-4 for i in pval2_list])
        print(f"Recovered count: ", count_rec_1, count_rec_2)

        p_value_rec_1 = 0 if count_rec_1/len(pval1_list)>0.4 else 1
        p_value_rec_2 = 0 if count_rec_2/len(pval2_list)>0.4 else 1
        print(f"P-rec-val: {p_value_rec_1, p_value_rec_2}.")

        return p_value_rec_1 * p_value_rec_2




class GMM_UInd(object):
    def __init__(self, kernelX='Gaussian', kernelY='Gaussian'):
        self.kernelX = kernelX
        self.kernelY = kernelY
    
    def compute_pval(self, data_x=None, data_y=None, K=0):
        # print(f"GMM UI is on!!!!!!! K={K}. x shape: {data_x.shape}. y shape: {data_y.shape}. {type(data_x)}")
        p_value = 1 # independent

        s1, s2 = data_y.shape 
        fed_data = data_y.reshape(K, int(s1/K), s2) 
        
        # data_joint = np.concatenate((data_y, data_y+0.5), axis=1)
        # s1, s2 = data_joint.shape 
        # fed_data_joint = data_joint.reshape(K, int(s1/K), s2) 

        # t_1 = time.time()

        """
            Using Gaussian Mixture Model to get an approximate distribution. 
        """
        res_dist = [] # shape: [K clients, 3 parameters, 2 clusters]
        for k in range(K):
            mu, sigma, pi = EM_single(fed_data[k], 2, 1e-8)
            # mu_j, sigma_j, pi_j = EM_multi(fed_data_joint[k], 3, 1e-8, mode='diag')
            res_dist.append([mu, sigma, pi])
        res_dist = np.array(res_dist)
        # print("Paras shape: ", res_dist.shape)

        # t_2 = time.time()
        # print(f"Time for getting GMM: {t_2-t_1}s.")

        """
            Using Monte Carlo Simulation to recover the samples from GMM. 
        """
        recovered_samples = [] # shape: [K clients, 3 parameters, 2 clusters]
        for k in range(K):
            
            N = 200
            mus, sigmas, ps = res_dist[k]
            samples = np.hstack([np.random.normal(mus[0], sigmas[0], int(ps[0]*N)), np.random.normal(mus[1], sigmas[1], max(int(ps[1]*N), N-int(ps[0]*N)) )])
            random.shuffle(samples)
            recovered_samples.append(samples)
        recovered_samples = np.array(recovered_samples, dtype=np.float64)
        # print("Recovered shape: ", recovered_samples.shape)


        # t_3 = time.time()
        # print(f"Time for getting recovered samples: {t_3-t_2}s.")


        # import matplotlib.pyplot as plt
        # import pandas as pd 
        # import seaborn as sns
        # plt.rcParams['figure.figsize'] = [20, 10]
        # ax = plt.figure()
        # for k in range(8):
        #     ax.add_subplot(2,4,k+1)
        #     sns.distplot(fed_data[k,:,:].flatten(), hist=False, kde=True, label="Original")
        #     sns.distplot(recovered_samples[k].flatten(), hist=False, kde=True, label="Recovered")
        # # plt.title('$X_0$')
        # plt.legend(fontsize=12, loc=1)
        # plt.savefig("result/expRes/x1.pdf")


        """
            Do Kolmogorov-Smirnov Test (KS Test) to determine whether two distributions are equal or not, according to their samples.
            Two key parameters: alpha=1e-4, ratio=0.4. 
        """
        rand_list = np.array(range(K))
        random.shuffle(rand_list)
        pval_list = []
        pval_rec_list = []
        for k in range(0,K,2):
            """
                Distribution Matching Test for all pairs.
            """
            i, j = rand_list[k], rand_list[k+1]
            res1 = stats.ks_2samp(fed_data[i].flatten(), fed_data[j].flatten())
            res2 = stats.ks_2samp(recovered_samples[i].flatten(), recovered_samples[j].flatten())
            print(f"GroundTrue: i-{i},j-{j}:", res1)
            print(f"Recovered: i-{i},j-{j}:", res2)
            pval_list.append(res1.pvalue)
            pval_rec_list.append(res2.pvalue)

        count = np.sum([i<=0.05 for i in pval_list])
        count_rec = np.sum([i<=1e-4 for i in pval_rec_list])
        print(f"Truth count: ", count)
        print(f"Recover count: ", count_rec)

        p_value = 0 if count/len(pval_list)>0.3 else 1
        print(f"P-val: {p_value}.")
        p_value_rec = 0 if count_rec/len(pval_rec_list)>0.4 else 1
        print(f"P-rec-val: {p_value_rec}.")


        # t_4 = time.time()
        # print(f"########### Time for KS test: {t_4-t_3}s.")

        return p_value_rec


def EM_single(data, K, threshold):
    # data shape: [n,1] -> [n]
    data = np.squeeze(data)

    # Initialization 
    mu0 = np.ones(K) 
    for k in range(K):
        idx = np.int(random.random() * len(data))
        mu0[k] += data[idx]
    sigma0 = np.ones(K) * np.var(data)
    pi0 = np.ones(K) * 1.0 / K
    current_log_likelihood = log_likelihood_single(data, K, mu0, sigma0, pi0)

    # EM method.
    mu, sigma, pi = mu0, sigma0, pi0
    max_iter = 100
    for it in range(max_iter):
        resp = e_step_single(data, K, mu, sigma, pi)
        mu, sigma, pi = m_step_single(data, K, resp)

        new_log_likelihood = log_likelihood_single(data, K, mu, sigma, pi)
        if (abs(new_log_likelihood - current_log_likelihood) < threshold):
            # print("iters=%d" % (it))
            break
        current_log_likelihood = new_log_likelihood
        # print(f"Iters:{it}", current_log_likelihood, mu, sigma, pi)
    return mu, sigma, pi 

def e_step_single(data, K, mu, sigma, pi):
    idvs = len(data)
    resp = np.zeros((idvs, K))
    for i in range(idvs):
        for k in range(K):
            resp[i][k] = pi[k] * gaussian_single(data[i], mu[k], sigma[k]) / likelihood_single(data[i], K, mu, sigma, pi)
    return resp

def m_step_single(data, K, resp):
    idvs = len(data)
    mu = np.zeros(K)
    sigma = np.zeros(K)
    pi = np.zeros(K)
    marg_resp = np.zeros(K)
    for k in range(K):
        for i in range(idvs):
            marg_resp[k] += resp[i][k]
            mu[k] += (resp[i][k]) * data[i]
        mu[k] /= marg_resp[k] # update mu.
        for i in range(idvs):
            x_mu = data[i] - mu[k]
            sigma[k] += (resp[i][k] / marg_resp[k]) * x_mu * x_mu # update sigma.

        pi[k] = marg_resp[k] / idvs # update pi.
    return mu, sigma, pi

def log_likelihood_single(data, K, mu, sigma, pi):
    log_likelihood = 0.0
    for n in range(len(data)):
        log_likelihood += np.log(likelihood_single(data[n], K, mu, sigma, pi)) # log P(X) = log \prod P(X_i) = \sum log P(X_i)
    return log_likelihood

def likelihood_single(x, K, mu, sigma, pi):
    rs = 0.0
    for k in range(K):
        rs += pi[k] * gaussian_single(x, mu[k], sigma[k])
    return rs

def gaussian_single(x, mu, sigma):
    if sigma==0:
        sigma = 1e-6 

    norm_factor = 2 * np.pi * sigma
    norm_factor = 1.0 / np.sqrt(norm_factor)
    rs = norm_factor * np.exp(-0.5 * (x-mu)*(x-mu) / sigma)
    return rs




class GMM_CInd(object):
    def __init__(self, kernelX='Gaussian', kernelY='Gaussian'):
        self.kernelX = kernelX
        self.kernelY = kernelY
    
    def compute_pval(self, data_x=None, data_y=None, data_z=None, K=0):
        # s1, s2 = data_z.shape
        # dz = 1 if s2==1 else 0
            
        # data_joint = np.concatenate((data_y, data_z), axis=1)
        # s1, s2 = data_joint.shape 
        # fed_data_joint = data_joint.reshape(K, int(s1/K), s2) 

        # data_marginal = data_z
        # s1, s2 = data_marginal.shape
        # fed_data_marginal = data_marginal.reshape(K, int(s1/K), s2)
        print("Hi, I am in Conditional Independent Test.")

        s1, s2 = data_y.shape
        fed_data_y = data_y.reshape(K, int(s1/K), s2)
        s1, s2 = data_z.shape
        fed_data_z = data_z.reshape(K, int(s1/K), s2)


        """
            Seperate y into y1,y2 according to the L2 norm values of z. 
            Using Gaussian Mixture Model to get approximate distributions for y1,y2. 
            We randomly choose 2 clients and make comparisons. 
        """
        res_dist_1 = []
        res_dist_2 = []
        rand_list = np.array(range(K))
        random.shuffle(rand_list)
        for k in range(0,K,2):
            i, j = rand_list[k], rand_list[k+1]

            dy1, dy2 = fed_data_y[k], fed_data_y[k+1]
            dz1, dz2 = fed_data_z[k], fed_data_z[k+1] 
            sep_data_1 = np.linalg.norm(dz1, axis=1, ord=2)
            sep_data_2 = np.linalg.norm(dz2, axis=1, ord=2)

            sep_num = math.floor((max(sep_data_1) + min(sep_data_1) + max(sep_data_2) + min(sep_data_2))/4)-0.5 # choose a seperation number 
            print(f"Sep_num: {sep_num}", sep_num)

            dy1 = np.squeeze(dy1) # [n,1] -> [n]
            dy2 = np.squeeze(dy2) # [n,1] -> [n]
            
            sep_idx_1_1 = np.where(sep_data_1 > sep_num)
            sep_idx_1_2 = np.where(sep_data_1 <= sep_num)
            dy_1_1 = dy1[sep_idx_1_1].reshape(-1,1)
            dy_1_2 = dy1[sep_idx_1_2].reshape(-1,1)
            print(f"dy1 shape: {dy_1_1.shape},{dy_1_2.shape}.")
            print(f"sep idx 1: {sep_idx_1_1}, sep idx 2: {sep_idx_1_2}.")
            print(f"length: {len(sep_idx_1_1)}, {len(sep_idx_1_2)}.")

            sep_idx_2_1 = np.where(sep_data_2 > sep_num)
            sep_idx_2_2 = np.where(sep_data_2 <= sep_num)
            dy_2_1 = dy2[sep_idx_2_1].reshape(-1,1)
            dy_2_2 = dy2[sep_idx_2_2].reshape(-1,1)
            print(f"dy2 shape: {dy_2_1.shape},{dy_2_2.shape}.")

            if dy_1_1.shape[0]<3:
                print("######## Zero happens.")
                dy_1_1 = dy1
            if dy_1_2.shape[0]<3:
                print("######## Zero happens.")
                dy_1_2 = dy1
            if dy_2_1.shape[0]<3:
                print("######## Zero happens.")
                dy_2_1 = dy2
            if dy_2_2.shape[0]<3:
                print("######## Zero happens.")
                dy_2_2 = dy2 

            mu, sigma, pi = EM_single(dy_1_1, 2, 1e-8)
            res_dist_1.append([mu, sigma, pi])
            mu, sigma, pi = EM_single(dy_2_1, 2, 1e-8)
            res_dist_1.append([mu, sigma, pi])

            mu, sigma, pi = EM_single(dy_1_2, 2, 1e-8)
            res_dist_2.append([mu, sigma, pi])
            mu, sigma, pi = EM_single(dy_2_2, 2, 1e-8)
            res_dist_2.append([mu, sigma, pi])

        
        """
            Using Monte Carlo Simulation to recover the samples from GMM. 
        """
        recovered_samples_1 = [] # shape: [K clients, 3 parameters, 2 clusters]
        recovered_samples_2 = [] 
        for k in range(K):
            N = 200
            mus, sigmas, ps = res_dist_1[k]
            samples = np.hstack([np.random.normal(mus[0], sigmas[0], int(ps[0]*N)), np.random.normal(mus[1], sigmas[1], max(int(ps[1]*N), N-int(ps[0]*N)) )])
            random.shuffle(samples)
            recovered_samples_1.append(samples)

            mus, sigmas, ps = res_dist_2[k]
            samples = np.hstack([np.random.normal(mus[0], sigmas[0], int(ps[0]*N)), np.random.normal(mus[1], sigmas[1], max(int(ps[1]*N), N-int(ps[0]*N)) )])
            random.shuffle(samples)
            recovered_samples_2.append(samples)
        recovered_samples_1 = np.array(recovered_samples_1, dtype=np.float64)
        recovered_samples_2 = np.array(recovered_samples_2, dtype=np.float64)
        

        """
            Do Kolmogorov-Smirnov Test (KS Test) to determine whether two distributions are equal or not, according to their samples.
            Two key parameters: alpha=1e-4, ratio=0.4. 
        """
        pval1_list = []
        pval2_list = []
        for k in range(0,K,2):
            """
                Distribution Matching Test for all pairs.
            """
            # i, j = rand_list[k], rand_list[k+1]
            i, j = k, k+1
            res1 = stats.ks_2samp(recovered_samples_1[i].flatten(), recovered_samples_1[j].flatten())
            res2 = stats.ks_2samp(recovered_samples_2[i].flatten(), recovered_samples_2[j].flatten())
            print(f"Recovered pvalue1 and pvalue2: i-{i},j-{j}:", res1.pvalue, res2.pvalue)
            pval1_list.append(res1.pvalue)
            pval2_list.append(res2.pvalue)

        count_rec_1 = np.sum([i<=1e-4 for i in pval1_list])
        count_rec_2 = np.sum([i<=1e-4 for i in pval2_list])
        print(f"Recovered count: ", count_rec_1, count_rec_2)

        p_value_rec_1 = 0 if count_rec_1/len(pval1_list)>0.4 else 1
        p_value_rec_2 = 0 if count_rec_2/len(pval2_list)>0.4 else 1
        print(f"P-rec-val: {p_value_rec_1, p_value_rec_2}.")

        # for k in range(K):
        #     """
        #         Using Gaussian Mixture Model to get an approximate distribution, given dataX. 
        #     """
        #     mu_j, sigma_j, pi_j = EM_multi(fed_data_joint[k], 3, 1e-8, mode='diag')

        #     if dz==1:
        #         mu_m, sigma_m, pi_m = EM_single(fed_data_marginal[k], 3, 1e-8)
        #     else:
        #         mu_m, sigma_m, pi_m = EM_multi(fed_data_marginal[k], 3, 1e-8, mode='diag')
        return p_value_rec_1 * p_value_rec_2



def random_parameters(data, K,mode=None):
    cols = (data.shape)[1]
    mu = np.zeros((K, cols))
    for k in range(K):
        idx = np.int(random.random() * len(data))
        for col in range(cols):

            mu[k][col] += data[idx, col]

    sigma = np.zeros((K, cols, cols))
    for k in range(K):
        sigma[k]=np.cov(data.T)

    sigma1 = np.zeros((K, cols, cols))
    if mode == 'diag':
        for k in range(K):
            sigma1[k] = np.eye(cols) * sigma[k]
    elif mode == 'd+s':
        for k in range(K):
            sigma1[k] = np.eye(cols) * np.eys(cols) * sigma[k]
    else:
        sigma1 = sigma
    np.mean(np.array(sigma),axis=0)
    pi = np.ones(K) * 1.0 / K
    return mu, sigma1, pi


def e_step(data, K, mu, sigma, pi):
    idvs = (data.shape)[0]

    resp = np.zeros((idvs, K))

    for i in range(idvs):
        for k in range(K):
            resp[i][k] = pi[k] * gaussian(data[i], mu[k], sigma[k]) / likelihood(data[i], K, mu, sigma, pi)

    return resp


def log_likelihood(data, K, mu, sigma, pi):
    log_likelihood = 0.0
    for n in range(len(data)):
        log_likelihood += np.log(likelihood(data[n], K, mu, sigma, pi)) # log P(X) = log \prod P(X_i) = \sum log P(X_i)
    return log_likelihood


def likelihood(x, K, mu, sigma, pi):
    rs = 0.0
    for k in range(K):
        rs += pi[k] * gaussian(x, mu[k], sigma[k])
    return rs


def m_step(data, K, resp,mode=None):
    idvs = (data.shape)[0]
    cols = (data.shape)[1]

    mu = np.zeros((K, cols))
    sigma = np.zeros((K, cols, cols))
    pi = np.zeros(K)

    marg_resp = np.zeros(K)
    for k in range(K):
        for i in range(idvs):
            marg_resp[k] += resp[i][k]
            mu[k] += (resp[i][k]) * data[i]
        mu[k] /= marg_resp[k]

        for i in range(idvs):
            # x_i = (np.zeros((1,cols))+data[k])
            x_mu = np.zeros((1, cols)) + data[i] - mu[k]
            sigma[k] += (resp[i][k] / marg_resp[k]) * x_mu * x_mu.T

        pi[k] = marg_resp[k] / idvs

    sigma1 = np.zeros((K, cols, cols))
    if mode == 'diag':
        for k in range(K):
            sigma1[k]=np.eye(cols) * sigma[k]
    elif mode == 'share':
        for k in range(K):
            sigma1[k]=np.mean(sigma,axis=0)
    elif mode == 'd+s':
        for k in range(K):
            sigma1[k]=np.eye(cols) * np.mean(np.array(sigma), axis=0)
    else:
        sigma1=sigma

    return mu, sigma1, pi


def gaussian(x, mu, sigma):
    idvs = len(x)
    norm_factor = (2 * np.pi) ** idvs

    norm_factor *= np.linalg.det(sigma)
    norm_factor = 1.0 / np.sqrt(norm_factor)

    x_mu = np.matrix(x - mu)

    rs = norm_factor * np.exp(-0.5 * x_mu * np.linalg.inv(sigma) * x_mu.T)
    return rs


def EM_multi(data, K, threshold, mode):
    mu0, sigma0, pi0 = random_parameters(data, K, mode =mode)
    current_log_likelihood = log_likelihood(data, K, mu0, sigma0, pi0)

    max_iter = 30
    mu, sigma, pi = mu0, sigma0, pi0
    for it in range(max_iter):
        resp = e_step(data, K, mu, sigma, pi)
        mu, sigma, pi = m_step(data, K, resp, mode=mode)

        new_log_likelihood = log_likelihood(data, K, mu, sigma, pi)
        if (abs(new_log_likelihood - current_log_likelihood) < threshold):       
            print("iters=%d" % (it))
            break
        current_log_likelihood = new_log_likelihood
        print(f"Iters:{it}", current_log_likelihood, pi)

    return mu, sigma, resp
