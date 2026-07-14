import numpy as np

import sys
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from ARMUL import ARMUL, Baselines
from preprocessing import load, split

class experiment_synthetic: # for synthetic data experiments
    def __init__(self,  n = 100, m = 50, d = 50):
        self.n, self.m, self.d = n, m, d
        
    def getsamples(self, setting, K = 3, signal_norm = 2, sigma = 1, delta = 0.1, epsilon = 0, norm_outliers = 2, seed = 1000):
        assert setting == 'vanilla' or setting == 'clustered' or setting == 'lowrank'
        assert K <= self.d
        
        self.setting, self.K, self.seed = setting, K, seed
        
        np.random.seed(seed)
        data_X, data_y = [], []
        Theta = np.zeros((self.d, self.m))

        if setting == 'vanilla':
            Theta[0, :] = signal_norm
        elif setting == 'clustered':
            r = int(self.m / K)
            for k in range(K):
                if k < K - 1:
                    Theta[k, (k * r):(k * r + r)] = signal_norm
                else:
                    Theta[k, (k * r):self.m] = signal_norm
        elif setting == 'lowrank':
            for j in range(self.m):
                z_j = np.random.randn(K)
                Theta[0:K, j] = signal_norm * z_j / np.linalg.norm(z_j)

        S_outliers = np.random.choice(self.m, size = int(self.m * epsilon), replace = False)
        S = np.delete(np.array(range(self.m)), S_outliers)
        S_outliers = set(S_outliers)
        S = set(S)
        
        
                
        for j in range(self.m):
            # get coefficient vectors
            delta_j = np.random.randn(self.d) 
            if j in S_outliers: # outlier tasks
                # random vector
                Theta[:, j] = norm_outliers * delta_j / np.linalg.norm(delta_j)
            else: # non-outliers
                # add perturbation
                Theta[:, j] = Theta[:, j] + delta * delta_j / np.linalg.norm(delta_j)

            # generate X and y
            X_j = np.random.randn(self.n, self.d)
            data_X.append(X_j)
            y_j = (X_j @ Theta[:, j].reshape(-1, 1)).reshape(-1,) + sigma * np.random.randn(self.n)
            data_y.append(y_j)
            
        self.data = [data_X, data_y]
        self.Theta = Theta
        self.S = S
        self.Sc = S_outliers 
    
    def run(self, lbd_list, n_fold = 5, eta = 0.05, T = 100):        
        # ARMUL
        mtl = ARMUL(link = 'linear', n_class = 1, penalty = 'new')    
        mtl.cv(self.data, lbd_list = lbd_list, model = self.setting, n_fold = n_fold, K = self.K, eta_global = eta, eta_local = eta, eta_B = eta, eta_Z = eta, T_global = T, seed = self.seed, intercept = False)
        Theta_hat = mtl.models[self.setting][:, :, 0].T

        # baselines
        baseline = Baselines(link = 'linear', n_class = 1)

        # STL
        baseline.STL_train(self.data, eta = eta, T = T, intercept = False)
        Theta_hat_STL = baseline.models['STL'][:, :, 0].T
        # DP
        baseline.DP_train(self.data, eta = eta, T = T, intercept = False)
        Theta_hat_DP = baseline.models['DP'][:, :, 0].T

        # baseline (clustered or lowrank)
        if self.setting == 'clustered':
            baseline.clustered_train(self.data, K = self.K, eta_B = eta, T = T, intercept = False)
            Theta_hat_base = baseline.models[self.setting][:, :, 0].T
        elif self.setting == 'lowrank':
            baseline.lowrank_train(self.data, K = self.K, eta_B = eta, eta_Z = eta, T = T, intercept = False)
            Theta_hat_base = baseline.models[self.setting][:, :, 0].T    
            
        self.err, self.err_S = dict(), dict()

        err_ARMUL, err_STL, err_DP, err_base = [], [], [], []
        for j in range(self.m):
            err_ARMUL.append( np.linalg.norm((Theta_hat - self.Theta)[:, j]) )
            err_STL.append( np.linalg.norm((Theta_hat_STL - self.Theta)[:, j]) )
            err_DP.append( np.linalg.norm((Theta_hat_DP - self.Theta)[:, j]) )
            if self.setting == 'clustered' or self.setting == 'lowrank':
                err_base.append( np.linalg.norm((Theta_hat_base - self.Theta)[:, j]) )

        self.err['ARMUL'] = err_ARMUL
        self.err['STL'] = err_STL
        self.err['DP'] = err_DP
        self.err['baseline'] = err_base

        self.err_S['ARMUL'] = [err_ARMUL[j] for j in self.S]
        self.err_S['STL'] = [err_STL[j] for j in self.S]
        self.err_S['DP'] = [err_DP[j] for j in self.S]
        if self.setting == 'clustered' or self.setting == 'lowrank':
            self.err_S['baseline'] = [err_base[j] for j in self.S]
        else:
            self.err_S['baseline'] = []
    



class experiment_real: # for real data experiments
    def __init__(self, dataset, PCs, path, prop = None, seed = 1000):
        [data_raw, self.link, self.n_class] = load(dataset, path, PCs = PCs)
        [self.data_train, self.data_test] = split(data = data_raw, prop = prop, seed = seed)
        self.seed = seed


    def run(self, lbd_factors = None, model = 'vanilla', n_fold = 5, K = 2, eta_global = 0.01, eta_local = 0.01, eta_B = 0.01, eta_Z = 0.01, T_global = 1000, T_local = 1, T_B = 1, T_Z = 1, intercept = True, penalty = 'new', seed = None, standardization = True):
        # new method
        if seed is None:
            seed = self.seed
        np.random.seed(seed)
 
        mtl = ARMUL(link = self.link, n_class = self.n_class, penalty = penalty)

        # list of lambda-configurations
        if lbd_factors is None:
            L = 20
            lbd_factors = [(i / L) for i in range(1, L + 1)]
            
        d = self.data_train[0][0].shape[1]
        n_list = np.array([len(l) for l in self.data_train[0]])
        ratios = np.sqrt(d / n_list)
        lbd_list = [factor * ratios for factor in lbd_factors]

        # cross validation
        mtl.cv(self.data_train, lbd_list = lbd_list, model = model, n_fold = n_fold, K = K, eta_global = eta_global, eta_local = eta_local, eta_B = eta_B, eta_Z = eta_Z, T_global = T_global, T_local = T_local, T_B = T_B, T_Z = T_Z, seed = self.seed, standardization = standardization, intercept = intercept)                 
        self.lbd_list = mtl.lbd_list # list of all lambda's        
        self.lbd_cv = mtl.lbd_cv # optimal lambda
        self.errors_cv = mtl.errors_cv # CV errors    

        # OOS evaluation
        self.results = mtl.evaluate(self.data_test, model = model)
    

    def run_baseline(self, model = 'DP', n_fold = 5, intercept = True, K_list = [2], eta = 0.01, eta_B = 0.01, T = 100, T_B = 1, T_Z = 1, eta_Z = 0.01, seed = None, standardization = True):
        # baseline
        if seed is None:
            seed = self.seed
        np.random.seed(seed)

        # training
        baseline = Baselines(link = self.link, n_class = self.n_class)
        if model == 'DP':
            baseline.DP_train(self.data_train, eta = eta, T = T, standardization = standardization, intercept = intercept)
            self.errors_cv_baseline = None
        if model == 'STL':
            baseline.STL_train(self.data_train, eta = eta, T = T, standardization = standardization, intercept = intercept)
            self.errors_cv_baseline = None
        if model == 'clustered' or model == 'lowrank':
            # cross validation
            baseline.cv(self.data_train, K_list = K_list, model = model, n_fold = n_fold, eta = eta, eta_B = eta_B, T = T, T_B = T_B, T_Z = T_Z, eta_Z = eta_Z, seed = seed, standardization = standardization, intercept = intercept)
            self.errors_cv_baseline = baseline.errors_cv # CV errors

        # OOS evaluation
        self.results_baseline = baseline.evaluate(self.data_test, model = model)
