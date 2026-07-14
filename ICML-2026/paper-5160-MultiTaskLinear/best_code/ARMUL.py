import numpy as np
from MTL import MTL, baselines, prediction, evaluation
from preprocessing import split_cv

class ARMUL:
    def __init__(self, link = 'linear', n_class = 2, penalty = 'new'):
        # task j has n_j samples and an empirical risk f_j (sample average)
        # data = [X, y]
        # X: a list of m feature matrices, each of which is (n_j, d)
        # y: a list of m response vectors, each of which is (n_j, 1)
        # link = 'linear' or 'logistic'
        # n_class: number of classes in logistic regression, ignored for linear regression
        # class indices in y range from 0, 1, ..., n_class - 1
        # penalty: 'new' or 'ridge'
            # 'new': lbd_j * ||v|| on the j-th node
            # 'ridge': lbd_j * ||v||^2
        
        self.link = link      
        self.n_class = n_class
        self.penalty = penalty
        self.models = dict()
    
    def vanilla(self, data, lbd = None, eta_global = 0.01, eta_local = 0.01, T_global = 1000, T_local = 1, standardization = True, intercept = True):
        # vanilla ARMUL
        # lbd: a list of m penalty parameters
        mtl = MTL(data, link = self.link, intercept = intercept, n_class = self.n_class, penalty = self.penalty, standardization = standardization)  
        self.X_means, self.X_stds, self.y_mean, self.y_std, self.n_list = mtl.X_means, mtl.X_stds, mtl.y_mean, mtl.y_std, mtl.n_list

        if lbd is None:
            lbd = 0.1 * np.sqrt(data[0][0].shape[1] / self.n_list)
        mtl.vanilla_train(lbd, eta_global, eta_local, T_global, T_local)
        # local models
        self.models['vanilla'] = mtl.models['vanilla']
        # global model
        self.models['vanilla_gamma'] = mtl.models['vanilla_gamma']


    def clustered(self, data, lbd = None, K = 2, eta_B = 0.01, eta_local = 0.01, T_global = 1000, T_B = 1, T_local = 1, standardization = True, intercept = True):
        # clustered ARMUL
        mtl = MTL(data, link = self.link, intercept = intercept, n_class = self.n_class, penalty = self.penalty, standardization = standardization)  
        self.X_means, self.X_stds, self.y_mean, self.y_std, self.n_list = mtl.X_means, mtl.X_stds, mtl.y_mean, mtl.y_std, mtl.n_list

        if lbd is None:
            lbd = 0.1 * np.sqrt(data[0][0].shape[1] / self.n_list)
        mtl.clustered_train(lbd = lbd, K = K, eta_B = eta_B, eta_local = eta_local, T_global = T_global, T_B = T_B, T_local = T_local)
        self.models['clustered'] = mtl.models['clustered']
        self.models['clustered_B'] = mtl.models['clustered_B']
        self.models['clustered_Z'] = mtl.models['clustered_Z']


    def lowrank(self, data, lbd = None, K = 1, eta_B = 0.01, eta_Z = 0.01, eta_local = 0.01, T_global = 100, T_B = 1, T_Z = 1, T_local = 1, standardization = True, intercept = True):
        # low-rank ARMUL
        mtl = MTL(data, link = self.link, intercept = intercept, n_class = self.n_class, penalty = self.penalty, standardization = standardization)          
        self.X_means, self.X_stds, self.y_mean, self.y_std, self.n_list = mtl.X_means, mtl.X_stds, mtl.y_mean, mtl.y_std, mtl.n_list

        if lbd is None:
            lbd = 0.1 * np.sqrt(data[0][0].shape[1] / self.n_list)
        mtl.lowrank_train(lbd = lbd, K = K, eta_B = eta_B, eta_Z = eta_Z, eta_local = eta_local, T_global = T_global, T_B = T_B, T_Z = T_Z, T_local = T_local)
        self.models['lowrank'] = mtl.models['lowrank']
        self.models['lowrank_B'] = mtl.models['lowrank_B']
        self.models['lowrank_Z'] = mtl.models['lowrank_Z']


    def predict(self, X_test, model = 'vanilla'):
        return prediction(X_test, self.models[model], self.link, self.X_means, self.X_stds, self.y_mean, self.y_std)


    def evaluate(self, data_test, model = 'vanilla'):
        y_pred = self.predict(data_test[0], model)
        return evaluation(y_pred, data_test[1], self.link)
    

    def cv(self, data, lbd_list = None, model = 'vanilla', n_fold = 5, K = 2, eta_global = 0.01, eta_local = 0.01, eta_B = 0.01, eta_Z = 0.01, T_global = 1000, T_local = 1, T_B = 1, T_Z = 1, seed = 1000, standardization = True, intercept = True):
        # cross validation
        # lbd_list: a list of lambda-configurations
        np.random.seed(seed)
        m = len(data[0]) # number of tasks
        n_list = np.zeros(m).astype(int)
        for j in range(m):
            n_list[j] = len(data[0][j])
        splits = split_cv(n_list, n_fold, seed)

        if lbd_list is None:
            d = data[0][0].shape[1]
            lbd_list = [0.05 * i * np.sqrt(d / n_list) for i in range(1, 21)] # from 0.05 to 1 by 0.05
        L = len(lbd_list) # number of lambda configurations
        results = np.zeros((L, n_fold))

        for i in range(L):
            for k in range(n_fold):
                X_train, X_test = [], []
                y_train, y_test = [], []
                for j in range(m):
                    idx_test = splits[j][k]
                    X_test.append(data[0][j][idx_test])
                    y_test.append(data[1][j][idx_test])
                    idx_train = np.delete(np.array(range(n_list[j])), idx_test)
                    X_train.append(data[0][j][idx_train])
                    y_train.append(data[1][j][idx_train])
                
                data_train = [X_train, y_train]
                data_test = [X_test, y_test]

                if model == 'vanilla':
                    self.vanilla(data_train, lbd_list[i], eta_global, eta_local, T_global, T_local, standardization, intercept)
                if model == 'clustered':
                    self.clustered(data_train, lbd_list[i], K = K, eta_B = eta_B, eta_local = eta_local, T_global = T_global, T_B = T_B, T_local = T_local, standardization = standardization, intercept = intercept)
                if model == 'lowrank':
                    self.lowrank(data_train, lbd_list[i], K = K, eta_B = eta_B, eta_Z = eta_Z, eta_local = eta_local, T_global = T_global, T_B = T_B, T_Z = T_Z, T_local = T_local, standardization = standardization, intercept = intercept)

                tmp = self.evaluate(data_test, model = model)
                results[i, k] = tmp['average error']
        cv_err = np.mean(results, axis = 1)
        
        # cv errors
        self.errors_cv = cv_err
        
        # hyperparameter selection
        idx = np.argmin(cv_err)
        self.lbd_cv = lbd_list[idx] # optimal lambda
        self.lbd_list = lbd_list # list of all lambda's

        # refitting
        if model == 'vanilla':
            self.vanilla(data, lbd_list[idx], eta_global, eta_local, T_global, T_local, standardization, intercept)
        if model == 'clustered':
            self.clustered(data, lbd_list[idx], K = K, eta_B = eta_B, eta_local = eta_local, T_global = T_global, T_B = T_B, T_local = T_local, standardization = standardization, intercept = intercept)
        if model == 'lowrank':
            self.lowrank(data, lbd_list[idx], K = K, eta_B = eta_B, eta_Z = eta_Z, eta_local = eta_local, T_global = T_global, T_B = T_B, T_Z = T_Z, T_local = T_local, standardization = standardization, intercept = intercept)




# baselines
# single-task learning, hard parameter sharing, clustered MTL and low-rank MTL
class Baselines:
    def __init__(self, link = 'linear', n_class = 2):
        self.link = link      
        self.n_class = n_class
        self.models = dict()
    

    def STL_train(self, data, eta = 0.01, T = 1000, standardization = True, intercept = True): # STL   
        base = baselines(data, link = self.link, intercept = intercept, n_class = self.n_class, standardization = standardization)  
        self.X_means, self.X_stds, self.y_mean, self.y_std, self.n_list = base.X_means, base.X_stds, base.y_mean, base.y_std, base.n_list
        base.STL_train(eta, T)
        self.models['STL'] = base.models['STL']


    def DP_train(self, data, eta = 0.01, T = 1000, standardization = True, intercept = True):
        base = baselines(data, link = self.link, intercept = intercept, n_class = self.n_class, standardization = standardization)  
        self.X_means, self.X_stds, self.y_mean, self.y_std, self.n_list = base.X_means, base.X_stds, base.y_mean, base.y_std, base.n_list
        base.DP_train(eta, T)
        self.models['DP'] = base.models['DP']
    

    def clustered_train(self, data, K = 2, eta_B = 0.01, T = 100, T_B = 1, standardization = True, intercept = True):
        base = baselines(data, link = self.link, intercept = intercept, n_class = self.n_class, standardization = standardization)
        self.X_means, self.X_stds, self.y_mean, self.y_std, self.n_list = base.X_means, base.X_stds, base.y_mean, base.y_std, base.n_list
        base.clustered_train(K = K, eta_B = eta_B, T = T, T_B = T_B)
        self.models['clustered'] = base.models['clustered']
        self.models['clustered_B'] = base.models['clustered_B']
        self.models['clustered_Z'] = base.models['clustered_Z']        
    

    def lowrank_train(self, data, K = 2, eta_B = 0.01, eta_Z = 0.01, T = 100, T_B = 1, T_Z = 1, standardization = True, intercept = True):
        base = baselines(data, link = self.link, intercept = intercept, n_class = self.n_class, standardization = standardization)
        self.X_means, self.X_stds, self.y_mean, self.y_std, self.n_list = base.X_means, base.X_stds, base.y_mean, base.y_std, base.n_list
        base.lowrank_train(K = K, eta_B = eta_B, eta_Z = eta_Z, T = T, T_B = T_B, T_Z = T_Z)
        self.models['lowrank'] = base.models['lowrank']
        self.models['lowrank_B'] = base.models['lowrank_B']
        self.models['lowrank_Z'] = base.models['lowrank_Z']


    def predict(self, X_test, model = 'DP'):
        return prediction(X_test, self.models[model], self.link, self.X_means, self.X_stds, self.y_mean, self.y_std)


    def evaluate(self, data_test, model = 'DP'):
        y_pred = self.predict(data_test[0], model)
        return evaluation(y_pred, data_test[1], self.link)


    def cv(self, data, K_list = [2], model = 'lowrank', n_fold = 5, eta = 0.01, eta_B = 0.01, T = 100, T_B = 1, T_Z = 1, eta_Z = 0.01, seed = 1000, standardization = True, intercept = True):
        # cross validation
        # K_list: a list of K's
        assert model == 'lowrank' or model == 'clustered' # o/w no need for CV
        np.random.seed(seed)       

        m = len(data[0]) # number of tasks
        n_list = np.zeros(m).astype(int)
        for j in range(m):
            n_list[j] = len(data[0][j])
        splits = split_cv(n_list, n_fold, seed)

        L = len(K_list) # number of lambda configurations
        results = np.zeros((L, n_fold))

        for i in range(L):
            for k in range(n_fold):
                X_train, X_test = [], []
                y_train, y_test = [], []
                for j in range(m):
                    idx_test = splits[j][k]
                    X_test.append(data[0][j][idx_test])
                    y_test.append(data[1][j][idx_test])
                    idx_train = np.delete(np.array(range(n_list[j])), idx_test)
                    X_train.append(data[0][j][idx_train])
                    y_train.append(data[1][j][idx_train])
                
                data_train = [X_train, y_train]
                data_test = [X_test, y_test]

                if model == 'clustered':
                    self.clustered_train(data_train, K = K_list[i], eta_B = eta_B, T = T, T_B = T_B, standardization = standardization, intercept = intercept)
                if model == 'lowrank':
                    self.lowrank_train(data_train, K = K_list[i], eta_B = eta_B, eta_Z = eta_Z, T = T, T_B = T_B, T_Z = T_Z, standardization = standardization, intercept = intercept)
                tmp = self.evaluate(data_test, model = model)
                results[i, k] = tmp['average error']
        cv_err = np.mean(results, axis = 1)
        
        # cv errors
        self.errors_cv = cv_err
        
        # hyperparameter selection
        idx = np.argmin(cv_err)
        self.K_cv = K_list[idx] # optimal K
        self.K_list = K_list # list of all K's

        # refitting
        if model == 'clustered':
            self.clustered_train(data, K = self.K_cv, eta_B = eta_B, T = T, T_B = T_B, standardization = standardization, intercept = intercept)
        if model == 'lowrank':
            self.lowrank_train(data, K = self.K_cv, eta_B = eta_B, eta_Z = eta_Z, T = T, T_B = T_B, T_Z = T_Z, standardization = standardization, intercept = intercept)
                
