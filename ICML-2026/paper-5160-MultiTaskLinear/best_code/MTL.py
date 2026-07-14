import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD

from preprocessing import MTL_preprocessing

class MTL:
    def __init__(self, data, link = 'linear', intercept = True, n_class = 2, penalty = 'new', standardization = True):
        # task j has n_j samples and an empirical risk f_j (sample average)
        # objective fcn: \sum_j n_j [ f_j(w - v_j) + lambda_j \| v_j \|_2 ]
        # data = [X, y]
        # X: a list of m feature matrices, each of which is (n_j, d)
        # y: a list of m response vectors, each of which is (n_j, 1)
        # link = 'linear' or 'logistic'
        # n_class: number of classes in logistic regression, ignored for linear regression
        # class indices in y range from 0, 1, ..., n_class - 1
        # penalty: 'new' or 'ridge'
            # 'new': lbd_j * ||v|| on the j-th node
            # 'ridge': lbd_j * ||v||^2

        [self.X, self.y, self.X_means, self.X_stds, self.y_mean, self.y_std, self.n_list, self.d_out] = MTL_preprocessing(data, link, intercept, n_class, standardization)
        self.y_raw = data[1] # used in clustered and low-rank ARMUL

        self.intercept, self.n_class, self.standardization = intercept, n_class, standardization

        self.m = len(data[0])
        self.d = data[0][0].shape[1]
        if intercept: # No. of feature + 1 (when intercept = True)
            self.d += 1
        self.N = np.sum(self.n_list)
        self.link = link      
        self.penalty = penalty
        self.models = dict()

    
    def vanilla_train(self, lbd = None, eta_global = 0.01, eta_local = 0.01, T_global = 100, T_local = 1):
        # vanilla ARMUL
        # lbd: a list of m penalty parameters
        if lbd is None:
            lbd = 0.1 * np.sqrt(self.d / self.n_list)
        gamma = np.zeros((self.d, self.d_out)) # global model
        V = np.zeros((self.m, self.d, self.d_out)) # local corrections

        for t in range(T_global):
            g = np.zeros((self.d, self.d_out))
            for idx in range(self.m):
                # update v_idx
                v = pgd_v(self.X[idx], self.y[idx], self.link, lbd[idx], gamma, V[idx], self.penalty, eta_local, T_local)
                V[idx] = v

                # compute the local gradient of gamma
                g += gradient_loss(self.X[idx], self.y[idx], gamma - v, self.link) / self.N

            # gd on gamma
            gamma -= eta_global * g

        # global model
        self.models['vanilla_gamma'] = gamma

        # get local models
        self.models['vanilla'] = gamma - V


    def clustered_train(self, lbd = None, K = 2, eta_B = 0.01, eta_local = 0.01, T_global = 100, T_B = 1, T_local = 1):
        if lbd is None:
            lbd = 0.1 * np.sqrt(self.d / self.n_list)

        ####################
        # warm start by STL
        if self.link == 'linear' or self.n_class == 1:
            base = baselines([self.X, self.y], self.link, False, self.n_class, False)
        else:
            base = baselines([self.X, self.y_raw], self.link, False, self.n_class, False)
        
        base.STL_train(eta = eta_local, T = T_global)
        stl = base.models['STL'] # STL

        kmeans = KMeans(n_clusters = K, random_state = 0).fit(stl.reshape(self.m, self.d * self.d_out))
        B = np.zeros((K, self.d, self.d_out))
        Z = np.zeros((self.m, K))
        for k in range(K):
            B[k] = kmeans.cluster_centers_[k].reshape(self.d, self.d_out)
        Z[:, kmeans.labels_] = 1
        V = np.zeros((self.m, self.d, self.d_out))
        for j in range(self.m):
            V[j] = B[kmeans.labels_[j]] - stl[j] 
 
        ####################
        
        for t in range(T_global):
            # update v
            for idx in range(self.m):
                # compute the global model
                gamma = np.zeros((self.d, self.d_out))
                for r in range(self.d_out):
                    tmp1 = B[:, :, r].T
                    tmp2 = Z[idx].reshape(-1, 1)
                    gamma[:, r] = (tmp1 @ tmp2).reshape(-1,)
                # update v_idx
                v = pgd_v(self.X[idx], self.y[idx], self.link, lbd[idx], gamma, V[idx], self.penalty, eta_local, T_local)
                V[idx] = v

            # update B and Z
            B = gd_B(self.X, self.y, self.link, B, Z, V, eta_B, T_B)
            Z = hard_Z(self.X, self.y, self.link, B, V)

        # get local models
        Theta_clustered = np.zeros((self.m, self.d, self.d_out))
        for r in range(self.d_out):
            Br = B[:, :, r]
            Theta_clustered[:, :, r] = Z @ Br - V[:, :, r] # m-by-d
        
        self.models['clustered'] = Theta_clustered
        self.models['clustered_B'] = B
        self.models['clustered_Z'] = Z


    def lowrank_train(self, lbd = None, K = 2, eta_B = 0.01, eta_Z = 0.01, eta_local = 0.01, T_global = 100, T_B = 1, T_Z = 1, T_local = 1):
        if lbd is None:
            lbd = 0.1 * np.sqrt(self.d / self.n_list)

        ####################
        # warm start by STL
        if self.link == 'linear' or self.n_class == 1:
            base = baselines([self.X, self.y], self.link, False, self.n_class, False)
        else:
            base = baselines([self.X, self.y_raw], self.link, False, self.n_class, False)

        base.STL_train(eta = eta_local, T = T_global)
        stl = base.models['STL'] # STL

        ########################################

        svd = TruncatedSVD(n_components = K, n_iter = 10, random_state = 0)
        svd.fit( stl.reshape(self.m, self.d * self.d_out).T )
        Z = svd.components_.T # m by K
        B = np.zeros((K, self.d, self.d_out))
        tmp = np.zeros((self.m, self.d, self.d_out)) # low-rank model
        for r in range(self.d_out):
            B[:, :, r] = Z.T @ stl[:, :, r]
            tmp[:, :, r] = Z @ B[:, :, r]
        V = tmp - stl

        #####################
        # balancing
        tmp = np.linalg.norm(B.reshape(-1,))
        B, Z = B / np.sqrt(tmp), Z * np.sqrt(tmp)
        
        for t in range(T_global):            
            # Step 1: update v
            for idx in range(self.m):
                # compute the global model
                gamma = np.zeros((self.d, self.d_out))
                for r in range(self.d_out):
                    tmp1 = B[:, :, r].T
                    tmp2 = Z[idx].reshape(-1, 1)
                    gamma[:, r] = (tmp1 @ tmp2).reshape(-1,)
                # update v_idx
                v = pgd_v(self.X[idx], self.y[idx], self.link, lbd[idx], gamma, V[idx], self.penalty, eta_local, T_local)
                V[idx] = v
            
            # Step 2: update B and Z
            # compute the current objective value and save the current B, Z
            obj = self.get_loss_lowrank(B, Z, V)
            B_cache, Z_cache = B.copy(), Z.copy()

            # update B and Z
            B = gd_B(self.X, self.y, self.link, B, Z, V, eta_B, T_B)
            Z = gd_Z(self.X, self.y, self.link, B, Z, V, eta_Z, T_Z)

            # if the objective increases, use the old B, Z
            tmp = self.get_loss_lowrank(B, Z, V)
            if tmp > obj: 
                B, Z = B_cache.copy(), Z_cache.copy()
            obj = tmp

        self.models['lowrank'] = np.zeros((self.m, self.d, self.d_out))
        for r in range(self.d_out):
            self.models['lowrank'][:, :, r] = Z @ B[:, :, r] - V[:, :, r] # m-by-d
        self.models['lowrank_B'] = B
        self.models['lowrank_Z'] = Z


    def predict(self, X_test, model):
        return prediction(X_test, self.models[model], self.link, self.X_means, self.X_stds, self.y_mean, self.y_std)        


    def evaluate(self, data_test, model):
        y_pred = self.predict(data_test[0], model)
        return evaluation(y_pred, data_test[1], self.link)


    def get_loss_lowrank(self, B, Z, V):
        model = np.zeros((self.m, self.d, self.d_out))
        for r in range(self.d_out):
            model[:, :, r] = Z @ B[:, :, r] - V[:, :, r] # m-by-d
        ans = 0
        for j in range(self.m):
            ans += self.n_list[j] * loss(self.X[j], self.y[j], model[j], link = self.link)
        return ans


# baselines
# single-task learning, hard parameter sharing, clustered MTL and low-rank MTL
class baselines:
    def __init__(self, data, link = 'linear', intercept = True, n_class = 2, standardization = True):
        [self.X, self.y, self.X_means, self.X_stds, self.y_mean, self.y_std, self.n_list, self.d_out] = MTL_preprocessing(data, link, intercept, n_class, standardization)
        self.m = len(self.X)
        self.d = self.X[0].shape[1]
        self.N = np.sum(self.n_list)
        self.link = link      
        self.models = dict()

        
    def STL_train(self, eta = 0.01, T = 1000): # STL     
        Theta_STL = np.zeros((self.m, self.d, self.d_out))
        for t in range(T):
            for j in range(self.m):
                theta = Theta_STL[j]
                g = gradient_loss(self.X[j], self.y[j], theta, self.link) / self.n_list[j]
                Theta_STL[j] = theta - eta * g
        self.models['STL'] = Theta_STL


    def DP_train(self, eta = 0.01, T = 1000): # DP
        gamma = np.zeros((self.d, self.d_out))
        for t in range(T):
            g = np.zeros((self.d, self.d_out))
            for j in range(self.m):
                g += gradient_loss(self.X[j], self.y[j], gamma, self.link) / self.N
            gamma -= eta * g
        Theta_DP = np.zeros((self.m, self.d, self.d_out))
        for j in range(self.m):
            Theta_DP[j] = gamma
        self.models['DP'] = Theta_DP


    def clustered_train(self, K = 2, eta_B = 0.01, T = 100, T_B = 1):

        ####################
        # warm start by STL
        self.STL_train(eta = eta_B, T = T)
        stl = self.models['STL'] # STL

        kmeans = KMeans(n_clusters = K, random_state = 0).fit(stl.reshape(self.m, self.d * self.d_out))
        B = np.zeros((K, self.d, self.d_out))
        Z = np.zeros((self.m, K))
        for k in range(K):
            B[k] = kmeans.cluster_centers_[k].reshape(self.d, self.d_out)
        Z[:, kmeans.labels_] = 1
 
        #########################

        for t in range(T):
            # update B and Z
            B = gd_B(X = self.X, y = self.y, link = self.link, B = B, Z = Z, eta = eta_B, T = T_B)
            Z = hard_Z(X = self.X, y = self.y, link = self.link, B = B)

        # get local models
        Theta_clustered = np.zeros((self.m, self.d, self.d_out))
        for r in range(self.d_out):
            Br = B[:, :, r]
            Theta_clustered[:, :, r] = Z @ Br # m-by-d
        
        self.models['clustered'] = Theta_clustered
        self.models['clustered_B'] = B
        self.models['clustered_Z'] = Z

    
    def lowrank_train(self, K = 2, eta_B = 0.01, eta_Z = 0.01, T = 100, T_B = 1, T_Z = 1):
        
        ####################
        # warm start by STL
        self.STL_train(eta = eta_B, T = T)
        stl = self.models['STL'] # STL
        svd = TruncatedSVD(n_components = K, n_iter = 10, random_state = 0)
        svd.fit( stl.reshape(self.m, self.d * self.d_out).T )
        Z = svd.components_.T # m by K
        B = np.zeros((K, self.d, self.d_out)) 
        for r in range(self.d_out):
            B[:, :, r] = Z.T @ stl[:, :, r] 
        #####################
        # balancing
        tmp = np.linalg.norm(B.reshape(-1,))
        B, Z = B / np.sqrt(tmp), Z * np.sqrt(tmp)
        self.models['lowrank'] = np.zeros((self.m, self.d, self.d_out))
        for r in range(self.d_out):
            self.models['lowrank'][:, :, r] = Z @ B[:, :, r] # m-by-d
        obj = self.get_loss(model = 'lowrank')

        for t in range(T):
            # save the current model
            model_cache = self.models['lowrank'].copy

            # update B and Z
            B = gd_B(X = self.X, y = self.y, link = self.link, B = B, Z = Z, eta = eta_B, T = T_B)
            Z = gd_Z(X = self.X, y = self.y, link = self.link, B = B, Z = Z, eta = eta_Z, T = T_Z)
            #####################
            self.models['lowrank'] = np.zeros((self.m, self.d, self.d_out))
            for r in range(self.d_out):
                self.models['lowrank'][:, :, r] = Z @ B[:, :, r] # m-by-d
            tmp = self.get_loss(model = 'lowrank')

            if tmp > obj:
                self.models['lowrank'] = model_cache
                break
            obj = tmp
            #####################


        # get local models
        self.models['lowrank'] = np.zeros((self.m, self.d, self.d_out))
        for r in range(self.d_out):
            self.models['lowrank'][:, :, r] = Z @ B[:, :, r] # m-by-d
        self.models['lowrank_B'] = B
        self.models['lowrank_Z'] = Z


    def predict(self, X_test, model):
        return prediction(X_test, self.models[model], self.link, self.X_means, self.X_stds, self.y_mean, self.y_std)        


    def evaluate(self, data_test, model):
        y_pred = self.predict(data_test[0], model)
        return evaluation(y_pred, data_test[1], self.link)

    def get_loss(self, model):
        ans = 0
        for j in range(self.m):            
            ans += self.n_list[j] * loss(self.X[j], self.y[j], self.models[model][j], link = self.link)
        return ans


####################################################
## Auxiliaries

def gradient_loss(X, y, b, link = 'linear'):
    z = X @ b

    if link == 'logistic':
        d_out = b.shape[1]
        if d_out == 1: # y_i = 0 or 1
            z = 1 / (1 + np.exp(-z))
        else: # one-hoc encoding 
            tmp = np.exp(z)
            tmp2 = tmp @ np.ones((d_out, 1))
            z = tmp / tmp2
            
    return X.T @ (z - y)


def pgd_v(X, y, link, lbd, w, v, penalty = 'new', eta = 0.01, T = 100):
    # find an (approximately) optimal v_j for a given w 
    # i.e. approximate minimizer of f_j(w - v_j) + lambda_j \| v_j \|_2
    # optimization algorithm: proximal gd (new) or gd (ridge)
    
    # X: (n, d), y: (n, 1)
    # lbd: penalty parameter
    # w: global model
    # v: local correction
    # eta: stepsize
    # T: number of iterations

    v_t = v                
    for t in range(T):
        g = - gradient_loss(X, y, w - v_t, link) / len(y)
        tmp = v_t - eta * g # gradient descent

        if penalty == 'new':
            r = np.linalg.norm(tmp, ord = 'fro')
            if r > 0:
                v_t = max(0, 1 - lbd * eta / r) * tmp
            else:
                v_t = 0
        else: # 'ridge'
            v_t -= eta * (g + lbd * v_t)

    return v_t


def gd_B(X, y, link, B, Z, V = None, eta = 0.01, T = 100):
    # B: K-d-d_out
    # Z: m-K
    # V: m-d-d_out
    K, d, d_out = B.shape
    m = Z.shape[0]
    N = sum([len(l) for l in y])
    if V is None:
        V = np.zeros((m, d, d_out))        
    B_t = B
    for t in range(T):
        g = np.zeros((d, d_out))
        for idx in range(m):
            # compute the global model
            gamma = np.zeros((d, d_out))
            for r in range(d_out):
                gamma[:, r] = ( B_t[:, :, r].T @ Z[idx].reshape(-1, 1) ).reshape(-1,)                
            g += gradient_loss(X[idx], y[idx], gamma - V[idx], link) / N
        
        # GD on B
        for k in range(K):
            B_t[k] = B_t[k] - eta * np.sum(Z[:, k]) * g
    return B_t


def gd_Z(X, y, link, B, Z, V = None, eta = 0.01, T = 100):
    # B: K-d-d_out
    # Z: m-K
    # V: m-d-d_out
    K, d, d_out = B.shape
    m = Z.shape[0]
    n_list = [len(l) for l in y]
    if V is None:
        V = np.zeros((m, d, d_out))
    Z_t = Z
    for t in range(T):
        for idx in range(m):
            # compute the global model
            gamma = np.zeros((d, d_out))
            for r in range(d_out):
                gamma[:, r] = ( B[:, :, r].T @ Z_t[idx].reshape(-1, 1) ).reshape(-1,)
            g_idx = gradient_loss(X[idx], y[idx], gamma - V[idx], link) / n_list[idx]
        
            # GD on Z
            for k in range(K):
                Z_t[idx, k] = Z_t[idx, k] - eta * np.sum(g_idx * B[k])        
    return Z_t


def hard_Z(X, y, link, B, V = None):
    # B: K-d-d_out
    # Z: m-K
    # V: m-d-d_out
    K, d, d_out = B.shape
    m = len(X)
    Z = np.zeros((m, K))
    if V is None:
        V = np.zeros((m, d, d_out))

    for idx in range(m):
        tmp = np.zeros(K)
        for k in range(K):
            tmp[k] = loss(X[idx], y[idx], B[k] - V[idx], link)
        Z[idx, np.argmin(tmp)] = 1
    return Z


def loss(X, y, b, link = 'linear'):
    # X: (n, d), y: (n, d_out)
    n = X.shape[0]
    z = X @ b   # (n, d_out)
    if link == 'linear':
        return np.mean( (z - y) ** 2 )
    if link == 'logistic':
        d_out = b.shape[1]
        if d_out == 1: # y_i = 0 or 1
            return np.mean( np.log(1 + np.exp(z)) - y * z )
        else: # one-hoc encoding 
            tmp = np.exp(z)
            tmp2 = tmp @ np.ones((d_out, 1))
            return np.mean( np.log(tmp2) ) - np.sum(y * z) / n


def prediction(X_test, Theta, link = 'linear', X_means = np.zeros((1, 1)), X_stds = np.ones((1, 1)), y_mean = 0, y_std = 1):
    # X_test: m tasks
    # output: m arrays of shape = (n,)

    intercept = True
    if X_test[0].shape[1] == Theta.shape[1]: # no all-one column added
        intercept = False

    y_pred = []
    for (j, X) in enumerate(X_test):
        if intercept: # add an all-one column
            X = np.concatenate((np.ones((X.shape[0], 1)), X), axis = 1)
            
        # standardization using the means and stds of the training data
        X0 = (X - X_means.T) / X_stds.T
        z = X0 @ Theta[j]

        if link == 'linear':
            y_pred_j = (z * y_std + y_mean)
        if link == 'logistic':
            if Theta.shape[2] == 1:
                y_pred_j = ((z >= 0) + 0)
            else:
                y_pred_j = np.argmax(z, axis = 1)

        y_pred.append(y_pred_j.reshape(-1,))
    return y_pred
    
    
def evaluation(y_pred, y_true, link = 'linear'):
    # both y_pred and y_true have m tasks
    m = len(y_pred)
    n_list = np.zeros(m) # local sample sizes
    err = np.zeros(m) # MSE or misclassification error
    tmp = np.zeros(m) # for computing R2 in linear regression

    for (j, y_pred_j) in enumerate(y_pred):
        y_pred_j = y_pred_j.reshape(-1,)
        n_list[j] = len(y_pred_j)
        y_true_j = y_true[j].reshape(-1,)
        if link == 'linear':
            err[j] = np.mean( (y_pred_j - y_true_j) ** 2 )
            tmp[j] = np.mean( ( y_true[j] - np.mean(y_true_j) ) ** 2 )
        if link == 'logistic':
            err[j] = np.mean(y_pred_j != y_true_j)
    N = np.sum(n_list)

    ans = dict()
    ans['errors'] = err
    ans['average error'] = np.sum(err * n_list) / N

    if link == 'linear':
        total_variance = np.sum(tmp * n_list) / N
        ans['R2'] = 1 - ans['average error'] / total_variance
    
    return ans
