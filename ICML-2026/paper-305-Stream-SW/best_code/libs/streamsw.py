from .kll import KLL
import numpy as np
from .sw import *
import torch
class  StreamSW:
    def __init__(self, L,d,k,p=2, c = 2.0/3.0, thetas=None,lazy=True, alternate=True):
        if k<=0: raise ValueError("k must be a positive integer.")
        if c <= 0.5 or c > 1.0: raise ValueError("c must larger than 0.5 and at most 1.0.")
        self.L = L
        self.d=d
        self.k=k
        self.c=c
        self.lazy=lazy
        self.p=p
        self.alternate=alternate
        self.klls_mu = [KLL(k=k, c = c, lazy=lazy, alternate=alternate) for _ in range(L)]
        self.klls_nu = [KLL(k=k, c=c, lazy=lazy, alternate=alternate) for _ in range(L)]
        if(thetas is None):
            self.theta = np.random.randn(L,d)
            self.theta = self.theta/np.sqrt(np.sum(self.theta**2,axis=1,keepdims=True))
        else:
            self.theta = thetas

    def reset(self):
        self.klls_mu = [KLL(k=self.k, c=self.c, lazy=self.lazy, alternate=self.alternate) for _ in range(self.L)]
        self.klls_nu = [KLL(k=self.k, c=self.c, lazy=self.lazy, alternate=self.alternate) for _ in range(self.L)]
        self.theta = np.random.randn(self.L, self.d)
        self.theta = self.theta / np.sqrt(np.sum(self.theta ** 2, axis=1, keepdims=True))

    def sizes(self):
        return sum([len(kll.items()) for kll in self.klls_mu]),sum([len(kll.items()) for kll in self.klls_nu])
    def update_mu(self,X):
        projected_X = np.dot(self.theta,X.T)#Lxn
        for l in range(self.L):
            self.klls_mu[l].update_batch(projected_X[l])

    def update_nu(self, Y):
        projected_Y = np.dot(self.theta, Y.T)  # Lxn
        for l in range(self.L):
            self.klls_nu[l].update_batch(projected_Y[l])

    def compute_distance(self,X=None,Y=None):
        if X is not None:
            self.update_mu(X)
        if Y is not None:
            self.update_nu(Y)
        supports_mus=[]
        supports_nus=[]
        weights_mus=[]
        weights_nus=[]
        for l in range(self.L):
            supports_mu, weights_mu= self.klls_mu[l].empirical_distribution()
            supports_mu = np.array(supports_mu)
            weights_mu = np.array(weights_mu)
            weights_mu = weights_mu/np.sum(weights_mu)

            supports_nu, weights_nu = self.klls_nu[l].empirical_distribution()
            supports_nu = np.array(supports_nu)
            weights_nu = np.array(weights_nu)
            weights_nu = weights_nu / np.sum(weights_nu)

            supports_mus.append(supports_mu)
            supports_nus.append(supports_nu)
            weights_mus.append(weights_mu)
            weights_nus.append(weights_nu)
        supports_mus = torch.from_numpy(np.array(supports_mus).T)
        supports_nus = torch.from_numpy(np.array(supports_nus).T)
        weights_mus= torch.from_numpy(np.array(weights_mus).T)
        weights_nus = torch.from_numpy(np.array(weights_nus).T)
        distance = one_dimensional_Wasserstein(supports_mus, supports_nus, u_weights=weights_mus, v_weights=weights_nus, p=self.p).mean()**(1./self.p)
        return distance

class  OnesidedStreamSW:
    def __init__(self, L,d,k,p=2, c = 2.0/3.0, thetas=None,lazy=True, alternate=True):
        if k<=0: raise ValueError("k must be a positive integer.")
        if c <= 0.5 or c > 1.0: raise ValueError("c must larger than 0.5 and at most 1.0.")
        self.L = L
        self.d=d
        self.k=k
        self.c=c
        self.lazy=lazy
        self.p=p
        self.alternate=alternate
        self.klls_mu = [KLL(k=k, c = c, lazy=lazy, alternate=alternate) for _ in range(L)]
        if(thetas is None):
            self.theta = np.random.randn(L,d)
            self.theta = self.theta/np.sqrt(np.sum(self.theta**2,axis=1,keepdims=True))
        else:
            self.theta = thetas

    def reset(self):
        self.klls_mu = [KLL(k=self.k, c=self.c, lazy=self.lazy, alternate=self.alternate) for _ in range(self.L)]
        self.theta = np.random.randn(self.L, self.d)
        self.theta = self.theta / np.sqrt(np.sum(self.theta ** 2, axis=1, keepdims=True))

    def sizes(self):
        return sum([len(kll.items()) for kll in self.klls_mu]),sum([len(kll.items()) for kll in self.klls_nu])
    def update(self,X):
        projected_X = np.dot(self.theta,X.T)#Lxn
        for l in range(self.L):
            self.klls_mu[l].update_batch(projected_X[l])


    def compute_distance(self,X,a,Y=None):
        if Y is not None:
            self.update(Y)
        supports_mus=[]
        weights_mus=[]
        for l in range(self.L):
            supports_mu, weights_mu= self.klls_mu[l].empirical_distribution()
            supports_mu = np.array(supports_mu)
            weights_mu = np.array(weights_mu)
            weights_mu = weights_mu/np.sum(weights_mu)
            supports_mus.append(supports_mu)
            weights_mus.append(weights_mu)
        supports_mus = torch.from_numpy(np.array(supports_mus).T)
        weights_mus= torch.from_numpy(np.array(weights_mus).T)

        projected_X = torch.from_numpy(np.dot(self.theta, X.T).T)
        a = torch.from_numpy(np.tile(a, (self.L, 1)).T)

        distance = one_dimensional_Wasserstein(supports_mus, projected_X, u_weights=weights_mus, v_weights=a, p=self.p).mean()**(1./self.p)
        return distance
    def compute_distance_torch(self,X,a,Y=None):
        if Y is not None:
            self.update(Y)
        supports_mus=[]
        weights_mus=[]
        for l in range(self.L):
            supports_mu, weights_mu= self.klls_mu[l].empirical_distribution()
            supports_mu = np.array(supports_mu)
            weights_mu = np.array(weights_mu)
            weights_mu = weights_mu/np.sum(weights_mu)
            supports_mus.append(supports_mu)
            weights_mus.append(weights_mu)
        supports_mus = torch.from_numpy(np.array(supports_mus).T).to(X.device)
        weights_mus= torch.from_numpy(np.array(weights_mus).T).to(X.device)
        projected_X = torch.matmul(torch.from_numpy(self.theta).to(X.device),X.T).T
        a = torch.from_numpy(np.tile(a, (self.L, 1)).T).to(X.device)
        distance = one_dimensional_Wasserstein(supports_mus, projected_X, u_weights=weights_mus, v_weights=a, p=self.p).mean()**(1./self.p)
        return distance
