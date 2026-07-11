DEFAULT_BATCH_SIZE = 20

from numpy.linalg import norm as L2
from scipy.stats import norm as univariate_normal
# from scipy.stats import multivariate_normal
from torch.distributions.multivariate_normal import MultivariateNormal

from tqdm import tqdm
import numpy as np 
import torch

import time


class KernelDensityEstimator:
    def __init__(self,kernel="multivariate_gaussian", bandwidth_estimator = "silverman",univariate_bandwidth = None):
        
        kernels = {"multivariate_gaussian":self.kernel_multivariate_gaussian}
                #    "univariate_gaussian": self.kernel_univariate_gaussian}
        bandwidth_estimators = {"silverman":self.est_bandwidth_silverman,
                               "scott":self.est_bandwidth_scott,
                                "identity": self.est_bandwidth_identity}
        compatible_estimators = {"multivariate_gaussian":["silverman","scott","identity"],
                               "univariate":[]}
                    
            
        self.kernel =  kernels[kernel]

        self.bandwidth_estimator = bandwidth_estimators[bandwidth_estimator]
        
        # if multivariate gaussian kernel is chosen, choose an estimator
        # if kernel=="multivariate_gaussian":
        #     self.bandwidth_estimator = bandwidth_estimators[bandwidth_estimator]
        
        # # if choosing univariate kernel without bandwidth clarified, print out a warning
        # elif kernel=="univariate_gaussian" and (not univariate_bandwidth):
        #     print("Please define your \"univariate_bandwidth\" parameters since the bandwidth cannot \
        #             automatically estimated using univariate kernel yet")
        
        # else:
        #     self.univariate_bandwidth = univariate_bandwidth

        # Bandwidth for estimating density
        self.bandwidth = None
        
        # Store data
        self.data = None
        
    def kernel_multivariate_gaussian(self,x):
        # Estimate density using multivariate gaussian kernel

        # Retrieve data
        data = self.data
        
        # Get dim of data
        d = data.shape[1]
        
        # Estimate bandwidth
        H = self.bandwidth_estimator()
        self.bandwidth = H

        # Calculate determinant of non zeros entry
        # diag_H = np.diagonal(H).copy()
        # diag_H[diag_H==0]=1
        # det_H = np.prod(diag_H)

        # diag_H = torch.diagonal(H).to(data.device)

        # Multivariate normal density estimate of x
        # var = multivariate_normal(mean=np.zeros(d), cov=H,allow_singular=True)
        # density = np.expand_dims(var.pdf(x),1)
        
        var = MultivariateNormal(torch.zeros(d,device=x.device), covariance_matrix=H)
        log_density = var.log_prob(x)
        # density = torch.exp(log_density)
        density = log_density
        
        
        return density
    
    # def kernel_univariate_gaussian(self,x):
    #     # Estimate density using univariate gaussian kernel

    #     # Retrieve data
    #     data = self.data
        
    #     # Get dim of data
    #     d = data.shape[1]
        
    #     # Estimate bandwidth
    #     h = self.univariate_bandwidth
    #     # Calculate density
    #     density = univariate_normal.pdf(L2(x,axis=1)/h)/h
        
    #     return density

    def fit(self,X,y=None):
        
        self.data = X # Make a pointer to the data variable
        
        return self
        
      
    def est_bandwidth_scott(self):
        # Estimate bandwidth using scott's rule

        # Retrieve data
        data = self.data
        
        # Get number of samples
        n = data.shape[0]
        
        # Get dim of data
        d = data.shape[1]
        
        # Compute standard along each i-th variable
        std = torch.std(data,dim=0) 
        
        # Construct the H diagonal bandwidth matrix with std along the diag
        H = (n**(-1/(d+4))*torch.diag(std))**2
        
        return H

    def est_bandwidth_identity(self):
        # Generate an identity matrix of density for bandwidth

        # Retrieve data
        data = self.data
        
        # Get number of samples
        n = data.shape[0]
        
        # Get dim of data
        d = data.shape[1]

        # Construct the H bandwidth matrix
        H = torch.eye(d)
        return H
    def est_bandwidth_silverman(self):
        # Estimate bandwidth using silverman's rule of thumbs

        # Retrieve data
        data = self.data
        
        # Get number of samples
        n = data.shape[0]
        
        # Get dim of data
        d = data.shape[1]
        
        # Compute standard along each i-th variable
        std = torch.std(data,axis=0) 
        
        # Construct the H diagonal bandwidth matrix with std along the diag
        H = (4/(d+2))**(1/(d+4))*(n**(-1/(d+4)))*torch.diag(std)
        return H
    
    def predict_proba(self,X,batch_size=10):
        # Predict proba for an input matrix X

        kernel_func = self.kernel

        # Retrieve data
        # data = self.data.half()
        # X = X.half()
        data = self.data
        X = X
        
        # number of samples in data
        n_data = data.shape[0]
        # number of samples in input set
        n_X = X.shape[0]
        dim = data.shape[1]
        
        # Init the estimated probabilities list
        # est_probs = torch.empty(0, device=data.device)
        

        # Add third dimension for broardcasting                          
        ## shape (1,dim,n_X)
        # X_ = np.expand_dims(X,0).transpose((0,2,1)) 
        X_ = X.unsqueeze(0).permute(0,2,1) 
        
          
        ## shape(n_data,dim,1)
        # data_ = np.expand_dims(data,2) 
        data_ = data.unsqueeze(2)
        
        
        # The difference of input set and data set pairwise (using broadcasting)
        
        ## shape (n_data,dim,n_X)
        delta_mid = X_ - data_ 

        # Flatten the delta into matrix
        # delta = delta.reshape(n_data*n_X,-1) # shape (n_data*n_X,dim)
        ####################################################################
        # this may be not correct 
        # delta = delta_mid.reshape(n_data*n_X,-1) # shape (n_data*n_X,dim)
        # should be changed to this 2024/11/12
        delta = delta_mid.permute(0,2,1).reshape(n_data*n_X,-1) # shape (n_data*n_X,dim)
        ####################################################################
        
        

        est_prob = kernel_func(delta) # (n_data*n_X,)

        # Calculate mean sum of probability for each sample
        est_prob = 1/n_data*est_prob.reshape(n_data,n_X).T.sum(axis=1) # shape (n_X,)
            
        return est_prob, delta_mid

    def predict(self,X,batch_size=DEFAULT_BATCH_SIZE):
        # Predict proba for a given X to belong to a dataset
        
        # if x is a vector (has 1 axis)
        if len(X.shape) == 1:
            # expand one more axis to represent a matrix
            # X = np.expand_dims(X,0)
            X = X.unsqueeze(0)
        with torch.cuda.amp.autocast(True):
            proba, delta = self.predict_proba(X,batch_size=batch_size)
                        
        return proba, delta
    
# input_data = np.load('test_data.npy')
# input_data = torch.from_numpy(input_data).to('cuda')

# import time
# start = time.time()


# Estimator = KernelDensityEstimator(kernel="multivariate_gaussian", bandwidth_estimator="silverman")
# Estimator.fit(input_data)
# proba = Estimator.predict(input_data, input_data.shape[0])

# end = time.time()
# time_count = start-end
# print(time_count)


# cu_ver_proba = np.load('cupy_proba.npy')
# cu_ver_proba = torch.from_numpy(cu_ver_proba)


