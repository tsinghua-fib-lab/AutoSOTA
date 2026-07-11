import numpy as np
import cvxpy as cvx
from scipy import linalg
import warnings

def cov2cor(Sigma):
    """
    Converts a covariance matrix to a correlation matrix
    :param Sigma : A covariance matrix (p x p)
    :return: A correlation matrix (p x p)
    """
    sqrtDiagSigma = np.sqrt(np.diag(Sigma))
    scalingFactors = np.outer(sqrtDiagSigma,sqrtDiagSigma)
    return np.divide(Sigma, scalingFactors)

def solve_sdp(Sigma, tol=1e-3):
    """
    Computes s for sdp-correlated Gaussian knockoffs
    :param Sigma : A covariance matrix (p x p)
    :param mu    : An array of means (p x 1)
    :return: A matrix of knockoff variables (n x p)
    """

    # Convert the covariance matrix to a correlation matrix
    # Check whether Sigma is positive definite
    if(np.min(np.linalg.eigvals(Sigma))<0):
        corrMatrix = cov2cor(Sigma + (1e-8)*np.eye(Sigma.shape[0]))
    else:
        corrMatrix = cov2cor(Sigma)
        
    p,_ = corrMatrix.shape
    s = cvx.Variable(p)
    objective = cvx.Maximize(sum(s))
    constraints = [ 2.0*corrMatrix >> cvx.diag(s) + cvx.diag([tol]*p), 0<=s, s<=1]
    prob = cvx.Problem(objective, constraints)
    prob.solve(solver='CVXOPT')
    
    assert prob.status == cvx.OPTIMAL

    s = np.clip(np.asarray(s.value).flatten(), 0, 1)
	
    # Scale back the results for a covariance matrix
    return np.multiply(s, np.diag(Sigma))
    
class GaussianKnockoffs:
    """
    Class GaussianKnockoffs
    Knockoffs for a multivariate Gaussian model
    """

    def __init__(self, Sigma, method="equi", mu=[], tol=1e-3):
        """
        Constructor
        :param model  : A multivariate Gaussian model object containing the covariance matrix
        :param method : Specifies how to determine the free parameters of Gaussian knockoffs.
                        Allowed values: "equi", "sdp" (default "equi")
        :return:
        """
        
        if len(mu)==0:
            self.mu = np.zeros((Sigma.shape[0],))
        else:
            self.mu = mu
        self.p = len(self.mu)
        self.Sigma = Sigma
        self.method = method

        # Initialize Gaussian knockoffs by computing either SDP or min(Eigs)

        if self.method=="equi":
            lambda_min = linalg.eigh(self.Sigma, eigvals_only=True, eigvals=(0,0))[0]
            s = min(1,2*(lambda_min-tol))
            self.Ds = np.diag([s]*self.Sigma.shape[0])
        elif self.method=="sdp":
            self.Ds = np.diag(solve_sdp(self.Sigma,tol=tol))
        else:
            raise ValueError('Invalid Gaussian knockoff type: '+self.method)
        self.SigmaInvDs = linalg.lstsq(self.Sigma,self.Ds)[0]
        self.V = 2.0*self.Ds - np.dot(self.Ds, self.SigmaInvDs)
        self.LV = np.linalg.cholesky(self.V+1e-10*np.eye(self.p))
        if linalg.eigh(self.V, eigvals_only=True, eigvals=(0,0))[0] <= tol:
            warnings.warn("Warning...........\
            The conditional covariance matrix for knockoffs is not positive definite. \
            Knockoffs will not have any power.")

    def generate(self, X):
        """
        Generate knockoffs for the multivariate Gaussian model
        :param X: A matrix of observations (n x p)
        :return: A matrix of knockoff variables (n x p)
        """
        n, p = X.shape
        muTilde = X - np.dot(X-np.tile(self.mu,(n,1)), self.SigmaInvDs)
        N = np.random.normal(size=muTilde.shape)
        return muTilde + np.dot(N,self.LV.T)
