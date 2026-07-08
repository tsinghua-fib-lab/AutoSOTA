import numpy as np
from scipy.optimize import minimize
from .metrics import sqdist_euclid, grad_sqdist_euclid, mixhess_sqdist_euclid

        
def KExponentialNumpy(ConstrObj, ConstrGrad, x0, v, K, sqdist = sqdist_euclid, gradsqdist=grad_sqdist_euclid, 
                      mixhesssqdist = mixhess_sqdist_euclid, constrfact=1, firstStep='step'):
    """
    Compute K steps of the exponential map on a constrained manifold given bei the approximate zero level set
    of ConstrObj.
    
    This function uses a numerical scheme that reproduces discrete variational 
    geodesics minimizing the path energy starting from point x0 with initial (discrete) velocity v.
    'BFGS' from scipy.optimize is used for the inner optimization problem in each step.

    Parameters
    ----------
    ConstrObj : callable
        Constraint function that implicitly defines the manifold.
        Signature: ConstrObj(coord) -> array of shape (n-d, m)
        See GeodesicSolverNumpy documentation for details.
        
    ConstrGrad : callable
        Gradient (Jacobian) of the constraint function.
        Signature: ConstrGrad(coord) -> array of shape (n-d, m, n)
        See GeodesicSolverNumpy documentation for details.
        
    x0 : array-like, shape (n,)
        Initial point on the manifold. Should satisfy ConstrObj(x0) ≈ 0.
        
    v : array-like, shape (n,)
        Initial velocity vector in the tangent space at x0.
        The total geodesic length is approximately ||v||.
        
    K : int
        Number of discrete steps to compute along the geodesic.
        
    sqdist : callable, optional (default: sqdist_euclid)
        Squared distance function defining the metric.
        Signature: sqdist(x, y) -> float
        See GeodesicSolverNumpy documentation for details.
        
    gradsqdist : callable, optional (default: grad_sqdist_euclid)
        Gradient of the squared distance function.
        Signature: gradsqdist(x, y, wrt_first=False) -> array of shape (n,)
        See GeodesicSolverNumpy documentation for details.
        
    mixhesssqdist : callable, optional (default: mixhess_sqdist_euclid)
        Mixed Hessian (second derivatives) of the squared distance function.
        Signature: mixhesssqdist(x, y, wrt_first=False) -> array of shape (n, n)
        
        Returns ∂²sqdist/∂x∂y, the mixed partial derivatives.
        - If wrt_first=True: first derivative wrt first argument
        - If wrt_first=False: first derivative wrt second argument
        
        For Euclidean metric:
            mixhess_sqdist_euclid(x, y, wrt_first=False) = 2*I (identity matrix)
            
    constrfact : float, optional (default: 1)
        Weight factor for the constraint penalty term.
        Depends on the accuracy of the constraint

    firstStep : {'step', 'linear'}, optional (default: 'step')
        Method for computing the first step:
        - 'step': Use exponential step with midpoint x0 + v/(2K)
        - 'linear': Use simple linear step x0 + v/K
            
    Returns
    -------
    result : ndarray, shape (K+1, n)
        Discrete exponential paths with K+1 points
    """
    print('Computing K-exponential map')
    v = v/K
    x = np.zeros((K+1, x0.size))
    x[0,:] = x0
    if firstStep == 'step':
        x[1,:] = ExponentialStep(ConstrObj, ConstrGrad, x[0,:], x0 + v/2, sqdist, gradsqdist, mixhesssqdist, 
                                 constrfact=constrfact)
    if firstStep == 'linear':
        x[1,:] = x0 + v
    for i in range(1,K):
        x[i+1,:] = ExponentialStep(ConstrObj,ConstrGrad, x[i-1,:],x[i,:], sqdist, gradsqdist, mixhesssqdist, 
                                   constrfact=constrfact)

    return x.reshape( -1, x0.size)

def ExponentialStep(ConstrObj, ConstrGrad, x0, x1, sqdist = sqdist_euclid, gradsqdist=grad_sqdist_euclid, 
                    mixhesssqdist = mixhess_sqdist_euclid, constrfact=1):
    
    n = x0.size
    d = n - ConstrObj(np.reshape(x0, (-1,n))).shape[0]

    lam = np.zeros(n-d)
    x_init = np.append(x1,lam)
    
    args = (ConstrObj,ConstrGrad, x0, x1, sqdist, gradsqdist, mixhesssqdist, constrfact)
    
    options = {'disp':True, 'gtol':1e-09}
    optResult = minimize(F,x_init, jac = DF,args = args, method = 'BFGS', options = options)

    result = optResult.get('x')
    return result[:n]

def F(x_it, ConstrObj, ConstrGrad, x0, x1, sqdist=sqdist_euclid, gradsqdist=grad_sqdist_euclid, 
      mixhesssqdist = mixhess_sqdist_euclid, constrfact=1):
    n = x0.size
    x2 = x_it[:n]
    lam = x_it[n:]
    l = lam.size

    dphi_x1 = ConstrGrad(np.reshape(x1, (-1,n)))

    geod = 2*(gradsqdist(x0, x1, wrt_first = False) + gradsqdist(x1,x2, wrt_first = True))
    for i in range(l):
        geod -= lam[i]*dphi_x1[i,0,:]
    geod = 0.5*np.linalg.norm(geod)**2 
    constr = 0.5*np.linalg.norm( ConstrObj( np.reshape(x2, (-1,n)) ) )**2
    return geod + constrfact*constr

def DF(x_it, ConstrObj, ConstrGrad, x0, x1, sqdist=sqdist_euclid, gradsqdist=grad_sqdist_euclid, 
       mixhesssqdist = mixhess_sqdist_euclid, constrfact=1):
    n = x0.size
    x2 = x_it[:n]
    lam = x_it[n:]
    l = lam.size
   
    dphi_x2 = ConstrGrad(np.reshape(x2, (-1,n)))
    dphi_x1 = ConstrGrad(np.reshape(x1, (-1,n)))
    inner = 2*(gradsqdist(x0, x1, wrt_first = False) +gradsqdist(x1,x2, wrt_first = True))
    for i in range(l):
        inner -= lam[i]*dphi_x1[i,0,:]
    dF_x = np.reshape( 2*inner.dot(mixhesssqdist(x1,x2, wrt_first = True).T) 
                      + constrfact*ConstrObj(np.reshape(x2, (-1,n)))[:,0].dot(dphi_x2[:,0,:]), -1)    
    dF_lam = -(inner).dot(dphi_x1[:,0,:].T)
    deriv = np.append(dF_x,dF_lam)
    return deriv