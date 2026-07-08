import numpy as np
import scipy.optimize

from .metrics import sqdist_euclid, grad_sqdist_euclid

class GeodesicSolverNumpy:
    def __init__(self, n, ConstrObj, ConstrGrad, sqdist=sqdist_euclid, gradsqdist=grad_sqdist_euclid):
        """
        Initialize Geodesic Solver that computes geodesics on the approximate zero levelset of ConstrObj.
        
        The solver uses an augmented Lagrangian method to find minimum-energy paths constrained to a 
        manifold defined implicitly by ConstrObj(x) ≈ 0.

        If you want to use pullback metrics defined through the decoder we recommend using the 
        torch version implemented in GeodesicSolverTorch instead.
        
        Parameters
        ----------
        n : int
            Dimension of the ambient space (i.e., latent space dimension).
            
        ConstrObj : callable
            Constraint function that implicitly defines the manifold.
            We do **not** require that for a d-dimensional manifold embedded in n dimmension, 
            the constraint function returns n-d values.
            E.g. it also works for a projection function that returns n values but as approximate rank n-d.
            If you have learned an implicit manifold representation with this repository's LatentProjector class, 
            you can obtain ConstrObj as follows:
            ConstrObj = latentprojector.phi(use_torch=false)
            This automatically provides the appropriate constraint function from your trained model.
            Signature: ConstrObj(coord) -> array
            
            Input:
                coord : array of shape (m, n)
                    m points in n-dimensional space (batch of coordinates)
            
            Output:
                array of shape (n-d, m)
                    Constraint values for each point. The manifold is defined by the 
                    zero levelset where ConstrObj(x) ≈ 0.
                    - (n-d) is the codimension of the manifold
                    - If ConstrObj has full rank, the manifold dimension is d
                    - Each column corresponds to one input point
                
        ConstrGrad : callable
            Gradient (Jacobian) of the constraint function.
            If you have learned an implicit manifold representation with this repository's LatentProjector class, 
            you can obtain ConstrObj as follows:
            ConstrGrad = latentprojector.dphi(use_torch=false)

            Signature: ConstrGrad(coord) -> array
            
            Input:
                coord : array of shape (m, n)
                    m points in n-dimensional space
            
            Output:
                array of shape (n-d, m, n)
                    Jacobian of ConstrObj at each point.
                    - First axis: constraint component index (n-d components)
                    - Second axis: batch index (m points)
                    - Third axis: spatial derivative direction (n dimensions)
                    
                    For constraint i and point j: ConstrGrad[i, j, :] = ∇ConstrObj_i(coord[j])
        
        sqdist : callable, optional (default: sqdist_euclid)
            (Approximation) of squared distance function defining the metric on the ambient space.
            
            Signature: sqdist(x, y) -> float
            
            Input:
                x : array of shape (n,) or (1, n)
                    First point
                y : array of shape (n,) or (1, n)
                    Second point
            
            Output:
                float
                    Squared distance between x and y
            
            Default (Euclidean):
                sqdist(x, y) = ||y - x||²
                        
        gradsqdist : callable, optional (default: grad_sqdist_euclid)
            Gradient of the squared distance function.
            
            Signature: gradsqdist(x, y, wrt_first=False) -> array
            
            Input:
                x : array of shape (n,) or (m, n)
                    First point(s)
                y : array of shape (n,) or (m, n)
                    Second point(s)
                wrt_first : bool, optional (default: False)
                    If True, compute gradient with respect to x
                    If False, compute gradient with respect to y
            
            Output:
                array of shape (n,) or (m, n)
                    Gradient ∇_x sqdist(x,y) if wrt_first=True
                    Gradient ∇_y sqdist(x,y) if wrt_first=False
            
            Default (Euclidean):
                gradsqdist(x, y, wrt_first=False) = 2*(y - x)
                gradsqdist(x, y, wrt_first=True)  = 2*(x - y) = -2*(y - x)
        
        Examples
        --------
        >>> # Sphere in 3D (2D manifold)
        >>> def sphere_constraint(coord):
        ...     # coord shape: (m, 3)
        ...     return (np.sum(coord**2, axis=1) - 1).reshape(1, -1)
        >>> 
        >>> def sphere_gradient(coord):
        ...     # coord shape: (m, 3)
        ...     return (2 * coord).reshape(1, -1, 3)  # shape: (1, m, 3)
        >>> 
        >>> solver = GeodesicSolverNumpy(n=3, ConstrObj=sphere_constraint, 
        ...                              ConstrGrad=sphere_gradient)
        """
        #constraints must handle input as mxn, m batch dimension
        self.ConstrObj = ConstrObj # output (n-d,m)
        self.ConstrGrad = ConstrGrad # (n-d,m,n)
        self.sqdist = sqdist
        self.gradsqdist= gradsqdist
        
        self.n = n
        self.d = n - self.ConstrObj(np.zeros((1,self.n))).shape[0] #image domain of constraint is n-d

    def Lagrangian(self, coord, lam, mu, K, xA = None, xB = None):

        coord = np.reshape(coord, (-1,self.n))
        valPhi = self.ConstrObj(coord)
        coord = np.concatenate(([xA],coord,[xB]))

        energy = np.sum((coord[1:] - coord[:-1]) ** 2)
        constraint = np.sum(np.einsum('ij,ij->i', lam[:self.n-self.d, :], valPhi[:self.n-self.d, :]))

        L = K*energy - constraint + 0.5*mu*(np.linalg.norm(valPhi))**2 
      
        return L

    def DLagrangian(self, coord, lam, mu, K, xA=None, xB=None):
        #gradient of augmented lagrangian
        n = self.n
        coord = coord.reshape(-1, n)
        dof = coord.shape[0]

        valPhi  = self.ConstrObj(coord)       # (n-d, dof)
        valDphi = self.ConstrGrad(coord)       

        coord = np.vstack((xA, coord, xB))

        # gradient constraint term
        factors = mu * valPhi - lam
        constraint = np.sum(factors[:, :, None] * valDphi, axis=0 )

        # gradient energy
        grad_prev = self.gradsqdist(coord[:-2], coord[1:-1], wrt_first=False)
        grad_next = self.gradsqdist(coord[1:-1], coord[2:], wrt_first=True)

        gradL = K * (grad_prev + grad_next) + constraint

        return gradL.reshape(n * dof)
    
    def AugLagrangeMinimize(self, resolution, xA, xB, x0 = None, lam = None, mu = 10, maxmu = 3e4, 
                            tol = 1e-5, tolConstraint = 1e-6, alpha = 2, disp = True):
        """
        Compute a geodesic path between two points using augmented Lagrangian optimization.
        
        This method finds the minimum-energy path constrained to the manifold by iteratively
        solving an augmented Lagrangian problem. The inner optimization uses scipy.optimize.minimize
        with the BFGS method.
        Note: When using learned representation functions the gradient may not reach very small values 
        due to inherent approximation errors in the learned manifold.

        Parameters
        ----------
        resolution : int
            Number of intermediate points along the geodesic path (excluding endpoints).
            
        xA : array-like, shape (n,)
            Starting point of the geodesic. Must lie on (or near) the manifold.
            
        xB : array-like, shape (n,)
            Ending point of the geodesic. Must lie on (or near) the manifold.
            
        x0 : array-like, shape (resolution, n), optional
            Initial guess for intermediate points. If None (default), initializes with
            a piecewise constant path that jumps from xA to xB at the midpoint.

        lam : array-like, shape (n-d, resolution), optional
            Initial Lagrange multipliers for the constraint. Default is zeros.

        mu : float, optional (default: 10)
            Initial penalty parameter for the augmented Lagrangian. Controls the strength
            of the quadratic penalty term. mu=0 is allowed (pure Lagrangian unconstrained problem).
            
        maxmu : float, optional (default: 3e4)
            Maximum allowed penalty parameter. Optimization terminates if mu exceeds this
            value. Increase for hard constraints.
            
        tol : float, optional (default: 1e-5)
            Convergence tolerance for the gradient norm. Optimization terminates when
            ||∇L|| < tol and constraint satisfaction is met.
            
        tolConstraint : float, optional (default: 1e-6)
            Convergence tolerance for constraint violation: ||ConstrObj(x)|| < tolConstraint.
            Should be set based on the accuracy of your implicit function representation.
            For learned manifolds, approximately the mean constraint violation on training samples.
            
        alpha : float, optional (default: 2)
            Multiplication factor for increasing mu when constraints are not satisfied.
            New penalty: mu_new = alpha * mu. Larger values accelerate constraint 
            satisfaction but may cause instability. 
            For ground truth constraints alpha=100 is still safe.
            
        disp : bool, optional (default: True)
            If True, print optimization progress of inner itrations
        
        Returns
        -------
        result : ndarray, shape (resolution + 2, n)
            The computed geodesic path including endpoints.
            result[0] = xA, result[-1] = xB, result[1:-1] are optimized intermediate points.
        
        Notes
        -----
        Algorithm Overview:
        The augmented Lagrangian method alternates between:
        1. Minimizing L(x, λ, μ) = K*Energy(x) - λ^T*φ(x) + (μ/2)||φ(x)||²
        2. Updating multipliers: λ ← λ - μ*φ(x)
        3. Increasing penalty: μ ← α*μ (if constraints not satisfied)
        
        Convergence Criteria:
        - Terminates when both ||∇L|| < tol AND ||φ(x)|| < tolConstraint
        - Or when μ > maxmu (reports constraint violation at termination)
        - Or when maximum outer iterations (200) is reached
        
        Inner Optimization Details:
        - Uses scipy.optimize.minimize with method='BFGS'
        - Analytical gradients provided via DLagrangian method
        
        """
        n = self.n
        dof = resolution #degrees of freedom, number of intermediate points

        K = dof +1
        if xA is None or xB is None:
            print('xA or xB missing')
            return 0
        if x0 is None:
            # use constant path that jumps from xA to xB
            a = resolution // 2
            b = resolution - a
            x0 = np.concatenate([a*[xA], b*[xB]])
            x0 = x0.reshape(x0.shape[0], x0.shape[-1]) 

        if alpha is None : alpha = 100
        if lam is None: lam = np.zeros((self.n-self.d,dof))

        finaltolGrad = tol
        finaltolConstr = tolConstraint
        
        maxIt = 200
        i = 0
        x = np.reshape(x0, self.n*dof)

        args = (lam, mu, K,xA,xB)

        #for tolerances
        normGrad = np.max(np.abs(self.DLagrangian(x,*args)))
        normConstraint = np.linalg.norm(self.ConstrObj(np.reshape(x, (-1,n))))

        hess = None
        method = 'BFGS'
        if mu != 0:
            tolMethod = 1/mu  #tol for gradient
            tolConstraint = 1/mu**(0.1)
        else:
            tolMethod = 0.1
            tolConstraint = 0.1
        
        options = {'gtol':tolMethod, 'disp': disp}
        print('Starting Augmented Lagrangian Optimization for computing geodesic...')
        while (normGrad > finaltolGrad or normConstraint > finaltolConstr) and i<maxIt:

            args = (lam, mu, K,xA,xB)
            normEnergy = np.linalg.norm(self.Lagrangian(x,np.zeros((self.n-self.d,dof)), 0, K, xA = xA, xB = xB))
            normConstraint = np.linalg.norm(self.ConstrObj(np.reshape(x, (-1,n))))

            print('NormEnergyBefore:', normEnergy)
            print('NormConstraintBefore: ', normConstraint)

            optResult = scipy.optimize.minimize(self.Lagrangian, x, args, method = method, jac = self.DLagrangian, 
                                                hess = hess, tol = tolMethod, options = options)
            x = optResult.get('x')

            normGrad = np.max(np.abs(optResult.get('jac')))
            normConstraint = np.linalg.norm(self.ConstrObj(np.reshape(x, (-1,n))))
            normEnergy = np.linalg.norm(self.Lagrangian(x,np.zeros((self.n-self.d,dof)), 0, K, xA = xA, xB = xB))
            
            print('NormEnergy:', normEnergy)
            print('NormGrad: ', normGrad)
            print('NormConstraint', normConstraint)
            
            # Early convergence: if final tolerances are already met,
            # return immediately to avoid wasted penalty-increase iterations.
            if normConstraint < finaltolConstr and normGrad < finaltolGrad:
                result = x.reshape((-1,n))
                result = np.concatenate(([xA],result,[xB]))
                print('Early convergence: final tolerances satisfied')
                return result

            #refine tolerances
            if  normConstraint < tolConstraint:
                if normConstraint < finaltolConstr and normGrad < finaltolGrad:
                    result = x.reshape((-1,n))
                    result = np.concatenate(([xA],result,[xB]))
                    return result
                else:
                    #update multipliers, tighten tolerances
                    print('lam update')
                    lam = lam - np.reshape(mu*self.ConstrObj(np.reshape(x,(-1,n))), (self.n-self.d,dof))
                    tolConstraint = tolConstraint/(mu**0.9)
                    tolMethod = tolMethod/mu
            else:
                #increase penalty parameter, tighten tolerances
                print('increase penalty')
                mu = alpha*mu
                if mu != 0:
                    tolMethod = 1/mu
                    tolConstraint = 1/(mu**(0.1))
                else:
                    tolMethod = 0.1*tolMethod
                    tolConstraint = 0.1*tolConstraint

            options = {'gtol':tolMethod, 'disp': disp, 'return_all':True}
            if mu >= maxmu:
                result = x.reshape((-1,n))
                result = np.concatenate(([xA],result,[xB]))
                print('reached max penalty factor: ', maxmu)
                print('NormGrad: ', normGrad)
                print('NormConstraint', normConstraint)
                return result
            
            i = i+1
        
        result = x.reshape((-1,n))
        result = np.concatenate(([xA],result,[xB]))
        print('Iterations: ', i)
        print('mu: ', mu/alpha)
        print('NormGrad: ', normGrad)
        print('NormConstraint', normConstraint)   
        return result

