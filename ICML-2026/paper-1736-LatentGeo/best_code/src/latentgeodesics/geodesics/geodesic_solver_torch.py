import torch
from .metrics import sqdist_euclid_torch

class GeodesicSolverTorch:
    def __init__(self, n, ConstrObj, sqdist = sqdist_euclid_torch):
        """
        Initialize Geodesic Solver that computes geodesics on the approximate zero levelset of ConstrObj.
        
        This PyTorch implementation uses automatic differentiation (autograd) for gradient computation of the constraint
        and the squared distance function. 
        Especially for pullback metrics (e.g., distances computed through decoder networks), this implementation is more efficient 
        as it avoids conversions between PyTorch tensors and NumPy arrays.

        The solver uses an augmented Lagrangian method to find minimum-energy paths constrained to a 
        manifold defined implicitly by ConstrObj(x) ≈ 0.
        
        Parameters
        ----------
        n : int
            Dimension of the ambient space (i.e., latent space dimension).
        
        ConstrObj : callable
            Constraint function that implicitly defines the manifold.
            We do **not** require that for a d-dimensional manifold embedded in n dimmension, the constraint function returns n-d values.
            E.g. it also works for a projection function that returns n values but as approximate rank n-d.
            If you have learned an implicit manifold representation with this repository's LatentProjector class, you can obtain ConstrObj as follows:
            ConstrObj = latentprojector.phi()
            This automatically provides the appropriate constraint function from your trained model.


            Signature: ConstrObj(coord) -> torch.Tensor
            
            Input:
                coord : torch.Tensor of shape (m, n)
                    m points in n-dimensional space (batch of coordinates)
                    Should have requires_grad=True for autograd to work
            
            Output:
                torch.Tensor of shape (n-d, m)
                    Constraint values for each point. The manifold is defined by the 
                    zero levelset where ConstrObj(x) ≈ 0.
                    - (n-d) is the codimension of the manifold
                    - If ConstrObj has full rank, the manifold dimension is d
                    - Each row corresponds to one constraint component
                    - Each column corresponds to one input point

        sqdist : callable, optional (default: sqdist_euclid_torch)
            Squared distance function defining the metric on the ambient space.
            
            Signature: sqdist(x, y) -> torch.Tensor (scalar)
            
            Input:
                x : torch.Tensor of shape (n,) or (1, n)
                    First point
                y : torch.Tensor of shape (n,) or (1, n)
                    Second point
            
            Output:
                torch.Tensor (scalar)
                    Squared distance between x and y
            
            Default (Euclidean):
                def sqdist_euclid_torch(x, y):
                    return torch.norm(y - x)**2
            
            Custom Metric Examples:
                1. Pullback Euclidean (for latent spaces):
                   def pullback_sqdist(x, y):
                       # Distance in data space via decoder
                       return torch.norm(decoder(y) - decoder(x))**2
                
                2. KL-Divergence induced metric (for probability distributions):
                   def kl_sqdist(x, y):
                       # Assuming decoder outputs distribution parameters
                       p = decoder(x)
                       q = decoder(y)
                       return kl_divergence(p, q)

        """
        #constraints must handle input as mxn, m batch dimension
        self.ConstrObj = ConstrObj # output (n-d,m)
        self.sqdist = sqdist
        
        self.n = n
        self.d = n - self.ConstrObj(torch.zeros((1,self.n))).shape[0] #image domain of constraint is n-d


    def Lagrangian(self, coord, lam, mu, K, xA = None, xB = None):

        coord = coord.reshape((-1, self.n))
        valPhi = self.ConstrObj(coord)
        coord = torch.cat(([xA.unsqueeze(0),coord,xB.unsqueeze(0)]), dim=0)

        energy = torch.sum(torch.stack([self.sqdist(coord[s - 1], coord[s]) for s in range(1, coord.shape[0])]))
        constraint = torch.sum(torch.einsum('ij,ij->i', lam[:self.n-self.d, :], valPhi[:self.n-self.d, :]))

        L = K*energy - constraint + 0.5*mu*(torch.norm(valPhi))**2 
      
        return L

    def AugLagrangeMinimize(self, resolution, xA, xB, x0 = None, lam = None, mu = 10, maxmu = 3e4,
                            tol = 1e-5, tolConstraint = 1e-6, alpha = 2, max_it_solver=1000, 
                            tol_change_solver=1e-5, lr=1.,disp=True):

        """
        Compute a geodesic path between two points using augmented Lagrangian optimization.
        
        This method finds the minimum-energy path constrained to the manifold by
        solving an augmented Lagrangian problem. The inner optimization uses torch.optim.LBFGS
        with automatic differentiation.
        Note: When using learned representation functions the gradient may not reach very small values 
        due to inherent approximation errors in the learned manifold.
        
        Parameters
        ----------
        resolution : int
            Number of intermediate points along the geodesic path (excluding endpoints).
            
        xA : torch.Tensor or array-like, shape (n,)
            Starting point of the geodesic. Must lie on (or near) the manifold.
            
        xB : torch.Tensor or array-like, shape (n,)
            Ending point of the geodesic. Must lie on (or near) the manifold.
            
        x0 : torch.Tensor or array-like, shape (resolution, n), optional
            Initial guess for intermediate points. If None (default), initializes with
            a piecewise constant path that jumps from xA to xB at the midpoint.
            Good initialization can significantly speed up convergence.
            
        lam : torch.Tensor or array-like, shape (n-d, resolution), optional
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
            For learned manifolds approximately the mean constraint violation on training samples.
            
        alpha : float, optional (default: 2)
            Multiplication factor for increasing mu when constraints are not satisfied.
            New penalty: mu_new = alpha * mu. Larger values accelerate constraint 
            satisfaction but may cause instability. 
            For ground truth constraints alpha=100 is still safe.
            
        max_it_solver : int, optional (default: 1000)
            Maximum number of iterations for the inner LBFGS optimizer per outer iteration.
            
        tol_change_solver : float, optional (default: 1e-5)
            Tolerance for relative change in the objective function for LBFGS termination.
            
        lr : float, optional (default: 1.0)
            Learning rate for the LBFGS optimizer. Usually 1.0 is appropriate; reduce if
            optimization is unstable.
            
        disp : bool, optional (default: True)
            Only for API compatibility, not used in this implementation.
        
        Returns
        -------
        result : torch.Tensor, shape (resolution + 2, n)
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
            x0 = torch.cat([xA.repeat(a, 1), xB.repeat(b, 1)], dim=0)
            x0 = x0.view(x0.shape[0], x0.shape[-1]) 

        
        if alpha is None : alpha = 100
        if lam is None: lam = torch.zeros((self.n-self.d,dof), dtype=x0.dtype)
 
        finaltolGrad = tol
        finaltolConstr = tolConstraint
        
        maxIt = 200
        i = 0
        x = x0.view((self.n*dof))
        
        #for tolerances
        x_eval = x.clone().detach().requires_grad_(True)
        loss = self.Lagrangian(x_eval, lam, mu, K, xA, xB)
        grad, = torch.autograd.grad(loss, x_eval)
        normGrad = grad.abs().max()
        with torch.no_grad():
            normConstraint = torch.norm(self.ConstrObj(torch.reshape(x_eval, (-1,n))))

        if mu != 0:
            tolMethod = 1/mu  #tol for gradient
            tolConstraint = 1/mu**(0.1)
        else:
            tolMethod = 1e-6
            tolConstraint = 0.1
        
        while (normGrad > finaltolGrad or normConstraint > finaltolConstr) and i<maxIt:

            normEnergy = torch.norm(self.Lagrangian(x,torch.zeros((self.n-self.d,dof),dtype=xA.dtype), 0, K, xA = xA, xB = xB))
            normConstraint = torch.norm(self.ConstrObj(torch.reshape(x, (-1,n))))
            
            print('NormGrad: ', normGrad.item())
            print('NormEnergyBefore:', normEnergy.item())
            print('NormConstraintBefore: ', normConstraint.item())
            x = x.clone().detach().requires_grad_(True) 

            optimizer = torch.optim.LBFGS([x], lr=lr, max_iter=max_it_solver, tolerance_grad=tolMethod, tolerance_change= tol_change_solver)
            def objective():
                optimizer.zero_grad()
                L = self.Lagrangian(x, lam, mu, K, xA, xB)
                L.backward()
                return L

            optimizer.step(objective)
            x = x.detach()

            x_eval = x.clone().detach().requires_grad_(True)
            loss = self.Lagrangian(x_eval, lam, mu, K, xA, xB)
            grad, = torch.autograd.grad(loss, x_eval)
            normGrad = grad.abs().max()
            with torch.no_grad():
                normConstraint = torch.norm(self.ConstrObj(torch.reshape(x_eval, (-1,n))))
                normEnergy = torch.norm(self.Lagrangian(x_eval,torch.zeros((self.n-self.d,dof), dtype=xA.dtype), 0, K, xA = xA, xB = xB))

            print('NormEnergy:', normEnergy.item())
            print('NormGrad: ', normGrad.item())
            print('NormConstraint', normConstraint.item())

            #refine tolerances
            if  normConstraint < tolConstraint:
                if normConstraint < finaltolConstr and normGrad < finaltolGrad:
                    result = x.reshape((-1,n))
                    result  = torch.cat(([xA.unsqueeze(0),result,xB.unsqueeze(0)]), dim=0)
                    return result
                else:
                    #update multipliers, tighten tolerances
                    print('lam update')
                    with torch.no_grad():
                        lam = lam - torch.reshape(mu*self.ConstrObj(torch.reshape(x,(-1,n))), (self.n-self.d,dof))
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

            if mu >= maxmu:
                result = x.reshape((-1,n))
                result  = torch.cat(([xA.unsqueeze(0),result,xB.unsqueeze(0)]), dim=0)
                print('reached max penalty factor: ', maxmu)
                print('NormGrad: ', normGrad.item())
                print('NormConstraint', normConstraint.item())
                return result
            i = i+1
        
        result = x.reshape((-1,n))
        result  = torch.cat(([xA.unsqueeze(0),result,xB.unsqueeze(0)]), dim=0)
        print('Iterations: ', i)
        print('mu: ', mu/alpha)
        print('NormGrad: ', normGrad.item())
        print('NormConstraint', normConstraint.item())   
        return result
