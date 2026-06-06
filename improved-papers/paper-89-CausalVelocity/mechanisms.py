import jax
import jax.numpy as jnp

from diffrax import diffeqsolve, ODETerm, Tsit5, SaveAt, PIDController, ConstantStepSize, Kvaerno3

class flow_from_mechanism:
    """
    Base class to obtain flow/velocity field from a given bijective mechanism with known inverse.
    """

    def __init__(self):
        self.has_normalized = False

    def forward(self,x,e):
        """
        compute y = f_{x}(e)
        """
        raise NotImplementedError("Forward map not implemented for this mechanism.")

    def inverse(self,x,y):
        """
        compute e = f^{-1}_x(y) 
        """
        raise NotImplementedError("Inverse map not implemented for this mechanism.")
    
    def forward_normalized(self, x, e):
        """
        compute y = f_{x}(e) on original scale then return normalized y
        """
        assert self.has_normalized, "Normalization statistics not set."
        return (self.forward((x*self.x_std + self.x_mean), e) - self.y_mean)/self.y_std
    
    def inverse_normalized(self, x, y):
        """
        compute e = f^{-1}_x(y) on original scale then return normalized e
        """
        assert self.has_normalized, "Normalization statistics not set."
        return (self.inverse((x*self.x_std + self.x_mean), (y*self.y_std + self.y_mean)))
    
    def flow_map_normalized(self, x, x0, y0):
        """
        compute phi_{x, x'}(y) = f_{x'}\circ f_{x}^{-1}(y) on original scale then return normalized y
        """
        e = self.inverse_normalized(x0, y0)
        return self.forward_normalized(x, e)
    
    def flow_normalized(self, x, x0, y0):
        return self.flow_map_normalized(x, x0, y0)
    
    def velocity_normalized(self, y, x):
        """
        compute v(y,x) = d/dx' phi_{x', x}(y) | x' = x
        """
        assert self.has_normalized, "Normalization statistics not set."
        return self.velocity((y*self.y_std), (x*self.x_std)) * self.x_std / self.y_std  
    
    def flow_map(self, x, x0, y0):
        """
        compute phi_{x, x'}(y) = f_{x'}\circ f_{x}^{-1}(y)
        """
        e = self.inverse(x0, y0)
        return self.forward(x, e)

    def flow(self, times, x0, y0):
        """
        evaluate flow map phi_{x, x'}(y) = f_{x'}\circ f_{x}^{-1}(y)
        """
        return self.flow_map(times, x0, y0)
    
    def velocity(self, y, x):
        """
        compute v(y,x) = d/dx' phi_{x', x}(y) | x' = x
        """
        ddx = jax.grad(self.flow_map)
        return ddx(x, x, y) 
    
    def set_normalization_statistics(self, x_mean, x_std, y_mean, y_std):
        """
        set normalization statistics
        """
        self.x_mean = x_mean
        self.x_std = x_std
        self.y_mean = y_mean
        self.y_std = y_std
        self.has_normalized = True

def fwd_ode_solve(times, x0, y0, term, solver, stepsize_controller):
    """
    Solve ODE forward from x0 to times, where times are not necessarily increasing.
    """
    fwd_argsort = jnp.argsort(times)
    fwd_times = times[fwd_argsort]
    sol = diffeqsolve(term, solver, t0 = x0, t1 = fwd_times[-1], dt0 = None, y0 = y0,
                    saveat = SaveAt(ts = fwd_times), stepsize_controller=stepsize_controller, max_steps = 10000)
    y_fwd = sol.ys
    y_fwd = y_fwd[jnp.argsort(fwd_argsort)]
    return y_fwd

def bwd_ode_solve(times, x0, y0, term, solver, stepsize_controller):
    """
    Solve ODE backwards from x0 to times, where times are not necessarily decreasing.
    """
    bwd_argsort = jnp.argsort(times)
    bwd_times = jnp.flip(times[bwd_argsort])
    sol = diffeqsolve(term, solver, t0 = x0, t1 = bwd_times[-1], dt0 = None, y0 = y0,
                    saveat = SaveAt(ts = bwd_times), stepsize_controller=stepsize_controller, max_steps = 10000)
    y_bwd = sol.ys
    y_bwd = jnp.flip(y_bwd, axis = 0)[jnp.argsort(bwd_argsort)]
    return y_bwd    


class mechanism_from_flow(flow_from_mechanism):
    """
    Base class to obtain flow map from a given velocity field using numerical integration. 
    """

    def __init__(self, stiff = False):
        """
        define ode solver parameters here
        stiff: parameter for whether a stiff ode integrator should be used 
        """
        super().__init__()
        if stiff:
            self.solver=Kvaerno3()
            self.stepsize_controller = PIDController(rtol=1e-3, atol=1e-3)
        else:
            self.solver = Tsit5()
            self.stepsize_controller = PIDController(rtol=1e-5, atol=1e-5)

    def forward(self, x, e):
        """
        compute y = f_{x}(e)
        """
        out = self.flow(x, 0., e)
        return jnp.diag(out).squeeze()

    def inverse(self, x, y):
        """
        compute e = f^{-1}_x(y) 
        """
        out = self.flow(jnp.array([0.]), x, y)
        return jnp.diag(out).squeeze()
    
    def flow(self, times, x0, y0):
        """
        evaluate flow map phi_{x, x'}(y) = f_{x'}\circ f_{x}^{-1}(y)
        """
        v = lambda t, y, args : self.velocity(jnp.array(y), jnp.array(t)).squeeze()
        term = ODETerm(v)
        times = jnp.array(times).reshape(-1,)
        sol = self.integrate(term, times, x0, y0)
        return sol
    
    def velocity(self, y, x):
        """
        compute v(y,x)
        """
        raise NotImplementedError("Velocity field needs to be implemented for this mechanism.")
    
    def flow_normalized(self, times, x0, y0):
        """
        evaluate flow map phi_{x, x'}(y) = f_{x'}\circ f_{x}^{-1}(y)
        """
        v = lambda t, y, args : self.velocity_normalized(jnp.array(y), jnp.array(t)).squeeze()
        term = ODETerm(v)
        sol = self.integrate(term, times, x0, y0)
        return sol
    
    def integrate(self, term, times, x0, y0):
        """
        Integrate from x_0 to x2 handling non monotonicity
        """
        out = jnp.zeros((len(times),))
        ## forward times
        fwd_times = jnp.where(times > x0, times, x0) ## for x2 > x0, integrate from x0 to x2. Otherwise, just return y0. 
        y_fwd = fwd_ode_solve(fwd_times, x0, y0, term, self.solver, self.stepsize_controller)
        y_fwd = jnp.where(jnp.isinf(y_fwd), y0, y_fwd) ## if y_fwd is inf, x_0 is the maximum time, so just return y0.
        out = out + y_fwd ## gives y_t at x2 > xi and y0 at x2 <= xi
        ## backward times
        bwd_times = jnp.where(times < x0, times, x0)
        y_bwd = bwd_ode_solve(bwd_times, x0, y0, term, self.solver, self.stepsize_controller)
        y_bwd = jnp.where(jnp.isinf(y_bwd), y0, y_bwd) ## if y_bwd is inf, x_0 is the minimum time, so set to y0.
        out = out + y_bwd ## gives y_t at x2 < xi and y0 at x2 >= xi
        ## equal times
        out = out - y0 ## for x2 == xi, output is 2 * y0, everywhere else is y_t + y0 (forward integration + backward y0 or vice versa).
        return out
