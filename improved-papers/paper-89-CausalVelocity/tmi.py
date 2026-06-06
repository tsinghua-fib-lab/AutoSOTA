from numpy.polynomial.legendre import leggauss
from models import nn_model
import jax 
import jax.numpy as jnp

class softplus_nn(nn_model):

    def forward(self, x, params):
        return jax.nn.softplus(super(softplus_nn, self).forward(x, params))
    
class zero_nn(nn_model):
    def forward(self, x, params):
        return 0.
    
    def __call__(self, x, params):
        return jnp.zeros((x.shape[0], 1))

class tmi_nn:
    def __init__(self, d, seed = 0, layers = 2, hidden_size = 32):
        self.d = d
        self.nns_f = []
        self.nns_h = [] 
        for i in range(d):
            if i == 0:
                self.nns_f.append(zero_nn())
                nn_h = softplus_nn(in_features = 1, layers = layers, hidden_size = hidden_size)
                self.nns_h.append(nn_h)
            else:
                nn_f = nn_model(in_features = i, layers = layers, hidden_size = hidden_size)
                self.nns_f.append(nn_f)
                nn_h = softplus_nn(in_features = i+1, layers = layers, hidden_size = hidden_size)
                self.nns_h.append(nn_h)

    def params_init(self, init_weight = 1.0, seed = 0):
        self.params = {}
        for i, nn in enumerate(self.nns_f):
            if nn != "zero":
                self.params[f"f_{i}"] = nn.params_init(init_weight = init_weight, seed = seed + i)
        for i, nn in enumerate(self.nns_h):
            self.params[f"h_{i}"] = nn.params_init(init_weight = init_weight, seed = seed + i + 22222)
        return self.params
    
    def forward(self, x, params):
        """
        Unbatched forward pass
        """        
        def t_i(x, i):
            vi = lambda _: self.nns_h[i].forward(_, params[f"h_{i}"])
            fi = self.nns_f[i].forward(x[:i], params[f"f_{i}"])
            ei = fi + self.integrate_leggauss(vi, x[i], cond = x[:i])
            return ei.squeeze()  
        
        t = jnp.array([t_i(x, i) for i in range(self.d)])
        
        return t

    def __call__(self, x):
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        def t_i(x, i):
            vi = lambda _: self.nns_h[i](_, self.params[f"h_{i}"])
            fi = self.nns_f[i](x[:,:i], self.params[f"f_{i}"])
            ei = fi + self.integrate_leggauss_batched(vi, x[:,i], cond = x[:, :i])
            return ei

        t = jnp.hstack([t_i(x, i) for i in range(self.d)])
        
        return t

    def eval(self, x):
        return self.forward(x, self.params)
    
    def eval_h(self, x):
        return jnp.array([self.nns_h[i].forward(x[:(i+1)], self.params[f"h_{i}"]) for i in range(self.d)])
    
    def eval_f(self, x):
        return jnp.array([self.nns_f[i].forward(x[:i], self.params[f"f_{i}"]) for i in range(self.d)])
    
    def logjac(self, x, params, analytic = True):
        """
        Computes the log determinant of the Jacobian of the triangular transformation, averaged over the batch.
        Analytic: if true, then evaluate the Jacobian of each component analytically (i.e., differentiating through the integral sign)
        Otherwise, autodiff through the numerically integrated output. 
        """
        out_list = []
        for i in range(self.d):
            if analytic:
                out_list.append(jnp.log(self.nns_h[i](x[:,:(i+1)], params[f"h_{i}"])))
            else:
                vi = lambda _: self.nns_h[i](_, params[f"h_{i}"])
                if i == 0:
                    int_i = lambda _: self.integrate_leggauss_batched(vi, _)
                    _, grad = jax.jvp(int_i, (x[:, i],), (jnp.ones_like(x[:, i]),))
                else:
                    int_i = lambda _: self.integrate_leggauss_batched(vi, _, cond = x[:, :i])
                    _, grad = jax.jvp(int_i, (x[:, i],), (jnp.ones_like(x[:, i]),)) 
                out_list.append(jnp.log(grad))        
        out_list = jnp.array(out_list).reshape(x.shape[0], self.d)
        return jnp.mean(out_list, axis = 0).sum(), out_list
    
    def logprob(self, x):
        e = self.eval(x)
        gauss_logprobs = -e**2 / 2 
        jacs = jnp.log(self.eval_h(x))
        return ((gauss_logprobs) + jacs).sum()
        
    def integrate_leggauss(self, f, x, cond, degree = 100):
        """
        Gauss-Legendre quadrature integrating f from 0 to x. 
        x: (1,) array
        f: function to integrate
        degree: number of quadrature points
        cond: additional conditioning variables for f (will be concatenated to x and can be empty)
        """

        t, w = leggauss(degree)
        t_rescaled = (x/2 * t + x/2)
        w_rescaled = (x/2 * w)
        if len(cond) == 0:
            x_eval = t_rescaled
        else:
            cond = jnp.repeat(cond.reshape(1, -1), degree, axis = 0)
            x_eval = jnp.hstack([cond, t_rescaled.reshape(-1, 1)])
        ft = jax.vmap(f)(x_eval)
        sol = (ft * w_rescaled).sum()
        return jnp.array([sol])


    def integrate_leggauss_batched(self, f, x, cond, degree = 100):
        
        x = x.reshape(-1, 1, 1)

        n = x.shape[0]

        t, w = leggauss(degree)

        t, w = jnp.array(t).reshape(-1, 1), jnp.array(w).reshape(-1, 1)

        t_rep = jnp.tile(t, (n, 1)).reshape(n, degree, 1) ## n x deg x 1

        w_rep = jnp.tile(w, (n, 1)).reshape(n, degree, 1) ## n x deg x 1 

        t_rescaled = (x/2 * t_rep + x/2) 

        w_rescaled = (x/2 * w_rep) 

        cond = jnp.hstack([jnp.repeat(cond, degree, axis = 0), t_rescaled.reshape(-1, 1)]) ## n * degree x (d+1)

        ft = f(cond) ## n * deg x 1

        ft = ft.reshape(n, degree, 1) ## back to n x deg x 1

        sol = (ft * w_rescaled).sum(axis = 1) ## n x 1

        return sol
    
if __name__ == "__main__":
    x_data = jax.random.normal(shape = (1000,), key = jax.random.PRNGKey(22222))
    tmi = tmi_nn(d = 1, seed = 0)
    tmi.params_init(init_weight = 1.0)
    import matplotlib.pyplot as plt 
    x_data = tmi(x_data)
    x_data = x_data - x_data.mean()
    x_data = x_data / x_data.std()

    plt.hist(x_data, bins = 25, density = True, alpha = 0.5, label = "TMI output")
    plt.show()


