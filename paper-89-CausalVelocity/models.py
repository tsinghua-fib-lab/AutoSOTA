import jax 
import jax.numpy as jnp

class parametric_model:
    """
    class for family of velocity models with linear combination of basis functions
    """

    def __init__(self, basis_name = "linear", add_exponential_terms = False, add_fourier_terms = False):
        assert basis_name in ["linear", "quadratic", "cubic", "quartic"], "basis_name must be either 'linear' or 'quadratic'."
        self.basis_name = basis_name
        self.add_exponential_terms = add_exponential_terms
        self.add_fourier_terms = add_fourier_terms
        if basis_name == "linear":
            self.nparams = 3
        elif basis_name == "quadratic":
            self.nparams = 6
        elif basis_name == "cubic":
            self.nparams = 10 
        elif basis_name == "quartic":
            self.nparams = 15 
        if add_exponential_terms:
            self.nparams += 3
        if add_fourier_terms:
            self.nparams += 4

    def basis(self, y, t):
        if y.ndim == 0:
            y = jnp.expand_dims(y, (0,1))
            t = jnp.expand_dims(t, (0,1))
        if y.ndim == 1:
            y = jnp.expand_dims(y, -1)
            t = jnp.expand_dims(t, -1)
        if self.basis_name == "linear":
            basis = jnp.hstack((jnp.ones_like(t), t, y))
        elif self.basis_name == "quadratic":
            basis = jnp.hstack((jnp.ones_like(t), t, y, t**2, y**2, t*y))
        elif self.basis_name == "cubic":
            basis = jnp.hstack((jnp.ones_like(t), t, y, t**2, y**2, t*y, t**3, y**3, t**2*y, t*y**2))
        elif self.basis_name == "quartic":
            basis = jnp.hstack((jnp.ones_like(t), t, y, t**2, y**2, t*y, t**3, y**3, t**2*y, t*y**2, t**4, y**4, t**3*y, t**2*y**2, t * y**3))
        if self.add_exponential_terms:
            basis = jnp.hstack((basis, jnp.exp(-t**2), jnp.exp(-y**2), jnp.exp(-t**2 - y**2)))
        if self.add_fourier_terms:
            basis = jnp.hstack((basis, jnp.sin(t), jnp.cos(t), jnp.sin(y), jnp.cos(y)))
        return basis

    def __call__(self, y, t, params):
        return jnp.dot(self.basis(y, t), params["params_l2"])
    
    def params_init(self, seed = 0, init_weight = 1.0):
        p_init = {
            'params_l2': jax.random.normal(jax.random.PRNGKey(seed), (self.nparams,)) /jnp.sqrt(self.nparams) * init_weight
        }
        return p_init
    
class nn_model:
    def __init__(self, in_features = 1, layers = 2, hidden_size = 32):
        self.in_features = in_features
        self.layers = layers
        self.hidden_size = hidden_size
    
    def forward(self, x, params):
        """
        evaluate the NN unbatched at x with parameters params
        x: float, or (1,) array
        out: (1,) array
        """
        if self.layers >1 :
            h = x
            for i in (range(self.layers-1)):
                weight = params[f"w{i+1}_l2"]
                h = jnp.dot(h, weight) + params[f"b{i+1}"]
                h = jax.nn.tanh(h)
            out = jnp.dot(h, params[f"w{self.layers}_l2"]) + params[f"b{self.layers}"]
        elif self.layers == 1:
            out = jnp.dot(x, params["w1_l2"]) + params["b1"]
        else:
            out = x
        return out.squeeze()
    
    def __call__(self, x, params):
        """
        Batched version of forward.
        """
        if isinstance(x, float):
            x = jnp.array([x])
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        # out = jax.vmap(self.forward, in_axes = (0, None))(x, params)
        return self.forward(x, params).reshape(-1, 1)
    
    def params_init(self, seed = 0, init_weight = 1.0):

        p_init = {'w1_l2': jax.random.normal(shape=(self.in_features, self.hidden_size), key=jax.random.PRNGKey(seed + 0)) * init_weight,
                'b1': jnp.zeros(shape=(self.hidden_size))}

        seed = seed + 1
        
        if self.layers > 1:

            for i in range(self.layers-2):
                p_init[f'w{i+2}_l2'] = jax.random.normal(shape=(self.hidden_size, self.hidden_size), key=jax.random.PRNGKey(seed))/jnp.sqrt(self.hidden_size) * init_weight
                p_init[f'b{i+2}'] = jnp.zeros(shape=(self.hidden_size))
                seed = seed + 1

            p_init[f'w{self.layers}_l2'] = jax.random.normal(shape=(self.hidden_size, 1), key=jax.random.PRNGKey(seed)) * init_weight
            p_init[f'b{self.layers}'] = jnp.zeros(shape=()) 

            seed = seed + 1

        return p_init


class nn_velocity_model(nn_model):
    """
    base class for neural network velocity models 
    """
    def __init__(self, in_features = 2, layers = 2, hidden_size = 32):
        super().__init__(in_features, layers, hidden_size)
    
    def forward(self, y, t, params):
        x = jnp.stack((y,t), axis = -1)
        return super().forward(x, params)

    
    def __call__(self, y, t, params):
        out = self.forward(y, t, params)
        out = out.reshape(-1,)
        return out


class anm_model(nn_model):
    """
    velocity implementation of anm (v(y,x) = f(x))
    """

    def __init__(self, in_features = 1, layers = 2, hidden_size = 32):
        super().__init__(in_features, layers, hidden_size)
        self.in_features = 1

    def forward(self, y, t, params):
        x = t.reshape(-1, 1)      
        return super().forward(x, params)
        

class lsnm_model(anm_model):
    """
    velocity implementation of lsnm 
    """
    def __init__(self, in_features = 1, layers = 2, hidden_size = 32):
        super().__init__(in_features, layers, hidden_size)
        self.in_features = 1

    def params_init(self, seed = 0, init_weight=1):
        p = {}
        p["offset"] = super().params_init(seed = seed, init_weight = init_weight)
        p["scale"] = super().params_init(seed = seed + 444, init_weight = init_weight)
        return p
    
    def __call__(self, y, t, params):
        m = lambda x: self.forward(y, x, params["offset"])
        s = lambda x: self.forward(y, x, params["scale"])

        m_, dm_ = jax.jvp(m, (t,), (jnp.ones_like(t),))
        s_, ds_ = jax.jvp(s, (t,), (jnp.ones_like(t),)) ## assume s is exp(h), then d log s = dh. 

        m_, dm_, s_, ds_ = m_.reshape(-1,), dm_.reshape(-1,), s_.reshape(-1,), ds_.reshape(-1,)

        return dm_ + ds_ * ((y - m_)) 
    
class anm_model(nn_model):
    """
    velocity implementation of anm (v(y,x) = f(x))
    """

    def __init__(self, in_features = 1, layers = 2, hidden_size = 32):
        super().__init__(in_features, layers, hidden_size)
        self.in_features = 1

    def forward(self, y, t, params):
        x = t.reshape(-1, 1)      
        return super().forward(x, params)
    
    def __call__(self, y, t, params):
        out = self.forward(y, t, params)
        out = out.reshape(-1,)
        return out

        

class lsnm_model(anm_model):
    """
    velocity implementation of lsnm 
    """
    def __init__(self, in_features = 1, layers = 2, hidden_size = 32):
        super().__init__(in_features, layers, hidden_size)
        self.in_features = 1

    def params_init(self, seed = 0, init_weight=1):
        p = {}
        p["offset"] = super().params_init(seed = seed, init_weight = init_weight)
        p["scale"] = super().params_init(seed = seed + 444, init_weight = init_weight)
        return p
    
    def __call__(self, y, t, params):
        m = lambda x: self.forward(y, x, params["offset"])
        s = lambda x: self.forward(y, x, params["scale"])

        m_, dm_ = jax.jvp(m, (t,), (jnp.ones_like(t),))
        s_, ds_ = jax.jvp(s, (t,), (jnp.ones_like(t),)) ## assume s is exp(h), then d log s = dh. 

        m_, dm_, s_, ds_ = m_.reshape(-1,), dm_.reshape(-1,), s_.reshape(-1,), ds_.reshape(-1,)

        return dm_ + ds_ * ((y - m_)) 