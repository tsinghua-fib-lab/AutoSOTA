import jax
import jax.numpy as jnp
import numpy as onp
import argparse
from jax.scipy.stats.norm import ppf as norm_ppf
from jax.scipy.stats.norm import cdf as norm_cdf
from jax.scipy.stats.norm import pdf as norm_pdf 
from jax.scipy.stats.laplace import pdf as laplace_pdf
from utils import standardize_data
from tmi import tmi_nn


from mechanisms import flow_from_mechanism, mechanism_from_flow


def parse_arguments():
    parser = argparse.ArgumentParser(description='Synthetic data generation')
    parser.add_argument('--seed', type=int, default=0, help='seed for data generation')
    parser.add_argument('--n_data', type=int, default=2000, help='number of data points')
    parser.add_argument('--weight', type=float, default=0.2, help='initialization weight for neural network')
    parser.add_argument('--layers', type=int, default=2, help='number of layers for neural network')
    parser.add_argument('--hidden_size', type=int, default=64, help='hidden size for neural network')
    parser.add_argument('--noise', type=str, default="normal", help='noise distribution')
    parser.add_argument('--noise_scale', type=float, default=1, help='noise scale for data generation')
    parser.add_argument('--mech', type=str, default="lsnm", help='mechanism')
    parser.add_argument('--name', type=str, default="lsnm", help='output name of the data file')
    parser.add_argument('--metafile', type=str, default="pairmeta.txt", help='meta file for data')
    parser.add_argument('--compute_score', action='store_true', help='compute score or not')
    parser.add_argument('--tmi_tx', action='store_true', help='use TMI transformation on x data')
    return parser.parse_args()

if __name__ == "__main__":

    args = parse_arguments()
    DATA_SEED = args.seed
    N_DATA = args.n_data
    WEIGHT = args.weight
    LAYERS = args.layers
    HIDDEN_SIZE = args.hidden_size
    NOISE = args.noise
    NOISE_SCALE = args.noise_scale
    MECH = args.mech
    NAME = args.name
    METAFILE = args.metafile
    SCORE = args.compute_score
    TMI_TX = args.tmi_tx

    def score_oracle(x,y, model):
        """
        Compute the score at (x,y) using the model
        """
        e = model.inverse_normalized(x, y)
        pdf_e = lambda _: norm_pdf(_/NOISE_SCALE)/NOISE_SCALE

        
        df = jax.vmap(jax.grad(model.inverse_normalized, (0, 1))) ## returns tuple (df-1/dx, df-1/dy)
        dfdy = jax.grad(model.inverse_normalized, 1)
        dfdxdy = jax.vmap(jax.grad(dfdy, (0,1))) ## returns tuple (d^2f-1/dxdy, d^2f-1/dy^2)

        dx_data, dy_data = df(x, y)
        dxdy_data, dydy_data = dfdxdy(x, y)

        sx = -x_data ## std. gaussian score 
        sxyx = -x_data + (-e)*(dx_data) + dxdy_data/dy_data
        sxyy = - e * dy_data + dydy_data/dy_data

        sxy = jnp.stack((sxyx, sxyy), axis = -1)

        ### marginal expressions to compute sy 

        def marginal_y(y_):
            e = jax.vmap(model.inverse_normalized, (0, None))(x, y_)
            dfdy_data = jnp.abs(jax.vmap(jax.grad(model.inverse_normalized, 1), (0, None))(x, y_))
            return (pdf_e(e)*dfdy_data).mean(axis = 0).squeeze()
        
        py = jax.vmap(marginal_y)(y)
        sy = jax.vmap(jax.grad(lambda _: jnp.log(marginal_y(_))))(y)

        return sx, sy, sxy
    

    class nn_cond_mechanism(flow_from_mechanism):

        def __init__(self, mech = "anm", **kwargs):
            super().__init__()
            self.mech = mech
            if mech == "anm":
                n_out = 1
            elif mech == "lsnm":
                n_out = 2
            elif mech == "periodic":
                n_out = 4
            elif mech == "sigmoid":
                n_out = 4
            self.build_nn_params(n_out, **kwargs)

        def build_nn_params(self, n_out, layers = 3, hidden_size = 32, seed = 0):
            params = {"w1": jax.random.normal(shape=(1, hidden_size), key=jax.random.PRNGKey(seed))*WEIGHT,
                        "b1": jax.random.normal(shape=(hidden_size), key=jax.random.PRNGKey(seed + 1))*WEIGHT}
            for i in range(layers - 2):
                params[f"w{i+2}"] = jax.random.normal(shape=(hidden_size, hidden_size), key=jax.random.PRNGKey(seed))*WEIGHT
                params[f"b{i+2}"] = jax.random.normal(shape=(hidden_size), key=jax.random.PRNGKey(seed + 1))*WEIGHT
                seed = seed + 2
            params[f"w{layers}"] = jax.random.normal(shape=(hidden_size, n_out), key=jax.random.PRNGKey(seed))*WEIGHT
            params[f"b{layers}"] = jax.random.normal(shape=(n_out), key=jax.random.PRNGKey(seed + 1))*WEIGHT
            self.params = params
            self.layers = layers

        def nn_conditioner(self, x):
            if x.ndim == 1:
                x = x[:, None]
            h = x
            for i in range(self.layers-1):
                h = jnp.dot(h, self.params[f"w{i+1}"]) + self.params[f"b{i+1}"]
                h = jax.nn.tanh(h)
            out = jnp.dot(h, self.params[f"w{self.layers}"]) + self.params[f"b{self.layers}"]
            return out

        def forward(self, x, e):
            """
            compute y = f_{x}(e)
            """
            cond = self.nn_conditioner(x)
            if self.mech == "anm":
                return (e + cond.squeeze()).squeeze()
            elif self.mech == "lsnm":
                mu, sigma = cond[:, 0], jnp.exp(-cond[:, 1]**2)
                return (mu + (sigma+0.2) * e).squeeze()
            elif self.mech == "periodic":
                a, b, c, d= cond[:, 0], jnp.exp(-cond[:, 1] ** 2), jnp.exp(-cond[:, 2] ** 2), jnp.exp(-cond[:, 3] ** 2)
                return (a + b*jnp.sin(c*e)).squeeze()
            elif self.mech == "sigmoid":
                a, b, c, d = cond[:, 0], jnp.exp(-cond[:, 1]**2), cond[:, 2], jnp.exp(-cond[:, 3]**2)
                sigmoid = jax.nn.sigmoid(a + b * e)
                return (c + norm_ppf(sigmoid)*d).squeeze()
            
        def inverse(self, x, y):
            """
            compute e = f_{x}^{-1}(y)
            """
            cond = self.nn_conditioner(x)
            if self.mech == "anm":
                return (y - cond.squeeze()).squeeze()
            elif self.mech == "lsnm":
                mu, sigma = cond[:, 0], jnp.exp(-cond[:, 1]**2)
                return ((y - mu) / (sigma + 0.2)).squeeze()
            elif self.mech == "periodic":
                raise NotImplementedError
            elif self.mech == "sigmoid":
                a, b, c, d = cond[:, 0], jnp.exp(-cond[:, 1]**2), cond[:, 2], jnp.exp(-cond[:, 3]**2)
                sigmoid = norm_cdf((y-c)/d) 
                invsigmoid = -jnp.log(1/sigmoid - 1) 
                return ((invsigmoid - a)/b).squeeze()
            
    class velocity_mechanism(mechanism_from_flow):
        def __init__(self, seed = 0):
            super().__init__()
            self.params = {
                "a": jax.random.normal(jax.random.PRNGKey(seed+1))*WEIGHT,
                "b": jax.random.normal(jax.random.PRNGKey(seed+1))*WEIGHT,
                "c": jax.random.normal(jax.random.PRNGKey(seed+1))*WEIGHT,
                "d": jax.random.normal(jax.random.PRNGKey(seed+1))*WEIGHT,
                "e": jax.random.normal(jax.random.PRNGKey(seed+1))*WEIGHT,
                "f": jax.random.normal(jax.random.PRNGKey(seed+1))*WEIGHT
            }

        def velocity(self, y, x):
            """
            compute v(y,x)
            """
            a, b, c, d, e, f = self.params["a"], self.params["b"], self.params["c"], self.params["d"], self.params["e"], self.params["f"]
            #return a + b * y + c * x  + d * jnp.exp(-x**2) + e * jnp.exp(-y**2) + f * jnp.exp(-(x-y)**2)
            return a + b * jnp.sin(x) + c * jnp.sin(y) + d * jnp.cos(x) + e * jnp.cos(y) + f * jnp.sin(x+y)
    if MECH == "velocity":
        m = velocity_mechanism(seed = DATA_SEED)
    else:
        m = nn_cond_mechanism(mech = MECH, layers = LAYERS, hidden_size = HIDDEN_SIZE, seed = DATA_SEED)
    if NOISE == "normal":
        x_data = jax.random.normal(shape = (N_DATA,), key = jax.random.PRNGKey(DATA_SEED+22222))
        noise = jax.random.normal(shape = (N_DATA,), key = jax.random.PRNGKey(DATA_SEED+33333))
        if TMI_TX: 
            tmi_x = tmi_nn(d = 1, layers = 3, hidden_size = 64)
            tmi_x.params_init(init_weight = 0.3, seed = DATA_SEED+55555)
            x_data = tmi_x(x_data) 
            x_data = x_data.reshape(-1)
            tmi_e = tmi_nn(d = 1, layers = 3, hidden_size = 64)
            tmi_e.params_init(init_weight = 0.3, seed = DATA_SEED+66666)
            noise = tmi_e(noise)
            noise = noise.reshape(-1)
        noise = noise * NOISE_SCALE
    elif NOISE == "laplace":
        raise NotImplementedError
        # x_data = jax.random.laplace(shape = (N_DATA,), key = jax.random.PRNGKey(DATA_SEED+22222))
        # noise = jax.random.laplace(shape = (N_DATA,), key = jax.random.PRNGKey(DATA_SEED+33333)) * NOISE_SCALE
    elif NOISE == "uniform":
        raise NotImplementedError 
        # x_data = jax.random.uniform(shape = (N_DATA,), key = jax.random.PRNGKey(DATA_SEED+22222), minval = -5, maxval = 5)
        # noise = jax.random.uniform(shape = (N_DATA,), key = jax.random.PRNGKey(DATA_SEED+33333), minval = -5, maxval = 5) * NOISE_SCALE
    y_data = m.forward(x_data, noise)

    dat = jnp.concat((x_data[:, None], y_data[:, None]), axis = -1)

    dat, dat_mean, dat_std = standardize_data(dat, return_statistics = True, trim_outliers = 0) ## normalize data before computing the true score 
    x_data, x_data_mean, x_data_std = dat[:,0], dat_mean[0], dat_std[0]
    y_data, y_data_mean, y_data_std = dat[:,1], dat_mean[1], dat_std[1]

    m.set_normalization_statistics(x_data_mean, x_data_std, y_data_mean, y_data_std)

    cause = 1
    effect = 2




    flip = jax.random.uniform(shape = (), key = jax.random.PRNGKey(DATA_SEED+44444)) > 0.5
    if flip:
        dat = dat[:, [1, 0]]
        cause = 2
        effect = 1

    dat = onp.array(dat)

    if SCORE: 
        s_x, s_y, s_xy = score_oracle(x_data, y_data, m) ## x is always the cause. 
        if flip:
            s_xy = s_xy[:, [1, 0]]
        s_cause = onp.array(s_x)
        s_effect = onp.array(s_y)
        s_joint = onp.array(s_xy)
        onp.savetxt(f"data/{NAME}/s_cause{DATA_SEED:04d}.txt", s_cause)
        onp.savetxt(f"data/{NAME}/s_effect{DATA_SEED:04d}.txt", s_effect)
        onp.savetxt(f"data/{NAME}/s_joint{DATA_SEED:04d}.txt", s_joint)

    onp.savetxt(f"data/{NAME}/pair{DATA_SEED:04d}.txt", dat)

    with open(f"data/{NAME}/{METAFILE}", "a") as f:
        f.write(f"{DATA_SEED:04d} {cause} {cause} {effect} {effect} 1 \n")
    
