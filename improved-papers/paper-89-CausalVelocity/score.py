import jax.numpy as jnp
import jax
import optax

from kernel import gauss_kernel_gram, median_heuristic, exp_kernel_gram, silverman, silverman_log, silverman_sqrt, eb_mx

@jax.jit
def jnp_xdiff_(x,z):
    """
    For computing pairwise differences (not their norm) in the stein score.
    (x_i): n_1 x d
    (z_i): n_2 x d 
    out: x_i - z_j (n_1 x n_2 x d)
    """
    return x[:, None] - z # n_1 x n_2 x d broadcasting

def stein_score(x, reg = 0.1, score_kernel = "gauss", bandwidth_factor = 1):
    """
    Computes the stein gradient estimator of the score function by solving
    ridge regression with regularization parameter reg. 
    Choice of kernel: "gauss" or "exp".
    """
    assert score_kernel in ["gauss", "exp"], "kernel must be either 'gauss' or 'exp'."
    n_train = len(x)
    mx = median_heuristic(x) * bandwidth_factor
    if score_kernel == "gauss":
        kx = gauss_kernel_gram(x, x, lengthscale = mx, scale = 1)
        nabla_kx = -jnp.einsum("kij, ik -> kj", jnp_xdiff_(x,x), kx) / (mx**2)
    elif score_kernel == "exp":
        kx = exp_kernel_gram(x, x, lengthscale = mx, scale = 1)
        xdiff = jnp_xdiff_(x,x) # n x n x d
        xdiff_norm = jnp.linalg.norm(xdiff, axis = -1, keepdims = True) # n x n x 1
        nabla_kx = -jnp.einsum("kij, ik -> kj", jnp.where(xdiff_norm!= 0, xdiff / xdiff_norm, 0), kx) / (2 * mx)
    kx_inv = jnp.linalg.inv(kx + reg * jnp.eye(n_train))
    sx = jnp.matmul(kx_inv, nabla_kx)
    return sx

def kde(x, reg = 0.01, score_kernel = "gauss", x_eval = None, bandwidth_factor = 1, eb = True):
    """
    KDE with the silverman rule for bandwidth selection.
    The score is estimated with the Empirical Bayes regularization with parameter reg (smoothing out low-density regions). 
    Returns both the KDE estimate and the score estimate.
    """
    assert score_kernel in ["gauss", "exp"], "kernel must be either 'gauss' or 'exp'."
    if eb: 
        mx = eb_mx(x) * bandwidth_factor
    else:
        mx = silverman(x) * bandwidth_factor
        # mx = median_heuristic(x) * bandwidth_factor
    reg = x.shape[0]**(-2) ## Wibisino Theorem 1
    if x_eval is None:
        x_eval = x
    if score_kernel == "gauss":
        kx = gauss_kernel_gram(x_eval, x, lengthscale = mx, scale = 1)
        kde_ = jnp.mean(kx, axis = -1)/jnp.prod(mx)**0.5 # n_test of kde likelihoods
        nabla = jnp.mean(- (jnp_xdiff_(x_eval, x) /  (mx ** 3)) * jnp.expand_dims(kx, -1), axis = -2) # n_test x n_train x d -> n_test x d
        sx = nabla / jnp.expand_dims(jnp.maximum(kde_, reg), -1)
        kde_ = kde_/(2 * jnp.pi)**0.5 ## normalizing constant if returning kde 
        #sx_test = (- jnp.expand_dims(kx_test, -1) * jnp_xdiff_(x_test, x_train)/mx**2).mean(axis = -2) / mx  
    elif score_kernel == "exp":
        kx = exp_kernel_gram(x_eval, x, lengthscale = mx, scale = 1)
        kde_ = jnp.mean(kx, axis = -1)/jnp.prod(mx)**0.5
        xdiff = jnp_xdiff_(x_eval,x) 
        xdiff_norm = jnp.linalg.norm(xdiff, axis = -1, keepdims = True)
        sx = (- jnp.expand_dims(kx, -1) * jnp.where(xdiff_norm!= 0, xdiff / xdiff_norm, 0)/(2 * mx**2)).mean(axis = -2) / jnp.expand_dims(jnp.maximum(kde_, reg), -1)
        kde_ = kde_/2 ## normalizing constant if returning kde

    return sx, kde_

def stein_score_all(x, y, reg, score_kernel = "gauss", bandwidth_factor = 1, bandwidth_factor_joint = 1):
    """
    Computes the stein gradient estimator of the score function.
    Optionally takes a list of kernels to use for each component.
    Return: tuple of marginal and joint scores: (s_x, s_y, s_xy).
    """
    if x.ndim == 1:
        x = x[:, None]
    if y.ndim == 1:
        y = y[:, None]

    xy = jnp.concatenate([x, y], axis = -1) 

    if not isinstance(score_kernel, list):
        score_kernel = [score_kernel] * 3
    
    assert all(k in ["gauss", "exp"] for k in score_kernel), f"method must be either 'gauss' or 'exp'., got {score_kernel}"

    sx = stein_score(x, reg, score_kernel[0], bandwidth_factor=bandwidth_factor)
    sy = stein_score(y, reg, score_kernel[1], bandwidth_factor=bandwidth_factor)
    sxy = stein_score(xy, reg, score_kernel[2], bandwidth_factor=bandwidth_factor_joint)

    return sx, sy, sxy

def kde_score_all(x, y, reg, score_kernel = "gauss", bandwidth_factor = 1, bandwidth_factor_joint = 1):
    """
    Kernel density estimation of the score function.
    Optionally takes a list of kernels to use for each component.
    Return: tuple of marginal and joint scores: (s_x, s_y, s_xy).
    """
    if x.ndim == 1:
        x = x[:, None]
    if y.ndim == 1:
        y = y[:, None]

    if not isinstance(score_kernel, list):
        score_kernel = [score_kernel] * 3
    
    assert all(k in ["gauss", "exp"] for k in score_kernel), f"method must be either 'gauss' or 'exp'., got {score_kernel}"

    xy = jnp.concatenate([x, y], axis = -1) 

    sx, kde_x = kde(x, reg, score_kernel[0], bandwidth_factor=bandwidth_factor, eb = False)
    sy, kde_y = kde(y, reg, score_kernel[1], bandwidth_factor=bandwidth_factor, eb = False)
    sxy, kde_xy = kde(xy, reg, score_kernel[2], bandwidth_factor=bandwidth_factor_joint, eb = False)

    return sx, sy, sxy

def hybrid_score_all(x, y, reg, score_kernel = "gauss", bandwidth_factor = 1, bandwidth_factor_joint = 1):
    """
    Kernel density estimation of the marginal score functions, stein score of the joint score function.
    """
    if x.ndim == 1:
        x = x[:, None]
    if y.ndim == 1:
        y = y[:, None]

    if not isinstance(score_kernel, list):
        score_kernel = [score_kernel] * 3
    
    assert all(k in ["gauss", "exp"] for k in score_kernel), f"method must be either 'gauss' or 'exp'., got {score_kernel}"

    xy = jnp.concatenate([x, y], axis = -1) 

    sx, kde_x = kde(x, reg, score_kernel[0], bandwidth_factor=bandwidth_factor)
    sy, kde_y = kde(y, reg, score_kernel[1], bandwidth_factor=bandwidth_factor)
    sxy = stein_score(xy, 0.1, score_kernel[2], bandwidth_factor=bandwidth_factor_joint)

    return sx, sy, sxy

def score_matching(x, hidden_size = 32, lr = 0.1, lam = 1, epochs = 50, seed = 0, denoising = False, sigma = 0.1, init_weight = 0.2, n_reinit=1):
    """
    (not currently used)
    Implements score matching, standard version is the regularized score matching of Kingma and Lecun https://papers.nips.cc/paper_files/paper/2010/hash/6f3e29a35278d71c7f65495871231324-Abstract.html
    Or optionally, denoising score matching of Vincent https://www.iro.umontreal.ca/~vincentp/Publications/smdae_techreport.pdf
    """
    d = x.shape[1]

    def mlp(x_, params):
        h = jnp.dot(x_, params["w1"]) + params["b1"]
        h = jax.nn.relu(h)
        out = jnp.dot(h, params["w2"]) + params["b2"]
        return out
    
    def loss(params, x, eps):

        ## denoising score matching (Vincent 2011)
        if denoising: 
            tilde_x = x + sigma * eps
            s = mlp(x, params)
            SM_loss_ = (s - (-(tilde_x - x) / sigma**2)).sum(-1) # denoising score matching loss
            # SM_loss_ = (1. / (2. * sigma)) * ((s + eps)**2.).sum(-1) # denoising score matching loss
            mlp_x = lambda x_: mlp(x_, params)
            s, dsdx = jax.jvp(mlp_x, (x,), (jnp.ones_like(x),))
            SM_loss = (s**2 / 2.+ dsdx).sum(-1) ## return SM loss for validation 

        # vanilla score matching (Hyvarinen) with regularization w/ weight 0.1 (Lecun and Kingma 2010)
        else: 
            mlp_x = lambda x_: mlp(x_, params)
            s, dsdx = jax.jvp(mlp_x, (x,), (jnp.ones_like(x),))
            SM_loss = (s**2 / 2.+ dsdx).sum(-1)
            SM_loss_ = SM_loss + lam * (dsdx**2).sum(-1)

        return SM_loss_.mean(), SM_loss.mean()
    
    def step(params, opt, opt_state, x, eps):
        _, grads = jax.value_and_grad(loss, has_aux = True)(params, x, eps)
        updates, opt_state = opt.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        _, loss_after = loss(params, x, eps)
        return params, opt_state, loss_after
    
    best_loss = float('inf')
    
    for i in range(n_reinit):

        params = {'w1': jax.random.normal(shape=(d, hidden_size), key=jax.random.PRNGKey(seed + 0)) * init_weight,
            'b1': jax.random.normal(shape=(hidden_size), key=jax.random.PRNGKey(seed + 1)) * init_weight, 
            'w2': jax.random.normal(shape=(hidden_size, d), key=jax.random.PRNGKey(seed + 2)) * init_weight,
            'b2': jax.random.normal(shape=(d), key=jax.random.PRNGKey(seed + 3)) * init_weight}

        opt = optax.adam(lr)
        opt_state = opt.init(params)
        
        for i in range(epochs):
            eps = jax.random.normal(key = jax.random.PRNGKey(seed + 4 + i), shape = x.shape)
            params, opt_state, loss_value = step(params, opt, opt_state, x, eps)
            if loss_value < best_loss:
                best_loss = loss_value
                best_params = params.copy()
    
    score = mlp(x, best_params)

    return score 
    
def score_matching_all(x, y, **kwargs):
    """
    Train score matching on data x_train and y_train, and return the score evaluation at x_test and y_test
    """
    if len(x.shape) == 1: ## handle 1d case
        x = x[:, None]
        y = y[:, None]

    xy = jnp.concatenate([x, y], axis = -1) 

    sx = score_matching(x, **kwargs)

    sy = score_matching(y, **kwargs)

    sxy = score_matching(xy, **kwargs)

    return sx, sy, sxy
    

