import jax 
import jax.numpy as jnp
from .pcg import pcg
from ..preconditioner.nystrom import Nys_Precond, rand_nys_appx
from ..utils.metric_utils import mse, classification_accuracy
from ..utils.model_utils import optimal_weights_transform
from ..utils.proximal_utils import batch_proxl2_tensor

def admm(model,admm_params):
    rank = admm_params['rank']
    beta = admm_params['beta']
    gamma_ratio = admm_params['gamma_ratio']
    admm_iters = admm_params['admm_iters']
    pcg_iters = admm_params['pcg_iters']
    check_opt = admm_params['check_opt']

    validate = False
    if model.Xtst is not None:
        validate = True
        ntst = model.Xtst.shape[0]
    
    n, d = model.X.shape

    metrics = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []
               }
    
    Y = jax.nn.one_hot(model.y, model.n_classes)

        # --------------- Init Optim Params ---------------
    # u contains u1 ... uP, z1... zP 
    u = jnp.zeros((model.n_classes, 2, d, model.P_S))
    # v contrains v1 ... vP, w1 ... wP
    v = jnp.zeros((model.n_classes, 2, d, model.P_S))
    # slacks s1 ... sP, t1 ... tP
    s = jnp.zeros((model.n_classes, 2, n, model.P_S))
    # lam contains lam11 lam12 ... lam1P lam21 lam22 ... lam2P
    lam = jnp.zeros((model.n_classes, 2, d, model.P_S))
    # nu contains nu11 nu12 ... nu1P nu21 nu22 ... nu2P
    nu = jnp.zeros((model.n_classes, 2, n, model.P_S))

    U, S, model.seed = rand_nys_appx(model, rank, model.seed)

    Mnys = Nys_Precond(U, S, d, model.rho, model.P_S)

    b_1 = model.batch_rmatvec_F(Y.T)/model.rho

    
    def _admm_step (u, v, s, lam, nu):
        
        # u update
        b = b_1 + v - lam + model.batch_rmatvec_G(s-nu)
        u ,_ , _ = pcg(b, model, Mnys, pcg_iters)

        # updates on v = (v1...vP, w1...wP) via prox operator
        v = v.at[:, 0, :].set(batch_proxl2_tensor(u[:, 0, :]+lam[:, 0, :], beta = beta, gamma = 1/model.rho))
        v = v.at[:, 1, :].set(batch_proxl2_tensor(u[:, 1, :]+lam[:, 1, :], beta = beta, gamma = 1/model.rho))

        # updates on s = (s1...sP, t1...tP)
        Gu = model.batch_matvec_G(u)
        s = jax.nn.relu(Gu + nu)

        # dual updates on lam=(lam11...lam2P), nu=(nu11...nu2P)
        lam += (u - v) * gamma_ratio
        nu += (Gu - s) * gamma_ratio

        return u, v, s, lam, nu, Gu
    
    
    def _opt_conds(u, v, s, lam, nu, Gu):
        y_hat = ((model.batch_matvec_F(u)).sum(axis = 2)).T
        u_v_dist = jnp.linalg.norm(u - v) + jnp.linalg.norm(Gu - s)
        u_optimality = jnp.linalg.norm(model.batch.rmatvec_F(y_hat - model.y.squeeze()) + model.rho * (lam + model.batch_matvec_F(nu)))
        v_optimality = jnp.linalg.norm(beta * v / jnp.linalg.norm(v, axis=2, keepdims=True) - model.rho * lam)
        return u_v_dist, u_optimality, v_optimality
    
    for k in range(admm_iters):

        u, v, s, lam, nu, Gu = _admm_step(u, v, s, lam, nu)

        if check_opt == True:
           u_v_dist, u_optimality, v_optimality = _opt_conds(u, v, s, lam, nu, Gu)
           print(f"iter: {k}\n  u-v dist = {u_v_dist}, u resid = {u_optimality}, v resid = {v_optimality}")

        if validate == True:
           y_hat = (model.batch_matvec_F(u)).T
           W1, W2 = model.get_ncvx_weights(v)
           Y_hat_val = model.stacked_predict(model.Xtst, W1, W2)
           
           metrics['train_loss'].append(mse(y_hat, model.y))
           metrics['val_loss'].append(mse(Y_hat_val, model.ytst))
           metrics['train_acc'].append(classification_accuracy(y_hat, model.y))
           metrics['val_acc'].append(classification_accuracy(Y_hat_val, model.ytst))

    model.u = u
    model.v = v
    model.s = s

    # Save the weights in the model
    W1, w2 = model.get_ncvx_weights(v)
    model.theta1 = W1
    model.theta2 = w2

    return v, metrics
