"""
Defines trainer classes for FBPINNs and PINNs.

This is the main entry point for training FBPINNs and PINNs.

To train a FBPINN / PINN, use a Constants object to set up the problem and define its hyperparameters, and pass it
to one of the trainer classes defined here
"""

import time
from functools import partial
from jax.tree_util import tree_map

import jax
import jax.numpy as jnp
from jax import jit, vmap, value_and_grad, jvp
from jax import random
import optax
import numpy as np

from fbpinns.trainers_base import _Trainer
from fbpinns import networks, plot_trainer
from fbpinns.util.logger import logger
from fbpinns.util.jax_util import tree_index, total_size, str_tensor, partition, combine


# LABELLING CONVENTIONS

# xd = dimensionality of point
# ud = dimensionality of solution
# dims = (ud, xd)
# n = number of points
# m = number of models (i.e. subdomains)
# c = number of constraints

# x = single coordinate (xd)
# x_batch = batch of coordinates (n, xd)
# uj = solution and gradient component list

# j = index in uj
# im = index of model
# ip = index of point
# ic = index of constraint
# i = generic index

# nm = shape of rectangular DDs
# ii = for grid index in nm


# STATIC FUNCTIONS

def tree_map_dicts(f, *trees):
    "Apply function to top-level dicts in tree(s)"

    is_dict = lambda x: isinstance(x, dict)
    def apply(leaf, *leaves):
        if is_dict(leaf):# if top-level dict
            return f(leaf, *leaves)
        else:
            return leaf# if leaf (i.e. at bottom of tree), return first tree's leaf only (!)
    return jax.tree_util.tree_map(apply, *trees, is_leaf=is_dict)# stop traverse on top-level dicts


def get_jmaps(required_ujs):
    "Generate tree for computing chained jacobians"

    logger.debug("get_jmaps")

    # build tree of required gradients
    tree = {}
    for iu,ixs in required_ujs:
        t = tree
        for ix in ixs:
            if ix not in t:
                t[ix] = {}
            t = t[ix]

    # parse tree nodes
    def get_nodes(t, n=(), ks=()):
        ni = len(n)-1 + 1# index of parent node (including u at start)
        for k in t:
            ks_ = ks+(k,)
            if t[k]:
                n += (((ni,k),ks_,0),)# node
                n = get_nodes(t[k], n, ks_)
            else:
                n += (((ni,k),ks_,1),)# leaf
        return n

    # list of chained grad functions
    nodes = get_nodes(tree)
    logger.debug(nodes)

    # list of grad functions to evaluate
    leaves = tuple((i + 1, node[1]) for i,node in enumerate(nodes) if node[2])
    if not leaves:
        leaves = ((0,()),)# special case where only solution required. tree/nodes are empty in this case
    logger.debug(leaves)

    # get map between required_ujs and list of chained gradients
    jac_is = ()# il (leaf index), io (order index), iu (u index)
    for iu,ixs in required_ujs:
        io = len(ixs)
        il = [leaf[1][:io] for leaf in leaves].index(ixs)# also works for 0,()
        jac_is += ((il, io, iu),)
    logger.debug(jac_is)

    return nodes, leaves, jac_is



# JITTED FUNCTIONS

def FBPINN_model_inner(params, x, norm_fn, network_fn, unnorm_fn, window_fn):
    x_norm = norm_fn(params, x)# normalise
    u_raw = network_fn(params, x_norm)# network
    u = unnorm_fn(params, u_raw)# unnormalise
    w = window_fn(params, x)# window
    return u*w, w, u_raw

def PINN_model_inner(all_params, x, norm_fn, network_fn, unnorm_fn):
    x_norm = norm_fn(all_params, x)# normalise
    u_raw = network_fn(all_params, x_norm)# network
    u = unnorm_fn(u_raw)# unnormalise
    return u, u_raw

def FBPINN_model(all_params, x_batch, takes, model_fns, verbose=True):
    "Defines FBPINN model"

    norm_fn, network_fn, unnorm_fn, window_fn, constraining_fn = model_fns
    m_take, n_take, p_take, np_take, npou = takes

    # take x_batch
    x_take = x_batch[n_take]# (s, xd)
    log_ = logger.info if verbose else logger.debug
    log_("x_batch")
    log_(str_tensor(x_batch))# (n, xd)
    log_("x_take")
    log_(str_tensor(x_take))

    # take subdomain params
    d = all_params
    all_params_take = {t_k: {cl_k: {k: tree_map(lambda p:p[m_take], d[t_k][cl_k][k]) if k=="subdomain" else d[t_k][cl_k][k]
        for k in d[t_k][cl_k]}
        for cl_k in d[t_k]}
        for t_k in ["static", "trainable"]}
    f = {t_k: {cl_k: {k: tree_map(lambda p: 0, d[t_k][cl_k][k]) if k=="subdomain" else tree_map(lambda p: None, d[t_k][cl_k][k])
        for k in d[t_k][cl_k]}
        for cl_k in d[t_k]}
        for t_k in ["static", "trainable"]}
    logger.debug("all_params")
    logger.debug(tree_map(lambda x: str_tensor(x), all_params))
    logger.debug("all_params_take")
    logger.debug(tree_map(lambda x: str_tensor(x), all_params_take))
    logger.debug("vmap f")
    logger.debug(f)

    # batch over parameters and points
    us, ws, us_raw = vmap(FBPINN_model_inner, in_axes=(f,0,None,None,None,None))(all_params_take, x_take, norm_fn, network_fn, unnorm_fn, window_fn)# (s, ud)
    logger.debug("u")
    logger.debug(str_tensor(us))

    # apply POU and sum
    u = jnp.concatenate([us, ws], axis=1)# (s, ud+1)
    u = jax.ops.segment_sum(u, p_take, indices_are_sorted=False, num_segments=len(np_take))# (_, ud+1)
    wp = u[:,-1:]
    u = u[:,:-1]/wp
    logger.debug(str_tensor(u))
    u = jax.ops.segment_sum(u, np_take, indices_are_sorted=False, num_segments=len(x_batch))# (n, ud)
    logger.debug(str_tensor(u))
    u = u/npou
    logger.debug(str_tensor(u))

    # then apply constraining operator
    u = constraining_fn(all_params, x_batch, u)# (n, ud)
    logger.debug(str_tensor(u))

    return u, wp, us, ws, us_raw

def PINN_model(all_params, x_batch, model_fns, verbose=True):
    "Defines PINN model"

    norm_fn, network_fn, unnorm_fn, constraining_fn = model_fns
    log_ = logger.info if verbose else logger.debug
    log_("x_batch")
    log_(str_tensor(x_batch))# (n, xd)

    # batch over parameters and points
    u, u_raw = vmap(PINN_model_inner, in_axes=(None,0,None,None,None))(all_params, x_batch, norm_fn, network_fn, unnorm_fn)# (n, ud)
    logger.debug("u")
    logger.debug(str_tensor(u))

    # then apply constraining operator
    u = constraining_fn(all_params, x_batch, u)

    return u, u_raw

def FBPINN_forward(all_params, x_batch, takes, model_fns, jmaps):
    "Computes gradients of FBPINN model"

    # isolate model function
    def u(x_batch):
        return FBPINN_model(all_params, x_batch, takes, model_fns)[0], ()
    return _get_ujs(x_batch, jmaps, u)

def PINN_forward(all_params, x_batch, model_fns, jmaps):
    "Computes gradients of PINN model"

    # isolate model function
    def u(x_batch):
        return PINN_model(all_params, x_batch, model_fns)[0], ()
    return _get_ujs(x_batch, jmaps, u)

def _get_ujs(x_batch, jmaps, u):

    nodes, leaves, jac_is = jmaps
    vs = jnp.tile(jnp.eye(x_batch.shape[1]), (x_batch.shape[0],1,1))

    # chain required jacobian functions
    fs = [u]
    for (ni, ix), _, _ in nodes:
        fs.append(jacfwd(fs[ni], vs[:,ix]))

    # evaluate required jacobian functions
    jacs = []
    for ie,_ in leaves:
        fin, jac = fs[ie](x_batch)
        jacs.append(jac+(fin,))

    # index required jacobians
    ujs = [jacs[il][io][:,iu:iu+1] for il,io,iu in jac_is]

    logger.debug("fs")
    logger.debug(fs)
    logger.debug("jacs")
    for jac in jacs: logger.debug([j.shape for j in jac])# (n, ud)
    logger.debug("ujs")
    for uj in ujs: logger.debug(str_tensor(uj))

    return ujs

def jacfwd(f, v):
    "Computes jacobian for single x, for all y, fully chained"
    def jacfun(x):
        y, j, aux = jvp(f, (x,), (v,), has_aux=True)
        aux = aux + (y,)
        return j, aux
    return jacfun

def FBPINN_loss(active_params, fixed_params, static_params, takess, constraints, model_fns, jmapss, loss_fn):

    # add fixed params to active, recombine all_params
    d, da = active_params, fixed_params
    trainable_params = {cl_k: {k: tree_map(lambda p1, p2:jnp.concatenate([p1,p2],0), d[cl_k][k], da[cl_k][k]) if k=="subdomain" else d[cl_k][k]
        for k in d[cl_k]}
        for cl_k in d}
    all_params = {"static":static_params, "trainable":trainable_params}

    # run FBPINN for each constraint, with shared params
    constraints_ = []
    for takes, jmaps, constraint in zip(takess, jmapss, constraints):
        logger.debug("constraint")
        for c_ in constraint:
            logger.debug(str_tensor(c_))
        x_batch = constraint[0]
        ujs = FBPINN_forward(all_params, x_batch, takes, model_fns, jmaps)
        constraints_.append(constraint+ujs)
    return loss_fn(all_params, constraints_)

def PINN_loss(active_params, static_params, constraints, model_fns, jmapss, loss_fn):

    # recombine all_params
    all_params = {"static":static_params, "trainable":active_params}

    # run PINN for each constraint, with shared params
    constraints_ = []
    for jmaps, constraint in zip(jmapss, constraints):
        logger.debug("constraint")
        for c_ in constraint:
            logger.debug(str_tensor(c_))
        x_batch = constraint[0]
        ujs = PINN_forward(all_params, x_batch, model_fns, jmaps)
        constraints_.append(constraint+ujs)
    return loss_fn(all_params, constraints_)

@partial(jit, static_argnums=(0, 5, 8, 9, 10))
def FBPINN_update(optimiser_fn, active_opt_states,
                  active_params, fixed_params, static_params_dynamic, static_params_static,
                  takess, constraints, model_fns, jmapss, loss_fn):
    # recombine static params
    static_params = combine(static_params_dynamic, static_params_static)
    # update step
    lossval, grads = value_and_grad(FBPINN_loss, argnums=0)(
        active_params, fixed_params, static_params, takess, constraints, model_fns, jmapss, loss_fn)
    updates, active_opt_states = optimiser_fn(grads, active_opt_states, active_params)
    active_params = optax.apply_updates(active_params, updates)
    return lossval, active_opt_states, active_params

@partial(jit, static_argnums=(0, 4, 6, 7, 8))
def PINN_update(optimiser_fn, active_opt_states,
                active_params, static_params_dynamic, static_params_static,
                constraints, model_fns, jmapss, loss_fn):
    # recombine static params
    static_params = combine(static_params_dynamic, static_params_static)
    # update step
    lossval, grads = value_and_grad(PINN_loss, argnums=0)(
        active_params, static_params, constraints, model_fns, jmapss, loss_fn)
    updates, active_opt_states = optimiser_fn(grads, active_opt_states, active_params)
    active_params = optax.apply_updates(active_params, updates)
    return lossval, active_opt_states, active_params


# For fast test inference only

@partial(jax.jit, static_argnums=(1,4,5))
def _FBPINN_model_jit(all_params_dynamic, all_params_static, x_batch, takes, model_fns, verbose):
    all_params = combine(all_params_dynamic, all_params_static)
    return FBPINN_model(all_params, x_batch, takes, model_fns, verbose)
def FBPINN_model_jit(all_params, x_batch, takes, model_fns, verbose=True):
    all_params_dynamic, all_params_static = partition(all_params)
    return _FBPINN_model_jit(all_params_dynamic, all_params_static, x_batch, takes, model_fns, verbose)

@partial(jax.jit, static_argnums=(1,3,4))
def _PINN_model_jit(all_params_dynamic, all_params_static, x_batch, model_fns, verbose):
    all_params = combine(all_params_dynamic, all_params_static)
    return PINN_model(all_params, x_batch, model_fns, verbose)
def PINN_model_jit(all_params, x_batch, model_fns, verbose=True):
    all_params_dynamic, all_params_static = partition(all_params)
    return _PINN_model_jit(all_params_dynamic, all_params_static, x_batch, model_fns, verbose)



def get_inputs(x_batch, active, all_params, decomposition):
    "Get the inputs to the FBPINN model based on x_batch and the active models"

    # get the ims inside x_batch
    n_take, m_take, training_ims = decomposition.inside_points(all_params, x_batch)# (nc, m)

    # get active_ims and fixed_ims
    # scheduler should return
    # 0 = inactive (but still trained if it intersects with active models)
    # 1 = active
    # 2 = fixed
    # now modify active to
    # 0 = discard (not in current training points)
    # 1 = active
    # 2 = fixed
    active = jnp.array(active).copy()
    assert jnp.isin(active, jnp.array([0,1,2])).all()
    assert active.shape == (all_params["static"]["decomposition"]["m"],)

    active = active.at[active==0].set(1)# set inactive models to active
    mask = jnp.zeros_like(active)# mask out models in training points
    mask = mask.at[training_ims].set(1)
    active = active*mask
    ims_ = jnp.arange(all_params["static"]["decomposition"]["m"])
    active_ims = ims_[active==1]# assume unsorted
    fixed_ims = ims_[active==2]
    logger.debug("updated active")
    logger.debug(active)
    logger.debug("active_ims")
    logger.debug(active_ims)
    logger.debug("fixed_ims")
    logger.debug(fixed_ims)

    # note, numbers in all_ims == numbers in training_ims == numbers in m_take
    # which also means we need all m_take points above
    all_ims = jnp.concatenate([active_ims, fixed_ims])

    # re-index m_take to all_ims index
    inv = jnp.zeros(all_params["static"]["decomposition"]["m"], dtype=int)
    inv = inv.at[all_ims].set(jnp.arange(len(all_ims)))# assumes all_ims is unique
    m_take = inv[m_take]

    # (!) note: make sure n_take, pous (and therefore p_take / np_take) are sorted - makes segment_sum quicker
    logger.debug("takes")
    logger.debug(str_tensor(m_take))
    logger.debug(str_tensor(n_take))

    # get POUs
    pous = all_params["static"]["decomposition"]["subdomain"]["pou"][all_ims].astype(int)
    np = jnp.stack([n_take, pous[m_take,0]], axis=-1).astype(int)# points and pous
    logger.debug(str_tensor(np))
    npu,p_take = jnp.unique(np, axis=0, return_inverse=True)# unique points and pous (sorted), point-pou takes
    p_take = p_take.reshape(-1)# (!) fixes output shape inconsistency after jax 0.4.24
    np_take = npu[:,0]
    logger.debug(str_tensor(p_take))
    logger.debug(str_tensor(np_take))
    npou = len(jnp.unique(all_params["static"]["decomposition"]["subdomain"]["pou"].astype(int)))# global npou
    logger.debug(f"Total number of POUs: {npou}")

    takes = (m_take, n_take, p_take, np_take, npou)

    # cut active and fixed parameter trees
    def cut_active(d):
        "Cuts active_ims from param dict"
        return {cl_k: {k: tree_map(lambda p:p[active_ims], d[cl_k][k]) if k=="subdomain" else d[cl_k][k]
                for k in d[cl_k]}
                for cl_k in d}
    def cut_fixed(d):
        "Cuts fixed_ims from param dict"
        return {cl_k: {k: tree_map(lambda p:p[fixed_ims],  d[cl_k][k]) if k=="subdomain" else d[cl_k][k]
                for k in d[cl_k]}
                for cl_k in d}
    def cut_all(d):
        "Cuts all_ims from param dict"
        return {cl_k: {k: tree_map(lambda p:p[all_ims],    d[cl_k][k]) if k=="subdomain" else d[cl_k][k]
                for k in d[cl_k]}
                for cl_k in d}
    def merge_active(da, d):
        "Merges active_ims from param dict da to d"
        for cl_k in d:
            for k in d[cl_k]:
                if k=="subdomain":
                    d[cl_k][k] = tree_map(lambda pa, p: p.copy().at[active_ims].set(pa), da[cl_k][k], d[cl_k][k])
                else:
                    d[cl_k][k] = da[cl_k][k]
        return d

    return takes, all_ims, (active, cut_active, cut_fixed, cut_all, merge_active)


def _common_train_initialisation(c, key, all_params, problem, domain):

    # print stats
    logger.info("Total number of trainable parameters:")
    for k in all_params["trainable"]:
        logger.info(f'\t{k}: {total_size(all_params["trainable"][k]):,}')

    # initialise optimiser
    optimiser = optax.adam(**c.optimiser_kwargs)
    all_opt_states = optimiser.init(all_params["trainable"])
    logger.debug("all_opt_states")
    logger.debug(tree_map(lambda x: str_tensor(x), all_opt_states))
    optimiser_fn, loss_fn = optimiser.update, problem.loss_fn

    # get global constraints (training points)
    key, subkey = random.split(key)
    constraints_global = problem.sample_constraints(all_params=all_params, domain=domain, key=subkey, sampler=c.sampler, batch_shapes=c.ns)
    for constraint_ in constraints_global:
        for c_ in constraint_[:-1]:
            assert c_.shape[0] == constraint_[0].shape[0]
    # parse global constraints
    x_batch_global = jnp.concatenate([constraint_[0] for constraint_ in constraints_global])# (n, xd)
    constraint_offsets_global = jnp.array([0]+[constraint_[0].shape[0] for constraint_ in constraints_global[:-1]], dtype=int).cumsum()# (c,) offset index of each constraint
    constraint_fs_global = jnp.zeros((x_batch_global.shape[0], len(constraints_global)), dtype=bool)# (n, c)
    for ic in range(len(constraints_global)):# fill in constraint filters
        constraint_fs_global = constraint_fs_global.at[
            constraint_offsets_global[ic]:constraint_offsets_global[ic]+constraints_global[ic][0].shape[0], ic].set(True)
    required_ujss = [constraint_[-1] for constraint_ in constraints_global]
    constraints_global = [constraint_[:-1] for constraint_ in constraints_global]
    logger.info(f"Total number of constraints: {len(constraints_global)}")
    logger.debug("constraints_global")
    logger.debug(tree_map(lambda x: str_tensor(x), constraints_global))
    logger.debug(constraint_offsets_global)
    logger.debug(str_tensor(constraint_fs_global))
    logger.debug(required_ujss)
    logger.debug("x_batch_global")
    logger.debug(str_tensor(x_batch_global))

    # get jac maps
    jmapss = tuple(get_jmaps(required_ujs) for required_ujs in required_ujss)

    # get test points - for now, just use global interior points
    x_batch_test = domain.sample_interior(all_params=all_params, key=None, sampler="grid", batch_shape=c.n_test)
    logger.debug("x_batch_test")
    logger.debug(str_tensor(x_batch_test))

    # get exact solution if it exists
    logger.info("Computing exact solution..")
    u_exact = problem.exact_solution(all_params=all_params, x_batch=x_batch_test, batch_shape=c.n_test)
    logger.info("Computing done")
    logger.debug("u_exact")
    logger.debug(str_tensor(u_exact))

    return (optimiser, all_opt_states, optimiser_fn, loss_fn, key,
            constraints_global, x_batch_global, constraint_offsets_global, constraint_fs_global, jmapss,
            x_batch_test, u_exact)

class FBPINNTrainer(_Trainer):
    "FBPINN model trainer class"

    def _get_x_batch(self, i, active, all_params, x_batch_global, constraints_global, constraint_fs_global, constraint_offsets_global, decomposition):
        "Get the x_batch points from x_batch_global which are inside active models"

        # cut active points out of x_batch_global
        ims = jnp.arange(all_params["static"]["decomposition"]["m"])[active==1]
        training_ips, _d = decomposition.inside_models(all_params, x_batch_global, ims)# (n, mc)
        x_batch = x_batch_global[training_ips]

        # report
        logger.info(f"[i: {i}/{self.c.n_steps}] Average number of points/dimension in active subdomains: {_d:.2f}")
        logger.debug("x_batch")
        logger.debug(str_tensor(x_batch))

        # cut same points out of constraints_global
        constraint_fs = constraint_fs_global[training_ips]# for each training point, whether in constraint
        ix_ = jnp.arange(x_batch.shape[0])
        constraint_ips = [ix_[f] for f in constraint_fs.T]# indices of training points in each constraint
        constraints = [[c_[training_ips[constraint_ips[ic]]-constraint_offsets_global[ic]]
                        for c_ in constraints_global[ic]]
                       for ic in range(len(constraints_global))]# cut constraints
        logger.debug("constraints")
        logger.debug(tree_map(lambda x: str_tensor(x), constraints))
        logger.debug(tree_map(lambda x: str_tensor(x), constraint_ips))
        logger.debug(str_tensor(constraint_fs))

        return x_batch, constraints, constraint_fs, constraint_ips

    def _get_update_inputs(self, i, active, all_params, all_opt_states, x_batch_global, constraints_global, constraint_fs_global, constraint_offsets_global, decomposition, problem):
        "Get inputs to the FBPINN update step based on active models"

        start0 = time.time()
        logger.info(f"[i: {i}/{self.c.n_steps}] Updating active inputs..")

        # check active
        logger.debug(active)
        active = jnp.array(active).copy()
        assert jnp.isin(active, jnp.array([0,1,2])).all()
        assert active.shape == (all_params["static"]["decomposition"]["m"],)

        # get x_batch / constraints inputs from x_batch_global based on active
        x_batch, constraints, constraint_fs, constraint_ips = self._get_x_batch(i, active, all_params, x_batch_global, constraints_global, constraint_fs_global, constraint_offsets_global, decomposition)

        # get model takes / scheduler cuts given x_batch and active
        takes, _, (active, cut_active, cut_fixed, cut_all, merge_active) = get_inputs(x_batch, active, all_params, decomposition)

        # cut params / opt states (schedule)
        active_params = cut_active(all_params["trainable"])
        fixed_params = cut_fixed(all_params["trainable"])
        static_params = cut_all(all_params["static"])
        active_opt_states = tree_map_dicts(cut_active, all_opt_states)# because all_opt_states has more complex structure
        logger.debug("active_params")
        logger.debug(tree_map(lambda x: str_tensor(x), active_params))
        logger.debug("fixed_params")
        logger.debug(tree_map(lambda x: str_tensor(x), fixed_params))
        logger.debug("static_params")
        logger.debug(tree_map(lambda x: str_tensor(x), static_params))
        logger.debug("active_opt_states")
        logger.debug(tree_map(lambda x: str_tensor(x), active_opt_states))

        # split takes into many using constraint_ips (multiple constraints)
        # choice 1: use serial constraints - to stop too many gradients
        # choice 2: combine constraints into joint params/x_batch - 1) for single inside computation 2) needed for joint loss (shared params)
        (m_take, n_take, p_take, np_take, npou) = takes
        takess = []
        iu_ = jnp.arange(np_take.shape[0])
        for f, ips in zip(constraint_fs.T, constraint_ips):

            f1 = f# cut x_batch
            f2 = f[np_take]# cut unique point-pou

            #ips = ix_[f1] cut x_batch index (already done above)
            ius = iu_[f2]# cut unique point-pou index

            f3 = f1[n_take]# cut n_take
            f4 = f2[p_take]# cut p_take

            # re-index takes based on cuts
            inv = jnp.zeros(x_batch.shape[0], dtype=int)
            inv = inv.at[ips].set(jnp.arange(len(ips)))

            inv2 = jnp.zeros(np_take.shape[0], dtype=int)
            inv2 = inv2.at[ius].set(jnp.arange(len(ius)))

            m_take_ic = m_take[f3]# not necessary to re-index as params are not cut per constraint
            n_take_ic = inv[n_take[f3]]

            p_take_ic = inv2[p_take[f4]]
            np_take_ic = inv[np_take[f2]]

            takess.append((m_take_ic, n_take_ic, p_take_ic, np_take_ic, npou))

        logger.info(f"[i: {i}/{self.c.n_steps}] Updating active inputs done ({time.time()-start0:.2f} s)")

        return active, merge_active, active_opt_states, active_params, fixed_params, static_params, takess, constraints, x_batch

    def train(self, use_pretrained=False, pretrained_params=None):
        "Train model"

        c, writer = self.c, self.writer

        # generate root key
        key = random.PRNGKey(c.seed)
        np.random.seed(c.seed)

        # define all_params
        all_params = {"static":{},"trainable":{}}

        # initialise domain, problem and decomposition params
        domain, problem, decomposition = c.domain, c.problem, c.decomposition
        for tag, cl, kwargs in zip(["domain", "problem", "decomposition"], [domain, problem, decomposition],
                                   [c.domain_init_kwargs, c.problem_init_kwargs, c.decomposition_init_kwargs]):
            ps_ = cl.init_params(**kwargs)
            if ps_[0]: all_params["static"][tag] = ps_[0]
            if ps_[1]: all_params["trainable"][tag] = ps_[1]
        assert (all_params["static"]["domain"]["xd"] ==\
                all_params["static"]["problem"]["dims"][1] ==\
                all_params["static"]["decomposition"]["xd"])
        logger.info(f'Total number of subdomains: {all_params["static"]["decomposition"]["m"]}')

        # initialise subdomain network params
        network = c.network

        if use_pretrained:
            assert pretrained_params is not None
            all_params["trainable"] = pretrained_params["trainable"]
            all_params["static"] = pretrained_params["static"]
            print("using pretrained params")
        else:
            key, *subkeys = random.split(key, all_params["static"]["decomposition"]["m"]+1)
            args_ = c.network_init_kwargs.values()
            ps_ = vmap(network.init_params, in_axes=(0,)+(None,)*len(args_))(jnp.array(subkeys), *args_)
            if ps_[0]: all_params["static"]["network"] = {"subdomain": ps_[0]}# add subdomain key
            if ps_[1]: all_params["trainable"]["network"] = {"subdomain": ps_[1]}# add subdomain key

        logger.debug("all_params")
        logger.debug(tree_map(lambda x: str_tensor(x), all_params))
        model_fns = (decomposition.norm_fn, network.network_fn, decomposition.unnorm_fn, decomposition.window_fn, problem.constraining_fn)

        # initialise scheduler
        scheduler = c.scheduler(all_params=all_params, n_steps=c.n_steps, **c.scheduler_kwargs)

        # common initialisation
        (optimiser, all_opt_states, optimiser_fn, loss_fn, key,
        constraints_global, x_batch_global, constraint_offsets_global, constraint_fs_global, jmapss,
        x_batch_test, u_exact) = _common_train_initialisation(c, key, all_params, problem, domain)

        # fix test data inputs
        logger.info("Getting test data inputs..")
        active_test_ = jnp.ones(all_params["static"]["decomposition"]["m"], dtype=int)
        takes_, all_ims_, (_, _, _, cut_all_, _)  = get_inputs(x_batch_test, active_test_, all_params, decomposition)
        test_inputs = (takes_, all_ims_, cut_all_)

        # train loop
        pstep, fstep, u_test_losses = 0, 0, []
        start0, start1, report_time = time.time(), time.time(), 0.
        merge_active, active_params, active_opt_states, fixed_params = None, None, None, None
        lossval = None
        for i,active_ in enumerate(scheduler):

            # update active
            if active_ is not None:
                active = active_

                # first merge latest all_params / all_opt_states
                if i != 0:
                    all_params["trainable"] = merge_active(active_params, all_params["trainable"])
                    all_opt_states = tree_map_dicts(merge_active, active_opt_states, all_opt_states)

                # then get new inputs to update step
                active, merge_active, active_opt_states, active_params, fixed_params, static_params, takess, constraints, x_batch = \
                     self._get_update_inputs(i, active, all_params, all_opt_states, x_batch_global, constraints_global, constraint_fs_global, constraint_offsets_global, decomposition, problem)

                # AOT compile update function
                startc = time.time()
                logger.info(f"[i: {i}/{self.c.n_steps}] Compiling update step..")
                static_params_dynamic, static_params_static = partition(static_params)
                update = FBPINN_update.lower(optimiser_fn, active_opt_states,
                                             active_params, fixed_params, static_params_dynamic, static_params_static,
                                             takess, constraints, model_fns, jmapss, loss_fn).compile()
                logger.info(f"[i: {i}/{self.c.n_steps}] Compiling done ({time.time()-startc:.2f} s)")
                cost_ = update.cost_analysis()
                cost_ = [cost_]
                p,f = total_size(active_params["network"]), cost_[0]["flops"] if (cost_ and "flops" in cost_[0]) else 0
                logger.debug("p, f")
                logger.debug((p,f))

            # report initial model
            if i == 0:
                u_test_losses, start1, report_time = \
                self._report(i, pstep, fstep, u_test_losses, start0, start1, report_time,
                            u_exact, x_batch_test, test_inputs, all_params, all_opt_states, model_fns, problem, decomposition,
                            active, merge_active, active_opt_states, active_params, x_batch,
                            lossval)

            # take a training step
            lossval, active_opt_states, active_params = update(active_opt_states,
                                         active_params, fixed_params, static_params_dynamic,
                                         takess, constraints)# note compiled function only accepts dynamic arguments
            pstep, fstep = pstep+p, fstep+f

            # report
            u_test_losses, start1, report_time = \
            self._report(i + 1, pstep, fstep, u_test_losses, start0, start1, report_time,
                        u_exact, x_batch_test, test_inputs, all_params, all_opt_states, model_fns, problem, decomposition,
                        active, merge_active, active_opt_states, active_params, x_batch,
                        lossval)

        # cleanup
        writer.close()
        logger.info(f"[i: {i+1}/{self.c.n_steps}] Training complete")

        # return trained parameters
        all_params["trainable"] = merge_active(active_params, all_params["trainable"])
        all_opt_states = tree_map_dicts(merge_active, active_opt_states, all_opt_states)

        return all_params

    def _report(self, i, pstep, fstep, u_test_losses, start0, start1, report_time,
                u_exact, x_batch_test, test_inputs, all_params, all_opt_states, model_fns, problem, decomposition,
                active, merge_active, active_opt_states, active_params, x_batch,
                lossval):
        "Report results"

        c = self.c
        summary_,test_,model_save_ = [(i % f == 0) for f in
                                      [c.summary_freq, c.test_freq, c.model_save_freq]]
        if summary_ or test_ or model_save_:

            # print summary
            if i != 0 and summary_:
                rate = c.summary_freq / (time.time()-start1-report_time)
                self._print_summary(i, lossval.item(), rate, start0)
                start1, report_time = time.time(), 0.

            if test_ or model_save_:

                start2 = time.time()

                # merge latest all_params / all_opt_states
                all_params["trainable"] = merge_active(active_params, all_params["trainable"])
                all_opt_states = tree_map_dicts(merge_active, active_opt_states, all_opt_states)

                # take test step
                if test_:
                    u_test_losses = self._test(
                        x_batch_test, u_exact, u_test_losses, x_batch, test_inputs, i, pstep, fstep, start0, active, all_params, model_fns, problem, decomposition)

                # save model
                if model_save_:
                    self._save_model(i, (i, all_params, all_opt_states, active, jnp.array(u_test_losses)))

                report_time += time.time()-start2

        return u_test_losses, start1, report_time

    def _test(self, x_batch_test, u_exact, u_test_losses, x_batch, test_inputs, i, pstep, fstep, start0, active, all_params, model_fns, problem, decomposition):
        "Test step"

        c, writer = self.c, self.writer
        n_test = c.n_test

        # get FBPINN solution using test data
        takes, all_ims, cut_all = test_inputs
        all_params_cut = {"static":cut_all(all_params["static"]),
                          "trainable":cut_all(all_params["trainable"])}
        u_test, wp_test_, us_test_, ws_test_, us_raw_test_ = FBPINN_model_jit(all_params_cut, x_batch_test, takes, model_fns, verbose=False)
        if all_params["static"]["problem"]["dims"][1] == 1:# 1D plots require full lines, not just hist stats

            m, ud, n = all_params["static"]["decomposition"]["m"], all_params["static"]["problem"]["dims"][0], x_batch_test.shape[0]

            us_test = jnp.full((m, n, ud), jnp.nan)
            us_test = us_test.at[all_ims[takes[0]], takes[1], :].set(us_test_)

            ws_test = jnp.full((m, n, 1), jnp.nan)
            ws_test = ws_test.at[all_ims[takes[0]], takes[1], :].set(ws_test_)

            us_raw_test = jnp.full((m, n, ud), jnp.nan)
            us_raw_test = us_raw_test.at[all_ims[takes[0]], takes[1], :].set(us_raw_test_)

            # apply POU
            us_test = us_test.at[all_ims[takes[0]], takes[1], :].divide(wp_test_[takes[2]])/takes[4]
            ws_test = ws_test.at[all_ims[takes[0]], takes[1], :].divide(wp_test_[takes[2]])/takes[4]

            # apply constraining operator
            us_test = vmap(model_fns[-1], in_axes=(None,None,0))(all_params, x_batch_test, us_test)

        else:
            us_test, ws_test, us_raw_test = us_test_, ws_test_, us_raw_test_

        # get losses over test data
        if u_exact.shape[-1] == 3:
            l1_ux = (jnp.mean(jnp.abs(u_exact[:,0]-u_test[:,0]))/jnp.mean(jnp.abs(u_exact[:,0]))).item()
            l1_uy = (jnp.mean(jnp.abs(u_exact[:,1]-u_test[:,1]))/jnp.mean(jnp.abs(u_exact[:,1]))).item()
            l1_rho = (jnp.mean(jnp.abs(u_exact[:,2]-u_test[:,2]))/jnp.mean(jnp.abs(u_exact[:,2]))).item()
            print(f"ux Relative L1 Error: {l1_ux*100:.4f}%")
            print(f"uy Relative L1 Error: {l1_uy*100:.4f}%")
            print(f"rho Relative L1 Error: {l1_rho*100:.4f}%")
            l2_ux = (jnp.sqrt(jnp.mean((u_exact[:,0]-u_test[:,0])**2))/jnp.sqrt(jnp.mean((u_exact[:,0])**2))).item()
            l2_uy = (jnp.sqrt(jnp.mean((u_exact[:,1]-u_test[:,1])**2))/jnp.sqrt(jnp.mean((u_exact[:,1])**2))).item()
            l2_rho = (jnp.sqrt(jnp.mean((u_exact[:,2]-u_test[:,2])**2))/jnp.sqrt(jnp.mean((u_exact[:,2])**2))).item()
            print(f"ux Relative L2 Error: {l2_ux*100:.4f}%")
            print(f"uy Relative L2 Error: {l2_uy*100:.4f}%")
            print(f"rho Relative L2 Error: {l2_rho*100:.4f}%")
            l1 = l1_ux + l1_uy + l1_rho
        else:
            l1 = (jnp.mean(jnp.abs(u_exact-u_test))/jnp.mean(jnp.abs(u_exact))).item()
            print(f"Relative L1 Error: {l1*100:.4f}%")
            l2 = (jnp.sqrt(jnp.mean((u_exact-u_test)**2))/jnp.sqrt(jnp.mean((u_exact)**2))).item()
            print(f"Relative L2 Error: {l2*100:.4f}%")
            l1 = jnp.mean(jnp.abs(u_exact-u_test)).item()
        l1n = l1 / u_exact.std().item()
        u_test_losses.append([i, pstep, fstep, time.time()-start0, l1, l1n])
        writer.add_scalar("loss/test/l1_istep", l1, i)

        # create figures
        if i % (c.test_freq * 10) == 0:
            fs = plot_trainer.plot("FBPINN", all_params["static"]["problem"]["dims"],
                x_batch_test, u_exact, u_test, us_test, ws_test, us_raw_test, x_batch, all_params, i, active, decomposition, n_test)
            if fs is not None:
                self._save_figs(i, fs)

        return u_test_losses





class PINNTrainer(_Trainer):
    "PINN model trainer class"

    def train(self, use_pretrained=False, pretrained_params=None):
        "Train model"
        c, writer = self.c, self.writer

        # generate root key
        key = random.PRNGKey(c.seed)
        np.random.seed(c.seed)

        # define all_params
        all_params = {"static":{},"trainable":{}}

        # initialise domain, problem and decomposition params
        domain, problem = c.domain, c.problem
        for tag, cl, kwargs in zip(["domain", "problem"], [domain, problem],
                                   [c.domain_init_kwargs, c.problem_init_kwargs]):
            ps_ = cl.init_params(**kwargs)
            if ps_[0]: all_params["static"][tag] = ps_[0]
            if ps_[1]: all_params["trainable"][tag] = ps_[1]
        assert (all_params["static"]["domain"]["xd"] ==\
                all_params["static"]["problem"]["dims"][1])

        # initialise network params
        network = c.network

        if use_pretrained:
            assert pretrained_params is not None
            all_params["trainable"] = pretrained_params["trainable"]
            all_params["static"] = pretrained_params["static"]
            print("using pretrained params")

        else:
            key, subkey = random.split(key)
            ps_ = network.init_params(key=subkey, **c.network_init_kwargs)
        
            if ps_[0]: all_params["static"]["network"] = {"subdomain": ps_[0]}# add subdomain key
            if ps_[1]: all_params["trainable"]["network"] = {"subdomain": ps_[1]}# add subdomain key
        logger.debug("all_params")
        logger.debug(tree_map(lambda x: str_tensor(x), all_params))

        # define unnorm function
        mu_, sd_ = c.decomposition_init_kwargs["unnorm"]
        unnorm_fn = lambda u: networks.unnorm(mu_, sd_, u)
        model_fns = (domain.norm_fn, network.network_fn, unnorm_fn, problem.constraining_fn)

        # common initialisation
        (optimiser, all_opt_states, optimiser_fn, loss_fn, key,
        constraints_global, x_batch_global, _, _, jmapss,
        x_batch_test, u_exact) = _common_train_initialisation(c, key, all_params, problem, domain)

        # get implicit jitted update function
        active_params = all_params["trainable"]
        static_params = all_params["static"]
        active_opt_states = all_opt_states
        x_batch = x_batch_global
        constraints = constraints_global

        # AOT compile update function
        startc = time.time()
        logger.info(f"[i: {0}/{self.c.n_steps}] Compiling update step..")
        static_params_dynamic, static_params_static = partition(static_params)
        update = PINN_update.lower(optimiser_fn, active_opt_states,
                                   active_params, static_params_dynamic, static_params_static,
                                   constraints, model_fns, jmapss, loss_fn).compile()
        logger.info(f"[i: {0}/{self.c.n_steps}] Compiling done ({time.time()-startc:.2f} s)")
        cost_ = update.cost_analysis()
        cost_ = [cost_]
        p,f = total_size(active_params["network"]), cost_[0]["flops"] if (cost_ and "flops" in cost_[0]) else 0
        logger.debug("p, f")
        logger.debug((p,f))

        # train loop
        pstep, fstep, u_test_losses = 0, 0, []
        start0, start1, report_time = time.time(), time.time(), 0.
        lossval = None
        for i in range(c.n_steps):

            if i == 0:
                # report initial model
                u_test_losses, start1, report_time = \
                self._report(i, pstep, fstep, u_test_losses, start0, start1, report_time,
                            u_exact, x_batch_test, all_params, all_opt_states, model_fns, problem,
                            active_opt_states, active_params,
                            x_batch,
                            lossval)

            # take a training step
            lossval, active_opt_states, active_params = update(active_opt_states,
                                       active_params, static_params_dynamic,
                                       constraints)# note compiled function only accepts dynamic arguments
            pstep, fstep = pstep+p, fstep+f

            # report
            u_test_losses, start1, report_time = \
            self._report(i + 1, pstep, fstep, u_test_losses, start0, start1, report_time,
                        u_exact, x_batch_test, all_params, all_opt_states, model_fns, problem,
                        active_opt_states, active_params,
                        x_batch,
                        lossval)

        # cleanup
        writer.close()
        logger.info(f"[i: {i+1}/{self.c.n_steps}] Training complete")

        # return trained parameters
        all_params["trainable"] = active_params
        all_opt_states = active_opt_states

        return all_params

    def _report(self, i, pstep, fstep, u_test_losses, start0, start1, report_time,
                u_exact, x_batch_test, all_params, all_opt_states, model_fns, problem,
                active_opt_states, active_params,
                x_batch,
                lossval):
        "Report results"

        c = self.c
        summary_,test_,model_save_ = [(i % f == 0) for f in
                                      [c.summary_freq, c.test_freq, c.model_save_freq]]
        if summary_ or test_ or model_save_:

            # print summary
            if i != 0 and summary_:
                rate = c.summary_freq / (time.time()-start1-report_time)
                self._print_summary(i, lossval.item(), rate, start0)
                start1, report_time = time.time(), 0.

            if test_ or model_save_:

                start2 = time.time()

                # merge latest params
                all_params["trainable"] = active_params
                all_opt_states = active_opt_states

                # take test step
                if test_:
                    u_test_losses = self._test(
                        x_batch_test, u_exact, u_test_losses, x_batch, i, pstep, fstep, start0, all_params, model_fns, problem)

                # save model
                if model_save_:
                    self._save_model(i, (i, all_params, all_opt_states, jnp.array(u_test_losses)))

                report_time += time.time()-start2


        return u_test_losses, start1, report_time

    def _test(self, x_batch_test, u_exact, u_test_losses, x_batch, i, pstep, fstep, start0, all_params, model_fns, problem):
        "Test step"

        c, writer = self.c, self.writer
        n_test = c.n_test

        # get PINN solution using test data
        u_test, u_raw_test = PINN_model_jit(all_params, x_batch_test, model_fns, verbose=False)
        
        # get losses over test data
        if u_exact.shape[-1] == 3:
            l1_ux = (jnp.mean(jnp.abs(u_exact[:,0]-u_test[:,0]))/jnp.mean(jnp.abs(u_exact[:,0]))).item()
            l1_uy = (jnp.mean(jnp.abs(u_exact[:,1]-u_test[:,1]))/jnp.mean(jnp.abs(u_exact[:,1]))).item()
            l1_rho = (jnp.mean(jnp.abs(u_exact[:,2]-u_test[:,2]))/jnp.mean(jnp.abs(u_exact[:,2]))).item()
            print(f"ux Relative L1 Error: {l1_ux*100:.4f}%")
            print(f"uy Relative L1 Error: {l1_uy*100:.4f}%")
            print(f"rho Relative L1 Error: {l1_rho*100:.4f}%")
            l2_ux = (jnp.sqrt(jnp.mean((u_exact[:,0]-u_test[:,0])**2))/jnp.sqrt(jnp.mean((u_exact[:,0])**2))).item()
            l2_uy = (jnp.sqrt(jnp.mean((u_exact[:,1]-u_test[:,1])**2))/jnp.sqrt(jnp.mean((u_exact[:,1])**2))).item()
            l2_rho = (jnp.sqrt(jnp.mean((u_exact[:,2]-u_test[:,2])**2))/jnp.sqrt(jnp.mean((u_exact[:,2])**2))).item()
            print(f"ux Relative L2 Error: {l2_ux*100:.4f}%")
            print(f"uy Relative L2 Error: {l2_uy*100:.4f}%")
            print(f"rho Relative L2 Error: {l2_rho*100:.4f}%")
            l1 = l1_ux + l1_uy + l1_rho
        else:
            l1 = (jnp.mean(jnp.abs(u_exact-u_test))/jnp.mean(jnp.abs(u_exact))).item()
            print(f"Relative L1 Error: {l1*100:.4f}%")
            l2 = (jnp.sqrt(jnp.mean((u_exact-u_test)**2))/jnp.sqrt(jnp.mean((u_exact)**2))).item()
            print(f"Relative L2 Error: {l2*100:.4f}%")
            l1 = jnp.mean(jnp.abs(u_exact-u_test)).item()
        l1n = l1 / u_exact.std().item()
        u_test_losses.append([i, pstep, fstep, time.time()-start0, l1, l1n])
        
        writer.add_scalar("loss/test/l1_istep", l1, i)

        # create figures
        if i % (c.test_freq * 10) == 0:
            fs = plot_trainer.plot("PINN", all_params["static"]["problem"]["dims"],
                x_batch_test, u_exact, u_test, u_raw_test, x_batch, all_params, i, n_test)
            
            if fs is not None:
                self._save_figs(i, fs)

        return u_test_losses



if __name__ == "__main__":

    from fbpinns.constants import Constants
    from fbpinns.problems import HarmonicOscillator1D, HarmonicOscillator1DHardBC, HarmonicOscillator1DInverse

    logger.setLevel("DEBUG")

    c = Constants(
        run="test",
        #problem=HarmonicOscillator1D,
        #problem=HarmonicOscillator1DHardBC,
        problem=HarmonicOscillator1DInverse,
        network_init_kwargs = dict(layer_sizes=[1, 32, 32, 1]),
        )

    run = FBPINNTrainer(c)
    #run = PINNTrainer(c)

    all_params = run.train()
    print(all_params["static"]["problem"])
    if "problem" in all_params["trainable"]:
        print(all_params["trainable"]["problem"])