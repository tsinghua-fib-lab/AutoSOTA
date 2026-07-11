import jax.numpy as jnp
from collections import defaultdict

"""Utilities related to the multicomponent implementation."""

def reverse_indices(inds):
    # find inverse of indices
    kk = 0
    start = 0
    lenparts = sum([sum([len(part) for part in comp]) for comp in inds])
    ddinv = jnp.zeros(lenparts,dtype=jnp.int32)
    for comp in inds:
        for part in comp:
            ddinv = ddinv.at[start:start+len(part)].set(kk)
            start += len(part)
            kk+=1
    return ddinv

def flatten(lst):
    result = []
    for item in lst:
        if isinstance(item, list):
            result.extend(flatten(item))
        else:
            result.append(item)
    return result

def starts(inds):
    count = 0
    start = []
    end = []
    for component in inds:
        for instance in component:
            len_inst = len(instance)
            start.append(count)
            count += len_inst
            end.append(count)
    return start, end

def repetitions(inds):
    idx_out = flatten(inds)
    Dout = max(idx_out) + 1
    Din = len(idx_out)
    idx_in = jnp.arange(Din)

    seen = defaultdict(list)
    for i, val in enumerate(idx_out):
        seen[val].append(i)

    rep_out = []
    rep_in = []

    for val, idxs in seen.items():
        if len(idxs) > 1:
            rep_out.extend([val] * len(idxs))
            rep_in.extend(idxs)

    rep = jnp.array([rep_out,rep_in]).T
    idx = jnp.array([idx_out,idx_in]).T

    phi_in = jnp.zeros((Dout,Din))
    phi = jnp.expand_dims(phi_in.at[idx_out,idx_in].set(1),axis=0)
    return rep, idx, phi

def reverse_and_compact(inds):

    _, _, phi = repetitions(inds)
    inverse = reverse_indices(inds)

    return phi, inverse