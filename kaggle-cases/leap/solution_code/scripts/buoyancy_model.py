"""Add zero-initialized RH and paper-defined buoyancy input channels."""
import numpy as np


def encode_with_physics(encoded,rh,buoyancy):
    return np.concatenate([encoded[:,:1680],rh,buoyancy,encoded[:,1680:]],axis=1).astype(np.float32)


def build_physics_model(runtime,checkpoint,config,warmstart):
    old=runtime.build_model(checkpoint,config['dim'],config['head_dim']) if warmstart else None
    runtime.namespace['col_len']=1800
    model=runtime.namespace['get_model'](dim=config['dim'],head_dim=config['head_dim'])
    if warmstart:
        previous=old.get_weights();expanded=model.get_weights()
        if len(previous)!=len(expanded):raise ValueError('Augmented weight count differs')
        changes=0
        for i,(p,n) in enumerate(zip(previous,expanded)):
            if p.shape!=n.shape:
                if p.ndim!=2 or p.shape[0]!=44 or n.shape!=(46,p.shape[1]):raise ValueError('Unexpected architecture change')
                expanded[i]=np.concatenate([p[:28],np.zeros((2,p.shape[1]),dtype=p.dtype),p[28:]],axis=0);changes+=1
            else:expanded[i]=p
        if changes!=1:raise ValueError('Expected one expanded kernel')
        model.set_weights(expanded)
    else:model.load_weights(str(checkpoint))
    runtime.model=model
    return model,old
