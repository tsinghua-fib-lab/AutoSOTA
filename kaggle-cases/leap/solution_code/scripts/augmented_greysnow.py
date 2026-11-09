"""Extend the original input embedding with a zero-initialized RH channel."""
import numpy as np


def encode_with_rh(encoded,rh):
    return np.concatenate([encoded[:,:1680],rh,encoded[:,1680:]],axis=1).astype(np.float32)


def build_rh_model(runtime,checkpoint,config,warmstart):
    old_model=runtime.build_model(checkpoint,config['dim'],config['head_dim']) if warmstart else None
    runtime.namespace['col_len']=1740
    model=runtime.namespace['get_model'](dim=config['dim'],head_dim=config['head_dim'])
    if warmstart:
        old_weights=old_model.get_weights();new_weights=model.get_weights()
        if len(old_weights)!=len(new_weights):raise ValueError('Architecture weights differ')
        inserted=0
        for i,(old,new) in enumerate(zip(old_weights,new_weights)):
            if old.shape!=new.shape:
                if old.ndim!=2 or old.shape[0]!=44 or new.shape!=(45,old.shape[1]):raise ValueError('Unexpected weight shape change')
                new_weights[i]=np.concatenate([old[:28],np.zeros((1,old.shape[1]),dtype=old.dtype),old[28:]],axis=0);inserted+=1
            else:new_weights[i]=old
        if inserted!=1:raise ValueError('Expected exactly one new input channel')
        model.set_weights(new_weights)
    else:model.load_weights(str(checkpoint))
    runtime.model=model
    return model,old_model
