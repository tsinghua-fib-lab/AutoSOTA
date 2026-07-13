import numpy as np
from collections import defaultdict

def nested_dict(levels):
    if levels == 1:
        return defaultdict(dict)
    return defaultdict(lambda: nested_dict(levels - 1))

def flatten_sample(sample):
    flat = []
    def _flatten(value):
        if isinstance(value, dict):
            for v in value.values():
                _flatten(v)
        elif isinstance(value, list):
            for item in value:
                _flatten(item)
        else:
            flat.append(value)
    
    _flatten(sample)
    return flat

def set_sample(x, template):
    for k1 in template:
        for k2 in template[k1]:
            x[k1][k2] = template[k1][k2]
    return x

def match_sample(xhat, T):
    matches = []
    
    for i, template in enumerate(T):
        match = True
        
        for k1 in template:  # first-level key in T
            if k1 not in xhat:
                match = False
                break
            
            for k2 in template[k1]:  # k2 is index into xhat[k1]
                sub_template = template[k1][k2]
                sub_xhat = xhat[k1]
                
                if isinstance(sub_template, dict):
                    # 3-key case: xhat[k1] is list of lists
                    if not (0 <= k2 < len(sub_xhat)):
                        match = False
                        break
                    
                    for k3 in sub_template:
                        if not (0 <= k3 < len(sub_xhat[k2])) or sub_xhat[k2][k3] != sub_template[k3]:
                            match = False
                            break
                        
                    if not match:
                        break
                
                else:
                    # 2-key case: xhat[k1] is a list
                    if not (0 <= k2 < len(sub_xhat)) or sub_xhat[k2] != sub_template:
                        match = False
                        break
            
            if not match:
                break
        
        if match:
            matches.append(i)
    
    return matches


def encode_data(xhat, T):
    x = np.zeros(len(T))
    matches = match_sample(xhat, T)
    x[matches] = 1
    no_norm_x = x
    if x.sum() > 0:
        x = x/np.sqrt(x.sum())
    return no_norm_x, x

