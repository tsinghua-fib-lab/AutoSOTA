from collections.abc import MutableMapping

__all__ = [
    'MinMeter',
    'MaxMeter',
    'update_nested_dict',
    'flatten_dict'
]

class MinMeter:
    def __init__(self, delta: float=0.0):
        self.delta = delta
        self.reset()

    def update(self, value):
        update = False
        if self._best is not None: print('MinMeter:', round(self._best, 4), round(value, 4))
        if self._best is None:
            self._best = value
            update = True
        elif value <= (self._best - self.delta):
            self._best = value
            update = True
        
        return update
        
    def reset(self):
        self._best = None
    
    @property
    def min_value(self):
        return self._best

 
class MaxMeter:
    def __init__(self, delta: float=0.0):
        self.delta = delta
        self.reset()

    def update(self, value):
        update = False
        #if self._best is not None: print('MaxMeter:', round(self._best, 4), round(value, 4))
        if self._best is None:
            self._best = value
            update = True
        elif value >= (self._best + self.delta):
            self._best = value
            update = True
        
        return update
        
    
    def reset(self):
        self._best = None
    
    @property
    def max_value(self):
        return self._best


def build_nested_dict(pairs_dict):
    res = {}
    for key, value in pairs_dict.items():
        keys = key.split('.')
        current_dict = res

        for k in keys[:-1]:
            if k not in current_dict:
                current_dict[k] = {}
            current_dict = current_dict[k] 

        current_dict[keys[-1]] = value
    return res


def update_dict(d, updates):
    for k, v in updates.items():
        if isinstance(v, dict):
            if k not in d:
                d[k] = {}
            update_dict(d[k], v)
        else:
            d[k] = v
            

def update_nested_dict(base_dict, updates_dict):
    updates_nested = build_nested_dict(updates_dict)
    update_dict(base_dict, updates_nested)


def _flatten_dict_gen(d, parent_key, sep):
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            yield from flatten_dict(v, new_key, sep=sep).items()
        else:
            yield new_key, v


def flatten_dict(d: MutableMapping, parent_key: str = '', sep: str = '.'):
    return dict(_flatten_dict_gen(d, parent_key, sep))