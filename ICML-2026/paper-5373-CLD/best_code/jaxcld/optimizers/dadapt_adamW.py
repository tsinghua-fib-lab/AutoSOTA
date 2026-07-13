import optax
from jax import jit, tree_util
from optax.contrib import dadapt_adamw

class Dadapt_AdamW:
    def __init__(self, 
                 lr: float, 
                 weight_decay: float) -> object:
        self.lr = lr
        self.weight_decay = weight_decay
        self.optimizer = dadapt_adamw(learning_rate = lr, weight_decay = weight_decay)
    
    @jit 
    def step(self,
        grads: dict,
         params: dict,
         opt_state):
  
        #Update parameters
        updates, opt_state = self.optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state
    
    def _tree_flatten(self):
        children = ()  # arrays / dynamic values
        aux_data = {'lr': self.lr, 'weight_decay': self.weight_decay}  # static values
        return (children, aux_data)
    
    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        () = children
        lr = aux_data['lr']
        rho = aux_data['weight_decay']
        return cls(lr, rho)

tree_util.register_pytree_node(Dadapt_AdamW,
                                Dadapt_AdamW._tree_flatten,
                                Dadapt_AdamW._tree_unflatten)