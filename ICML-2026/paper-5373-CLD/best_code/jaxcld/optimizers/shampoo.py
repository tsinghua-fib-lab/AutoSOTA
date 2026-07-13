import optax
from jax import jit, tree_util
from .dist_shampoo.distributed_shampoo import distributed_shampoo

class Shampoo:
    def __init__(self, 
                 lr: float, 
                ) -> object:
        self.lr = lr
        self.optimizer = distributed_shampoo(learning_rate=lr, 
                                             block_size=128)
    
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
        aux_data = {'lr': self.lr}  # static values
        return (children, aux_data)
    
    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        () = children
        lr = aux_data['lr']
        return cls(lr)

tree_util.register_pytree_node(Shampoo,
                                Shampoo._tree_flatten,
                                Shampoo._tree_unflatten)