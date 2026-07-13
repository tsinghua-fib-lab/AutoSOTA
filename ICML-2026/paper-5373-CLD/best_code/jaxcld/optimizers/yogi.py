import optax
from jax import jit, tree_util
from optax import yogi

class Yogi:
    def __init__(self, 
                 lr: float, 
                ) -> object:
        self.lr = lr
        self.optimizer = yogi(learning_rate = lr)
    
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

tree_util.register_pytree_node(Yogi,
                                Yogi._tree_flatten,
                                Yogi._tree_unflatten)