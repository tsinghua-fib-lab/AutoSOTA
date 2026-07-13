import optax
from jax import jit, tree_util
from optax import sgd

class SGD:
    def __init__(self, 
                 lr: float,
                  momentum: float) -> object:
        self.lr = lr
        self.momentum = momentum
        self.optimizer = sgd(learning_rate = lr, momentum = momentum)
    
    @jit
    def step(self,
        grads: dict,
         params: dict,
         opt_state):
    
        # Update parameters
        updates, opt_state = self.optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state
    
    def _tree_flatten(self):
        children = ()  # arrays / dynamic values
        aux_data = {'lr': self.lr, 'momentum': self.momentum}  # static values
        return (children, aux_data)
    
    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        () = children
        lr = aux_data['lr']
        rho = aux_data['momentum']
        return cls(lr, rho)

tree_util.register_pytree_node(SGD,
                                SGD._tree_flatten,
                                SGD._tree_unflatten)