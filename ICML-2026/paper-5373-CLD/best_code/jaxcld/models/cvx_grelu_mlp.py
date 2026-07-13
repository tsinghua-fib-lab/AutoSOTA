from .cvx_mlp import Convex_MLP
from ..utils.model_utils import get_grelu_patterns, grelu_optimal_weights_transform
import jax.numpy as jnp
from jax import jit, tree_util

class CVX_GReLU_MLP:
    def __init__(self, X, y, P_S, seed, beta = None, d_diags = None, gates = None):
        self.X = X
        self.y = y
        self.P_S = P_S
        self.seed = seed
        self.beta = beta 
        self.d_diags = d_diags
        self.gates = gates
    
    def init_model(self):
        self.d_diags, self.gates, self.seed = get_grelu_patterns(self.X, self.P_S, self.seed)
    
    @jit
    def matvec_Fi(self, i, vec):
        return self.d_diags[:,i] * (self.X @ vec)
    
    @jit
    def rmatvec_Fi(self, i, vec):
        return  self.X.T @ (self.d_diags[:,i] * vec)
    
    @jit
    def matvec_F(self, vec):
        n = self.X.shape[0]
        out = jnp.zeros((n, ))
        for i in range(self.P_S):
            out += self.matvec_Fi(i, vec[:, i])
        return out
    
    @jit
    def rmatvec_F(self, vec):
        n, d = self.X.shape
        out = jnp.zeros((d, self.P_S))
        for i in range(self.P_S):
            rFi_v = self.rmatvec_Fi(i, vec)
            out = out.at[:,i].set(rFi_v)
        return out
    
    
    @jit
    def matvec_A0(self, vec):
        return self.rmatvec_F(self.matvec_F(vec)/self.X.shape[0])
    
    @jit
    def matvec_A(self, vec):
        return self.rmatvec_F(self.matvec_F(vec)/self.X.shape[0])+self.beta*vec
    
    def get_nvcx_weights(self, u):
        return grelu_optimal_weights_transform(u, self.P_S, self.X.shape[1])
    
    def predict(self, data, W1, w2):
        d_g = (data@self.gates)>=0
        return (d_g*(data@W1))@w2
    
    def _tree_flatten(self):
        children = (self.X, self.y, self.seed, self.d_diags, self.gates)  # arrays / dynamic values
        aux_data = {'P_S': self.P_S, 'beta': self.beta}  # static values
        return (children, aux_data)
    
    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        X, y, seed, d_diags, gates = children
        P_S = aux_data['P_S']
        beta = aux_data['beta']
        return cls(X, y, P_S, seed, beta, d_diags, gates)
  
tree_util.register_pytree_node(CVX_GReLU_MLP,
                                CVX_GReLU_MLP._tree_flatten,
                                CVX_GReLU_MLP._tree_unflatten)