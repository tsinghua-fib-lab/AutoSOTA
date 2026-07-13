import jax
from .cvx_mlp import Convex_MLP
from ..utils.model_utils import get_hyperplane_cuts, optimal_weights_transform
import jax.numpy as jnp
from jax import jit, vmap, tree_util

class CVX_ReLU_MLP(Convex_MLP):
    def __init__(self, X, y, n_classes, P_S, beta, rho, seed,
    d_diags = None, e_diags = None, Xtst = None, ytst = None):
        super().__init__(X, y, P_S, beta, rho, seed)
        self.n_classes = n_classes
        self.d_diags = d_diags
        self.e_diags = e_diags
        self.Xtst = Xtst
        self.ytst = ytst
        
        self.theta1 = None
        self.theta2 = None
    
    def init_model(self):
        self.d_diags, self.seed = get_hyperplane_cuts(self.X, self.P_S, 
        self.seed)
        self.e_diags = 2*self.d_diags-1

    @jit
    def matvec_F(self, vec):
      # Input vec has dimensions (2, d, P)
      diff = vec[0, :, :] - vec[1, :, :]
      Xdiff = self.X@diff
      return (self.d_diags*Xdiff).sum(axis = 1)

    def batch_matvec_F(self, vecs):
      return vmap(self.matvec_F)(vecs)
    
    @jit
    def rmatvec_Fi(self, i, vec):
        return  self.X.T @ (self.d_diags[:,i] * vec)

    @jit 
    def rmatvec_F(self, vec):
        n, d = self.X.shape
        out = jnp.zeros((2, d, self.P_S))
        for i in range(self.P_S):
            rFi_v = self.rmatvec_Fi(i,vec)
            out = out.at[0,:,i].set(rFi_v)
            out = out.at[1,:,i].set(-rFi_v)
        return out
    
    def batch_rmatvec_F(self, vecs):
      return vmap(self.rmatvec_F)(vecs)
     
    @jit
    def matvec_G(self, vec):
      n, d = self.X.shape
      out = jnp.zeros((2, n, self.P_S))
      out = out.at[0, :, :].set(self.e_diags*(self.X@vec[0,:,:]))
      out = out.at[1, :, :].set(self.e_diags*(self.X@vec[1,:,:]))
      return out

    
    def batch_matvec_G(self, vecs):
      return vmap(self.matvec_G)(vecs)

    
    @jit
    def rmatvec_G(self, vec):
       n, d = self.X.shape
       out = jnp.zeros((2, d, self.P_S))
       out = out.at[0,:,:].set((self.X.T@(self.e_diags*vec[0,:,:])))
       out = out.at[1,:,:].set((self.X.T@(self.e_diags*vec[1,:,:])))
       return out
      
    def batch_rmatvec_G(self, vecs):
      return vmap(self.rmatvec_G)(vecs)
    
    @jit
    def matvec_A(self, vec):
         b = vec  
         b = b + 1/self.rho * self.rmatvec_F(self.matvec_F(vec))
         b = b + self.rmatvec_G(self.matvec_G(vec))
         return b
    
    def batch_matvec_A(self, vecs):
       return vmap(self.matvec_A)(vecs)

    def get_ncvx_weights(self, v):
      d = self.X.shape[1]
      m = 2*self.P_S
      W1 = jnp.zeros((self.n_classes, d, m))
      W2 = jnp.zeros((self.n_classes, m))
      for i in range(self.n_classes):
        w1, w2 = optimal_weights_transform(v[i, 0, :], v[i, 1, :], 
        self.P_S, d)
        W1 = W1.at[i,:,:].set(w1)
        W2 = W2.at[i,:].set(w2) 
      return W1, W2
    
    def predict(self, X, w1, w2):
      return jax.nn.relu(X@w1)@w2
    
    def stacked_predict(self, X, W1, W2):
      return (vmap(lambda w1, w2: self.predict(X, w1, w2))(W1, W2)).T
    
    def _tree_flatten(self):
        children = (self.X, self.y, self.beta, self.seed, 
        self.d_diags, self.e_diags, self.Xtst, self.ytst)  # arrays / dynamic values
        aux_data = {'n_classes': self.n_classes, 'P_S': self.P_S, 'rho': self.rho}  # static values
        return (children, aux_data)
    
    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        X, y, beta, seed, d_diags, e_diags, Xtst, ytst = children
        n_classes = aux_data['n_classes']
        P_S = aux_data['P_S']
        rho = aux_data['rho']
        return cls(X, y, n_classes, P_S, beta, rho, seed, d_diags, e_diags, Xtst, ytst)
  
tree_util.register_pytree_node(CVX_ReLU_MLP,
                                CVX_ReLU_MLP._tree_flatten,
                                CVX_ReLU_MLP._tree_unflatten)