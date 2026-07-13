from flax import linen as nn
from flax.linen.module import Module
from flax.linen.module import compact
from jax.nn.initializers import xavier_uniform
from numpy import False_

class GReLU(Module):
  output_dim: int = 256
  
  @compact
  def __call__(self, x):
    d = nn.Dense(self.output_dim)(x)>=0
    x = d*x
    return x 

# An MLP with ReLU activated features layer and 2-Layer Gated ReLU head 
class GReLU_MLP(nn.Module):
  def setup(self):
      self.features = Features()
      self.head = Head()
 
  def __call__(self, x):
      x = self.features(x)
      x = self.head(x)
      return x
    
  def features_transform(self, x):
      return self.features(x)

class Features(nn.Module):
  @nn.compact
  def __call__(self, x):
    x =  nn.Dense(features=256, use_bias=False)(x)
    x = nn.relu(x)
    x = nn.Dense(features=256, use_bias=False)(x)
    x = nn.relu(x)

class Head(nn.Module):
  @nn.compact 
  def __call__(self, x):
    x = nn.Dense(features=256, use_bias=False)
    x = GReLU()(x)
    x = nn.Dense(features=1, use_bias=False)(x)
    return x 