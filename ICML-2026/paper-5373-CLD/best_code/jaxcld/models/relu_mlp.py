from flax import linen as nn
from jax.nn.initializers import xavier_uniform

# Standard ReLU MLP
class ReLU_MLP(nn.Module):
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
    return x

class Head(nn.Module):
  @nn.compact 
  def __call__(self, x):
    x = nn.Dense(features=20, use_bias=False)(x)
    x = nn.relu(x)
    x = nn.Dense(features=1, use_bias=False)(x)
    return x 

##############################################################################################################

# Variable projection ReLU MLP 
class VarPro_MLP(nn.Module):
    def setup(self):
      self.outer_layers = VPro_Outer_Layers()
      self.last_layer = VPro_Last_Layer()

   
    def __call__(self, x):
        x = self.outer_layers(x)
        x = self.last_layer(x)
        return x
    
    def apply_outer_layers(self,x):
      x = self.outer_layers(x)
      return x


class VPro_Outer_Layers(nn.Module):
        @nn.compact
        def __call__(self,x):
             x = nn.Dense(features=256,use_bias = False)(x)
             x = nn.relu(x)
             x = nn.Dense(features = 256,use_bias = False)(x)
             x = nn.relu(x)
             x = nn.Dense(features=20,use_bias = False)(x)
             x = nn.relu(x)
             return x
        
class VPro_Last_Layer(nn.Module):
    @nn.compact
    def __call__(self,x):
        x = nn.Dense(features=1, use_bias = False)(x)
        return x

          
