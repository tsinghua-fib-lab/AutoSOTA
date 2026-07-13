from flax import linen as nn

# Standard ReLU MLP
class Two_Layer_ReLU_MLP(nn.Module):
  
  @nn.compact
  def __call__(self, x):
    x =  nn.Dense(features=20, use_bias=False)(x)
    x = nn.relu(x)
    x = nn.Dense(features=1, use_bias=False)(x)
    return x