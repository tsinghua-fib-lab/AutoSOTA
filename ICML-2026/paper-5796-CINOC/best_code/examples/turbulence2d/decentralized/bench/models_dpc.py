import jax
import jax.numpy as jnp
import flax.linen as nn

class CentralizedFCNControlNet2D_Turb(nn.Module):
    """
    Single-agent (Centralized) Fully Convolutional Controller for 2D Turbulence.
    Maps the 64x64 vorticity field directly to an 8x8 grid of forcing commands
    using strided convolutions to preserve spatial alignment.
    """
    u_max: float = 75.0

    @nn.compact
    def __call__(self, xi, z):
        # z is the 2D vorticity field (..., N_grid, N_grid)
        # xi is the actuator positions (..., n_agents, 2) - ignored in FCN 
        # as the network infers locations naturally via spatial downsampling.
        
        x = jnp.expand_dims(z, -1) 
        
        # Conv Block 1: (64, 64) -> (32, 32)
        x = nn.Conv(features=32, kernel_size=(3, 3), strides=(2, 2), padding='SAME')(x)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)
        
        # Conv Block 2: (32, 32) -> (16, 16)
        x = nn.Conv(features=64, kernel_size=(3, 3), strides=(2, 2), padding='SAME')(x)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)
        
        # Conv Block 3: (16, 16) -> (8, 8)
        x = nn.Conv(features=128, kernel_size=(3, 3), strides=(2, 2), padding='SAME')(x)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)
        
        # Final Output Layer: ZERO initialization to start as uncontrolled 
        # (critical for BPTT stability in chaotic turbulence).
        zero_init = nn.initializers.zeros
        x = nn.Conv(features=1, kernel_size=(3, 3), strides=(1, 1), padding='SAME', 
                    kernel_init=zero_init, bias_init=zero_init)(x)
        
        # Reshape from (..., 8, 8, 1) to a flat actuator array (..., 64)
        x = x.reshape((*x.shape[:-3], 64))
        
        return self.u_max * jnp.tanh(x)