import jax
import jax.numpy as jnp
import flax.linen as nn

class FCNActor(nn.Module):
    @nn.compact
    def __call__(self, z):
        x = jnp.expand_dims(z, -1) 
        
        # Conv Block 1
        x = nn.Conv(features=32, kernel_size=(3, 3), strides=(2, 2), padding='SAME')(x)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)
        
        # Conv Block 2
        x = nn.Conv(features=64, kernel_size=(3, 3), strides=(2, 2), padding='SAME')(x)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)
        
        # Conv Block 3
        x = nn.Conv(features=128, kernel_size=(3, 3), strides=(2, 2), padding='SAME')(x)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)
        
        # Final Output Layer: initialized with tiny weights to prevent early saturation
        init_fn = jax.nn.initializers.variance_scaling(scale=0.01, mode="fan_in", distribution="uniform")
        x = nn.Conv(features=1, kernel_size=(3, 3), strides=(1, 1), padding='SAME', kernel_init=init_fn)(x)
        
        x = x.reshape((*x.shape[:-3], 64))
        
        return jnp.tanh(x)

class FCNCritic(nn.Module):
    @nn.compact
    def __call__(self, z, actions):
        x = jnp.expand_dims(z, -1)
        
        # State Features (Downsample 64x64 -> 8x8)
        x = nn.Conv(features=32, kernel_size=(3, 3), strides=(2, 2), padding='SAME')(x)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)
        
        x = nn.Conv(features=64, kernel_size=(3, 3), strides=(2, 2), padding='SAME')(x)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)
        
        state_features = nn.Conv(features=128, kernel_size=(3, 3), strides=(2, 2), padding='SAME')(x)
        state_features = nn.LayerNorm()(state_features)
        state_features = nn.relu(state_features) 
        
        # Spatial Concatenation: Actions are perfectly mapped to their physical locations
        act_spatial = actions.reshape((*actions.shape[:-1], 8, 8, 1))
        xu = jnp.concatenate([state_features, act_spatial], axis=-1)
        
        # Fully Convolutional Q-Network (No Dense layers, No Flattening)
        def build_q(features):
            q = nn.Conv(features=256, kernel_size=(3, 3), padding='SAME')(features)
            q = nn.LayerNorm()(q)
            q = nn.relu(q)
            
            q = nn.Conv(features=256, kernel_size=(3, 3), padding='SAME')(q)
            q = nn.LayerNorm()(q)
            q = nn.relu(q)
            
            # Map down to exactly 1 Q-value per grid cell (Shape: 8x8x1)
            q = nn.Conv(features=1, kernel_size=(3, 3), padding='SAME')(q)
            
            # Global Average Pooling: Average the local Q-values into one Global Q-value
            # Output shape becomes (batch, 1) matching the reward shape exactly
            return jnp.mean(q, axis=(-3, -2))
        
        q1 = build_q(xu)
        q2 = build_q(xu)
        
        return q1, q2