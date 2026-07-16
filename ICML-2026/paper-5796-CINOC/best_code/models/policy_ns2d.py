"""
NS2D Smoke Control Policy Network

Based on Heat2DControlNet architecture:
- Branch: CNN to process 2D error field (NO POOLING)
- Trunk: Fourier encoding of 2D actuator positions  
- Fusion: Broadcast + concatenate
- Heads: Separate outputs for u (injection intensity) and v (velocity)
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Sequence


class NS2DControlNet(nn.Module):
    """
    Fan-Only Controller for 2D Navier-Stokes Smoke Control.
    
    Agents act as "fans" that push smoke without injecting new smoke.
    Policy outputs only push velocity (v).
    """
    features: Sequence[int] = (16, 32)
    v_max: float = 0.5
    
    def setup(self):
        self.frequencies = jnp.array([1.0, 2.0, 4.0, 8.0])
    
    def branch_net(self, error, error_grad_x, error_grad_y):
        x = jnp.stack([error, error_grad_x, error_grad_y], axis=-1)
        for feat in self.features:
            x = nn.Conv(feat, kernel_size=(3, 3), padding='SAME')(x)
            x = nn.relu(x)
        x = x.reshape(-1)
        x = x / (jnp.linalg.norm(x) + 1e-8)
        x = nn.Dense(64)(x)
        x = nn.tanh(x)
        return x
    
    def trunk_net(self, xi):
        angle_x = xi[:, 0, None] * self.frequencies * jnp.pi
        angle_y = xi[:, 1, None] * self.frequencies * jnp.pi
        encoded = jnp.concatenate([
            jnp.sin(angle_x), jnp.cos(angle_x),
            jnp.sin(angle_y), jnp.cos(angle_y)
        ], axis=-1)
        for feat in [64, 64]:
            encoded = nn.Dense(feat)(encoded)
            encoded = nn.tanh(encoded)
        return encoded
    
    @nn.compact
    def __call__(self, smoke_curr, smoke_target, xi_curr):
        xi_curr = jnp.clip(xi_curr, 0.0, 1.0)
        error = smoke_curr - smoke_target
        error_grad = jnp.gradient(error)
        error_grad_x = error_grad[1]
        error_grad_y = error_grad[0]
        
        branch_out = self.branch_net(error, error_grad_x, error_grad_y)
        trunk_out = self.trunk_net(xi_curr)
        
        branch_repeated = jnp.tile(branch_out, (xi_curr.shape[0], 1))
        combined = jnp.concatenate([branch_repeated, trunk_out], axis=-1)
        
        h = nn.Dense(64)(combined)
        h = nn.tanh(h)
        v_raw = nn.Dense(2)(h)
        v = self.v_max * jnp.tanh(v_raw)
        
        return v


class DecentralizedNS2DControlNet(nn.Module):
    """
    Decentralized NS2D Smoke Controller - Redesigned Observation.
    
    Each agent receives:
    - Local SMOKE patch (not error!) - "Is there smoke near me?"
    - Direction to TARGET centroid - "Where should smoke go?"
    - Direction to SMOKE centroid - "Where is smoke?" (for exploration)
    - Distance to smoke - "Am I near the action?"
    """
    features: Sequence[int] = (16, 32)
    v_max: float = 0.5
    patch_size: int = 12
    
    def setup(self):
        self.frequencies = jnp.array([1.0, 2.0, 4.0, 8.0])
    
    def compute_centroid(self, field):
        """Compute weighted centroid of a field."""
        Nx, Ny = field.shape
        x_coords = jnp.linspace(0, 1, Nx)
        y_coords = jnp.linspace(0, 1.25, Ny)
        X, Y = jnp.meshgrid(x_coords, y_coords, indexing='ij')
        
        total_mass = jnp.sum(field) + 1e-8
        cx = jnp.sum(field * X) / total_mass
        cy = jnp.sum(field * Y) / total_mass
        
        return jnp.array([cx, cy])
    
    def extract_local_patch(self, field, xi, n_grid_x, n_grid_y):
        """Extract local patch around agent position."""
        i = jnp.clip((xi[0] * (n_grid_x - 1)).astype(int), 0, n_grid_x-1)
        j = jnp.clip((xi[1] * (n_grid_y - 1)).astype(int), 0, n_grid_y-1)
        half_patch = self.patch_size // 2
        padded_field = jnp.pad(field, ((half_patch, half_patch), (half_patch, half_patch)), mode='edge')
        patch = jax.lax.dynamic_slice(padded_field, (i, j), (self.patch_size, self.patch_size))
        return patch
    
    def branch_net(self, local_patch):
        """Process local smoke patch through CNN."""
        x = local_patch[..., None]  # Add channel dim: (P, P, 1)
        for feat in self.features:
            x = nn.Conv(feat, kernel_size=(3, 3), padding='SAME')(x)
            x = nn.relu(x)
        x = x.reshape(-1)
        x = x / (jnp.linalg.norm(x) + 1.0)  # Safe normalization
        x = nn.Dense(32)(x)
        x = nn.tanh(x)
        return x
    
    def trunk_net(self, xi, target_centroid, smoke_centroid):
        """Encode position + directions to target AND smoke."""
        # Fourier encoding of own position
        angle_x = xi[:, 0, None] * self.frequencies * jnp.pi
        angle_y = xi[:, 1, None] * self.frequencies * jnp.pi
        pos_encoded = jnp.concatenate([
            jnp.sin(angle_x), jnp.cos(angle_x),
            jnp.sin(angle_y), jnp.cos(angle_y)
        ], axis=-1)  # (M, 16)
        
        # Direction to TARGET (where smoke should go)
        dir_to_target = target_centroid - xi
        dist_to_target = jnp.linalg.norm(dir_to_target, axis=-1, keepdims=True) + 1e-8
        dir_to_target_norm = dir_to_target / dist_to_target
        
        # Direction to SMOKE (for exploration - find the smoke)
        dir_to_smoke = smoke_centroid - xi
        dist_to_smoke = jnp.linalg.norm(dir_to_smoke, axis=-1, keepdims=True) + 1e-8
        dir_to_smoke_norm = dir_to_smoke / dist_to_smoke
        
        # Combine all: position + target info + smoke info
        encoded = jnp.concatenate([
            pos_encoded,           # (M, 16)
            dir_to_target,         # (M, 2) raw direction to target
            dir_to_target_norm,    # (M, 2) unit direction to target
            dist_to_target,        # (M, 1) distance to target
            dir_to_smoke,          # (M, 2) raw direction to smoke
            dir_to_smoke_norm,     # (M, 2) unit direction to smoke  
            dist_to_smoke,         # (M, 1) distance to smoke
        ], axis=-1)  # (M, 26)
        
        for feat in [32, 32]:
            encoded = nn.Dense(feat)(encoded)
            encoded = nn.tanh(encoded)
        return encoded
    
    @nn.compact  
    def __call__(self, smoke_curr, smoke_target, xi_curr):
        xi_curr = jnp.clip(xi_curr, 0.0, 1.0)
        n_grid_x, n_grid_y = smoke_curr.shape
        
        # Compute centroids (global info, but just 4 floats total!)
        target_centroid = self.compute_centroid(smoke_target)
        smoke_centroid = self.compute_centroid(smoke_curr)
        
        # Extract local SMOKE patches (not error!)
        def get_local_smoke(xi_single):
            return self.extract_local_patch(smoke_curr, xi_single, n_grid_x, n_grid_y)
        
        local_patches = jax.vmap(get_local_smoke)(xi_curr)  # (M, P, P)
        branch_outs = jax.vmap(self.branch_net)(local_patches)
        trunk_outs = self.trunk_net(xi_curr, target_centroid, smoke_centroid)
        
        combined = jnp.concatenate([branch_outs, trunk_outs], axis=-1)
        h = nn.Dense(64)(combined)
        h = nn.tanh(h)
        v_raw = nn.Dense(2)(h)
        v = self.v_max * jnp.tanh(v_raw)
        
        return v


class StarVisionNS2DControlNet(nn.Module):
    """
    Star Vision Decentralized NS2D Smoke Controller.
    
    Each agent sees 8 fixed rays emanating from its position:
    - Horizontal (left/right)
    - Vertical (up/down)  
    - 4 Diagonals (45 degrees)
    
    No velocity dependency = stable training.
    """
    features: Sequence[int] = (64, 64)
    v_max: float = 0.5
    ray_length: int = 32
    
    def setup(self):
        self.frequencies = jnp.array([1.0, 2.0, 4.0, 8.0])
        sqrt2 = 0.7071
        self.directions = jnp.array([
            [1.0, 0.0],   [0.0, 1.0],   [-1.0, 0.0],  [0.0, -1.0],
            [sqrt2, sqrt2], [-sqrt2, sqrt2], [-sqrt2, -sqrt2], [sqrt2, -sqrt2],
        ])
    
    def sample_ray(self, field, start_pos, direction, n_samples, n_grid_x, n_grid_y):
        """Sample field along a ray from agent position to domain boundary."""
        # Compute intersection with domain boundary [0,1] x [0,1.25]
        dx, dy = direction[0], direction[1]
        x0, y0 = start_pos[0], start_pos[1]
        
        # Find t_max where ray hits boundary (parametric form: x = x0 + t*dx)
        # Safe defaults
        t_max = 1.5  # Maximum possible extent
        
        # X boundaries
        t_max = jnp.where(dx > 0.01, jnp.minimum(t_max, (1.0 - x0) / dx), t_max)
        t_max = jnp.where(dx < -0.01, jnp.minimum(t_max, -x0 / dx), t_max)
        
        # Y boundaries (domain is [0, 1.25])
        t_max = jnp.where(dy > 0.01, jnp.minimum(t_max, (1.25 - y0) / dy), t_max)
        t_max = jnp.where(dy < -0.01, jnp.minimum(t_max, -y0 / dy), t_max)
        
        t_max = jnp.clip(t_max, 0.1, 1.5)  # Ensure reasonable range
        
        # Sample along ray from start to boundary
        t = jnp.linspace(0, t_max, n_samples)
        ray_x = x0 + t * dx
        ray_y = y0 + t * dy
        
        # Convert to grid indices (clamped)
        ix = jnp.clip(jnp.round(ray_x * (n_grid_x - 1)), 0, n_grid_x - 1).astype(jnp.int32)
        iy = jnp.clip(jnp.round(ray_y / 1.25 * (n_grid_y - 1)), 0, n_grid_y - 1).astype(jnp.int32)
        
        return field[ix, iy]
    
    def extract_star_observation(self, error, xi, n_grid_x, n_grid_y):
        """Extract rays in all 8 directions."""
        def sample_one_dir(d):
            return self.sample_ray(error, xi, d, self.ray_length, n_grid_x, n_grid_y)
        all_rays = jax.vmap(sample_one_dir)(self.directions)
        return all_rays.reshape(-1)
    
    def branch_net(self, obs):
        x = obs / (jnp.linalg.norm(obs) + 1.0)
        for feat in self.features:
            x = nn.Dense(feat)(x)
            x = nn.tanh(x)
        return x
    
    def trunk_net(self, xi):
        angle_x = xi[:, 0, None] * self.frequencies * jnp.pi
        angle_y = xi[:, 1, None] * self.frequencies * jnp.pi
        encoded = jnp.concatenate([jnp.sin(angle_x), jnp.cos(angle_x),
                                   jnp.sin(angle_y), jnp.cos(angle_y)], axis=-1)
        for feat in [32, 32]:
            encoded = nn.Dense(feat)(encoded)
            encoded = nn.tanh(encoded)
        return encoded
    
    @nn.compact  
    def __call__(self, smoke_curr, smoke_target, xi_curr, prev_vel=None):
        xi_curr = jnp.clip(xi_curr, 0.0, 1.0)
        error = jnp.clip(smoke_curr - smoke_target, -5.0, 5.0)
        n_grid_x, n_grid_y = smoke_curr.shape
        
        star_obs = jax.vmap(lambda xi: self.extract_star_observation(
            error, xi, n_grid_x, n_grid_y))(xi_curr)
        
        branch_outs = jax.vmap(self.branch_net)(star_obs)
        trunk_outs = self.trunk_net(xi_curr)
        
        combined = jnp.concatenate([branch_outs, trunk_outs], axis=-1)
        h = nn.tanh(nn.Dense(64)(combined))
        v = self.v_max * jnp.tanh(nn.Dense(2)(h))
        
        return v


# Aliases for compatibility
PlusVisionNS2DControlNet = StarVisionNS2DControlNet
VelocityDirectedVisionNS2DControlNet = StarVisionNS2DControlNet
