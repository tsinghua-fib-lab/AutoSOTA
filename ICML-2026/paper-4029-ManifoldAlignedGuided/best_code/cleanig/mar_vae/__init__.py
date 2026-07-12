"""
MAR VAE module for cleanig.
Provides VAE models compatible with the Manifold AGI implementation.
"""

from .vae import AutoencoderKL
from .utils import download_pretrained_vae, load_partial_pretrained_weights

# Create MAR_VAE_models dictionary for compatibility with test scripts
def mar_base(img_size=256, patch_size=16, **kwargs):
    """
    Create a MAR VAE base model.
    
    Args:
        img_size: Image size (not used, kept for compatibility)
        patch_size: Patch size (not used, kept for compatibility)
        **kwargs: Additional arguments
    
    Returns:
        AutoencoderKL instance with MAR configuration
    """
    # MAR base configuration
    return AutoencoderKL(
        embed_dim=16,
        ch_mult=[1, 1, 2, 2, 4],
        use_variational=True
    )

# Dictionary of available models
MAR_VAE_models = {
    'mar_base': mar_base
}

__all__ = [
    'AutoencoderKL',
    'MAR_VAE_models',
    'download_pretrained_vae',
    'load_partial_pretrained_weights'
]