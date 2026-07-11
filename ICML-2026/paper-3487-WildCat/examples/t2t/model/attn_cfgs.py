"""Attention configurations for the T2T-ViT attention experiments.

These configurations are hard-coded with the settings used by Insu.
"""

from argparse import ArgumentParser


def get_attn_cfg1(attn1: str, args: ArgumentParser | None = None) -> dict:
    """Get the attention configuration for the first attention layer.
    Baselines are hard-coded with the settings used by Insu.

    Args:
        attn1 (str): name of the attention method
        args (argparse.Namespace | None): command line arguments to be used
            to configure wildcat, optional for other methods

    """
    if attn1 == "full":
        return {"name": "full"}

    elif attn1 == "performer":
        return {
            "name": "performer",
            "dim_heads": 64,
            "nb_features": 49,
            "softmax_eps": 0.0,
            "normalization_eps": 0.0,
        }

    elif attn1 == "reformer":
        return {
            "name": "reformer",
            "bucket_size": 49,
            "n_hashes": 2,
        }

    elif attn1 == "scatterbrain":
        return {
            "name": "scatterbrain",
            "dim_heads": 64,
            "local_context": 49,
            "nb_features": 48,
        }

    elif attn1 == "kdeformer":
        return {
            "name": "kdeformer",
            "sample_size": 64,
            "Bucket_size": 32,
        }

    elif attn1 == "thinformer":
        return {
            "name": "thinformer",
            "g": 2,
            "use_torch_spda": False,
        }
    
    elif attn1 == "wildcat":
        return {
            "name": "wildcat",
            "r": args.rank1,
            # "q_kernel": args.q_kernel,
            #"mode": args.mode,
            "num_bins": args.bins1,
            #"dim_bins": args.dim_bins1
        }
    else:
        raise ValueError(f"Invalid attention method: {attn1}")


def get_attn_cfg2(attn2: str, args: ArgumentParser | None = None) -> dict:
    """Get the attention configuration for the first attention layer.
    Baselines are hard-coded with the settings used by Insu.

    Args:
        attn2 (str): name of the attention method
        args (argparse.Namespace | None): command line arguments to be used
            to configure thinformer, optional for other methods

    """
    if attn2 == "full":
        return {"name": "full"}

    elif attn2 == "performer":
        return {
            "name": "performer",
            "dim_heads": 64,
            "nb_features": 64,
            #"nb_features": 12,
            "softmax_eps": 0.0,
            "normalization_eps": 0.0,
        }

    elif attn2 == "reformer":
        return {
            "name": "reformer",
            "bucket_size": 12,
            "n_hashes": 2,
        }

    elif attn2 == "scatterbrain":
        return {
            "name": "scatterbrain",
            "dim_heads": 64,
            "local_context": 12,
            "nb_features": 6,
        }

    elif attn2 == "kdeformer":
        return {
            "name": "kdeformer",
            #"sample_size": 56,
            "sample_size": 64,
            "Bucket_size": 32,
        }

    elif attn2 == "thinformer":
        return {
            "name": "thinformer",
            "g": 4,
            "use_torch_spda": False,
        }
    elif attn2 == "wildcat":
        return {
            "name": "wildcat",
            "r": args.rank2, 
            # "q_kernel": args.q_kernel,
            #"mode": args.mode,
            "num_bins": args.bins2,
            #"dim_bins": args.dim_bins2
        }
    else:
        raise ValueError(f"Invalid attention method: {attn2}")
