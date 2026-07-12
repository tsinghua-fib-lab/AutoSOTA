import torch
import os
import requests
from tqdm import tqdm


def download_pretrained_vae(
    overwrite=False, 
    download_dir="pretrained_models/vae",
    ckpt_name="kl16.ckpt"
):
    if not os.path.exists(os.path.join(download_dir, ckpt_name)) or overwrite:
        headers = {'user-agent': 'Wget/1.16 (linux-gnu)'}
        r = requests.get(f"https://www.dropbox.com/scl/fi/hhmuvaiacrarfg28qxhwz/{ckpt_name}?rlkey=l44xipsezc8atcffdp4q7mwmh&dl=0", stream=True, headers=headers)
        print("Downloading KL-16 VAE...")
        with open(os.path.join(download_dir, ckpt_name), 'wb') as f:
            for chunk in tqdm(r.iter_content(chunk_size=1024*1024), unit="MB", total=254):
                if chunk:
                    f.write(chunk)


def load_partial_pretrained_weights(model, ckpt_path, target_z_channels=16, source_z_channels=16):
    """
    Load pretrained weights with partial channel loading when z_channels don't match.
    
    Args:
        model: The target model to load weights into
        ckpt_path: Path to the pretrained checkpoint
        target_z_channels: Number of z_channels in the target model (e.g., 8)
        source_z_channels: Number of z_channels in the source pretrained model (e.g., 16)
    """
    print(f"\nLoading pretrained weights from {ckpt_path}")
    print(f"Source z_channels: {source_z_channels}, Target z_channels: {target_z_channels}")
    
    # Load pretrained state dict
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    pretrained_sd = checkpoint["model"]
    
    # Get current model state dict
    model_sd = model.state_dict()
    
    # Track what we're loading
    loaded_keys = []
    partial_keys = []
    skipped_keys = []
    
    for key, pretrained_value in pretrained_sd.items():
        if key in model_sd:
            model_value = model_sd[key]
            
            # Check if shapes match
            if pretrained_value.shape == model_value.shape:
                # Perfect match - load directly
                model_sd[key] = pretrained_value
                loaded_keys.append(key)
            else:
                # Shape mismatch - need special handling
                # These are likely the conv layers dealing with z_channels
                if 'quant_conv' in key or 'post_quant_conv' in key:
                    # For quant_conv and post_quant_conv layers
                    print(f"\nHandling layer: {key}")
                    print(f"  Pretrained shape: {pretrained_value.shape}")
                    print(f"  Model shape: {model_value.shape}")
                    
                    if 'weight' in key:
                        # Conv weight shape: [out_channels, in_channels, kernel_h, kernel_w]
                        if 'quant_conv' in key:
                            # quant_conv: input is from encoder (2*embed_dim channels)
                            # output depends on use_variational (2*embed_dim or 1*embed_dim)
                            # We need to handle both input and output channel differences
                            
                            # Take only the first target_z_channels from both dimensions
                            if len(pretrained_value.shape) == 4:  # Conv2d weight
                                # Determine how many channels to copy
                                out_channels_to_copy = min(pretrained_value.shape[0], model_value.shape[0])
                                in_channels_to_copy = min(pretrained_value.shape[1], model_value.shape[1])
                                
                                # Copy the subset of channels
                                model_sd[key][:out_channels_to_copy, :in_channels_to_copy, :, :] = \
                                    pretrained_value[:out_channels_to_copy, :in_channels_to_copy, :, :]
                                
                                partial_keys.append(key)
                                print(f"  Partially loaded: [{out_channels_to_copy}, {in_channels_to_copy}] channels")
                        
                        elif 'post_quant_conv' in key:
                            # post_quant_conv: both input and output are embed_dim
                            if len(pretrained_value.shape) == 4:  # Conv2d weight
                                # Take only the first target_z_channels
                                channels_to_copy = min(target_z_channels, source_z_channels)
                                model_sd[key][:channels_to_copy, :channels_to_copy, :, :] = \
                                    pretrained_value[:channels_to_copy, :channels_to_copy, :, :]
                                
                                partial_keys.append(key)
                                print(f"  Partially loaded: {channels_to_copy} channels")
                    
                    elif 'bias' in key:
                        # Bias shape: [out_channels]
                        channels_to_copy = min(pretrained_value.shape[0], model_value.shape[0])
                        model_sd[key][:channels_to_copy] = pretrained_value[:channels_to_copy]
                        partial_keys.append(key)
                        print(f"  Partially loaded bias: {channels_to_copy} channels")
                
                elif 'encoder.conv_out' in key or 'decoder.conv_in' in key:
                    # These layers interface with the latent space
                    print(f"\nHandling layer: {key}")
                    print(f"  Pretrained shape: {pretrained_value.shape}")
                    print(f"  Model shape: {model_value.shape}")
                    
                    if 'weight' in key and len(pretrained_value.shape) == 4:
                        if 'encoder.conv_out' in key:
                            # Encoder output: produces 2*z_channels (for mean and logvar)
                            # Weight shape: [2*z_channels, in_channels, k, k]
                            out_channels_to_copy = min(pretrained_value.shape[0], model_value.shape[0])
                            model_sd[key][:out_channels_to_copy, :, :, :] = \
                                pretrained_value[:out_channels_to_copy, :, :, :]
                            partial_keys.append(key)
                            print(f"  Partially loaded encoder.conv_out: {out_channels_to_copy} out channels")
                        
                        elif 'decoder.conv_in' in key:
                            # Decoder input: takes z_channels
                            # Weight shape: [out_channels, z_channels, k, k]
                            in_channels_to_copy = min(pretrained_value.shape[1], model_value.shape[1])
                            model_sd[key][:, :in_channels_to_copy, :, :] = \
                                pretrained_value[:, :in_channels_to_copy, :, :]
                            partial_keys.append(key)
                            print(f"  Partially loaded decoder.conv_in: {in_channels_to_copy} in channels")
                    
                    elif 'bias' in key:
                        if 'encoder.conv_out' in key:
                            channels_to_copy = min(pretrained_value.shape[0], model_value.shape[0])
                            model_sd[key][:channels_to_copy] = pretrained_value[:channels_to_copy]
                            partial_keys.append(key)
                            print(f"  Partially loaded bias: {channels_to_copy} channels")
                        else:
                            # For decoder.conv_in bias, we can load it fully as output channels match
                            model_sd[key] = pretrained_value
                            loaded_keys.append(key)
                else:
                    # Other shape mismatches - skip
                    skipped_keys.append(key)
                    print(f"  Skipping {key} due to shape mismatch")
        else:
            # Key not in model - skip
            skipped_keys.append(key)
    
    # Load the modified state dict
    model.load_state_dict(model_sd, strict=False)
    
    # Print summary
    print(f"\n=== Weight Loading Summary ===")
    print(f"Fully loaded layers: {len(loaded_keys)}")
    print(f"Partially loaded layers: {len(partial_keys)}")
    if partial_keys:
        print(f"Partial layers: {partial_keys}")
    print(f"Skipped layers: {len(skipped_keys)}")
    
    # Print some specific important layers
    print(f"\n=== Key Layer Status ===")
    important_layers = ['encoder.conv_in', 'encoder.conv_out', 'decoder.conv_in', 'decoder.conv_out',
                       'quant_conv', 'post_quant_conv']
    for layer_name in important_layers:
        matching_keys = [k for k in model_sd.keys() if layer_name in k and 'weight' in k]
        for key in matching_keys:
            if key in loaded_keys:
                status = "✓ Fully loaded"
            elif key in partial_keys:
                status = "⚠ Partially loaded"
            else:
                status = "✗ Not loaded"
            print(f"{key}: {status}")
    
    return model