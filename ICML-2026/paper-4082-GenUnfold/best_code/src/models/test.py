import torch

from diffusion_model import create_diffusion

from components.diffusion_transformer import DiT1D

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# --- Test DiT1D without class conditioning ---
print("\n--- Testing DiT1D (No Class Conditioning) ---")
seq_length = 64
patch_s = 4
in_chans = 1
hidden_s = 128
n_heads = 4
n_classes_test = 10  # No class conditioning
batch_size = 8

model = DiT1D(
    seq_len=seq_length,
    patch_size=patch_s,
    in_channels=in_chans,
    hidden_size=hidden_s,
    num_heads=n_heads,
    condition_feature_size=n_classes_test,  # Set to 0
    depth=2  # Small depth for quick test
).to(device)


diffusion = create_diffusion(timestep_respacing='')

x = torch.randn(batch_size, in_chans, seq_length).to(device)

y = torch.randn(batch_size, n_classes_test).to(device)


t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)

print(model(x, t, y).shape)

model_kwargs = dict(y=y)

loss_dict = diffusion.training_losses(model, x, t, model_kwargs)
print(loss_dict)