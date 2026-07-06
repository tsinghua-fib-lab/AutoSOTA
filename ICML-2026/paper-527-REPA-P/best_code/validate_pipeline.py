import yaml, torch, sys
from pathlib import Path
sys.path.insert(0, '/repo')

# Test config parsing
with open("/repo/configs/model.darcy.repap.repro.yaml") as f:
    config = yaml.safe_load(f.read())
print(f"Config loaded: gov_eqs={config['gov_eqs']}, dim=32, diff_steps={config['diff_steps']}")

# Test model creation
from src_pr.unet_new import Unet3D
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Unet3D(
    dim=32,
    channels=2,
    sigmoid_last_channel=False,
    use_projection_heads=True,
    projection_positions=["bottleneck"],
    projection_hidden_dim=128
).to(device)
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Model created: {num_params:,} parameters")

# Test DenoisingDiffusion
from src_pr.denoising_utils import DenoisingDiffusion
diffusion = DenoisingDiffusion(100, device, residual_grad_guidance=False)
print(f"Diffusion created with 100 steps")

print("Pipeline validation OK")
