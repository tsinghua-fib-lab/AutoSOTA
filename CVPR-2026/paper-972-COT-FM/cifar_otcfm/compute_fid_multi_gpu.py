# modified from torchcfm
import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from glob import glob
from torchvision import models, transforms
from scipy import linalg
from torchdiffeq import odeint
from torchdyn.core import NeuralODE
import torch.distributed as dist
from cleanfid import fid

from torchcfm.models.unet.unet import UNetModelWrapper
from ppo_utils import sample_noise
from absl import app, flags
from prdc import compute_prdc
import torchvision

FLAGS = flags.FLAGS

# -------------------------
# absl flags definitions
# -------------------------
flags.DEFINE_integer("batch_size_fid", 1024, "Batch size for FID computation.")
flags.DEFINE_string("integration_method", "euler", "ODE solver: euler or dopri5.")
flags.DEFINE_integer("euler_steps", 50, "Euler steps (only used for euler).")
flags.DEFINE_string("path_pattern", "./results/*/*400000.pt", "Checkpoint path glob pattern.")
flags.DEFINE_integer("fid_device_id", 0, "Device id used for CleanFID.")
flags.DEFINE_integer("num_gen", 50000, "Number of generated samples for FID.")
flags.DEFINE_integer("inception_batch_size", 256, "Batch size for Inception feature extraction.")


# -------------------------
# ImageNet normalization
# -------------------------
imagenet_normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)


# -------------------------
# InceptionWrapper
# -------------------------
class InceptionWrapper(torch.nn.Module):
    """
    Wrapper to extract pool3 features (FID) from InceptionV3.
    Supports batched inference to avoid OOM.
    """
    def __init__(self, device="cuda", batch_size=256):
        super().__init__()
        inception = models.inception_v3(pretrained=True, transform_input=False)
        inception.fc = torch.nn.Identity()
        inception.eval()
        self.inception = inception.to(device)
        self.device = device
        self.batch_size = batch_size

    @torch.no_grad()
    def get_features(self, x: torch.Tensor):
        """
        Args:
            x: (N, 3, H, W) in [0, 255] uint8
        Returns:
            features: (N, 2048) on the same device as input
        """
        # Convert to float and normalize to [0, 1]
        x = x.float() / 255.0
        
        # Resize to 299x299 for Inception
        x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)

        features_list = []

        for i in range(0, len(x), self.batch_size):
            batch = x[i:i + self.batch_size]
            # Apply ImageNet normalization
            batch = torch.stack([imagenet_normalize(img) for img in batch], dim=0).to(self.device)

            feats = self.inception(batch)
            features_list.append(feats)

            torch.cuda.empty_cache()

        features = torch.cat(features_list, dim=0)
        return features


# -------------------------
# FID calculation
# -------------------------
def calculate_fid(real_feats, fake_feats):
    """
    Calculate FID between real and fake features.
    Args:
        real_feats: numpy array (N, 2048)
        fake_feats: numpy array (N, 2048)
    Returns:
        fid: float
    """
    mu1, sigma1 = real_feats.mean(axis=0), np.cov(real_feats, rowvar=False)
    mu2, sigma2 = fake_feats.mean(axis=0), np.cov(fake_feats, rowvar=False)

    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
    return float(fid)


# -------------------------
# distributed setup
# -------------------------
def setup_distributed():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl")
        return True, rank, world_size, local_rank
    return False, 0, 1, 0


# ------------------------------------------------
# main logic wrapped into main(argv)
# ------------------------------------------------
def main(argv):
    # load FLAGS
    batch_size_fid = FLAGS.batch_size_fid
    integration_method = FLAGS.integration_method
    euler_steps = FLAGS.euler_steps
    path_pattern = FLAGS.path_pattern
    path_pattern = f"./results/{path_pattern}/*300000.pt"
    result_csv_path = f"./fid_{FLAGS.path_pattern}_{integration_method}_{euler_steps}.csv"
    fid_device_id = FLAGS.fid_device_id
    inception_batch_size = FLAGS.inception_batch_size

    # setup distributed
    is_distributed, rank, world_size, local_rank = setup_distributed()
    local_device = f"cuda:{local_rank}" if is_distributed else "cuda:0"

    # model
    new_net = UNetModelWrapper(
        dim=(3, 32, 32),
        num_res_blocks=4,
        num_channels=128,
        channel_mult=[1, 2, 2, 2],
        num_heads=4,
        num_head_channels=64,
        attention_resolutions="16",
        dropout=0.1,
    ).to(local_device)
    
    # Each rank has its own Inception feature extractor
    feat_extractor = InceptionWrapper(device=local_device, batch_size=inception_batch_size)

    # load checkpoints
    path_list = glob(path_pattern)

    if not os.path.exists(result_csv_path):
        pd.DataFrame(columns=["path", "fid-50k", "precision", "recall"]).to_csv(result_csv_path, index=False)

    df = pd.read_csv(result_csv_path)

    if len(path_list) == 0:
        if rank == 0:
            print("No new checkpoints to evaluate.")
        return

    path = path_list[0]
    if rank == 0:
        print("Evaluate path:", path)

    checkpoint = torch.load(path, map_location=local_device)
    state_dict = checkpoint["ema_model"]

    try:
        new_net.load_state_dict(state_dict)
    except RuntimeError:
        # remove "model." prefix
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            new_state_dict[k[7:]] = v
        new_net.load_state_dict(new_state_dict)

    new_net.eval()

    # solver
    if integration_method == "euler":
        node = NeuralODE(new_net, solver=integration_method).to(local_device)

    # --------------------------------------------
    # generator wrapper (must be inside main)
    # --------------------------------------------
    def gen_1_img(unused_latent):
        local_batch = batch_size_fid // world_size if is_distributed else batch_size_fid

        with torch.no_grad():
            if path_pattern.find("cotfm") != -1:
                x = sample_noise(batch_size=local_batch, device=local_device)
            else:
                x = torch.randn(local_batch, 3, 32, 32, device=local_device)
            if integration_method == "euler":
                t_span = torch.linspace(0, 1, euler_steps + 1, device=local_device)
                traj = node.trajectory(x, t_span=t_span)
            else:
                t_span = torch.linspace(0, 1, 2, device=local_device)
                traj = odeint(new_net, x, t_span, rtol=1e-5, atol=1e-5, method=integration_method)

        img = (traj[-1] * 127.5 + 128).clip(0, 255).to(torch.uint8)

        # distributed gather
        if not is_distributed:
            return img

        # gather from all ranks
        gathered = [torch.zeros_like(img) for _ in range(world_size)]
        dist.all_gather(gathered, img)
        return torch.cat(gathered, dim=0)

    # --------------------------------------------
    # Extract real features (multi-GPU)
    # --------------------------------------------
    if rank == 0:
        print("Extracting real features on all GPUs...")
    
    # Load CIFAR-10 dataset
    dataset = torchvision.datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
    )
    
    # Calculate samples per rank
    samples_per_rank = FLAGS.num_gen // world_size if is_distributed else FLAGS.num_gen
    remainder = FLAGS.num_gen % world_size if is_distributed else 0
    
    # Adjust for remainder
    if is_distributed and rank < remainder:
        local_num_samples = samples_per_rank + 1
        start_idx = rank * (samples_per_rank + 1)
    elif is_distributed:
        local_num_samples = samples_per_rank
        start_idx = remainder * (samples_per_rank + 1) + (rank - remainder) * samples_per_rank
    else:
        local_num_samples = samples_per_rank
        start_idx = 0
    
    # Extract features for this rank's portion of CIFAR-10
    real_images_list = []
    for i in range(start_idx, start_idx + local_num_samples):
        img, _ = dataset[i]
        # Convert PIL to tensor [3, 32, 32] and scale to [0, 255]
        img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1)
        real_images_list.append(img_tensor)
    
    real_images = torch.stack(real_images_list).to(local_device)
    real_feats_local = feat_extractor.get_features(real_images)
    
    if rank == 0:
        print(f"Rank {rank}: extracted {len(real_feats_local)} real features")
    
    # --------------------------------------------
    # Extract fake features (multi-GPU)
    # --------------------------------------------
    if rank == 0:
        print("Extracting fake features on all GPUs...")
    
    fake_images_list = []
    num_batches = (local_num_samples + (batch_size_fid // world_size) - 1) // (batch_size_fid // world_size) if is_distributed else \
                  (local_num_samples + batch_size_fid - 1) // batch_size_fid
    from tqdm import tqdm
    with torch.no_grad():
        for _ in tqdm(range(num_batches)):
            local_batch = min(batch_size_fid // world_size if is_distributed else batch_size_fid,
                            local_num_samples - len(fake_images_list))
            
            if local_batch <= 0:
                break
            
            if path_pattern.find("cotfm") != -1:
                x = sample_noise(batch_size=local_batch, device=local_device)
            else:
                x = torch.randn(local_batch, 3, 32, 32, device=local_device)
            
            if integration_method == "euler":
                t_span = torch.linspace(0, 1, euler_steps + 1, device=local_device)
                traj = node.trajectory(x, t_span=t_span)
            else:
                t_span = torch.linspace(0, 1, 2, device=local_device)
                traj = odeint(new_net, x, t_span, rtol=1e-5, atol=1e-5, method=integration_method)
            
            imgs = (traj[-1] * 127.5 + 128).clip(0, 255).to(torch.uint8)
            
            for img in imgs:
                fake_images_list.append(img)
                if len(fake_images_list) >= local_num_samples:
                    break
    
    fake_images = torch.stack(fake_images_list[:local_num_samples]).to(local_device)
    fake_feats_local = feat_extractor.get_features(fake_images)
    
    if rank == 0:
        print(f"Rank {rank}: extracted {len(fake_feats_local)} fake features")
    
    # --------------------------------------------
    # Gather all features to rank 0
    # --------------------------------------------
    if is_distributed:
        if rank == 0:
            print("Gathering features from all ranks...")
        
        # Gather real and fake features
        if rank == 0:
            # Create list to hold features from all ranks (on GPU)
            max_size = samples_per_rank + 1  # Maximum possible size per rank
            gathered_real = [torch.zeros(max_size, real_feats_local.shape[1], device=local_device) for _ in range(world_size)]
            gathered_fake = [torch.zeros(max_size, fake_feats_local.shape[1], device=local_device) for _ in range(world_size)]
        else:
            gathered_real = None
            gathered_fake = None
        
        # Pad features to same size for gathering (keep on GPU)
        max_size = samples_per_rank + 1
        real_feats_padded = torch.zeros(max_size, real_feats_local.shape[1], device=local_device)
        real_feats_padded[:len(real_feats_local)] = real_feats_local.to(local_device)
        fake_feats_padded = torch.zeros(max_size, fake_feats_local.shape[1], device=local_device)
        fake_feats_padded[:len(fake_feats_local)] = fake_feats_local.to(local_device)
        
        # Gather to rank 0
        dist.gather(real_feats_padded, gathered_real, dst=0)
        dist.gather(fake_feats_padded, gathered_fake, dst=0)
        
        if rank == 0:
            # Concatenate and trim to exact size
            real_feats_all = []
            fake_feats_all = []
            
            for i in range(world_size):
                if i < remainder:
                    size = samples_per_rank + 1
                else:
                    size = samples_per_rank
                real_feats_all.append(gathered_real[i][:size])
                fake_feats_all.append(gathered_fake[i][:size])
            
            real_feats_final = torch.cat(real_feats_all, dim=0)[:FLAGS.num_gen]
            fake_feats_final = torch.cat(fake_feats_all, dim=0)[:FLAGS.num_gen]
        else:
            real_feats_final = None
            fake_feats_final = None
    else:
        real_feats_final = real_feats_local
        fake_feats_final = fake_feats_local
    
    # --------------------------------------------
    # Compute FID and Precision/Recall (only on rank 0)
    # --------------------------------------------
    if rank == 0:
        print(f"Computing FID with {len(real_feats_final)} samples...")
        
        # Convert to CPU and numpy
        real_feats_np = real_feats_final.cpu().numpy()
        fake_feats_np = fake_feats_final.cpu().numpy()
        
        # Calculate FID
        fid_score = calculate_fid(real_feats_np, fake_feats_np)
        
        # Calculate Precision and Recall
        print(f"Computing precision and recall...")
        prdc_metrics = compute_prdc(
            real_features=real_feats_np,
            fake_features=fake_feats_np,
            nearest_k=5,
        )
        
        precision = prdc_metrics["precision"]
        recall = prdc_metrics["recall"]
        print(f"FID = {fid_score}")
        print(f"Precision = {precision}")
        print(f"Recall = {recall}")
        
        new_row = pd.DataFrame([{"path": path, "fid-50k": fid_score, "precision": precision, "recall": recall}])
        new_row.to_csv(result_csv_path, mode="a", header=False, index=False)

    if rank == 0:
        print(f"FID = {score}")
        print(f"Precision = {precision}")
        print(f"Recall = {recall}")
        new_row = pd.DataFrame([{"path": path, "fid-50k": score, "precision": precision, "recall": recall}])
        new_row.to_csv(result_csv_path, mode="a", header=False, index=False)


# -------------------------
# run with absl
# -------------------------
if __name__ == "__main__":
    app.run(main)