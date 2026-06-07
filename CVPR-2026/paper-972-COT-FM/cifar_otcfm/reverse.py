# Reverse ODE: from images to noise
import os
import torch
import numpy as np
from torchdyn.core import NeuralODE
import torch.distributed as dist
from torch.utils.data import DataLoader
from collections import OrderedDict

from torchcfm.models.unet.unet import UNetModelWrapper
from absl import app, flags
from ppo_utils import load_ppo_resources

FLAGS = flags.FLAGS

# -------------------------
# absl flags definitions
# -------------------------
flags.DEFINE_integer("batch_size", 128)
flags.DEFINE_integer("euler_steps", 100)
flags.DEFINE_string("checkpoint_path", "./results/vanilla_fm_blocks_4/icfm_cifar10_weights_step_400000.pt")
flags.DEFINE_string("data_root", "./data/cifar10_processed")
flags.DEFINE_string("output_path", "./reverse_states_cov_class.pth")
flags.DEFINE_integer("num_classes", 100)


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


# -------------------------
# reverse ODE integration
# -------------------------
def reverse_ode_integration(model, images, euler_steps, device):
    """
    Integrate from t=1 (images) to t=0 (noise) using reverse ODE.
    dx/dt = -f(x, t)
    """
    with torch.no_grad():
        x = images
        x = x.to(device)
        
        t_span = torch.linspace(1, 0, euler_steps + 1, device=device)
        
        for i in range(euler_steps):
            t_curr = t_span[i]
            t_next = t_span[i + 1]
            dt = t_next - t_curr  
            
        
            t_batch = torch.ones(x.shape[0], device=device) * t_curr
            v_t = model(t_batch, x)
            
            x = x + dt * (v_t)
        
        return x


# -------------------------
# compute statistics for one class
# -------------------------
def compute_statistics_for_class(model, class_id, data_root, batch_size, euler_steps, device, rank, ppo_resources):
    """
    Compute mean and covariance for one class.
    Returns: dict with 'mean', 'cov', 'num_samples', 'class_id'
    """

    images = ppo_resources['ppo_data']['images'][ppo_resources['class_to_indices'][class_id]]
    
    print(f"[Rank {rank}] Processing class {class_id} with {len(images)} images...")
    
    dataset = torch.utils.data.TensorDataset(images)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    noise_samples = []
    
    for batch_idx, (batch_images,) in enumerate(dataloader):
        noise = reverse_ode_integration(model, batch_images, euler_steps, device)
        noise_samples.append(noise.cpu())


    all_noise = torch.cat(noise_samples, dim=0)
    

    N = all_noise.shape[0]
    noise_flat = all_noise.reshape(N, -1)
    
    mean = noise_flat.mean(dim=0)  
    cov = torch.cov(noise_flat.T)
    
    print(f"[Rank {rank}] Completed class {class_id}: {N} samples")
    
    return {
        'mean': mean,  
        'cov': cov,   
        'num_samples': N,
        'class_id': class_id
    }


# ------------------------------------------------
# main
# ------------------------------------------------
def main(argv):
    batch_size = FLAGS.batch_size
    euler_steps = FLAGS.euler_steps
    checkpoint_path = FLAGS.checkpoint_path
    data_root = FLAGS.data_root
    output_path = FLAGS.output_path
    num_classes = FLAGS.num_classes

    is_distributed, rank, world_size, local_rank = setup_distributed()
    local_device = f"cuda:{local_rank}" if is_distributed else "cuda:0"

    if rank == 0:
        print(f"Running on {world_size} GPUs")
        print(f"Checkpoint: {checkpoint_path}")
        print(f"Euler steps: {euler_steps}")
        print(f"Total classes: {num_classes}")

    model = UNetModelWrapper(
        dim=(3, 32, 32),
        num_res_blocks=4,
        num_channels=128,
        channel_mult=[1, 2, 2, 2],
        num_heads=4,
        num_head_channels=64,
        attention_resolutions="16",
        dropout=0.1,
    ).to(local_device)

    checkpoint = torch.load(checkpoint_path, map_location=local_device)
    state_dict = checkpoint.get("ema_model", checkpoint.get("model_state_dict", checkpoint))

    try:
        model.load_state_dict(state_dict)
    except RuntimeError:
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            new_k = k[7:] if k.startswith("module.") else k
            new_state_dict[new_k] = v
        model.load_state_dict(new_state_dict)

    model.eval()

    classes_per_gpu = []
    for i in range(world_size):
        start_idx = i * num_classes // world_size
        end_idx = (i + 1) * num_classes // world_size
        classes_per_gpu.append(list(range(start_idx, end_idx)))

    my_classes = classes_per_gpu[rank]
    print(f"[Rank {rank}] Assigned classes: {my_classes[0]} to {my_classes[-1]} ({len(my_classes)} classes)")

    local_stats = []
    ppo_resources = load_ppo_resources(device=local_device)
    for class_id in my_classes:
        stats = compute_statistics_for_class(
            model, class_id, data_root, batch_size, euler_steps, local_device, rank, ppo_resources
        )
        if stats is not None:
            local_stats.append(stats)

    print(f"[Rank {rank}] Finished processing {len(local_stats)} classes")

    if is_distributed:
        num_local_classes = len(local_stats)
        all_num_classes = [None] * world_size
        dist.all_gather_object(all_num_classes, num_local_classes)

        if rank == 0:
            print(f"Classes processed per rank: {all_num_classes}")
            
            all_stats = [None] * world_size
            dist.gather_object(local_stats, all_stats if rank == 0 else None, dst=0)
            
            all_stats_flat = []
            for stats_list in all_stats:
                if stats_list is not None:
                    all_stats_flat.extend(stats_list)
            
            all_stats_flat.sort(key=lambda x: x['class_id'])
            
            final_stats = {
                'means': torch.stack([s['mean'] for s in all_stats_flat]),  # [num_classes, 3072]
                'covs': torch.stack([s['cov'] for s in all_stats_flat]),    # [num_classes, 3072, 3072]
                'num_samples': torch.tensor([s['num_samples'] for s in all_stats_flat]),
                'class_ids': torch.tensor([s['class_id'] for s in all_stats_flat]),
                'euler_steps': euler_steps,
                'checkpoint_path': checkpoint_path
            }

            torch.save(final_stats, output_path)
            print(f"\nSaved all statistics to: {output_path}")
            print(f"Total classes: {len(all_stats_flat)}")
            print(f"Means shape: {final_stats['means'].shape}")
            print(f"Covs shape: {final_stats['covs'].shape}")
        else:
            dist.gather_object(local_stats, None, dst=0)
    else:
        local_stats.sort(key=lambda x: x['class_id'])
        
        final_stats = {
            'means': torch.stack([s['mean'] for s in local_stats]),
            'covs': torch.stack([s['cov'] for s in local_stats]),
            'num_samples': torch.tensor([s['num_samples'] for s in local_stats]),
            'class_ids': torch.tensor([s['class_id'] for s in local_stats]),
            'euler_steps': euler_steps,
            'checkpoint_path': checkpoint_path
        }
        
        torch.save(final_stats, output_path)
        print(f"\nSaved all statistics to: {output_path}")
        print(f"Total classes: {len(local_stats)}")

if __name__ == "__main__":
    app.run(main)