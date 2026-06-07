import argparse
import datetime
import numpy as np
import os
import time
import pickle
import io
from tqdm import tqdm
import torch.distributed as dist

import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from PIL import Image
import lmdb

from diffusers.models import AutoencoderKL
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution

def center_crop_arr(pil_image, image_size):

    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


class LMDBImageNetReader(torch.utils.data.Dataset):
    def __init__(self, lmdb_path, transform=None):
        self.env = lmdb.open(lmdb_path, 
                            readonly=True,
                            lock=False,
                            readahead=False,
                            meminit=False)
        
        with self.env.begin() as txn:
            self.length = int(txn.get('num_samples'.encode()).decode())
        
        self.transform = transform
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        with self.env.begin() as txn:
            data = txn.get(f'{index}'.encode())
            if data is None:
                raise IndexError(f'Index {index} is out of bounds')
            
            data = pickle.loads(data)
            img_bin = data['image']
            label = data['label']
            
            buffer = io.BytesIO(img_bin)
            img = Image.open(buffer).convert('RGB')
            
            if self.transform:
                img = self.transform(img)
            
            class_name = f'n{label:08d}'
            filename = f'{class_name}/img_{index:07d}.JPEG'
                
            return img, label, filename, index 
    
    def __del__(self):
        if hasattr(self, 'env') and self.env is not None:
            try:
                self.env.close()
            except Exception as e:
                print(f"Error closing LMDB environment: {e}")

def process_batch(args, vae, device, images, labels, filenames, original_indices, env):
    images = images.to(device)
    
    with torch.no_grad():
        posterior = DiagonalGaussianDistribution(vae._encode(images))
        moments = posterior.parameters
        posterior_flip = DiagonalGaussianDistribution(vae._encode(images.flip(dims=[3])))
        moments_flip = posterior_flip.parameters
    
    try:
        with env.begin(write=True) as txn:
            for i, (label, filename, orig_idx) in enumerate(zip(labels, filenames, original_indices)):
                data = {
                    'moments': moments[i].cpu().numpy(),
                    'moments_flip': moments_flip[i].cpu().numpy(),
                    'label': label.item(),
                    'filename': filename
                }
                
                txn.put(f'{orig_idx}'.encode(), pickle.dumps(data))
    except lmdb.Error as e:
        print(f"LMDB error during batch processing: {e}")
        return 0
        
    return len(images)

def preprocess_latents(args):
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    cudnn.benchmark = True
    
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-ema").to(device)
    vae.eval()
    
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    if rank == 0:
        print(f"Loading source dataset from {args.source_lmdb}")
    dataset = LMDBImageNetReader(args.source_lmdb, transform=transform)
    
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, 
        num_replicas=world_size, 
        rank=rank,
        shuffle=False,
        drop_last=False
    )
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    os.makedirs(os.path.dirname(args.target_lmdb) if os.path.dirname(args.target_lmdb) else '.', exist_ok=True)
    
    map_size = 1024 * 1024 * 1024 * args.lmdb_size_gb
    if rank == 0:
        print(f"Creating target LMDB at {args.target_lmdb} with size {args.lmdb_size_gb}GB")
    
    env = None
    try:
        env = lmdb.open(args.target_lmdb, map_size=map_size, max_readers=world_size*2, max_spare_txns=world_size*2)
        
        total_processed = 0
        start_time = time.time()
        
        pbar = None
        if rank == 0:
            pbar = tqdm(total=len(dataloader), desc=f"Processing")
        
        for batch_idx, (images, labels, filenames, original_indices) in enumerate(dataloader):
            num_processed = process_batch(
                args, vae, device, images, labels, filenames, original_indices, env
            )
            
            total_processed += num_processed
            if pbar is not None:
                pbar.update(1)
        
        if pbar is not None:
            pbar.close()
        
        dist.barrier()
        
        if rank == 0:
            try:
                with env.begin(write=True) as txn:
                    txn.put('num_samples'.encode(), str(len(dataset)).encode())
                    txn.put('created_at'.encode(), str(datetime.datetime.now()).encode())
            except lmdb.Error as e:
                print(f"Error writing metadata: {e}")
        
        dist.barrier()
        
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        if rank == 0:
            print(f'Preprocessing completed in {total_time_str}')
            print(f'Each process processed approximately {total_processed} samples')
            print(f'Target LMDB saved at: {args.target_lmdb}')
    
    except Exception as e:
        print(f"Process {rank} encountered error: {e}")
        if env is not None:
            try:
                env.close()
            except:
                pass
        raise
    
    finally:
        if env is not None:
            try:
                env.sync()
                env.close()
                if rank == 0:
                    print("LMDB environment closed successfully")
            except Exception as e:
                print(f"Process {rank} error closing LMDB: {e}")

def parse_args():
    parser = argparse.ArgumentParser(description='Preprocess ImageNet to VAE latents')
    parser.add_argument('--source_lmdb', type=str, required=True,
                        help='Path to source ImageNet LMDB')
    parser.add_argument('--target_lmdb', type=str, required=True,
                        help='Path to save target latents LMDB')
    parser.add_argument('--img_size', type=int, default=256,
                        help='Image size for preprocessing')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for processing')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='Number of workers for data loading')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--lmdb_size_gb', type=int, default=300,
                        help='LMDB size in GB')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    
    if 'RANK' not in os.environ:
        raise RuntimeError("Please use torchrun to run this script. Example: torchrun --nproc_per_node=8 main_cache.py ...")
    
    preprocess_latents(args)
