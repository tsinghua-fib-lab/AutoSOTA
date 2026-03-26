"""
Standalone evaluation script for MeanFlow CIFAR-10 FID computation.
"""
import sys
sys.path.insert(0, '/py-meanflow/meanflow')

import os
import logging
import math
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchmetrics.image.fid import FrechetInceptionDistance

from train_arg_parser import get_args_parser
from models.model_configs import instantiate_model

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(filename)s - %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    # Parse arguments with defaults
    args = get_args_parser().parse_args([])

    # Configuration
    args.use_edm_aug = True  # The checkpoint was trained with EDM aug
    args.distributed = False
    args.batch_size = 128
    args.compute_fid = True
    args.seed = 0
    args.fid_samples = 50000
    args.data_path = '/tmp/cifar10_data/cifar10_data'  # Pre-extracted dataset

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")

    # Load checkpoint
    chkpt_path = '/checkpoints/cifar10_meanflow.pth'
    logger.info(f"Loading checkpoint from {chkpt_path}")
    checkpoint = torch.load(chkpt_path, map_location='cpu', weights_only=False)
    logger.info(f"Checkpoint epoch: {checkpoint['epoch']}")

    # Instantiate model
    model = instantiate_model(args)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()
    logger.info("Model loaded.")

    # Check which EMA networks are available
    logger.info(f"EMA net ema_decay: {model.net_ema.ema_decay}")
    for i, decay in enumerate(args.ema_decays):
        net = model._modules[f'net_ema{i+1}']
        logger.info(f"net_ema{i+1} ema_decay: {net.ema_decay}")

    # Use net_ema1 as in demo notebook (ema_decay=0.99995)
    net_eval = model.net_ema1

    # Load CIFAR-10 (already downloaded)
    logger.info("Loading CIFAR-10 dataset...")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    dataset = datasets.CIFAR10(
        root=args.data_path,
        train=True,
        download=False,  # Already downloaded
        transform=transform,
    )
    logger.info(f"Dataset size: {len(dataset)}")

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # Compute FID
    fid_metric = FrechetInceptionDistance(normalize=True).to(device)

    num_synthetic = 0
    fid_samples = args.fid_samples

    logger.info(f"Computing FID with {fid_samples} samples...")

    for data_iter_step, (samples, _) in enumerate(data_loader):
        if data_iter_step % 10 == 0:
            logger.info(f"Step {data_iter_step}/{len(data_loader)}, synthetic: {num_synthetic}/{fid_samples}")

        samples = samples.to(device, non_blocking=True)

        # Real samples: scale from [-1,1] to [0,1]
        real_samples_01 = torch.clamp(samples * 0.5 + 0.5, min=0.0, max=1.0)
        fid_metric.update(real_samples_01, real=True)

        if num_synthetic < fid_samples:
            with torch.no_grad():
                synthetic_samples = model.sample(
                    samples_shape=samples.shape,
                    net=net_eval,
                    device=device
                )
            torch.cuda.synchronize()

            # Scale from [-1,1] to [0,1]
            synthetic_samples = torch.clamp(
                synthetic_samples * 0.5 + 0.5, min=0.0, max=1.0
            )
            synthetic_samples = torch.floor(synthetic_samples * 255)
            synthetic_samples = torch.clamp(synthetic_samples / 255.0, 0.0, 1.0)

            fid_metric.update(synthetic_samples.float(), real=False)
            num_synthetic += samples.shape[0]

    fid_score = fid_metric.compute().item()
    logger.info(f"=== FID score (net_ema1): {fid_score:.4f} ===")
    print(f"\n=== FID RESULT ===")
    print(f"FID (1-NFE, CIFAR-10, net_ema1): {fid_score:.4f}")

    return fid_score


if __name__ == '__main__':
    main()
