import sys, os, time
sys.path.insert(0, '/repo')

from diner_ipod_train import FastMRIDataset, DinerReptileTrainer
import torch
import numpy as np

print("Testing DINER-IPOD pipeline...")
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

# Test data loading
dataset = FastMRIDataset('/datasets/fastmri_processed', preload=False)
print("Dataset size:", len(dataset))

# Test loading one sample
task_data = dataset.get_samples(0, num_samples=1)
print("Task:", task_data['task_id'])
print("Num samples:", len(task_data['samples']))
sample = task_data['samples'][0]
print("Sample keys:", sorted(sample.keys()))
print("  gt_img shape:", sample['gt_img'].shape)
print("  mask_transposed shape:", sample['mask_transposed'].shape)
print("  csmp_transposed shape:", sample['csmp_transposed'].shape)
print("  coordinates shape:", sample['coordinates'].shape)

# Quick model test with short inner loop
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
trainer = DinerReptileTrainer(
    inner_lr=2e-2,
    meta_lr=5e-4,
    inner_steps=5,
    device=device,
)

# Test inner loop on one sample
samples = task_data['samples']
print("\nTesting inner loop adaptation (5 steps)...")
t0 = time.time()
adapted_model, loss = trainer.inner_loop_adaptation(samples)
t1 = time.time()
print("Final loss:", loss)
print("Time per step: %.3f s" % ((t1 - t0) / 5))
print("Estimated time per epoch (15 tasks x 300 steps): %.1f min" % (15 * 300 * (t1 - t0) / 5 / 60))
print("Estimated total training time (2500 epochs): %.1f hours" % (2500 * 15 * 300 * (t1 - t0) / 5 / 3600))

print("\nPipeline test PASSED!")
