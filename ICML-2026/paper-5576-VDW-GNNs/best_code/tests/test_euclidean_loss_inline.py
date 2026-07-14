"""
Simple inline test for Euclidean distance loss - can be copy-pasted into a notebook.
"""

# Cell 1: Imports and setup
import sys
sys.path.insert(0, '/home/davejohnson/Research/vdw/code')

import torch
import torch.nn.functional as F
from models.base_module import euclidean_distance_loss, MultiTaskLoss

print("✓ Imports successful")

# Cell 2: Test basic functionality
print("\n" + "="*60)
print("TEST 1: Basic Euclidean Distance Loss")
print("="*60)

preds = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
targets = torch.tensor([[1.5, 2.5], [3.0, 4.0]])

euclidean_loss = euclidean_distance_loss(preds, targets, reduction='mean')
mse_loss = F.mse_loss(preds, targets)

print(f"Predictions:     {preds.tolist()}")
print(f"Targets:         {targets.tolist()}")
print(f"\nEuclidean loss:  {euclidean_loss.item():.6f}")
print(f"MSE loss:        {mse_loss.item():.6f}")
print(f"\nRatio (MSE/Euclidean): {(mse_loss / euclidean_loss).item():.3f}x")

# Cell 3: Demonstrate the difference with coordinate-wise errors
print("\n" + "="*60)
print("TEST 2: Why Euclidean is Better for 2D Vectors")
print("="*60)

# Case A: Error only in x-direction
preds_a = torch.tensor([[1.0, 0.0]])
targets_a = torch.tensor([[0.0, 0.0]])
eucl_a = euclidean_distance_loss(preds_a, targets_a)
mse_a = F.mse_loss(preds_a, targets_a)

# Case B: Error only in y-direction  
preds_b = torch.tensor([[0.0, 1.0]])
targets_b = torch.tensor([[0.0, 0.0]])
eucl_b = euclidean_distance_loss(preds_b, targets_b)
mse_b = F.mse_loss(preds_b, targets_b)

# Case C: Equal error in both directions
preds_c = torch.tensor([[0.707, 0.707]])
targets_c = torch.tensor([[0.0, 0.0]])
eucl_c = euclidean_distance_loss(preds_c, targets_c)
mse_c = F.mse_loss(preds_c, targets_c)

print("Case A - Error only in X direction:")
print(f"  Euclidean: {eucl_a.item():.4f}, MSE: {mse_a.item():.4f}")
print("Case B - Error only in Y direction:")
print(f"  Euclidean: {eucl_b.item():.4f}, MSE: {mse_b.item():.4f}")
print("Case C - Equal error in both (same distance):")
print(f"  Euclidean: {eucl_c.item():.4f}, MSE: {mse_c.item():.4f}")
print("\nKey insight: Euclidean treats (1,0) and (0,1) the same (both distance=1)")
print("             MSE treats them the same too, but Euclidean treats vectors")
print("             as geometric entities, not independent coordinates")

# Cell 4: Test MultiTaskLoss with Euclidean
print("\n" + "="*60)
print("TEST 3: MultiTaskLoss with Euclidean Distance")
print("="*60)

loss_fn = MultiTaskLoss(task_names=['pos', 'vel'])

preds_dict = {
    'pos': torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
    'vel': torch.tensor([[0.5, 0.5], [1.0, 1.0]]),
}
targets_dict = {
    'pos': torch.tensor([[1.1, 2.1], [3.1, 4.1]]),
    'vel': torch.tensor([[0.6, 0.6], [1.1, 1.1]]),
}

total_loss, per_task = loss_fn(preds_dict, targets_dict)

print(f"Total weighted loss: {total_loss.item():.6f}")
print(f"\nPer-task base losses (before uncertainty weighting):")
for task_name, task_loss in per_task.items():
    print(f"  {task_name}: {task_loss:.6f}")

# Cell 5: Test gradient flow
print("\n" + "="*60)
print("TEST 4: Gradient Flow")
print("="*60)

preds = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
targets = torch.tensor([[1.5, 2.5], [3.0, 4.0]])

loss = euclidean_distance_loss(preds, targets, reduction='mean')
loss.backward()

print(f"Loss value: {loss.item():.6f}")
print(f"Gradients shape: {preds.grad.shape}")
print(f"Gradients:\n{preds.grad}")
print("\n✓ Gradients computed successfully")

print("\n" + "="*60)
print("ALL TESTS PASSED!")
print("="*60)

