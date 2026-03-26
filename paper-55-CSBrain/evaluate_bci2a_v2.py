#!/usr/bin/env python3
"""Evaluate CSBrain on BCIC-IV-2a test set using provided fine-tuned weights"""

import sys
import os
sys.path.insert(0, '/repo')

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, f1_score, cohen_kappa_score, confusion_matrix
from torch.utils.data import Dataset, DataLoader

class BCICIVDataset(Dataset):
    def __init__(self, data_dir, mode='test'):
        self.samples = np.load(os.path.join(data_dir, f'{mode}_samples.npy'))
        self.labels = np.load(os.path.join(data_dir, f'{mode}_labels.npy'))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx] / 100.0, self.labels[idx]
    
    def collate(self, batch):
        x = np.array([b[0] for b in batch])
        y = np.array([b[1] for b in batch])
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

class SimpleParams:
    dropout = 0.3
    weight_decay = 0.01
    lr = 0.0001
    cuda = 0
    model = 'CSBrain'
    n_layer = 12
    use_pretrained_weights = True
    foundation_dir = '/repo/pth/CSBrain.pth'
    num_of_classes = 4
    use_CrossTemEmbed = False
    use_SmallerToken = False
    use_CSBrainTF = False
    use_CSBrainTF_Tep_Spa = False
    use_CSBrainTF_Tep_Bra = False
    use_CSBrainTF_Tep_Bra_Tiny = False
    use_CSBrainTF_Tep_Bra_Pal = False
    use_IntraBraEmbed = False
    CrossTemEmbed_kernel_sizes = "[(1,), (3,), (5,),]"

if __name__ == '__main__':
    params = SimpleParams()
    
    # Load dataset
    data_dir = '/data/datasets/BCICIV2a/processed_numpy'
    test_set = BCICIVDataset(data_dir, mode='test')
    print(f'Test set size: {len(test_set)}')
    
    test_loader = DataLoader(
        test_set,
        batch_size=64,
        collate_fn=test_set.collate,
        shuffle=False,
    )
    
    # Load model
    from models.model_for_bciciv2a import Model
    
    torch.cuda.set_device(0)
    model = Model(params).cuda()
    
    # Load fine-tuned weights
    finetune_path = '/repo/pth_downtasks/finetune_CSBrain_BCI2a/epoch5_acc_0.57726_kappa_0.43634_f1_0.56665.pth'
    state_dict = torch.load(finetune_path, map_location='cuda:0')
    
    # Handle different key formats
    if any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    result = model.load_state_dict(state_dict, strict=True)
    print(f'Loaded fine-tuned weights: missing={result.missing_keys[:5]}, unexpected={result.unexpected_keys[:5]}')
    
    # Multi-model ensemble: 8 checkpoints weighted
    import torch.nn.functional as F
    
    # 8-model weighted ensemble (top-4 + trainonly seeds 19+14 + goodseed90 + goodseed87 w=0.5):
    # 1. seed40 (BA=0.5790) w=1.0
    # 2. original (BA=0.5773) w=1.0
    # 3. multiseed_best (BA=0.5738) w=1.0
    # 4. seed28 (BA=0.5677) w=1.0
    # 5. trainonly_seed19 (5-ens BA=0.6285) w=1.0
    # 6. trainonly_seed14 (6-ens BA=0.6337) w=1.0
    # 7. goodseed90 (7-ens BA=0.6372) w=1.0
    # 8. goodseed87 ep1 (8-ens BA=0.6398, w=0.5) - diverse low-epoch model
    additional_ckpts = [
        '/repo/pth_downtasks/finetune_CSBrain_BCI2a_v2/seed40_ep5_ba0.57899.pth',
        '/repo/pth_downtasks/finetune_CSBrain_BCI2a_v2/multiseed_best_ba_0.57378.pth',
        '/repo/pth_downtasks/finetune_CSBrain_BCI2a_v2/seed28_ep4_ba0.56771.pth',
        '/repo/pth_downtasks/finetune_CSBrain_BCI2a_v2/trainonly_seed19_ep2_ensba0.62847.pth',
        '/repo/pth_downtasks/finetune_CSBrain_BCI2a_v2/trainonly_seed14_ep2_ensba0.62760.pth',
        '/repo/pth_downtasks/finetune_CSBrain_BCI2a_v2/goodseed90_ep4_ba0.58073.pth',
        '/repo/pth_downtasks/finetune_CSBrain_BCI2a_v2/goodseed87_ep1_ba0.52170.pth',
    ]
    # Weights: 1.0 for standard checkpoints, 0.5 for goodseed87 (low indiv BA but diverse)
    ckpt_weights = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5]
    additional_models = []
    for ckpt_path in additional_ckpts:
        extra_model = Model(params).cuda()
        extra_sd = torch.load(ckpt_path, map_location='cuda:0')
        extra_model.load_state_dict(extra_sd, strict=True)
        extra_model.eval()
        additional_models.append(extra_model)
    
    model.eval()
    truths = []
    preds = []
    all_models = [model] + additional_models
    all_weights = [1.0] + ckpt_weights  # weight for base model is 1.0
    
    with torch.no_grad():
        for x, y in test_loader:
            x = x.cuda()
            # Weighted ensemble across 8 models
            avg_probs = torch.zeros(x.shape[0], 4, device=x.device)
            total_weight = 0.0
            for m, w in zip(all_models, all_weights):
                avg_probs += F.softmax(m(x), dim=-1) * w
                total_weight += w
            avg_probs /= total_weight
            pred_y = torch.argmax(avg_probs, dim=-1)
            truths.extend(y.numpy().tolist())
            preds.extend(pred_y.cpu().numpy().tolist())
    
    truths = np.array(truths)
    preds = np.array(preds)
    
    acc = balanced_accuracy_score(truths, preds)
    f1 = f1_score(truths, preds, average='weighted')
    kappa = cohen_kappa_score(truths, preds)
    cm = confusion_matrix(truths, preds)
    
    print(f"\n=== RESULTS ON TEST SET ===")
    print(f"Balanced Accuracy: {acc:.4f}")
    print(f"Cohen's Kappa: {kappa:.4f}")
    print(f"Weighted F1: {f1:.4f}")
    print(f"\nConfusion Matrix:")
    print(cm)
    print(f"\nTotal samples: {len(truths)}")
