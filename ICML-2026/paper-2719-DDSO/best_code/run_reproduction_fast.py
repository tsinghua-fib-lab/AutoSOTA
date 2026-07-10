import os, sys
import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
from torch.optim import Adam
from utils import evaluate
from model_all import ICALD_Classifier
from datasets import ClassificationDataModule, classification_shapes
from typing import Dict, Tuple, List

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

print('='*70)
print('HICALD Reproduction - ICALD on Heart-Disease')
print(f'PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}')
print(f'GPU: {torch.cuda.get_device_name(0)}')
print('='*70)

dataset_str = 'heart-disease'
model_str = 'ICALD_Classifier'
epochs = 600
batch_size = 128
lr = 1e-3
num_runs = 5
lambda_reg = 0.9  # corresponds to beta=0.1 (1-lambda_reg = beta)
t_val = 0.25       # corresponds to t0=0.5
patience = 50
min_epochs = 20
# Fast per-epoch evaluation, full MC at final
mc_fast = 100
mc_full = 2000

def suggest_n_hidden(input_dim):
    if input_dim <= 16:
        return int(2 * input_dim)
    elif input_dim <= 64:
        return int(1.6 * input_dim)
    else:
        return min(2.5 * input_dim, 256)

all_metrics = []
all_run_train_hist = []
all_run_test_hist = []

for run in range(num_runs):
    print(f'=== {model_str} | {dataset_str} | Epochs {epochs} | Run {run + 1}/{num_runs} ===')
    sys.stdout.flush()
    
    data_module = ClassificationDataModule(
        dataset_str, data_dir='datasets', normalize=True,
        batch_size=batch_size, test_batch_size=batch_size,
        random_seed=42 + run,
    )
    data_module.setup()
    input_dim, num_classes = classification_shapes[dataset_str]
    n_hidden = suggest_n_hidden(input_dim)
    
    model = ICALD_Classifier(input_dim=input_dim, n_hidden=n_hidden, output_dim=num_classes)
    model = model.to(torch.device('cuda'))
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    
    train_loss_hist = []
    test_loss_hist = []
    best_score = -1e9
    best_epoch = -1
    best_state_dict = None
    epochs_no_improve = 0
    
    for epoch in range(epochs):
        # ---- TRAIN ----
        model.train()
        epoch_train_losses = []
        for x, y in data_module.train_dataloader():
            optimizer.zero_grad()
            x = x.cuda(); y = y.cuda()
            
            # ICALD loss
            q_clean = torch.ones(x.shape[0], 1, device=x.device)
            probs_clean = model(x, q=q_clean, return_probs=True, t=t_val)
            loss_nll = F.nll_loss((probs_clean + 1e-8).log(), y.view(-1))
            
            K = 5
            x_aug = x.repeat_interleave(K, dim=0)
            q_aug = torch.rand(x_aug.shape[0], 1, device=x.device)
            with torch.no_grad():
                feat_aug = model.extract_features(x_aug)
            feat_aug = feat_aug.detach()
            probs_aug = model.predict_from_features(feat_aug, q_aug, return_probs=True, t=t_val)
            conf, _ = probs_aug.max(1)
            loss_cal = (conf - q_aug.view(-1)).abs().mean()
            
            loss = lambda_reg * loss_nll + (1 - lambda_reg) * loss_cal
            loss.backward()
            optimizer.step()
            epoch_train_losses.append(loss.item())
        
        mean_train_loss = float(np.mean(epoch_train_losses))
        
        # ---- EVALUATE (fast) ----
        model.eval()
        with torch.no_grad():
            x_test_all = []; y_test_all = []
            for xt, yt in data_module.test_dataloader():
                x_test_all.append(xt.cuda()); y_test_all.append(yt.cuda())
            x_test_cat = torch.cat(x_test_all, dim=0)
            y_test_cat = torch.cat(y_test_all, dim=0)
            
            # Fast evaluation for early stopping (fewer MC samples)
            val_metrics = evaluate(model_str, model, x_test_cat, y_test_cat.view(-1), t=t_val, mc_samples=mc_fast)
        
        # Test loss for logging
        model.eval()
        epoch_test_losses = []
        for xt, yt in data_module.test_dataloader():
            xt = xt.cuda(); yt = yt.cuda()
            q_clean = torch.ones(xt.shape[0], 1, device=xt.device)
            probs_clean = model(xt, q=q_clean, return_probs=True, t=t_val)
            test_loss = F.nll_loss((probs_clean + 1e-8).log(), yt.view(-1))
            epoch_test_losses.append(test_loss.item())
        mean_test_loss = float(np.mean(epoch_test_losses))
        
        train_loss_hist.append(mean_train_loss)
        test_loss_hist.append(mean_test_loss)
        
        acc = val_metrics.get('Accuracy', float('nan'))
        ece = val_metrics.get('ECE', float('nan'))
        kce = val_metrics.get('KCE', float('nan'))
        
        if not np.isnan(acc) and not np.isnan(ece):
            score = acc - ece
        elif not np.isnan(acc):
            score = acc
        else:
            score = -1e9
        
        improved = score > best_score + 1e-6
        if improved:
            best_score = score
            best_epoch = epoch
            best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f'Run {run+1} | Ep {epoch+1}/{epochs} | TrLoss: {mean_train_loss:.4f} | TeLoss: {mean_test_loss:.4f} | Acc: {acc:.4f} | ECE: {ece:.4f} | KCE: {kce:.4f}')
            sys.stdout.flush()
        
        if epoch + 1 >= min_epochs and epochs_no_improve >= patience:
            print(f'[EarlyStop] Stop at epoch {epoch+1}, best epoch = {best_epoch+1}, best score = {best_score:.4f}')
            break
    
    all_run_train_hist.append(train_loss_hist)
    all_run_test_hist.append(test_loss_hist)
    
    # Load best model
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
    
    # ---- FINAL EVALUATION (full MC) ----
    model.eval()
    with torch.no_grad():
        x_test_all = []; y_test_all = []
        for xt, yt in data_module.test_dataloader():
            x_test_all.append(xt.cuda()); y_test_all.append(yt.cuda())
        x_test = torch.cat(x_test_all, dim=0)
        y_test = torch.cat(y_test_all, dim=0)
        test_metrics = evaluate(model_str, model, x_test, y_test.view(-1), t=t_val, mc_samples=mc_full)
    
    all_metrics.append(test_metrics)
    print(f'Test Metrics (final, best epoch {best_epoch+1}): {test_metrics}')
    sys.stdout.flush()

# Aggregate
print()
print('='*70)
print('REPRODUCTION RESULTS')
print('='*70)
metric_keys = sorted(set().union(*[m.keys() for m in all_metrics]))
results = {}
for k in metric_keys:
    vals = [m.get(k, float('nan')) for m in all_metrics]
    vals = np.array(vals, dtype=float)
    mean, std = np.nanmean(vals), np.nanstd(vals)
    results[k] = (mean, std)
    if k in ['Accuracy', 'AP', 'AUC']:
        print(f'{k:15s}: {mean*100:.3f} +/- {std*100:.3f}')
    else:
        print(f'{k:15s}: {mean:.6f} +/- {std:.6f}')

# Save
with open('reproduction_summary.txt', 'w') as f:
    for k in sorted(results.keys()):
        mean, std = results[k]
        if k in ['Accuracy', 'AP', 'AUC']:
            f.write(f'{k}: {mean*100:.3f} +/- {std*100:.3f}\n')
        else:
            f.write(f'{k}: {mean:.6f} +/- {std:.6f}\n')

print()
print('Done! Results saved to reproduction_summary.txt')
