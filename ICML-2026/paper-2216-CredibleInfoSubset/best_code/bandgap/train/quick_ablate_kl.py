'''Quick KL weight ablation: test 4 values at 100 epochs each'''
import os, sys
os.chdir('/repo/bandgap')
sys.path.append('/repo/bandgap')
import numpy as np, pandas as pd, torch, random, json
from torch.utils.data import DataLoader, random_split, ConcatDataset
from dataset.dataset import CompositionDataset
from model.model_mine import BandModelSE, combined_vae_evidential_loss_SE

device = torch.device('cuda')

def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

SEED = 1024; MAX_EPOCHS = 100; LR = 1e-4; BATCH_SIZE = 128
set_seed(SEED)

full_df = pd.read_csv('./data/bandgap.csv')
hse_df = full_df[full_df.state == 0].reset_index(drop=True)
GGA_df = full_df[full_df.state == 1].reset_index(drop=True)
HSE_dataset = CompositionDataset(hse_df, 'material formula', 'Band_gap', 'state')
GGA_dataset = CompositionDataset(GGA_df, 'material formula', 'Band_gap', 'state')

hse_len = len(HSE_dataset); gga_len = len(GGA_dataset)
train_hse = int(0.8 * hse_len); val_hse = int(0.1 * hse_len); test_hse = hse_len - train_hse - val_hse
train_gga = int(0.8 * gga_len); val_gga = int(0.1 * gga_len)

train_HSE, val_HSE, test_HSE = random_split(HSE_dataset, [train_hse, val_hse, test_hse])
train_GGA, val_GGA, _ = random_split(GGA_dataset, [train_gga, val_gga, gga_len-train_gga-val_gga])
train_dataset = ConcatDataset([train_HSE, train_GGA])
val_dataset = ConcatDataset([val_HSE, val_GGA])
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_HSE, batch_size=BATCH_SIZE, shuffle=False)

results = {}
for kl_w in [0.0, 1e-4, 1e-3, 1e-2]:
    set_seed(SEED)
    model = BandModelSE().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    best_val_mae = float('inf')
    best_epoch = 0
    best_state = None
    
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        for batch in train_loader:
            x_comp = batch['x_comp'].to(device)
            x_total = batch['x_total_feats'].to(device)
            x_state = batch['state'].to(device)
            y = batch['y_bandgap'].to(device)
            optimizer.zero_grad()
            loss, _ = combined_vae_evidential_loss_SE(model, x_comp, x_total, x_state, y, kl_weight=kl_w)
            loss.backward()
            optimizer.step()
        
        model.eval()
        val_preds, val_trues = [], []
        with torch.no_grad():
            for batch in val_loader:
                x_comp = batch['x_comp'].to(device)
                x_total = batch['x_total_feats'].to(device)
                x_state = batch['state'].to(device)
                y = batch['y_bandgap'].to(device)
                _, ld = combined_vae_evidential_loss_SE(model, x_comp, x_total, x_state, y, kl_weight=kl_w)
                val_preds.append(ld['pred'].detach().cpu().ravel())
                val_trues.append(y.detach().cpu().ravel())
        val_mae = float(np.mean(np.abs(np.concatenate(val_preds) - np.concatenate(val_trues))))
        
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_epoch = epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
    
    # Final test
    model.load_state_dict(best_state)
    model.eval()
    all_preds, all_trues = [], []
    with torch.no_grad():
        for batch in test_loader:
            x_comp = batch['x_comp'].to(device)
            x_total = batch['x_total_feats'].to(device)
            x_state = batch['state'].to(device)
            y = batch['y_bandgap'].to(device)
            _, ld = combined_vae_evidential_loss_SE(model, x_comp, x_total, x_state, y, kl_weight=kl_w)
            all_preds.append(ld['pred'].detach().cpu().ravel())
            all_trues.append(y.detach().cpu().ravel())
    all_preds = np.concatenate(all_preds); all_trues = np.concatenate(all_trues)
    mae = float(np.mean(np.abs(all_preds - all_trues)))
    rmse = float(np.sqrt(np.mean((all_preds - all_trues)**2)))
    from scipy.stats import kendalltau
    tau, _ = kendalltau(all_trues, all_preds); tau = float(tau)
    
    results[kl_w] = {'val_mae': best_val_mae, 'best_epoch': best_epoch, 'test_mae': mae, 'test_rmse': rmse, 'test_tau': tau}
    print(f'kl_weight={kl_w:.0e}: val_mae={best_val_mae:.4f} @ ep{best_epoch}, test_mae={mae:.4f}, test_rmse={rmse:.4f}, tau={tau:.4f}')

print('\n=== ABLATION SUMMARY ===')
for kl_w in sorted(results.keys()):
    r = results[kl_w]
    print(f'kl_weight={kl_w:.0e}: test_mae={r["test_mae"]:.4f}, test_rmse={r["test_rmse"]:.4f}, tau={r["test_tau"]:.4f}, best_ep={r["best_epoch"]}')
with open('./kl_ablation.json', 'w') as f:
    json.dump({str(k): v for k, v in results.items()}, f, indent=2)
