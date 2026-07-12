import torch
import torch.nn.functional as F
from tqdm import tqdm

def train_dgl(model, train_loader, optimizers, criterion, num_classes, device):
    model.train()
    total_loss = 0
    pbar = tqdm(train_loader, desc="DGL Train", leave=False)
    
    for x, y in pbar:
        x, y = x.to(device), y.to(device)
        batch_loss = 0
        
        opt_0 = optimizers[0]
        opt_0.zero_grad()
        
        shared_features = model.feature_extractor(x)
        p_in = torch.full((x.size(0), num_classes), 1.0/num_classes, device=device)
        logits_0 = model.processing_modules[0](p_in, shared_features)
        
        loss_0 = criterion(logits_0, y)
        loss_0.backward()
        opt_0.step()
        
        batch_loss += loss_0.item()
        
        p_prev = F.softmax(logits_0.detach(), dim=1)
        features_detached = shared_features.detach()

        for i in range(1, model.num_modules):
            module = model.processing_modules[i]
            opt_i = optimizers[i]
            opt_i.zero_grad()
            
            logits_i = module(p_prev, features_detached)
            
            loss_i = criterion(logits_i, y)
            loss_i.backward()
            opt_i.step()
            
            batch_loss += loss_i.item()
            
            p_prev = F.softmax(logits_i.detach(), dim=1)
            
        total_loss += batch_loss / model.num_modules
        pbar.set_postfix(loss=f"{batch_loss:.4f}")

    return total_loss / len(train_loader)
