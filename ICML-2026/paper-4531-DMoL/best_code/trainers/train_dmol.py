import torch
import torch.nn.functional as F
from tqdm import tqdm

def train_dmol(model, train_loader, optimizers, num_classes, device, alpha, label_smoothing=0.0):
    model.train()
    total_loss = 0
    pbar = tqdm(train_loader, desc="DMoL Train", leave=False)
    opt_cnn, *opt_mlps = optimizers
    for x, y in pbar:
        x, y = x.to(device), y.to(device)
        if label_smoothing > 0:
            y_one_hot = torch.full((x.size(0), num_classes), label_smoothing / num_classes, device=device)
            y_one_hot.scatter_(1, y.unsqueeze(1), 1.0 - label_smoothing + label_smoothing / num_classes)
        else:
            y_one_hot = F.one_hot(y, num_classes=num_classes).float()
        for opt in optimizers:
            opt.zero_grad()
        shared_features = model.feature_extractor(x)
        
        with torch.no_grad():
            p_current = torch.full((x.size(0), num_classes), 1.0/num_classes, device=device)
            p_inputs = [p_current]
            for module in model.processing_modules:
                p_logits = module(p_inputs[-1], shared_features.detach())
                p_inputs.append(F.softmax(p_logits, dim=1))
        
        loss = 0
        for i, module in enumerate(model.processing_modules):
            log_p = F.log_softmax(module(p_inputs[i].detach(), shared_features), dim=1)
            loss += (alpha * F.kl_div(log_p, y_one_hot, reduction='batchmean') + (1 - alpha) * F.kl_div(log_p, p_inputs[i+1].detach(), reduction='batchmean'))
            
        loss.backward()
        for opt in optimizers:
            opt.step()
            
        total_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    return total_loss / len(train_loader)
