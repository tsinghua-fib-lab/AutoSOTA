import torch
import torch.optim as optim
from tqdm import tqdm
from models.hsic import hsic

def train_hsic(model, train_loader, criterion, device, args):
    model.train()
    optimizer = optim.Adam(
        list(model.feature_extractor.parameters()) + list(model.decoders[0].parameters()),
        lr=args.lr
    )
    pbar_base = tqdm(train_loader, desc="  HSIC Base Train", leave=False)
    for x, y in pbar_base:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        
        h_base = model.feature_extractor(x)
        logits = model.decoders[0](h_base)
        loss_ce = criterion(logits, y)

        x_flat = x.view(x.size(0), -1)
        loss_hsic_val = hsic(h_base, x_flat, kernel_type=args.hsic_kernel)
        
        total_loss = loss_ce + args.hsic_lambda * loss_hsic_val
        total_loss.backward()
        optimizer.step()
        pbar_base.set_postfix(loss=f"{total_loss.item():.4f}")

    h_in = None
    final_layer_loss = 0
    
    for l in range(model.num_modules):
        optimizer = optim.Adam(
            list(model.hsic_layers[l].parameters()) + list(model.decoders[l].parameters()),
            lr=args.hsic_decoder_lr
        )
        
        layer_loss_accumulator = 0
        pbar_layer = tqdm(train_loader, desc=f"  HSIC Layer {l+1} Train", leave=False)
        for x, y in pbar_layer:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            with torch.no_grad():
                reps = model.forward_all_reps(x)
                h_in = reps[l].detach()

            h_out = model.hsic_layers[l](h_in)
            
            logits = model.decoders[l](h_out)
            loss_ce = criterion(logits, y)

            x_flat = x.view(x.size(0), -1)
            loss_hsic_val = hsic(h_out, x_flat, kernel_type=args.hsic_kernel)

            total_loss = loss_ce + args.hsic_lambda * loss_hsic_val
            total_loss.backward()
            optimizer.step()
            
            layer_loss_accumulator += total_loss.item()
            pbar_layer.set_postfix(loss=f"{total_loss.item():.4f}")
        
        if l == model.num_modules - 1:
            final_layer_loss = layer_loss_accumulator / len(train_loader)
            
    return final_layer_loss
