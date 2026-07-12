import torch
from tqdm import tqdm

def train_es(model, train_loader, optimizer, criterion, device, sigma, population_size, lr):
    model.train()
    total_loss = 0
    pbar = tqdm(train_loader, desc="ES Train", leave=False)
    main_params = [p for p in model.parameters() if p.requires_grad]
    for x, y in pbar:
        x, y = x.to(device), y.to(device)
        original_params = [p.data.clone() for p in main_params]
        perturbations = [[torch.randn_like(p) for p in main_params] for _ in range(population_size)]
        rewards = torch.zeros(population_size, device=device)
        
        for i in range(population_size):
            for p, noise in zip(main_params, perturbations[i]): 
                p.data = original_params[i % len(original_params)] + sigma * noise
            with torch.no_grad(): 
                outputs = model(x)
                loss = criterion(outputs, y)
                rewards[i] = -loss
                
        if rewards.std() > 1e-6: 
            rewards = (rewards - rewards.mean()) / rewards.std()
        else: 
            rewards = rewards - rewards.mean()
            
        grad_estimate = [torch.zeros_like(p) for p in main_params]
        for i in range(population_size):
            for grad_p, noise_p in zip(grad_estimate, perturbations[i]): 
                grad_p += rewards[i] * noise_p
                
        for p, orig_p in zip(main_params, original_params): 
            p.data = orig_p
            
        optimizer.zero_grad()
        for p, grad_est in zip(main_params, grad_estimate): 
            p.grad = -grad_est / (population_size * sigma)
        optimizer.step()
        
        with torch.no_grad(): 
            final_loss = criterion(model(x), y).item()
        total_loss += final_loss
        pbar.set_postfix(loss=f"{final_loss:.4f}")
        
    return total_loss / len(train_loader)
