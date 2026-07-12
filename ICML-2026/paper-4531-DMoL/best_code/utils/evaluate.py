import torch

def evaluate(model, test_loader, device, method):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            if method == 'ff': 
                outputs = model(pos_x=x)
            else: 
                outputs = model(x)
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
    return 100 * correct / total

def fgsm_attack(model, criterion, images, labels, epsilon):
    images.requires_grad = True
    outputs = model(images)
    loss = criterion(outputs, labels)
    model.zero_grad()
    loss.backward()
    attack_images = images + epsilon * images.grad.sign()
    attack_images = torch.clamp(attack_images, -1, 1) # Normalize friendly
    return attack_images

def evaluate_robustness(model, test_loader, device, method, criterion, fgsm_epsilon):
    model.eval()
    results = {}
    print("\nRunning Robustness Benchmark...")
    results['clean_acc'] = evaluate(model, test_loader, device, method)
    print(f"  - Clean Accuracy: {results['clean_acc']:.2f}%")
    
    # Gaussian Noise
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            noisy_images = torch.clamp(x + torch.randn_like(x) * 0.1, -1, 1)
            outputs = model(noisy_images)
            _, p = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (p == y).sum().item()
    results['gaussian_noise_acc'] = 100 * correct / total
    print(f"  - Gaussian Noise Acc: {results['gaussian_noise_acc']:.2f}%")
    
    # FGSM Attack
    correct, total = 0, 0
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        attack_images = fgsm_attack(model, criterion, x, y, fgsm_epsilon)
        outputs = model(attack_images)
        _, p = torch.max(outputs.data, 1)
        total += y.size(0)
        correct += (p == y).sum().item()
    results['fgsm_acc'] = 100 * correct / total
    print(f"  - FGSM Attack Acc (eps={fgsm_epsilon}): {results['fgsm_acc']:.2f}%")
    
    return results
