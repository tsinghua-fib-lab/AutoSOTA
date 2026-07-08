import torch
import json
from itertools import islice
from dataloaders.get_loaders import get_loaders
from attacks.fgsm import fgsm
from attacks.pgd import pgd
from utils import load_checkpoint
from training.utils import MetricTracker, get_input_dimensions


def test_step(model, attack, loader, upper_limit, lower_limit, mu, std, epsilon, attack_norm, device):
    num_batches_to_evaluate = len(loader) // 10
    model.eval()
    total, correct = 0, 0
    for data in islice(loader, num_batches_to_evaluate):
        img, lbl = data[0].to(device), data[1].to(device)
        img.requires_grad_(False)
        if attack:
            if attack == "FGSM":
                delta, _ = fgsm(model, img, lbl, upper_limit, lower_limit, mu, std, epsilon, 2 * epsilon)
            elif attack == "PGD":
                delta, _ = pgd(model, img, lbl, upper_limit, lower_limit, mu, std, epsilon, epsilon / 4, attack_iters=10)
            else:
                raise ValueError("Invalid Attack!")
            img += delta
            model.eval()
        pred = model(img)
        total += lbl.shape[0]
        correct += sum(torch.argmax(pred, axis=1) == lbl).item()

    return 100 * correct / total

def test(args, device, max_eps: int = 32):
    """
    Evaluation script for robustness curves using Clean, FGSM, and PGD.

    This module loads a trained model checkpoint, computes baseline clean accuracy,
    and measures adversarial accuracy for a sweep of ε values using:
        - Fast Gradient Sign Method (FGSM)
        - Projected Gradient Descent (PGD)

    Results are logged and saved as JSON for plotting robustness curves in research
    analysis.

    References:
        - Goodfellow, I. J., Shlens, J., & Szegedy, C. (2015).
        "Explaining and Harnessing Adversarial Examples."
        arXiv:1412.6572 (https://arxiv.org/abs/1412.6572)
        - Madry, A., Makelov, A., Schmidt, L., Tsipras, D., & Vladu, A. (2019).
        "Towards Deep Learning Models Resistant to Adversarial Attacks."
        arXiv:1706.06083 (https://arxiv.org/abs/1706.06083)
    """
    # Initialize tracker
    accs_vs_epps_tracker = MetricTracker()
    # Get dataset loaders
    trainloader, testloader, upper_limit, lower_limit, mu, std, _, num_classes, _, _ = get_loaders(args, False, device)
    _, C, H, W = get_input_dimensions(trainloader, False)
    
    final_checkpoint_path = f"{args.root_path}/Results/{args.dataset}/{args.model}/{args.attack}/checkpoints_{args.seed}/model{str(args.epochs).zfill(3)}.pt"
    
    model, _, _ = load_checkpoint(args, final_checkpoint_path, num_classes, H, C, len(trainloader), device)
    
    model.eval()
    
    clean_acc = test_step(model, None, testloader, upper_limit, lower_limit, mu, std, 0, args.attack_norm, device)
    accs_vs_epps_tracker.update(clean_acc=clean_acc)
    for eps in range(1, max_eps + 1):
        # Perform the tests
        fgsm_acc = test_step(model, "FGSM", testloader, upper_limit, lower_limit, mu, std, eps / 255, args.attack_norm, device)
        pgd_acc = test_step(model, "PGD", testloader, upper_limit, lower_limit, mu, std, eps / 255, args.attack_norm, device)
    
        accs_vs_epps_tracker.update(fgsm_acc=fgsm_acc, pgd_acc=pgd_acc)
        # Progress bar
        print(f'Epsilon {str(eps).zfill(2)}: Clean {clean_acc:.2f}% | FGSM {fgsm_acc:.2f}% | PGD {pgd_acc:.2f}%')

    metrics_to_save = {
        "accs_vs_eps_metrics": accs_vs_epps_tracker.to_dict()
    }
    with open(f"{args.root_path}/Results/{args.dataset}/{args.model}/{args.attack}/raw_results_{args.seed}/accs_vs_eps_metrics.json", "w") as f:
        json.dump(metrics_to_save, f, indent=4)
    
