import torch
import torch.nn.functional as F
import time
import json
from dataloaders.get_loaders import get_loaders
from utils import load_checkpoint
from attacks.get_attack import get_attack
from attacks.fgsm import fgsm
from attacks.pgd import pgd
from attacks.attack_params import get_attack_params
from training.utils import MetricTracker, calculate_batch_corrects, get_input_dimensions

def evaluate(args, device):
    """
    Evaluate the model.
    
    Args:
        args (argparse.Namespace): Arguments for the training.
        device (torch.device): Device to use for the training.
    """
    
    index_dataset = args.attack in ["ATAS"]
    # Get dataset loaders
    trainloader, testloader, upper_limit, lower_limit, mu, std, _, num_classes, num_train_samples, num_test_samples = get_loaders(args, index_dataset, device)
    _, C, H, W = get_input_dimensions(trainloader, index_dataset)
    # Get attack parameters
    attack_params = get_attack_params(args).get(args.attack, {}).copy()

    use_regularizer = args.attack in ["TRADES", "GradAlign", "ELLE", "NuAT"]

    # Setup metric trackers
    trackers = {
        "attack": {"batch": MetricTracker(), "epoch": MetricTracker()},
        "benign": {"batch": MetricTracker(), "epoch": MetricTracker()},
        "fgsm": {"batch": MetricTracker(), "epoch": MetricTracker()},
        "pgd": {"batch": MetricTracker(), "epoch": MetricTracker()},
        "reg": MetricTracker()
    }

    total_evaluation_time = 0
    for epoch in range(args.epochs + 1):
        start_time = time.time()
        model, _, _ = load_checkpoint(args, f"{args.root_path}/Results/{args.dataset}/{args.model}/{args.attack}/checkpoints_{args.seed}/model{str(epoch).zfill(3)}.pt", num_classes, H, C, len(trainloader), device)
        model.eval()

        # Determine attack
        attack = get_attack(args.attack)
        for i, data in enumerate(testloader):
            images, labels = data[0].to(device), data[1].to(device)

            match args.attack:
                case "Benign":
                    delta = torch.zeros_like(images)
                case args.attack if args.attack in ["FGSM", "FGM", "FGSM-RS", "FGM-RS", "NFGSM", "ZeroGrad", "MultiGrad", "PGD", "PGD2"]:
                    delta, _ = attack(model, images, labels, upper_limit, lower_limit, mu, std, **attack_params)
                case args.attack if args.attack in ["TRADES", "GradAlign", "ELLE", "NuAT"]:
                    delta, reg, _ = attack(model, images, labels, upper_limit, lower_limit, mu, std, **attack_params)
                case "SORA":
                    delta, _, _ = attack(model, images, labels, upper_limit, lower_limit, mu, std, **attack_params)
                case "AAER":
                    delta, _, _, _ = attack(model, images, labels, upper_limit, lower_limit, mu, std, **attack_params)
                case "ATAS":
                    delta, _, _, _, _ = attack(args.dataset, model, images, labels, None, upper_limit, lower_limit, mu, std, **attack_params)
                case _:
                    raise ValueError("Invalid Attack Method!")
            # Add perturbation to original images
            adv_images = images + delta
            # Forward pass with adversarial examples
            preds = model(adv_images)
            loss = F.cross_entropy(preds, labels)

            #Track Regularizer Value Per Batch
            if use_regularizer:
                reg = float(reg) if reg is not None else 0.0
                trackers["reg"].update(batch_test_reg=reg)
            
            # Calculate attack accuracy
            batch_corrects = calculate_batch_corrects(preds, labels)
            trackers["attack"]["batch"].update(loss=loss.item(), accuracy=batch_corrects.item())
            # Calculate Benign accuracy
            eval_and_track(model, images, labels, torch.zeros_like(images), trackers["benign"]["batch"])
            # Calculate FGSM accuracy
            delta_fgsm, _ = fgsm(model, images, labels, upper_limit, lower_limit, mu, std, args.epsilon, 2 * args.epsilon, args.attack_norm)
            eval_and_track(model, images, labels, delta_fgsm, trackers["fgsm"]["batch"])
            
            # Calculate PGD accuracy # TODO adapt for dataset normalization
            delta_pgd, _ = pgd(model, images, labels, upper_limit, lower_limit, mu, std, args.epsilon, args.epsilon / 4, args.attack_norm, 10, 1)
            eval_and_track(model, images, labels, delta_pgd, trackers["pgd"]["batch"])

        
        attack_epoch_loss = trackers["attack"]["batch"].average("loss")
        attack_epoch_accuracy = trackers["attack"]["batch"].sum("accuracy") / num_test_samples
        benign_epoch_loss = trackers["benign"]["batch"].average("loss")
        benign_epoch_accuracy = trackers["benign"]["batch"].sum("accuracy") / num_test_samples
        fgsm_epoch_loss = trackers["fgsm"]["batch"].average("loss")
        fgsm_epoch_accuracy = trackers["fgsm"]["batch"].sum("accuracy") / num_test_samples
        pgd_epoch_loss = trackers["pgd"]["batch"].average("loss")
        pgd_epoch_accuracy = trackers["pgd"]["batch"].sum("accuracy") / num_test_samples
        
        finish_time = time.time()
        epoch_time = finish_time - start_time
        total_evaluation_time += epoch_time
        # Print epoch loss and accuracy
        print(f"Epoch {epoch} - {args.attack} Accuracy: {attack_epoch_accuracy:.2%}, Benign Accuracy: {benign_epoch_accuracy:.2%}, FGSM Accuracy: {fgsm_epoch_accuracy:.2%}, PGD Accuracy: {pgd_epoch_accuracy:.2%}, , Time {epoch_time:.4f}")
        trackers["attack"]["epoch"].update(loss=attack_epoch_loss, accuracy=attack_epoch_accuracy)
        trackers["benign"]["epoch"].update(loss=benign_epoch_loss, accuracy=benign_epoch_accuracy)
        trackers["fgsm"]["epoch"].update(loss=fgsm_epoch_loss, accuracy=fgsm_epoch_accuracy)
        trackers["pgd"]["epoch"].update(loss=pgd_epoch_loss, accuracy=pgd_epoch_accuracy)
        trackers["attack"]["batch"].reset()
        trackers["benign"]["batch"].reset()
        trackers["fgsm"]["batch"].reset()
        trackers["pgd"]["batch"].reset()
    
    # Save training metrics for processing and visualization
    metrics_to_save = {
        "attack_epoch_metrics": trackers["attack"]["epoch"].to_dict(),
        "benign_epoch_metrics": trackers["benign"]["epoch"].to_dict(),
        "fgsm_epoch_metrics": trackers["fgsm"]["epoch"].to_dict(),
        "pgd_epoch_metrics": trackers["pgd"]["epoch"].to_dict(),
        "regularizer_values": trackers["reg"].to_dict()
    }

    with open(f"{args.root_path}/Results/{args.dataset}/{args.model}/{args.attack}/raw_results_{args.seed}/evaluation_metrics.json", "w") as f:
        json.dump(metrics_to_save, f, indent=4)
    
    print('Finished Evaluating')
    print("Total Evaluation Time: ", total_evaluation_time)


def eval_and_track(model, images, labels, delta, tracker):
    with torch.no_grad():
        preds = model(images + delta)
        loss = F.cross_entropy(preds, labels)
        num_corrects = calculate_batch_corrects(preds, labels)
        tracker.update(loss=loss.item(), accuracy=num_corrects.item())
