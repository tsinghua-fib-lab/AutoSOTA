from training.train import train
from evaluation.evaluation import evaluate
from evaluation.epsilons import test
from visualization.visualize import visualize
from utils import create_directories, get_device, set_seed
import argparse

def parse_args():
    """
    Parse arguments for the main function.

    Args:
        root_path (str): Path to the root directory of the project.
        seed (int): Seed for the random number generator.
        dataset (str): Dataset to use.
        batch_size (int): Batch size for the data loader.
        num_workers (int): Number of workers for the data loader.
        normalize_dataset (bool): Whether to normalize the dataset.
        model (str): Model to use.
        attack (str): Attack to use.
        epsilon (float): Epsilon for the attack.
        epochs (int): Number of epochs to train for.
        initial_lr (float): Initial learning rate.
        optimizer (str): Optimizer to use.
        momentum (float): Momentum for the optimizer.
        weight_decay (float): Weight decay for the optimizer.
        scheduler (str): Scheduler to use.
        track_alignment (bool): Whether to track alignment.
        device (str): Device to use.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset", choices=["CIFAR10", "CIFAR100", "TinyImageNet", "ImageNet100", "PathMNIST", "TissueMNIST"], default="CIFAR10")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--normalize_dataset", action="store_true")
    parser.add_argument("--model", choices=["PreActResNet18", "ResNet18", "WideResNet28", "SENet18"], default="PreActResNet18")
    parser.add_argument("--attack", choices=["SORA", "FGSM", "FGSM-RS", "GradAlign", "NuAT", "NFGSM", "AAER", "ZeroGrad", "MultiGrad", "ATAS", "ELLE", "TRADES", "PGD", "PGD2", "Benign"], required=True)
    parser.add_argument("--attack_norm", choices=["Linf", "L2"], default="Linf", help="Attack norm")
    parser.add_argument("--epsilon", type=float, default=8, help="Epsilon ball * 255")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--initial_lr", type=float, default=0.01, help="May be overwritten by scheduler")
    parser.add_argument("--max_lr", type=float, default=0.2, help="Maximum learning rate for Cyclic scheduler")
    parser.add_argument("--optimizer", choices=["SGD", "Adam", "AdamW"], default="SGD")
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--scheduler", choices=["Cyclic", "MultiStep", "CosineAnnealing", "WarmupLambda", "None"], default="Cyclic")
    parser.add_argument("--track_alignment", action="store_true")
    parser.add_argument("--evaluate_aa", action="store_true")
    parser.add_argument("--push_to_hf", action="store_true", help="Pushes results to Hugging Face")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--label_smoothing", type=float, default=0.1, help="Label smoothing for cross-entropy loss (0=disable)")
    
    return parser.parse_args()

def main():
    """
    Main function for the project.
    This function is used to train the model, evaluate the model, and visualize the results.
    It also sets the seed for the random number generator and creates the necessary directories.
    It also gets the device to use for the training.
    It also trains the model, evaluates the model, and visualizes the results.
    It also sets the seed for the random number generator and creates the necessary directories.
    It also gets the device to use for the training.
    It also trains the model, evaluates the model, and visualizes the results.
    """
    
    # Parse arguments
    args = parse_args()
    # Set seed
    set_seed(args.seed)
    # Create nescessary directories
    create_directories(args)
    # Get device
    device = get_device(args.device)
    # Train model
    args.epsilon /= 255
    train(args, device)
    # Evaluate training
    evaluate(args, device)
    # Evaluate accuracy vs epsilons
    test(args, device, max_eps=32)
    # Visualize results
    visualize(args)

if __name__ == "__main__":
    main()
