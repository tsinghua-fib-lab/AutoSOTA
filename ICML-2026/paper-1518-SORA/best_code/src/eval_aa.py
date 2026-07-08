import torch
from torch import nn
import argparse
from dataloaders.get_loaders import get_loaders
from architectures.get_model import get_model
from training.utils import get_input_dimensions
from utils import create_directories, get_device, set_seed, load_checkpoint
from huggingface_hub import login, hf_hub_download

# simple Module to normalize an image
class Normalize(nn.Module):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        # Convert to tensor (if not already) and reshape to (1, C, 1, 1)
        # register_buffer ensures these tensors move to the correct device with the model
        self.register_buffer('mean', torch.as_tensor(mean).view(1, -1, 1, 1))
        self.register_buffer('std', torch.as_tensor(std).view(1, -1, 1, 1))

    def forward(self, x):
        # Broadcasting handles the (B, C, H, W) math automatically
        return (x - self.mean) / self.std
        

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset", choices=["CIFAR10", "CIFAR100", "CINIC10", "TinyImageNet", "PathMNIST", "TissueMNIST", "OrganAMNIST", "BloodMNIST"], default="CIFAR10")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--normalize_dataset", action="store_true")
    parser.add_argument("--model", choices=["PreActResNet18", "ResNet18", "WideResNet28", "SENet18"], default="PreActResNet18")
    parser.add_argument("--attack", choices=["SORA", "FGSM", "FGSM-RS", "GradAlign", "NFGSM", "AAER", "ZeroGrad", "MultiGrad", "ATAS", "ELLE", "TRADES", "PGD", "PGD2", "Benign"], required=True)
    parser.add_argument("--epsilon", type=float, default=8/255)
    parser.add_argument('--norm', type=str, default='Linf', choices=['L2', 'Linf'])
    parser.add_argument('--n_ex', type=int, default=-1)
    parser.add_argument('--log_path', type=str, default='aa_log.txt')
    parser.add_argument('--individual', default=False, action='store_true')
    parser.add_argument('--version', type=str, default='standard')
    parser.add_argument("--push_to_hf", action="store_true", help="Pushes results to Hugging Face")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--hf_token", type=str)
    parser.add_argument("--repo_id", type=str)
    
    return parser.parse_args()


def download_weights(args):
    login(token=args.hf_token, args.repo_id)
    path = f"{args.dataset}/{args.model}/{args.attack}/final_checkpoints_{args.seed}/model030.pt"
    local_file = hf_hub_download(repo_id=args.repo_id, filename=path, repo_type="model")
    return local_file
    

def main():    
    # Parse arguments
    args = parse_args()
    # Set seed
    set_seed(args.seed)
    # Create nescessary directories
    create_directories(args)
    # Get device
    device = get_device(args.device)
    # Get dataset loaders
    _, _, _, _, mean, std, _, _, _, _ = get_loaders(args, False, device)
    args.normalize_dataset = False
    _, testloader, _, _, _, _, _, num_classes, _, _ = get_loaders(args, False, device)
    _, C, H, W = get_input_dimensions(testloader, False)
    args.n_ex = args.n_ex if args.n_ex > 0 else len(testloader.dataset)
    # Get model
    net = get_model(args.model, num_classes, H, C)
    path = download_weights(args, hf_token, repo_id) 
    # path = f"{args.root_path}/Results/{args.dataset}/{args.model}/{args.attack}/final_checkpoints_{args.seed}/model030.pt"
    checkpoint = torch.load(path, weights_only=True)
    net.load_state_dict(checkpoint['model_state_dict'])
    model = nn.Sequential(Normalize(mean=mean, std=std), net).to(device)
    model.eval()
    
    # load attack    
    from autoattack import AutoAttack
    log_path = f'{args.root_path}/Results/{args.dataset}/{args.model}/{args.attack}/raw_results_{args.seed}/{args.log_path}'
    args.epsilon = 0.01176470588235294117647058823529 if args.dataset == "TissueMNIST" else args.epsilon # Epsilon = 3/255 for TissueMNIST
    adversary = AutoAttack(model, norm=args.norm, eps=args.epsilon , log_path=log_path)
    # set the number of classes
    adversary.fab.n_target_classes = min(num_classes - 1, adversary.fab.n_target_classes)
    adversary.apgd_targeted.n_target_classes = min(num_classes - 1, adversary.apgd_targeted.n_target_classes)
    
    l = [x for (x, y) in testloader]
    x_test = torch.cat(l, 0)
    l = [y for (x, y) in testloader]
    y_test = torch.cat(l, 0)
    
    # cheap version
    # example of custom version
    if args.version == 'custom':
        adversary.attacks_to_run = ['apgd-ce', 'fab']
        adversary.apgd.n_restarts = 2
        adversary.fab.n_restarts = 2

    # run attack and save images
    if not args.individual:
        adv_complete = adversary.run_standard_evaluation(x_test[:args.n_ex], y_test[:args.n_ex], bs=args.batch_size)

if __name__ == "__main__":
    main()
