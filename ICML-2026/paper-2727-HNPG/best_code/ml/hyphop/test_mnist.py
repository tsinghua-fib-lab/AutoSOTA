import argparse
import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import math
import copy

from models.wrappers import SingleInstanceClassifier as ModelWrapper
from utils.device import select_device

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        # ALGO-5: Update EMA shadow weights
        with torch.no_grad():
            for ema_p, p in zip(_ema_model.parameters(), model.parameters()):
                ema_p.data.mul_(_ema_decay).add_(p.data, alpha=1.0 - _ema_decay)
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    
    accuracy = 100. * correct / len(test_loader.dataset)
    return accuracy

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Core MNIST')
    parser.add_argument('--model', type=str, default='kf_attention', 
                        choices=['kf_attention', 'kf_layer', 'kf_pooling', 
                                 'hf_attention', 'hf_layer', 'hf_pooling',
                                 'ein_attention', 'ein_layer', 'ein_pooling'])
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--gamma', type=float, default=0.96, metavar='M',
                        help='Learning rate step gamma (default: 0.96)')
    parser.add_argument('--hidden-dim', type=int, default=8,
                        help='hidden dimension d (default: 8)')
    parser.add_argument('--beta', type=float, default=None,
                        help='scaling factor beta; default derives from d')
    parser.add_argument('--num-states', type=int, default=1,
                        help='num pooling states (default: 1)')
    parser.add_argument('--num-memories', type=int, default=64,
                        help='num static memories (default: 64)')
    parser.add_argument('--no-accel', action='store_true',
                        help='disables accelerator')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cpu', 'cuda', 'mps'],
                        help='device to use (default: auto)')
    parser.add_argument('--dry-run', action='store_true',
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', 
                        help='For Saving the current Model')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = select_device(args.device, no_accel=args.no_accel)

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if device.type != "cpu":
        accel_kwargs = {'num_workers': 1,
                        'persistent_workers': True,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(accel_kwargs)
        test_kwargs.update(accel_kwargs)

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    dataset1 = datasets.MNIST('./datasets', train=True, download=True,
                       transform=transform)
    dataset2 = datasets.MNIST('./datasets', train=False,
                       transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    input_dim = dataset1[0][0].numel()

    model = ModelWrapper(
        mode=args.model,
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        num_classes=10,
        beta=args.beta,
        num_states=args.num_states,
        num_memories=args.num_memories,
    ).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # ALGO-5: EMA of model weights for inference stability
    global _ema_model, _ema_decay
    _ema_decay = 0.999
    _ema_model = copy.deepcopy(model)
    for p in _ema_model.parameters():
        p.requires_grad = False

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), "mnist_core.pt")
    
    # ALGO-5: Use EMA weights for final evaluation
    # Save original weights, load EMA, test, restore
    orig_state = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(_ema_model.state_dict())
    final_acc = test(model, device, test_loader)
    model.load_state_dict(orig_state)
    return final_acc

if __name__ == '__main__':
    main()

