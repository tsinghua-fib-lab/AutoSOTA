import numpy as np
import torch


def clip_gradients(net: torch.nn.Module, loss_train: torch.Tensor, num_train: int, dp_mech: str, grad_clip: float):
    clipped_grads = {
        name: torch.zeros_like(param)
        for name, param in net.named_parameters()
    }
    loss_train.mean().backward()

    if dp_mech == 'laplace':
        torch.nn.utils.clip_grad_norm_(
            net.parameters(), max_norm=grad_clip, norm_type=1)
    elif dp_mech == 'gaussian':
        torch.nn.utils.clip_grad_norm_(
            net.parameters(), max_norm=grad_clip, norm_type=2)
    for name, param in net.named_parameters():
        clipped_grads[name] += param.grad
    net.zero_grad()
    for name, param in net.named_parameters():
        clipped_grads[name] /= num_train
        param.grad = clipped_grads[name]


def Laplace_noise(args, dataset_size: int, x: torch.FloatTensor):
    times = args.num_rounds * args.client_frac
    each_query_eps = args.dp_eps / times
    sensitivity = 2 * args.lr * args.grad_clip / dataset_size
    scale = sensitivity / each_query_eps
    noise = torch.from_numpy(np.random.laplace(
        loc=0, scale=scale, size=x.shape)).to(x.device)
    return noise


def Gaussian_noise(args, dataset_size: int, x: torch.FloatTensor):
    times = args.num_rounds * args.client_frac
    each_query_eps = args.dp_eps / times
    each_query_delta = args.dp_delta / times
    sensitivity = args.grad_clip / dataset_size
    scale = sensitivity * \
        np.sqrt(2 * np.log(1.25 / each_query_delta)) / each_query_eps
    noise = torch.from_numpy(np.random.normal(
        loc=0, scale=scale, size=x.shape)).to(x.device)
    return noise


def add_noise(args, net: torch.nn.Module, dataset_size: int):
    with torch.no_grad():
        if args.dp_mech == 'laplace':
            for param in net.parameters():
                if param.requires_grad:
                    noise = Laplace_noise(args, dataset_size, param.data)
                    param.data.add_(noise)
        elif args.dp_mech == 'gaussian':
            for param in net.parameters():
                if param.requires_grad:
                    noise = Gaussian_noise(args, dataset_size, param.data)
                    param.data.add_(noise)
        else:
            raise NotImplementedError("This mechanism is not implemented!")
