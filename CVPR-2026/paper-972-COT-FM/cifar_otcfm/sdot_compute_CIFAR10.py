import torch
from torchvision import datasets, transforms

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
import os
USE_MULTI_GPU = "LOCAL_RANK" in os.environ


dataset = datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    ),
)

data_tensor = torch.stack([data for data, label in dataset]).cuda()
data_augmentation = transforms.RandomHorizontalFlip(p=1)
data_augmented = data_augmentation(data_tensor)
data_tensor = torch.concat([data_tensor, data_augmented])
data_tensor_flatten = data_tensor.reshape(data_tensor.shape[0], -1)
dual_weight = torch.zeros(len(data_tensor_flatten), device=data_tensor.device)
torch.save(data_tensor, './pickled_cifar10_augmented.pt')
exit()
#
#
dual_weight = dual_weight.cuda()
filename = "./CIFAR10_dual_weight_flip.pt"

dual_weight, info_dict = sdot_solve(
    data_tensor_flatten,
    use_multi_gpu=USE_MULTI_GPU,
    lr=10,
    num_step=1_000,
    batch_size=1024,
    dual_weight_init=dual_weight,
    eps_entropic=1,
    save_every=1000,
    ema_param=0.99,
    filename=filename,
)

dual_weight, info_dict = sdot_solve(
    data_tensor_flatten,
    use_multi_gpu=USE_MULTI_GPU,
    lr=0.1,
    num_step=5_000,
    batch_size=4096,
    dual_weight_init=dual_weight,
    eps_entropic=1,
    save_every=1000,
    ema_param=0.999,
    filename=filename,
    info_dict=info_dict,
)

dual_weight, info_dict = sdot_solve(
    data_tensor_flatten,
    use_multi_gpu=USE_MULTI_GPU,
    lr=0.1,
    num_step=5_000,
    batch_size=16384,
    dual_weight_init=dual_weight,
    eps_entropic=0.01,
    save_every=1000,
    ema_param=0.999,
    filename=filename,
    info_dict=info_dict,
)

torch.save(dual_weight, filename)
torch.save(info_dict, "./info_dict.pt")
