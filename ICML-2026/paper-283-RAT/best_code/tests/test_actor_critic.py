import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from utils.utils import build_cnn
from utils.utils import SharedActorCritic, SeparateActorCritic
from utils.likelihoods import FISH_LIKELIHOODS

def read_cifar(path, preprocess):
    # code from https://github.com/jeonsworld/MLP-Mixer-Pytorch/blob/main/utils/data_utils.py
    image_size = 64
    transform_train = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            preprocess,
        ]
    )
    transform_test = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            preprocess,
        ]
    )

    train_data = datasets.CIFAR10(
        root="data", train=True, download=True, transform=transform_train, target_transform=transforms.Lambda(
            lambda y: torch.zeros(10).scatter_(0, torch.tensor(y), value=1)
        )
    )
    test_data = datasets.CIFAR10(
        root="data", train=False, download=True, transform=transform_test, target_transform=transforms.Lambda(
            lambda y: torch.zeros(10).scatter_(0, torch.tensor(y), value=1)
        )
    )

    class DataSets(object):
        pass

    data_sets = DataSets()

    data_sets.train = train_data
    data_sets.test = test_data

    return data_sets

def load_dataset(batch_size, preprocess):
    dataset = read_cifar("data/", preprocess)

    ## Dataset
    train_dataset = dataset.train
    test_dataset = dataset.test
    print("Number of training samples: ", len(train_dataset))
    print("Number of testing samples: ", len(test_dataset))
    # print("Image shape: ", train_dataset[0][0].shape)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    aux_loader = torch.utils.data.DataLoader(
        train_dataset, shuffle=True, batch_size=batch_size
    )
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)

    return train_loader, aux_loader, test_loader

def class_accuracy(predictions, labels):
    y = torch.max(predictions, 1)[1]
    y_labels = torch.max(labels, 1)[1]

    return torch.mean(y.eq(y_labels).float())

def test_actor_critic():
    if torch.cuda.is_available(): # i.e. for NVIDIA GPUs
        device_type = "cuda"
    else:
        device_type = "cpu"
    device = torch.device(device_type) # Select best available device

    embed_dim=256
    kwargs = {'with_norm_layer': True, 'p_dropblock': 0.1, 'device': device}
    embed_nets, preprocess = build_cnn(64, embed_dim, **kwargs)

    shared_ac = SharedActorCritic(embed_nets, embed_dim, 10, False).to(device)
    sep_ac = SeparateActorCritic(embed_nets, embed_dim, 10, False).to(device)

    train_loader, _, _ = load_dataset(500, preprocess)

    lr = 0.0005
    weight_decay = 1e-5

    likelihood = FISH_LIKELIHOODS["softmax"](device=device)

    opt_shared = torch.optim.Adam(
        shared_ac.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )
    opt_sep = torch.optim.Adam(
        sep_ac.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    batch_data, batch_labels = next(iter(train_loader)) 

    batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)

    ## test shared actor critic
    shared_ac.eval()
    # actor part
    opt_shared.zero_grad()
    vals_before, logits = shared_ac(batch_data) # check logits only
    loss = likelihood(logits, batch_labels)
    loss.backward()
    opt_shared.step()
    vals_after, new_logits = shared_ac(batch_data) # check logits only
    assert not torch.allclose(vals_before, vals_after)
    assert not torch.allclose(logits, new_logits)
    # critic part
    opt_shared.zero_grad()
    vals, logits_before = shared_ac(batch_data) # check logits only
    loss = torch.sum(vals)
    loss.backward()
    opt_shared.step()
    new_vals, logits_after = shared_ac(batch_data) # check logits only
    assert not torch.allclose(logits_before, logits_after)
    assert not torch.allclose(vals, new_vals)

    ## test separate actor critic
    sep_ac.eval()
    # actor
    opt_sep.zero_grad()
    vals_before, logits = sep_ac(batch_data) # check logits only
    loss = likelihood(logits, batch_labels)
    loss.backward()
    opt_sep.step()
    vals_after, new_logits = sep_ac(batch_data) # check logits only
    assert torch.allclose(vals_before, vals_after)
    assert not torch.allclose(logits, new_logits)
    # critic
    opt_sep.zero_grad()
    vals, logits_before = sep_ac(batch_data) # check logits only
    loss = torch.sum(vals)
    loss.backward()
    opt_sep.step()
    new_vals, logits_after = sep_ac(batch_data) # check logits only
    assert torch.allclose(logits_before, logits_after)
    assert not torch.allclose(vals, new_vals)