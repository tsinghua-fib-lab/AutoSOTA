import torch

import torchvision.datasets as datasets
import torchvision.transforms as transforms

from utils.utils import ResNetImpala, count_vars
from utils.likelihoods import FISH_LIKELIHOODS

def read_cifar(path, if_autoencoder=False):
    # code from https://github.com/jeonsworld/MLP-Mixer-Pytorch/blob/main/utils/data_utils.py
    image_size = 64
    transform_train = transforms.Compose(
        [
            transforms.RandomResizedCrop((image_size, image_size), scale=(0.05, 1.0)),
            transforms.ToTensor(),
        ]
    )
    transform_test = transforms.Compose(
        [
            transforms.RandomResizedCrop((image_size, image_size)),
            transforms.ToTensor(),
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

    if if_autoencoder:
        train_data.targets = train_data.data
        test_data.targets = test_data.data

    data_sets.train = train_data
    data_sets.test = test_data

    return data_sets

def load_dataset(batch_size):
    dataset = read_cifar("data/", if_autoencoder=False)

    ## Dataset
    train_dataset = dataset.train
    test_dataset = dataset.test
    print("Number of training samples: ", len(train_dataset))
    print("Number of testing samples: ", len(test_dataset))
    print("Image shape: ", train_dataset[0][0].shape)

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

def test_resnet():
    train_loader, _, test_loader = load_dataset(500)

    embed_dim=256
    kwargs = {'with_norm_layer': True, 'depths': [32, 64, 64]}
    embed_nets = ResNetImpala(64, embed_dim, **kwargs)
    
    # embed_nets = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)

    class TestModel(torch.nn.Module):
        def __init__(self, embed_net, embed_dim, n_actions):
            super().__init__()
            self.n_actions = n_actions # used for one-hot encoding
            self.embed_net = embed_net

            self.logits_net = torch.nn.Sequential(
                torch.nn.Linear(embed_dim, n_actions),
            )

        def forward(self, obs):
            latents = self.embed_net(obs)
            logits = self.logits_net(latents)
            return logits

    model = TestModel(embed_nets, embed_dim, 10)
    print('Number of parameters: ', count_vars(model))

    lr = 0.0005
    weight_decay = 1e-5

    if torch.cuda.is_available(): # i.e. for NVIDIA GPUs
        device_type = "cuda"
    else:
        device_type = "cpu"
    device = torch.device(device_type) # Select best available device

    model = model.to(device)
    likelihood = FISH_LIKELIHOODS["softmax"](device=device)

    opt = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    import time
    st = time.time()
    eval_time = 0

    n_steps = 0
    n_test_steps = 0
    from tqdm import tqdm
    for epoch in range(1, 50 + 1):
        with tqdm(train_loader, unit="batch") as tepoch:
            running_loss = 0
            running_acc = 0
            for n, (batch_data, batch_labels) in enumerate(tepoch, start=1):
                tepoch.set_description(f"Epoch {epoch}")

                batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)

                opt.zero_grad()
                output = model(batch_data)
                loss = likelihood(output, batch_labels)
                loss.backward()
                opt.step()

                acc = class_accuracy(output, batch_labels)

                running_loss += loss.item()
                running_acc += acc.item()

                et = time.time()     

                if n % 50 == 0:
                    model.eval()

                    running_test_loss = 0
                    running_test_acc = 0

                    for m, (test_batch_data, test_batch_labels) in enumerate(test_loader, start=1):
                        test_batch_data, test_batch_labels = test_batch_data.to(device), test_batch_labels.to(device)

                        test_output = model(test_batch_data)

                        test_loss = likelihood(test_output, test_batch_labels).item()
                        test_acc = class_accuracy(test_output, test_batch_labels).item()

                        running_test_loss += test_loss
                        running_test_acc += test_acc

                    running_test_loss /= m
                    running_test_acc /= m

                    n_test_steps += 1
                    tepoch.set_postfix(acc=100 * running_acc / n, test_acc=running_test_acc * 100)
                    model.train()
                    eval_time += time.time() - et
            
                n_steps += 1

            epoch_time = time.time() - st - eval_time
            tepoch.set_postfix(loss=running_loss / n, test_loss=running_test_loss, epoch_time=epoch_time)