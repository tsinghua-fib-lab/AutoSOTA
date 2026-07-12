# pylint: disable=no-value-for-parameter,missing-module-docstring,missing-function-docstring
from easydict import EasyDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os

from src.utils.group_convolution import layers as l
from src.utils.group_convolution import sampling as s
from src import datasets, groups

class Net(nn.Module):
    def __init__(self, config):
        super().__init__()
        fl_feature_dims = [1, 40, 40]
        scales = [1, 1]
        fl_MLP_dims = [20, 20]
        self.first_layer = l.FirstLayer(config.group_name, fl_feature_dims, config.embedding_dim, fl_MLP_dims, [1,20], scales, config.device)

        if config.group_name == 'affine':
            self.G = groups.ImageAffine(config.device)
        elif config.group_name == 'homography':
            self.G = groups.ImageHomography(config.device)
        else:
            raise NotImplementedError
        g_dim = self.G.num_generators_per_component
        hl_MLP_dims = [config.embedding_dim * g_dim, config.width]
        self.second_layer = l.FilterMLP(config.embedding_dim, hl_MLP_dims, config.output_dims1, config.device)

        self.fc1 = nn.Linear(config.width, config.width)
        self.fc2 = nn.Linear(config.width, config.width)

        pooling_input_dim = config.width
        pooling_output_dim = 10
        self.pooling_layer = l.PoolingLayer(pooling_input_dim, pooling_output_dim, config.device)

    def forward(self, data, v):
        z = self.first_layer(data, v[-1])
        z = F.silu(z)
        z = (1/v[0].shape[0])*torch.sum(torch.mul(z.unsqueeze(-1), self.second_layer(v[0])), dim=(-2, -3))
        z = F.silu(self.fc1(z)) + z
        z = F.silu(self.fc2(z)) + z
        z = self.pooling_layer(z)
        return z
    
class AddChannelWrapper(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        img = img.unsqueeze(0)
        return img, label


def main(config):
    device = config.device
    torch.cuda.empty_cache()

    # setup datamodule and group
    datamodule = datasets.PaddedMNIST(config)
    if config.dataset == "affNIST":
        g_datamodule = datasets.AffNIST(config)
        lie_scales = torch.tensor([0.168, 0.168, 0.841, 0.168, 0.168, 0.841]).to(device)

    elif config.dataset == "homNIST":
        g_datamodule = datasets.HomNIST(config)
        lie_scales = torch.tensor([0.15, 0.35, 0.5, 0.35, 0.15, 0.5, 0.15, 0.15]).to(device)
    else:
        raise NotImplementedError
    
    datamodule.prepare_data()
    datamodule.setup("fit")
    datamodule.setup("test")
    wrapped_train_dataset = AddChannelWrapper(datamodule.train_dataset)
    wrapped_test_dataset = AddChannelWrapper(datamodule.test_dataset)

    train_loader = torch.utils.data.DataLoader(wrapped_train_dataset, batch_size=config.train_batch, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(wrapped_test_dataset, batch_size=config.test_batch, shuffle=False, num_workers=2)

    g_datamodule.prepare_data()
    g_datamodule.setup("test")
    g_wrapped_test_dataset = AddChannelWrapper(g_datamodule.test_dataset)
    g_test_loader = torch.utils.data.DataLoader(g_wrapped_test_dataset, config.test_batch, shuffle=False, num_workers=2)

    # define model, optimizer, criterion
    net = Net(config).to(device)

    optimizer = optim.Adam(net.parameters(), lr=1e-3, betas=(0.9, 0.999), weight_decay=1e-8)
    criterion = nn.CrossEntropyLoss()

    epochs = config.epochs
    train_acc = np.zeros(epochs)
    train_loss = np.zeros(epochs)
    val_loss = np.zeros(epochs)
    test_acc = np.zeros(epochs)
    generate = s.generate_haar_PI_parallel(config.group_name, 100, 100, 1, lie_scales, device)

    for e in range(epochs):
        # --- training ---
        correct = 0
        total = 0
        running_loss = 0.0

        print("Epoch: ", e)

        for data in train_loader:                
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            v = generate.generate()
            outputs = net(inputs, v)

            _, predicted = torch.max(outputs.detach(), 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            loss = criterion(outputs, labels)
            running_loss += loss.item()

            loss.backward()
            optimizer.step()
            
        train_acc[e] = correct / total
        train_loss[e] = running_loss / len(train_loader)
        print(f"Training accuracy: {train_acc[e]:.4f}, loss: {train_loss[e]:.4f}")

        # --- validation ---
        correct = 0
        total = 0
        val_running_loss = 0.0

        with torch.no_grad():
            for data in test_loader:
                images, labels = data[0].to(device), data[1].to(device)
                v = generate.generate()

                outputs = net(images, v)

                # accuracy
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                # loss
                loss = criterion(outputs, labels)
                val_running_loss += loss.item()

            test_acc[e] = correct / total
            val_loss[e] = val_running_loss / len(test_loader)
            print(f"validation accuracy: {test_acc[e]:.4f}, loss: {val_loss[e]:.4f}")

        # --- checkpointing ---
        checkpoint = {
            "epoch": e,
            "model_state_dict": net.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_acc": train_acc[e],
            "val_acc": test_acc[e],
            "train_loss": train_loss[e],
            "val_loss": val_loss[e]
        }

        save_dir = f"./group_conv/{config.group_name}"
        os.makedirs(save_dir, exist_ok=True)
        save_path = f"{save_dir}/checkpoint_epoch_{e}.pth"
        torch.save(checkpoint, save_path)
        print(f"Checkpoint saved: {save_path}")

    # --- evaluation ---
    with torch.no_grad():

        correct = 0
        total = 0

        for data in g_test_loader:
            images, labels = data[0].to(device), data[1].to(device)
            v = generate.generate()

            outputs = net(images, v)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        test_g_acc_final = correct / total
        
    print("Final {} test accuracy: {}".format(config.dataset, test_g_acc_final))


if __name__ == "__main__":
    from easydict import EasyDict
    config_ = EasyDict({
        'device': torch.device("cuda:7"),
        'data_dir': './experiments/datasets',
        'batch_size': 32,
        'num_workers': 4,
        'group_name': 'affine', # 'affine' or 'homography'
        'dataset': 'affNIST', # 'affNIST' or 'homNIST'
        'width' : 128, # width used in experiments
        'embedding_dim' : 10,
        'output_dims1' : [20, 128], # second value must be the same as width
        'train_batch' : 60,
        'test_batch' : 290,
        'epochs' : 50, # originally 150
    })
    main(config_)
    

