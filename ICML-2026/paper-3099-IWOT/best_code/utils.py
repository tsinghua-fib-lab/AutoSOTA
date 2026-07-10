import os
import time
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader


class ForeverDataIterator:
    """A data iterator that will never stop producing data.
    Taken from fDAL code (MIT License): https://github.com/nv-tlabs/fdal,
    who also took it from somewhere else it seems."""

    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader
        self.iter = iter(self.data_loader)

    def __next__(self):
        try:
            data = next(self.iter)
        except StopIteration:
            self.iter = iter(self.data_loader)
            data = next(self.iter)
        return data

    def __len__(self):
        return len(self.data_loader)


class GenericDataset(Dataset):
    def __init__(self, X_train, y_train, transform=None):
        self.data = X_train
        self.targets = y_train
        self.transform = transform

    def __getitem__(self, index):
        data, targets = self.data[index], self.targets[index]
        if self.transform is not None:
            data = self.transform(data)
        return data, targets

    def __len__(self):
        return self.data.shape[0]


def one_hot(x, k):
    return torch.nn.functional.one_hot(x, k).float()


def test(dataloader, model, loss_fun, fabric):
    model.eval()
    test_loss, correct = 0, 0
    num_points = 0
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            # We expect loss to reduce to mean here!
            test_loss += loss_fun(pred, one_hot(y, model.num_classes)).item() * y.size(
                0
            )
            correct += (pred.argmax(1) == y).sum()
            num_points += y.size(0)
    num_points = fabric.all_reduce(num_points, reduce_op="sum")
    test_loss = fabric.all_reduce(test_loss, reduce_op="sum") / num_points
    correct = fabric.all_reduce(correct, reduce_op="sum") / num_points
    print(f"Accuracy: {(100*correct):0.2f}%, Avg loss: {test_loss:.6f} \n")
    return test_loss, correct


def report_metrics(
    scenario, model, loss_fun, report_source_train, report_target_train, fabric
):
    # These are very slow
    if report_source_train:
        print(
            f"Reporting accuracy/loss on source {scenario.source_name} training dataset..."
        )
        test(scenario.source_dataloader, model, loss_fun, fabric)

    if report_target_train:
        print(
            f"Reporting accuracy/loss on target {scenario.target_name} training dataset..."
        )
        test(scenario.target_dataloader, model, loss_fun, fabric)

    print(f"Reporting accuracy/loss on {scenario.source_name} test dataset...")
    loss_source, acc_source = test(
        scenario.source_test_dataloader, model, loss_fun, fabric
    )

    print(f"Reporting accuracy/loss on {scenario.target_name} test dataset...")
    loss_target, acc_target = test(
        scenario.target_test_dataloader, model, loss_fun, fabric
    )
    return torch.tensor([loss_source, acc_source, loss_target, acc_target])


def train_model_on_source(config, model, loss_fun, scenario, opt, fabric):
    folder_path = "save_files/" + scenario.name + "/"
    save_path = folder_path + model.name + ".pth"
    train(
        scenario.source_dataloader,
        model,
        loss_fun,
        opt,
        config["num_pretrain_epochs"],
        fabric,
        report_every=10,
    )
    # Report accuracy/loss on whole training dataset
    test(scenario.source_dataloader, model, loss_fun, fabric)
    fabric.barrier()
    if fabric.global_rank == 0:
        print(f"Saving parameters to file: {save_path}")
        os.makedirs(folder_path, exist_ok=True)
        torch.save(model.state_dict(), save_path)
    fabric.barrier()
    return model


# Checks that the same network architecture used for both source *and* target can achieve high accuracy
def train_model_on_source_and_target(config, model, loss_fun, scenario, opt, fabric):
    folder_path = "save_files/" + scenario.name + "/train_on_both/"
    save_path = folder_path + model.name + ".pth"
    combined_data = torch.utils.data.ConcatDataset(
        [scenario.source_data, scenario.target_data]
    )
    dataloader = torch.utils.data.DataLoader(
        combined_data, **scenario.dataloader_options
    )
    train(
        dataloader,
        model,
        loss_fun,
        opt,
        config["num_pretrain_epochs"],
        fabric,
        report_every=10,
    )   
    # Report accuracy/loss on whole training dataset
    test(scenario.source_dataloader, model, loss_fun, fabric)
    fabric.barrier()
    if fabric.global_rank == 0:
        print(f"Saving parameters to file: {save_path}")
        os.makedirs(folder_path, exist_ok=True)
        torch.save(model.state_dict(), save_path)
    fabric.barrier()
    return model


def train(
    dataloader,
    model,
    loss_fun,
    optimizer,
    num_epochs,
    fabric,
    report_every=10,
    report_metrics=False,
):
    size = len(dataloader.dataset)
    t0 = time.perf_counter()
    model.train()
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}\n-------------------------------")
        for batch, (X, y) in enumerate(dataloader):
            y = one_hot(y, model.num_classes)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fun(model(X), y)
            fabric.backward(loss)
            optimizer.step()
            if batch % report_every == 0:
                loss, current = loss.item(), (batch + 1) * len(X)
                print(f"loss: {loss:>7f} epoch:{epoch+1} [{current:>5d}/{size:>5d}]")
        if report_metrics is True:
            print("Train dataset metrics:")
            test(dataloader, model, loss_fun, fabric)
    print(f"Method took {time.perf_counter() - t0} sec")


# Assuming Euclidean distance is to be used for W_{1,l} computation,
# the result is equal to \sqrt(2) / 2 * \sum_i |p_i - q_i|
def calc_w_distance_label_shift(scenario):
    num_cond_source = torch.zeros(scenario.num_classes)
    num_cond_target = torch.zeros(scenario.num_classes)
    for (X_train, y_train), (X_shift, y_shift) in zip(
        scenario.source_dataloader, scenario.target_dataloader
    ):
        for i in range(scenario.num_classes):
            num_cond_source[i] += torch.sum(y_train == i)
            num_cond_target[i] += torch.sum(y_shift == i)
    p_y = num_cond_source / torch.sum(num_cond_source)
    q_y = num_cond_target / torch.sum(num_cond_target)
    w_1_euclidean_dist = np.sqrt(2) * torch.sum(torch.abs(p_y - q_y)) / 2
    print(f"W1_distance_labels for {scenario.name} is {w_1_euclidean_dist}")


def calc_margin(preds, labels):
    # Get correct points with matching labels
    # Find the gap between max and second max (by sorting for now)
    pred_sorted_val, pred_sorted_ind = torch.sort(preds, dim=1, descending=True)
    correct = pred_sorted_ind[:, 0] == labels.argmax(1)
    margin = pred_sorted_val[correct, 0] - pred_sorted_val[correct, 1]
    std_margin, mean_margin = torch.std_mean(margin)
    return std_margin, mean_margin, margin, correct
