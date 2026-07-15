import torch
from torch.utils.data import DataLoader


def get_executable_func(func_cfg):
    if func_cfg.module != "":
        import importlib

        mdl = importlib.import_module(func_cfg.module)
        return getattr(mdl, func_cfg.call)
    elif func_cfg.source != "":
        exec(func_cfg.source, globals())
        return eval(func_cfg.call)


def mse_loss(pred, y):
    return torch.nn.MSELoss()(pred.float(), y.float().unsqueeze(-1))


def get_loss_func(cfg):
    if cfg.loss == "":
        return get_executable_func(cfg.get_loss)()
    elif cfg.loss == "CrossEntropy":
        return torch.nn.CrossEntropyLoss()
    elif cfg.loss == "MSE":
        return mse_loss


def get_dataloader(cfg, dataset, mode):
    """Create a torch `DataLoader` object from the dataset, configuration, and set mode."""
    if dataset is None:
        return None
    if len(dataset) == 0:
        return None
    assert mode in ["train", "val", "test"]
    if mode == "train":
        ## Configure training at client
        batch_size = cfg.train_data_batch_size
        shuffle = cfg.train_data_shuffle
    else:
        batch_size = cfg.test_data_batch_size
        shuffle = cfg.test_data_shuffle
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=cfg.num_workers,
        shuffle=shuffle,
        pin_memory=True,
    )
