_MODEL_KW = {
    "optimizer",
    "lr",
    "epochs",
    "model_size",
    "model_checkpoint",
    "save_model_ckpt",
}


def get_model(cfg, device, dataset_class, train_loader, val_loader, used_labels=None):
    model_kwargs = get_model_kwargs(cfg)
    model_kwargs["used_labels"] = used_labels

    model = dataset_class.get_model(device, train_loader, val_loader, **model_kwargs)

    return model


def get_model_kwargs(cfg):
    # Create subdict of cfg for the keys relevant to models
    model_kw = _MODEL_KW
    model_kwargs = {k: cfg[k] for k in cfg.keys() & model_kw}

    return model_kwargs
