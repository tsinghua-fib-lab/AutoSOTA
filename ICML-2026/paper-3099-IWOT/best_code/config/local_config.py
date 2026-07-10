def setup_local_config():
    config = {
        # Experiment details
        "device": "auto",  # 'cpu' or 'auto' to find gpu automatically
        # Model and optimizer (MLP, ConvNet, ConvNet2, LeNet, SmallCNN, ResNet)
        "model": "ResNet",
        "resnet_size": 18,  # 18 or 50
        "load_imagenet_weights": True,
        "pretrain": True,
        "num_pretrain_epochs": 5,  # if pretrain is True
        "loss": "margin",  # 'margin', 'euclidean', 'cross-entropy'
        "optimizer": "adam",  # alternatives: adam, sgd
        "adam_beta2": 0.98,
        "learning_rate": 1e-3,  # use 1e-4 for ResNets or a learning scheduler
        "momentum": 0.9,  # for SGD
        "weight_decay": 0.0,
        "num_epochs": 5,
        "num_runs": 5,
        "algs": [
            "wrr",
        ],  # wrr, weighted_wrr, cons_wrr, jdot, lje, erm, cc, dann, fdal, reverse-kl, mmd
        # Debugging algorithms
        "debug": False,
        "debug_every_n": 50,
        "n_batches_per_epoch": -1,  # if -1 uses all batches
        "report_source_train_risk": False,
        "report_target_train_risk": False,
        "pretrain_on_both": False,  # starting from a model that 'cheats'!
    }

    scenario_config = {
        # Distribution shift scenario
        # MNIST_TO_USPS, USPS_TO_MNIST, MNIST_TO_MNIST_M, SVHN_TO_MNIST,
        # CIFAR-10-C, PORTRAITS, OFFICE_31, OFFICEHOME, IMAGECLEFDA, VISDA17
        "scenario": "MNIST_TO_USPS",
        # Enable only when running a dataset for the first-time
        "preprocess": False,
        # Data loader options
        "batch_size": 64,
        # Test set dataloader options
        "test_batch_size": 512,
        "shuffle": True,
        "test_shuffle": True,
        # Dataset-specific settings
        "cifar-10-corruptions": ["fog", "frost", "snow"],
        "portraits-size": [186, 171],
        "portraits-grayscale": False,
        "officehome-target": "real world",  # 'art', 'clipart', 'product', 'real world'
        "officehome-size": [224, 224],
        "office-31-target": "webcam",  # 'amazon', 'dslr', 'webcam'
        "office-31-size": [300, 300],
        "imageclef-size": [300, 300],
        "imageclef-target": "pascal",  # 'bing', 'caltech', 'imagenet', 'pascal'
        "visda17-size": [384, 216],
    }

    debug_config = {
        "calc_label_shift": False,
        "calc_entanglement": False,
        "calc_margin": True,
        "calc_wrr": True,
        "calc_weighted_wrr": True,
        "verbose_weighted_wrr": False,
        "calc_weight_info": False,
        "calc_grad_info": False,
        "calc_gradual_shift": True,
        "est_lambda": True,
    }

    config["scenario_options"] = scenario_config
    config["debug_options"] = debug_config

    # Algorithms and their hyperparameters/options
    config = setup_alg_config(config)
    return config


def setup_alg_config(config):
    config["wrr"] = {
        "scale": 1.0,
        "norm": 2,
        "entropy_reg": 1e-3,
        "print_info": False,
        "propagate_labels": False,
        "compute_ultrametric": False,
        "estimate_entanglement": False,
        "softmax_temperature": 0.1,
    }

    config["weighted_wrr"] = {
        "scale": 1.0,
        "entropy_reg": 1e-1,  # only for sinkhorn uot
        "add_source_loss": True,
        "separate_optim": True,
        "uot_alg": "mm",  # sinkhorn or mm
        "uot_init": False,  # initialize MM with semi-relaxed UOT
        "uot_iter_max": 1000,
        "autograd_at_convergence": True,
        "reg_m": (1.0, 100.0),
        "print_info": False,
    }

    config["pseudolabel"] = {
        "linkage": "single",
    }

    config["jdot"] = {
        "alpha": 0.001,
        "lambda": 0.001,
        "track_layer": "flatten", #"flatten" for ConvNet, -2 for MLP
        "add_source_loss": True,
        "use_squared_dist": False,
    }

    config["cons_wrr"] = {"norm": 2, "entropy_reg": 1e-3, "scale": 1.0, "thresh": 0.01}

    config["mmd"] = {
        "alpha": 0.1,
        "gammas": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
        "use_squared_dist": True,
        "use_features": True, # if false uses predictions
        "track_layer": "flatten", #"flatten" for ConvNet, -2 for MLP
    }

    config["dann"] = {
        "conv_feat_layer": "flatten",
        "mlp_feat_layer": -2,  # ignored for ResNets
        "discriminator": "conv",  # conv or mlp
        "learning_rate": 1e-3,  # for the internal optimizer
        "weight_decay": 0.0,
        "num_epochs": 2,
        "num_batches": 1000,  # rough estimate should be enough
    }

    config["fdal"] = {
        "juncture": -1,  # For now we keep backbone/taskhead juncture at the last layer only
        "auxhead": "conv",  # conv or none
        "grl": {"max_iters": 3000, "hi": 0.6, "auto_step": True},
        "divergence": "pearson",
        "learning_rate": 1e-4,  # for the internal optimizer
        "weight_decay": 0.0,
        "clip_grad_val": 10,
        "reg_coef": 0.1,
    }

    config["cc"] = {
        "entropy_reg": 1e-3,
        "norm": 2,
        "mode": "joint",  # or 'weighted_joint' or 'conditional'
        "add_source_loss": False,  # only for weighted_joint
    }

    config["reverse_kl"] = {
        "alpha_reverse": 0.1,
        "alpha_forward": 0.1,
        "augment_softmax": 0.0,
    }

    return config
