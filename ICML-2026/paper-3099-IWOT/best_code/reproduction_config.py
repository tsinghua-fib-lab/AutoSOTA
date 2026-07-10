"""Reproduction config for paper 3099: MNIST->USPS MLP W2R2 experiment.
Matches rubric conditions: MLP, Adam, lr=1e-3, batch_size=64, epochs=10,
pretrain_epochs=5, n_trials=5, margin multiclass loss, mm-UOT solver, lambda=1.
"""
def setup_reproduction_config():
    config = {
        # Experiment details
        "device": "cuda",  # use 'cuda' explicitly (not 'auto') for Lightning 2.x compat
        # Model and optimizer
        "model": "MLP",
        "resnet_size": 18,
        "load_imagenet_weights": True,
        "pretrain": True,
        "use_batchnorm": True,  # BatchNorm1d after each hidden Linear layer
        "num_pretrain_epochs": 5,
        "loss": "margin",  # margin multiclass loss (Crammer & Singer formulation)
        "optimizer": "adam",
        "adam_beta2": 0.98,
        "learning_rate": 1e-3,
        "momentum": 0.9,
        "weight_decay": 0.0,
        "num_epochs": 5,  # 5 adaptation epochs (paper: 10 total = 5 pretrain + 5 adapt)
        "num_runs": 5,  # 5 trials
        "algs": [
            "weighted_wrr",
        ],  # W2R2 = weighted WRR with mm-UOT solver
        # Debugging
        "debug": False,
        "debug_every_n": 50,
        "n_batches_per_epoch": -1,  # use all batches
        "report_source_train_risk": False,
        "report_target_train_risk": False,
        "pretrain_on_both": False,
    }

    scenario_config = {
        "scenario": "MNIST_TO_USPS",
        "preprocess": False,
        "batch_size": 64,
        "test_batch_size": 512,
        "shuffle": True,
        "test_shuffle": True,
        "cifar-10-corruptions": ["fog", "frost", "snow"],
        "portraits-size": [186, 171],
        "portraits-grayscale": False,
        "officehome-target": "real world",
        "officehome-size": [224, 224],
        "office-31-target": "webcam",
        "office-31-size": [300, 300],
        "imageclef-size": [300, 300],
        "imageclef-target": "pascal",
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

    # Algorithm configs
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
        "entropy_reg": 1e-1,       # only for sinkhorn uot
        "add_source_loss": True,
        "separate_optim": False,
        "uot_alg": "mm",            # mm-UOT solver
        "uot_init": False,          # initialize MM with semi-relaxed UOT
        "uot_iter_max": 1000,
        "autograd_at_convergence": True,
        "reg_m": (1.0, 100.0),      # lambda=1 (rho=1, rho2=100 per paper eq 14)
        "print_info": False,
    }

    config["pseudolabel"] = {
        "linkage": "single",
    }

    config["jdot"] = {
        "alpha": 0.001,
        "lambda": 0.001,
        "track_layer": "flatten",
        "add_source_loss": True,
        "use_squared_dist": False,
    }

    config["cons_wrr"] = {"norm": 2, "entropy_reg": 1e-3, "scale": 1.0, "thresh": 0.01}

    config["mmd"] = {
        "alpha": 0.1,
        "gammas": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
        "use_squared_dist": True,
        "use_features": True,
        "track_layer": "flatten",
    }

    config["dann"] = {
        "conv_feat_layer": "flatten",
        "mlp_feat_layer": -2,
        "discriminator": "conv",
        "learning_rate": 1e-3,
        "weight_decay": 0.0,
        "num_epochs": 2,
        "num_batches": 1000,
    }

    config["fdal"] = {
        "juncture": -1,
        "auxhead": "conv",
        "grl": {"max_iters": 3000, "hi": 0.6, "auto_step": True},
        "divergence": "pearson",
        "learning_rate": 1e-4,
        "weight_decay": 0.0,
        "clip_grad_val": 10,
        "reg_coef": 0.1,
    }

    config["cc"] = {
        "entropy_reg": 1e-3,
        "norm": 2,
        "mode": "joint",
        "add_source_loss": False,
    }

    config["reverse_kl"] = {
        "alpha_reverse": 0.1,
        "alpha_forward": 0.1,
        "augment_softmax": 0.0,
    }

    return config
