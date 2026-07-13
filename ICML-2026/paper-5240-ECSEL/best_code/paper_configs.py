"""
Published per-dataset hyperparameter configurations for the ECSEL classification benchmark.
"""

PAPER_CONFIGS = {
    "iris": {
        "K": 1,
        "l1_strength": 0.00019489008462344248,
        "batch_size": 64,
        "lr": 0.0022093834415066287,
        "num_epochs": 953,
        "patience": 50,
        "use_sigmoid": False,
        "random_state": 42,
        "verbose": False,
    },
    "hearts": {
        "K": 2,
        "l1_strength": 0.005661697479095379,
        "batch_size": 128,
        "lr": 0.00034547828892679517,
        "num_epochs": 200,
        "patience": 50,
        "use_sigmoid": True,
        "random_state": 42,
        "verbose": False,
    },
    
    "seeds": {
        "K": 1,
        "l1_strength": 0.002106397978144515,
        "batch_size": 32,
        "lr": 0.003884102911126366,
        "num_epochs": 200,
        "patience": 50,
        "use_sigmoid": False,
        "random_state": 42,
        "verbose": False,
    },
    "ilpd": {
        "K": 2,
        "l1_strength": 0.006536264082965748,
        "batch_size": 256,
        "lr": 0.009423326114108493,
        "num_epochs": 200,
        "patience": 50,
        "use_sigmoid": False,
        "random_state": 42,
        "verbose": False,
    },
    
    "transfusion": {
        "K": 2,
        "l1_strength": 0.0011913656840629686,
        "batch_size": 64,
        "lr": 0.0011045892849928793,
        "num_epochs": 907,
        "patience": 20,
        "use_sigmoid": False,
        "random_state": 42,
        "verbose": False,
    },
    
    "loan": {
        "K": 3,
        "l1_strength": 6.00853605176827e-05,
        "batch_size": 256,
        "lr": 0.00021064934599103905,
        "num_epochs": 1194,
        "patience": 500,
        "use_sigmoid": True,
        "sigmoid_threshold": 0.6,
        "random_state": 42,
        "verbose": False,
    },

}