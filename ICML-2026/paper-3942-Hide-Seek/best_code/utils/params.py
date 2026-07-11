CONSTANT_PARAMS = {'hide_and_seek': {'epochs': 500,
                            'batch_size': None},
            'invase': {'epochs': 10_000,
                    'batch_size': 1_000},
            'realx': {'epochs': 500,
                    'batch_size': 1_000},
        'l2x': {'epochs': None,
                    'batch_size': None}, #l2x uses 1 epoch and batch size 1000, but that is hard-coded in l2x_for_testing.py
        'lime': {'epochs': 500,
                    'batch_size': None},
        'shap_xgboost': {'epochs': None,
                    'batch_size': None}
}