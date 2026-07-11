from wmcal.calibrators.grid_boost import GridBoostCalibratorConfig
from wmcal.data.datasets.synthetic import TopKSyntheticDatasetConfig
from wmcal.experiments import BaseExperimentConfig
from wmcal.predictors import SimpleNetConfig

WORKERS = 1

experiments = [
    BaseExperimentConfig(
        seed=seed,
        predictor_config=SimpleNetConfig(
            input_dim=10,
            output_dim=4,
            epochs=1000,
            lr=0.01,
        ),
        dataset_config=TopKSyntheticDatasetConfig(
            test_size=4096,
            predictor_size=10000,
            input_dim=10,
            poly_degree=2,
            output_dim=4,
            spread=spread,
            top_k=1,
        ),
        calibrator_config=GridBoostCalibratorConfig(
            output_dim=4,
            eps=0.1,
            grid_resolution=0.25,
            grid_iter_size=256,
            grid_size=1024,
            batch_size=1024,
            max_iter=2048,
            early_stop=False,
            check2_prob=1.0,
            eps_start=0.5,
        ),
    )
    for seed in [42, 43, 44, 45, 46]
    for spread in [2, 4]
]
