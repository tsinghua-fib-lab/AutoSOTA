import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from einops import rearrange
from omegaconf import DictConfig, OmegaConf

# Avoid slow plugin discovery during Trainer init unless explicitly enabled.
os.environ.setdefault("LIGHTNING_DISABLE_PLUGINS", "1")

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "datasets"


DATA_ROOT.mkdir(parents=True, exist_ok=True)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from basicts.data.gpvar import GPVARDataset
from basicts.runners.base_predictor import BasePredictor
from basicts.utils.data_utils import create_residuals_frame
from tsl import logger
from tsl.data import SpatioTemporalDataset, SpatioTemporalDataModule
from tsl.data.datamodule.splitters import CustomSplitter
from tsl.data.datamodule import splitters as dm_splitters
from tsl.data.preprocessing import StandardScaler
import importlib
from tsl.experiment import Experiment, NeptuneLogger
from tsl.metrics import torch_metrics

from basicts import config
import tsl.nn.models as tsl_models
from tsl.utils.casting import torch_to_numpy

from foundation_model.nn.base import STGNNModel  # , MLPModel
from basicts.data.air_quality import AirQuality


def get_model_class(model_str):
    """Return the stage-1 forecasting model class.

    We keep our local baselines (RNN/STGNN) and additionally expose TSL's
    built-in spatiotemporal forecasting models for extensibility.
    """
    raw_name = str(model_str)
    model_str = raw_name.lower()

    # Local baseline models.
    if model_str == 'stgnn':
        return STGNNModel

    # TSL built-in models (TorchSpatiotemporal/tsl).
    tsl_aliases = {
        # canonical names
        'transformer': 'TransformerModel',
        'rnn': 'RNNModel',
        'dcrnn': 'DCRNNModel',
        'graphwavenet': 'GraphWaveNetModel',
        'gwnet': 'GraphWaveNetModel',
        'agcrn': 'AGCRNModel',
        'evolvegcn': 'EvolveGCNModel',
        'stid': 'STIDModel',
        'tcn': 'TCNModel',
        'var': 'VARModel',
    }
    cls_name = tsl_aliases.get(model_str)
    if cls_name is not None and hasattr(tsl_models, cls_name):
        return getattr(tsl_models, cls_name)

    # Allow passing the exact class name (case-insensitive), e.g., "GraphWaveNetModel".
    for name in dir(tsl_models):
        if name.lower() == model_str and name.endswith("Model"):
            return getattr(tsl_models, name)

    raise NotImplementedError(
        f'Model "{model_str}" not available. '
        f'Use one of: rnn, stgnn, transformer, dcrnn, gwnet, agcrn, evolvegcn, stid, tcn, var.'
    )


def _get_dataset_class(name: str):
    ds = importlib.import_module("tsl.datasets")
    candidates = {
        "la": ["MetrLA"],
        "pems03": ["PeMS03", "Pems03", "PEMS03"],
        "pems04": ["PeMS04", "Pems04", "PEMS04"],
        "pems07": ["PeMS07", "Pems07", "PEMS07"],
        "pems08": ["PeMS08", "Pems08", "PEMS08"],
        "pems_bay": ["PemsBay", "PeMSBay", "PEMSBay"],
        "large_st": ["LargeST"],
    }
    for cls_name in candidates.get(name, []):
        if hasattr(ds, cls_name):
            return getattr(ds, cls_name)
    raise ImportError(f"Dataset class for '{name}' not found in tsl.datasets.")


def _make_dataset(ds_cls, *args, **kwargs):
    """Instantiate a TSL dataset class under DATA_ROOT/<DatasetClassName>/...

    Some TSL datasets interpret the optional `root` argument as the *dataset directory itself*
    (i.e., they don't add an extra wrapping <DatasetName> folder). To guarantee the wrapping
    folder exists, we override `tsl.config.data_dir` and instantiate the dataset WITHOUT `root`.
    """
    try:
        from tsl import config as tsl_config
        tsl_config.data_dir = str(DATA_ROOT)
    except Exception:
        pass
    return ds_cls(*args, **kwargs)


def get_dataset(dataset_cfg):
    name = dataset_cfg.name
    if name == 'la':
        dataset = _make_dataset(_get_dataset_class("la"))
    elif name in {'pems03', 'pems04', 'pems07', 'pems08', 'pems_bay'}:
        dataset = _make_dataset(_get_dataset_class(name))
    elif name == 'large_st':
        hparams = dict(dataset_cfg.get('hparams', {}))
        root = hparams.pop('root', None) or str(DATA_ROOT / "large_st")
        dataset = _make_dataset(_get_dataset_class(name), root=root, **hparams)
    elif name == 'air':
        dataset = _make_dataset(AirQuality)
    elif name == 'gpvar':
        # GPVAR is generated locally; keep original behavior.
        dataset = GPVARDataset(**dataset_cfg.hparams, p_max=0)
    else:
        raise ValueError(f"Dataset {name} not available.")
    return dataset


def run_experiment(cfg: DictConfig):
    ########################################
    # data module                          #
    ########################################
    dataset = get_dataset(cfg.dataset)

    covariates = dict()
    if cfg.get('add_exogenous'):
        assert cfg.dataset.name not in {'gpvar'}
        # encode time of the day and use it as exogenous variable
        day_sin_cos = dataset.datetime_encoded('day').values
        weekdays = dataset.datetime_onehot('weekday').values
        covariates.update(u=np.concatenate([day_sin_cos, weekdays], axis=-1))

    if cfg.dataset.name in {'gpvar', 'toy', 'mso'}:
        ds_index = pd.Index(dataset.index)
        index_type = 'scalar'
    else:
        ds_index = dataset.index
        index_type = 'datetime'

    torch_dataset = SpatioTemporalDataset(index=ds_index,
                                          target=dataset.dataframe(),
                                          mask=dataset.mask,
                                          covariates=covariates,
                                          horizon=cfg.horizon,
                                          window=cfg.window,
                                          stride=cfg.stride,
                                          delay=cfg.get('delay', 0))

    if cfg.apply_scaler is False:
        transform = {}
    else:
        scale_axis = (0,) if cfg.get('scale_axis') == 'node' else (0, 1)
        transform = {
            'target': StandardScaler(axis=scale_axis)
        }

    if cfg.dataset.splitting.method == 'random':
        splitter = CustomSplitter(
            val_split_fn=dm_splitters.random,
            test_split_fn=dm_splitters.random,
            val_kwargs={'length': cfg.dataset.splitting.val_len},
            test_kwargs={'length': cfg.dataset.splitting.test_len},
            mask_test_indices_in_val=True,
        )
    else:
        splitter = dataset.get_splitter(**cfg.dataset.splitting)

    dm = SpatioTemporalDataModule(
        dataset=torch_dataset,
        scalers=transform,
        splitter=splitter,
        batch_size=cfg.batch_size,
        workers=cfg.workers
    )
    dm.setup()

    ########################################
    # training                             #
    ########################################

    # get adjacency matrix
    adj = dataset.get_connectivity(**cfg.dataset.connectivity,
                                   train_slice=dm.train_slice)
    dm.torch_dataset.set_connectivity(adj)

    ########################################
    # Create model                         #
    ########################################

    model_cls = get_model_class(cfg.model.name)

    d_exog = torch_dataset.input_map.u.shape[-1] if 'u' in torch_dataset else 0
    model_kwargs = dict(n_nodes=torch_dataset.n_nodes,
                        input_size=torch_dataset.n_channels,
                        exog_size=d_exog,
                        output_size=torch_dataset.n_channels,
                        weighted_graph=torch_dataset.edge_weight is not None,
                        window=torch_dataset.window,
                        horizon=torch_dataset.horizon)

    model_cls.filter_model_args_(model_kwargs)
    model_kwargs.update(cfg.model.hparams)

    ########################################
    # predictor                            #
    ########################################

    loss_fn = torch_metrics.MaskedMAE()

    log_metrics = {'mae': torch_metrics.MaskedMAE(),
                   'mse': torch_metrics.MaskedMSE(),
                   'mre': torch_metrics.MaskedMRE()}

    if cfg.dataset.name in ['la', 'bay']:
        multistep_metrics = {
            'mape': torch_metrics.MaskedMAPE(),
            'mae@15': torch_metrics.MaskedMAE(at=2),
            'mae@30': torch_metrics.MaskedMAE(at=5),
            'mae@60': torch_metrics.MaskedMAE(at=11),
        }
        log_metrics.update(multistep_metrics)

    # setup predictor
    predictor = BasePredictor(
        model_class=model_cls,
        model_kwargs=model_kwargs,
        optim_class=getattr(torch.optim, cfg.optimizer.name),
        optim_kwargs=dict(cfg.optimizer.hparams),
        loss_fn=loss_fn,
        metrics=log_metrics,
        scale_target=cfg.scale_target,
    )

    ########################################
    # logging options                      #
    ########################################

    run_args = exp.get_config_dict()
    run_args['model']['trainable_parameters'] = predictor.trainable_parameters

    exp_logger = TensorBoardLogger(save_dir=cfg.run.dir, name=cfg.run.name)

    ########################################
    # training                             #
    ########################################

    early_stop_callback = EarlyStopping(
        monitor='val_mae',
        patience=cfg.patience,
        mode='min'
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.run.dir,
        save_top_k=1,
        monitor='val_mae',
        mode='min',
    )

    val_batches = .25

    trainer = Trainer(max_epochs=cfg.epochs,
                      limit_train_batches=cfg.train_batches,
                      limit_val_batches=val_batches,
                      default_root_dir=cfg.run.dir,
                      logger=exp_logger,
                      accelerator='gpu' if torch.cuda.is_available() else 'cpu',
                      devices=1,
                      gradient_clip_val=cfg.grad_clip_val,
                      callbacks=[early_stop_callback, checkpoint_callback])

    load_model_path = cfg.get('load_model_path')
    if load_model_path is not None:
        predictor.load_model(load_model_path)
    else:
        trainer.fit(predictor, train_dataloaders=dm.train_dataloader(), val_dataloaders=dm.val_dataloader())

    predictor.freeze()

    ########################################
    # compute residuals                    #
    ########################################

    random_split = cfg.dataset.splitting.method == 'random'
    if random_split:
        full_loader = dm.get_dataloader(None, shuffle=False)
        output = trainer.predict(predictor, dataloaders=full_loader)
        output = predictor.collate_prediction_outputs(output)
        output = torch_to_numpy(output)
        y_hat, y_true, mask = (output['y_hat'], output['y'], output.get('mask', None))
        residuals = (y_true - y_hat).squeeze(-1)
        all_indices = np.arange(torch_dataset.n_samples)
        index_for_lagged = dm.torch_dataset.data_timestamps(all_indices)['horizon']
        target_index = index_for_lagged[:, 0]
    else:
        output = trainer.predict(predictor, dataloaders=[dm.val_dataloader(),
                                                         dm.test_dataloader()])  # has size [[len_val], [len_test]]
        output = predictor.collate_prediction_outputs(output)  # has size [len_val + len_test]
        output = torch_to_numpy(output)
        y_hat, y_true, mask = (output['y_hat'], output['y'], output.get('mask', None))
        residuals = (y_true - y_hat).squeeze(-1)
        calib_indices = dm.valset.indices
        test_indices = dm.testset.indices
        val_index = dm.torch_dataset.data_timestamps(calib_indices)['horizon']
        test_index = dm.torch_dataset.data_timestamps(test_indices)['horizon']
        index_for_lagged = np.concatenate([val_index, test_index], axis=0)
        target_index = index_for_lagged[:, 0]

    calib_indices = dm.valset.indices
    test_indices = dm.testset.indices

    # Remove residuals at the beginning and at the end of the time series that have less than window and horizon time steps respectively
    # The second dimension in the dataframe is nodes x horizon
    # Input: [samples, nodes, horizon]
    # Output: [filtered_samples, nodes x horizon]
    col_idx = [(c[0], f'{c[1]}_{i}') for c in dataset._columns_multiindex() for i in range(cfg.horizon)]

    lagged_residuals = create_residuals_frame(residuals,
                                              index_for_lagged,
                                              channels_index=dataset._columns_multiindex(),
                                              horizon=cfg.horizon,
                                              idx_type=index_type)

    # create a dataframe with the residuals arranged in shape [samples, nodes x horizon]
    target_df = pd.DataFrame(data=rearrange(residuals, "t h n ... -> t (n ... h)"),
                             index=target_index,
                             columns=pd.MultiIndex.from_tuples(col_idx))
    if mask is not None:
        mask_df = pd.DataFrame(data=rearrange(mask, "t h n ... -> t (n ... h)"),
                               index=target_index,
                               columns=pd.MultiIndex.from_tuples(col_idx))
    else:
        mask_df = None

    # filter calib and test indices
    valid_input_indices = torch_dataset.index.get_indexer(lagged_residuals.index)
    valid_target_indices = torch_dataset.index.get_indexer(target_index)

    ########################################
    # save residuals for CP                #
    ########################################
    if cfg.save_outputs:
        lagged_residuals.to_hdf(os.path.join(cfg.run.dir,
                                             'residuals.h5'),
                                key='input')
        target_df.to_hdf(os.path.join(cfg.run.dir,
                                             'residuals.h5'),
                                key='target')
        if mask_df is not None:
            mask_df.to_hdf(os.path.join(cfg.run.dir,
                                             'residuals.h5'),
                                key='target_mask')
        np.savez(os.path.join(cfg.run.dir,'indices.npz'),
                 calib_indices=calib_indices,
                 test_indices=test_indices,
                 valid_input_indices=valid_input_indices,
                 valid_target_indices=valid_target_indices)

    return 'done'


if __name__ == '__main__':
    exp = Experiment(run_fn=run_experiment,
                     config_path=str(ROOT / "foundation_model" / "training"),
                     config_name="default")
    res = exp.run()
    logger.info(res)
