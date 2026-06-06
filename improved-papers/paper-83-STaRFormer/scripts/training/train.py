import os
import os.path as osp
import hydra
import wandb
import yaml
import torch
from omegaconf import DictConfig, OmegaConf
from datetime import datetime
from typing import Union, Dict

from lightning import seed_everything, Trainer, LightningDataModule, LightningDataModule
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.strategies import DDPStrategy, FSDPStrategy
from lightning.pytorch.profilers import PyTorchProfiler
from lightning.pytorch.utilities import rank_zero_only
from collections.abc import MutableMapping


from initializer import Initializer

def imports() -> None:
    from src.runtime import (
        LossLoggingCallback, GeolifeMetricLoggerCallback, 
        PAMMetricLoggerCallback, UEAMetricLoggerCallback, P12P19MetricLoggerCallback,
        TSRMetricLoggerCallback, AnomalyMetricLoggerCallback,
        ExponentialMovingAverage
    )
    from src.utils import (
        recusively_convert_dict_for_tensorboard, update_nested_dict, flatten_dict, 
        DatasetOptions, ModelOptions, TaskOptions, ClassificationMethodOptions, TrainingMethodOptions
    )
    from src.nn import TransformerEncoderLayer
    for key, value in locals().items():
        globals()[key] = value
try:
    imports()
except:
    import sys
    file_path = osp.abspath(__file__)
    dir_path = "/".join(file_path.split("/")[:-3])
    sys.path.append(dir_path)
    imports()

DIR_PATH = "/".join(osp.abspath(__file__).split("/")[:-3])

@hydra.main(version_base=None, config_path=f"{DIR_PATH}/configs", config_name="config")
def main(config: DictConfig) -> None:
    wandb.require("core")
    
    # seed everything for reproducability
    seed_everything(seed=config.seed, workers=True)
    
    # training script
    if config.get('run_url', 0) != 0:
        print(f'\nLoading configs from {config.run_url}\n')
        #print(config)
        change_config = config.change_config if config.get('change_config', 0) != 0 else None
        config = load_run_config_from_wandb(run_url=config.run_url, change_config=change_config)

    initializer = Initializer(config)

    # setup experiment logger
    dataset = config.dataset.lower()
    #print(config, '\n', change_config)
    try:
        training_method = config.datamodule.training_method
    except: 
        training_method = TrainingMethodOptions.centralized

    exp_logger, instance = initializer.init_exp_logger(
        **{'dir_path': DIR_PATH, 'dataset': dataset, 'training_method': training_method}
    )
    #print(exp_logger, instance)
    #print(exp_logger.experiment.sweep_id)
    config = update_sweep_config(config=config, exp_logger=exp_logger)

    print(f'Configs:\n{OmegaConf.to_yaml(config)}')
    
    initializer.config = config
    # setup data 
    datamodule, _, config = initializer.init_datamodule()
    # setup model and lightning model
    if config.task == 'regression':
        max_seq_len = 144
    else:
        max_seq_len = datamodule.dataset._max_seq_len + 1 if config.model.sequence_model.name in ModelOptions.all and config.model.output_head.task == TaskOptions.classification and \
            config.model.output_head.cls_method == ClassificationMethodOptions.cls_token else datamodule.dataset._max_seq_len
    print(max_seq_len)
    dataset_obj = None

    lightning_model = initializer.init_model(max_seq_len=max_seq_len, dataset=dataset_obj)
    
    config['run'] = instance
    
    # setup trainer
    checkpoint_callback = ModelCheckpoint(
        dirpath=osp.join(DIR_PATH, 'model_ckpts', dataset, 'centralized', 'harbor', instance),#f'version_{exp_logger.version}'), 
        monitor=config.callbacks.model_ckpt.monitor, 
        verbose=config.callbacks.model_ckpt.verbose, 
        save_last=config.callbacks.model_ckpt.save_last,
        mode=config.callbacks.model_ckpt.mode,
    )
    early_stopping_callback = EarlyStopping(
        monitor=config.callbacks.early_stop.monitor,
        mode=config.callbacks.early_stop.mode,
        patience=config.callbacks.early_stop.patience,
        verbose=config.callbacks.early_stop.verbose
    )
    
    callbacks = [LossLoggingCallback(config=config), early_stopping_callback]
    mckpt_callback_list = []
    mckpt_callback_list.extend(DatasetOptions.tsr)
    mckpt_callback_list.extend([DatasetOptions.p12, DatasetOptions.p19])
    mckpt_callback_list.extend(DatasetOptions.anomaly)
    if config.dataset.lower() not in mckpt_callback_list and not config.logger.sweep:
        callbacks.append(checkpoint_callback)
    
    if config.dataset.lower() == DatasetOptions.dkt:
        raise NotImplementedError
    elif config.dataset.lower() == DatasetOptions.geolife:
        callbacks.append(GeolifeMetricLoggerCallback(config=config, log_cm_train=False, log_cm_val=True))
    elif config.dataset.lower() == DatasetOptions.pam:
        callbacks.append(PAMMetricLoggerCallback(config=config, log_cm_train=False, log_cm_val=True))
    elif config.dataset.lower() in DatasetOptions.uea:
        callbacks.append(UEAMetricLoggerCallback(config=config, log_cm_train=False, log_cm_val=True))
        if config.callbacks.ema.decay:
            callbacks.append(ExponentialMovingAverage(decay=config.callbacks.ema.decay))

    elif config.dataset.lower() in [DatasetOptions.p12, DatasetOptions.p19]:
        callbacks.append(P12P19MetricLoggerCallback(config=config, log_cm_train=False, log_cm_val=True))
    elif config.dataset.lower() in DatasetOptions.tsr:
        callbacks.append(TSRMetricLoggerCallback(config=config, log_cm_train=False, log_cm_val=True))
    elif config.dataset.lower() in DatasetOptions.anomaly:
        callbacks.append(AnomalyMetricLoggerCallback(config=config, log_cm_train=False, log_cm_val=True))

    if config.callbacks.lr_scheduler.apply:
        callbacks.append(LearningRateMonitor(logging_interval='step'))
    

    if dataset.lower() in [DatasetOptions.uea]: 
        log_every_n_steps = 1
    else:
        log_every_n_steps = 50

    if dataset.lower() in [DatasetOptions.pam, DatasetOptions.geolife, DatasetOptions.p12, DatasetOptions.p19, DatasetOptions.kpi]:
        log_every_n_steps = config.training.log_every_n_steps

    if config.training.get('multi_gpu_training', None) is not None: 
        if config.training.multi_gpu_training: 

            if config.system.accelerator != "gpu":
                raise RuntimeError(f'{config.system.accelerator} can only be "gpu".')
            if config.system.devices is None:
                raise RuntimeError(f'{config.system.devices} cannot be None!')
            if config.system.devices is not None:
                if config.system.devices < 1:
                    raise RuntimeError(f'{config.system.devices} have to be larger than 1!')
            if dataset.lower() in ['eigenworms', 'ethanolconcentration']:
                lightning_model.cli_logger.info(f'Training multi-gpu using FSDPStrategy on {config.system.devices} devices')
                policy = {TransformerEncoderLayer}
                trainer = Trainer(
                    max_epochs=config.training.epochs,
                    accelerator=config.system.accelerator,
                    logger=exp_logger,
                    fast_dev_run=config.training.fast_dev,
                    callbacks=callbacks,
                    precision=str(config.training.precision),
                    #strategy=FSDPStrategy(state_dict_policy='full', auto_wrap_policy=policy, cpu_offload=True),
                    strategy=FSDPStrategy(),
                    devices=config.system.devices,
                    log_every_n_steps=log_every_n_steps,
                    deterministic=True,
                    #profiler=profiler
                )
            else: 
                lightning_model.cli_logger.info(f'Training multi-gpu using DDPStrategy on {config.system.devices} devices')
                trainer = Trainer(
                    max_epochs=config.training.epochs,
                    accelerator=config.system.accelerator, 
                    logger=exp_logger,
                    fast_dev_run=config.training.fast_dev, 
                    callbacks=callbacks,
                    precision=str(config.training.precision),
                    strategy=DDPStrategy(find_unused_parameters=False), # find_unused_parameters=True --> reduces speed
                    devices=config.system.devices,
                    log_every_n_steps=log_every_n_steps,
                    deterministic=True,
                    #profiler=profiler
                )
        else:
            trainer = Trainer(
                max_epochs=config.training.epochs,
                accelerator=config.system.accelerator,
                logger=exp_logger,
                fast_dev_run=config.training.fast_dev,
                callbacks=callbacks,
                precision=str(config.training.precision),
                log_every_n_steps=log_every_n_steps,
                devices=config.system.devices,
                deterministic=True,
            )
    else:
        trainer = Trainer(
                max_epochs=config.training.epochs,
                accelerator=config.system.accelerator,
                logger=exp_logger,
                fast_dev_run=config.training.fast_dev, 
                callbacks=callbacks,
                precision=str(config.training.precision),
                log_every_n_steps=log_every_n_steps,
                devices=config.system.devices,
                deterministic=True,
            )

    # run training
    if config.task == 'regression':
        trainer.fit(model=lightning_model, train_dataloaders=train_dataloader, val_dataloaders=test_dataloader)
    else:
        trainer.fit(model=lightning_model, datamodule=datamodule)

    # testing
    if datamodule.test_dataset is not None:
        test(lightning_model=lightning_model, config=config, dataset=dataset, instance=instance, 
            exp_logger=exp_logger, callbacks=callbacks, datamodule=datamodule, trainer=trainer)
    else:
        datamodule.cli_logger.info(f'No test dataset given, datamodule.test_dataset={datamodule.test_dataset}!')

    print('\n'+'-'*50)
    print(f'Training finished for run: {config["run"]}')
    print('-'*50+'\n')


def load_run_config_from_wandb(run_url: str, change_config: Dict=None) -> DictConfig:
    return get_run_configs(run_url=run_url, change_config=change_config)


@rank_zero_only
def test(
    lightning_model: LightningDataModule, 
    config: DictConfig, 
    dataset: str, 
    instance: str, 
    exp_logger: Union[TensorBoardLogger, WandbLogger], 
    callbacks: list, 
    datamodule: LightningDataModule, 
    trainer: Trainer,
    log_every_n_steps: int=50):
    if config.training.get('devices', None) is not None and config.system.accelerator == "gpu":
        #if config.dataset.lower().lower() in ['eigenworms', 'ethanolconcentration']:
        #    lightning_model.cli_logger.info(f'Testing multi-gpu for FSDPStrategy on {config.system.devices} device')
            #lightning_model.cli_logger.info(f'Testing')
        #    trainer.test(ckpt_path='best', datamodule=datamodule)
        
        if config.system.devices > 1:
            lightning_model.cli_logger.info(f'Testing multi-gpu for DDPStrategy on 1 device')
            test_trainer = Trainer(
                max_epochs=config.training.epochs,
                accelerator=config.system.accelerator,
                logger=exp_logger,
                fast_dev_run=config.training.fast_dev, 
                callbacks=callbacks,
                precision=str(config.training.precision),
                devices=1,
                log_every_n_steps=log_every_n_steps,
                deterministic=True,
            )
            ckpt_dir = osp.join(DIR_PATH, 'model_ckpts', dataset, 'centralized', 'harbor', instance)
            best_ckpt_path = [osp.join(ckpt_dir, p) for p in os.listdir(ckpt_dir) if p.split('.')[0] != 'last'][0]
            if osp.exists(best_ckpt_path):
                test_trainer.test(model=lightning_model, ckpt_path=best_ckpt_path, datamodule=datamodule)
            else:
                raise Exception(f'Path {best_ckpt_path} does not exist.')
        else:
            lightning_model.cli_logger.info(f'Testing')
            #print(datamodule._test_dataset)
            trainer.test(ckpt_path='best', datamodule=datamodule)
    else:
        lightning_model.cli_logger.info(f'Testing')
        trainer.test(ckpt_path='best', datamodule=datamodule)


def update_sweep_config(
    config: DictConfig, 
    exp_logger: Union[TensorBoardLogger, WandbLogger]
    ):
    
    if config.logger.name == 'wandb':# and config.training.get('devices', None) is not None:
        if config.logger.sweep:
            if isinstance(exp_logger.experiment.config, wandb.sdk.wandb_config.Config):
                if config.training.get('devices', None) is None:
                    wandb_api = wandb.Api()
                    if config.logger.entity is None:
                        raise RuntimeError('Add wandb entity!')
                    sweep = wandb_api.sweep(f'{config.logger.entity}/{config.logger.project}/{exp_logger.experiment.sweep_id}')
                    if len(sweep.runs) <= 1 and config.logger.run_1_config_path is not None:
                        print('\n######################################')
                        print('Setting - default args for first run!')
                        exp_logger.experiment.config.__dict__["_locked"] = {}  # `delete lock on sweep parameters

                        run_1_config_file_path = osp.join(DIR_PATH, config.logger.run_1_config_path)

                        with open(run_1_config_file_path, 'r') as file:
                            default_run_1 = yaml.safe_load(file)
                        default_run_1 = flatten_dict(d=default_run_1)

                        exp_logger.experiment.config.update(
                            default_run_1, allow_val_change=True
                        )
                
                elif config.training.get('devices', None) is not None:
                    if config.logger.entity is None:
                        raise RuntimeError('Add wandb entity!')
                    #if config.system.devices < 2:
                    if config.system.devices < 2:
                        wandb_api = wandb.Api()
                        sweep = wandb_api.sweep(f'{config.logger.entity}/{config.logger.project}/{exp_logger.experiment.sweep_id}')
                        if len(sweep.runs) <= 1 and config.logger.run_1_config_path is not None:
                            print('\n######################################')
                            print('Setting - default args for first run!')
                            exp_logger.experiment.config.__dict__["_locked"] = {}  # `delete lock on sweep parameters

                            run_1_config_file_path = osp.join(DIR_PATH, config.logger.run_1_config_path)

                            with open(run_1_config_file_path, 'r') as file:
                                default_run_1 = yaml.safe_load(file)
                            default_run_1 = flatten_dict(d=default_run_1)

                            exp_logger.experiment.config.update(
                                default_run_1, allow_val_change=True
                            )

                print('######################################')
                wand_config = exp_logger.experiment.config.as_dict() # get wandb config as dict

            if not isinstance(exp_logger.experiment.config, wandb.sdk.wandb_config.Config): 
                # using DDPStrategy during sweep 
                # wandb_logger.experiment (the wandb run) is only initialized on rank0,
                # but we need every proc to get the wandb sweep config, which happens on .init
                # so we have to call .init on non rank0 procs, but we disable creating a new run
                wandb.init(config={}, mode="disabled")
                #print('wandb.config', wandb.config, type(wandb.config))
                assert isinstance(wandb.config, wandb.sdk.wandb_config.Config)
                #print('wandb.config', wandb.config)
                wand_config = wandb.config.as_dict() # get wandb config as dict
            
            print('wand_config', wand_config)
            if wand_config.get('training.batch_size', None) is not None:
                # update batch_size
                config.training.val_batch_size = wand_config['training.batch_size']
                config.training.test_batch_size = wand_config['training.batch_size']
                if config.model.sequence_model.name == ModelOptions.starformer:
                    config.model.output_head.batch_size = wand_config['training.batch_size']
                    config.datamodule.batch_size = wand_config['training.batch_size']
                    #if config.dataset.lower() not in DatasetOptions.uea:
                    config.datamodule.val_batch_size = wand_config['training.batch_size']
                    config.datamodule.test_batch_size = wand_config['training.batch_size']
            config_dict = OmegaConf.to_container(config, resolve=True) # convert DictConfig to dict
            
            #print(config_dict)
            #config_dict['training']['val_batch_size']=wand_config[training.batch_size]
            #config_dict['training']['test_batch_size']=wand_config[training.batch_size]
            #print('config_dict', config_dict)
            assert isinstance(wand_config, dict)
            update_nested_dict(base_dict=config_dict, updates_dict=wand_config) # update config dict
            config = OmegaConf.create(config_dict) # convert config dict to DictConfig

    return config 


def get_run_configs(run_url: str, change_config: Dict) -> DictConfig:
    try:
        #api = WandBApi()
        wandb_api = wandb.Api()
        run = wandb_api.run(run_url)
#        run = api.run("your-project-name/your-run-id")
        #print(f"Run summary: {run.summary}")
    except Exception as e:
        print(f"Error accessing the run: {e}")
        raise Exception(e)
        #wandb_api = wandb.Api()
        #run = wandb_api.run(run_url)
    configs = run.config

    def get_sweep_configs(configs: dict):
        sweep_model_configs = {}
        for k in configs.keys():
            if k.split('.')[0] == 'model' and k != 'model':
                if k.split('.')[1] in ['sequence_model', 'output_head']:
                    if sweep_model_configs.get(k.split('.')[1], None) is None:
                        sweep_model_configs[k.split('.')[1]] = {}
                    sweep_model_configs[k.split('.')[1]][k.split('.')[2]] = configs[k]
                elif k.split('.')[1] == 'text_model':
                    if sweep_model_configs.get(k.split('.')[1], None) is None:
                        sweep_model_configs[k.split('.')[1]] = {}
                    
                    if k.split('.')[2] == 'aligner':
                        if sweep_model_configs[k.split('.')[1]].get(k.split('.')[2], None) is None:
                            sweep_model_configs[k.split('.')[1]][k.split('.')[2]] = {}
                        
                        sweep_model_configs[k.split('.')[1]][k.split('.')[2]][k.split('.')[3]] = configs[k]
                    else:
                       sweep_model_configs[k.split('.')[1]][k.split('.')[2]] = configs[k] 
                    
                ## original 
                #if k.split('.')[1] == 'output_head':
                #    sweep_model_configs[k.split('.')[1]] = {k.split('.')[2]: configs[k]}
                else:
                    sweep_model_configs[k.split('.')[1]] = configs[k]
        return sweep_model_configs

    def check_sweep_configs_are_applied_model(configs):
        sweep_model_configs = get_sweep_configs(configs)
        for k, v in sweep_model_configs.items():
            if isinstance(v, dict):
                for kk, vv in v.items():
                    if isinstance(vv, dict):
                        for kkk, vvv in vv.items():
                            assert not isinstance(vvv, dict)
                            assert configs['model'][k][kk][kkk] == vvv, f"{k}: Issues with configs {configs['model'][k][kk][kkk]}, {vvv}"
                    else:        
                        assert configs['model'][k][kk] == vv, f"{k}: Issues with configs {configs['model'][k][kk]}, {vv}, {v}"
            else:
                if k == 'output_head':
                    key = 'reduced' if configs['model'][k].get('reduced') else 'reduce'
                    assert configs['model'][k][key] == v[key], f"{k}: Issues with configs {configs['model'][k][key]}, {v[key]}"
                else:
                    assert configs['model'][k] == v, f"{k}: Issues with configs {configs['model'][k]}, {v}"
        

    check_sweep_configs_are_applied_model(configs)

    return_configs = {}
    for key, value in configs.items():
      if len(key.split('.')) <= 1:
            return_configs[key] = value
    
    if change_config is not None:
        # update with change keys
        keys = return_configs.keys()
        for k, v in change_config.items():
            if k == 'task':
                return_configs[k] = v
            else:  
                assert k in keys, f'{k}'
                if isinstance(v, dict) or isinstance(v, DictConfig):
                    for kk, vv in v.items():
                        return_configs[k][kk] = vv
                else:
                    return_configs[k] = v

    return OmegaConf.create(return_configs)


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    main()