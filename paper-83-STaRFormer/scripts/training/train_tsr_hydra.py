import os
import os.path as osp
import hydra 
import torch
import numpy as np
import yaml

from lightning import seed_everything
from omegaconf import DictConfig, OmegaConf
from datetime import datetime

try:
    from src.utils.runtime.tsr.data_loader import load_from_tsfile_to_dataframe
    from src.utils.runtime.tsr.regression_tools import process_data_tsr, fit_regressor_hydra, calculate_regression_metrics
    from src.utils.runtime.tsr.tools import create_directory
    from src.utils.runtime.tsr.transformer_tools import fit_transformer
    from src.utils import DatasetOptions, ModelOptions, flatten_dict, update_nested_dict
except:
    import sys
    file_path = osp.abspath(__file__)
    dir_path = "/".join(file_path.split("/")[:-3])
    sys.path.append(dir_path)
    
    from src.utils.runtime.tsr.data_loader import load_from_tsfile_to_dataframe
    from src.utils.runtime.tsr.regression_tools import process_data_tsr, fit_regressor_hydra, calculate_regression_metrics
    from src.utils.runtime.tsr.tools import create_directory
    from src.utils.runtime.tsr.transformer_tools import fit_transformer
    from src.utils import DatasetOptions, ModelOptions, flatten_dict, update_nested_dict


DIR_PATH = "/".join(osp.abspath(__file__).split("/")[:-3])
MODULE = "RegressionExperiment"
# default transformer parameters 
transformer_name = "none"  # see transformer_tools.transformers
flatten = False  # if flatten, do not transform per dimension
n_components = 10  # number of principal components
n_basis = 10  # number of basis functions
bspline_order = 4  # bspline order

RAW_FILE_NAME = {
    DatasetOptions.appliancesenergy: "3902637.zip",
    DatasetOptions.australiarainfall: '3902654.zip',
    DatasetOptions.beijingpm10quality: "3902667.zip",
    DatasetOptions.beijingpm25quality: "3902671.zip",
    DatasetOptions.benzeneconcentration: "3902673.zip",
    DatasetOptions.bidmcrr: '4001463.zip', # newer version
    DatasetOptions.bidmchr: '4001456.zip', # newer version
    DatasetOptions.bidmcspo2: '4001464.zip', # newer version
    DatasetOptions.covid3month: "3902690.zip",
    DatasetOptions.floodmodeling1: "3902694.zip",
    DatasetOptions.floodmodeling2: "3902696.zip",
    DatasetOptions.floodmodeling3: "3902698.zip",
    DatasetOptions.householdpowerconsumption1: "3902704.zip",
    DatasetOptions.householdpowerconsumption2: "3902706.zip",
    DatasetOptions.ieeeppg: "3902710.zip",
    DatasetOptions.livefuelmoisturecontent: "4632439.zip",
    DatasetOptions.newsheadlinesentiment: '3902718.zip',
    DatasetOptions.newstitlesentiment: '3902726.zip',
    DatasetOptions.ppgdalia: '3902728.zip',
}

DATASET_URLS = {
    DatasetOptions.appliancesenergy: f"https://zenodo.org/api/records/{RAW_FILE_NAME[DatasetOptions.appliancesenergy].split('.')[0]}/files-archive",
    DatasetOptions.australiarainfall: f"https://zenodo.org/api/records/{RAW_FILE_NAME[DatasetOptions.australiarainfall].split('.')[0]}/files-archive",
    DatasetOptions.beijingpm10quality: f"https://zenodo.org/api/records/{RAW_FILE_NAME[DatasetOptions.beijingpm10quality].split('.')[0]}/files-archive",
    DatasetOptions.beijingpm25quality: f"https://zenodo.org/api/records/{RAW_FILE_NAME[DatasetOptions.beijingpm25quality].split('.')[0]}/files-archive",
    DatasetOptions.benzeneconcentration: f"https://zenodo.org/api/records/{RAW_FILE_NAME[DatasetOptions.benzeneconcentration].split('.')[0]}/files-archive",
    DatasetOptions.bidmcrr: f"https://zenodo.org/api/records/{RAW_FILE_NAME[DatasetOptions.bidmcrr].split('.')[0]}/files-archive",
    DatasetOptions.bidmchr: f"https://zenodo.org/api/records/{RAW_FILE_NAME[DatasetOptions.bidmchr].split('.')[0]}/files-archive",
    DatasetOptions.bidmcspo2: f"https://zenodo.org/api/records/{RAW_FILE_NAME[DatasetOptions.bidmcspo2].split('.')[0]}/files-archive",
    DatasetOptions.covid3month: f"https://zenodo.org/api/records/{RAW_FILE_NAME[DatasetOptions.covid3month].split('.')[0]}/files-archive",
    DatasetOptions.floodmodeling1: f"https://zenodo.org/api/records/{RAW_FILE_NAME[DatasetOptions.floodmodeling1].split('.')[0]}/files-archive",
    DatasetOptions.floodmodeling2: f"https://zenodo.org/api/records/{RAW_FILE_NAME[DatasetOptions.floodmodeling2].split('.')[0]}/files-archive",
    DatasetOptions.floodmodeling3: f"https://zenodo.org/api/records/{RAW_FILE_NAME[DatasetOptions.floodmodeling3].split('.')[0]}/files-archive",
    DatasetOptions.householdpowerconsumption1: f"https://zenodo.org/api/records/{RAW_FILE_NAME[DatasetOptions.householdpowerconsumption1].split('.')[0]}/files-archive",
    DatasetOptions.householdpowerconsumption2: f"https://zenodo.org/api/records/{RAW_FILE_NAME[DatasetOptions.householdpowerconsumption2].split('.')[0]}/files-archive",
    DatasetOptions.ieeeppg: f"https://zenodo.org/api/records/{RAW_FILE_NAME[DatasetOptions.ieeeppg].split('.')[0]}/files-archive",
    DatasetOptions.livefuelmoisturecontent: f"https://zenodo.org/api/records/{RAW_FILE_NAME[DatasetOptions.livefuelmoisturecontent].split('.')[0]}/files-archive",
    DatasetOptions.newsheadlinesentiment: f"https://zenodo.org/api/records/{RAW_FILE_NAME[DatasetOptions.newsheadlinesentiment].split('.')[0]}/files-archive",
    DatasetOptions.newstitlesentiment: f"https://zenodo.org/api/records/{RAW_FILE_NAME[DatasetOptions.newstitlesentiment].split('.')[0]}/files-archive",
    DatasetOptions.ppgdalia: f"https://zenodo.org/api/records/{RAW_FILE_NAME[DatasetOptions.ppgdalia].split('.')[0]}/files-archive",
}


from pathlib import Path
from src.utils.datasets.tsr import download

def _download(dataset: str, data_raw_path: str | Path):
    download(
        DATASET_URLS[dataset.lower()], # lower
        dataset_name=dataset, # in Caps
        dataset_file_path=data_raw_path + '/' + RAW_FILE_NAME[dataset.lower()],
        dataset_path=data_raw_path,
        uncompress=True
    )

def initialize_data(module, data_path, problem):
    # loading the data. X_train and X_test are dataframe of N x n_dim
    print("[{}] Loading data".format(module))
    if problem == DatasetOptions.appliancesenergy:
       problem = 'AppliancesEnergy'
    elif problem == DatasetOptions.australiarainfall:
        problem = 'AustraliaRainfall'
    elif problem == DatasetOptions.benzeneconcentration:   
       problem = 'BenzeneConcentration'
    elif problem == DatasetOptions.beijingpm10quality:
       problem = 'BeijingPM10Quality'
    elif problem == DatasetOptions.beijingpm25quality:
       problem = 'BeijingPM25Quality'
    elif problem == DatasetOptions.bidmcrr:
        problem = 'BIDMCRR'
    elif problem == DatasetOptions.bidmchr:
        problem = 'BIDMCHR'
    elif problem == DatasetOptions.bidmcspo2:
        problem = 'BIDMCSpO2'
    elif problem == DatasetOptions.covid3month:
       problem = 'Covid3Month'
    elif problem == DatasetOptions.floodmodeling1:
       problem = 'FloodModeling1'
    elif problem == DatasetOptions.floodmodeling2:
       problem = 'FloodModeling2'
    elif problem == DatasetOptions.floodmodeling3:
       problem = 'FloodModeling3'
    elif problem == DatasetOptions.householdpowerconsumption1:
       problem = 'HouseholdPowerConsumption1'
    elif problem == DatasetOptions.householdpowerconsumption2:
       problem = 'HouseholdPowerConsumption2'
    elif problem == DatasetOptions.ieeeppg:
       problem = 'IEEEPPG'
    elif problem == DatasetOptions.livefuelmoisturecontent:
       problem = 'LiveFuelMoistureContent'
    elif problem == DatasetOptions.newsheadlinesentiment:
       problem = 'NewsHeadlineSentiment'
    elif problem == DatasetOptions.newstitlesentiment:
       problem = 'NewsTitleSentiment'
    elif problem == DatasetOptions.ppgdalia:
       problem = 'PPGDalia'
    else:
        raise ValueError(f'Dataset {problem} not known!')
    # set data folder, train & test
    data_folder = osp.join(data_path, problem, "raw")
    if problem.lower() in [DatasetOptions.bidmcrr, DatasetOptions.bidmchr, DatasetOptions.bidmcspo2]:
        if problem.lower() == DatasetOptions.bidmcrr:
            train_file = osp.join(data_folder, "BIDMC32RR_TRAIN.ts")
            test_file = osp.join(data_folder, "BIDMC32RR_TEST.ts")
        elif problem.lower() == DatasetOptions.bidmchr:
            train_file = osp.join(data_folder, "BIDMC32HR_TRAIN.ts")
            test_file = osp.join(data_folder, "BIDMC32HR_TEST.ts")
        elif problem.lower() == DatasetOptions.bidmcspo2:
            train_file = osp.join(data_folder, "BIDMC32SpO2_TRAIN.ts")
            test_file = osp.join(data_folder, "BIDMC32SpO2_TEST.ts")
        else:
            raise RuntimeError
    else:
        train_file = osp.join(data_folder, problem+"_TRAIN.ts")
        test_file = osp.join(data_folder, problem+"_TEST.ts")

    if not osp.exists(data_folder):
        os.makedirs(data_folder)
        # download 
        _download(dataset=problem, data_raw_path=data_folder)
        
    X_train, y_train = load_from_tsfile_to_dataframe(train_file)
    X_test, y_test = load_from_tsfile_to_dataframe(test_file)
    print("[{}] X_train: {}".format(module, X_train.shape))
    print("[{}] X_test: {}".format(module, X_test.shape))
    return X_train, y_train, X_test, y_test

def find_minimum_length(module, X_train, X_test):
    # in case there are different lengths in the dataset, we need to consider that.
    # assume that all the dimensions are the same length
    print("[{}] Finding minimum length".format(module))
    min_len = np.inf
    for i in range(len(X_train)):
        x = X_train.iloc[i, :]
        all_len = [len(y) for y in x]
        min_len = min(min(all_len), min_len)
    for i in range(len(X_test)):
        x = X_test.iloc[i, :]
        all_len = [len(y) for y in x]
        min_len = min(min(all_len), min_len)
    print("[{}] Minimum length: {}".format(module, min_len))
    return min_len


def process_data(module, X_train, X_test, norm, min_len):
    # process the data into numpy array with (n_examples, n_timestep, n_dim)
    print("[{}] Reshaping data".format(module))
    x_train = process_data_tsr(X_train, normalise=norm, min_len=min_len)
    x_test = process_data_tsr(X_test, normalise=norm, min_len=min_len)

    # transform the data if needed
    if transformer_name != "none":
        if transformer_name == "pca":
            kwargs = {"n_components": n_components}
        elif transformer_name == "fpca":
            kwargs = {"n_components": n_components}
        elif transformer_name == "fpca_bspline":
            kwargs = {"n_components": n_components,
                      "n_basis": n_basis,
                      "order": bspline_order,
                      "smooth": "bspline"}
        else:
            kwargs = {}
        x_train, transformer = fit_transformer(transformer_name, x_train, flatten=flatten, **kwargs)
        x_test = transformer.transform(x_test)

    print("[{}] X_train: {}".format(module, x_train.shape))
    print("[{}] X_test: {}".format(module, x_test.shape))
    return x_train, x_test

@hydra.main(version_base=None, config_path=f"{DIR_PATH}/configs", config_name="config")
def main(config: DictConfig) -> None:
    if not config.logger.sweep:
        # training
        train(config=config)
    else:
        # hyparameter sweeps
        sweep(config=config)

def train(config: DictConfig, save_output_locally: bool=True):
    if config.get('run_url', 0) != 0:
        print(f'\nLoading configs from {config.run_url}\n')
        change_config = config.change_config if config.get('change_config', 0) != 0 else None
        config = load_run_config_from_wandb(run_url=config.run_url, change_config=change_config)
    print(f'Configs:\n{OmegaConf.to_yaml(config)}')

    data_path = osp.join(DIR_PATH, "data")
    problem = config.dataset  # see data_loader.regression_datasets
    regressor_name = config.model.sequence_model.name  # see regressor_tools.all_models
    seed = config.seed
    norm = config.norm  # none, standard, minmax
    pytorch = config.pytorch
    
    seed_everything(seed=config.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # create output directory
    output_directory = "output/regression/"
    if norm != "none":
        output_directory = "output/regression_{}/".format(norm)
    output_directory = output_directory + regressor_name + '/' + problem + '/seed_' + str(seed) + '/'
    create_directory(output_directory)

    print("=======================================================================")
    print("[{}] Starting Holdout Experiments".format(MODULE))
    print("=======================================================================")
    print("[{}] Data path: {}".format(MODULE, data_path))
    print("[{}] Output Dir: {}".format(MODULE, output_directory))
    print("[{}] Iteration (Seed): {}".format(MODULE, seed))
    print("[{}] Problem: {}".format(MODULE, problem))
    print("[{}] Regressor: {}".format(MODULE, regressor_name))
    print("[{}] Transformer: {}".format(MODULE, transformer_name))
    print("[{}] Normalisation: {}".format(MODULE, norm))
    print("[{}] PyTorch: {}".format(MODULE, pytorch))

    X_train, y_train, X_test, y_test = initialize_data(MODULE, data_path, problem)

    if config.dataset.lower() in [DatasetOptions.australiarainfall]:
        # filter outliers 
        from tqdm.auto import tqdm
        import pandas as pd
        assert isinstance(y_train, np.ndarray)
        assert isinstance(X_train, pd.DataFrame), f'{type(X_train)} is not a {pd.DataFrame}'
        assert isinstance(y_test, np.ndarray)
        assert isinstance(X_test, pd.DataFrame), f'{type(X_test)} is not a {pd.DataFrame}'
        percentile_999 = np.percentile(y_train, 99.9) 
        train_index_filtered = [i
            for i, y in enumerate(tqdm(y_train, desc="Filtering outliers in train set."))
            if y <= percentile_999
        ]
        percentile_999_test = np.percentile(y_test, 99.9) 
        test_index_filtered = [i
            for i, y in enumerate(tqdm(y_test, desc='Filtering outliers in test set.'))
            if y <= percentile_999_test
        ]
        X_train, y_train = X_train.iloc[train_index_filtered], y_train[train_index_filtered]
        X_test, y_test = X_test.iloc[test_index_filtered], y_test[test_index_filtered]

    min_len = find_minimum_length(MODULE, X_train, X_test)
    x_train, x_test = process_data(MODULE, X_train, X_test, norm=norm, min_len=min_len)

    # fit the regressor
    regressor = fit_regressor_hydra(config, output_directory, regressor_name, x_train, y_train, x_test, y_test)

    # start testing
    y_pred = regressor.predict(x_test)
    df_metrics = calculate_regression_metrics(y_test, y_pred)
    if regressor_name.lower() == ModelOptions.starformer:
        import wandb
        wandb.log({
            'test/rmse': df_metrics['rmse'].values,
            'test/mae': df_metrics['mae'].values,
        })


    print(f'[{MODULE}]: Test Results:\n{df_metrics}')
    if save_output_locally:
        # save the outputs
        df_metrics.to_csv(output_directory + 'regression_experiment.csv', index=False)
    else: 
        # delete locally stored runs
        for f in os.listdir(output_directory):
            os.remove(os.path.join(output_directory, f))
        
        os.rmdir(output_directory)

def sweep(config: DictConfig):
    if config.get('run_url', 0) != 0:
        print(f'\nLoading configs from {config.run_url}\n')
        change_config = config.change_config if config.get('change_config', 0) != 0 else None
        config = load_run_config_from_wandb(run_url=config.run_url, change_config=change_config)

    # give required sweep_id
    # currently has to be hard coded
    if config.dataset == DatasetOptions.appliancesenergy:
        raise RuntimeError(f'{config.dataset} not in {DatasetOptions.tsr}')
    elif config.dataset == DatasetOptions.australiarainfall:
        config.sweep_id = 'exbaa0q1' 
    elif config.dataset == DatasetOptions.beijingpm10quality:
        config.sweep_id = '4v820mvw' 
    elif config.dataset == DatasetOptions.beijingpm25quality:
        config.sweep_id = 'b6a6lk2q' 
    elif config.dataset == DatasetOptions.benzeneconcentration:
        config.sweep_id = 'ugfjcg1c' 
    elif config.dataset == DatasetOptions.bidmcrr:
        config.sweep_id = 'tehbiinw' 
    elif config.dataset == DatasetOptions.bidmchr:
        config.sweep_id = 'ufc4n46x'
    elif config.dataset == DatasetOptions.bidmcspo2:
        config.sweep_id = 'lhrtp6qh'
    elif config.dataset == DatasetOptions.floodmodeling1:
        config.sweep_id = 'mwp8k3vj' 
    elif config.dataset == DatasetOptions.floodmodeling2:
        config.sweep_id = 'je3jngb8'
    elif config.dataset == DatasetOptions.floodmodeling3:
        config.sweep_id = 'kdpq0vrp'
    elif config.dataset == DatasetOptions.ieeeppg:
        config.sweep_id = 'aw5ifzfn'
    elif config.dataset == DatasetOptions.householdpowerconsumption1:
        config.sweep_id = 't3573rf1'
    elif config.dataset == DatasetOptions.householdpowerconsumption2:
        config.sweep_id = 'pcp875j1' 
    elif config.dataset == DatasetOptions.livefuelmoisturecontent:
        config.sweep_id = '7k4j82rf'
    elif config.dataset == DatasetOptions.covid3month:
        config.sweep_id = 'abiuid32'
    elif config.dataset == DatasetOptions.newsheadlinesentiment:
        config.sweep_id = '12rgxusn'
    elif config.dataset == DatasetOptions.newstitlesentiment:
        config.sweep_id = 'i73pcywa'
    elif config.dataset == DatasetOptions.ppgdalia:
        config.sweep_id = 'gt842yiu'
    else:
        print(config.dataset, config.dataset in DatasetOptions.tsr, config.dataset == DatasetOptions.householdpowerconsumption2)
        raise RuntimeError(f'{config.dataset} not in {DatasetOptions.tsr}')
    
    print(config.dataset, DatasetOptions.ieeeppg, config.sweep_id)

    # get wandb config
    wandb.init(
        entity=config.logger.entity,
        project=config.logger.project,
        name=datetime.now().strftime("%Y-%m-%d_%H:%M:%S"),
        config=OmegaConf.to_container(config, resolve=True),
    )
    current_sweep_config = wandb.config
    # update wandb config for inital run of hyperparameter sweep
    config = update_sweep_config(config=config, wandb_sweep_config=current_sweep_config)
    # delete lock on sweep parameters
    wandb.config.__dict__["_locked"] = {}  
    wandb.config.update(OmegaConf.to_container(config, resolve=True))

    train(config=config, save_output_locally=False)

import wandb 
from typing import Dict


def load_run_config_from_wandb(run_url: str, change_config: Dict) -> DictConfig:
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

def update_sweep_config(
    config: DictConfig,
    wandb_sweep_config: wandb.sdk.wandb_config.Config
    ):
    if config.logger.name == 'wandb':
        if config.logger.sweep:
            if config.logger.entity is None:
                raise RuntimeError('Add wandb entity!')
            
            wandb_api = wandb.Api()
            print(f'{config.logger.entity}/{config.logger.project}/{config.sweep_id}')
            sweep = wandb_api.sweep(f'{config.logger.entity}/{config.logger.project}/{config.sweep_id}')
            if len(sweep.runs) <= 1 and config.logger.run_1_config_path is not None:
                print('\n######################################')
                print('Setting - default args for first run!')
                wandb_sweep_config.__dict__["_locked"] = {}  # `delete lock on sweep parameters

                run_1_config_file_path = osp.join(DIR_PATH, config.logger.run_1_config_path)

                with open(run_1_config_file_path, 'r') as file:
                    default_run_1 = yaml.safe_load(file)
                default_run_1 = flatten_dict(d=default_run_1)
                wandb_sweep_config.update(
                    default_run_1, allow_val_change=True
                )

            print('######################################')
            wand_config = wandb_sweep_config.as_dict() # get wandb config as dict
            
            print('wand_config', wand_config)
            if wand_config.get('training.batch_size', None) is not None:
                # update batch_size
                config.training.val_batch_size = wand_config['training.batch_size']
                config.training.test_batch_size = wand_config['training.batch_size']
                if config.model.sequence_model.name == ModelOptions.starformer:
                    config.model.output_head.batch_size = wand_config['training.batch_size']
            config_dict = OmegaConf.to_container(config, resolve=True) # convert DictConfig to dict
            assert isinstance(wand_config, dict)
            update_nested_dict(base_dict=config_dict, updates_dict=wand_config) # update config dict
            config = OmegaConf.create(config_dict) # convert config dict to DictConfig

    return config 

if __name__ == '__main__':
    main()
