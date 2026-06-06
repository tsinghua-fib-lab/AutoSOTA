import os.path as osp
from omegaconf import DictConfig, open_dict

try:
    from src.utils import TrainingMethodOptions, DatasetOptions, ModelOptions
    from src.datamodule import (
        GeoLifeDatamodule,
        PAMDatamodule,
        UEADatamodule,
        P12Datamodule,
        P19Datamodule, 
        AnomalyDatamodule,
    )

except:
    import sys
    file_path = osp.abspath(__file__)
    dir_path = "/".join(file_path.split("/")[:-4])
    sys.path.append(dir_path)
    from src.utils import TrainingMethodOptions, DatasetOptions, ModelOptions
    from src.datamodule import (
        GeoLifeDatamodule,
        PAMDatamodule,
        UEADatamodule,
        P12Datamodule,
        P19Datamodule,
        AnomalyDatamodule,
    )
    

def initialize_datamodule(config: DictConfig):
    if config.dataset.lower() == DatasetOptions.dkt:
        raise NotImplementedError
    elif config.dataset.lower() == DatasetOptions.geolife:
        datamodule, instance, config = geolife(config)
    elif config.dataset.lower() in DatasetOptions.uea:
        datamodule, instance, config = uea(config)
    elif config.dataset.lower() in DatasetOptions.irregulary_sampled:
        datamodule, instance, config = irregulary_sampled(config)
    elif config.dataset.lower() in DatasetOptions.anomaly: # anomaly datasets
        datamodule, instance, config = anomaly(config)
    else:
        raise RuntimeError(f'{config.dataset} is not found!')
    
    return datamodule, instance, config


def geolife(config: DictConfig=None):
    # centralized
    if config.datamodule.training_method == TrainingMethodOptions.centralized:
        dm_params = {
            'batch_size': int(config.datamodule.batch_size),
            'num_workers': int(config.datamodule.num_workers),
            'seed': int(config.seed),
            'training_method': config.datamodule.training_method,
            'train_test_split': config.datamodule.train_test_split,
            'identical_training_class_label_distribution': config.datamodule.identical_training_class_label_distribution,
            'aws_profile': config.datamodule.aws_profile,
            's3_bucket_path': config.datamodule.s3_bucket_path,
            'pad_sequence': config.datamodule.pad_sequence,
            'val_batch_size': config.datamodule.val_batch_size,
            'test_batch_size':config.datamodule.test_batch_size,
            'num_train': config.datamodule.num_train,
            'num_val': config.datamodule.num_val,
            'num_test': config.datamodule.num_test,
            'max_trajectory_length': config.datamodule.max_trajectory_length,
            'synthetic_minority_upsampling': config.datamodule.synthetic_minority_upsampling,
            'noise_level': config.datamodule.noise_level,
        }

        datamodule = GeoLifeDatamodule(**dm_params)
        datamodule.setup()
        instance = datamodule.dataset.dataset_dir_paths['centralized_instance'].split("/")[-1]
        with open_dict(config): # add dataset instance to datamodule config
            config.datamodule.dataset_instance = datamodule.dataset.ds_config['instance']
    
    # federated
    elif config.datamodule.training_method == TrainingMethodOptions.federated:
        raise RuntimeError

    else:
        raise RuntimeError
    
    return datamodule, instance, config


def uea(config: DictConfig=None):
    # centralized
    #if config.datamodule.training_method == TrainingMethodOptions.centralized:
    
    dm_params = {
        'batch_size': int(config.datamodule.batch_size),
        'num_workers': int(config.datamodule.num_workers),
        'seed': int(config.seed),
        'dataset_name': config.datamodule.dataset,
        'max_trajectory_length': config.datamodule.max_trajectory_length,
        #'training_method': config.datamodule.training_method,
        #'train_splits': dict(config.datamodule.train_splits),
        'aws_profile': config.datamodule.aws_profile,
        's3_bucket_path': config.datamodule.s3_bucket_path,
        'use_threads': config.datamodule.use_threads,
        'val_batch_size': config.datamodule.val_batch_size,
        'test_batch_size':config.datamodule.test_batch_size,
        'num_train': config.datamodule.num_train,
        'num_val': config.datamodule.num_val,
        'num_test': config.datamodule.num_test,
        
    }
    if config.dataset in [DatasetOptions.eigenworms, DatasetOptions.ethanolconcentration]:
        dm_params['max_trajectory_length'] = config.datamodule.max_trajectory_length

    datamodule = UEADatamodule(**dm_params)
    datamodule.setup()
    instance = datamodule.dataset.dataset_dir_paths['centralized_instance'].split("/")[-1]
    with open_dict(config): # add dataset instance to datamodule config
        config.datamodule.dataset_instance = datamodule.dataset.ds_config['instance']
    
    # federated
    #if config.datamodule.training_method == TrainingMethodOptions.federated:
    #    raise RuntimeError
    #else:
    #    raise RuntimeError
    
    return datamodule, instance, config

def irregulary_sampled(config: DictConfig=None):
    # centralized
    if config.datamodule.training_method == TrainingMethodOptions.centralized:
        datamodule_objects_map = {
            DatasetOptions.p19: P19Datamodule,
            DatasetOptions.p12: P12Datamodule,
            DatasetOptions.pam: PAMDatamodule,
        }
        dm_params = {
            'batch_size': int(config.datamodule.batch_size),
            'num_workers': int(config.datamodule.num_workers),
            'seed': int(config.seed),
            'training_method': config.datamodule.training_method,
            'train_split_index': config.datamodule.train_split_index,
            'aws_profile': config.datamodule.aws_profile,
            's3_bucket_path': config.datamodule.s3_bucket_path,
            'use_threads': config.datamodule.use_threads,
            'val_batch_size': config.datamodule.val_batch_size,
            'test_batch_size':config.datamodule.test_batch_size,
            'num_train': config.datamodule.num_train,
            'num_val': config.datamodule.num_val,
            'num_test': config.datamodule.num_test,   
            'pad_sequence': config.datamodule.pad_sequence,
        }
        if config.dataset in [DatasetOptions.p12, DatasetOptions.p19]:
            dm_params['preprocessing_method'] = config.datamodule.preprocessing_method
            dm_params['min_seq_length'] = config.datamodule.min_seq_length
            dm_params['percentile_of_features_used'] = config.datamodule.percentile_of_features_used
            dm_params['balance'] = config.datamodule.balance
            dm_params['upsample_percentage'] = config.datamodule.upsample_percentage
            
        
        datamodule = datamodule_objects_map[config.dataset](**dm_params)
        datamodule.setup()
        instance = datamodule.dataset.dataset_dir_paths['centralized_instance'].split("/")[-1]
        with open_dict(config): # add dataset instance to datamodule config
            config.datamodule.dataset_instance = datamodule.dataset.ds_config['instance']
    
    # federated
    elif config.datamodule.training_method == TrainingMethodOptions.federated:
        raise RuntimeError

    else:
        raise RuntimeError
    
    return datamodule, instance, config

def anomaly(config: DictConfig=None):
    # centralized
    if config.datamodule.training_method == TrainingMethodOptions.centralized:
        dm_params = {
            'dataset': config.datamodule.dataset,
            'batch_size': int(config.datamodule.batch_size),
            'num_workers': int(config.datamodule.num_workers),
            'seed': int(config.seed),
            'training_method': config.datamodule.training_method,
            'aws_profile': config.datamodule.aws_profile,
            's3_bucket_path': config.datamodule.s3_bucket_path,
            'use_threads': config.datamodule.use_threads,
            'window_size': config.datamodule.window_size,
            'stride': config.datamodule.stride,
            'pad_sequence': config.datamodule.pad_sequence,
            'val_batch_size': config.datamodule.val_batch_size,
            'test_batch_size':config.datamodule.test_batch_size,
            'num_train': config.datamodule.num_train,
            'num_val': config.datamodule.num_val,
            'num_test': config.datamodule.num_test,
        }
        #if config.dataset in [DatasetOptions.eigenworms, DatasetOptions.ethanolconcentration]:
        #    dm_params['max_trajectory_length'] = config.datamodule.max_trajectory_length

        datamodule = AnomalyDatamodule(**dm_params)
        datamodule.setup()
        instance = datamodule.dataset.dataset_dir_paths['centralized_instance'].split("/")[-1]
        with open_dict(config): # add dataset instance to datamodule config
            config.datamodule.dataset_instance = datamodule.dataset.ds_config['instance']
    
    # federated
    elif config.datamodule.training_method == TrainingMethodOptions.federated:
        raise RuntimeError

    else:
        raise RuntimeError
    
    return datamodule, instance, config