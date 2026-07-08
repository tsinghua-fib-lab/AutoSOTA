from pathlib import Path
import numpy as np
import pandas as pd
import torch
from neuralhydrology.datautils.utils import load_basin_file
from neuralhydrology.datasetzoo.camelsus import load_camels_us_attributes, load_camels_us_discharge, load_camels_us_forcings
from neuralhydrology.utils.config import Config as NhConfig

# Dataset type is "hydro-<basin_file>"
HYDRO_TYPE_PREFIX = 'hydro'

TARGET_COL_NAME = 'QObs(mm/d)'


def get_hydro_data(dataset_type, path, **kwargs):
    camels_path = path / 'CAMELS_US'
    assert camels_path.is_dir()
    basin_file = path / dataset_type.replace(f'{HYDRO_TYPE_PREFIX}-', '')
    assert basin_file.is_file()
    basin_list = load_basin_file(basin_file)

    nh_config_path = Path(kwargs['nh_config_path'])
    fc_hydro_cfg = NhConfig(nh_config_path)
    use_static_attributes = kwargs['use_static_attributes']

    train_start_date = fc_hydro_cfg.train_start_date
    if not kwargs.get("calib_as_calib", False):
        eval_start_date = fc_hydro_cfg.test_start_date
    else:
        eval_start_date = fc_hydro_cfg.validation_start_date
    test_end_date = fc_hydro_cfg.test_end_date
    assert train_start_date < eval_start_date

    if use_static_attributes:
        attribute_list = fc_hydro_cfg.static_attributes
        basin_attributes = load_camels_us_attributes(camels_path, basin_list)[attribute_list]
        # Normalize static attributes here, since they need to be handled differently than dynamic variables.
        given_norm_param = kwargs.get('hydro_static_norm_param', None)
        if given_norm_param is None:
            given_norm_param = basin_attributes.mean(axis=0), basin_attributes.std(axis=0)
        attribute_means, attribute_stds = given_norm_param
        basin_attributes = (basin_attributes - attribute_means) / attribute_stds
    else:
        given_norm_param = None
        basin_attributes = None

    datasets = []
    static_attribute_indices = []
    for basin in basin_list:
        
        _id = f"{basin}_{nh_config_path.parent.name}_statics{use_static_attributes}"

        # Load forcings (i.e., dynamic input variables).
        all_features = []
        area = None
        for forcings in fc_hydro_cfg.forcings:
            suffix = f'_{forcings}' if len(fc_hydro_cfg.forcings) > 1 else ''
            features = [feature for feature in fc_hydro_cfg.dynamic_inputs if feature.endswith(suffix)]
            features_without_suffix = [feature.replace(suffix, '') for feature in features]

            forcing_df, forcing_area = load_camels_us_forcings(data_dir=camels_path, basin=basin, forcings=forcings)
            # It shouldn't matter from which forcings we take the area, but just to be sure we always use it from the
            # same forcings. We need the area to area-normalize the discharge.
            if forcings == sorted(fc_hydro_cfg.forcings)[0]:
                area = forcing_area
            
            # Make sure names don't overlap across forcing products.
            forcing_df = forcing_df[features_without_suffix].rename(dict(zip(features_without_suffix, features)),
                                                                    axis=1)
            all_features.append(forcing_df)
        basin_df = pd.concat(all_features, axis=1).loc[train_start_date:test_end_date]

        # Add normalized static attributes to every time step.
        df_columns = sorted(basin_df.columns)  # Sort to make sure the column order is consistent.
        if use_static_attributes:
            for attribute in fc_hydro_cfg.static_attributes:
                basin_df[attribute] = basin_attributes.loc[basin, attribute]
            df_columns += sorted(fc_hydro_cfg.static_attributes)
            static_attribute_indices = [i for i, col in enumerate(df_columns) if col in fc_hydro_cfg.static_attributes]

        # Load discharge (i.e., target values).
        basin_discharge = load_camels_us_discharge(data_dir=camels_path, basin=basin, area=area)
        basin_discharge.loc[basin_discharge < 0] = np.nan  # CAMELS uses -999 to encode NaN.
        basin_df[TARGET_COL_NAME] = basin_discharge.loc[train_start_date:test_end_date]
        
        # We only allow NaNs in the beginning of the timeseries, since we can just drop those. We don't allow NaNs
        # anywhere in the cal/test series.
        num_nans = basin_df.isna().sum().sum()
        if num_nans > 0:
            failed = True
            # It's fine if the records just start later and we lose some training data.
            discharge_nans = basin_df[TARGET_COL_NAME].isna().sum()
            if basin_df.iloc[discharge_nans:].isna().sum().sum() == 0:
                basin_df = basin_df.iloc[discharge_nans:]
                if basin_df.index[0] < eval_start_date:
                    failed = False
            if failed:
                raise ValueError(f'Basin {basin} has NaN values.')


        train_steps = (basin_df.index < eval_start_date).sum()
        if eval_start_date != fc_hydro_cfg.test_start_date:
            calib_steps = (basin_df.index < fc_hydro_cfg.test_start_date).sum() - train_steps
        else:
            calib_steps = None

        Y_full = basin_df[TARGET_COL_NAME]
        X_full = basin_df[df_columns]
        datasets.append((
            torch.from_numpy(X_full.to_numpy()).float(),
            torch.from_numpy(Y_full.to_numpy()).float(),
            _id,
            train_steps,
            calib_steps
        ))
    return datasets, static_attribute_indices, given_norm_param
