import numpy as np
import pandas as pd
import argparse
import json
from sklearn.preprocessing import StandardScaler
from scipy.special import expit
# import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import repo_paths  # noqa: F401

from tools import run_feature_selection_model


# Define city centers (latitude, longitude)
POPULATION_CENTRES = {
    'la_metro': (34.05, -118.25),
    'sf_bay': (37.77, -122.42)
}
# RADIUS_KM = 100
RANDOM_STATE = 42

def bool_or_none(v):
    if v is None:
        return None
    v = str(v).strip().lower()
    if v in {'none', 'null'}:
        return None
    if v in {'true', '1', 'yes', 'y', 't'}:
        return True
    if v in {'false', '0', 'no', 'n', 'f'}:
        return False
    raise argparse.ArgumentTypeError('Expected one of true/false/none')


def int_or_none(v):
    if v is None:
        return None
    v = str(v).strip().lower()
    if v in {'none', 'null'}:
        return None
    return int(v)


def parse_args():
    parser = argparse.ArgumentParser(description='Run California housing feature selection experiments')
    parser.add_argument('--model-type', type=str, default='hide_and_seek')
    parser.add_argument('--folder-for-pickle', type=str, default=None)
    parser.add_argument('--lmbda', type=float, default=0.06)
    parser.add_argument('--seed', type=int, default=RANDOM_STATE)
    parser.add_argument('--epochs', type=int_or_none, default=500)
    parser.add_argument('--batch-size', type=int_or_none, default=None)
    parser.add_argument('--num-important-features', type=int, default=3)
    parser.add_argument('--location-cols-to-use', nargs='+', default=['latitude', 'longitude'])
    parser.add_argument('--eval-split', type=str, choices=['val', 'test'], default='val')
    parser.add_argument('--n-ensemble', type=int_or_none, default=None)
    parser.add_argument('--colsample', type=float, default=None)
    parser.add_argument('--ensemble-parallel', type=bool_or_none, default=None)
    parser.add_argument('--ensemble-n-jobs', type=int_or_none, default=None)
    parser.add_argument('--ensemble-backend', type=str, default='loky')
    parser.add_argument('--xgb-params', type=json.loads, default=None, help='JSON string for XGBoost params')
    return parser.parse_args()

def drop_extreme_rows(df, lower=0.02, upper=0.98, ignore_cols=None):
    """
    Drop rows where any feature value lies outside [lower, upper] quantiles.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    lower : float
        Lower quantile (default=0.02).
    upper : float
        Upper quantile (default=0.98).
    ignore_cols : list, optional
        List of column names to ignore when filtering.
    
    Returns
    -------
    pd.DataFrame
        Filtered dataframe with ignored columns untouched.
    """
    if ignore_cols is None:
        ignore_cols = []

    # Only apply filtering to numeric, non-ignored columns
    cols = [c for c in df.columns if c not in ignore_cols and pd.api.types.is_numeric_dtype(df[c])]
    
    lower_bounds = df[cols].quantile(lower)
    upper_bounds = df[cols].quantile(upper)
    
    mask = ((df[cols] >= lower_bounds) & (df[cols] <= upper_bounds)).all(axis=1)
    return df[mask]

# def assign_geography_location(row):
#     if row['dist_la_metro']<RADIUS_KM:
#         return 0
#     elif row['dist_sf_bay']<RADIUS_KM:
#         return 1
#     else:
#         return 2
    
def assign_geography_location(row): #longitude
    if row['longitude']<-121.5:
        return 0
    elif row['longitude']<-118.5:
        return 1
    else:
        return 2
    
# def assign_geography_location(row): #latitude
#     if row['latitude']<34.5:
#         return 0
#     elif row['latitude']<37.6:
#         return 1
#     else:
#         return 2
    
def distance_from_city(lat, lon, city_name, population_centers=POPULATION_CENTRES):
    from geopy.distance import distance
    """
    Compute distance from a given city center.
    
    Parameters
    ----------
    lat, lon : float
        Coordinates of the point.
    city_name : str
        One of 'la_metro' or 'sf_bay'.
    
    Returns
    -------
    float : distance in kilometers
    """
    if city_name not in population_centers:
        raise ValueError(f"city_name must be one of {list(population_centers.keys())}")
    
    city_lat, city_lon = population_centers[city_name]
    return distance((lat, lon), (city_lat, city_lon)).km

def create_logit(row, three_loc_groups=[0,1,2]):
    assert len(three_loc_groups)==3
    if row['location']==three_loc_groups[0]:
        return (row['averooms']-row['avebedrms']) #non-bedrooms per household
    elif row['location']==three_loc_groups[1]:
        return row['population']/(row['aveoccup']) #number of households in the block
    elif row['location']==three_loc_groups[2]:
        return 5*(row['medinc'])-2*(row['houseage'])**2 #penalise very new and very old
    else:
        raise ValueError("create logit error - didn't match groups.")

def create_ground_truth(x_val: pd.DataFrame, location: pd.Series) -> pd.DataFrame:
    """
    Create a ground truth dataframe with the same columns as x_val.
    For each row, set 1 for the features used based on location, else 0.

    Parameters:
        x_val (pd.DataFrame): Input dataframe with named columns.
        location (pd.Series): Series of 0, 1, or 2 indicating location.

    Returns:
        pd.DataFrame: Ground truth dataframe of 0s and 1s.
    """
    # Initialize a dataframe of zeros with the same shape and columns as x_val
    gt_df = pd.DataFrame(0, index=x_val.index, columns=x_val.columns)

    # Define which features correspond to each location
    features_by_location = {
        0: ['averooms','avebedrms','longitude'],
        1: ['population','aveoccup','longitude'],
        2: ['medinc','houseage','longitude']
    }

    # Iterate over rows and set 1s for the used features
    for loc_value, features in features_by_location.items():
        mask = (location == loc_value)
        available_features = [f for f in features if f in gt_df.columns]
        if available_features:
            gt_df.loc[mask, available_features] = 1

    return gt_df


def logit_to_prob(logit, eps=1e-6):
    p = expit(logit) # numerically stable sigmoid
    p = np.clip(p, eps, 1 - eps)
    return p

def add_location_features(df, city_centres=['sf_bay', 'la_metro']):

    for city_centre in city_centres:
        df[f'dist_{city_centre}'] = df.apply(lambda row: 
                                       distance_from_city(row['latitude'],row['longitude'],
                                                         city_name=city_centre),axis=1)
    
    df['location'] = df.apply(lambda row: 
                            assign_geography_location(row),axis=1)

    return df

def add_y(df, leakage_cols=['logit', 'prob', 'raw_score']):
    df = df.copy()
    df['logit'] = df.apply(lambda row: create_logit(row), axis=1)
    df['prob'] = df['logit'].apply(logit_to_prob)

    df_y = df['prob']
    df_x = df.drop(leakage_cols, axis=1, errors='ignore')

    df_y = df_y.values.reshape(1, -1)
    df_y = np.vstack([1 - df_y, df_y]).T
    return df_x, df_y

def scale_some_cols(df, is_train=False, scaler=None, ignore_cols=['location']):
    df = df.reset_index(drop=True)
    scale_cols = [col for col in df.columns if col not in ignore_cols]
    df_scaled = df[scale_cols].copy()
    print(scale_cols)
    if is_train == True:
        if scaler is not None:
            raise ValueError("Cannot be train data and provide a scaler.")
        
        scaler = StandardScaler()
        df_scaled = scaler.fit_transform(df_scaled)
    
    else:
        if scaler == None:
            raise ValueError("Must provide the train scaler for non-train data.")
        df_scaled = scaler.transform(df_scaled)
        
    df_scaled = pd.DataFrame(df_scaled, columns = scale_cols)
    for col in ignore_cols:
        df_scaled[col] = df[col]

    return df_scaled, scaler

def run_experiment(df, location_cols_to_use=['longitude'],
                  lmbda=0.3,
                  epochs=500,
             batch_size=None,
               eval_split='val',
                  model_type='hide_and_seek',
                  pickle_cali_results_folder=None,
               num_important_features=3,
               seed=RANDOM_STATE, #should update to match seed passed in via args
               n_ensemble=None,
               colsample=None,
               ensemble_parallel=None,
               ensemble_n_jobs=None,
                    ensemble_backend='loky',
                    xgb_params=None):
    
    _df = df.copy().reset_index(drop=True)
    _df = add_location_features(_df)

    # Split into train/val/test first, then scale features, then create y using create_logit.
    train_df, temp_df = train_test_split(
        _df,
        test_size=0.2,
        random_state=seed
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        random_state=seed
    )

    ignore_cols=['location'] #in scaling

    train_scaled, scaler = scale_some_cols(df=train_df,
                                           is_train=True, 
                                           scaler=None, ignore_cols=ignore_cols)
    val_scaled, _ = scale_some_cols(df=val_df,
                                    is_train=False,
                                    scaler=scaler, ignore_cols=ignore_cols)
    test_scaled, _ = scale_some_cols(df=test_df,
                                     is_train=False,
                                     scaler=scaler, ignore_cols=ignore_cols)

    x_train, y_train = add_y(train_scaled)
    x_val, y_val = add_y(val_scaled)
    x_test, y_test = add_y(test_scaled)

    #only use specified location features
    location_cols = ['latitude', 'longitude', 'dist_sf_bay', 'dist_la_metro', 'location']

    # Preserve location labels for GT/cluster evaluation even if the location column is dropped from model inputs.
    val_location = x_val['location'].reset_index(drop=True).copy()
    test_location = x_test['location'].reset_index(drop=True).copy()

    location_cols_to_drop = [col for col in location_cols if col not in location_cols_to_use]
    x_train = x_train.drop(location_cols_to_drop,axis=1)
    x_val = x_val.drop(location_cols_to_drop,axis=1)
    x_test = x_test.drop(location_cols_to_drop,axis=1)
    
    print('x_train cols',x_train.columns)
    #run model

    lmbdas = lmbda if isinstance(lmbda, list) else [lmbda]

    if eval_split == 'val':
        x_eval_df = x_val.reset_index(drop=True).copy()
        y_eval = y_val
        eval_location = val_location
    elif eval_split == 'test':
        x_eval_df = x_test.reset_index(drop=True).copy()
        y_eval = y_test
        eval_location = test_location
    else:
        raise ValueError("eval_split must be either 'val' or 'test'")

    # Ground-truth must be built from the same post-drop feature set given to the model.
    ground_truth = create_ground_truth(
        x_val=x_eval_df,
        location=eval_location
    )

    # Keep dataframes for post-hoc cluster analysis and mask column naming.
    x_eval_cluster_df = x_eval_df.copy()
    x_eval_cluster_df['location'] = eval_location.values

    x_train = x_train.values
    x_eval = x_eval_df.values
    full_data_dict = {'x_train':x_train,
                     'y_train':y_train,
                     'x_test':x_eval,
                     'y_test':y_eval,
                     'g_test':ground_truth.values
                     }

    for lmbda in lmbdas:
        results = run_feature_selection_model(
                            full_data_dict=full_data_dict,
                            data_type='cali_housing',
                            folder_for_pickle=pickle_cali_results_folder,
                            model_type=model_type,
                            lmbda=lmbda,
                            epochs=epochs,
                            batch_size=batch_size,
                            seed=seed,
                            num_important_features=num_important_features,
                            n_ensemble=n_ensemble,
                            colsample=colsample,
                            ensemble_parallel=ensemble_parallel,
                            ensemble_n_jobs=ensemble_n_jobs,
                            ensemble_backend=ensemble_backend,
                            xgb_params=xgb_params)

        print('TPR',results['TPR_mean'])
        print('FDR',results['FDR_mean'])
        print('f1',results['f1'])
        print('pct_sig',results['pct_sig'])
        
    return


# def plotTwo(df, lst):
#     """
#     Plot Two Geopandas Plots Side by Side.
#     If lst has two features, the less common value is overlaid on the more common one in red stars.
#     """
#     import geopandas as gpd
#     import geoplot as gplt
#     import geoplot.crs as gcrs
#     import matplotlib.pyplot as plt
    
#     # Load California shapefile
#     cali = gpd.read_file(gplt.datasets.get_path('california_congressional_districts'))
#     cali = cali.assign(area=cali.to_crs('EPSG:3310').geometry.area)
    
#     # Create GeoDataFrame
#     gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude))
#     gdf = gdf.set_crs('EPSG:4326')
    
#     proj = gcrs.AlbersEqualArea(central_latitude=37.16611, central_longitude=-119.44944)
#     fig, ax = plt.subplots(1, 2, figsize=(21, 6), subplot_kw={'projection': proj})
    
#     for ii, i in enumerate(lst):
#         tgdf = gdf.sort_values(by=i, ascending=True)
#         gplt.polyplot(cali, projection=proj, ax=ax[ii])
        
#         # Special case: only two unique values in the feature
#         unique_vals = tgdf[i].value_counts()
#         if len(unique_vals) == 2:
#             # Identify more common and less common values
#             more_common = unique_vals.idxmax()
#             less_common = unique_vals.idxmin()
            
#             # Plot more common as green circles
#             gplt.pointplot(tgdf[tgdf[i] == more_common], ax=ax[ii], color='green', s=2, alpha=1)
#             # Overlay less common as red circles
#             gplt.pointplot(tgdf[tgdf[i] == less_common], ax=ax[ii], color='red', s=2, alpha=0.5)
#             # Overlay with invisible all data to keep all of map
#             gplt.pointplot(tgdf, ax=ax[ii], color='black', s=0.001, alpha=0.001)
#         else:
#             from matplotlib.colors import ListedColormap

#             my_colors = ['black', 'blue','#ff7f0e']  # black, blue, orange
#             cmap = ListedColormap(my_colors)

#             # Default continuous color mapping
#             gplt.pointplot(tgdf, ax=ax[ii], hue=i, cmap=cmap, legend=False, alpha=1.0, s=2)
        
#         ax[ii].set_title(i)
    
#     plt.tight_layout()
#     plt.subplots_adjust(wspace=-0.5)


if __name__ == '__main__':
    args = parse_args()
    print(os.getcwd())
    csv_path = os.path.join(os.path.dirname(__file__), 'df.csv')
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing required dataset file: {csv_path}")
    df = pd.read_csv(csv_path)
    
    print(len(df))
    # Apply legacy filtering before model pipeline.
    df = df[df['houseage'] < 50]
    df = drop_extreme_rows(df, lower=0.01, upper=1, ignore_cols=['latitude', 'longitude'])
    df = drop_extreme_rows(df, lower=0, upper=0.97, ignore_cols=['houseage', 'latitude', 'longitude'])
    print(len(df))

    location_cols_to_use = args.location_cols_to_use

    cali_results = run_experiment(df,
                                location_cols_to_use=location_cols_to_use,
                                lmbda=args.lmbda,
                                epochs=args.epochs,
                                batch_size=args.batch_size,
                                eval_split=args.eval_split,
                                model_type=args.model_type,
                                pickle_cali_results_folder=args.folder_for_pickle,
                                num_important_features=args.num_important_features,
                                seed=args.seed,
                                n_ensemble=args.n_ensemble,
                                colsample=args.colsample,
                                ensemble_parallel=args.ensemble_parallel,
                                ensemble_n_jobs=args.ensemble_n_jobs,
                                ensemble_backend=args.ensemble_backend,
                                xgb_params=args.xgb_params)
