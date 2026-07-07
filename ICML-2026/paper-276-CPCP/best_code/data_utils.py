import os
import requests
import glob
import numpy as np
import pandas as pd
from scipy.io import arff
from sklearn.preprocessing import StandardScaler

# Loader for datasets in ./Datasets

def load_diamonds(path_prefix="./Datasets/"):
    save_name = "diamonds.csv"
    file_path = os.path.join(path_prefix, save_name)
    df = pd.read_csv(file_path)
    
    for col in ['cut', 'color', 'clarity']:
        if col in df.columns: 
            df = pd.get_dummies(df, columns=[col], drop_first=True)
            
    Y = df['price'].values.reshape(-1, 1)
    X = df.drop(columns=['price']).values
    return StandardScaler().fit_transform(X), StandardScaler().fit_transform(Y)

def load_bike(path_prefix="./Datasets/"):
    save_name = "bike_hour.csv"
    file_path = os.path.join(path_prefix, save_name)
    df = pd.read_csv(file_path)
    
    drop_cols = ['dteday', 'casual', 'registered', 'cnt']
    X_cols = [c for c in df.columns if c not in drop_cols]
    Y = df['cnt'].values.reshape(-1, 1)
    X = df[X_cols].values
    return StandardScaler().fit_transform(X), StandardScaler().fit_transform(Y)

def load_superconductivity(path_prefix="./Datasets/"):
    save_name = "super.csv"
    file_path = os.path.join(path_prefix, save_name)
    df = pd.read_csv(file_path)
    Y = df['critical_temp'].values.reshape(-1, 1)
    X = df.drop(columns=['critical_temp']).values
    return StandardScaler().fit_transform(X), StandardScaler().fit_transform(Y)



# --- Multi-Dimensional Datasets ---

def load_naval(path_prefix="./Datasets/"):    
    save_name = "naval.csv"
    file_path = os.path.join(path_prefix, save_name)
    df = pd.read_csv(file_path)
    Y = df.iloc[:, -2:].values
    X = df.iloc[:, :-2].values
    return StandardScaler().fit_transform(X), StandardScaler().fit_transform(Y)

def load_gas_turbine(path_prefix="./Datasets/"):
    save_name = "gt_full.csv"
    file_path = os.path.join(path_prefix, save_name)
    df = pd.read_csv(file_path, index_col=0)
    Y = df.iloc[:, -2:].values
    X = df.iloc[:, :-2].values
    return StandardScaler().fit_transform(X), StandardScaler().fit_transform(Y)

def load_sgemm_product(path_prefix="./Datasets/"):
    save_name = "sgemm_product.csv"
    file_path = os.path.join(path_prefix, save_name)
    if not os.path.exists(file_path):
        file_path = os.path.join(path_prefix, "newData", save_name)
    
    df = pd.read_csv(file_path)
    if len(df) > 50000:
        df = df.sample(n=50000, random_state=42)
    
    target_cols = ['Run1 (ms)', 'Run2 (ms)', 'Run3 (ms)', 'Run4 (ms)']
    found_targets = [c for c in target_cols if c in df.columns]
    
    if len(found_targets) == 4:
        Y = df[found_targets].values
        X = df.drop(columns=found_targets).values
    else:
        Y = df.iloc[:, -4:].values
        X = df.iloc[:, :-4].values
    return StandardScaler().fit_transform(X), StandardScaler().fit_transform(Y)

def load_wec(path_prefix="./Datasets/"):
    save_name = "WEC_Perth_49.csv"
    file_path = os.path.join(path_prefix, save_name)
    if not os.path.exists(file_path): # Fallback search
        possible_paths = [
            os.path.join(path_prefix, "Large-scale Wave Energy Farm", save_name),
            os.path.join(path_prefix, "WEC", save_name),
            os.path.join(path_prefix, "newData", save_name)
        ]
        for p in possible_paths:
            if os.path.exists(p):
                file_path = p
                break
    
    if not os.path.exists(file_path):
        return None, None
        
    df = pd.read_csv(file_path)
    # Heuristic column extraction
    pos_cols = [c for c in df.columns if (c.startswith('X') or c.startswith('Y')) and c[1:].isdigit()]
    p_cols = [c for c in df.columns if (c.startswith('P') or c.startswith('Power')) and c[-1].isdigit()]
    
    if len(pos_cols) == 0 or len(p_cols) == 0:
        X = df.iloc[:, :98].values
        Y = df.iloc[:, 98:98+49].values
    else:
        X = df[pos_cols].values
        Y = df[p_cols].values
    return StandardScaler().fit_transform(X), StandardScaler().fit_transform(Y)

def load_transcoding(path_prefix="./Datasets/"):
    save_name = "transcoding_measurement.tsv"
    file_path = os.path.join(path_prefix, save_name)
    df = pd.read_csv(file_path, sep='\t')
    target_cols = ['utime', 'umem']
    drop_cols = ['id', 'url'] 
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    
    if 'codec' in df.columns: df = pd.get_dummies(df, columns=['codec'], drop_first=True)
    if 'o_codec' in df.columns: df = pd.get_dummies(df, columns=['o_codec'], drop_first=True)
    
    X = df.drop(columns=target_cols).values
    Y = df[target_cols].values
    return StandardScaler().fit_transform(X), StandardScaler().fit_transform(Y)