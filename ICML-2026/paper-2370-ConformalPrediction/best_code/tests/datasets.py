import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '../'))
import numpy as np
import pandas as pd

def load_dataset(name):
    if name == "daily-climate":
        df = pd.read_csv('./datasets/raw/daily-climate.csv')
        df.rename({'date': 'timestamp', 'meantemp': 'y'}, axis='columns', inplace=True)
        df = df.drop("Unnamed: 0", axis='columns')
        data = df.melt(id_vars=['timestamp'], value_name='target')
        data.rename({'variable': 'item_id'}, axis='columns', inplace=True)
    if name.endswith("-stock"):
        ticker = name.replace("-stock", "")
        df = pd.read_csv('./datasets/raw/djia.csv')
        df = df.drop(["High", "Low", "Close", "Volume"], axis='columns')
        df.rename({'Date': 'timestamp', 'Name': 'item_id'}, axis='columns', inplace=True)
        df.replace({'item_id': {ticker: 'y'}}, inplace=True)
        data = df.melt(id_vars=['timestamp', 'item_id'], value_name='target')
        data.drop("variable", axis='columns', inplace=True)
    if os.name.endswith("-COVID-deaths-4wk"):
        state = name.split('-')[0]
        file_path = f'./datasets/processed/COVID/{state}_proc_4wkdeaths.pkl'
        df = pd.read_pickle(file_path)
        df.rename({'variable': 'item_id'}, axis='columns', inplace=True)
        data = df
        data = data[data.timestamp <=  np.sort(data.timestamp.unique())[105]]
    # if name == "COVID-deaths4wk":
    #     df = pd.read_pickle('./datasets/covid-ts-proc/proc_4wkdeaths.pkl')
    #     df.rename({'variable' : 'item_id'}, axis='columns', inplace=True)
    #     data = df
    # if name == "tx-COVID-deaths-4wk":
    #     df = pd.read_pickle('./datasets/covid-ts-proc/statewide/tx_proc_4wkdeaths.pkl')
    #     df.rename({'variable' : 'item_id'}, axis='columns', inplace=True)
    #     data = df
    # if name == "ca-COVID-deaths-4wk":
    #     df = pd.read_pickle('./datasets/covid-ts-proc/statewide/ca_proc_4wkdeaths.pkl')
    #     df.rename({'variable' : 'item_id'}, axis='columns', inplace=True)
    #     data = df
    #     data = data[data.timestamp <=  np.sort(data.timestamp.unique())[105]]
    # if name == "ga-COVID-deaths-4wk":
    #     df = pd.read_pickle('./datasets/covid-ts-proc/statewide/ga_proc_4wkdeaths.pkl')
    #     df.rename({'variable' : 'item_id'}, axis='columns', inplace=True)
    #     data = df
    #     data = data[data.timestamp <=  np.sort(data.timestamp.unique())[105]]
    # if name == "fl-COVID-deaths-4wk":
    #     df = pd.read_pickle('./datasets/covid-ts-proc/statewide/fl_proc_4wkdeaths.pkl')
    #     df.rename({'variable' : 'item_id'}, axis='columns', inplace=True)
    #     data = df
    #     data = data[data.timestamp <=  np.sort(data.timestamp.unique())[105]]
    # if name == "ny-COVID-deaths-4wk":
    #     df = pd.read_pickle('./datasets/covid-ts-proc/statewide/ny_proc_4wkdeaths.pkl')
    #     df.rename({'variable' : 'item_id'}, axis='columns', inplace=True)
    #     data = df
    #     data = data[data.timestamp <=  np.sort(data.timestamp.unique())[105]]
    # if name == "COVID-cases3wk":
    #     df = pd.read_pickle('./datasets/covid-ts-proc/proc_3wkcases.pkl')
    #     df.rename({'variable' : 'item_id'}, axis='columns', inplace=True)
    #     data = df
    if name == "elec2":
        df = pd.read_csv('./datasets/raw/elec2.csv')
        df['timestamp'] = pd.date_range(start='1996-5-7', end='1998-12-6 23:30:00', freq='30T', inclusive='both')
        df['class'] = (df['class'] == 'UP').astype(float)
        df.rename({'nswdemand': 'y'}, axis='columns', inplace=True)
        df = df[:2000]
        data = df.melt(id_vars=['timestamp'], value_name='target')
        data.rename({'variable': 'item_id'}, axis='columns', inplace=True)
        data.astype({'target': 'float64'})
    # if name == "M4":
    #     data = pd.read_csv("https://autogluon.s3.amazonaws.com/datasets/timeseries/m4_hourly_subset/train.csv")
    data = data.pivot(columns="item_id", index="timestamp", values="target")
    data['y'] = data['y'].astype(float)
    data = data.interpolate()
    data.index = pd.to_datetime(data.index)
    return data

def generate_synthetic_ensemble(name="changepoint", n_seeds=5):
    data_ensemble = []
    for _ in range(n_seeds):
        data = generate_synthetic_data(name)
        data_ensemble.append(data)
    
    return data_ensemble
    # if not aggregate:
    #     return data_ensemble
    # else:
    #     all_xs = [data['x'] for data in data_ensemble]
    #     all_ys = [data['y'] for data in data_ensemble]
        
    #     agg_x = np.mean(all_xs, axis=0)
    #     agg_y = np.mean(all_ys, axis=0)
        
    #     return {'x': agg_x, 'y': agg_y}

def generate_synthetic_data(name="changepoint"):
    n_steps = 3000
    d = 4 # Number of features
    x = np.random.normal(0, 1, (n_steps, d))
        
    if name == "changepoint":
        betas = np.zeros((n_steps, d))
        # Phase 1: t=1..500
        betas[:500] = [2, 1, 0, 0]
        # # Phase 2: t=501..1500
        betas[500:1500] = [0, -2, -1, 0]
        # # Phase 3: t=1501..2000
        betas[1500:] = [0, 0, 2, 1]
        variance = 0.25
        epsilon = np.random.normal(0, np.sqrt(variance), n_steps)
        y = np.sum(x * betas, axis=1) + epsilon
    elif name == "changevariance":
        # Static setup
        beta = np.array([2, 1, 0, 0])
        variances = np.zeros(n_steps)

        variances[:500] = 4.0**2
        variances[500:1500] = 1.5**2
        variances[1500:] = 0.5**2
        epsilon = np.random.normal(0, np.sqrt(variances), n_steps)
        y = np.sum(x * beta, axis=1) + epsilon
    else:
        raise ValueError("Invalid synthetic dataset name")
    
    return {'x': x, 'y': y}

if __name__ == "__main__":
    # Iterate through all the datasets and attempt loading them
    datasets = ['tx-COVID-deaths-4wk', 'ca-COVID-deaths-4wk']
    for dataset in datasets:
        print(f"Loading {dataset} dataset")
        data = load_dataset(dataset)
        print(f"Loaded {dataset} dataset")
        print(data.columns)
