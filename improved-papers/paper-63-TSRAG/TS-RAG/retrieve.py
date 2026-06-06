import os
import pdb
import math
import torch
import faiss
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from chronos import ChronosPipeline

from utils.tools import get_borders

frequency_dict = {'ETTh1': 'hour', 'ETTh2': 'hour', 'ETTm1': 'minute', 'ETTm2': 'minute',
                    'electricity': 'hour', 'weather': '10minutes', 'traffic': 'hour', 'exchange_rate': 'hour', 'illness': 'hour'}
subdir_name_dict = {'ETTh1': 'ETT-small', 'ETTh2': 'ETT-small', 'ETTm1': 'ETT-small', 'ETTm2': 'ETT-small', 
                    'electricity': 'electricity', 'weather': 'weather', 'traffic': 'traffic'}

def create_database(raw_data, timestamps, lookback_length, embedding_model, metadata):
    embeddings = []
    sliced_timestamps = []

    # batch embedding
    batch_size = 512
    num_batchs = (len(raw_data) - lookback_length + 1 + batch_size - 1) // batch_size

    for batch_idx in tqdm(range(num_batchs)):
        # get start_idx and end_idx
        start_idx = batch_idx * batch_size
        end_idx = min(len(raw_data) - lookback_length + 1, start_idx + batch_size)

        # get batch data
        batch_slices = [raw_data[i:i + lookback_length] for i in range(start_idx, end_idx)]
        batch_slices = torch.tensor(batch_slices)

        # embedding
        batch_embeddings, _ = embedding_model.embed(batch_slices)
        eos_embeddings = batch_embeddings[:, -1, :].float().numpy()

        # append to embeddings
        embeddings.extend(eos_embeddings)
        sliced_timestamps.extend(timestamps[start_idx + lookback_length - 1:end_idx + lookback_length - 1])
    
    embeddings = np.array(embeddings)
    sliced_timestamps = np.array(sliced_timestamps)

    database = {
        'raw_data': raw_data,
        'timestamps': sliced_timestamps,
        'embeddings': embeddings,
        'metadata': metadata
    }
    return database

def save_database(database, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(database, f)

def load_database(file_path):
    with open(file_path, 'rb') as f:
        database = pickle.load(f)
    return database

def generate_retrieval_database(dataset_name, lookback_length, embedding_model, database_dir, root_dir):
    root_dir = Path(root_dir)
    database_dir = Path(database_dir)
    data_path = root_dir / (dataset_name+'.csv')
    frequency = frequency_dict[dataset_name]
    df = pd.read_csv(data_path)
    variables = df.columns[1:]

    databases = {}
    for variable in variables:
        raw_data = df[variable].tolist()
        timestamps = df['date'].tolist()
        metadata = {
            'dataset_name': dataset_name,
            'variable_name': variable,
            'lookback_length': lookback_length,
            'frequency': frequency,
            }
        database = create_database(raw_data, timestamps, lookback_length, embedding_model, metadata)
        databases[variable] = database
    
    save_database(databases, os.path.join(database_dir, f'{dataset_name}_{frequency}_{lookback_length}.pkl'))
    

class Retriever():
    def __init__(self, database_dir, root_dir, metadata, seed, dimension, embedding_model, embedding_tuning):
        self.database_dir = database_dir
        self.metadata = metadata
        self.d = dimension #768
        self.index = None
        self.Y = None
        self.seed = seed
        self.embedding_model = embedding_model
        self.root_dir = root_dir
        self.embedding_tuning = embedding_tuning

    def build_index(self, y_length, begin=None, end=None, variable_filter=None):
        self.raw_data = []
        self.retrieved_metadata = []
        self.timestamps = []
        self.boundary = [0]
        self.index = faiss.IndexFlatL2(self.d)  # euclidean distance

        database_paths = []
        for database_name in self.metadata['database_name']:
            database_path = f'{database_name}_{self.metadata["frequency"]}_{self.metadata["lookback_length"]}.pkl'
            if self.embedding_tuning:
                self.database_dir = os.path.join(self.database_dir, self.embedding_tuning)
            if not os.path.exists(self.database_dir):
                print(f'{self.database_dir} does not exist, building the dir...')
                os.makedirs(self.database_dir)
            if os.path.exists(os.path.join(self.database_dir, database_path)):
                database_paths.append(database_path)
            else:
                print(f'{database_path} does not exist, building the database...')
                generate_retrieval_database(dataset_name=database_name, lookback_length=self.metadata['lookback_length'], embedding_model=self.embedding_model, database_dir=self.database_dir, root_dir=self.root_dir)
                database_paths.append(database_path)
        
        print(f'Build index with database: {database_paths}')

        # load database
        for database_path in database_paths:
            print(f'load database: {database_path}')
            database = load_database(os.path.join(self.database_dir, database_path))
            # filter by metadata['variable']
            for key in database.keys():
                if variable_filter is None or key in variable_filter:
                    embeddings = database[key]['embeddings']
                    embeddings = embeddings.reshape(-1, self.d).astype('float32')  # reshape to (n, d)
                    
                    # filter embeddings
                    if begin == None:
                        filter_begin = 0
                    else:
                        filter_begin = begin
                    if end == None:
                        filter_end = -y_length
                    else:
                        filter_end = end

                    embeddings = embeddings[filter_begin:filter_end, :]

                    self.index.add(embeddings)

                    self.timestamps.append(database[key]['timestamps'])
                    self.raw_data.append(database[key]['raw_data'])
                    self.retrieved_metadata.append(database[key]['metadata'])
                    self.boundary.append(embeddings.shape[0])                       
                
            self.boundary = [sum(self.boundary[:i]) for i in range(1, len(self.boundary)+1)]

    def search(self, query_vector, top_k, drop_first=False, params=None):
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)
        # drop first or last
        if params is None:
            distances, indices = self.index.search(query_vector, top_k + 1)
        else:
            distances, indices = self.index.search(query_vector, top_k + 1, params=params)
        if drop_first:
            distances = distances[:, 1:]
            indices = indices[:, 1:]
        else:
            distances = distances[:, :-1]
            indices = indices[:, :-1]
        
        # find boundary_idx and timestamp_idx
        boundary_idx_batch = np.digitize(indices, self.boundary) - 1
        timestamp_idx_batch = indices - np.array(self.boundary)[boundary_idx_batch]
        # if np.any(timestamp_idx_batch < 0):
        #     print('timestamp_idx_batch has negative values')
        #     pdb.set_trace()

        return distances, boundary_idx_batch, timestamp_idx_batch

def do_retrieve(original_data_name, retrieval_database_dir, root_dir, metadata, mode, top_k, context_length, prediction_length, seed, dimension, embedding_model, save=True, embedding_tuning=None):
    '''
    input: the original data, retrieval database, metadata and retrieve mode
    output: retrieved_data
    '''
    # load original data
    original_data_path = os.path.join(root_dir, original_data_name+'.csv')
    original_data = pd.read_csv(original_data_path)
    variable_names = original_data.columns[1:]  # exclude 'date'
    print(f'There are {len(variable_names)} variables in the original data')

    # initialize the retrieved data
    boundary_idx_matrix = np.full((len(original_data), len(variable_names), top_k), np.nan)
    timestamp_idx_matrix = np.full((len(original_data), len(variable_names), top_k), np.nan)
    distance_matrix = np.full((len(original_data), len(variable_names), top_k), np.nan)

    # get borders
    border1s, border2s = get_borders(original_data_name, context_length, len(original_data))

    if mode == 'only_self_train':
        print(f'----------For one variable, we retrieve data from itself training set----------')
        # each variable retrieve from historical data of itself
        for var_idx, var_name in enumerate(variable_names):
            print(f'----------Retrieving for variable: {var_name}')
            retriever = Retriever(database_dir=retrieval_database_dir, metadata=metadata, seed=seed, dimension=dimension, embedding_model=embedding_model, root_dir=root_dir, embedding_tuning=embedding_tuning)

            # retriever.build_index(y_length=prediction_length, variable_filter=[var_name])
            retriever.build_index(y_length=prediction_length, variable_filter=[var_name], begin=border1s[0], end=border2s[0])
            
            sequence = original_data[var_name].values

            ## batch search
            start_idx_list = list(range(0, len(sequence) - context_length - prediction_length + 1))
            end_idx_list = [start_idx + context_length for start_idx in start_idx_list]
            
            # get batch of start_idx and end_idx
            search_batch_size = 512
            batch_num = math.ceil(len(start_idx_list) / search_batch_size)
            for batch_idx in tqdm(range(batch_num)):
                start_idx_batch = start_idx_list[batch_idx*search_batch_size:min((batch_idx+1)*search_batch_size, len(start_idx_list))]
                end_idx_batch = end_idx_list[batch_idx*search_batch_size:min((batch_idx+1)*search_batch_size, len(start_idx_list))]
                
                # for training and validation set, do not need to search
                if end_idx_batch[-1] <= border2s[0]:
                    boundary_idx_batch = np.zeros((len(start_idx_batch), top_k))
                    timestamp_idx_batch = np.zeros((len(start_idx_batch), top_k))
                    distance_batch = np.zeros((len(start_idx_batch), top_k))
                    boundary_idx_matrix[start_idx_batch, var_idx, :] = boundary_idx_batch
                    timestamp_idx_matrix[start_idx_batch, var_idx, :] = timestamp_idx_batch
                    distance_matrix[start_idx_batch, var_idx, :] = distance_batch

                else:
                    seq_x_batch = np.array([sequence[start_idx:end_idx] for start_idx, end_idx in zip(start_idx_batch, end_idx_batch)])
                    query_vector_batch, _ = embedding_model.embed(torch.tensor(seq_x_batch))
                    query_vector_batch = query_vector_batch[:,-1,:].squeeze().float().numpy()

                    distances_batch, boundary_idx_batch, timestamp_idx_batch = retriever.search(query_vector_batch, top_k=top_k)
                    boundary_idx_matrix[start_idx_batch, var_idx, :] = boundary_idx_batch
                    timestamp_idx_matrix[start_idx_batch, var_idx, :] = timestamp_idx_batch
                    distance_matrix[start_idx_batch, var_idx, :] = distances_batch
            # import pdb; pdb.set_trace()
            
    boundary_idx_df = pd.DataFrame(boundary_idx_matrix.reshape(len(original_data), -1), columns=[f'boundary_idx_{var}_{k}' for var in variable_names for k in range(top_k)])
    timestamp_idx_df = pd.DataFrame(timestamp_idx_matrix.reshape(len(original_data), -1), columns=[f'timestamp_idx_{var}_{k}' for var in variable_names for k in range(top_k)])
    distance_df = pd.DataFrame(distance_matrix.reshape(len(original_data), -1), columns=[f'distance_{var}_{k}' for var in variable_names for k in range(top_k)])
    retrieved_data = pd.concat([original_data, boundary_idx_df, timestamp_idx_df, distance_df], axis=1)

    assert (pd.concat([boundary_idx_df.isna().sum().reset_index(drop=True), 
                   timestamp_idx_df.isna().sum().reset_index(drop=True), 
                   distance_df.isna().sum().reset_index(drop=True)], axis=1).nunique(axis=1) == 1).all(), "NaN counts are not the same in all columns"
    
    if save:
        retrieval_database_names = '_'.join(metadata['database_name'])
        retrieved_data_path = os.path.join(root_dir, f'{original_data_name}_retrieve_{retrieval_database_names}_{metadata["lookback_length"]}_{mode}_{embedding_tuning}.csv')
        print(f'Saving the retrieved data to {retrieved_data_path}')
        retrieved_data.to_csv(retrieved_data_path, index=False)

    return retrieved_data


if __name__ == "__main__":
    if 1:
        original_data_path = './datasets/ETT-small/ETTh1.csv'
        original_data_name = 'ETTh1'
        retrieval_database_dir = '../retrieval_database/'
        root_dir = './datasets/ETT-small'
        metadata = {
            'database_name': ['ETTh2'],
            'lookback_length': 96,
            'frequency': 'hour',
        }
        mode = 'only_self'
        dimension = 768
        save = True
        seed = 42
        top_k = 20
        context_length = 96
        prediction_length = 96

        embedding_model = ChronosPipeline.from_pretrained(
            "amazon/chronos-t5-base",
            device_map="cuda",
            torch_dtype=torch.bfloat16,
        )

        do_retrieve(original_data_name, retrieval_database_dir, root_dir, metadata, mode, top_k, context_length, prediction_length, seed, dimension, embedding_model, save)
    
    if 0:
        retrieval_database_dir = '../retrieval_database/'
        root_dir = './datasets/ETT-small'
        metadata = {
            'database_name': ['ETTh2'],
            'lookback_length': 96,
            'frequency': 'hour',
        }
        seed = 42
        dimension = 768
        embedding_model = ChronosPipeline.from_pretrained(
            "amazon/chronos-t5-base",
            device_map="cuda",
            torch_dtype=torch.bfloat16,
        )
        retriever = Retriever(database_dir=retrieval_database_dir, root_dir=root_dir, metadata=metadata, seed=seed, dimension=dimension, embedding_model=embedding_model)
        retriever.build_index(y_length=96)

        query_vector = torch.randn(32, 768)
        distances, boundary_idx_list, timestamp_idx_list = retriever.search(query_vector, top_k=20)
        pdb.set_trace()
        print('done')