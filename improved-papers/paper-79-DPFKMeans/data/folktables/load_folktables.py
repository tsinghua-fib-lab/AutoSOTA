import numpy as np
import os
import pandas as pd
import pathlib
from folktables import ACSDataSource, ACSIncome, generate_categories
import h5py

from pfl.data.dataset import Dataset

# Maps filter label to name of column
COW_LABELS = {
        1: "Employee of a private for-profit company or business, or of an individual, "
           "for wages, salary, or commissions",
        2: "Employee of a private not-for-profit, tax-exempt, or charitable organization",
        3: "Local government employee (city, county, etc.)",
        4: "State government employee",
        5: "Federal government employee",
        6: "Self-employed in own not incorporated business, professional practice, or farm",
        7: "Self-employed in own incorporated business, professional practice or farm",
        8: "Working without pay in family business or farm"
    }

STATE_LIST = ['AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA', 'HI',
              'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD', 'MA', 'MI',
              'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ', 'NM', 'NY', 'NC',
              'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT',
              'VT', 'VA', 'WA', 'WV', 'WI', 'WY', 'PR'
              ]


def make_folktables_datasets(args, make_clients=True, make_server=True):
    for filter_label in [2, 5, 6]:
        print(f'Running filter_label = {filter_label}')
        args.filter_label = filter_label
        full_columns_list = None
        train_client_datasets, val_client_datasets = [], []
        for state in STATE_LIST:
            print(f'State: {state}')
            data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')
            state_data = data_source.get_data(states=[state], download=True)
            definition_df = data_source.get_definitions(download=True)
            categories = generate_categories(features=ACSIncome.features, definition_df=definition_df)
            if full_columns_list is None:
                full_columns_list = []
                for cat, col_dict in categories.items():
                    for idx, col_name in col_dict.items():
                        full_columns_list.append(f'{cat}_{col_name}')

            partial_features, labels, _ = ACSIncome.df_to_pandas(state_data, categories=categories, dummies=True)
            partial_features = partial_features.astype(bool)
            features = pd.DataFrame(
                data=np.zeros((len(partial_features), len(full_columns_list)), dtype=bool),
                columns=full_columns_list
            )
            features[partial_features.columns[2:]] = partial_features[partial_features.columns[2:]]

            columns_without_cow = [c for c in features.columns if c[:3] != 'COW']
            filter_column_name = f'COW_{COW_LABELS[args.filter_label]}'
            filtered_data = features.loc[features[filter_column_name]][columns_without_cow].to_numpy().astype(bool)
            filtered_labels = labels.loc[features[filter_column_name]].to_numpy().reshape(-1)

            dataset = Dataset((filtered_data, filtered_labels))
            train_dataset, val_dataset = dataset.split(fraction=0.9)

            train_client_datasets.append(train_dataset)
            val_client_datasets.append(val_dataset)
            path_to_data_file = pathlib.Path(__file__).parent.absolute()
            path_to_data = os.path.join(path_to_data_file, 'cow_extracted_data',
                                        f'fl-{args.filter_label}')

            # path_to_data = os.path.join('cow_extracted_data', f'fl-{args.filter_label}')
            if not os.path.exists(path_to_data):
                os.makedirs(path_to_data)

            if make_clients:
                for split, split_dataset in zip(['train', 'val'], [train_dataset, val_dataset]):
                    with h5py.File(os.path.join(path_to_data, f'{split}_users.hdf5'), 'a') as f:
                        x, y = split_dataset.raw_data
                        f.create_dataset(f'/{state}/features', data=x)
                        f.create_dataset(f'/{state}/labels', data=y)

            if make_server:
                server_features = features.loc[~features[filter_column_name]][columns_without_cow].to_numpy().astype(bool)
                server_labels = labels.loc[~features[filter_column_name]].to_numpy().astype(bool)
                assert len(server_features) == len(server_labels)
                sampled_idxs = np.random.choice(range(len(server_features)), size=len(server_features) // 10, replace=False)
                with h5py.File(os.path.join(path_to_data, 'server_data.hdf5'), 'a') as server_f:
                    server_f.create_dataset(f'/{state}/features', data=server_features[sampled_idxs])
                    server_f.create_dataset(f'/{state}/labels', data=server_labels[sampled_idxs])
