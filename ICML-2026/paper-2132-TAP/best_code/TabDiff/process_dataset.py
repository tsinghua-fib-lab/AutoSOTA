import numpy as np
import pandas as pd
import os
import sys
import json
import argparse

from sklearn.preprocessing import OrdinalEncoder
from sklearn import model_selection

TYPE_TRANSFORM ={
    'float', np.float32,
    'str', str,
    'int', int
}

INFO_PATH = 'data/Info'

parser = argparse.ArgumentParser(description='process dataset')

# General configs
parser.add_argument('--dataname', type=str, default=None, help='Name of dataset.')
args = parser.parse_args()

def preprocess_beijing():
    with open(f'{INFO_PATH}/beijing.json', 'r') as f:
        info = json.load(f)
    
    data_path = info['raw_data_path']

    data_df = pd.read_csv(data_path)
    columns = data_df.columns

    data_df = data_df[columns[1:]]


    df_cleaned = data_df.dropna()
    df_cleaned.to_csv(info['data_path'], index = False)
    
def preprocess_beijing_dcr():
    with open(f'{INFO_PATH}/beijing_dcr.json', 'r') as f:
        info = json.load(f)
    
    data_path = info['raw_data_path']

    data_df = pd.read_csv(data_path)
    columns = data_df.columns

    data_df = data_df[columns[1:]]


    df_cleaned = data_df.dropna()
    df_cleaned.to_csv(info['data_path'], index = False)

def preprocess_news(remove_cat=False):
    name = 'news' if not remove_cat else 'news_nocat'
    with open(f'{INFO_PATH}/{name}.json', 'r') as f:
        info = json.load(f)

    data_path = info['raw_data_path']
    data_df = pd.read_csv(data_path)
    data_df = data_df.drop('url', axis=1)

    columns = np.array(data_df.columns.tolist())

    cat_columns1 = columns[list(range(12,18))]
    cat_columns2 = columns[list(range(30,38))]
    
    if not remove_cat:
        cat_col1 = data_df[cat_columns1].astype(int).to_numpy().argmax(axis = 1)
        cat_col2 = data_df[cat_columns2].astype(int).to_numpy().argmax(axis = 1)

    data_df = data_df.drop(cat_columns2, axis=1)
    data_df = data_df.drop(cat_columns1, axis=1)

    if not remove_cat:
        data_df['data_channel'] = cat_col1
        data_df['weekday'] = cat_col2
    
    data_save_path = f'data/{name}/{name}.csv'
    data_df.to_csv(f'{data_save_path}', index = False)

    columns = np.array(data_df.columns.tolist())
    num_columns = columns[list(range(45))]
    cat_columns = ['data_channel', 'weekday'] if not remove_cat else []
    target_columns = columns[[45]]

    info['num_col_idx'] = list(range(45))
    info['cat_col_idx'] = [46, 47] if not remove_cat else []
    info['target_col_idx'] = [45]
    info['data_path'] = data_save_path
    
    with open(f'{INFO_PATH}/{name}.json', 'w') as file:
        json.dump(info, file, indent=4)
        
def preprocess_news_dcr(remove_cat=False):
    name = 'news_dcr' if not remove_cat else 'news_nocat_dcr'
    with open(f'{INFO_PATH}/{name}.json', 'r') as f:
        info = json.load(f)

    data_path = info['raw_data_path']
    data_df = pd.read_csv(data_path)
    data_df = data_df.drop('url', axis=1)

    columns = np.array(data_df.columns.tolist())

    cat_columns1 = columns[list(range(12,18))]
    cat_columns2 = columns[list(range(30,38))]
    
    if not remove_cat:
        cat_col1 = data_df[cat_columns1].astype(int).to_numpy().argmax(axis = 1)
        cat_col2 = data_df[cat_columns2].astype(int).to_numpy().argmax(axis = 1)

    data_df = data_df.drop(cat_columns2, axis=1)
    data_df = data_df.drop(cat_columns1, axis=1)

    if not remove_cat:
        data_df['data_channel'] = cat_col1
        data_df['weekday'] = cat_col2
    
    data_save_path = f'data/{name}/{name}.csv'
    data_df.to_csv(f'{data_save_path}', index = False)

    columns = np.array(data_df.columns.tolist())
    num_columns = columns[list(range(45))]
    cat_columns = ['data_channel', 'weekday'] if not remove_cat else []
    target_columns = columns[[45]]

    info['num_col_idx'] = list(range(45))
    info['cat_col_idx'] = [46, 47] if not remove_cat else []
    info['target_col_idx'] = [45]
    info['data_path'] = data_save_path
    
    with open(f'{INFO_PATH}/{name}.json', 'w') as file:
        json.dump(info, file, indent=4)
    
def preprocess_diabetes():
    """
    Preprocesses the diabetes dataset is aligned with the concurrent work
    Continuous Diffusion for Mixed-Type Tabular Data (CDTD):
    https://github.com/muellermarkus/cdtd
    """
    with open(f'{INFO_PATH}/diabetes.json', 'r') as f:
        info = json.load(f)

    info['num_col_idx'] = list(range(9))
    info['cat_col_idx'] = list(range(9, 36))
    info['target_col_idx'] = [36]
    
    data_path = info['raw_data_path']
    df = pd.read_csv(data_path, sep=',')
    df = df[info['column_names']]
    df = df.replace(r' ', np.nan)
    df = df.replace(r'?', np.nan)
    df = df.replace(r'', np.nan)
    
    num_features = [info['column_names'][idx] for idx in info['num_col_idx']]
    cat_features = [info['column_names'][idx] for idx in info['cat_col_idx']]
    target = info['column_names'][info['target_col_idx'][0]]
    df[target] = np.where(df[target] == 'NO', 0, 1)
    enc = OrdinalEncoder()
    df['age'] = enc.fit_transform(df['age'].to_numpy().reshape(-1,1))
    
    # remove rows with missings in targets
    idx_target_nan = df[target].isna().to_numpy().nonzero()[0]
    df.drop(labels = idx_target_nan, axis = 0, inplace = True)
    
    # for categorical features, replace missings with 'empty', which will be counted as a new category
    df[cat_features] = df[cat_features].fillna('empty')
    
    # for continuous data, drop missing
    df.dropna(inplace = True)
    
    # ensure correct types
    X_cat = df[cat_features].to_numpy().astype('str')
    X_cont = df[num_features].to_numpy().astype('float')
    y = df[[target]].to_numpy()
    
    val_prop, test_prop = 0.2, 0.2
    prop = val_prop / (1 - test_prop) 
    
    stratify = None if info['task_type'] == 'regression' else y
    X_cat_train, X_cat_test, X_cont_train, X_cont_test, y_train, y_test = \
        model_selection.train_test_split(X_cat, X_cont, y, test_size = test_prop, 
                                        stratify = stratify, random_state = 42)
    if val_prop > 0:
        stratify = None if info['task_type'] == 'regression' else y_train
        X_cat_train, X_cat_val, X_cont_train, X_cont_val, y_train, y_val = \
            model_selection.train_test_split(X_cat_train, X_cont_train, y_train,
                                            stratify = stratify, test_size = prop, 
                                            random_state = 42)
    
    train_df = pd.DataFrame(np.concatenate([X_cont_train, X_cat_train, y_train], axis = 1), columns = num_features + cat_features + [target])
    val_df = pd.DataFrame(np.concatenate([X_cont_val, X_cat_val, y_val], axis = 1), columns = num_features + cat_features + [target])
    test_df = pd.DataFrame(np.concatenate([X_cont_test, X_cat_test, y_test], axis = 1), columns = num_features + cat_features + [target])

    # Save the splited data
    train_df.to_csv(info['data_path'], index = False)
    val_df.to_csv(info['val_path'], index = False)
    test_df.to_csv(info['test_path'], index = False)
    # Save updated info
    with open(f'{INFO_PATH}/diabetes.json', 'w') as file:
        json.dump(info, file, indent=4)
        
def preprocess_diabetes_dcr():
    """
    Preprocesses the diabetes dataset is aligned with the concurrent work
    Continuous Diffusion for Mixed-Type Tabular Data (CDTD):
    https://github.com/muellermarkus/cdtd
    """
    with open(f'{INFO_PATH}/diabetes_dcr.json', 'r') as f:
        info = json.load(f)

    info['num_col_idx'] = list(range(9))
    info['cat_col_idx'] = list(range(9, 36))
    info['target_col_idx'] = [36]
    
    data_path = info['raw_data_path']
    df = pd.read_csv(data_path, sep=',')
    df = df[info['column_names']]
    df = df.replace(r' ', np.nan)
    df = df.replace(r'?', np.nan)
    df = df.replace(r'', np.nan)
    
    num_features = [info['column_names'][idx] for idx in info['num_col_idx']]
    cat_features = [info['column_names'][idx] for idx in info['cat_col_idx']]
    target = info['column_names'][info['target_col_idx'][0]]
    df[target] = np.where(df[target] == 'NO', 0, 1)
    enc = OrdinalEncoder()
    df['age'] = enc.fit_transform(df['age'].to_numpy().reshape(-1,1))
    
    # remove rows with missings in targets
    idx_target_nan = df[target].isna().to_numpy().nonzero()[0]
    df.drop(labels = idx_target_nan, axis = 0, inplace = True)
    
    # for categorical features, replace missings with 'empty', which will be counted as a new category
    df[cat_features] = df[cat_features].fillna('empty')
    
    # for continuous data, drop missing
    df.dropna(inplace = True)
    
    # ensure correct types
    X_cat = df[cat_features].to_numpy().astype('str')
    X_cont = df[num_features].to_numpy().astype('float')
    y = df[[target]].to_numpy()
    
    val_prop, test_prop = 0.0, 0.5      # 50-50 split for dcr eval
    prop = val_prop / (1 - test_prop) 
    
    stratify = None if info['task_type'] == 'regression' else y
    X_cat_train, X_cat_test, X_cont_train, X_cont_test, y_train, y_test = \
        model_selection.train_test_split(X_cat, X_cont, y, test_size = test_prop, 
                                        stratify = stratify, random_state = 42)
    if val_prop > 0:
        stratify = None if info['task_type'] == 'regression' else y_train
        X_cat_train, X_cat_val, X_cont_train, X_cont_val, y_train, y_val = \
            model_selection.train_test_split(X_cat_train, X_cont_train, y_train,
                                            stratify = stratify, test_size = prop, 
                                            random_state = 42)
    
    train_df = pd.DataFrame(np.concatenate([X_cont_train, X_cat_train, y_train], axis = 1), columns = num_features + cat_features + [target])
    if val_prop > 0:
        val_df = pd.DataFrame(np.concatenate([X_cont_val, X_cat_val, y_val], axis = 1), columns = num_features + cat_features + [target])
    else:
        val_df = pd.DataFrame(columns = num_features + cat_features + [target]).astype(train_df.dtypes)
    test_df = pd.DataFrame(np.concatenate([X_cont_test, X_cat_test, y_test], axis = 1), columns = num_features + cat_features + [target])

    # Save the splited data
    train_df.to_csv(info['data_path'], index = False)
    val_df.to_csv(info['val_path'], index = False)
    test_df.to_csv(info['test_path'], index = False)
    # Save updated info
    with open(f'{INFO_PATH}/diabetes_dcr.json', 'w') as file:
        json.dump(info, file, indent=4)
    


def get_column_name_mapping(data_df, num_col_idx, cat_col_idx, target_col_idx, column_names = None):
    
    if not column_names:
        column_names = np.array(data_df.columns.tolist())
    

    idx_mapping = {}

    curr_num_idx = 0
    curr_cat_idx = len(num_col_idx)
    curr_target_idx = curr_cat_idx + len(cat_col_idx)

    for idx in range(len(column_names)):

        if idx in num_col_idx:
            idx_mapping[int(idx)] = curr_num_idx
            curr_num_idx += 1
        elif idx in cat_col_idx:
            idx_mapping[int(idx)] = curr_cat_idx
            curr_cat_idx += 1
        else:
            idx_mapping[int(idx)] = curr_target_idx
            curr_target_idx += 1


    inverse_idx_mapping = {}
    for k, v in idx_mapping.items():
        inverse_idx_mapping[int(v)] = k
        
    idx_name_mapping = {}
    
    for i in range(len(column_names)):
        idx_name_mapping[int(i)] = column_names[i]

    return idx_mapping, inverse_idx_mapping, idx_name_mapping


def train_val_test_split(data_df, cat_columns, num_train = 0, num_test = 0):
    total_num = data_df.shape[0]
    idx = np.arange(total_num)


    seed = 1234

    while True:
        np.random.seed(seed)
        np.random.shuffle(idx)

        train_idx = idx[:num_train]
        test_idx = idx[-num_test:]


        train_df = data_df.loc[train_idx]
        test_df = data_df.loc[test_idx]



        flag = 0
        for i in cat_columns:
            if len(set(train_df[i])) != len(set(data_df[i])):
                flag = 1
                break

        if flag == 0:
            break
        else:
            seed += 1
        
    return train_df, test_df, seed    


def process_data(name):

    if name == 'news':
        preprocess_news()
    elif name == 'news_nocat':
        preprocess_news(remove_cat=True)
    elif name == 'news_dcr':
        preprocess_news_dcr()
    elif name == 'beijing':
        preprocess_beijing()
    elif name == 'beijing_dcr':
        preprocess_beijing_dcr()
    elif name == 'diabetes':
        preprocess_diabetes()
    elif name == 'diabetes_dcr':
        preprocess_diabetes_dcr()

    with open(f'{INFO_PATH}/{name}.json', 'r') as f:
        info = json.load(f)

    data_path = info['data_path']
    if info['file_type'] == 'csv':
        data_df = pd.read_csv(data_path, header = info['header'])

    elif info['file_type'] == 'xls':
        data_df = pd.read_excel(data_path, sheet_name='Data', header=1)
        data_df = data_df.drop('ID', axis=1)

    num_data = data_df.shape[0]

    column_names = info['column_names'] if info['column_names'] else data_df.columns.tolist()
 
    num_col_idx = info['num_col_idx']
    cat_col_idx = info['cat_col_idx']
    target_col_idx = info['target_col_idx']

    num_columns = [column_names[i] for i in num_col_idx]
    cat_columns = [column_names[i] for i in cat_col_idx]
    target_columns = [column_names[i] for i in target_col_idx]
    
    idx_mapping, inverse_idx_mapping, idx_name_mapping = get_column_name_mapping(data_df, num_col_idx, cat_col_idx, target_col_idx, column_names)

    has_val = bool(info['val_path'])
    val_df = pd.DataFrame(columns=data_df.columns).astype(data_df.dtypes)   # by default (val_path is not provided), set val_Df to be empty
    if info['test_path']:

        # if testing data is given
        test_path = info['test_path']
        
        if "adult" in name:     # BUG: currently data saved at adult's test_path cannot be directly loaded. Consider integrate the following code to a preprocesing function for adult
            with open(test_path, 'r') as f:
                lines = f.readlines()[1:]
                test_save_path = f'data/{name}/test.data'
                if not os.path.exists(test_save_path):
                    with open(test_save_path, 'a') as f1:     
                        for line in lines:
                            save_line = line.strip('\n').strip('.')
                            f1.write(f'{save_line}\n')

            test_df = pd.read_csv(test_save_path, header = None)
        else:
            test_df = pd.read_csv(test_path, header = info['header'])
            
        if has_val:     # currently you cannot have a val path without a test path
            val_path = info['val_path']
            val_df = pd.read_csv(val_path, header = info['header'])
            
        train_df = data_df
        
        if "dcr" in name and "diabetes" not in name:   # create 50/50 splits for dcr datasets; no need for this for diabetes dataset as it's done in preprocessing
            complete_df = pd.concat([train_df, test_df, val_df], axis = 0, ignore_index=True)
            num_data = complete_df.shape[0]
            num_train = int(num_data*0.5)
            num_test = num_data - num_train
            complete_df.rename(columns = idx_name_mapping, inplace=True)
            train_df, test_df, seed = train_val_test_split(complete_df, cat_columns, num_train, num_test)

    else:  
        # Train/ Test Split, 90% Training (50% for dcr eval exclusively), 10% Testing (Validation set will be selected from Training set)
        if "dcr" in name:
            num_train = int(num_data*0.5)
        else:
            num_train = int(num_data*0.9)
        num_test = num_data - num_train

        train_df, test_df, seed = train_val_test_split(data_df, cat_columns, num_train, num_test)
    
    complete_df = pd.concat([train_df, test_df, val_df], axis = 0)
    name_idx_mapping = {val: key for key, val in idx_name_mapping.items()}
    int_columns = []
    int_col_idx = []
    int_col_idx_wrt_num = []
    for i, col_idx in enumerate(num_col_idx):
        col = column_names[col_idx]
        col_data = complete_df.iloc[:,col_idx]
        is_int = (col_data%1 == 0).all()
        if is_int:
            int_columns.append(col)
            int_col_idx.append(name_idx_mapping[col])
            int_col_idx_wrt_num.append(i)
    info['int_col_idx'] = int_col_idx
    info['int_columns'] = int_columns
    info['int_col_idx_wrt_num'] = int_col_idx_wrt_num

    train_df.columns = range(len(train_df.columns))
    test_df.columns = range(len(test_df.columns))
    val_df.columns = range(len(val_df.columns))

    print(name, train_df.shape, val_df.shape, test_df.shape, data_df.shape)

    col_info = {}
    
    for col_idx in num_col_idx:
        col_info[col_idx] = {}
        col_info['type'] = 'numerical'
        col_info['max'] = float(train_df[col_idx].max())
        col_info['min'] = float(train_df[col_idx].min())
     
    for col_idx in cat_col_idx:
        col_info[col_idx] = {}
        col_info['type'] = 'categorical'
        col_info['categorizes'] = list(set(train_df[col_idx]))    

    for col_idx in target_col_idx:
        if info['task_type'] == 'regression':
            col_info[col_idx] = {}
            col_info['type'] = 'numerical'
            col_info['max'] = float(train_df[col_idx].max())
            col_info['min'] = float(train_df[col_idx].min())
        else:
            col_info[col_idx] = {}
            col_info['type'] = 'categorical'
            col_info['categorizes'] = list(set(train_df[col_idx]))      

    info['column_info'] = col_info

    train_df.rename(columns = idx_name_mapping, inplace=True)
    test_df.rename(columns = idx_name_mapping, inplace=True)
    val_df.rename(columns = idx_name_mapping, inplace=True)

    for col in num_columns:
        if (train_df[col] == ' ?').sum() > 0:
            print(col)
            import pdb; pdb.set_trace()
        if (train_df[col] == '?').sum() > 0:
            print(col)
            import pdb; pdb.set_trace()
        train_df.loc[train_df[col] == '?', col] = np.nan
    for col in cat_columns:
        train_df.loc[train_df[col] == '?', col] = 'nan'
    for col in num_columns:
        if (test_df[col] == ' ?').sum() > 0:
            print(col)
            import pdb; pdb.set_trace()
        if (test_df[col] == '?').sum() > 0:
            print(col)
            import pdb; pdb.set_trace()
        test_df.loc[test_df[col] == '?', col] = np.nan
    for col in cat_columns:
        test_df.loc[test_df[col] == '?', col] = 'nan'
    for col in num_columns:
        val_df.loc[val_df[col] == '?', col] = np.nan
    for col in cat_columns:
        val_df.loc[val_df[col] == '?', col] = 'nan'
    
    if train_df.isna().any().any():
        print("Training data contains nan in the numerical cols")
        import pdb; pdb.set_trace()


    
    X_num_train = train_df[num_columns].to_numpy().astype(np.float32)
    X_cat_train = train_df[cat_columns].to_numpy()
    y_train = train_df[target_columns].to_numpy()


    X_num_test = test_df[num_columns].to_numpy().astype(np.float32)
    X_cat_test = test_df[cat_columns].to_numpy()
    y_test = test_df[target_columns].to_numpy()

    X_num_val = val_df[num_columns].to_numpy().astype(np.float32)
    X_cat_val = val_df[cat_columns].to_numpy()
    y_val = val_df[target_columns].to_numpy()
 
    save_dir = f'data/{name}'
    np.save(f'{save_dir}/X_num_train.npy', X_num_train)
    np.save(f'{save_dir}/X_cat_train.npy', X_cat_train)
    np.save(f'{save_dir}/y_train.npy', y_train)

    np.save(f'{save_dir}/X_num_test.npy', X_num_test)
    np.save(f'{save_dir}/X_cat_test.npy', X_cat_test)
    np.save(f'{save_dir}/y_test.npy', y_test)
    
    if has_val:
        np.save(f'{save_dir}/X_num_val.npy', X_num_val)
        np.save(f'{save_dir}/X_cat_val.npy', X_cat_val)
        np.save(f'{save_dir}/y_val.npy', y_val)

    train_df[num_columns] = train_df[num_columns].astype(np.float32)
    test_df[num_columns] = test_df[num_columns].astype(np.float32)
    val_df[num_columns] = val_df[num_columns].astype(np.float32)


    train_df.to_csv(f'{save_dir}/train.csv', index = False)
    test_df.to_csv(f'{save_dir}/test.csv', index = False)
    if has_val:
        val_df.to_csv(f'{save_dir}/val.csv', index = False)

    if not os.path.exists(f'synthetic/{name}'):
        os.makedirs(f'synthetic/{name}')
    
    train_df.to_csv(f'synthetic/{name}/real.csv', index = False)
    test_df.to_csv(f'synthetic/{name}/test.csv', index = False)
    
    if has_val:
        val_df.to_csv(f'synthetic/{name}/val.csv', index = False)

    print('Numerical', X_num_train.shape)
    print('Categorical', X_cat_train.shape)

    info['column_names'] = column_names
    info['train_num'] = train_df.shape[0]
    info['test_num'] = test_df.shape[0]
    info['val_num'] = val_df.shape[0]

    info['idx_mapping'] = idx_mapping
    info['inverse_idx_mapping'] = inverse_idx_mapping
    info['idx_name_mapping'] = idx_name_mapping 

    metadata = {'columns': {}}
    task_type = info['task_type']
    num_col_idx = info['num_col_idx']
    cat_col_idx = info['cat_col_idx']
    target_col_idx = info['target_col_idx']

    for i in num_col_idx:
        metadata['columns'][i] = {}
        metadata['columns'][i]['sdtype'] = 'numerical'
        metadata['columns'][i]['computer_representation'] = 'Float'

    for i in cat_col_idx:
        metadata['columns'][i] = {}
        metadata['columns'][i]['sdtype'] = 'categorical'


    if task_type == 'regression':
        
        for i in target_col_idx:
            metadata['columns'][i] = {}
            metadata['columns'][i]['sdtype'] = 'numerical'
            metadata['columns'][i]['computer_representation'] = 'Float'

    else:
        for i in target_col_idx:
            metadata['columns'][i] = {}
            metadata['columns'][i]['sdtype'] = 'categorical'

    info['metadata'] = metadata

    with open(f'{save_dir}/info.json', 'w') as file:
        json.dump(info, file, indent=4)

    print(f'Processing and Saving {name} Successfully!')

    print(name)
    print('Total', info['train_num'] + info['test_num'])
    print('Train', info['train_num'])
    print('Val', info['val_num'])
    print('Test', info['test_num'])
    if info['task_type'] == 'regression':
        num = len(info['num_col_idx'] + info['target_col_idx'])
        cat = len(info['cat_col_idx'])
    else:
        cat = len(info['cat_col_idx'] + info['target_col_idx'])
        num = len(info['num_col_idx'])
    print('Num', num)
    print('Int', len(info['int_col_idx']))
    print('Cat', cat)


if __name__ == "__main__":

    if args.dataname:
        process_data(args.dataname)
    else:
        for name in [
                'adult', 'default', 'shoppers', 'magic', 'beijing', 'news', 'news_nocat', 'diabetes',
                'adult_dcr',
                'default_dcr',
                'shoppers_dcr',
                'beijing_dcr',
                'news_dcr', 
                'diabetes_dcr'
            ]:    
            process_data(name)

        

