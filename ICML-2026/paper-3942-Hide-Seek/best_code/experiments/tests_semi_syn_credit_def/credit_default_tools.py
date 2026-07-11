import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from datetime import datetime
import gc

from pathlib import Path

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import repo_paths  # noqa: F401
from tools import run_feature_selection_model, parse_args

RANDOM_STATE = 0

def load_data(return_y=False):
    # -----------------------------
    # Load data
    # -----------------------------
    # directory containing credit_default_tools.py
    this_dir = Path(__file__).resolve().parent
    file_path = this_dir / "default of credit card clients.xls"   # assumes the .xls is in tests_credit_default/

    df = pd.read_excel(file_path, header=1)

    # -----------------------------
    # Define X and y
    # -----------------------------
    target_col = "default payment next month"
    
    X = df.drop(columns=[target_col])
    X = X.drop(columns = 'ID')
    
    col_order = ['LIMIT_BAL', 'BILL_AMT1', 'BILL_AMT2',
           'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1',
           'PAY_AMT2', 'PAY_AMT3', 'AGE', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6', 'SEX', 'EDUCATION', 'MARRIAGE', 'PAY_0', 'PAY_2',
           'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
    
    X = X[col_order]
    y = df[target_col].astype(int)
    
    # -----------------------------
    # Train / Val / Test split
    # 80% train, 10% val, 10% test
    # -----------------------------

    # First split: 80% train, 20% temp
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,          # important for class balance
        random_state=RANDOM_STATE
    )
    
    # Second split: split temp into 10% val, 10% test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.5,
        stratify=y_temp,
        random_state=RANDOM_STATE
    )
    
     # -----------------------------
    # Standard scaling (FIT ON TRAIN ONLY)
    # -----------------------------
    scaler = StandardScaler()

    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )

    X_val_scaled = pd.DataFrame(
        scaler.transform(X_val),
        columns=X_val.columns,
        index=X_val.index
    )

    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )

    # -----------------------------
    # Sanity checks
    # -----------------------------
    print("Shapes:")
    print("Train:", X_train_scaled.shape, y_train.shape)
    print("Val:  ", X_val_scaled.shape, y_val.shape)
    print("Test: ", X_test_scaled.shape, y_test.shape)

    print("\nDefault rate:")
    print("Train:", y_train.mean())
    print("Val:  ", y_val.mean())
    print("Test: ", y_test.mean())

    print("\nTrain means (≈0):")
    print(X_train_scaled.mean().round(3).head())

    print("\nTrain stds (≈1):")
    print(X_train_scaled.std().round(3).head())

    if return_y == False:
        return X_train_scaled, X_val_scaled, X_test_scaled#, y_train, y_val, y_test
    else:
        return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test

def stack_y(y):
    """ Stack binary y into two-column format """
    y = y.values.reshape(1,-1)
    y = np.vstack([1-y, y]).T #have now applied correct stacking order
    assert (y.sum(axis=1)==1).all()
    return y

if __name__ == '__main__':
    val_or_test = 'test'
    if val_or_test == 'val': #to match the process we did for syn data, using test for val and val for test.
        X_train, _, X_test, y_train, _, y_test = load_data(return_y=True)
    elif val_or_test == 'test':
        X_train, X_test, _, y_train, y_test, _ = load_data(return_y=True)

    y_train = stack_y(y_train)
    y_test = stack_y(y_test)
    
    y_train = y_train.astype("float32")
    y_test  = y_test.astype("float32")

    full_data_dict = {'x_train':X_train.values,
                     'y_train':y_train,
                     'x_test':X_test.values,
                     'y_test':y_test,
                     'g_test':None
                     }


    args = parse_args(lmbda_default=0.1, 
                      epochs_default=10_000, 
                      batch_size_default=1_000,
                     seed_default=0, 
                     syn_idx_default=3, 
                     num_syn_features_default=11,
                     train_N_default=10_000, 
                     test_N_default=10_000,
                     hide_hidden_dim_default=32, 
                     seek_hidden_dim_default=32,
                     hide_num_hidden_layers_default=2, 
                     seek_num_hidden_layers_default=2,
                     lmbda_exponent_default=2,
                     data_mode_default='not relevant')
                     #'credit_data_val', #'synthetic')
    
    lmbda = args.lmbda
    batch_size = args.batch_size
    epochs = args.epochs

    seed = args.seed
    syn_idx = args.syn_idx
    num_syn_features = args.num_syn_features
    train_N = args.train_N
    test_N = args.test_N

    data_mode = args.data_mode

    hide_hidden_dim = args.hide_hidden_dim
    seek_hidden_dim = args.seek_hidden_dim
    hide_num_hidden_layers = args.hide_num_hidden_layers
    seek_num_hidden_layers = args.seek_num_hidden_layers
    lmbda_exponent = args.lmbda_exponent
    
    batchnorm_hs = args.batchnorm_hs
    
    # args.return_losses_on_val = True #to edit manually

    return_losses_on_val = args.return_losses_on_val
    
    print(f"Running with seed={seed}, lmbda={lmbda}")

    single_data_set=True

    # data_sets = ['Syn1','Syn2','Syn3','Syn4','Syn5','Syn6', 'Syn7']
    folder_for_pickle = 'ICML_experiments/semi_syn_cred/after_tuning_lambda_true_y_true_stack'
    model_type = "hide_and_seek"
    task = 'classification'
    save_experiment_data = True
    num_important_features = 'none' #not needed for hide&seek, invase, realx. Will fail for other models.
    data_mode = 'none'
    return_results=True
    
    if model_type == 'invase' and (epochs != 10_000 or batch_size != 1_000 or lmbda != 0.05): raise ValueError("invase usually requires lambda=0.05 epochs=10_000, batch_size=1_000")
    if model_type == 'hide_and_seek' and (epochs != 500 or batch_size != None or lmbda != 0.1): raise ValueError("hide_and_seek usually requires lambda=0.1, epochs=500, batch_size=None")
    if model_type == 'lime' and (epochs != 500 or batch_size != None): raise ValueError("lime baseline usually requires uses epochs=500, batch_size=None")
    if model_type == 'realx' and (epochs != 500 or batch_size != 1_000 or lmbda != 0.05): raise ValueError("realx baseline usually requires lambda=0.05 epochs=500, batch_size=1000")
    if num_syn_features != 11:
        print("WARNING NOT TESTING ON STANDARD NUMBER OF SYN FEATURES")
    
    if single_data_set == True:
        data_set = 'credit_data_real_y'
        results = run_feature_selection_model(data_type=data_set, 
                            folder_for_pickle=folder_for_pickle,
                            num_important_features=num_important_features,
                            full_data_dict=full_data_dict,
                            model_type=model_type,
                            batch_size=batch_size,
                            epochs=epochs,
                            lmbda=lmbda,
                            task=task,
                            hide_hidden_dim=hide_hidden_dim, 
                            seek_hidden_dim=seek_hidden_dim, 
                            hide_num_hidden_layers=hide_num_hidden_layers, 
                                seek_num_hidden_layers=seek_num_hidden_layers,
                                train_N = train_N,
                                test_N = test_N,
                                seed=seed,
                                num_syn_features=num_syn_features,
                                return_results=return_results,
                                batchnorm_hs=batchnorm_hs,
                                save_experiment_data=save_experiment_data,
                                lmbda_exponent=lmbda_exponent,
                                return_losses_on_val=return_losses_on_val,
                                data_mode=data_mode
                                )
        print(results['accuracy'])
        print(results['roc_auc'])
        print(results['pr_auc'])

    gc.collect()
    if model_type != 'l2x' and model_type != 'realx':
        import torch
        torch.cuda.empty_cache()
