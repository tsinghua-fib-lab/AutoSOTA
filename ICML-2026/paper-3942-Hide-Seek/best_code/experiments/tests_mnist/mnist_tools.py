import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import matplotlib.patches as patches

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import repo_paths  # noqa: F401


from tools import save_results_as_pickle, run_feature_selection_model

RANDOM_STATE = 42

def load_mnist_data():
    mnist = fetch_openml('mnist_784', version=1)
    X, y = mnist['data'], mnist['target']
    y = y.astype(int)

    three_eight_idxs = np.where((y==3) | (y==8))[0]

    X_mnist = X.loc[three_eight_idxs].reset_index(drop=True)
    y_mnist = y[three_eight_idxs].reset_index(drop=True)
    print('original data shape:')
    print(X_mnist.shape, y_mnist.shape)
    return X_mnist, y_mnist

def create_train_val_test(X_mnist, y_mnist):

    # binary classification. Turn 3s to 1s and 8s to 0s
    assert len(y_mnist.value_counts()) == 2
    one_hot = (y_mnist.values==3).astype(float) #changed from int
    one_hot = np.vstack((one_hot,1-one_hot)).T #should swap order of stacking in future to align with hide_and_seek y_train.argmax(axis=1) implementation, but leaving as is to match previous experiments. No major impact as balanced classes.

    # 80% train, 20% leftover
    X_train, X_temp, y_train, y_temp = train_test_split(
        np.array(X_mnist), one_hot, test_size=0.2, random_state=RANDOM_STATE
    )

    # 10% val, 10% test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=RANDOM_STATE
    )

    return X_train, X_val, X_test, y_train, y_val, y_test

def report_training_results(results, y_val,
                            verbose=True):
                            
    y_val_pred = results['y_test_pred']
    y_val_pred = (y_val_pred>0.5).astype(int)

    acc = (y_val[:,0] == y_val_pred[:,0]).mean()
    

    mask = results['mask']
    pct_significant = (mask>0.5).mean()
    
    if verbose==True:
        print(f"Accuracy is: {acc*100:.2f}%")
        print(f"Percentage significant masks: {pct_significant*100:.2f}%")

    return acc, pct_significant

def patch_importance(mask, patch_size=4, stride=1):
    from numpy.lib.stride_tricks import sliding_window_view
    if not isinstance(mask, np.ndarray):
        mask = np.array(mask, dtype=np.float32)
    out = sliding_window_view(mask.astype(np.float32), (patch_size, patch_size)).sum(axis=(-2, -1))
    if stride > 1:
        out = out[::stride, ::stride]
    return out

def show_patch_explainer(data_X, data_y, y_preds, mask,
                         n_images=6,
                        num_top_patches=4,
                        patch_size=3,
                        cols=3,
                        return_indices=False,
                        indices=None, #to specify indices
                        save_path=None
                        ):
    
    n_images = n_images
    k = num_top_patches
    patch_size = patch_size
    
    cols = cols
    rows = int(np.ceil(n_images/cols))
    
    # Random indices
    # np.random.seed(2)
    if indices is None:
        indices = np.random.randint(0, data_X.shape[0], size=n_images)
    
    fig, axes = plt.subplots(rows, cols, figsize=(12, rows*3))
    
    for ax, i in zip(axes.flatten(), indices):

        # Original image & mask
        image = data_X[i].reshape(28, 28)
        mask_i = mask[i].reshape(28, 28)
        
        # Compute patch scores
        patch_scores = patch_importance(mask_i, patch_size=patch_size, stride=1)
        
        flat_idx = np.argsort(-patch_scores.ravel())[:k]
        top_patches = np.array(np.unravel_index(flat_idx, patch_scores.shape)).T
        
        # Show digit
        ax.imshow(image, cmap="gray")
        
        # Overlay important patches
        for (r, c) in top_patches:
            rect = patches.Rectangle(
                (c, r), patch_size, patch_size,  # (x, y), width, height
                linewidth=3, edgecolor="red", facecolor="none"
            )
            ax.add_patch(rect)
        
        # Labels
        true_label = 3 if data_y[i][0] == 1 else 8
        pred_label = 3 if y_preds[i][0]>0.5 else 8
        # ax.set_title(f"True: {true_label}, Pred: {pred_label}")
        ax.axis("off")
    print(indices)
    plt.tight_layout()

    if save_path is not None:
        save_path = os.path.expanduser(save_path)
        plt.savefig(save_path, format='pdf', bbox_inches='tight')
        
    plt.show()

    save_dict = {
            "indices": indices,
            "images": [data_X[i].reshape(28, 28) for i in indices],
            "masks": [mask[i].reshape(28, 28) for i in indices],
            "labels": [(3 if data_y[i][0]==1 else 8, 3 if data_y[i][0]==1 else 8) for i in indices],
            "patch_size": patch_size,
            "k": k
        }
    if return_indices == True:
        return save_dict  

def run_mnist_experiment(model_type = 'hide_and_seek',
                         num_important_features = 3, #not used in hide_and_seek and invase
                         pickle_mnist_results_folder=None,
                         lmbda = 0.3,
                            epochs = 500,
                        batch_size = None,
                        xgb_params = None,
                        indices_to_use = None,
                        use_custom_nn_for_lime = False,
                        scale_data = 'min_max', #'min_max' for [0,1] scaling (/255), 'standardize' for StandardScaler, False for no scaling
                        val_or_test = 'test'
                         ):
     
    
    #load data
    X_mnist, y_mnist = load_mnist_data()
    # i=34
    # plt.imshow(X_mnist.iloc[i].values.reshape(28, 28), cmap='gray')
    # plt.title(f"Label: {y_mnist.iloc[i]}")
    # plt.axis('off')
    # plt.show()

    if scale_data == 'min_max':
        X_mnist = X_mnist / 255
        tools_scale_data = False
    elif scale_data == 'standardize':
        tools_scale_data = True  # StandardScaler applied in run_feature_selection_model (fit on train only)
    elif scale_data == False or scale_data is None:
        tools_scale_data = False
    else:
        raise ValueError(f"scale_data must be 'min_max', 'standardize', or False. Got: {scale_data}")

    if val_or_test == 'val':
        #used for validating
        X_train, X_test, X_val, y_train, y_test, y_val = create_train_val_test(X_mnist, y_mnist)
    elif val_or_test == 'test':
        #used for testing. Note as such 'X_test' is our val set and 'X_val' is our test set.
        X_train, X_val, X_test, y_train, y_val, y_test = create_train_val_test(X_mnist, y_mnist)
    else:
        raise ValueError(f"val_or_test must be 'val' or 'test'. Got: {val_or_test}")


    # run model
    
    data_type = 'mnist'
    full_data_dict = {'x_train':X_train,
                     'y_train':y_train,
                     'x_test':X_val,
                     'y_test':y_val,
                     'g_test':None
                     }
    
    results = run_feature_selection_model(
                full_data_dict=full_data_dict,
            lmbda=lmbda,
            epochs=epochs,
            batch_size=batch_size,
            model_type=model_type,
            data_type=data_type,
            folder_for_pickle=None,
            num_important_features=num_important_features,
            xgb_params=xgb_params,
            use_custom_nn_for_lime=use_custom_nn_for_lime,
            scale_data=tools_scale_data)

    #report results
    acc, pct_significant = report_training_results(results, y_val)

    #show patches
    train_val_test = 'val'
    mask = results['mask']
    y_preds = results['y_test_pred']

    if train_val_test == 'train':
        data_X = X_train.copy()
        data_y = y_train.copy()
    elif train_val_test == 'val':
        data_X = X_val.copy()
        data_y = y_val.copy()
    elif train_val_test == 'test':
        data_X = X_test.copy()
        data_y = y_test.copy()
    else:
        raise ValueError(f"train_val_test must be 'train', 'val', or 'test'. Got: {train_val_test}")

    indices = show_patch_explainer(data_X=data_X, 
                        data_y=data_y,
                        y_preds=y_preds,
                        mask=mask,
                            n_images=6,
                            num_top_patches=4,
                            patch_size=3,
                            cols=6,
                            return_indices=True,
                            indices=indices_to_use
                            )
    
    mnist_results = {}
    mnist_results['results']=results
    mnist_results['acc']=acc
    mnist_results['pct_significant']=pct_significant
    mnist_results['indices']=indices
    mnist_results['data_X']=data_X
    mnist_results['data_y']=data_y
    mnist_results['mask']=mask
    mnist_results['scale_data']=scale_data

    if pickle_mnist_results_folder is not None:
        config_str = "_".join(
            str(v) if not isinstance(v, list) else "-".join(map(str, v))
            for v in [train_val_test, lmbda, num_important_features]
        )
        
        save_results_as_pickle(results=mnist_results,
                                syn_type=config_str,
                                model_type=model_type,
                                folder=pickle_mnist_results_folder,
                                name_end='mnist_run',
                                timestamp=results['time_run'])
    return mnist_results

if __name__ == '__main__':
    
    import gc
    import torch

    results_list = []
    
    for lamda in [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]:
        mnist_results = run_mnist_experiment(model_type = 'invase',
                            num_important_features = 3, #not used in hide_and_seek, realx, invase
                            pickle_mnist_results_folder='ICML_experiments/mnist/invase',
                            lmbda = lamda,
                            epochs = 10_000,
                            batch_size = 1_000,
                            scale_data = 'min_max'
                            )
    # for num_important_features in [int(np.round(0.2*28*28)), 
    #                            int(np.round(0.1*28*28)),
    #                            int(np.round(0.5*28*28)),
    #                            int(np.round(0.4*28*28)),
    #                            int(np.round(0.05*28*28))
    #                           ]:
    #     print(num_important_features)
    #     mnist_results = run_mnist_experiment(model_type = 'lime',
    #                         num_important_features = num_important_features, #not used in hide_and_seek, realx, invase
    #                         pickle_mnist_results_folder='ICML_experiments/mnist/lime',
    #                         epochs = 100,
    #                         indices_to_use=None,
    #                         use_custom_nn_for_lime=True,
    #                         batch_size = None,
    #                         scale_data = 'min_max'
    #                         )
        gc.collect()
        torch.cuda.empty_cache()
