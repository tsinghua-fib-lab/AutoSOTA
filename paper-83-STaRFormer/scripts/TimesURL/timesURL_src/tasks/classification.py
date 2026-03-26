import numpy as np
from . import _eval_protocols as eval_protocols
from sklearn.preprocessing import label_binarize
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm.auto import tqdm


def eval_classification(model, train_data, train_labels, test_data, test_labels, eval_protocol='linear', batch_size: int = None, seed: int=42):
    assert train_labels.ndim == 1 or train_labels.ndim == 2
    # train_data = {'x': ..., 'mask': ...}

    if batch_size is None:    
        train_repr = model.encode(train_data, encoding_window='full_series' if train_labels.ndim == 1 else None)
        test_repr = model.encode(test_data, encoding_window='full_series' if train_labels.ndim == 1 else None)
    else:
        len_train_data = len(train_data['x'])
        r_ = list(range(0, len_train_data, batch_size)) + [len_train_data]
        train_repr = []
        for i in tqdm(range(len(r_)-1), desc="Getting Representation from Training Data"):
            batch = {
                'x': train_data['x'][r_[i]:r_[i+1], ...],
                'mask': train_data['mask'][r_[i]:r_[i+1], ...]
                }
            assert batch['x'].shape[0] <= batch_size
            assert batch['x'].shape[0] == batch['mask'].shape[0]
            train_repr.append(model.encode(batch, encoding_window='full_series' if train_labels.ndim == 1 else None))

        len_test_data = len(test_data['x'])
        r_ = list(range(0, len_test_data, batch_size)) + [len_test_data]
        test_repr = []
        for i in tqdm(range(len(r_)-1), desc="Getting Representation from Training Data"):
            batch = {
                'x': test_data['x'][r_[i]:r_[i+1], ...],
                'mask': test_data['mask'][r_[i]:r_[i+1], ...]
                }
            assert batch['x'].shape[0] <= batch_size
            assert batch['x'].shape[0] == batch['mask'].shape[0]
            test_repr.append(model.encode(batch, encoding_window='full_series' if train_labels.ndim == 1 else None))
        
        train_repr = np.concatenate(train_repr, axis=0)
        test_repr = np.concatenate(test_repr, axis=0)
        

    if eval_protocol == 'linear':
        fit_clf = eval_protocols.fit_lr
    elif eval_protocol == 'svm':
        fit_clf = eval_protocols.fit_svm
    elif eval_protocol == 'knn':
        fit_clf = eval_protocols.fit_knn
    else:
        assert False, 'unknown evaluation protocol'

    def merge_dim01(array):
        return array.reshape(array.shape[0], *array.shape[1]*array.shape[2:])

    if True: # for dkt dataset
        train_repr = train_repr.reshape(train_repr.shape[0], -1)
        train_labels = train_labels.reshape(train_labels.shape[0], -1).ravel()
        test_repr = test_repr.reshape(test_repr.shape[0], -1)
        test_labels = test_labels.reshape(test_labels.shape[0], -1).ravel()

    elif train_labels.ndim == 2:
        train_repr = merge_dim01(train_repr)
        train_labels = merge_dim01(train_labels)
        test_repr = merge_dim01(test_repr)
        test_labels = merge_dim01(test_labels)
    
    print('Fitting Evaluation....')
    clf = fit_clf(train_repr, train_labels, seed=seed)
    print('Scoring Evaluation....')
    acc = clf.score(test_repr, test_labels)
    if eval_protocol == 'linear':
        y_score = clf.predict_proba(test_repr)
    else:
        y_score = clf.decision_function(test_repr)
    test_labels_onehot = label_binarize(test_labels, classes=np.arange(train_labels.max()+1))
    auprc = average_precision_score(test_labels_onehot, y_score)
    print('Done')

    return y_score, { 'acc': acc, 'auprc': auprc}
