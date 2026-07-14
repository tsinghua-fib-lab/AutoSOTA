import numpy as np
from sklearn.decomposition import TruncatedSVD
from pathlib import Path


def load(dataset, path, label_list = [4], PCs = float('inf')):
    # convert HAR (6 classes) to binary classification if needed
    # label_list: e.g. [1, 2, 3]
    # PCs: number of principal components for dimension reduction (HAR binary)
    assert dataset == 'HAR' or dataset == 'HAR_binary'

    X_raw, y_raw = [], []
    
    link, n_class = 'logistic', 6
    
    base = Path(path)
    if (base / 'train' / 'X_train.txt').exists():
        dataset_root = base
    elif (base / 'UCI_HAR_Dataset' / 'train' / 'X_train.txt').exists():
        dataset_root = base / 'UCI_HAR_Dataset'
    elif (base / 'HAR Data' / 'UCI HAR Dataset' / 'train' / 'X_train.txt').exists():
        dataset_root = base / 'HAR Data' / 'UCI HAR Dataset'
    else:
        raise FileNotFoundError(
            f"Could not locate the UCI HAR dataset under {base}. "
            "Expected `UCI_HAR_Dataset/train/X_train.txt`."
        )
    
    X_train = np.loadtxt(dataset_root / 'train' / 'X_train.txt')
    idx_train = np.loadtxt(dataset_root / 'train' / 'subject_train.txt')
    y_train = np.loadtxt(dataset_root / 'train' / 'y_train.txt')

    X_test = np.loadtxt(dataset_root / 'test' / 'X_test.txt')
    idx_test = np.loadtxt(dataset_root / 'test' / 'subject_test.txt')
    y_test = np.loadtxt(dataset_root / 'test' / 'y_test.txt')

    X_all = np.concatenate((X_train, X_test))
    
    # Dimension reduction by PCA
    d = X_all.shape[1]
    r = min(PCs, d)
    if r < d:
        svd = TruncatedSVD(n_components = r, n_iter = 10, random_state = 0)
        svd.fit(X_all)
        X_all = svd.transform(X_all)

    y_all = np.concatenate((y_train, y_test))
    idx_raw = np.concatenate((idx_train, idx_test))
    idx_raw = idx_raw.astype(int)

    ## prepare data
    m = max(idx_raw)
    tmp = [[] for _ in range(m)]
    for (i, j) in enumerate(idx_raw):
        tmp[j - 1].append(i)

    for j in range(m):
        X_raw.append( X_all[tmp[j]] )
        y_raw.append( y_all[tmp[j]].astype(int) - 1 )

    # convert to binary classification if needed
    # 6 classes: 1 WALKING, 2 WALKING_UPSTAIRS 3 WALKING_DOWNSTAIRS, 4 SITTING, 5 STANDING, 6 LAYING
    if dataset == 'HAR_binary':
        n_class = 2 # binary classification
        for j in range(m):
            y_raw[j] = np.array([q + 1 in label_list for q in y_raw[j]]).astype(int)     

    return [[X_raw, y_raw], link, n_class]


def split(data, prop = None, seed = 1000):
    # train-test split
    [X_raw, y_raw] = data
    np.random.seed(seed)
    m = len(X_raw)
    if prop is None:
        prop = 0.2 * np.ones(m)
    X_train, X_test = [], []
    y_train, y_test = [], []
    for j in range(m):
        n_j = len(y_raw[j])
        idx_test = np.random.choice(n_j, int(prop[j] * n_j), replace = False)
        X_test.append(X_raw[j][idx_test])
        y_test.append(y_raw[j][idx_test])
        idx_train = np.delete(np.array(range(n_j)), idx_test)
        X_train.append(X_raw[j][idx_train])
        y_train.append(y_raw[j][idx_train])
    data_train = [X_train, y_train]
    data_test = [X_test, y_test]
    return [data_train, data_test]


def split_cv(n_list, n_fold = 5, seed = 1000):
    # train-test split for cross-validation
    np.random.seed(seed)
    m = len(n_list)
    ans = []
    for j in range(m):
        n_j = n_list[j]
        perm = np.random.permutation(n_j)
        q, r = int(n_j / n_fold), n_j % n_fold
        ans_j = []
        for k in range(n_fold):           
            if k < r:
                tmp = [perm[i * n_fold + k] for i in range(q + 1)]
            else:
                tmp = [perm[i * n_fold + k] for i in range(q)]
            ans_j.append(tmp)
        ans.append(ans_j)
    return ans


def MTL_preprocessing(data, link = 'linear', intercept = True, n_class = 1, standardization = True):
    # standardization of data
    m = len(data[0])
    d = data[0][0].shape[1]
    n_list = np.zeros(m).astype(int)

    if not standardization:
        X_means, X_stds = np.zeros((d, 1)), np.ones((d, 1))
        if intercept:
            X_means = np.vstack((np.zeros((1, 1)), X_means))
            X_stds = np.vstack((np.ones((1, 1)), X_stds))
        y_mean, y_std = 0, 1
        X, Y = [], []        
        for j in range(m):
            # load X
            tmp = data[0][j]
            n_list[j] = tmp.shape[0]
            if intercept:
                tmp = np.hstack((np.ones((n_list[j], 1)), tmp))
            X.append(tmp)

        # load y
        if link == 'linear':
            d_out = 1
            for y in data[1]:
                Y.append(y.reshape(-1, 1))
        if link == 'logistic':
            if n_class == 2:
                d_out = 1
                for y in data[1]:
                    Y.append(y.reshape(-1, 1))
            else: # n_class > 2, use one-hoc encoding
                d_out = n_class
                for y in data[1]:
                    rows = np.arange(y.shape[0])
                    tmp = np.zeros((y.shape[0], n_class))
                    tmp[rows, y.reshape(-1,)] = 1
                    Y.append(tmp)
        return [X, Y, X_means, X_stds, y_mean, y_std, n_list, d_out]

    # with standardization (default)
    tmp1 = np.zeros((m, d))
    tmp2 = np.zeros((m, d))
    for j in range(m):
        tmp = data[0][j]
        n_list[j] = tmp.shape[0]
        tmp1[j] = np.mean(tmp, axis = 0)
        tmp2[j] = np.mean(tmp ** 2, axis = 0)
    n_list = n_list.astype(int)
    N = np.sum(n_list)

    # means
    if intercept:
        X_means = tmp1.T @ n_list.reshape(-1, 1) / N # d-by-1
    else: # no intercept, no centering
        X_means = np.zeros((d, 1))
    # standard deviations
    X_stds = np.sqrt( tmp2.T @ n_list.reshape(-1, 1) / N - X_means ** 2 )
    X_stds = np.clip(X_stds, a_min = 1e-5, a_max = None) # avoid dividing by zero

    # Standardize features
    X = []
    for j in range(m):
        tmp = data[0][j]
        tmp = (tmp - X_means.T) / X_stds.T
        if intercept: # add an all-one column
            tmp = np.hstack((np.ones((n_list[j], 1)), tmp))
        X.append(tmp)
    if intercept:
        X_means = np.vstack((np.zeros((1, 1)), X_means))
        X_stds = np.vstack((np.ones((1, 1)), X_stds))

    Y = []
    if link == 'linear':
        d_out = 1
        tmp1 = np.zeros(m)
        tmp2 = np.zeros(m)
        for (j, y) in enumerate(data[1]):
            tmp1[j] = np.mean(y)
            tmp2[j] = np.mean(y ** 2)

        if intercept: # standardize y
            # means
            y_mean = np.sum(tmp1 * n_list) / N
            # standard deviations
            y_std = np.sqrt( np.sum(tmp2 * n_list) / N - y_mean ** 2 )

            # standardize y
            for y in data[1]:
                tmp = (y - y_mean) / y_std
                Y.append(tmp.reshape(-1, 1))
        else:
            y_mean, y_std = 0, 1
            for y in data[1]:
                Y.append(y.reshape(-1, 1))

    if link == 'logistic':
        y_mean, y_std = 0, 1
        if n_class == 2:
            d_out = 1
            for y in data[1]:
                Y.append(y.reshape(-1, 1))
        else: # n_class > 2, use one-hoc encoding
            d_out = n_class
            for y in data[1]:
                rows = np.arange(y.shape[0])
                tmp = np.zeros((y.shape[0], n_class))
                tmp[rows, y.reshape(-1,)] = 1
                Y.append(tmp)
    return [X, Y, X_means, X_stds, y_mean, y_std, n_list, d_out]
