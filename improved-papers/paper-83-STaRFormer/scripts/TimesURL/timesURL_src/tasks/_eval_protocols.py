import numpy as np
from sklearn.linear_model import Ridge
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV, train_test_split

def fit_svm(features, y, MAX_SAMPLES=10000, seed=None):
    nb_classes = np.unique(y, return_counts=True)[1].shape[0]
    train_size = features.shape[0]
    random_state = seed if seed is not None else 42
    # stick with 100
    svm = SVC(C=100, gamma='scale', verbose=True, max_iter=10000, kernel='rbf', 
              degree=3, coef0=0, shrinking=True, probability=False, tol=0.001, 
              cache_size=200, class_weight=None, random_state=random_state, 
              decision_function_shape='ovr')
    if False: #train_size // nb_classes < 5 or train_size < 50: # choose to always to soft margin classifaction setting C=0.1
        print('No Grid Search!')
        return svm.fit(features, y)
    else:
        print('Grid Search!')
        grid_search = GridSearchCV(
            svm, {
                'C': [
                    #0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000,
                    0.01, 0.1, 1, 10, 100, # 1000, 10000,
                    #float(np.inf)
                ],
                'kernel': ['rbf'],
                'degree': [3],
                'gamma': ['scale'],
                'coef0': [0],
                'shrinking': [True],
                'probability': [False],
                'tol': [0.001],
                'cache_size': [200],
                'class_weight': [None],
                'verbose': [False],
                'max_iter':  [10000], # [10000000],
                'decision_function_shape': ['ovr'],
                'random_state': [None]
            },
            cv=2, n_jobs=5, verbose=3
        )
        # If the training set is too large, subsample MAX_SAMPLES examples
        if train_size > MAX_SAMPLES:
            split = train_test_split(
                features, y,
                train_size=MAX_SAMPLES, random_state=0, stratify=y
            )
            features = split[0]
            y = split[2]
            
        grid_search.fit(features, y)
        return grid_search.best_estimator_

def fit_lr(features, y, MAX_SAMPLES=100000, **kwargs):
    # If the training set is too large, subsample MAX_SAMPLES examples
    if features.shape[0] > MAX_SAMPLES:
        split = train_test_split(
            features, y,
            train_size=MAX_SAMPLES, random_state=0, stratify=y
        )
        features = split[0]
        y = split[2]
        
    pipe = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            random_state=0,
            max_iter=1000000,
            multi_class='ovr',
            verbose=1,
        )
    )
    pipe.fit(features, y)
    return pipe

def fit_knn(features, y, **kwargs):
    pipe = make_pipeline(
        StandardScaler(),
        KNeighborsClassifier(n_neighbors=1)
    )
    pipe.fit(features, y)
    return pipe

def fit_ridge(train_features, train_y, valid_features, valid_y, MAX_SAMPLES=100000, **kwargs):
    # If the training set is too large, subsample MAX_SAMPLES examples
    if train_features.shape[0] > MAX_SAMPLES:
        split = train_test_split(
            train_features, train_y,
            train_size=MAX_SAMPLES, random_state=0
        )
        train_features = split[0]
        train_y = split[2]
    if valid_features.shape[0] > MAX_SAMPLES:
        split = train_test_split(
            valid_features, valid_y,
            train_size=MAX_SAMPLES, random_state=0
        )
        valid_features = split[0]
        valid_y = split[2]
    
    alphas = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
    valid_results = []
    for alpha in alphas:
        lr = Ridge(alpha=alpha).fit(train_features, train_y)
        valid_pred = lr.predict(valid_features)
        score = np.sqrt(((valid_pred - valid_y) ** 2).mean()) + np.abs(valid_pred - valid_y).mean()
        valid_results.append(score)
    best_alpha = alphas[np.argmin(valid_results)]
    
    lr = Ridge(alpha=best_alpha)
    lr.fit(train_features, train_y)
    return lr
