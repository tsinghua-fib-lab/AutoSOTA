
# DKT 
# CUDA_VISIBLE_DEVICES=0 python scripts/TimesURL/train_timesURL.py dkt dkt_run  --loader dkt --epochs 30 --batch_size 512 --eval --num_train 10000 --num_test 10000 --instance 2024-10-16_22:03:27
# Training time: 2:36:20.763222
#Training time: 2:36:20.763222
#
#x (10000, 256, 8) (10000, 256, 8)
#mask (10000, 256, 8) (10000, 256, 8)
#Getting Representation from Training Data: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████| 625/625 [00:02<00:00, 223.92it/s]
#Getting Representation from Training Data: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████| 625/625 [00:02<00:00, 224.24it/s]
#Fitting Evaluation....
#Grid Search!
#Fitting 5 folds for each of 5 candidates, totalling 25 fits
#/home/ubuntu/.pyenv/versions/fl-3.10.14/lib/python3.10/site-packages/sklearn/svm/_base.py:297: ConvergenceWarning: Solver terminated early (max_iter=10000).  Consider pre-processing your data with StandardScaler or MinMaxScaler.
#  warnings.warn(
#/home/ubuntu/.pyenv/versions/fl-3.10.14/lib/python3.10/site-packages/sklearn/svm/_base.py:297: ConvergenceWarning: Solver terminated early (max_iter=10000).  Consider pre-processing your data with StandardScaler or MinMaxScaler.
#  warnings.warn(
#/home/ubuntu/.pyenv/versions/fl-3.10.14/lib/python3.10/site-packages/sklearn/svm/_base.py:297: ConvergenceWarning: Solver terminated early (max_iter=10000).  Consider pre-processing your data with StandardScaler or MinMaxScaler.
#  warnings.warn(
#/home/ubuntu/.pyenv/versions/fl-3.10.14/lib/python3.10/site-packages/sklearn/svm/_base.py:297: ConvergenceWarning: Solver terminated early (max_iter=10000).  Consider pre-processing your data with StandardScaler or MinMaxScaler.
#  warnings.warn(
#/home/ubuntu/.pyenv/versions/fl-3.10.14/lib/python3.10/site-packages/sklearn/svm/_base.py:297: ConvergenceWarning: Solver terminated early (max_iter=10000).  Consider pre-processing your data with StandardScaler or MinMaxScaler.
#  warnings.warn(
#/home/ubuntu/.pyenv/versions/fl-3.10.14/lib/python3.10/site-packages/sklearn/svm/_base.py:297: ConvergenceWarning: Solver terminated early (max_iter=10000).  Consider pre-processing your data with StandardScaler or MinMaxScaler.
#  warnings.warn(
#/home/ubuntu/.pyenv/versions/fl-3.10.14/lib/python3.10/site-packages/sklearn/svm/_base.py:297: ConvergenceWarning: Solver terminated early (max_iter=10000).  Consider pre-processing your data with StandardScaler or MinMaxScaler.
#  warnings.warn(
#/home/ubuntu/.pyenv/versions/fl-3.10.14/lib/python3.10/site-packages/sklearn/svm/_base.py:297: ConvergenceWarning: Solver terminated early (max_iter=10000).  Consider pre-processing your data with StandardScaler or MinMaxScaler.
#  warnings.warn(
#/home/ubuntu/.pyenv/versions/fl-3.10.14/lib/python3.10/site-packages/sklearn/svm/_base.py:297: ConvergenceWarning: Solver terminated early (max_iter=10000).  Consider pre-processing your data with StandardScaler or MinMaxScaler.
#  warnings.warn(
#/home/ubuntu/.pyenv/versions/fl-3.10.14/lib/python3.10/site-packages/sklearn/svm/_base.py:297: ConvergenceWarning: Solver terminated early (max_iter=10000).  Consider pre-processing your data with StandardScaler or MinMaxScaler.
#  warnings.warn(
#/home/ubuntu/.pyenv/versions/fl-3.10.14/lib/python3.10/site-packages/sklearn/svm/_base.py:297: ConvergenceWarning: Solver terminated early (max_iter=10000).  Consider pre-processing your data with StandardScaler or MinMaxScaler.
#  warnings.warn(
#Scoring Evaluation....
#Done
#Evaluation result: {'acc': 0.7272, 'auprc': 0.7918707808063523}
#\Evaluation time: 14:26:40.639271

# CUDA_VISIBLE_DEVICES=1 python scripts/TimesURL/train_timesURL.py dkt dkt_run  --loader dkt --epochs 30 --batch_size 512 --eval --num_train 10000 --num_test 10000 --instance 2024-10-28_14:34:17 --seed 123

#Training time: 2:31:26.137686
#
#x (10000, 256, 8) (10000, 256, 8)
#mask (10000, 256, 8) (10000, 256, 8)
#Getting Representation from Training Data: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████| 625/625 [00:02<00:00, 221.20it/s]
#Getting Representation from Training Data: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████| 625/625 [00:02<00:00, 223.30it/s]
#Fitting Evaluation....
#Grid Search!
#Fitting 2 folds for each of 5 candidates, totalling 10 fits
#[CV 1/2] END C=1, cache_size=200, class_weight=None, coef0=0, decision_function_shape=ovr, degree=3, gamma=scale, kernel=rbf, max_iter=10000, probability=False, random_state=None, shrinking=True, tol=0.001, verbose=False;, score=0.702 total time=79.0min
#[CV 1/2] END C=0.1, cache_size=200, class_weight=None, coef0=0, decision_function_shape=ovr, degree=3, gamma=scale, kernel=rbf, max_iter=10000, probability=False, random_state=None, shrinking=True, tol=0.001, verbose=False;, score=0.674 total time=84.2min
#[CV 2/2] END C=0.1, cache_size=200, class_weight=None, coef0=0, decision_function_shape=ovr, degree=3, gamma=scale, kernel=rbf, max_iter=10000, probability=False, random_state=None, shrinking=True, tol=0.001, verbose=False;, score=0.668 total time=88.4min
#[CV 1/2] END C=0.01, cache_size=200, class_weight=None, coef0=0, decision_function_shape=ovr, degree=3, gamma=scale, kernel=rbf, max_iter=10000, probability=False, random_state=None, shrinking=True, tol=0.001, verbose=False;, score=0.612 total time=103.6min
#/home/ubuntu/.pyenv/versions/fl-3.10.14/lib/python3.10/site-packages/sklearn/svm/_base.py:297: ConvergenceWarning: Solver terminated early (max_iter=10000).  Consider pre-processing your data with StandardScaler or MinMaxScaler.
#  warnings.warn(
#[CV 2/2] END C=0.01, cache_size=200, class_weight=None, coef0=0, decision_function_shape=ovr, degree=3, gamma=scale, kernel=rbf, max_iter=10000, probability=False, random_state=None, shrinking=True, tol=0.001, verbose=False;, score=0.616 total time=119.0min
#/home/ubuntu/.pyenv/versions/fl-3.10.14/lib/python3.10/site-packages/sklearn/svm/_base.py:297: ConvergenceWarning: Solver terminated early (max_iter=10000).  Consider pre-processing your data with StandardScaler or MinMaxScaler.
#  warnings.warn(
#[CV 2/2] END C=1, cache_size=200, class_weight=None, coef0=0, decision_function_shape=ovr, degree=3, gamma=scale, kernel=rbf, max_iter=10000, probability=False, random_state=None, shrinking=True, tol=0.001, verbose=False;, score=0.692 total time=68.2min
#[CV 2/2] END C=10, cache_size=200, class_weight=None, coef0=0, decision_function_shape=ovr, degree=3, gamma=scale, kernel=rbf, max_iter=10000, probability=False, random_state=None, shrinking=True, tol=0.001, verbose=False;, score=0.707 total time=65.9min
#[CV 1/2] END C=10, cache_size=200, class_weight=None, coef0=0, decision_function_shape=ovr, degree=3, gamma=scale, kernel=rbf, max_iter=10000, probability=False, random_state=None, shrinking=True, tol=0.001, verbose=False;, score=0.714 total time=77.1min
#[CV 1/2] END C=100, cache_size=200, class_weight=None, coef0=0, decision_function_shape=ovr, degree=3, gamma=scale, kernel=rbf, max_iter=10000, probability=False, random_state=None, shrinking=True, tol=0.001, verbose=False;, score=0.722 total time=63.1min
#[CV 2/2] END C=100, cache_size=200, class_weight=None, coef0=0, decision_function_shape=ovr, degree=3, gamma=scale, kernel=rbf, max_iter=10000, probability=False, random_state=None, shrinking=True, tol=0.001, verbose=False;, score=0.707 total time=52.0min
#/home/ubuntu/.pyenv/versions/fl-3.10.14/lib/python3.10/site-packages/sklearn/svm/_base.py:297: ConvergenceWarning: Solver terminated early (max_iter=10000).  Consider pre-processing your data with StandardScaler or MinMaxScaler.
#  warnings.warn(
#Scoring Evaluation....
#Done
#Evaluation result: {'acc': 0.7261, 'auprc': 0.7903955768060431}
#\Evaluation time: 10:43:17.209653
#
#Finished.

# CUDA_VISIBLE_DEVICES=1 python scripts/TimesURL/train_timesURL.py dkt dkt_run  --loader dkt --epochs 30 --batch_size 512 --eval --num_train 10000 --num_test 10000 --instance 2024-10-28_17:57:21  --seed 0

#Training time: 2:32:55.214060
#
#x (10000, 256, 8) (10000, 256, 8)
#mask (10000, 256, 8) (10000, 256, 8)
#Getting Representation from Training Data: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████| 625/625 [00:02<00:00, 222.42it/s]
#Getting Representation from Training Data: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████| 625/625 [00:02<00:00, 224.54it/s]
#Fitting Evaluation....
#Grid Search!
#Fitting 2 folds for each of 5 candidates, totalling 10 fits
#[CV 1/2] END C=1, cache_size=200, class_weight=None, coef0=0, decision_function_shape=ovr, degree=3, gamma=scale, kernel=rbf, max_iter=10000, probability=False, random_state=None, shrinking=True, tol=0.001, verbose=False;, score=0.687 total time=81.0min
#[CV 1/2] END C=0.01, cache_size=200, class_weight=None, coef0=0, decision_function_shape=ovr, degree=3, gamma=scale, kernel=rbf, max_iter=10000, probability=False, random_state=None, shrinking=True, tol=0.001, verbose=False;, score=0.602 total time=106.5min
#[CV 1/2] END C=0.1, cache_size=200, class_weight=None, coef0=0, decision_function_shape=ovr, degree=3, gamma=scale, kernel=rbf, max_iter=10000, probability=False, random_state=None, shrinking=True, tol=0.001, verbose=False;, score=0.663 total time=106.5min
#[CV 2/2] END C=0.1, cache_size=200, class_weight=None, coef0=0, decision_function_shape=ovr, degree=3, gamma=scale, kernel=rbf, max_iter=10000, probability=False, random_state=None, shrinking=True, tol=0.001, verbose=False;, score=0.658 total time=117.6min
#/home/ubuntu/.pyenv/versions/fl-3.10.14/lib/python3.10/site-packages/sklearn/svm/_base.py:297: ConvergenceWarning: Solver terminated early (max_iter=10000).  Consider pre-processing your data with StandardScaler or MinMaxScaler.
#  warnings.warn(
#[CV 2/2] END C=0.01, cache_size=200, class_weight=None, coef0=0, decision_function_shape=ovr, degree=3, gamma=scale, kernel=rbf, max_iter=10000, probability=False, random_state=None, shrinking=True, tol=0.001, verbose=False;, score=0.604 total time=126.0min
#/home/ubuntu/.pyenv/versions/fl-3.10.14/lib/python3.10/site-packages/sklearn/svm/_base.py:297: ConvergenceWarning: Solver terminated early (max_iter=10000).  Consider pre-processing your data with StandardScaler or MinMaxScaler.
#  warnings.warn(
#[CV 2/2] END C=1, cache_size=200, class_weight=None, coef0=0, decision_function_shape=ovr, degree=3, gamma=scale, kernel=rbf, max_iter=10000, probability=False, random_state=None, shrinking=True, tol=0.001, verbose=False;, score=0.691 total time=87.0min
#[CV 1/2] END C=10, cache_size=200, class_weight=None, coef0=0, decision_function_shape=ovr, degree=3, gamma=scale, kernel=rbf, max_iter=10000, probability=False, random_state=None, shrinking=True, tol=0.001, verbose=False;, score=0.713 total time=65.9min
#[CV 1/2] END C=100, cache_size=200, class_weight=None, coef0=0, decision_function_shape=ovr, degree=3, gamma=scale, kernel=rbf, max_iter=10000, probability=False, random_state=None, shrinking=True, tol=0.001, verbose=False;, score=0.720 total time=71.6min
#[CV 2/2] END C=10, cache_size=200, class_weight=None, coef0=0, decision_function_shape=ovr, degree=3, gamma=scale, kernel=rbf, max_iter=10000, probability=False, random_state=None, shrinking=True, tol=0.001, verbose=False;, score=0.714 total time=85.1min
#[CV 2/2] END C=100, cache_size=200, class_weight=None, coef0=0, decision_function_shape=ovr, degree=3, gamma=scale, kernel=rbf, max_iter=10000, probability=False, random_state=None, shrinking=True, tol=0.001, verbose=False;, score=0.716 total time=67.8min
#/home/ubuntu/.pyenv/versions/fl-3.10.14/lib/python3.10/site-packages/sklearn/svm/_base.py:297: ConvergenceWarning: Solver terminated early (max_iter=10000).  Consider pre-processing your data with StandardScaler or MinMaxScaler.
#  warnings.warn(
#Scoring Evaluation....
#Done
#Evaluation result: {'acc': 0.7244, 'auprc': 0.786529284871008}
#\Evaluation time: 11:02:36.598744


# CUDA_VISIBLE_DEVICES=0 python scripts/TimesURL/train_timesURL.py dkt dkt_run  --loader dkt --epochs 30 --batch_size 512 --eval --num_train 10000 --num_test 10000 --instance 2024-11-08_15:27:47 --seed 63

#Training time: 2:32:48.632487
#
#x (10000, 256, 8) (10000, 256, 8)
#mask (10000, 256, 8) (10000, 256, 8)
#Getting Representation from Training Data: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████| 625/625 [00:02<00:00, 222.98it/s]
#Getting Representation from Training Data: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████| 625/625 [00:02<00:00, 224.66it/s]
#Fitting Evaluation....
#Grid Search!
#Fitting 2 folds for each of 5 candidates, totalling 10 fits
#[CV 2/2] END C=0.1, cache_size=200, class_weight=None, coef0=0, decision_function_shape=ovr, degree=3, gamma=scale, kernel=rbf, max_iter=10000, probability=False, random_state=None, shrinking=True, tol=0.001, verbose=False;, score=0.659 total time=90.3min
#[CV 1/2] END C=0.1, cache_size=200, class_weight=None, coef0=0, decision_function_shape=ovr, degree=3, gamma=scale, kernel=rbf, max_iter=10000, probability=False, random_state=None, shrinking=True, tol=0.001, verbose=False;, score=0.663 total time=98.9min
#[CV 1/2] END C=1, cache_size=200, class_weight=None, coef0=0, decision_function_shape=ovr, degree=3, gamma=scale, kernel=rbf, max_iter=10000, probability=False, random_state=None, shrinking=True, tol=0.001, verbose=False;, score=0.690 total time=103.9min
#[CV 2/2] END C=0.01, cache_size=200, class_weight=None, coef0=0, decision_function_shape=ovr, degree=3, gamma=scale, kernel=rbf, max_iter=10000, probability=False, random_state=None, shrinking=True, tol=0.001, verbose=False;, score=0.603 total time=106.5min
#/home/ubuntu/.pyenv/versions/fl-3.10.14/lib/python3.10/site-packages/sklearn/svm/_base.py:297: ConvergenceWarning: Solver terminated early (max_iter=10000).  Consider pre-processing your data with StandardScaler or MinMaxScaler.
#  warnings.warn(
#[CV 1/2] END C=0.01, cache_size=200, class_weight=None, coef0=0, decision_function_shape=ovr, degree=3, gamma=scale, kernel=rbf, max_iter=10000, probability=False, random_state=None, shrinking=True, tol=0.001, verbose=False;, score=0.619 total time=125.8min
#/home/ubuntu/.pyenv/versions/fl-3.10.14/lib/python3.10/site-packages/sklearn/svm/_base.py:297: ConvergenceWarning: Solver terminated early (max_iter=10000).  Consider pre-processing your data with StandardScaler or MinMaxScaler.
#  warnings.warn(
#[CV 1/2] END C=100, cache_size=200, class_weight=None, coef0=0, decision_function_shape=ovr, degree=3, gamma=scale, kernel=rbf, max_iter=10000, probability=False, random_state=None, shrinking=True, tol=0.001, verbose=False;, score=0.699 total time=62.6min
#[CV 2/2] END C=1, cache_size=200, class_weight=None, coef0=0, decision_function_shape=ovr, degree=3, gamma=scale, kernel=rbf, max_iter=10000, probability=False, random_state=None, shrinking=True, tol=0.001, verbose=False;, score=0.688 total time=80.9min
#[CV 2/2] END C=10, cache_size=200, class_weight=None, coef0=0, decision_function_shape=ovr, degree=3, gamma=scale, kernel=rbf, max_iter=10000, probability=False, random_state=None, shrinking=True, tol=0.001, verbose=False;, score=0.706 total time=71.8min
#[CV 1/2] END C=10, cache_size=200, class_weight=None, coef0=0, decision_function_shape=ovr, degree=3, gamma=scale, kernel=rbf, max_iter=10000, probability=False, random_state=None, shrinking=True, tol=0.001, verbose=False;, score=0.702 total time=82.9min
#[CV 2/2] END C=100, cache_size=200, class_weight=None, coef0=0, decision_function_shape=ovr, degree=3, gamma=scale, kernel=rbf, max_iter=10000, probability=False, random_state=None, shrinking=True, tol=0.001, verbose=False;, score=0.719 total time=65.0min
#/home/ubuntu/.pyenv/versions/fl-3.10.14/lib/python3.10/site-packages/sklearn/svm/_base.py:297: ConvergenceWarning: Solver terminated early (max_iter=10000).  Consider pre-processing your data with StandardScaler or MinMaxScaler.
#  warnings.warn(
#Scoring Evaluation....
#Done
#Evaluation result: {'acc': 0.7205, 'auprc': 0.7916296693576961}
#\Evaluation time: 11:09:03.381714
#
#Finished.

# CUDA_VISIBLE_DEVICES=0 python scripts/TimesURL/train_timesURL.py dkt dkt_run  --loader dkt --epochs 30 --batch_size 512 --eval --num_train 10000 --num_test 10000 --instance 2024-11-09_01:09:33 --seed 2024

#Training time: 2:31:53.907357
#
#x (10000, 256, 8) (10000, 256, 8)
#mask (10000, 256, 8) (10000, 256, 8)
#Getting Representation from Training Data: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████| 625/625 [00:02<00:00, 224.19it/s]
#Getting Representation from Training Data: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████| 625/625 [00:02<00:00, 224.07it/s]
#Fitting Evaluation....
#Grid Search!
#Fitting 2 folds for each of 5 candidates, totalling 10 fits
#[CV 2/2] END C=0.01, cache_size=200, class_weight=None, coef0=0, decision_function_shape=ovr, degree=3, gamma=scale, kernel=rbf, max_iter=10000, probability=False, random_state=None, shrinking=True, tol=0.001, verbose=False;, score=0.605 total time=81.3min
#[CV 2/2] END C=0.1, cache_size=200, class_weight=None, coef0=0, decision_function_shape=ovr, degree=3, gamma=scale, kernel=rbf, max_iter=10000, probability=False, random_state=None, shrinking=True, tol=0.001, verbose=False;, score=0.660 total time=84.7min
#[CV 1/2] END C=0.1, cache_size=200, class_weight=None, coef0=0, decision_function_shape=ovr, degree=3, gamma=scale, kernel=rbf, max_iter=10000, probability=False, random_state=None, shrinking=True, tol=0.001, verbose=False;, score=0.666 total time=93.4min
#[CV 1/2] END C=1, cache_size=200, class_weight=None, coef0=0, decision_function_shape=ovr, degree=3, gamma=scale, kernel=rbf, max_iter=10000, probability=False, random_state=None, shrinking=True, tol=0.001, verbose=False;, score=0.690 total time=94.7min
#/home/ubuntu/.pyenv/versions/fl-3.10.14/lib/python3.10/site-packages/sklearn/svm/_base.py:297: ConvergenceWarning: Solver terminated early (max_iter=10000).  Consider pre-processing your data with StandardScaler or MinMaxScaler.
#  warnings.warn(
#[CV 1/2] END C=0.01, cache_size=200, class_weight=None, coef0=0, decision_function_shape=ovr, degree=3, gamma=scale, kernel=rbf, max_iter=10000, probability=False, random_state=None, shrinking=True, tol=0.001, verbose=False;, score=0.606 total time=107.7min
#/home/ubuntu/.pyenv/versions/fl-3.10.14/lib/python3.10/site-packages/sklearn/svm/_base.py:297: ConvergenceWarning: Solver terminated early (max_iter=10000).  Consider pre-processing your data with StandardScaler or MinMaxScaler.
#  warnings.warn(
#[CV 1/2] END C=100, cache_size=200, class_weight=None, coef0=0, decision_function_shape=ovr, degree=3, gamma=scale, kernel=rbf, max_iter=10000, probability=False, random_state=None, shrinking=True, tol=0.001, verbose=False;, score=0.704 total time=58.0min
#[CV 2/2] END C=1, cache_size=200, class_weight=None, coef0=0, decision_function_shape=ovr, degree=3, gamma=scale, kernel=rbf, max_iter=10000, probability=False, random_state=None, shrinking=True, tol=0.001, verbose=False;, score=0.680 total time=80.7min
#[CV 1/2] END C=10, cache_size=200, class_weight=None, coef0=0, decision_function_shape=ovr, degree=3, gamma=scale, kernel=rbf, max_iter=10000, probability=False, random_state=None, shrinking=True, tol=0.001, verbose=False;, score=0.706 total time=81.2min
#[CV 2/2] END C=10, cache_size=200, class_weight=None, coef0=0, decision_function_shape=ovr, degree=3, gamma=scale, kernel=rbf, max_iter=10000, probability=False, random_state=None, shrinking=True, tol=0.001, verbose=False;, score=0.705 total time=72.9min
#[CV 2/2] END C=100, cache_size=200, class_weight=None, coef0=0, decision_function_shape=ovr, degree=3, gamma=scale, kernel=rbf, max_iter=10000, probability=False, random_state=None, shrinking=True, tol=0.001, verbose=False;, score=0.702 total time=67.8min
#/home/ubuntu/.pyenv/versions/fl-3.10.14/lib/python3.10/site-packages/sklearn/svm/_base.py:297: ConvergenceWarning: Solver terminated early (max_iter=10000).  Consider pre-processing your data with StandardScaler or MinMaxScaler.
#  warnings.warn(
#Scoring Evaluation....
#Done
#Evaluation result: {'acc': 0.7207, 'auprc': 0.7950338465012203}
#\Evaluation time: 11:41:47.796635
#
#Finished.

# WITH 100 Epochs

# CUDA_VISIBLE_DEVICES=0 python scripts/TimesURL/train_timesURL.py dkt dkt_run  --loader dkt --epochs 100 --batch_size 512 --eval --num_train 10000 --num_test 10000 --instance 2024-10-16_22:03:27

#Training time: 8:29:00.541632

#Fitting Evaluation....
#Grid Search!
#Fitting 2 folds for each of 5 candidates, totalling 10 fits
#Evaluation result: {'acc': 0.7244, 'auprc': 0.7942025667105461}
#\Evaluation time: 10:47:32.858584


# Geolife

# CUDA_VISIBLE_DEVICES=0 python scripts/TimesURL/train_timesURL.py geolife geolife_run --loader geolife --epochs 30 --batch_size 64 --eval --instance "2024-10-22_15:41:02" --s3_bucket_path data-phd

#Training time: 3:32:17.659829

#Getting Representation from Training Data: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████| 352/352 [00:04<00:00, 80.46it/s]
#Getting Representation from Training Data: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████| 98/98 [00:01<00:00, 80.38it/s]

#Fitting Evaluation....

#Grid Search!
#Fitting 5 folds for each of 5 candidates, totalling 25 fits
#Scoring Evaluation....

#Evaluation result: {'acc': 0.7660668380462725, 'auprc': 0.7356603700228475}
#Evaluation time: 6:21:54.111452
# Evaluation time for no grid Search is: 	1:28:43.229896


