#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sklearn.datasets import load_iris, load_wine, fetch_california_housing

import os
import pandas as pd

import wget

DATASETS = ['iris', 'wine', 'boston', 'california', 'parkinsons', \
            'climate_model_crashes', 'concrete_compression', \
            'yacht_hydrodynamics', 'airfoil_self_noise', \
            'connectionist_bench_sonar', 'ionosphere', 'qsar_biodegradation', \
            'seeds', 'glass', 'ecoli', 'yeast', 'libras', 'planning_relax', \
            'blood_transfusion', 'breast_cancer_diagnostic', \
            'connectionist_bench_vowel', 'concrete_slump', \
            'gauss', 'skew', 'skew_multi', 'multi', 'heavy',
            'wine_quality_red', 'wine_quality_white']


def dataset_loader(dataset):
    """
    Data loading utility for a subset of UCI ML repository datasets. Assumes
    datasets are located in './datasets'. If the called for dataset is not in
    this folder, it is downloaded from the UCI ML repo.

    Parameters
    ----------

    dataset : str
        Name of the dataset to retrieve.
        Valid values: see DATASETS.

    Returns
    ------
    X : ndarray
        Data values (predictive values only).
    """
    assert dataset in DATASETS, f"Dataset not supported: {dataset}"

    if dataset in DATASETS:
        if dataset == 'iris':
            X = load_iris()['data']
        elif dataset == 'wine':
            X = load_wine()['data']
        elif dataset == 'boston':
            X = load_boston()['data']
        elif dataset == 'california':
            X = fetch_california_housing()['data']
        elif dataset == 'parkinsons':
            X = fetch_parkinsons()['data']
        elif dataset == 'climate_model_crashes':
            X = fetch_climate_model_crashes()['data']
        elif dataset == 'concrete_compression':
            X = fetch_concrete_compression()['data']
        elif dataset == 'yacht_hydrodynamics':
            X = fetch_yacht_hydrodynamics()['data']
        elif dataset == 'airfoil_self_noise':
            X = fetch_airfoil_self_noise()['data']
        elif dataset == 'connectionist_bench_sonar':
            X = fetch_connectionist_bench_sonar()['data']
        elif dataset == 'ionosphere':
            X = fetch_ionosphere()['data']
        elif dataset == 'qsar_biodegradation':
            X = fetch_qsar_biodegradation()['data']
        elif dataset == 'seeds':
            X = fetch_seeds()['data']
        elif dataset == 'glass':
            X = fetch_glass()['data']
        elif dataset == 'ecoli':
            X = fetch_ecoli()['data']
        elif dataset == 'yeast':
            X = fetch_yeast()['data']
        elif dataset == 'libras':
            X = fetch_libras()['data']
        elif dataset == 'planning_relax':
            X = fetch_planning_relax()['data']
        elif dataset == 'blood_transfusion':
            X = fetch_blood_transfusion()['data']
        elif dataset == 'breast_cancer_diagnostic':
            X = fetch_breast_cancer_diagnostic()['data']
        elif dataset == 'connectionist_bench_vowel':
            X = fetch_connectionist_bench_vowel()['data']
        elif dataset == 'concrete_slump':
            X = fetch_concrete_slump()['data']
        elif dataset == 'wine_quality_red':
            X = fetch_wine_quality_red()['data']
        elif dataset == 'wine_quality_white':
            X = fetch_wine_quality_white()['data']
        #  'gauss', 'skew', 'skew_multi', 'multi',
        elif dataset == 'gauss':
            X = fetch_gauss()['data']
        elif dataset == 'skew':
            X = fetch_skew()['data']
        elif dataset == 'skew_multi':
            X = fetch_skew_multi()['data']
        elif dataset == 'multi':
            X = fetch_multi()['data']
        elif dataset == 'heavy':
            X = fetch_heavy()['data']

        return X


def label_loader(dataset):
    """
    Data loading utility for a subset of UCI ML repository datasets. Assumes
    datasets are located in './datasets'. If the called for dataset is not in
    this folder, it is downloaded from the UCI ML repo.

    Parameters
    ----------

    dataset : str
        Name of the dataset to retrieve.
        Valid values: see DATASETS.

    Returns
    ------
    X : ndarray
        Data values (predictive values only).
    """
    assert dataset in DATASETS, f"Dataset not supported: {dataset}"

    if dataset in DATASETS:
        if dataset == 'iris':
            X = load_iris()['target']
        elif dataset == 'wine':
            X = load_wine()['target']
        elif dataset == 'boston':
            X = load_boston()['target']
        elif dataset == 'california':
            X = fetch_california_housing()['target']
        elif dataset == 'parkinsons':
            X = fetch_parkinsons()['target']
        elif dataset == 'climate_model_crashes':
            X = fetch_climate_model_crashes()['target']
        elif dataset == 'concrete_compression':
            X = fetch_concrete_compression()['target']
        elif dataset == 'yacht_hydrodynamics':
            X = fetch_yacht_hydrodynamics()['target']
        elif dataset == 'airfoil_self_noise':
            X = fetch_airfoil_self_noise()['target']
        elif dataset == 'connectionist_bench_sonar':
            X = fetch_connectionist_bench_sonar()['target']
        elif dataset == 'ionosphere':
            X = fetch_ionosphere()['target']
        elif dataset == 'qsar_biodegradation':
            X = fetch_qsar_biodegradation()['target']
        elif dataset == 'seeds':
            X = fetch_seeds()['target']
        elif dataset == 'glass':
            X = fetch_glass()['target']
        elif dataset == 'ecoli':
            X = fetch_ecoli()['target']
        elif dataset == 'yeast':
            X = fetch_yeast()['target']
        elif dataset == 'libras':
            X = fetch_libras()['target']
        elif dataset == 'planning_relax':
            X = fetch_planning_relax()['target']
        elif dataset == 'blood_transfusion':
            X = fetch_blood_transfusion()['target']
        elif dataset == 'breast_cancer_diagnostic':
            X = fetch_breast_cancer_diagnostic()['target']
        elif dataset == 'connectionist_bench_vowel':
            X = fetch_connectionist_bench_vowel()['target']
        elif dataset == 'concrete_slump':
            X = fetch_concrete_slump()['target']
        elif dataset == 'wine_quality_red':
            X = fetch_wine_quality_red()['target']
        elif dataset == 'wine_quality_white':
            X = fetch_wine_quality_white()['target']

        return X


def fetch_parkinsons():
    if not os.path.isdir('datasets/parkinsons'):
        os.mkdir('datasets/parkinsons')
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data'
        wget.download(url, out='datasets/parkinsons/')

    with open('datasets/parkinsons/parkinsons.data', 'rb') as f:
        df = pd.read_csv(f, delimiter=',', header=0)
        Xy = {}
        Xy['data'] = df.values[:, 1:].astype('float')
        Xy['target'] = df.values[:, 0]

    return Xy


def fetch_climate_model_crashes():
    if not os.path.isdir('datasets/climate_model_crashes'):
        os.mkdir('datasets/climate_model_crashes')
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00252/pop_failures.dat'
        wget.download(url, out='datasets/climate_model_crashes/')

    with open('datasets/climate_model_crashes/pop_failures.dat', 'rb') as f:
        df = pd.read_csv(f, delimiter='\s+', header=0)
        Xy = {}
        Xy['data'] = df.values[:, 2:-1]
        Xy['target'] = df.values[:, -1]

    return Xy


def fetch_concrete_compression():
    if not os.path.isdir('datasets/concrete_compression'):
        os.mkdir('datasets/concrete_compression')
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/Concrete_Data.xls'
        wget.download(url, out='datasets/concrete_compression/')

    with open('datasets/concrete_compression/Concrete_Data.xls', 'rb') as f:
        df = pd.read_excel(io=f)
        Xy = {}
        Xy['data'] = df.values[:, :-2]
        Xy['target'] = df.values[:, -1]

    return Xy


def fetch_yacht_hydrodynamics():
    if not os.path.isdir('datasets/yacht_hydrodynamics'):
        os.mkdir('datasets/yacht_hydrodynamics')
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00243/yacht_hydrodynamics.data'
        wget.download(url, out='datasets/yacht_hydrodynamics/')

    with open('datasets/yacht_hydrodynamics/yacht_hydrodynamics.data', 'rb') as f:
        df = pd.read_csv(f, delimiter='\s+', header=None)
        Xy = {}
        Xy['data'] = df.values[:, :-1]
        Xy['target'] = df.values[:, -1]

    return Xy


def fetch_airfoil_self_noise():
    if not os.path.isdir('datasets/airfoil_self_noise'):
        os.mkdir('datasets/airfoil_self_noise')
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00291/airfoil_self_noise.dat'
        wget.download(url, out='datasets/airfoil_self_noise/')

    with open('datasets/airfoil_self_noise/airfoil_self_noise.dat', 'rb') as f:
        df = pd.read_csv(f, delimiter='\s+', header=None)
        Xy = {}
        Xy['data'] = df.values[:, :-1]
        Xy['target'] = df.values[:, -1]

    return Xy


def fetch_connectionist_bench_sonar():
    if not os.path.isdir('datasets/connectionist_bench_sonar'):
        os.mkdir('datasets/connectionist_bench_sonar')
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/sonar/sonar.all-data'
        wget.download(url, out='datasets/connectionist_bench_sonar/')

    with open('datasets/connectionist_bench_sonar/sonar.all-data', 'rb') as f:
        df = pd.read_csv(f, delimiter=',', header=None)
        Xy = {}
        Xy['data'] = df.values[:, :-1].astype('float')
        Xy['target'] = df.values[:, -1]

    return Xy


def fetch_ionosphere():
    if not os.path.isdir('datasets/ionosphere'):
        os.mkdir('datasets/ionosphere')
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/ionosphere/ionosphere.data'
        wget.download(url, out='datasets/ionosphere/')

    with open('datasets/ionosphere/ionosphere.data', 'rb') as f:
        df = pd.read_csv(f, delimiter=',', header=None)
        Xy = {}
        Xy['data'] = df.values[:, :-1].astype('float')
        Xy['target'] = df.values[:, -1]

    return Xy


def fetch_qsar_biodegradation():
    if not os.path.isdir('datasets/qsar_biodegradation'):
        os.mkdir('datasets/qsar_biodegradation')
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00254/biodeg.csv'
        wget.download(url, out='datasets/qsar_biodegradation/')

    with open('datasets/qsar_biodegradation/biodeg.csv', 'rb') as f:
        df = pd.read_csv(f, delimiter=';', header=None)
        Xy = {}
        Xy['data'] = df.values[:, :-1].astype('float')
        Xy['target'] = df.values[:, -1]

    return Xy


def fetch_seeds():
    if not os.path.isdir('datasets/seeds'):
        os.mkdir('datasets/seeds')
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00236/seeds_dataset.txt'
        wget.download(url, out='datasets/seeds/')

    with open('datasets/seeds/seeds_dataset.txt', 'rb') as f:
        df = pd.read_csv(f, delimiter='\s+', header=None)
        Xy = {}
        Xy['data'] = df.values[:, :-1].astype('float')
        Xy['target'] = df.values[:, -1]

    return Xy


def fetch_glass():
    if not os.path.isdir('datasets/glass'):
        os.mkdir('datasets/glass')
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/glass/glass.data'
        wget.download(url, out='datasets/glass/')

    with open('datasets/glass/glass.data', 'rb') as f:
        df = pd.read_csv(f, delimiter=',', header=None)
        Xy = {}
        Xy['data'] = df.values[:, 1:-1].astype('float')
        Xy['target'] = df.values[:, -1]

    return Xy


def fetch_ecoli():
    if not os.path.isdir('datasets/ecoli'):
        os.mkdir('datasets/ecoli')
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/ecoli/ecoli.data'
        wget.download(url, out='datasets/ecoli/')

    with open('datasets/ecoli/ecoli.data', 'rb') as f:
        df = pd.read_csv(f, delimiter='\s+', header=None)
        Xy = {}
        Xy['data'] = df.values[:, 1:-1].astype('float')
        Xy['target'] = df.values[:, -1]

    return Xy


def fetch_yeast():
    if not os.path.isdir('datasets/yeast'):
        os.mkdir('datasets/yeast')
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/yeast/yeast.data'
        wget.download(url, out='datasets/yeast/')

    with open('datasets/yeast/yeast.data', 'rb') as f:
        df = pd.read_csv(f, delimiter='\s+', header=None)
        Xy = {}
        Xy['data'] = df.values[:, 1:-1].astype('float')
        Xy['target'] = df.values[:, -1]

    return Xy


def fetch_libras():
    if not os.path.isdir('datasets/libras'):
        os.mkdir('datasets/libras')
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/libras/movement_libras.data'
        wget.download(url, out='datasets/libras/')

    with open('datasets/libras/movement_libras.data', 'rb') as f:
        df = pd.read_csv(f, delimiter=',', header=None)
        Xy = {}
        Xy['data'] = df.values[:, :-1].astype('float')
        Xy['target'] = df.values[:, -1]

    return Xy


def fetch_planning_relax():
    if not os.path.isdir('datasets/planning_relax'):
        os.mkdir('datasets/planning_relax')
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00230/plrx.txt'
        wget.download(url, out='datasets/planning_relax/')

    with open('datasets/planning_relax/plrx.txt', 'rb') as f:
        df = pd.read_csv(f, delimiter='\s+', header=None)
        Xy = {}
        Xy['data'] = df.values[:, :-1].astype('float')
        Xy['target'] = df.values[:, -1]

    return Xy


def fetch_blood_transfusion():
    if not os.path.isdir('datasets/blood_transfusion'):
        os.mkdir('datasets/blood_transfusion')
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/blood-transfusion/transfusion.data'
        wget.download(url, out='datasets/blood_transfusion/')

    with open('datasets/blood_transfusion/transfusion.data', 'rb') as f:
        df = pd.read_csv(f, delimiter=',')
        Xy = {}
        Xy['data'] = df.values[:, :-1].astype('float')
        Xy['target'] = df.values[:, -1]

    return Xy


def fetch_breast_cancer_diagnostic():
    if not os.path.isdir('datasets/breast_cancer_diagnostic'):
        os.mkdir('datasets/breast_cancer_diagnostic')
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data'
        wget.download(url, out='datasets/breast_cancer_diagnostic/')

    with open('datasets/breast_cancer_diagnostic/wdbc.data', 'rb') as f:
        df = pd.read_csv(f, delimiter=',', header=None)
        Xy = {}
        Xy['data'] = df.values[:, 2:].astype('float')
        Xy['target'] = df.values[:, 1]

    return Xy


def fetch_connectionist_bench_vowel():
    if not os.path.isdir('datasets/connectionist_bench_vowel'):
        os.mkdir('datasets/connectionist_bench_vowel')
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/vowel/vowel-context.data'
        wget.download(url, out='datasets/connectionist_bench_vowel/')

    with open('datasets/connectionist_bench_vowel/vowel-context.data', 'rb') as f:
        df = pd.read_csv(f, delimiter='\s+', header=None)
        Xy = {}
        Xy['data'] = df.values[:, 3:-1].astype('float')
        Xy['target'] = df.values[:, -1]

    return Xy


def fetch_concrete_slump():
    if not os.path.isdir('datasets/concrete_slump'):
        os.mkdir('datasets/concrete_slump')
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/slump/slump_test.data'
        wget.download(url, out='datasets/concrete_slump/')

    with open('datasets/concrete_slump/slump_test.data', 'rb') as f:
        df = pd.read_csv(f, delimiter=',')
        Xy = {}
        Xy['data'] = df.values[:, 1:-3].astype('float')
        Xy['target'] = df.values[:, -3:]

    return Xy


def fetch_wine_quality_red():
    if not os.path.isdir('datasets/wine_quality_red'):
        os.mkdir('datasets/wine_quality_red')
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
        wget.download(url, out='datasets/wine_quality_red/')

    with open('datasets/wine_quality_red/winequality-red.csv', 'rb') as f:
        df = pd.read_csv(f, delimiter=';')
        Xy = {}
        Xy['data'] = df.values[:, 1:-1].astype('float')
        Xy['target'] = df.values[:, -1]

    return Xy


def fetch_wine_quality_white():
    if not os.path.isdir('datasets/wine_quality_white'):
        os.mkdir('datasets/wine_quality_white')
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv'
        wget.download(url, out='datasets/wine_quality_white/')

    with open('datasets/wine_quality_white/winequality-white.csv', 'rb') as f:
        df = pd.read_csv(f, delimiter=';')
        Xy = {}
        Xy['data'] = df.values[:, :-1].astype('float')
        Xy['target'] = df.values[:, -1]

    return Xy

def fetch_gauss():

    # with open('datasets/toy_case/gauss.csv', 'rb') as f:
    df = pd.read_csv('datasets/toy_case/gauss.csv', index_col=0, header=0)
    Xy = {}
    Xy['data'] = df.values.astype('float')

    return Xy

def fetch_skew():

    # with open('datasets/toy_case/gauss.csv', 'rb') as f:
    df = pd.read_csv('datasets/toy_case/skew.csv', index_col=0, header=0)
    Xy = {}
    Xy['data'] = df.values.astype('float')

    return Xy


def fetch_skew_multi():

    # with open('datasets/toy_case/gauss.csv', 'rb') as f:
    df = pd.read_csv('datasets/toy_case/skew_multi.csv', index_col=0, header=0)
    Xy = {}
    Xy['data'] = df.values.astype('float')

    return Xy

def fetch_multi():

    # with open('datasets/toy_case/gauss.csv', 'rb') as f:
    df = pd.read_csv('datasets/toy_case/multi.csv', index_col=0, header=0)
    Xy = {}
    Xy['data'] = df.values.astype('float')

    return Xy

def fetch_heavy():

    # with open('datasets/toy_case/gauss.csv', 'rb') as f:
    df = pd.read_csv('datasets/toy_case/heavy.csv', index_col=0, header=0)
    Xy = {}
    Xy['data'] = df.values.astype('float')

    return Xy