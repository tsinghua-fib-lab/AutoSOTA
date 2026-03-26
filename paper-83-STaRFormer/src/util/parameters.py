from typing import List
from types import NoneType


__all__ = [
    'ClassificationMethodOptions',
    'DatasetOptions',
    'LossOptions',
    'MaskingOptions',
    'ModelOptions',
    'PredictionOptions',
    'TaskOptions',
    'TrainingMethodOptions',
    'TrainingModeOptions',
    'MetricOptions',
    'RegressionMethodOptions'
]


class OptionsBaseClass:
    @classmethod
    def get_options(cls) -> List[str | NoneType]:
        """ Returns all options (attributes) of the class. """
        #return {key: value for key, value in cls.__dict__.items() if not key.startswith('__')}
        return [value for key, value in cls.__dict__.items() if not key.startswith('__') and key != 'get_options']

class ClassificationMethodOptions(OptionsBaseClass):
    cls_token: str='cls_token'
    autoregressive: str='autoregressive'
    elementwise: str='elementwise'

class RegressionMethodOptions(OptionsBaseClass):
    regr_token: str='regr_token'
    autoregressive: str='autoregressive'
    elementwise: str='elementwise'


class DatasetOptions(OptionsBaseClass):
    mnist: str = 'mnist'
    dkt: str='dkt'
    geolife: str = 'geolife'
    # uea
    ethanolconcentration: str='ethanolconcentration'
    eigenworms: str='eigenworms'
    facedetection: str='facedetection'
    heartbeat: str='heartbeat'
    handwriting: str='handwriting'
    insectwingbeat: str='insectwingbeat'
    japanesevowels: str='japanesevowels'
    pendigits: str='pendigits'
    pemssf: str='pemssf'
    rightwhalecalls: str='rightwhalecalls'
    selfregulationscp1: str='selfregulationscp1'
    selfregulationscp2: str='selfregulationscp2'
    spokenarabicdigits: str='spokenarabicdigits'
    uwavegesturelibrary: str='uwavegesturelibrary'
    # irregularly sampled
    pam: str='pam'
    p12: str='p12'
    p19: str='p19'
    # tsr 
    appliancesenergy: str='appliancesenergy'
    benzeneconcentration: str='benzeneconcentration'
    beijingpm10: str='beijingpm10'
    beijingpm25: str='beijingpm25'
    livefuelmoisture: str='livefuelmoisture'
    ieeeppg: str='ieeeppg'


    uea: List[str]=[
        'ethanolconcentration', 'eigenworms', 'facedetection', 'heartbeat',
        'handwriting', 'insectwingbeat', 'japanesevowels', 'pendigits',
        'pemssf', 'rightwhalecalls', 'selfregulationscp1', 'selfregulationscp2',
        'spokenarabicdigits', 'uwavegesturelibrary']
    
    uea_binary: List[str]=['facedetection', 'heartbeat', 'rightwhalecalls', 'selfregulationscp1', 'selfregulationscp2']

    uea_multi: List[str]=[
        'ethanolconcentration', 'eigenworms', 'handwriting', 'insectwingbeat', 
        'japanesevowels', 'pendigits', 'pemssf', 'spokenarabicdigits', 'uwavegesturelibrary']
    
    irregulary_sampled: List[str] = [
        'pam', 'p12', 'p19'
    ]

    tsr: List[str]= [
        'appliancesenergy', 'benzeneconcentration', 'beijingpm10', 'beijingpm25', 'livefuelmoisture', 'ieeeppg'
    ]


class LossOptions(OptionsBaseClass):
    bce: str='bce'
    ce: str='ce'
    reconstruction: str = 'reconstruction'
    sscl: str = 'sscl'


class MaskingOptions(OptionsBaseClass):
    random: str = 'random'
    darem: str = 'darem'
    none: NoneType = None


class MetricOptions(OptionsBaseClass):
    accuracy: str = 'acc'
    confusion_matrix: str = 'cm'
    f05: str = 'f05'
    f1: str = 'f1'
    precision: str='precision'
    recall: str='recall'
    auroc: str='auroc'
    auprc: str='auprc'
    rmse: str='rmse'
    mae: str='mae'


class ModelOptions(OptionsBaseClass):
    gru: str = 'gru'
    lstm: str = 'lstm'
    rnn: str = 'rnn'
    starformer: str = 'starformer'
    all: list[str] = [
        'gru',
        'lstm',
        'rnn',
        'starformer'
    ]
    rnn_based: List[str] = [
        'gru', 'lstm', 'rnn'
    ]


class PredictionOptions(OptionsBaseClass):
    binary: str='binary'
    multiclass: str='multiclass'


class TaskOptions(OptionsBaseClass):
    classification: str='classification'
    regression: str='regression'
    forecasting: str='forecasting'


class TrainingMethodOptions(OptionsBaseClass):
    centralized: str = 'centralized'
    federated: str = 'federated'

class TrainingModeOptions(OptionsBaseClass):
    train: str = 'train'
    val: str = 'val'
    test: str = 'test'
