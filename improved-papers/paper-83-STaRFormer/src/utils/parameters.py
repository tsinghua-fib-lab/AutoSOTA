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
    """Base class for options containers used in task configuration.

    This class provides a utility method to retrieve all option values
    defined as class attributes (excluding special and private attributes).

    Methods:
        get_options: Class method to retrieve all option values as a list.
    """
    @classmethod
    def get_options(cls) -> List[str | NoneType]:
        """Retrieve the list of option values defined as class attributes.

        Returns:
            list: A list containing the values of all class attributes that do not
                start with '__' and are not the 'get_options' method.

        Example:
            >>> class MyOptions(OptionsBaseClass):
            ...     foo = 'foo'
            ...     bar = 'bar'
            >>> MyOptions.get_options()
            ['foo', 'bar']
        """
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
    # uea 1
    ethanolconcentration: str='ethanolconcentration'
    eigenworms: str='eigenworms'
    facedetection: str='facedetection'
    heartbeat: str='heartbeat'
    handwriting: str='handwriting'
    japanesevowels: str='japanesevowels'
    pendigits: str='pendigits'
    pemssf: str='pemssf'
    selfregulationscp1: str='selfregulationscp1'
    selfregulationscp2: str='selfregulationscp2'
    spokenarabicdigits: str='spokenarabicdigits'
    uwavegesturelibrary: str='uwavegesturelibrary'
    # uea 2
    articularywordrecognition: str='articularywordrecognition'
    atrialfibrillation: str='atrialfibrillation'
    basicmotions: str='basicmotions'
    charactertrajectories: str='charactertrajectories'
    cricket: str='cricket'
    duckduckgeese: str='duckduckgeese'
    epilepsy: str='epilepsy'
    ering: str='ering'
    fingermovements: str='fingermovements'
    handmovementdirection: str='handmovementdirection'
    insectwingbeat: str='insectwingbeat'
    libras: str='libras'
    lsst: str='lsst'
    motorimagery: str='motorimagery'
    natops: str='natops'
    phonemespectra: str='phonemespectra'
    racketsports: str='racketsports'
    standwalkjump: str='standwalkjump'
    # irregularly sampled
    pam: str='pam'
    p12: str='p12'
    p19: str='p19'
    # tsr 
    appliancesenergy: str='appliancesenergy'
    australiarainfall: str='australiarainfall'
    benzeneconcentration: str='benzeneconcentration'
    beijingpm10quality: str='beijingpm10quality'
    beijingpm25quality: str='beijingpm25quality'
    bidmcrr: str='bidmcrr' 
    bidmchr: str='bidmchr' 
    bidmcspo2: str='bidmcspo2' 
    covid3month: str="covid3month"
    floodmodeling1: str='floodmodeling1'
    floodmodeling2: str='floodmodeling2'
    floodmodeling3: str='floodmodeling3'
    householdpowerconsumption1: str='householdpowerconsumption1'
    householdpowerconsumption2: str='householdpowerconsumption2'
    ieeeppg: str='ieeeppg'
    livefuelmoisturecontent: str='livefuelmoisturecontent'
    newsheadlinesentiment: str='newsheadlinesentiment'
    newstitlesentiment: str='newstitlesentiment'
    ppgdalia: str='ppgdalia'
    # anomaly
    kpi: str='kpi'
    yahoo: str='yahoo'


    uea: List[str]=[
        'ethanolconcentration', 'eigenworms', 'facedetection', 'heartbeat',
        'handwriting', 'japanesevowels', 'pendigits',
        'pemssf', 'selfregulationscp1', 'selfregulationscp2',
        'spokenarabicdigits', 'uwavegesturelibrary', 
        # 2
        'articularywordrecognition', 'atrialfibrillation', 'basicmotions',
        'charactertrajectories', 'cricket', 'duckduckgeese', 'epilepsy',
        'ering', 'fingermovements', 'handmovementdirection', 'insectwingbeat',
        'libras', 'lsst', 'motorimagery', 'natops', 'phonemespectra',
        'racketsports', 'standwalkjump',
    ]
    
    uea_binary: List[str]=['facedetection', 'fingermovements',  'heartbeat', 'motorimagery', 'selfregulationscp1', 'selfregulationscp2']

    uea_multi: List[str]=[
        'articularywordrecognition', 'atrialfibrillation', 'basicmotions', 'charactertrajectories', 'cricket', 'duckduckgeese', 'eigenworms',
        'epilepsy', 'ering', 'ethanolconcentration', 'handmovementdirection', 'handwriting', 'insectwingbeat', 'japanesevowels', 'libras', 
        'lsst', 'natops', 'pendigits', 'pemssf', 'phonemespectra', 'racketsports', 'spokenarabicdigits', 'standwalkjump', 'uwavegesturelibrary']
    
    irregulary_sampled: List[str] = [
        'pam', 'p12', 'p19'
    ]

    tsr: List[str]= [ 
        'appliancesenergy',
        'australiarainfall',
        'benzeneconcentration',
        'beijingpm10quality',
        'beijingpm25quality',
        'bidmcrr',
        'bidmchr',
        'bidmcspo2',
        'covid3month',
        'floodmodeling1',
        'floodmodeling2',
        'floodmodeling3',
        'householdpowerconsumption1',
        'householdpowerconsumption2',
        'ieeeppg',
        'livefuelmoisturecontent',
        'newsheadlinesentiment',
        'newstitlesentiment',
        'ppgdalia',
    ]

    anomaly: List[str] = ['kpi', 'yahoo']


class LossOptions(OptionsBaseClass):
    bce: str='bce'
    ce: str='ce'
    reconstruction: str = 'reconstruction'
    sscl: str = 'sscl'
    mse: str = 'mean_squarred_error'



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
    anomaly_detection: str='anomaly_detection'


class TrainingMethodOptions(OptionsBaseClass):
    centralized: str = 'centralized'
    federated: str = 'federated'

class TrainingModeOptions(OptionsBaseClass):
    train: str = 'train'
    val: str = 'val'
    test: str = 'test'
