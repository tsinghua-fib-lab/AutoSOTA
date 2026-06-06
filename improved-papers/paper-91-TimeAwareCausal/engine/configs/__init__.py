from engine.configs.registry import Registry
# from .defaults import _C as cfg
import sys

sys.dont_write_bytecode = True

Loggers = Registry('Loggers')
Datasets = Registry('Datasets')
Dataloaders = Registry('DataLoaders')
LossFuncs = Registry('LossFuncs')
EvalFuncs = Registry('EvalFuncs')
Schedulers = Registry('Schedulers')
Embeddings = Registry('Embeddings')
Classifers = Registry('Classifiers')
Algorithms = Registry('Algorithms')