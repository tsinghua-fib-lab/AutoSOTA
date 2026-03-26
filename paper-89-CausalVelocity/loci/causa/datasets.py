import torch
import pandas as pd
import numpy as np
import gin

from causa import DATA_DIR


@gin.configurable
class CausalDataset(object):
    pass


@gin.configurable
class ANLSMN(CausalDataset):
    n_datasets = 100

    def __init__(self, pair_id=None, path=DATA_DIR, double=False, preprocessor=None):
        assert pair_id in list(range(1, self.n_datasets+1))
        directory = DATA_DIR + '/' + self.folder_name
        df = pd.read_csv(directory + f'/pair_{pair_id}.txt', delimiter=',')
        ground_truth = pd.read_csv(directory + '/pairs_gt.txt', header=None).iloc[pair_id-1].values[0]
        if ground_truth == 1:  # true order
            cause, effect = df.iloc[:, 1:2].values, df.iloc[:, 2:3].values
        elif ground_truth == 0:  # wrong order
            cause, effect = df.iloc[:, 2:3].values, df.iloc[:, 1:2].values
        else:
            raise ValueError()

        if preprocessor is not None:
            cause = preprocessor.fit_transform(cause)
            effect = preprocessor.fit_transform(effect)

        cause, effect = torch.from_numpy(cause), torch.from_numpy(effect)
        if double:  # default double
            cause, effect = cause.double(), effect.double()
        else:
            cause, effect = cause.float(), effect.float()
        self.cause, self.effect = cause, effect

    @property
    def folder_name(self):
        return 'ANLSMN_pairs'


@gin.configurable
class AN(ANLSMN):
    
    @property
    def folder_name(self):
        return super().folder_name + '/AN'
        

@gin.configurable
class ANs(ANLSMN):
    
    @property
    def folder_name(self):
        return super().folder_name + '/AN-s'


@gin.configurable
class LS(ANLSMN):
    
    @property
    def folder_name(self):
        return super().folder_name + '/LS'


@gin.configurable
class LSs(ANLSMN):
    
    @property
    def folder_name(self):
        return super().folder_name + '/LS-s'


@gin.configurable
class MNU(ANLSMN):
    
    @property
    def folder_name(self):
        return super().folder_name + '/MN-U'

        
@gin.configurable
class Tuebingen(CausalDataset):
    n_datasets = 108

    def __init__(self, pair_id=None, path=DATA_DIR, double=False, preprocessor=None):
        assert pair_id in list(range(1, self.n_datasets+1))
        directory = DATA_DIR + '/' + self.folder_name
        df = pd.read_csv(directory + f'/pair{pair_id:04d}.txt', delim_whitespace=True, header=None)
        # TODO: to use weight need to cast weight column to float..
        meta = pd.read_csv(directory + '/pairmeta.txt', delim_whitespace=True, 
                           header=None, 
                           names=['id', 'cause_start', 'cause_end', 'effect_start', 'effect_end', 'weight'],
                           index_col=0).astype(int)
        cause = df.iloc[:, meta.loc[pair_id, 'cause_start']-1:meta.loc[pair_id, 'cause_end']].values
        effect = df.iloc[:, meta.loc[pair_id, 'effect_start']-1:meta.loc[pair_id, 'effect_end']].values

        if preprocessor is not None:
            cause = preprocessor.fit_transform(cause)
            effect = preprocessor.fit_transform(effect)

        cause, effect = torch.from_numpy(cause), torch.from_numpy(effect)
        if double:  # default double
            cause, effect = cause.double(), effect.double()
        else:
            cause, effect = cause.float(), effect.float()

        self.cause, self.effect = cause, effect

    @property
    def folder_name(self):
        return 'Tuebingen'

        
@gin.configurable
class BenchmarkSimulated(Tuebingen):
    n_datasets = 100

    @property
    def folder_name(self):
        return 'Benchmark_simulated'


@gin.configurable
class SIM(BenchmarkSimulated):

    @property
    def folder_name(self):
        return super().folder_name + '/SIM'


@gin.configurable
class SIMc(BenchmarkSimulated):

    @property
    def folder_name(self):
        return super().folder_name + '/SIM-c'


@gin.configurable
class SIMG(BenchmarkSimulated):

    @property
    def folder_name(self):
        return super().folder_name + '/SIM-G'


@gin.configurable
class SIMln(BenchmarkSimulated):

    @property
    def folder_name(self):
        return super().folder_name + '/SIM-ln'


@gin.configurable
class Dataverse(CausalDataset):
    n_datasets = 300

    def __init__(self, pair_id=None, path=DATA_DIR, double=False, preprocessor=None):
        assert pair_id in list(range(1, self.n_datasets+1))
        directory = DATA_DIR + '/' + self.folder_name + '/' + self.file_name
        pairs = pd.read_csv(directory + '_pairs.csv')
        targets = pd.read_csv(directory + '_targets.csv')
        target = targets.loc[pair_id-1, 'Target']
        if target == 1:  
            cause = self.to_numpy(pairs.loc[pair_id-1, 'A'])
            effect = self.to_numpy(pairs.loc[pair_id-1, 'B'])
        elif target == -1:
            cause = self.to_numpy(pairs.loc[pair_id-1, 'B'])
            effect = self.to_numpy(pairs.loc[pair_id-1, 'A'])
        else:
            raise ValueError('Invalid target:', target)

        if preprocessor is not None:
            cause = preprocessor.fit_transform(cause)
            effect = preprocessor.fit_transform(effect)

        cause, effect = torch.from_numpy(cause), torch.from_numpy(effect)
        if double:  # default double
            cause, effect = cause.double(), effect.double()
        else:
            cause, effect = cause.float(), effect.float()
        self.cause, self.effect = cause, effect

    @staticmethod
    def to_numpy(data_string):
        return np.array([float(e) for e in data_string.strip().split(' ')]).reshape(-1, 1)

    @property
    def folder_name(self):
        return 'Dataverse_pairs'

    @property
    def file_name(self):
        raise NotImplementedError()


@gin.configurable
class Cha(Dataverse):
    
    @property
    def file_name(self):
        return 'CE-Cha'


@gin.configurable
class Multi(Dataverse):
    
    @property
    def file_name(self):
        return 'CE-Multi'


@gin.configurable
class Net(Dataverse):
    
    @property
    def file_name(self):
        return 'CE-Net'