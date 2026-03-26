from benchmark.sumo_net.pycox_local.pycox.datasets import kkbox,support,metabric,gbsg,flchain
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from benchmark.sumo_net.pycox_local.pycox.preprocessing.feature_transforms import *
import torch
# from .toy_data_generation import toy_data_class
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn_pandas import DataFrameMapper
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from lifelines import KaplanMeierFitter
import benchmark.sumo_net.pycox_local.pycox.utils as utils

def calc_km(durations,events):
    km = utils.kaplan_meier(durations, 1 - events)
    return km

class LogTransformer(BaseEstimator, TransformerMixin): #Scaling is already good. This leaves network architecture...
    def __init__(self):
        pass

    def fit_transform(self, input_array, y=None):
        return np.log(input_array)

    def fit(self, input_array, y=None):
        return self

    def transform(self, input_array, y=None):
        return np.log(input_array)

    def inverse_transform(self,input_array):
        return np.exp(input_array)

class IdentityTransformer(BaseEstimator, TransformerMixin): #Scaling is already good. This leaves network architecture...
    def __init__(self):
        pass

    def fit_transform(self, input_array, y=None):
        return input_array

    def fit(self, input_array, y=None):
        return self

    def transform(self, input_array, y=None):
        return input_array

    def inverse_transform(self,input_array):
        return input_array

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


class survival_dataset(Dataset):

    def __init__(self, data_train, data_val, data_test):

        super(survival_dataset, self).__init__()

        self.unique_cat_cols = []
        self.cat_X = []
        self.duration_mapper = MinMaxScaler()
  
        x_train = data_train['np.array']['x']
        x_val = data_val['np.array']['x']
        x_test = data_test['np.array']['x']

        self.y_train_ref = y_train = self.duration_mapper.fit_transform(data_train['np.array']['time'].reshape(-1, 1)).astype('float32')
        self.y_val_ref = y_val = self.duration_mapper.fit_transform(data_val['np.array']['time'].reshape(-1, 1)).astype('float32')
        self.y_test_ref = y_test = self.duration_mapper.fit_transform(data_test['np.array']['time'].reshape(-1, 1)).astype('float32')
                
        self.split(X=x_train,delta=data_train['pd.DataFrame']['event'],y=y_train,mode='train')
        self.split(X=x_val,delta=data_val['pd.DataFrame']['event'],y=y_val,mode='val')
        self.split(X=x_test,delta=data_test['pd.DataFrame']['event'],y=y_test,mode='test')
        self.set('train')


    def split(self,X,delta,y,mode='train'):
        min_dur,max_dur = y.min(),y.max()
        times = np.linspace(min_dur,max_dur,100)
        d = delta.values
        kmf = KaplanMeierFitter()
        kmf.fit(y,1-delta)
        setattr(self,f'{mode}_times', torch.from_numpy(times.astype('float32')).float().unsqueeze(-1))
        setattr(self,f'{mode}_delta', torch.from_numpy(delta.astype('float32').values).float())
        setattr(self,f'{mode}_y', torch.from_numpy(y).float())
        setattr(self, f'{mode}_X', torch.from_numpy(X).float())


    def set(self,mode='train'):
        self.X = getattr(self,f'{mode}_X')
        self.y = getattr(self,f'{mode}_y')
        self.times = getattr(self,f'{mode}_times')
        self.delta = getattr(self,f'{mode}_delta')
        self.min_duration = self.y.min().numpy()
        self.max_duration = self.y.max().numpy()

    def transform_x(self,x):
        return self.x_mapper.transform(x)

    def invert_duration(self,duration):
        return self.duration_mapper.inverse_transform(duration)

    def transform_duration(self,duration):
        return self.duration_mapper.transform(duration)

    def __getitem__(self, index):
        if self.cat_cols:
            return self.X[index,:],self.cat_X[index,:],self.y[index],self.delta[index]
        else:
            return self.X[index,:],self.cat_X,self.y[index],self.delta[index]

    def __len__(self):
        return self.X.shape[0]

class chunk_iterator():
    def __init__(self,X,delta,y,cat_X,shuffle,batch_size):
        self.X = X
        self.delta = delta
        self.y = y
        self.cat_X = cat_X
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.n = self.X.shape[0]
        self.chunks=self.n//batch_size+1
        self.perm = torch.randperm(self.n)
        self.valid_cat = not isinstance(self.cat_X, list)
        if self.shuffle:
            self.X = self.X[self.perm,:]
            self.delta = self.delta[self.perm]
            self.y = self.y[self.perm,:]
            if self.valid_cat: #F
                self.cat_X = self.cat_X[self.perm,:]
        self._index = 0
        self.it_X = torch.chunk(self.X,self.chunks)
        self.it_delta = torch.chunk(self.delta,self.chunks)
        self.it_y = torch.chunk(self.y,self.chunks)
        if self.valid_cat:  # F
            self.it_cat_X = torch.chunk(self.cat_X,self.chunks)
        else:
            self.it_cat_X = []
        self.true_chunks = len(self.it_X)

    def __next__(self):
        ''''Returns the next value from team object's lists '''
        if self._index < self.true_chunks:
            if self.valid_cat:
                result = (self.it_X[self._index],self.it_cat_X[self._index],self.it_y[self._index],self.it_delta[self._index])
            else:
                result = (self.it_X[self._index],[],self.it_y[self._index],self.it_delta[self._index])
            self._index += 1
            return result
        # End of Iteration
        raise StopIteration

    def __len__(self):
        return len(self.it_X)

class custom_dataloader():
    def __init__(self,dataset,batch_size=32,shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n = self.dataset.X.shape[0]
        self.len=self.n//batch_size+1
    def __iter__(self):
        return chunk_iterator(X =self.dataset.X,
                              delta = self.dataset.delta,
                              y = self.dataset.y,
                              cat_X = self.dataset.cat_X,
                              shuffle = self.shuffle,
                              batch_size=self.batch_size,
                              )
    def __len__(self):
        self.n = self.dataset.X.shape[0]
        self.len = self.n // self.batch_size + 1
        return self.len

def get_dataloader(data_train, data_val, data_test,bs,shuffle=True):
    d = survival_dataset(data_train, data_val, data_test)
    dat = custom_dataloader(dataset=d,batch_size=bs,shuffle=shuffle)
    return dat
