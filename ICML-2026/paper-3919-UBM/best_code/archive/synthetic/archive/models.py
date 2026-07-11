import torch
import torch.nn as nn
import numpy as np 
from sklearn.linear_model import LogisticRegression 
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import roc_auc_score, mean_squared_error
from skorch import NeuralNetRegressor
from torch import optim
from sklearn.ensemble import GradientBoostingClassifier

class Model: 
    
    def __init__(self, model_name='LogisticRegression', hp={}, model_type='binary'): 
        self.model_name = model_name
        if model_name == 'MLP': 
            if torch.cuda.is_available():
                self.device = torch.device('cuda:1')
            elif torch.backends.mps.is_available():
                self.device = torch.device('mps')
            else:
                self.device  = torch.device('cpu')
        self.model = self._init_model(model_name, hp)
        self.model_type = model_type
        
    
    def _init_model(self, name, params): 
        if name == 'LogisticRegression':
            C = params.get('C',1.)
            return Pipeline([('var_threshold', VarianceThreshold()), \
                             ('LR', LogisticRegression(C=C, max_iter = 1))])
        elif name == 'GradientBoostingClassifier':
            lr = params.get('learning_rate', 1.)
            ne = params.get('n_estimators', 1.)
            md = params.get('max_depth', 1.)
            msl = params.get('min_samples_leaf', 1.)
            mss = params.get('min_samples_split', 1.)
            mf = params.get('max_features', 1.)
            rs = params.get('random_state', 1.)
            return GradientBoostingClassifier(learning_rate = lr, n_estimators = ne,\
                                              max_depth = md, min_samples_leaf = msl,\
                                              min_samples_split = mss, max_features = mf,\
                                              random_state = rs)
        elif name == 'RandomForestRegressor': 
            nt = params.get('n_estimators',50)
            md = params.get('max_depth', 5)
            mn = params.get('min_samples_split', 10)
            mf = params.get('max_features', 'all')
            return RandomForestRegressor(n_estimators=nt, max_depth=md, \
                                          min_samples_split=mn, max_features=mf)         
        elif name == 'Lasso': 
            alpha = params.get('alpha',1.)
            return Lasso(alpha=alpha, max_iter=1000)
        elif name == 'LinearRegression': 
            return LinearRegression()
        elif name == 'MLP': 
            hidden_layer_sizes = params.get('hidden_layer_sizes', (50,50))
            activation = params.get('activation', 'relu')
            solver = params.get('solver', 'adam')
            alpha = params.get('alpha', .001)
            learning_rate = params.get('learning_rate', 'adaptive')
            learning_rate_init = params.get('learning_rate_init', 1e-3)
            max_iter = params.get('max_iter', 200)
            input_dim = params.get('input_dim',-1)
            
            if solver == 'adam': 
                o = optim.Adam
            elif solver == 'sgd': 
                o = optim.SGD
        
            return MLPRegressor(hidden_layer_sizes=hidden_layer_sizes,
                                activation=activation,
                                solver=solver,
                                alpha=alpha,
                                learning_rate=learning_rate,
                                learning_rate_init=learning_rate_init,
                                max_iter=max_iter)

    
    def fit(self, X, y): 
#         if self.model_name == 'MLP': 
#             X = X.astype(np.float32)
#             y = y.reshape(-1,1).astype(np.float32)
        self.model.fit(X,y)
    
    def predict(self, X): 
        if self.model_type == 'binary':
            return self.model.predict_proba(X)[:,1]
        else:
            return self.model.predict(X).squeeze()
    
    def compute_metric(self, y_true, y_predict): 
        if self.model_type == 'binary': 
            return roc_auc_score(y_true, y_predict)
        elif self.model_type == 'continuous': 
            return -mean_squared_error(y_true, y_predict)
        else: 
            raise ValueError('metric can only be computed for binary and continuous outcomes')
            

class MLP(nn.Module):
    def __init__(self, hidden_layer_sizes, activation, input_dim):
        super(MLP, self).__init__()
        self.num_layers = len(hidden_layer_sizes)+1
        self.layers = nn.ModuleList()
        for i in range(self.num_layers): 
            if i == 0: 
                self.layers.append(nn.Linear(input_dim,\
                                    hidden_layer_sizes[0]))
            elif i == self.num_layers-1: 
                self.layers.append(nn.Linear(hidden_layer_sizes[-1],1))
            else: 
                self.layers.append(nn.Linear(hidden_layer_sizes[i-1],\
                                    hidden_layer_sizes[i]))
        if activation == 'relu':
            self.m = nn.ReLU()
        elif activation == 'tanh':
            self.m = nn.Tanh()
        assert len(self.layers) == self.num_layers

    def forward(self, X, **kwargs):
        for i in range(self.num_layers-1): 
            X = self.layers[i](X)
            X = self.m(X)
        return self.layers[-1](X)
