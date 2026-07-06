import torch
from statistics import mode

class Dummy:
    def __init__(self, classification = False):
        self.constant = None
        self.classification = classification

    def train(self,X,y,p,dummy=0):
        if self.classification:
            self.constant = mode(y)
        else:
            self.constant = y.mean()

    def predict(self,X):
        return torch.ones(X.shape[0])*self.constant
