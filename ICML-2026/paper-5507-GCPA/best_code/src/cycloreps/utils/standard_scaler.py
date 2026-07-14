from torch import Tensor

class StandardScaler:
    '''
    Class to standardise a 2-D embedding matrix by centring it to zero and scaling it to unit variance.
    '''

    def __init__(self, mean=None, std=None):
        self.mean = mean
        # self.std = std

    def fit(self, X : Tensor) -> None:
        self.mean = X.mean(dim=0)
        # self.std = X.std(dim=0)

    def transform(self, X : Tensor) -> Tensor:
        return X - self.mean
        # return (X - self.mean) / self.std

    def inverse_transform(self, X : Tensor) -> Tensor:
        return X + self.mean
        # return X * self.std + self.mean