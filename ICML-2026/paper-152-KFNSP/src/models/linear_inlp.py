from models.inlp.INP import get_debiasing_projection
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
import numpy as np

class LinearINPWithSVR:
    def __init__(self,
        gamma=None,
        kernel="rbf",
        eps=0.01,
        C=1,
        alpha_prime=0.05):

        self.alpha_prime = alpha_prime
        self.model = SVR(kernel=kernel, gamma=gamma, epsilon=eps, max_iter=50000,C=C)


    def train(self, X, y, p, iterations):
        X = np.array(X, dtype=np.float64)
        y = np.array(y, dtype=np.float64)
        p = np.array(p, dtype=np.float64)

        input_dim = np.shape(X)[-1]

        is_autoregressive = True
        min_accuracy = 0.0


        num_nullspace_projections = iterations

        P, _, _ = get_debiasing_projection(Ridge, {"alpha":self.alpha_prime}, num_nullspace_projections, input_dim,
                                                                is_autoregressive, min_accuracy, X,
                                                                p, by_class=False)
        
        self.nullspace_projection = P

        self.model.fit(X @ self.nullspace_projection ,y)


    def predict(self, X):
        X = np.array(X, dtype=np.float64)
        
        return np.array(self.model.predict(X @ self.nullspace_projection),dtype=np.float32)
         



