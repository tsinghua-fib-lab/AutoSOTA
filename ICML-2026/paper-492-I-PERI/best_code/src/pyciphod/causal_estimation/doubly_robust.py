

class TMLE:
    def __init__(self, nuisance_estimator, max_iter=1000, tol=1e-6):
        self.nuisance_estimator = nuisance_estimator
        self.max_iter = max_iter
        self.tol = tol
        # TODO