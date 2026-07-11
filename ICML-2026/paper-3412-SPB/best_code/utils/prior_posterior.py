

class GaussianPrior:
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma


class GaussianPosterior:
    def __init__(self, model, mu, sigma):
        self.model = model
        self.mu = mu
        self.sigma = sigma
