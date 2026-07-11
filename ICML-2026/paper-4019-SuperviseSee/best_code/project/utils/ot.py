import torch

class OptimalTransport():
    def __init__(self,  epsilon=0.1, gamma=1, stoperr=1e-6, rho=1, semi_use=True, numItermax=1000):
        super().__init__()
        self.epsilon = epsilon
        self.gamma = gamma
        self.stoperr = stoperr
        self.rho = rho
        self.b = None
        self.semi_use = semi_use
        self.numItermax = numItermax
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'


    def solve(self, P):
        # Q is the cost matrix 
        P = P.double()
        P = -torch.log(torch.softmax(P, dim=1))
        n=P.shape[0]
        k=P.shape[1]
        mu = torch.zeros(n, 1).to(self.device)
        expand_cost = torch.cat([P, mu], dim=1)
        Q = torch.exp(- expand_cost / self.epsilon)
        
        # prior distribution
        Pa = torch.ones(n, 1).to(self.device) / n  # how many samples
        Pb = self.rho * torch.ones(Q.shape[1], 1).to(self.device)/ k # how many prototypes
        Pb[-1] = 1 - self.rho

        # init b
        b = torch.ones(Q.shape[1], 1).double().to(self.device)/ Q.shape[1] if self.b is None else self.b

        fi = self.gamma / (self.gamma + self.epsilon)
        err = 1
        last_b = b.clone()
        iternum = 0
        while err > self.stoperr and iternum < self.numItermax:
            a = Pa / (Q @ b)
            b =  Pb / (Q.t() @ a)
            if self.semi_use:
                b[:-1,:] = torch.pow(b[:-1,:], fi)

            err = torch.norm(b - last_b)
            last_b = b.clone()
            iternum += 1

        plan = Q.shape[0]*a*Q*b.T

        return plan.float()

