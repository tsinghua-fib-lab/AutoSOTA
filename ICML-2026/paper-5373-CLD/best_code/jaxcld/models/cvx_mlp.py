from abc import ABC, abstractmethod

class Convex_MLP(ABC):
    def __init__(self, X, y, P_S, beta, rho, seed):
        self.X = X
        self.y = y
        self.P_S = P_S
        self.beta = beta
        self.rho = rho
        self.seed = seed 
        self.d_diags = None
        self.e_diags = None
        self.Xtst = None
        self.ytst = None
    
    @abstractmethod
    def init_model(self):
        pass
    
    @abstractmethod
    def rmatvec_Fi(self, i, vec):
        pass

    @abstractmethod
    def matvec_F(self, vec):
        pass
    
    @abstractmethod 
    def batch_matvec_F(self, vecs):
      pass

    @abstractmethod
    def rmatvec_F(self, vec):
        pass
    
    @abstractmethod
    def batch_rmatvec_F(self, vecs):
      pass
    
    @abstractmethod
    def matvec_G(self, vec):
        pass
    
    @abstractmethod
    def batch_matvec_G(self, vecs):
      pass

    @abstractmethod
    def rmatvec_G(self, vec):
        pass
    
    @abstractmethod
    def batch_rmatvec_G(self,vecs):
      pass

    @abstractmethod
    def matvec_A(self, vec):
         pass
    
    @abstractmethod
    def batch_matvec_A(self,vecs):
      pass

