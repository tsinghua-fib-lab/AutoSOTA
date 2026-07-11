import torch
from src.Utils import *

class DF_2d(torch.nn.Module):
    def __init__(self, mu, iterations=10):
        super().__init__()
        self.mu = torch.nn.Parameter(torch.tensor([mu]), requires_grad=True)
        self.iterations = iterations

    def EHE(self,x, Coil, Mask):
        EHEx = torch.sum(IFFT(  Mask*(FFT(x*Coil))  ) * torch.conj(Coil),axis=1,keepdim=True)
        return EHEx
        
    def forward(self, z, zf, Coil, Mask):
        p_now = zf[:,0:1,:,:] + zf[:,1:2,:,:]*1j + self.mu*(z[:,0:1,:,:] + z[:,1:2,:,:]*1j)
        r_now = torch.clone( p_now)
        b_approx = torch.zeros_like(p_now)
        
        for i in range(self.iterations):
            
            q = self.EHE(p_now,Coil,Mask) + self.mu*p_now; 
            rrOverpq = torch.sum(r_now*torch.conj(r_now)) / torch.sum(q*torch.conj(p_now))  # rrOverpq = (r'*r)/(p'*q);
            b_next = b_approx + rrOverpq*p_now
            r_next = r_now - rrOverpq*q
            p_next = r_next + torch.sum(r_next*torch.conj(r_next)) / torch.sum(r_now*torch.conj(r_now)) * p_now # p = r_next + ( (r_next'*r_next)/(r'*r) )*p;
            b_approx = b_next
            p_now = torch.clone(p_next)
            r_now = torch.clone(r_next)

        return torch.cat([torch.real(b_approx), torch.imag(b_approx)], dim=1)
    
    
    
    

    
