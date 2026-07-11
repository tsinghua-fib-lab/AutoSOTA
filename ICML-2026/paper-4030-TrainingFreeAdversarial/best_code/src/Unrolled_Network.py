import torch
from src.DF import DF_2d
from src.ResNet import ResNet


class UnrolledNet(torch.nn.Module):
    def __init__(self, mu = 0.05, Unrolls = 10):
        super().__init__()

        self.Network         = ResNet(in_Chn = 2, inner_Chn = 64, out_Chn = 2, RB_Blocks = 15)  
        self.DataConsistency = DF_2d(mu = mu, iterations = 10)
        self.Unrolls         = Unrolls


    def forward(self, zf, Coil, Mask):
        zf    = torch.concat([torch.real(zf), torch.imag(zf)], axis=1) 
        recon = torch.nn.Parameter(torch.clone(zf), requires_grad=True)
        for OuterIter in range(self.Unrolls):
            recon = self.Network(recon)
            recon = self.DataConsistency(recon, zf, Coil, Mask) 
        
        recon = recon[:,0:1,:,:] + 1j*recon[:,1:2,:,:]
        return recon
