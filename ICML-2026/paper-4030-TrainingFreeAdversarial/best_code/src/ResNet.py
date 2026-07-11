import torch

class ResidualBlock(torch.nn.Module):
 
    def __init__(self, in_Chn, out_Chn, KS=3, C=0.1): 
        super().__init__() 
        self.convLayer  = torch.nn.Conv2d(in_Chn, out_Chn, kernel_size=KS, padding=(KS//2, KS//2),bias=False)
        self.activation = torch.nn.ReLU(inplace = True)
        self.C          = C
        
    def forward(self, x):
        out = self.convLayer(x)
        out = self.activation(out)
        out = self.convLayer(out)
        return x + self.C * out
        

class ResNet(torch.nn.Module):
    
    def __init__(self, in_Chn, inner_Chn, out_Chn, KS=3, RB_Blocks=15):
        super().__init__()
        
        self.FirstLayer = torch.nn.Conv2d(in_Chn, inner_Chn, kernel_size=KS, padding=(KS//2, KS//2),bias=False)
        
        self.ResNetConv = torch.nn.Sequential(*[ResidualBlock(inner_Chn, inner_Chn) for _ in range(RB_Blocks)], 
                                              torch.nn.Conv2d(inner_Chn, inner_Chn, kernel_size=KS, padding=KS//2, bias=False))
        
        self.LastLayer  = torch.nn.Conv2d(inner_Chn, out_Chn, kernel_size=KS, padding=(KS//2, KS//2),bias=False)
 
    def forward(self, x):
        y = self.FirstLayer(x)
        out = self.ResNetConv(y)
        return self.LastLayer(out + y)

